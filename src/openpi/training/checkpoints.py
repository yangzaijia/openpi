from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import errno
import logging
import os
import subprocess
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    return checkpoint_manager.save(step, items)


def _is_quota_or_space_error(error: BaseException) -> bool:
    """Only disk space errors may be skipped; model and checkpoint bugs must stop training."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in (errno.EDQUOT, errno.ENOSPC):
            return True
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
    return False


class ResilientCheckpointSaver:
    """Keep training through a bounded run of disk-full asynchronous save failures.

    The train state stays on device. After a failed save, the old Orbax manager
    must be drained and replaced: its failed finalize thread otherwise affects
    the next save. A later save uses the *current* train state, not a restored
    older checkpoint.
    """

    def __init__(
        self,
        manager: ocp.CheckpointManager,
        directory: epath.Path | str,
        *,
        keep_period: int | None,
        max_consecutive_failures: int | None = None,
    ):
        if max_consecutive_failures is None:
            max_consecutive_failures = int(os.environ.get("OPENPI_CKPT_MAX_CONSECUTIVE_FAILURES", "10"))
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive")
        self.manager = manager
        self.directory = epath.Path(directory)
        self.keep_period = keep_period
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.pending_step: int | None = None
        self.last_complete_step = manager.latest_step()

    def _alert(self, event: str, step: int, detail: str) -> None:
        """Optional nonblocking site-owned mail hook; no credentials live in Git."""
        command = os.environ.get("OPENPI_CKPT_ALERT_COMMAND")
        recipient = os.environ.get("OPENPI_CKPT_ALERT_EMAIL")
        if not command or not recipient:
            return
        try:
            subprocess.Popen(
                [command, recipient, event, str(step), str(self.directory), detail],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except OSError:
            logging.exception("Could not launch checkpoint alert command")

    def _replace_failed_manager(self, error: BaseException, step: int) -> None:
        if not _is_quota_or_space_error(error):
            raise error
        logging.error(
            "Checkpoint %s failed due to disk space/quota; last complete step=%s. "
            "Training will continue from the in-memory state.",
            step,
            self.last_complete_step,
            exc_info=error,
        )
        self.consecutive_failures += 1
        if self.consecutive_failures == 1:
            self._alert("failed", step, str(error))

        # Orbax 0.11.x may surface the same async failure once from its
        # finalize thread and again from the underlying checkpointer. Repeated
        # close() calls drain these public wait/close paths. If writers cannot
        # be drained, fail closed rather than open a second writer on this dir.
        for attempt in range(4):
            try:
                self.manager.close()
                break
            except Exception as close_error:
                if not _is_quota_or_space_error(close_error) or attempt == 3:
                    raise RuntimeError("Could not drain failed checkpoint writer") from close_error
                logging.warning("Draining failed checkpoint writer (%s/4): %s", attempt + 1, close_error)

        if self.consecutive_failures > self.max_consecutive_failures:
            raise RuntimeError(
                f"More than {self.max_consecutive_failures} consecutive checkpoint saves failed; "
                f"last complete step was {self.last_complete_step}"
            ) from error

        self.manager, _ = initialize_checkpoint_dir(
            self.directory,
            keep_period=self.keep_period,
            overwrite=False,
            resume=True,
        )
        self.last_complete_step = self.manager.latest_step()

    def _finish_pending(self) -> None:
        if self.pending_step is None:
            return
        step = self.pending_step
        self.pending_step = None
        try:
            self.manager.wait_until_finished()
        except Exception as error:
            self._replace_failed_manager(error, step)
        else:
            self.last_complete_step = step
            if self.consecutive_failures:
                logging.info("Checkpoint saving recovered at step %s", step)
                self._alert("recovered", step, f"last complete checkpoint: {step}")
                self.consecutive_failures = 0

    def save_state(self, state: training_utils.TrainState, data_loader: _data_loader.DataLoader, step: int) -> None:
        self._finish_pending()
        if self.last_complete_step is not None and self.last_complete_step >= step:
            logging.info("Checkpoint %s is already finalized; skipping duplicate save", step)
            return
        try:
            saved = save_state(self.manager, state, data_loader, step)
        except Exception as error:
            self._replace_failed_manager(error, step)
            if self.last_complete_step is not None and self.last_complete_step >= step:
                logging.info("Checkpoint %s finished despite manager metadata error", step)
                return
            # The previous async failure may only surface when saving this step.
            # Try this current state once with the fresh manager, so a disk
            # repaired just before step 20k can actually save step 20k.
            try:
                saved = save_state(self.manager, state, data_loader, step)
            except Exception as retry_error:
                self._replace_failed_manager(retry_error, step)
                return
        if not saved:
            raise RuntimeError(f"Checkpoint manager declined save at step {step}")
        self.pending_step = step

    def finish(self, state: training_utils.TrainState, data_loader: _data_loader.DataLoader, step: int) -> None:
        """Do not report training complete without a finalized last checkpoint."""
        self._finish_pending()
        if self.last_complete_step != step:
            self.save_state(state, data_loader, step)
            self._finish_pending()
        if self.last_complete_step != step:
            raise RuntimeError(f"Final checkpoint {step} did not finish; last complete step={self.last_complete_step}")
        self.manager.close()


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
