"""rr-sim (Industrial_Arm 单臂真机) 的 π0.5 输入/输出变换。

仿 openpi 官方 `libero_policy.py`（单臂骨架），针对 rr-sim 数据的两处差异改造：

  1. state 只取前 7 维（pose6 + gripper），**丢弃 force/torque 6 维**（用户 2026-09-20 决定去力）。
     原始 observation.state 是 13 维：x y z rx ry rz gripper fx fy fz tx ty tz。
  2. 相机映射：base_0_rgb = view1（外部第三人称），left_wrist_0_rgb = hand（腕部）。
     rr-sim 只有 2 个相机；第三槽 right_wrist_0_rgb 零填，pi05 下 image_mask=False（自动忽略）。

action 为 7 维 EE-pose（x y z rx ry rz gripper，rotvec）。delta 由 data config 的
extra_delta_transform 控制（DeltaActions(make_bool_mask(6,-1))：前 6 维相对当前 state 相减、gripper 绝对）。
详见 rr-sim/00_rr-sim_pi05_训练设定.zh.md。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_rrsim_example() -> dict:
    """随机输入样例（冒烟测试用）。state 13 维，会在 RrsimInputs 里被切到 7 维。"""
    return {
        "observation/state": np.random.rand(13),
        "observation/image": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(360, 480, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RrsimInputs(transforms.DataTransformFn):
    """把 rr-sim 数据整理成模型输入格式（训练与推理共用）。"""

    # 决定用哪个模型（不要改）。
    model_type: _model.ModelType

    # rr-sim 保留的 state 维度：去力后为前 7 维（pose6 + gripper）。
    state_dim: int = 7

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 去力：只取前 state_dim 维（丢弃 fx fy fz tx ty tz）。
        state = np.asarray(data["observation/state"])[..., : self.state_dim]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,          # view1（外部）
                "left_wrist_0_rgb": wrist_image,   # hand（腕部）
                # rr-sim 无右腕相机：零填 + pi05 下 mask=False。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 动作只在训练时出现；pad 到模型维度由后续 model_transforms 负责。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # B1/B2：透传离线动态码（照 aloha_policy.py），否则 compute_loss 里 observation.dyn_codes 恒为 None、CE 静默失效。
        # RR0 baseline 数据无 dyn_codes 时本段 no-op，行为逐字不变。
        for _k in ("dyn_codes", "dyn_codes_mask"):
            if _k in data:
                inputs[_k] = data[_k]

        return inputs


@dataclasses.dataclass(frozen=True)
class RrsimOutputs(transforms.DataTransformFn):
    """推理时把模型输出的动作截回 rr-sim 的 7 维（EE-pose）。"""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
