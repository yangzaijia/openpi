# Training through a disk-full checkpoint failure

`scripts/train.py` and `scripts/train_rrsim_v3_guarded_165180.py` use
`ResilientCheckpointSaver`. Only `ENOSPC` and `EDQUOT` errors are tolerated.
Other checkpoint, model, and data errors still stop training.

When an asynchronous save fails, the error may surface at the next scheduled
save. The saver waits for the old writer to finish, closes it, and opens a new
Orbax manager on the same checkpoint directory. It then attempts to save the
**current in-memory train state** at the current step. If the disk is still
full, training can continue to a later save point. The most recent finalized
checkpoint remains the only restart point until a newer save finishes.

The default limit is 10 consecutive failed saves. Set
`OPENPI_CKPT_MAX_CONSECUTIVE_FAILURES` to a positive integer to change it.
At a 2,500-step save interval, 10 failures allow up to 25,000 steps without a
new checkpoint. The final step must save successfully; otherwise the process
exits with an error. Existing processes do not pick up this code until they
restart.

For optional alerts, set `OPENPI_CKPT_ALERT_EMAIL` and
`OPENPI_CKPT_ALERT_COMMAND` in the launch environment. The command is invoked
without a shell, with arguments:

```text
<recipient> <failed|recovered> <step> <checkpoint-directory> <detail>
```

It is launched without waiting on the GPU training loop. The command must
provide its own mail transport; this repository does not store SMTP credentials
or assume that the cluster has a working mail relay. No email is sent when
either environment variable is unset.
