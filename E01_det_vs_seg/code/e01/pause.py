"""Pause/resume for E01 nnU-Net runs (pattern copied from 1.60 p3_train_nnunet.install_pause; not
imported, because importing 1.60 launcher modules would repoint nnUNet_* at 1.60).

- After every epoch: atomic checkpoint_latest.pth (tmp + os.replace), so a pause loses nothing.
  nnU-Net's own 10-epoch save is disabled (save_every = 1e9); checkpoint_final is unchanged.
- PAUSE flag (results\\PAUSE): checked at each epoch end; if present -> save, stop the
  augmentation workers, raise TrainingPaused. The launcher exits with EXIT_PAUSED (3) and the
  queue stops. Resume = remove the flag and re-run the queue (env\\resume.cmd).
"""
from __future__ import annotations

import os

from e01 import common as c

PAUSE_FLAG = c.RESULTS / "PAUSE"
EXIT_PAUSED = 3


class TrainingPaused(Exception):
    """args[0] = number of completed epochs."""


def pause_requested() -> bool:
    return PAUSE_FLAG.exists()


def install_pause(trainer_cls) -> None:
    original = trainer_cls.on_epoch_end
    if getattr(original, "_e01_patched", False):
        return

    def on_epoch_end(self):
        self.save_every = 10 ** 9            # replaced by the per-epoch atomic save below
        original(self)                        # increments current_epoch at its end
        finished = self.current_epoch
        if finished >= self.num_epochs:
            return                            # on_train_end writes checkpoint_final next
        latest = os.path.join(self.output_folder, "checkpoint_latest.pth")
        tmp = latest + ".tmp"
        self.current_epoch -= 1               # save_checkpoint stores current_epoch + 1
        self.save_checkpoint(tmp)
        self.current_epoch += 1
        if self.local_rank == 0 and os.path.exists(tmp):
            os.replace(tmp, latest)
        if pause_requested():
            self.print_to_log_file(f"PAUSE flag found after epoch {finished - 1}; checkpoint saved; pausing")
            for dl in (getattr(self, "dataloader_train", None), getattr(self, "dataloader_val", None)):
                if dl is not None and hasattr(dl, "_finish"):
                    try:
                        dl._finish()
                    except Exception:  # pragma: no cover  (best effort on shutdown)
                        pass
            raise TrainingPaused(finished)

    on_epoch_end._e01_patched = True
    trainer_cls.on_epoch_end = on_epoch_end
