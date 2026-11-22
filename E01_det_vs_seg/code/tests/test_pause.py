"""Pause hook on a minimal stand-in trainer (same on_epoch_end/save_checkpoint contract as nnU-Net)."""
import os

import pytest

from e01 import pause as pz


class FakeTrainer:
    def __init__(self, folder, num_epochs=5):
        self.output_folder = str(folder)
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.local_rank = 0
        self.save_every = 10
        self.saved = []
        self.logged = []

    def on_epoch_end(self):                    # nnU-Net increments at the end
        self.current_epoch += 1

    def save_checkpoint(self, fn):
        self.saved.append((os.path.basename(fn), self.current_epoch + 1))
        open(fn, "w").close()

    def print_to_log_file(self, *a):
        self.logged.append(" ".join(map(str, a)))


@pytest.fixture
def flag(tmp_path, monkeypatch):
    f = tmp_path / "PAUSE"
    monkeypatch.setattr(pz, "PAUSE_FLAG", f)
    return f


def make(tmp_path):
    cls = type("T", (FakeTrainer,), {})
    pz.install_pause(cls)
    return cls(tmp_path)


def test_saves_every_epoch_atomically_and_not_after_last(tmp_path, flag):
    t = make(tmp_path)
    for _ in range(5):
        t.on_epoch_end()
    assert t.save_every == 10 ** 9
    assert [s for s in t.saved] == [("checkpoint_latest.pth.tmp", e) for e in (1, 2, 3, 4)]
    assert (tmp_path / "checkpoint_latest.pth").exists() and not (tmp_path / "checkpoint_latest.pth.tmp").exists()


def test_pause_flag_raises_after_saving(tmp_path, flag):
    t = make(tmp_path)
    t.on_epoch_end()
    flag.touch()
    with pytest.raises(pz.TrainingPaused) as e:
        t.on_epoch_end()
    assert e.value.args[0] == 2
    assert t.saved[-1] == ("checkpoint_latest.pth.tmp", 2)      # saved before pausing
    assert t.current_epoch == 2


def test_install_is_idempotent(tmp_path, flag):
    cls = type("T2", (FakeTrainer,), {})
    pz.install_pause(cls)
    f = cls.on_epoch_end
    pz.install_pause(cls)
    assert cls.on_epoch_end is f
