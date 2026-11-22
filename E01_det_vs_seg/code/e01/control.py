"""E01 run control:  run.cmd -m e01.control {pause|resume|status}

pause   create results\\PAUSE; the running job saves after its current epoch and exits (code 3),
        and the queue stops. Nothing is lost.
resume  remove PAUSE and start the sequential queue (A then B, 3070 Ti) as a detached process,
        unless an E01 job is already running.
status  queue events, last epoch lines per arm (pseudo-Dice filtered out), GPUs, PAUSE flag.
"""
from __future__ import annotations

import json
import subprocess
import sys

from e01 import common as c
from e01 import pause as pz

PATTERN = r"e01\.(jobqueue|launch_train)"


def running() -> list[str]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '" + PATTERN +
          "' -and $_.Name -eq 'python.exe' } | ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def pause() -> None:
    pz.PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    pz.PAUSE_FLAG.touch()
    print(f"PAUSE requested ({pz.PAUSE_FLAG}). The current epoch finishes (~1.5 min), then the run stops.")


def resume() -> None:
    r = running()
    if r:
        print("An E01 job is already running; not starting another queue:\n  " + "\n  ".join(r))
        raise SystemExit(1)
    if pz.PAUSE_FLAG.exists():
        pz.PAUSE_FLAG.unlink()
    logs = c.RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with open(logs / "queue_stdout.log", "a", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, "-u", "-m", "e01.jobqueue"], cwd=str(c.E01), stdout=f,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
                             close_fds=True)
    print(f"E01 queue started (pid {p.pid}). Check progress with env\\status.cmd")


def status() -> None:
    print("PAUSE flag:", "SET" if pz.PAUSE_FLAG.exists() else "not set")
    r = running()
    print("running E01 processes:", len(r))
    for l in r:
        print("  ", l[:160])
    q = c.RESULTS / "logs" / "queue.jsonl"
    if q.exists():
        print("queue events (last 5):")
        for l in q.read_text(encoding="utf-8").splitlines()[-5:]:
            e = json.loads(l)
            print("  ", e["utc"], e["event"], " ".join(e.get("job", [])), e.get("exit", ""))
    for arm, ds, tr, fold in (("A", c.DATASET, "nnUNetTrainer_250epochs", c.FOLD),
                              ("B", c.DATASET, "E01HeatmapTrainer", c.FOLD),
                              ("C2", "Dataset902_E01C2Crops", "nnUNetTrainer_250epochs", "all")):
        fd = c.NN_RES / ds / f"{tr}__{c.PLANS}__{c.CONFIG}" / f"fold_{fold}"
        # by modification time: nnU-Net's unpadded dates make name order wrong (2026_11_9 > 2026_11_10)
        logs = sorted(fd.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime) if fd.exists() else []
        if not logs:
            print(f"arm {arm}: not started")
            continue
        lines = [l for l in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
                 if "pseudo" not in l.lower()]
        ep = [l for l in lines if ": Epoch " in l and "time" not in l]
        tm = [l for l in lines if "Epoch time" in l]
        tl = [l for l in lines if "train_loss" in l]
        done = (fd / "checkpoint_final.pth").exists()
        print(f"arm {arm}: {'FINISHED' if done else 'in progress'}; "
              f"{ep[-1].split(': ', 1)[1].strip() if ep else '-'}; last {tm[-1].split(': ', 1)[1].strip() if tm else '-'}"
              f"; last {tl[-1].split(': ', 1)[1].strip() if tl else 'train_loss -'}")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    print("GPUs:\n  " + out.strip().replace("\n", "\n  "))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"pause": pause, "resume": resume, "status": status}[cmd]()
