"""Sequential E01 training queue on ONE GPU: never two nnU-Net runs at once (PREREG D3).

    run.cmd -m e01.jobqueue            (normally via env\\resume.cmd, detached)

Jobs run as separate processes, in order. A finished job (checkpoint_final exists) is skipped
by launch_train; an interrupted one resumes from checkpoint_latest. The queue stops at the
first job that pauses (exit 3) or fails (any other non-zero exit).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from e01 import common as c
from e01 import pause as pz

JOBS = [
    ["--arm", "A", "--gpu", "3070ti"],
    ["--arm", "B", "--gpu", "3070ti"],
    ["--arm", "C2", "--gpu", "5060ti"],      # PREREG_C C-1.2 + DC1; owner go-ahead for the full run 2026-11-15
]
LOG = c.RESULTS / "logs" / "queue.jsonl"


def log(ev: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ev = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush=True)


def main() -> int:
    log({"event": "queue_start", "jobs": JOBS})
    for job in JOBS:
        if pz.pause_requested():
            log({"event": "queue_paused_before", "job": job})
            return pz.EXIT_PAUSED
        arm = job[1]
        out = c.RESULTS / "logs" / f"{arm}_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log({"event": "job_start", "job": job, "stdout": str(out)})
        with open(out, "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable, "-u", "-m", "e01.launch_train", *job],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=str(c.E01))
        log({"event": "job_end", "job": job, "exit": rc})
        if rc == pz.EXIT_PAUSED:
            log({"event": "queue_paused", "job": job})
            return rc
        if rc != 0:
            log({"event": "queue_failed", "job": job, "exit": rc})
            return rc
    log({"event": "queue_done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
