"""Sequential E02 queue on ONE GPU: never two nnU-Net runs at once (PREREG §4; E01 D3).

    run.cmd -m e02.jobqueue            (normally via env\\resume.cmd, detached)

Jobs run as separate processes, in order. A finished job (checkpoint_final exists) is skipped
by launch_train; an interrupted one resumes from checkpoint_latest. The queue stops at the
first job that pauses (exit 3) or fails (any other non-zero exit).

After the training jobs, the test-set evaluation (e02.run_eval --cases test) runs as the last
job, but only once results\\EVAL_GO exists (created by env\\eval.cmd: the owner's go-ahead for the
one test prediction run). It pauses and resumes like training: each stage continues where it
stopped, and a finished evaluation exits at once with nothing to do.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from e02 import common as c
from e02 import pause as pz

JOBS = [
    ["--arm", "A", "--gpu", "3070ti"],
    ["--arm", "B", "--gpu", "3070ti"],
    # C2 added 2026-11-17 after its crops were built and preprocessed (PREREG §9 step 4)
    ["--arm", "C2", "--gpu", "3070ti"],
]
EVAL_JOB = ["--cases", "test", "--gpus", "3070ti,5060ti"]   # training is over: both GPUs are free
EVAL_GO = c.RESULTS / "EVAL_GO"
LOG = c.RESULTS / "logs" / "queue.jsonl"


def log(ev: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ev = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush=True)


def run_job(module: str, job: list[str], log_name: str) -> int:
    if pz.pause_requested():
        log({"event": "queue_paused_before", "job": job})
        return pz.EXIT_PAUSED
    out = c.RESULTS / "logs" / f"{log_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log({"event": "job_start", "job": job, "stdout": str(out)})
    with open(out, "w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable, "-u", "-m", module, *job],
                             stdout=f, stderr=subprocess.STDOUT, cwd=str(c.E02))
    log({"event": "job_end", "job": job, "exit": rc})
    if rc == pz.EXIT_PAUSED:
        log({"event": "queue_paused", "job": job})
    elif rc != 0:
        log({"event": "queue_failed", "job": job, "exit": rc})
    return rc


def main() -> int:
    log({"event": "queue_start", "jobs": JOBS, "eval": EVAL_GO.exists()})
    for job in JOBS:
        rc = run_job("e02.launch_train", job, f"{job[1]}_train")
        if rc != 0:
            return rc
    if EVAL_GO.exists():
        rc = run_job("e02.run_eval", EVAL_JOB, "eval")
        if rc != 0:
            return rc
    log({"event": "queue_done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
