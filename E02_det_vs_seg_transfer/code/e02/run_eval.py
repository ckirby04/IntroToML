"""E02 test-set prediction and evaluation, end to end, pausable and resumable (PREREG §6-§11, §9 step 5).

    run.cmd -m e02.run_eval --cases test  [--gpus 3070ti,5060ti] [--workers 6]    the one test run
    run.cmd -m e02.run_eval --cases smoke --smoke-model [--gpus 5060ti] [--c2-checkpoint PATH]
    run.cmd -m e02.run_eval --cases preflight     8 training images, FINAL models (results\\eval_preflight)
                                                                     pre-flight on 2 TRAINING images

Normally started by the queue (env\\eval.cmd), never by hand, so it cannot share a GPU with a
training run. Stages, in order; each skips work already on disk:

  pred_A, pred_B  whole-image maps (predict.build_predictor, nnU-Net's own file path)
                  -> results\\pred\\{A,B}\\<case>.npz
  pred_C2         C2 stage 2 on a crop around every B peak -> results\\two_stage\\c2_pred\\<case>.npz
                  (c2_predict.CropPredictor: bit-identical to nnU-Net's single-array path, 3.2x faster)
  cases           per-image records (matching both hit rules, masks, Dice, descriptives), CPU pool
                  -> results\\eval\\cases\\<case>.pkl
  bootstrap       2,000 paired well resamples in chunks, CPU pool -> results\\eval\\boot\\rows_<start>.npz
  aggregate       CIs, verdicts H1-H6, per line, tertiles, multi-hit, descriptives, FROC curves
                  -> results\\eval\\summary.json, froc_curves.npz

GPU stages run in one worker process per GPU (--gpus). Workers claim images one at a time with
an atomic lock file, so no image is predicted twice; which GPU made each file is logged in the
folder's gpu_by_case.jsonl. Images differ only in which GPU ran them (TF32 numerics can differ
slightly between GPU models); every image is still predicted exactly once, with one checkpoint.

Pause / resume:
  * env\\pause.cmd creates results\\PAUSE (a smoke run uses results\\eval_smoke\\PAUSE instead). The run checks it between images (and between bootstrap
    chunks), saves nothing half-done, and exits with code 3; the queue stops.
  * env\\resume.cmd removes the flag and restarts the queue, which restarts this script; every
    stage continues where it stopped.
  * Every output is written to a temp file and renamed, so a crash or kill cannot leave a partial
    file that would later be skipped.
  * Each prediction folder pins its model (model.json: checkpoint sha256). A resumed stage refuses
    to continue with a different checkpoint, so no folder can mix two models.
  * The bootstrap weights are regenerated from the seed on resume and checked against a stored
    hash, so chunks from different sessions belong to one draw.

Smoke mode uses the 2-epoch A and B models, the 2 training images of predict.case_list("smoke"),
and A's smoke model as a STAND-IN for C2 (there is no C2 smoke model). It exercises every code
path; its numbers mean nothing. Outputs go to *_smoke folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e02 import common as c  # first: environment
from e02 import pause as pz

STAGES = ("pred_A", "pred_B", "pred_C2", "cases", "bootstrap", "aggregate")
GPU_STAGES = ("pred_A", "pred_B", "pred_C2")
DET_ARMS = ("A", "B", "C1", "C2")
SEG_ARMS = ("A", "C1", "C2", "C4")
CONTRASTS = (("B", "A"), ("A", "B"), ("C1", "B"), ("C1", "A"), ("C4", "A"),
             ("C2", "B"), ("C2", "A"), ("C4", "C1"))
BOOT_CHUNK = 20
# PREREG §2.1 / §3 test-set counts, checked against the records in test mode
TEST_N_IMAGES = 1460
TEST_BIN_COUNTS = {"small": 107660, "medium": 264573, "large": 72367}
TEST_EMPTIED_BY_GAP = 2


class EvalPaused(Exception):
    pass


# ---------------------------------------------------------------- run setup ------------------------
@dataclass
class Run:
    kind: str                   # "test" | "smoke"
    ids: list[str]
    folder: str                 # imagesTs | imagesTr
    split: str                  # Ts | Tr
    out: Path
    smoke_model: bool
    n_boot: int
    workers: int
    c2_checkpoint: Path | None = None     # smoke only: a copy of a C2 checkpoint instead of the A stand-in

    @property
    def pause_flag(self) -> Path:
        """The test run shares the queue's flag (env\\pause.cmd). A smoke run has its own, so
        pausing a pre-flight can never pause a training run."""
        return pz.PAUSE_FLAG if self.kind == "test" else self.out / "PAUSE"

    @property
    def cases_dir(self) -> Path:
        return self.out / "cases"

    @property
    def boot_dir(self) -> Path:
        return self.out / "boot"


def make_run(kind: str, smoke_model: bool, n_boot: int, workers: int, c2_checkpoint=None) -> Run:
    from e02.predict import case_list

    ids, folder = case_list(kind)
    from e02.predict import OUT_SUFFIX

    out = c.RESULTS / f"eval{OUT_SUFFIX[kind]}"
    return Run(kind, ids, folder, "Ts" if kind == "test" else "Tr", out, smoke_model, n_boot, workers,
               Path(c2_checkpoint) if c2_checkpoint else None)


def atomic_bytes(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def atomic_json(dst: Path, obj) -> None:
    atomic_bytes(dst, json.dumps(obj, indent=1, default=_json_default).encode("utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def clean_tmp(folder: Path) -> int:
    n = 0
    if folder.exists():
        for p in list(folder.glob("*.tmp")) + list(folder.glob("*.tmp.npz")):
            p.unlink()
            n += 1
    return n


def log(run: Run, ev: dict) -> None:
    run.out.mkdir(parents=True, exist_ok=True)
    ev = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **ev}
    with open(run.out / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, default=_json_default) + "\n")
    print(json.dumps(ev, default=_json_default), flush=True)


def progress(run: Run, stage: str, done: int, total: int, t0: float, done_at_start: int) -> None:
    rate = (time.time() - t0) / max(1, done - done_at_start)
    atomic_json(run.out / "progress.json", {
        "stage": stage, "done": done, "total": total, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sec_per_item_this_session": round(rate, 2), "eta_hours_stage": round(rate * (total - done) / 3600, 2)})


def check_pause(run: Run, stage: str) -> None:
    if run.pause_flag.exists():
        log(run, {"event": "paused", "stage": stage})
        raise EvalPaused(stage)


def pin_model(folder: Path, meta: dict) -> None:
    """The folder's predictions must all come from one checkpoint (model.json, else predict.py's
    run.json from an earlier pre-flight)."""
    folder.mkdir(parents=True, exist_ok=True)
    pinned = None
    for name in ("model.json", "run.json"):
        p = folder / name
        if p.exists():
            pinned = json.loads(p.read_text(encoding="utf-8")).get("checkpoint_sha256")
            break
    if pinned is not None and pinned != meta["checkpoint_sha256"]:
        raise SystemExit(f"{folder} holds predictions from checkpoint {pinned[:12]}..., "
                         f"not {meta['checkpoint_sha256'][:12]}...: refusing to mix models")
    if not (folder / "model.json").exists():
        atomic_json(folder / "model.json", meta)


# ---------------------------------------------------------------- stage: whole-image maps ----------
def pending_pred(run: Run, arm: str) -> list[str]:
    from e02.predict import out_dir

    od = out_dir(run.kind, arm)
    return [s for s in run.ids if not (od / f"{s}.npz").exists()]


def pred_dir(run: Run, stage: str) -> Path:
    from e02 import c2_predict as c2p
    from e02.predict import out_dir

    return c2p.out_dir(run.kind) if stage == "pred_C2" else out_dir(run.kind, stage[-1])


def pending_c2(run: Run) -> list[str]:
    od = pred_dir(run, "pred_C2")
    return [s for s in run.ids if not (od / f"{s}.npz").exists()]


def _claim(claims: Path, case: str, who: str) -> bool:
    """Atomic claim of one image by one worker (O_EXCL): two GPUs never predict the same image."""
    try:
        fd = os.open(claims / f"{case}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, who.encode())
    os.close(fd)
    return True


def _worker_predictor(run: Run, stage: str):
    from e02 import c2
    from e02.predict import build_predictor

    if stage == "pred_C2":
        if run.smoke_model:
            if run.c2_checkpoint:                     # pre-flight with a copy of a C2 checkpoint
                pr, meta = build_predictor("C2", dataset=c2.DATASET, checkpoint=run.c2_checkpoint)
                meta["c2_standin"] = f"C2 checkpoint copy, epoch {meta['epoch']} (pre-flight only)"
            else:
                pr, meta = build_predictor("A", smoke=True)
                meta["c2_standin"] = "A's 2-epoch model stands in for C2 (pre-flight only; numbers mean nothing)"
        else:
            pr, meta = build_predictor("C2", dataset=c2.DATASET)
    else:
        pr, meta = build_predictor(stage[-1], smoke=run.smoke_model)
    return pr, meta


def work_gpu_stage(run: Run, stage: str, gpu_key: str) -> int:
    """One GPU worker: claim pending images one at a time until none is left. Exit 0 / 3 (paused)."""
    from PIL import Image

    from e02 import c2_predict as c2p
    from e02.predict import load_prob

    od = pred_dir(run, stage)
    claims = od / "_claims"
    who = f"{gpu_key} pid {os.getpid()}"
    pr, meta = _worker_predictor(run, stage)
    pin_model(od, meta)
    crop_pred = c2p.CropPredictor(pr, batch=1) if stage == "pred_C2" else None   # bit-identical (c2_bench)
    tmp = od / f"_tmp_{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    img_dir = c.NN_RAW / c.DATASET / run.folder
    n = 0
    try:
        for s in run.ids:
            if (od / f"{s}.npz").exists():
                continue
            if run.pause_flag.exists():
                log(run, {"event": "worker_paused", "stage": stage, "gpu": gpu_key, "done_by_worker": n})
                return pz.EXIT_PAUSED
            if not _claim(claims, s, who):
                continue
            src = img_dir / f"{s}_0000.png"
            assert src.exists(), src
            if stage == "pred_C2":
                img = np.asarray(Image.open(src))
                c2p.save_atomic(od / f"{s}.npz", c2p.predict_case(pr, img, load_prob(run.kind, "B", s), crop_pred))
            else:
                pr.predict_from_files_sequential([[str(src)]], [str(tmp / s)], save_probabilities=True,
                                                 overwrite=True)
                os.replace(tmp / f"{s}.npz", od / f"{s}.npz")    # the only file kept, renamed when complete
                for ext in (".png", ".pkl"):
                    (tmp / f"{s}{ext}").unlink(missing_ok=True)
            with open(od / "gpu_by_case.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"case": s, "gpu": c.GPUS[gpu_key]}) + "\n")
            (claims / f"{s}.lock").unlink(missing_ok=True)
            n += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(run, {"event": "worker_done", "stage": stage, "gpu": gpu_key, "done_by_worker": n,
              "slow_path_crops": getattr(crop_pred, "n_slow", None)})
    return 0


def stage_gpu(run: Run, stage: str, gpus: list[str]) -> None:
    """Coordinator: one worker process per GPU, all claiming from the same pending list."""
    import subprocess

    od = pred_dir(run, stage)
    od.mkdir(parents=True, exist_ok=True)
    clean_tmp(od)
    shutil.rmtree(od / "_claims", ignore_errors=True)      # no worker is alive: old claims are stale
    (od / "_claims").mkdir()
    todo = [s for s in run.ids if not (od / f"{s}.npz").exists()]
    if not todo:
        return
    total, start, t0 = len(run.ids), len(run.ids) - len(todo), time.time()
    log(run, {"event": "stage_start", "stage": stage, "todo": len(todo), "total": total, "gpus": gpus})
    procs = {}
    for i, g in enumerate(gpus):
        cmd = [sys.executable, "-u", "-m", "e02.run_eval", "--cases", run.kind, "--worker", stage, "--gpu", g]
        if run.smoke_model:
            cmd.append("--smoke-model")
        if run.c2_checkpoint:
            cmd += ["--c2-checkpoint", str(run.c2_checkpoint)]
        f = open(run.out / f"worker_{stage}_{i}_{g}.log", "a", encoding="utf-8")
        procs[f"{i}_{g}"] = (subprocess.Popen(cmd, cwd=str(c.E02), stdout=f, stderr=subprocess.STDOUT), f)
        time.sleep(5)                                  # stagger model loading
    failed = None
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(10)
        done = sum((od / f"{s}.npz").exists() for s in run.ids)
        progress(run, stage, done, total, t0, start)
        for g, (p, _) in procs.items():
            rc = p.poll()
            if rc not in (None, 0, pz.EXIT_PAUSED) and failed is None:
                failed = (g, rc)
                for p2, _ in procs.values():          # outputs are atomic: killing loses only work in flight
                    if p2.poll() is None:
                        p2.terminate()
    for p, f in procs.values():
        p.wait()
        f.close()
    rcs = {g: p.returncode for g, (p, _) in procs.items()}
    done = sum((od / f"{s}.npz").exists() for s in run.ids)
    progress(run, stage, done, total, t0, start)
    if failed is not None:
        raise RuntimeError(f"{stage} worker on {failed[0]} failed (exit {failed[1]}); see "
                           f"{run.out / f'worker_{stage}_{failed[0]}.log'}")
    if any(rc == pz.EXIT_PAUSED for rc in rcs.values()) or run.pause_flag.exists():
        log(run, {"event": "paused", "stage": stage, "done": done, "total": total})
        raise EvalPaused(stage)
    if done != total:
        raise RuntimeError(f"{stage}: workers ended with {total - done} images missing ({rcs})")
    shutil.rmtree(od / "_claims", ignore_errors=True)
    log(run, {"event": "stage_done", "stage": stage, "seconds": round(time.time() - t0, 1), "workers": rcs})


# ---------------------------------------------------------------- stage: per-image records --------
def _det_record(points, scores, gt, rule: str) -> dict:
    from e02 import evaluate as ev

    m = ev.match_points(points, scores, gt, rule=rule)
    hits = np.asarray([len(h) for h in m.gt_hit_by])
    return {"gt_score": m.gt_score.astype(np.float64), "fp_score": m.pred_scores[m.pred_is_fp].astype(np.float64),
            "n_points": int(m.n_pred), "n_fp": int(m.pred_is_fp.sum()),
            "n_redundant": int(np.maximum(hits - 1, 0).sum())}


def eval_case(kind: str, split: str, case: str, well: str, line: str) -> dict:
    """Every per-image quantity the aggregate needs (PREREG §6, §7, §7.1, §10)."""
    from e02 import c2_predict as c2p
    from e02 import decode as dc
    from e02 import evaluate as ev
    from e02 import params as P
    from e02 import two_stage as ts
    from e02.predict import load_prob

    gt = ev.load_gt(case, split)
    pa, pb = load_prob(kind, "A", case), load_prob(kind, "B", case)
    assert pa.shape == pb.shape == gt.inst.shape, (case, pa.shape, pb.shape, gt.inst.shape)
    out2 = c2p.load_c2(c2p.out_dir(kind) / f"{case}.npz")

    pts_a, s_a, lab_a, n_comp_a = dc.points_from_prob(pa)
    pts_b, s_b = dc.points_from_heatmap(pb)
    cas = ts.cascade(pa, pb)
    assert np.array_equal(cas["points"], pts_b)
    assert np.array_equal(out2["points"], pts_b) and np.array_equal(out2["b"], s_b), \
        f"{case}: C2 crops were made from a different B map"
    s2, mask2 = ts.c2_scores_and_mask(out2, pb)
    s4, mask4 = ts.c4_scores_and_mask(cas, out2, pb)
    assert np.array_equal(s4, cas["s"])
    masks = {"A": pa >= P.TAU["A"], "C1": ts.cascade_mask(cas, pa, pb), "C2": mask2, "C4": mask4}
    det_in = {"A": (pts_a, s_a), "B": (pts_b, s_b), "C1": (pts_b, cas["s"]), "C2": (pts_b, s2)}

    esd = P.esd_px(gt.areas)
    bins = np.asarray(gt.bins, dtype=object)
    rec = {"case": case, "well": well, "line": line, "n_gt": gt.n, "bins": bins,
           "n_small": int((bins == "small").sum()), "max_area": int(gt.areas.max()) if gt.n else 0,
           "det": {rule: {arm: _det_record(*det_in[arm], gt, rule) for arm in DET_ARMS}
                   for rule in ("one", "multi")},
           "seg": {arm: {"image_dice": ev.dice(m, gt.label), "obj_dice": ev.object_dice(m, gt).astype(np.float64),
                         "mask_px": int(m.sum())} for arm, m in masks.items()},
           "a_components": int(n_comp_a), "a_merged": ev.merged_components(lab_a, gt),
           "a_saturated": int((s_a == 1.0).sum()), "max_esd": float(esd.max()) if gt.n else 0.0}
    return rec


def _eval_case_job(args) -> str:
    kind, split, case, well, line, dst = args
    rec = eval_case(kind, split, case, well, line)
    atomic_bytes(Path(dst), pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
    return case


def stage_cases(run: Run) -> None:
    stage = "cases"
    run.cases_dir.mkdir(parents=True, exist_ok=True)
    clean_tmp(run.cases_dir)
    rows = {r["case"]: r for r in c.image_rows(run.split)}
    todo = [s for s in run.ids if not (run.cases_dir / f"{s}.pkl").exists()]
    if not todo:
        return
    t0, total, start = time.time(), len(run.ids), len(run.ids) - len(todo)
    log(run, {"event": "stage_start", "stage": stage, "todo": len(todo), "total": total, "workers": run.workers})
    jobs = [(run.kind, run.split, s, rows[s]["well"], rows[s]["cell_line"], str(run.cases_dir / f"{s}.pkl"))
            for s in todo]
    done = start
    batch = max(1, 2 * run.workers)
    with ProcessPoolExecutor(max_workers=run.workers) as pool:
        for b0 in range(0, len(jobs), batch):
            check_pause(run, stage)                      # between batches: no batch is left half-done
            for _ in pool.map(_eval_case_job, jobs[b0:b0 + batch]):
                done += 1
            progress(run, stage, done, total, t0, start)
    log(run, {"event": "stage_done", "stage": stage, "seconds": round(time.time() - t0, 1)})


# ---------------------------------------------------------------- assembling records --------------
def load_records(run: Run) -> list[dict]:
    recs = []
    for s in run.ids:
        with open(run.cases_dir / f"{s}.pkl", "rb") as f:
            recs.append(pickle.load(f))
    return recs


def det_data(recs: list[dict], arm: str, rule: str = "one"):
    from jannus.evaluation.detection.metrics import concat_cases

    return concat_cases([(r["case"], r["well"], r["det"][rule][arm]["gt_score"], r["bins"],
                          r["det"][rule][arm]["fp_score"]) for r in recs])


def seg_data(recs: list[dict], arm: str):
    from e02 import evaluate as ev

    return ev.SegData(image_dice=np.asarray([r["seg"][arm]["image_dice"] for r in recs], float),
                      obj_dice=np.concatenate([r["seg"][arm]["obj_dice"] for r in recs]),
                      obj_case=np.concatenate([np.full(r["n_gt"], i, np.int64) for i, r in enumerate(recs)]),
                      obj_bin=np.concatenate([r["bins"] for r in recs]))


def boot_arms(recs: list[dict]) -> dict:
    """C4's detection equals C1's by construction (PREREG §7.1): C4 carries its mask only."""
    from e02 import params as P

    arms = {}
    for a in ("A", "B", "C1", "C2", "C4"):
        arms[a] = {"det": det_data(recs, a) if a in DET_ARMS else None,
                   "seg": seg_data(recs, a) if a in SEG_ARMS else None,
                   "tau": P.TAU[a] if a in DET_ARMS else None}
    return arms


def _w_hash(W: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(W, np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------- stage: bootstrap ----------------
_BOOT: dict = {}


def _boot_init(arms: dict, point: dict) -> None:
    _BOOT["arms"], _BOOT["point"] = arms, point


def _boot_chunk(job) -> int:
    from e02 import evaluate as ev

    s0, W_rows, dst = job
    smp = ev.bootstrap_samples(_BOOT["arms"], _BOOT["point"], W_rows)
    flat = {f"{a}|{k}": v for a in smp for k, v in smp[a].items()}
    dst = Path(dst)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, dst)
    return s0


def stage_bootstrap(run: Run, recs: list[dict], arms: dict):
    from e02 import evaluate as ev
    from e02 import params as P

    stage = "bootstrap"
    run.boot_dir.mkdir(parents=True, exist_ok=True)
    clean_tmp(run.boot_dir)
    W, n_wells, point = ev.bootstrap_weights(arms, run.n_boot, P.BOOT_SEED)
    meta = {"n_resamples": run.n_boot, "seed": P.BOOT_SEED, "n_wells": n_wells, "w_sha256": _w_hash(W),
            "chunk": BOOT_CHUNK, "case_ids_sha256": hashlib.sha256("\n".join(run.ids).encode()).hexdigest()}
    mp = run.boot_dir / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        assert old == meta, f"bootstrap chunks on disk belong to another draw: {old} != {meta}"
    else:
        atomic_json(mp, meta)
    starts = list(range(0, run.n_boot, BOOT_CHUNK))
    todo = [s for s in starts if not (run.boot_dir / f"rows_{s:05d}.npz").exists()]
    if todo:
        t0, start = time.time(), len(starts) - len(todo)
        log(run, {"event": "stage_start", "stage": stage, "chunks_todo": len(todo), "chunks": len(starts),
                  "workers": run.workers})
        # Each row's statistics depend only on that row of W, so chunks computed by different
        # processes, in any order, give exactly the one-pass result (test_run_eval).
        jobs = [(s0, W[s0:s0 + BOOT_CHUNK], str(run.boot_dir / f"rows_{s0:05d}.npz")) for s0 in todo]
        done = start
        if run.workers <= 1:
            _boot_init(arms, point)
            for job in jobs:
                check_pause(run, stage)
                _boot_chunk(job)
                done += 1
                progress(run, stage, done, len(starts), t0, start)
        else:
            with ProcessPoolExecutor(max_workers=run.workers, initializer=_boot_init,
                                     initargs=(arms, point)) as pool:
                for b0 in range(0, len(jobs), run.workers):
                    check_pause(run, stage)          # between rounds: every chunk in flight finishes
                    for _ in pool.map(_boot_chunk, jobs[b0:b0 + run.workers]):
                        done += 1
                    progress(run, stage, done, len(starts), t0, start)
        log(run, {"event": "stage_done", "stage": stage, "seconds": round(time.time() - t0, 1)})
    # assemble in row order
    samples = {a: {k: [] for k in point[a]} for a in point}
    for s0 in starts:
        z = np.load(run.boot_dir / f"rows_{s0:05d}.npz")
        for a in point:
            for k in point[a]:
                samples[a][k].append(z[f"{a}|{k}"])
    samples = {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in samples.items()}
    for a in samples:
        for k, v in samples[a].items():
            assert len(v) == run.n_boot, (a, k, len(v))
    return ev.summarise_bootstrap(point, samples, CONTRASTS, run.n_boot, P.BOOT_SEED, n_wells)


# ---------------------------------------------------------------- stage: aggregate ----------------
def _weights_for(recs, key) -> dict[str, np.ndarray]:
    vals = [key(r) for r in recs]
    return {v: np.asarray([x == v for x in vals], float) for v in sorted(set(vals))}


def tertile(n: int) -> str:
    from e02 import params as P

    lo, hi = P.CONFLUENCE_EDGES                 # an image on an edge goes in the upper tertile
    return "low" if n < lo else ("mid" if n < hi else "high")


def _point_stats(arms: dict, w) -> dict:
    from e02 import evaluate as ev

    return {a: ev.arm_stats(arms[a]["det"], arms[a]["seg"], w, arms[a]["tau"]) for a in arms}


def stage_aggregate(run: Run, recs: list[dict], arms: dict, boot: dict) -> dict:
    from e02 import evaluate as ev
    from e02 import params as P

    # per cell line (point estimates; condition B) and per-line bin counts
    lines = _weights_for(recs, lambda r: r["line"])
    per_line = {ln: _point_stats(arms, w) for ln, w in lines.items()}
    line_bins = {ln: {b: int(sum((r["bins"] == b).sum() for r in recs if r["line"] == ln)) for b in P.BIN_NAMES}
                 for ln in lines}
    verdicts = {}
    for hid, (x, y, metric, kind, b) in ev.HYPOTHESES.items():
        diff = {ln: per_line[ln][x].get(metric, np.nan) - per_line[ln][y].get(metric, np.nan) for ln in lines}
        cnt = {ln: line_bins[ln][b] for ln in lines} if b else None
        verdicts[hid] = ev.verdict(hid, boot["contrasts"][f"{x}-{y}"][metric], diff, cnt)

    # confluence tertiles (descriptive)
    terts = _weights_for(recs, lambda r: tertile(r["n_gt"]))
    per_tertile = {t: _point_stats(arms, w) for t, w in terts.items()}

    # E01 multi-hit rule (sensitivity analysis, point estimates)
    multi = {a: ev.det_stats(det_data(recs, a, "multi"), None, P.TAU[a]) for a in DET_ARMS}
    multi_contrasts = {f"{x}-{y}": {k: multi[x][k] - multi[y][k] for k in multi[x] if not k.startswith("reached")}
                       for x, y in CONTRASTS if x in multi and y in multi}

    # descriptives (§10)
    n_img = len(recs)
    desc = {
        "n_images": n_img, "n_gt": int(sum(r["n_gt"] for r in recs)),
        "gt_by_bin": {b: int(sum((r["bins"] == b).sum() for r in recs)) for b in P.BIN_NAMES},
        "gt_by_line_bin": line_bins,
        "images_by_line": {ln: int(w.sum()) for ln, w in lines.items()},
        "images_by_tertile": {t: int(w.sum()) for t, w in terts.items()},
        "objects_left_out_of_object_dice": int(sum(np.isnan(r["seg"]["A"]["obj_dice"]).sum() for r in recs)),
        "A_merged": {"total": int(sum(r["a_merged"] for r in recs)),
                     "per_image": float(np.mean([r["a_merged"] for r in recs])),
                     "by_tertile": {t: float(np.mean([r["a_merged"] for r in recs if tertile(r["n_gt"]) == t]))
                                    for t in terts}},
        "A_saturated_points": int(sum(r["a_saturated"] for r in recs)),
        "points_per_image": {rule: {a: float(np.mean([r["det"][rule][a]["n_points"] for r in recs]))
                                    for a in DET_ARMS} for rule in ("one", "multi")},
        "redundant_per_image": {rule: {a: float(np.mean([r["det"][rule][a]["n_redundant"] for r in recs]))
                                       for a in DET_ARMS} for rule in ("one", "multi")},
        "c4_detection": "equals C1 by construction (PREREG §7.1); C4's detection rows are C1's",
    }

    # unreached FP targets (flagged, never extrapolated)
    unreached = {a: [i + 1 for i in range(len(P.FP_TARGETS)) if not boot["arms"][a].get(f"reached@t{i + 1}", 1.0)]
                 for a in DET_ARMS}

    # §11 example images, chosen by GT only
    by_name = sorted(recs, key=lambda r: r["case"])
    ordered = sorted(by_name, key=lambda r: r["n_gt"])
    examples = {"most_small_cells": max(by_name, key=lambda r: r["n_small"])["case"],
                "median_cell_count": ordered[(len(ordered) - 1) // 2]["case"],
                "largest_cell": max(by_name, key=lambda r: r["max_area"])["case"]}

    # FROC curves (point estimates, one-hit rule)
    from jannus.evaluation.detection.metrics import froc
    curves = {}
    for a in DET_ARMS:
        d = arms[a]["det"]
        for b in ("all",) + tuple(P.BIN_NAMES):
            cv = froc(d) if b == "all" else froc(d, gt_mask=(d.gt_bin == b))
            for f in ("thresholds", "sensitivity", "fp_per_scan"):
                curves[f"{a}|{b}|{f}"] = getattr(cv, f)
    tmp = run.out / "froc_curves.tmp.npz"
    np.savez_compressed(tmp, **curves)
    os.replace(tmp, run.out / "froc_curves.npz")

    if run.kind == "test":                          # the records must match the built test set
        assert n_img == TEST_N_IMAGES, n_img
        assert desc["gt_by_bin"] == TEST_BIN_COUNTS, desc["gt_by_bin"]
        assert desc["objects_left_out_of_object_dice"] == TEST_EMPTIED_BY_GAP, desc["objects_left_out_of_object_dice"]

    return {"bootstrap": boot, "verdicts": verdicts, "per_line": per_line, "per_tertile": per_tertile,
            "multi_hit": {"arms": multi, "contrasts": multi_contrasts}, "descriptives": desc,
            "unreached_targets": unreached, "examples_by_gt": examples}


# ---------------------------------------------------------------- driver --------------------------
def pending_stages(run: Run) -> list[str]:
    p = []
    if pending_pred(run, "A"):
        p.append("pred_A")
    if pending_pred(run, "B"):
        p.append("pred_B")
    if pending_c2(run):
        p.append("pred_C2")
    if any(not (run.cases_dir / f"{s}.pkl").exists() for s in run.ids):
        p.append("cases")
    p += ["bootstrap", "aggregate"] if not (run.out / "summary.json").exists() else []
    return p


def training_running() -> list[str]:
    from e02 import control

    return [l for l in control.running() if "launch_train" in l]


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {f: c.sha256(here / f) for f in ("run_eval.py", "evaluate.py", "two_stage.py", "decode.py",
                                             "predict.py", "c2_predict.py", "params.py")}


def run_all(run: Run, gpus: list[str]) -> int:
    run.out.mkdir(parents=True, exist_ok=True)
    for stage in GPU_STAGES:                        # no worker is alive yet: any claim left is stale
        shutil.rmtree(pred_dir(run, stage) / "_claims", ignore_errors=True)
    pend = pending_stages(run)
    log(run, {"event": "session_start", "cases": run.kind, "n_cases": len(run.ids), "pending": pend,
              "gpus": [c.GPUS[g] for g in gpus], "prereg_sha256": c.sha256(c.PREREG), "code": code_hashes()})
    if not pend:
        log(run, {"event": "nothing_to_do"})
        return 0
    try:
        for stage in GPU_STAGES:                    # A and B maps first: C2's crops need B's peaks
            if stage in pend:
                stage_gpu(run, stage, gpus)
        if "cases" in pend:
            stage_cases(run)
        check_pause(run, "bootstrap")
        recs = load_records(run)
        arms = boot_arms(recs)
        boot = stage_bootstrap(run, recs, arms)
        check_pause(run, "aggregate")
        log(run, {"event": "stage_start", "stage": "aggregate"})
        summary = stage_aggregate(run, recs, arms, boot)
        summary["meta"] = {"cases": run.kind, "n_cases": len(run.ids), "smoke_model": run.smoke_model,
                           "prereg_sha256": c.sha256(c.PREREG), "code": code_hashes(),
                           "models": _models(run), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(run.out / "summary.json", summary)
        log(run, {"event": "done", "summary": str(run.out / "summary.json"),
                  "verdicts": {h: v["replicated"] for h, v in summary["verdicts"].items()}})
        atomic_json(run.out / "progress.json", {"stage": "done", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return 0
    except EvalPaused:
        return pz.EXIT_PAUSED


def _models(run: Run) -> dict:
    from e02 import c2_predict as c2p
    from e02.predict import out_dir

    out = {}
    for name, d in (("A", out_dir(run.kind, "A")), ("B", out_dir(run.kind, "B")), ("C2", c2p.out_dir(run.kind))):
        for f in ("model.json", "run.json"):
            if (d / f).exists():
                m = json.loads((d / f).read_text(encoding="utf-8"))
                out[name] = {k: m.get(k) for k in ("trainer", "checkpoint_sha256", "epoch", "c2_standin")}
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", choices=["test", "smoke", "preflight"], required=True,
                    help="preflight: 8 training images with the FINAL models (last check before test)")
    ap.add_argument("--smoke-model", action="store_true", help="2-epoch models; smoke cases only")
    ap.add_argument("--gpus", default="3070ti,5060ti",
                    help="GPU stages: one worker per GPU, comma-separated keys of common.GPUS")
    ap.add_argument("--workers", type=int, default=6, help="CPU processes for the cases and bootstrap stages")
    ap.add_argument("--n-boot", type=int, default=None, help="smoke only; the test run uses PREREG's 2,000")
    ap.add_argument("--c2-checkpoint", default=None, help="smoke only: a copy of a C2 checkpoint (any epoch)")
    ap.add_argument("--worker", choices=GPU_STAGES, default=None, help=argparse.SUPPRESS)   # internal
    ap.add_argument("--gpu", choices=sorted(c.GPUS), default=None, help=argparse.SUPPRESS)  # internal
    a = ap.parse_args(argv)
    from e02 import params as P

    if a.cases == "test":
        assert not a.smoke_model, "the smoke models are for the training-image pre-flight only"
        assert a.n_boot in (None, P.N_BOOT), "the test run uses PREREG §8's 2,000 resamples"
        assert a.c2_checkpoint is None, "the test run uses C2's checkpoint_final"
        c.assert_prereg()
    elif a.cases == "preflight":
        assert not a.smoke_model and a.c2_checkpoint is None, "the preflight uses the final models"
    else:
        assert a.smoke_model, "smoke cases are run with the smoke models"
    run = make_run(a.cases, a.smoke_model, a.n_boot or (P.N_BOOT if a.cases == "test" else 50), a.workers,
                   a.c2_checkpoint)
    if a.worker:                                   # a GPU worker started by stage_gpu
        assert a.gpu, "--worker needs --gpu"
        c.select_gpu(c.GPUS[a.gpu])                 # before torch initialises CUDA
        c.assert_paths()
        c.verify_gpu(c.GPUS[a.gpu])
        return work_gpu_stage(run, a.worker, a.gpu)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    assert gpus and all(g in c.GPUS for g in gpus), gpus
    assert a.cases != "test" or len(set(gpus)) == len(gpus), "one worker per GPU in the test run"
    if "3070ti" in gpus and training_running():
        raise SystemExit("an E02 training job is running on the 3070 Ti; evaluation waits for the queue")
    c.assert_paths()
    return run_all(run, gpus)


if __name__ == "__main__":
    sys.exit(main())
