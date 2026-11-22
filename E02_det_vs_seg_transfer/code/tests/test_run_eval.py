"""run_eval's pause/resume machinery, on synthetic data. Never touches the real results\\PAUSE flag
(that would pause a training run): every test redirects it to a temp folder."""
import json

import numpy as np
import pytest

from e02 import evaluate as ev
from e02 import jobqueue as jq
from e02 import params as P
from e02 import pause as pz
from e02 import run_eval as re_

from test_evaluate import _toy_data


@pytest.fixture
def flag(tmp_path, monkeypatch):
    """The smoke Run in these tests pauses on tmp_path/eval/PAUSE; the real flag is redirected too."""
    monkeypatch.setattr(pz, "PAUSE_FLAG", tmp_path / "REAL_PAUSE_REDIRECTED")
    f = tmp_path / "eval" / "PAUSE"
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def _arms():
    a, b = _toy_data(0.0), _toy_data(0.2)
    seg = ev.SegData(np.full(12, 0.8), np.full(36, 0.5), np.repeat(np.arange(12), 3),
                     np.array(["small", "medium", "large"] * 12, dtype=object))
    return {"A": {"det": a, "seg": seg, "tau": 0.5}, "B": {"det": b, "seg": None, "tau": 0.3}}


def _run(tmp_path, n_boot=50):
    return re_.Run(kind="smoke", ids=[f"c{i}" for i in range(12)], folder="imagesTr", split="Tr",
                   out=tmp_path / "eval", smoke_model=True, n_boot=n_boot, workers=1)


def test_bootstrap_pause_then_resume_equals_one_pass(tmp_path, flag, monkeypatch):
    monkeypatch.setattr(re_, "BOOT_CHUNK", 10)
    monkeypatch.setattr(re_, "CONTRASTS", (("B", "A"),))
    arms, run = _arms(), _run(tmp_path)
    real = ev.bootstrap_samples
    calls = []

    def pause_after_two(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            flag.touch()                         # owner pauses during chunk 2
        return real(*a, **k)

    monkeypatch.setattr(ev, "bootstrap_samples", pause_after_two)
    with pytest.raises(re_.EvalPaused):
        re_.stage_bootstrap(run, [], arms)
    assert sorted(p.name for p in run.boot_dir.glob("rows_*.npz")) == ["rows_00000.npz", "rows_00010.npz"]
    flag.unlink()                                # resume
    got = re_.stage_bootstrap(run, [], arms)
    assert len(calls) == 5                       # 2 before the pause + the 3 missing chunks
    want = ev.paired_bootstrap(arms, [("B", "A")], n=50, seed=P.BOOT_SEED)
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


def test_bootstrap_refuses_chunks_from_another_draw(tmp_path, flag, monkeypatch):
    monkeypatch.setattr(re_, "CONTRASTS", (("B", "A"),))
    arms = _arms()
    re_.stage_bootstrap(_run(tmp_path, 40), [], arms)
    with pytest.raises(AssertionError, match="another draw"):
        re_.stage_bootstrap(_run(tmp_path, 60), [], arms)


def test_partial_files_are_cleaned_not_skipped(tmp_path):
    d = tmp_path / "cases"
    d.mkdir()
    (d / "x.pkl.tmp").write_bytes(b"half")
    (d / "rows_00000.tmp.npz").write_bytes(b"half")
    (d / "y.pkl").write_bytes(b"done")
    assert re_.clean_tmp(d) == 2
    assert [p.name for p in d.iterdir()] == ["y.pkl"]
    re_.atomic_bytes(d / "z.pkl", b"ok")
    assert (d / "z.pkl").read_bytes() == b"ok" and not list(d.glob("*.tmp"))


def test_check_pause(tmp_path, flag):
    run = _run(tmp_path)
    re_.check_pause(run, "cases")                # no flag: nothing happens
    flag.touch()
    with pytest.raises(re_.EvalPaused):
        re_.check_pause(run, "cases")
    assert '"paused"' in (run.out / "run_log.jsonl").read_text()


def test_pin_model_refuses_a_second_checkpoint(tmp_path):
    d = tmp_path / "pred"
    re_.pin_model(d, {"checkpoint_sha256": "aaa", "arm": "A"})
    re_.pin_model(d, {"checkpoint_sha256": "aaa", "arm": "A"})         # same model resumes
    with pytest.raises(SystemExit, match="refusing to mix"):
        re_.pin_model(d, {"checkpoint_sha256": "bbb", "arm": "A"})


def test_pin_model_reads_preflight_run_json(tmp_path):
    d = tmp_path / "pred"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"checkpoint_sha256": "aaa"}))
    with pytest.raises(SystemExit):
        re_.pin_model(d, {"checkpoint_sha256": "bbb"})


def test_tertile_edges_go_up():
    lo, hi = P.CONFLUENCE_EDGES
    assert [re_.tertile(int(x)) for x in (lo - 1, lo, hi - 1, hi)] == ["low", "mid", "mid", "high"]


def test_queue_runs_eval_only_after_go_ahead(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jq, "run_job", lambda m, job, name: calls.append(m) or 0)
    monkeypatch.setattr(jq, "LOG", tmp_path / "queue.jsonl")
    monkeypatch.setattr(jq, "EVAL_GO", tmp_path / "EVAL_GO")
    assert jq.main() == 0 and "e02.run_eval" not in calls
    (tmp_path / "EVAL_GO").touch()
    calls.clear()
    assert jq.main() == 0 and calls[-1] == "e02.run_eval" and calls.count("e02.launch_train") == len(jq.JOBS)


def test_queue_stops_before_eval_if_training_paused(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jq, "run_job", lambda m, job, name: calls.append(m) or (pz.EXIT_PAUSED if m.endswith("train") else 0))
    monkeypatch.setattr(jq, "LOG", tmp_path / "queue.jsonl")
    (tmp_path / "EVAL_GO").touch()
    monkeypatch.setattr(jq, "EVAL_GO", tmp_path / "EVAL_GO")
    assert jq.main() == pz.EXIT_PAUSED and calls == ["e02.launch_train"]


def test_test_mode_guards():
    with pytest.raises(AssertionError):
        re_.main(["--cases", "test", "--smoke-model"])
    with pytest.raises(AssertionError):
        re_.main(["--cases", "test", "--n-boot", "10"])
    with pytest.raises(AssertionError):
        re_.main(["--cases", "smoke"])           # smoke cases need the smoke models


def test_smoke_flag_is_not_the_queue_flag(tmp_path):
    assert _run(tmp_path).pause_flag != pz.PAUSE_FLAG
    assert re_.Run("test", [], "imagesTs", "Ts", tmp_path, False, 2000, 1).pause_flag == pz.PAUSE_FLAG


def test_parallel_bootstrap_equals_one_pass(tmp_path, flag, monkeypatch):
    monkeypatch.setattr(re_, "BOOT_CHUNK", 10)
    monkeypatch.setattr(re_, "CONTRASTS", (("B", "A"),))
    arms, run = _arms(), _run(tmp_path)
    run.workers = 2                                    # separate processes, chunks in any order
    got = re_.stage_bootstrap(run, [], arms)
    want = ev.paired_bootstrap(arms, [("B", "A")], n=50, seed=P.BOOT_SEED)
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


def test_claims_are_exclusive(tmp_path):
    assert re_._claim(tmp_path, "img1", "gpu0")
    assert not re_._claim(tmp_path, "img1", "gpu1")   # the other worker skips it
    assert re_._claim(tmp_path, "img2", "gpu1")
    assert (tmp_path / "img1.lock").read_text() == "gpu0"


def test_test_run_needs_distinct_gpus():
    with pytest.raises(AssertionError):
        re_.main(["--cases", "test", "--gpus", "3070ti,3070ti"])
