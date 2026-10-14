"""The conditional probe and logit lens stage (docs/DECISIONS.md D46): targets, candidates and a toy run."""

import json

import numpy as np

from directions.config import load_varprobe_config
from directions.varmap import all_exprs
from directions.varprobe import (
    TARGETS,
    lens_eligible,
    run_varprobe,
    target_spec,
    variables,
)


def test_allowed_classes_hold_the_target_and_follow_the_condition():
    for task in ("two_lane", "one_lane"):
        v = variables(all_exprs(task))
        for name in TARGETS[task]:
            y, cond, allowed = target_spec(name, task, v)
            assert allowed[np.arange(len(y)), y].all(), (task, name)
            # every allowed class is consistent with the condition: same condition -> same allowed set
            for c in np.unique(cond):
                rows = allowed[cond == c]
                assert (rows == rows[0]).all(), (task, name, c)
        y, cond, allowed = target_spec("S1|Y", task, v)
        assert allowed.sum(axis=1).min() >= 1
        if task == "one_lane":  # S1 = Y + c: ten candidates per answer
            assert (allowed.sum(axis=1) == 10).all()


def test_lens_candidates_exclude_the_answer():
    v = variables(all_exprs("two_lane"))
    for name in ("S1", "S2", "a", "d"):
        e = lens_eligible(name, v)
        assert (v[name][e] != v["Y"][e]).all() and (v[name][e] <= 9).all()
    assert lens_eligible("Y", v).all()


def test_toy_run_writes_the_readouts(tmp_path):
    cfg = load_varprobe_config("configs/smoke_toy_varprobe.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.n_examples, cfg.n_perm = 40, 8
    root = run_varprobe(cfg, run_id="toy")
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["capture"]["identical"] and meta["determinism_check"]["probe"]["identical"]
    for task in ("two_lane", "one_lane"):
        r = json.loads((root / "core" / "tasks" / task / "varprobe.json").read_text())
        assert r["labels"][-1] == "␣" and set(r["probes"]) == set(TARGETS[task])
        M = len(r["read_points"])
        assert all(len(p["accuracy"]) == M for p in r["probes"].values())
        assert r["positive_control"]["carried"] == [True]  # the a token holds its own digit
        assert (root / "figures" / f"{task}.png").exists()
        z = np.load(root / "core" / "tasks" / task / "varprobe_arrays.npz")
        assert z["lens_digit"].shape == (M, r["n_kept"])
