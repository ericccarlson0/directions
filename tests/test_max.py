"""The max task (docs/DECISIONS.md D55): the expressions, the D45 sources, the probe targets, the per-variable DAS
sources, and toy runs of the four stages."""

from __future__ import annotations

import json

import numpy as np
import pytest

import directions.das as das
import directions.varmap as vm
from directions.config import ModelConfig, ToyModelConfig, VarmapConfig, load_das_config, load_varprobe_config, load_varsplit_config
from directions.model import ModelBackend
from directions.varprobe import TARGETS, lens_eligible, run_varprobe, target_spec, variables
from directions.varsplit import run_varsplit


@pytest.fixture(scope="module")
def toy():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                    toy=ToyModelConfig(family="qwen3")), run_seed=0)


def test_expressions():
    ex = vm.all_exprs("max")
    assert len(ex) == 2640 and len(set(ex)) == 2640
    assert all(e.S1 <= 9 and e.S2 <= 9 and e.S1 != e.S2 and e.Y == max(e.S1, e.S2) and e.C == int(e.S1 > e.S2) for e in ex)
    e = vm.Expr(3, 4, 2, 6, op="max")
    assert e.text() == "max(3+4, 2+6)" and e.Y == 8 and e.C == 0 and e.L == 7
    assert {e.text()[i]: name for i, name in e.digit_slots().items()} == {"3": "a", "4": "b", "2": "c", "6": "d"}
    assert vm.Expr(3, 4, 2, 6) != e  # the operation is part of the expression


def test_query_tokens_label_the_digits(toy):
    demos = [vm.Expr(1, 2, 0, 1, op="max"), vm.Expr(5, 4, 3, 3, op="max")]
    base, src = vm.Expr(4, 1, 2, 6, op="max"), vm.Expr(0, 9, 2, 6, op="max")
    pb, ps = vm.render(demos, base), vm.render(demos, src)
    assert pb.prompt.endswith("max(4+1, 2+6) = ") and pb.target == "8"
    pos_b, _, labels = vm.query_tokens(toy, pb, base)
    pos_s, _, _ = vm.query_tokens(toy, ps, src)
    assert pos_b == pos_s and labels[-1] == "␣" and [lab for lab in labels if lab in "abcd"] == list("abcd")


def test_predictions_and_sources():
    base = vm.Expr(5, 0, 1, 2, op="max")  # S1 5, S2 3, Y 5, C 1
    p = vm.predictions(base, vm.Expr(1, 1, 7, 1, op="max"))  # S1' 2, S2' 8, C' 0
    assert p == {"S1": 3, "S2": 8, "C": 3, "both": 8, "none": 5}
    bases, sources, demos, info = vm.sample_bases(1, "max", 30, 2, 8)
    assert len(bases) == 30 and info["n_expressions"] == 2640
    for b, srcs in zip(bases, sources):
        assert set(srcs) == set(vm.SOURCE_TYPES["max"]) and all(vm.valid_source(k, b, s) for k, s in srcs.items())
        assert vm.prediction("S1", b, srcs["S1"]) != b.Y and vm.prediction("S2", b, srcs["S2"]) != b.Y
        assert srcs["S1"].S2 == b.S2 and srcs["S2"].S1 == b.S1 and srcs["both"].Y != b.Y
        assert srcs["S1_same"].S1 == b.S1 and (srcs["S1_same"].a, srcs["S1_same"].b) != (b.a, b.b)
    # a loser that cannot overtake a 9 has no S1 source: max(S1', 9) is 9 for every S1'
    assert not any(vm.valid_source("S1", vm.Expr(1, 1, 4, 5, op="max"), s) for s in vm.all_exprs("max"))


def test_probe_targets_hold_their_classes():
    v = variables(vm.all_exprs("max"))
    assert (v["L"] == np.minimum(v["S1"], v["S2"])).all() and (v["C"] == (v["S1"] > v["S2"])).all()
    for name in TARGETS["max"]:
        y, cond, allowed = target_spec(name, "max", v)
        assert allowed[np.arange(len(y)), y].all(), name
    assert lens_eligible("L", v).all()  # the smaller sum is never the answer


def test_das_sources_follow_their_definitions():
    ex = vm.all_exprs("max")
    D = das.digits_of(ex)
    rng = np.random.default_rng(0)
    for base in [ex[i] for i in rng.choice(len(ex), 40, replace=False)]:
        for kind in das.TASK_SOURCE_KINDS["max"]:
            for i in np.flatnonzero(das.source_mask(kind, base, D))[:20]:
                s = ex[i]
                t = {v: das.targets(v, base, s) for v in das.TASK_VARIABLES["max"]}
                if kind.startswith("cause_"):
                    v = kind[len("cause_"):]
                    assert t[v] != base.Y and 0 <= t[v] <= 9
                    assert all(t[v] != t[u] for u in ("S1", "S2", "C") if u != v)
                elif kind in ("only_S1", "only_S2", "only_b", "same_C"):
                    assert s.Y != base.Y
                if kind == "only_S2":
                    assert (s.a, s.b) == (base.a, base.b) and s.S2 != base.S2
                if kind == "same_C":
                    assert s.C == base.C and s.S1 != base.S1 and s.S2 != base.S2
                if kind == "same_Y":
                    assert s.Y == base.Y and s.S1 != base.S1 and s.S2 != base.S2
    b = vm.Expr(4, 1, 2, 7, op="max")  # S1 5 loses to 9: nothing can make S1 matter without making it the answer
    assert das.targets("C", b, vm.Expr(8, 1, 0, 2, op="max")) == 5 and das.targets("S1", b, vm.Expr(8, 1, 0, 2, op="max")) == 9


def test_sample_splits_keeps_bases_without_some_sources():
    kinds = {"train": tuple(k for k in das.TASK_SOURCE_KINDS["max"] if k.startswith("cause")),
             "test": das.TASK_SOURCE_KINDS["max"]}
    splits, info = das.sample_splits(3, {"train": 60, "test": 60}, 2, 8, kinds, task="max")
    assert len(splits["train"].bases) == 60 and len(splits["test"].bases) == 60
    missing = sum(x is None for k in kinds["test"] for x in splits["test"].sources[k])
    assert missing > 0  # max leaves some bases without a source of some kind
    for sp in splits.values():
        for i, b in enumerate(sp.bases):
            assert any(sp.sources[k][i] is not None for k in sp.sources if k.startswith("cause"))


def test_varmap_toy_run(tmp_path):
    cfg = VarmapConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                       expressions=["max"], n_bases=4, n_shots=2, n_demo_pool=8, read_points=[0, 4], n_boot=30,
                       min_accuracy=0.0, gate=False)
    root = vm.run_varmap(cfg, run_id="smoke_max")
    r = json.loads((root / "core" / "tasks" / "max" / "varmap.json").read_text())
    assert r["template"] == "max(a+b, c+d) = " and set(r["rates"]) == set(vm.SOURCE_TYPES["max"])
    assert set(r["rates"]["both"]["hypotheses"]) == {"S1", "S2", "C", "both", "none"}
    assert (root / "figures" / "max.png").exists()


def test_varprobe_and_varsplit_toy_runs(tmp_path):
    cfg = load_varprobe_config("configs/smoke_toy_varprobe.yaml")
    cfg.output_dir, cfg.expressions, cfg.n_examples, cfg.n_perm = str(tmp_path), ["max"], 40, 8
    r = json.loads((run_varprobe(cfg, run_id="pmax") / "core" / "tasks" / "max" / "varprobe.json").read_text())
    assert set(r["probes"]) == set(TARGETS["max"]) and "L" in r["lens"]["candidates"]
    cfg = load_varsplit_config("configs/smoke_toy_varsplit.yaml")
    cfg.output_dir, cfg.expressions = str(tmp_path), ["max"]
    root = run_varsplit(cfg, run_id="smax")
    r = json.loads((root / "core" / "tasks" / "max" / "varsplit.json").read_text())
    assert set(r["probe"]["targets"]) == set(TARGETS["max"]) and "both" in r["heads"]["blocks"]["2"]["types"]


def test_das_toy_run(tmp_path):
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.task, cfg.variables = str(tmp_path), "max", ["S1", "S2", "C", "Y", "a"]
    cfg.sites, cfg.control_site, cfg.ks, cfg.null_k, cfg.epochs = [1, 4], 1, [2], 2, 1
    root = das.run_das(cfg, run_id="dmax")
    r = json.loads((root / "core" / "das.json").read_text())
    assert r["task"] == "max" and set(r["cells"]) == {f"{s}/{v}" for s in (1, 4) for v in cfg.variables}
    assert set(r["n"]["test"]["kept_by_var"]) == set(cfg.variables)
    assert set(r["cells"]["4/C"]["grid"]["2"]["isolate"]) == {"same_C"}
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["training"]["identical"] and meta["determinism_check"]["evaluation"]["identical"]


def test_das_two_token_shared_toy_run(tmp_path):
    """The 14B's design (D55 amended): max's per-variable sources with D54's shared subspace at the answer token
    and =; the swap spans the two tokens and determinism holds."""
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.task, cfg.variables = str(tmp_path), "max", ["S1", "S2", "C", "Y"]
    cfg.positions, cfg.shared_basis = "answer_eq", True
    cfg.sites, cfg.control_site, cfg.ks, cfg.null_k, cfg.epochs = [1, 4], 1, [2], 2, 1
    root = das.run_das(cfg, run_id="dmax_eq")
    r = json.loads((root / "core" / "das.json").read_text())
    assert r["positions"] == "answer_eq" and r["shared_basis"] and len(r["swap"]["tokens"]) == 2
    assert set(r["n"]["test"]["kept_by_var"]) == {"S1", "S2", "C", "Y"}
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["training"]["identical"] and meta["determinism_check"]["evaluation"]["identical"]
