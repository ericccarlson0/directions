"""The max/min task (docs/DECISIONS.md D57): the expressions, the operator swap and D45's swaps on the base's operator,
the balanced demonstrations, the operator token's label, the operator readouts and their readings, the probe targets,
and toy runs of the map and the probe."""

from __future__ import annotations

import json

import numpy as np
import pytest

import directions.varmap as vm
from directions.config import ModelConfig, ToyModelConfig, VarmapConfig, load_varprobe_config
from directions.model import ModelBackend
from directions.varprobe import TARGETS, run_varprobe, sample_probe_set, target_spec, variables


@pytest.fixture(scope="module")
def toy():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                    toy=ToyModelConfig(family="qwen3")), run_seed=0)


def test_expressions():
    ex = vm.all_exprs("maxmin")
    assert len(ex) == 5280 and len(set(ex)) == 5280
    assert sum(e.op == "max" for e in ex) == sum(e.op == "min" for e in ex) == 2640
    assert {(e.a, e.b, e.c, e.d) for e in ex if e.op == "max"} == {(e.a, e.b, e.c, e.d) for e in ex if e.op == "min"}
    assert [e for e in ex if e.op == "max"] == vm.all_exprs("max")  # D55's expressions, in D55's order
    e = vm.Expr(3, 4, 2, 6, op="min")
    assert e.text() == "min(3+4, 2+6)" and e.Y == 7 and e.C == 0 and e.L == 7
    assert {e.text()[i]: name for i, name in e.digit_slots().items()} == {"3": "a", "4": "b", "2": "c", "6": "d"}


def test_predictions_under_min():
    base = vm.Expr(5, 0, 1, 2, op="min")  # S1 5, S2 3, Y 3, C 1
    p = vm.predictions(base, vm.Expr(1, 1, 7, 1, op="min"))  # S1' 2, S2' 8, C' 0
    assert p == {"S1": 2, "S2": 5, "C": 5, "both": 2, "none": 3}
    # under max the same pair predicts the larger sums
    assert vm.predictions(vm.Expr(5, 0, 1, 2, op="max"), vm.Expr(1, 1, 7, 1, op="max"))["C"] == 3


def test_operator_sources_change_only_the_operator():
    bases, sources, demos, info = vm.sample_bases(2, "maxmin", 40, 8, 96)
    assert len(bases) == 40 and info["n_expressions"] == 5280
    assert {b.op for b in bases} == {"max", "min"}
    for b, srcs in zip(bases, sources):
        assert set(srcs) == set(vm.SOURCE_TYPES["maxmin"]) and all(vm.valid_source(k, b, s) for k, s in srcs.items())
        op = srcs["op"]
        assert (op.a, op.b, op.c, op.d) == (b.a, b.b, b.c, b.d) and op.op != b.op
        assert vm.prediction("op", b, op) == op.Y != b.Y  # the sums are unequal: the other operation moves the answer
        assert all(srcs[k].op == b.op for k in srcs if k != "op")  # D45's swaps keep the base's operator
        assert vm.prediction("S1", b, srcs["S1"]) != b.Y and vm.prediction("S2", b, srcs["S2"]) != b.Y
    for d in demos:
        assert len(d) == 8 and sum(e.op == "max" for e in d) == 4
    # a source of another operator is never valid for D45's swaps
    b = vm.Expr(1, 2, 3, 3, op="max")
    assert not vm.valid_source("S1", b, vm.Expr(4, 2, 3, 3, op="min")) and vm.valid_source("S1", b, vm.Expr(4, 3, 3, 3, op="max"))


def test_balanced_demos():
    pool = vm.all_exprs("maxmin")[:: 50]
    rng = np.random.default_rng(0)
    orders = set()
    for _ in range(20):
        d = vm.balanced_demos(pool, 8, rng)
        assert len(set(d)) == 8 and sum(e.op == "min" for e in d) == 4
        orders.add(tuple(e.op for e in d))
    assert len(orders) > 1  # the operators come in random order, not in blocks
    with pytest.raises(ValueError, match="even"):
        vm.balanced_demos(pool, 7, rng)
    with pytest.raises(ValueError, match="too few"):
        vm.balanced_demos([e for e in pool if e.op == "max"] + [vm.Expr(1, 2, 3, 3, op="min")], 4, rng)


def test_the_operator_token_is_labelled(toy):
    demos = [vm.Expr(1, 2, 0, 1, op="max"), vm.Expr(5, 4, 3, 3, op="min")]
    base, src = vm.Expr(4, 1, 2, 6, op="max"), vm.Expr(4, 1, 2, 6, op="min")
    pb, ps = vm.render(demos, base), vm.render(demos, src)
    assert pb.prompt.endswith("max(4+1, 2+6) = ") and ps.prompt.endswith("min(4+1, 2+6) = ") and ps.target == "5"
    pos_b, ids_b, lab_b = vm.query_tokens(toy, pb, base, "op")
    pos_s, ids_s, lab_s = vm.query_tokens(toy, ps, src, "op")
    assert pos_b == pos_s and lab_b == lab_s and lab_b[-1] == "␣"
    n_op = sum(lab == "op" for lab in lab_b)
    assert n_op >= 1 and lab_b[:n_op] == ["op"] * n_op and [lab for lab in lab_b if lab in "abcd"] == list("abcd")
    # only the operator's tokens differ between the two queries
    assert all(x == y for x, y, lab in zip(ids_b, ids_s, lab_b) if lab != "op") and ids_b != ids_s
    # without the label (D55's max task) the query is labelled as before: the operator word is not "op"
    assert "op" not in vm.query_tokens(toy, pb, base)[2]


LABELS = ["op", "(", "a", "+", "b", ",", " ", "c", "+", "d", ")", " =", "␣"]


def _read(present_at: dict[int, list[int]], gather: int, M: int = 12) -> dict[str, np.ndarray]:
    """present_O at (read point, token) pairs, and both sums present at the answer token from ``gather`` on."""
    P = len(LABELS)
    present = np.zeros((M, P), dtype=bool)
    for j, rps in present_at.items():
        present[rps, j] = True
    count = np.zeros((M, P), dtype=int)
    count[gather:, P - 1] = 2
    return {"present_O": present, "count": count}


@pytest.mark.parametrize("present_at, gather, reading, R, O_out", [
    ({0: list(range(7)), 12: list(range(6, 12))}, 6, "taken up where the operands meet", 6, 6),
    ({0: list(range(9)), 12: list(range(3, 12))}, 6, "carried ahead and held", 3, 8),
    ({0: list(range(9)), 12: list(range(8, 12))}, 6, "late", 8, 8),
    ({0: list(range(4)), 11: list(range(3, 7)), 12: list(range(6, 12))}, 6, "relayed", 6, 3),
    ({0: list(range(3)), 12: list(range(6, 12))}, 6, "relayed (spread over context)", 6, 2),
    ({0: list(range(12))}, 6, "undecided (no R_O or no G)", None, 11),
])
def test_operator_readouts(present_at, gather, reading, R, O_out):
    out = vm.operator_readouts(_read(present_at, gather), LABELS, list(range(12)))
    assert out["reading"] == reading and out["R_O"] == R and out["O_out"] == O_out and out["G"] == gather
    if reading == "relayed":
        assert out["route"] == {"11: =": [3, 4, 5, 6]}


def test_probe_targets_condition_on_the_operator():
    ex = vm.all_exprs("maxmin")
    v = variables(ex)
    assert (v["O"] == np.array([e.op == "max" for e in ex])).all()
    assert "O" not in variables(vm.all_exprs("max"))  # D55's probe arrays are unchanged
    for name in TARGETS["maxmin"]:
        y, cond, allowed = target_spec(name, "maxmin", v)
        assert allowed[np.arange(len(y)), y].all(), name
        # the null permutes the target within its conditioning classes: every class's allowed set is one set
        for c in np.unique(cond):
            assert (allowed[cond == c] == allowed[cond == c][0]).all(), name
    y, cond, allowed = target_spec("S1|Y", "maxmin", v)
    assert len(np.unique(cond)) > len(np.unique(v["Y"]))  # given Y and the operator
    assert not allowed[v["O"] == 1][:, 9].all() and allowed[v["O"] == 0][:, 9].all()  # min: S1 >= Y, any S1 may be 9
    _chosen, demos, info = sample_probe_set(3, "maxmin", 50, 8, 96)
    assert info["n_expressions"] == 5280 and all(sum(e.op == "max" for e in d) == 4 for d in demos)


def test_varmap_toy_run(tmp_path):
    cfg = VarmapConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                       expressions=["maxmin"], n_bases=6, n_shots=2, n_demo_pool=8, read_points=[0, 2, 4], n_boot=30,
                       min_accuracy=0.0, gate=False)
    root = vm.run_varmap(cfg, run_id="smoke_maxmin")
    r = json.loads((root / "core" / "tasks" / "maxmin" / "varmap.json").read_text())
    assert r["template"] == "op(a+b, c+d) = " and set(r["rates"]) == set(vm.SOURCE_TYPES["maxmin"])
    assert set(r["rates"]["both"]["hypotheses"]) == {"S1", "S2", "C", "both", "none"}
    assert "transfer" in r["rates"]["op"] and "present_O" in r["reading"]
    assert set(r["accuracy_by_op"]) == {"max", "min"} and r["labels"][0] == "op"
    assert set(r["operator"]) == {"R_O", "G", "O_out", "route", "reading"}
    s = json.loads((root / "core" / "summary.json").read_text())
    assert s["tasks"]["maxmin"]["operator"] == r["operator"]
    assert (root / "figures" / "maxmin.png").exists()


def test_varprobe_toy_run(tmp_path):
    cfg = load_varprobe_config("configs/smoke_toy_varprobe.yaml")
    cfg.output_dir, cfg.expressions, cfg.n_examples, cfg.n_perm, cfg.n_shots = str(tmp_path), ["maxmin"], 40, 8, 2
    root = run_varprobe(cfg, run_id="pmaxmin")
    r = json.loads((root / "core" / "tasks" / "maxmin" / "varprobe.json").read_text())
    assert set(r["probes"]) == set(TARGETS["maxmin"]) and r["labels"][0] == "op"
    assert set(r["lens"]["candidates"]) == {"Y", "S1", "S2", "a", "b", "c", "d"}
    arrays = np.load(root / "core" / "tasks" / "maxmin" / "varprobe_arrays.npz")
    assert "var__O" in arrays and set(np.unique(arrays["var__O"])) == {0, 1}


def test_d57_reading_script(tmp_path, monkeypatch, capsys):
    """scripts/d57_operator.py reads a map and a probe of one model (the toy's rows are all "fragile", because the
    untrained toy never gives the base answer: the re-read here drops none)."""
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("d57_operator", scripts / "d57_operator.py")
    d57 = importlib.util.module_from_spec(spec)
    sys.modules["d57_operator"] = d57
    spec.loader.exec_module(d57)
    monkeypatch.setattr(d57, "fragile_rows", lambda result, arrays, k, ids: np.array([], dtype=int))
    cfg = VarmapConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                       expressions=["maxmin"], n_bases=6, n_shots=2, n_demo_pool=8, read_points=None, n_boot=30,
                       min_accuracy=0.0, gate=False)
    vm.run_varmap(cfg, run_id="varmap_maxmin_qwen3_4b_toy")
    p = load_varprobe_config("configs/smoke_toy_varprobe.yaml")
    p.output_dir, p.expressions, p.n_examples, p.n_perm = str(tmp_path), ["maxmin"], 40, 8
    run_varprobe(p, run_id="varprobe_maxmin_qwen3_4b_toy")
    assert d57.main([str(tmp_path), "--out", str(tmp_path / "d57.json")]) == 0
    s = json.loads((tmp_path / "d57.json").read_text())["qwen3_4b"]
    assert s["handover"] == 24 and s["gather_max_d55"] == 24
    assert s["map"]["as_run"] == s["map"]["fragile_dropped"]  # nothing dropped: the same reading
    assert set(s["probe"]) >= {"O|Y", "C|Y", "S1|Y", "S2|Y", "O|Y_carried_from_G_on"}
    assert "qwen3_4b: accuracy" in capsys.readouterr().out
