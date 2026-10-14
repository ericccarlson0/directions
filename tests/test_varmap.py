"""The variable map (docs/DECISIONS.md D45): the expression tasks, their sources and predictions, the token
alignment, the readouts, and a smoke run on the toy model."""

import json

import numpy as np
import pytest

from directions.config import ModelConfig, ToyModelConfig, VarmapConfig
from directions.model import ModelBackend
from directions import varmap as vm


def test_expressions_and_their_answers():
    e = vm.Expr(3, 4, 2, 1)
    assert (e.S1, e.S2, e.Y, e.text()) == (7, 3, 4, "(3+4) - (2+1)")
    t = vm.Expr(3, 4, 5)
    assert (t.S1, t.S2, t.Y, t.text()) == (7, 5, 2, "(3+4) - 5")
    for task in vm.TASKS:
        exprs = vm.all_exprs(task)
        assert exprs and all(0 <= x.Y <= 9 for x in exprs)
    assert len(vm.all_exprs("one_lane")) < len(vm.all_exprs("two_lane"))


@pytest.mark.parametrize("task", vm.TASKS)
def test_sources_change_exactly_what_their_type_says(task):
    pool = vm.all_exprs(task)
    rng = np.random.default_rng(0)
    for base in pool[:: max(1, len(pool) // 40)]:
        for kind in vm.SOURCE_TYPES[task]:
            src = vm.draw_source(kind, base, pool, rng)
            if src is None:
                continue
            assert vm.valid_source(kind, base, src)
            pred = vm.prediction(kind, base, src)
            assert 0 <= pred <= 9
            if kind == "S1":
                assert (src.c, src.d) == (base.c, base.d) and src.S1 != base.S1 and pred == src.Y
            if kind == "S2":
                assert (src.a, src.b) == (base.a, base.b) and src.S2 != base.S2 and pred == src.Y
            if kind.endswith("_same"):
                assert pred == base.Y and src.Y == base.Y and src != base
            if kind == "both":
                p = vm.predictions(base, src)
                assert len({p["S1"], p["S2"], p["both"], p["none"]}) == 4 and pred == src.Y


@pytest.fixture(scope="module")
def toy():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                    toy=ToyModelConfig(family="qwen3")), run_seed=0)


def test_query_tokens_are_aligned_and_labelled(toy):
    demos = [vm.Expr(1, 2, 0, 1), vm.Expr(5, 4, 3, 3)]
    base, src = vm.Expr(3, 4, 2, 1), vm.Expr(5, 2, 2, 1)
    pb, ps = vm.render(demos, base), vm.render(demos, src)
    assert pb.prompt.endswith("(3+4) - (2+1) = ") and pb.target == "4"
    assert pb.prompt.splitlines()[0] == "(1+2) - (0+1) = 2"
    pos_b, ids_b, labels = vm.query_tokens(toy, pb, base)
    pos_s, ids_s, _ = vm.query_tokens(toy, ps, src)
    assert pos_b == pos_s and len(labels) == len("(3+4) - (2+1) = ")  # the toy tokenizer is per character
    assert labels[1] == "a" and labels[3] == "b" and labels[9] == "c" and labels[11] == "d" and labels[-1] == "␣"
    digit_cols = [j for j, lab in enumerate(labels) if lab in "abcd"]
    assert [j for j in range(len(ids_b)) if ids_b[j] != ids_s[j]] == [j for j in digit_cols if ids_b[j] != ids_s[j]]
    assert pos_b[-1] == len(toy._encode_prompt(pb)[0]) - 1  # the answer position is the query token


def test_bootstrap_and_reading():
    rng = np.random.default_rng(1)
    flips = np.zeros((40, 3, 2))
    flips[:, 2, 1] = 1.0
    r = vm.bootstrap_rate(flips, rng, 200, 0.05)
    assert r["mean"][2, 1] == 1.0 and r["lo"][2, 1] == 1.0 and r["hi"][0, 0] == 0.0
    rates = {"S1": {"flip": {"lo": np.array([[0.6, 0.1], [0.7, 0.0], [0.9, 0.9]])}},
             "S2": {"flip": {"lo": np.array([[0.0, 0.1], [0.6, 0.0], [0.2, 0.9]])}},
             "S1_same": {"change": {"hi": np.array([[0.9, 0.0], [0.1, 0.0], [0.2, 0.0]])}}}
    read = vm.reading(rates, "one_lane", 0.5, 0.25)
    assert read["count"].tolist() == [[1, 0], [2, 0], [1, 2]]
    assert read["value_S1"].tolist() == [[False, False], [True, False], [True, True]]
    assert read["digits_S1"].tolist() == [[True, False], [False, False], [False, False]]
    assert "value_S2" not in read
    num, den = np.ones((10, 2)), np.full((10, 2), 2.0)
    assert np.allclose(vm.bootstrap_ratio(num, den, rng, 50, 0.05)["mean"], 0.5)


def test_smoke_run_on_the_toy_model(tmp_path):
    cfg = VarmapConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                       n_bases=4, n_shots=2, n_demo_pool=8, read_points=[0, 2, 4], n_boot=30, min_accuracy=0.0, gate=False)
    root = vm.run_varmap(cfg, run_id="smoke")
    meta = json.loads((root / "metadata.json").read_text())
    assert all(meta["identity_check"].values()) and meta["determinism_check"]["swap"]["identical"]
    for task in vm.TASKS:
        res = json.loads((root / "core" / "tasks" / task / "varmap.json").read_text())
        P = len(res["labels"])
        assert np.asarray(res["rates"]["S1"]["flip"]["mean"]).shape == (3, P)
        assert set(res["rates"]) == set(vm.SOURCE_TYPES[task])
        assert (root / "figures" / f"{task}.png").exists()
        arr = np.load(root / "core" / "tasks" / task / "varmap_arrays.npz")
        assert arr["S1__flip"].shape == (res["n_kept"]["S1"], 3, P)
    # at the last read point the answer token's swap carries the source run's output entirely, whatever the model
    res = json.loads((root / "core" / "tasks" / "two_lane" / "varmap.json").read_text())
    for k in ("S1", "S2", "both"):
        assert abs(res["rates"][k]["transfer"]["mean"][-1][-1] - 1.0) < 1e-3
        assert abs(res["rates"][k]["transfer"]["mean"][0][0]) < 1e-3  # the first token at the embedding: the same "("
