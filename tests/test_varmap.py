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
    h = vm.Expr(9, 8, 4, 7, 2, 2)  # three brackets (D51)
    assert (h.S1, h.S2, h.S3, h.T, h.U, h.Y, h.text()) == (17, 11, 4, 6, 15, 2, "(9+8) - (4+7) - (2+2)")
    assert {i: h.text()[i] for i in h.digit_slots()} == {1: "9", 3: "8", 9: "4", 11: "7", 17: "2", 19: "2"}
    assert (e.S3, t.S3) == (0, 0)
    for task in vm.TASKS:
        exprs = vm.all_exprs(task)
        assert exprs and all(0 <= x.Y <= 9 for x in exprs)
    assert len(vm.all_exprs("one_lane")) < len(vm.all_exprs("two_lane")) < len(vm.all_exprs("three_lane")) == 113575


@pytest.mark.parametrize("task", vm.TASKS)
def test_sources_change_exactly_what_their_type_says(task):
    pool = vm.all_exprs(task)
    digits = np.array([[x.a, x.b, x.c, x.d, x.e, x.f] for x in pool]) if task == "three_lane" else None
    rng = np.random.default_rng(0)
    for base in pool[:: max(1, len(pool) // 40)]:
        for kind in vm.SOURCE_TYPES[task]:
            src = vm.draw_source(kind, base, pool, rng, digits)
            if src is None:
                continue
            assert vm.valid_source(kind, base, src)
            pred = vm.prediction(kind, base, src)
            assert 0 <= pred <= 9
            if kind == "S1":
                assert (src.c, src.d, src.e, src.f) == (base.c, base.d, base.e, base.f) and src.S1 != base.S1 and pred == src.Y
            if kind == "S2":
                assert (src.a, src.b, src.e, src.f) == (base.a, base.b, base.e, base.f) and src.S2 != base.S2 and pred == src.Y
            if kind == "S3":
                assert (src.a, src.b, src.c, src.d) == (base.a, base.b, base.c, base.d) and src.S3 != base.S3 and pred == src.Y
            if kind.endswith("_same"):
                assert pred == base.Y and src.Y == base.Y and src != base
            if kind == "both" and task == "max":  # max's predictions coincide (D55): both sums and the answer move
                assert src.S1 != base.S1 and src.S2 != base.S2 and src.Y != base.Y and pred == src.Y
            elif kind == "both":
                p = vm.predictions(base, src)
                assert len({p["S1"], p["S2"], p["both"], p["none"]}) == 4 and pred == src.Y
            if kind == "all":
                p = vm.predictions(base, src)
                assert set(p) == {"S1", "S2", "S3", "T", "U", "all", "none"} and len(set(p.values())) == 7 and pred == src.Y
                assert (p["T"], p["U"]) == (src.S1 - src.S2 - base.S3, base.S1 - src.S2 - src.S3)


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
    for task in cfg.expressions:
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


def test_three_lane_query_tokens_and_sampling(toy):
    """Three brackets (D51): six labelled digits; sampled sources valid; the two-lane draw is unchanged by the third task."""
    demos = [vm.Expr(1, 2, 0, 1, 0, 0), vm.Expr(5, 4, 3, 3, 1, 0)]
    base, src = vm.Expr(9, 8, 4, 7, 2, 2), vm.Expr(9, 8, 4, 7, 3, 0)
    pb, ps = vm.render(demos, base), vm.render(demos, src)
    assert pb.prompt.endswith("(9+8) - (4+7) - (2+2) = ") and pb.target == "2"
    pos_b, ids_b, labels = vm.query_tokens(toy, pb, base)
    pos_s, _ids_s, _ = vm.query_tokens(toy, ps, src)
    assert pos_b == pos_s and [labels[j] for j in (1, 3, 9, 11, 17, 19)] == list("abcdef") and labels[-1] == "␣"
    bases, sources, demos, info = vm.sample_bases(1, "three_lane", 6, 2, 8)
    assert len(bases) == 6 and info["n_expressions"] == 113575
    for b, srcs in zip(bases, sources):
        assert set(srcs) == set(vm.SOURCE_TYPES["three_lane"])
        assert all(vm.valid_source(k, b, s) for k, s in srcs.items())


def test_three_lane_smoke_run_on_the_toy_model(tmp_path):
    cfg = VarmapConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                       expressions=["three_lane"], n_bases=3, n_shots=2, n_demo_pool=8, read_points=[0, 4], n_boot=30,
                       min_accuracy=0.0, gate=False)
    root = vm.run_varmap(cfg, run_id="smoke3")
    res = json.loads((root / "core" / "tasks" / "three_lane" / "varmap.json").read_text())
    assert res["template"] == "(a+b) - (c+d) - (e+f) = " and set(res["rates"]) == set(vm.SOURCE_TYPES["three_lane"])
    assert set(res["rates"]["all"]["hypotheses"]) == {"S1", "S2", "S3", "T", "U", "all", "none"}
    assert set(res["reading"]) >= {"present_S3", "value_S3", "count"} and np.asarray(res["reading"]["count"]).max() <= 3
    for k in ("S1", "S2", "S3", "all"):  # the last read point's swap carries the source's output entirely
        assert abs(res["rates"][k]["transfer"]["mean"][-1][-1] - 1.0) < 1e-3
    assert (root / "figures" / "three_lane.png").exists()
