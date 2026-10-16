"""The position-set check (D52): the multi-token site swap against native runs, the sets, the paired gain, the rule,
and a smoke run on the toy model."""

from __future__ import annotations

import json

import numpy as np
import pytest

import directions.posset as ps
import directions.varmap as vm
from directions.config import ModelConfig, PossetConfig, ToyModelConfig, load_posset_config
from directions.model import ModelBackend, SitePatch


@pytest.fixture(scope="module")
def toy():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                    toy=ToyModelConfig(family="qwen3")), run_seed=0)


def _pair(toy):
    demos = [vm.Expr(1, 2, 0, 1, 0, 0), vm.Expr(5, 4, 3, 3, 1, 0)]
    base, src = vm.Expr(9, 8, 4, 7, 2, 2), vm.Expr(7, 6, 1, 2, 3, 0)
    pb, pr = vm.render(demos, base), vm.render(demos, src)
    pos, _, labels = vm.query_tokens(toy, pb, base)
    return pb, pr, np.asarray(pos), labels


def _ids(toy):
    return np.array([toy.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(10)])


def test_swapping_every_query_token_gives_the_source_run(toy):
    """Base and source share their demonstrations, so swapping every query token at a read point makes the rest of
    the base run the source's (a native reference for the multi-token swap)."""
    pb, pr, pos, _ = _pair(toy)
    tid = _ids(toy)
    src = toy.run([pr], token_ids=tid)
    states = toy.capture_sites([pr], pos[None])  # (1, L+1, P, d)
    for m in (0, 2, toy.n_layers):
        got = toy.run([pb], token_ids=tid, site_patch=SitePatch(np.array([m]), pos[None], states[:, m]))
        assert np.allclose(got.token_logprobs, src.token_logprobs, atol=1e-5)
        assert (got.query_argmax == src.query_argmax).all()


def test_a_set_with_the_runs_own_states_reduces_to_one_token(toy):
    """A set swap whose other tokens carry the base's own states equals the one-token swap, bit for bit; a set of
    one equals the one-token form."""
    pb, pr, pos, _ = _pair(toy)
    tid = _ids(toy)
    own = toy.capture_sites([pb], pos[None])
    src = toy.capture_sites([pr], pos[None])
    m, j = 2, len(pos) - 1
    one = toy.run([pb], token_ids=tid, site_patch=SitePatch(np.array([m]), pos[[j]], src[:, m, j]))
    vals = own[:, m].clone()
    vals[:, j] = src[:, m, j]
    many = toy.run([pb], token_ids=tid, site_patch=SitePatch(np.array([m]), pos[None], vals))
    single = toy.run([pb], token_ids=tid, site_patch=SitePatch(np.array([m]), pos[None, [j]], src[:, m, [j]]))
    assert np.array_equal(one.token_logprobs, many.token_logprobs) and np.array_equal(one.token_logprobs, single.token_logprobs)


def test_set_members_on_qwen_labels():
    labels = ["(", "a", "+", "b", ")", " -", " (", "c", "+", "d", ")", " -", " (", "e", "+", "f", ")", " =", "␣"]
    s = ps.set_members(labels)
    assert s["answer"] == [18] and s["answer_eq"] == [17, 18] and s["query"] == list(range(19))
    assert s["answer_digits"] == [1, 3, 7, 9, 13, 15, 18]
    assert len(s["non_digit"]) == 13 and not set(s["non_digit"]) & {1, 3, 7, 9, 13, 15} and 18 in s["non_digit"]
    with pytest.raises(ValueError):
        ps.set_members(labels[:-1])


def test_set_members_on_toy_labels(toy):
    _, _, _, labels = _pair(toy)
    s = ps.set_members(labels)
    assert s["answer"] == [len(labels) - 1] and len(s["answer_digits"]) == 7
    assert set(s["non_digit"]) | set(s["answer_digits"]) == set(range(len(labels)))


def test_paired_gain():
    rng = np.random.default_rng(0)
    ref = np.zeros((200, 2), dtype=bool)
    x = ref.copy()
    x[:40, 0] = True  # +0.2 at the first read point, nothing at the second
    g = ps.paired_gain(x, ref, rng, 2000, 0.01)
    assert np.allclose(g["mean"], [0.2, 0.0]) and g["lo"][0] > 0.1 and g["lo"][1] == 0.0


def _g(mean, lo):
    return {"mean": np.asarray(mean, dtype=float), "lo": np.asarray(lo, dtype=float)}


def test_decide():
    rps = [21, 24, 25]
    none = {"non_digit": _g([0.04, 0.0, 0.1], [0.01, -0.02, -0.01]), "answer_eq": _g([0, 0, 0], [0, 0, 0])}
    assert ps.decide({"all": none}, rps, 0.05, 0.8, 4, 21) == {"passes": False, "passing": []}
    some = {"non_digit": _g([0.0, 0.2, 0.1], [0.0, 0.1, 0.02]), "answer_eq": _g([0.0, 0.19, 0.02], [0, 0, 0])}
    d = ps.decide({"all": some}, rps, 0.05, 0.8, 4, 21)
    assert d["passes"] and d["positions"] == "non_digit" and d["sites"] == [21, 24, 25]  # 0.02 < 0.8 x 0.1 at 25
    eq = {"non_digit": _g([0.0, 0.2, 0.0], [0.0, 0.1, 0.0]), "answer_eq": _g([0.0, 0.19, 0.0], [0, 0, 0])}
    d = ps.decide({"all": eq}, rps, 0.05, 0.8, 1, None)
    assert d["positions"] == "answer_eq" and d["sites"] == [24]


def test_config_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("expressions: [three_lane]\nsources: [S1, all]\nread_points: [21, 24]\n")
    assert load_posset_config(p).sources == ["S1", "all"]
    p.write_text("expressions: [three_lane]\nsources: [both]\nread_points: [21]\n")
    with pytest.raises(ValueError):
        load_posset_config(p)
    p.write_text("expressions: [three_lane]\nread_points: []\n")
    with pytest.raises(ValueError):
        load_posset_config(p)


def test_smoke_run_on_the_toy_model(tmp_path):
    cfg = PossetConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16),
                       n_bases=5, n_shots=2, n_demo_pool=8, read_points=[1, 3], marks={"G3": 3}, n_boot=30,
                       n_boot_gain=200, min_accuracy=0.0, gate=False, control=1)
    root = ps.run_posset(cfg, run_id="smoke")
    meta = json.loads((root / "metadata.json").read_text())
    assert all(meta["identity_check"].values()) and meta["determinism_check"]["swap"]["identical"]
    res = json.loads((root / "core" / "tasks" / "three_lane" / "posset.json").read_text())
    assert set(res["rates"]) == {"S1", "S2", "S3", "all"} and set(res["rates"]["all"]) == {"n", *ps.SETS}
    assert np.asarray(res["rates"]["all"]["query"]["flip"]["mean"]).shape == (2,)
    assert set(res["gains"]["all"]) == set(ps.SETS) - {"answer"} and "passes" in res["decision"]
    arr = np.load(root / "core" / "tasks" / "three_lane" / "posset_arrays.npz")
    assert arr["all__non_digit__argmax"].shape == (res["n_kept"]["all"], 2)
    assert (root / "figures" / "three_lane.png").exists()


def test_query_set_gives_the_source_answer_on_the_toy_model(tmp_path):
    """The check set: with every query token swapped, the answer is the source run's own answer, at every read point."""
    cfg = PossetConfig(output_dir=str(tmp_path), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16),
                       n_bases=4, n_shots=2, n_demo_pool=8, read_points=[0, 2], n_boot=20, n_boot_gain=100,
                       min_accuracy=0.0, gate=False, figures=False, identity_check=False, determinism_check=False)
    root = ps.run_posset(cfg, run_id="query")
    res = json.loads((root / "core" / "tasks" / "three_lane" / "posset.json").read_text())
    arr = np.load(root / "core" / "tasks" / "three_lane" / "posset_arrays.npz")
    bases, sources, demos, _ = vm.sample_bases(cfg.seed, "three_lane", cfg.n_bases, cfg.n_shots, cfg.n_demo_pool)
    backend = ModelBackend(cfg.model, run_seed=cfg.seed)
    tid = np.array([backend.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(10)])
    rows = arr["all__bases"]
    src_am = backend.run([vm.render(demos[i], sources[i]["all"]) for i in rows], token_ids=tid).query_argmax
    assert (arr["all__query__argmax"] == src_am[:, None]).all()
    assert res["n_kept"]["all"] == len(rows)
