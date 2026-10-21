"""The operator direction (docs/DECISIONS.md D58): the backend's additive intervention at any token, the draws, the
readouts, and a toy run of the stage."""

from __future__ import annotations

import json

import numpy as np
import pytest

import directions.varmap as vm
from directions.config import ModelConfig, ToyModelConfig, load_opsteer_config
from directions.model import Intervention, ModelBackend, SitePatch
from directions.opsteer import mean_readouts, run_opsteer, sample_opsteer, shown, twin, window


@pytest.fixture(scope="module")
def toy():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                    toy=ToyModelConfig(family="qwen3")), run_seed=0)


def _prompts(n: int = 6) -> list:
    draw = sample_opsteer(1, 4, n, 2, 8)
    return [vm.render(d, shown(b)) for d, b in zip(draw["demos"], draw["bases"])], draw


def _query_pos(backend, prompts) -> np.ndarray:
    return np.array([len(backend._encode_prompt(p)[0]) - 1 for p in prompts])


def test_positioned_intervention_at_the_query_token_is_the_default_path(toy):
    prompts, _ = _prompts()
    tid = np.arange(10)
    vec = np.random.default_rng(0).standard_normal(toy.hidden_size).astype(np.float32)
    a = toy.run(prompts, interventions=[Intervention(2, vec, 0.7)], token_ids=tid)
    b = toy.run(prompts, interventions=[Intervention(2, vec, 0.7, _query_pos(toy, prompts))], token_ids=tid)
    assert np.array_equal(a.token_logprobs, b.token_logprobs)


def test_positioned_intervention_is_causal_and_sliced_per_batch(toy):
    prompts, draw = _prompts()
    tid = np.arange(10)
    op = np.array([vm.query_tokens(toy, p, shown(b), "op")[0][2] for p, b in zip(prompts, draw["bases"])])
    vec = 3 * np.random.default_rng(1).standard_normal(toy.hidden_size).astype(np.float32)
    iv = Intervention(2, vec, 1.0, op)
    plain = toy.run(prompts, token_ids=tid)
    steered = toy.run(prompts, interventions=[iv], token_ids=tid)
    assert not np.allclose(plain.token_logprobs, steered.token_logprobs)
    # per-batch slicing of the positions: two rows per batch gives the same as one batch
    small = toy.run(prompts, interventions=[iv], token_ids=tid, batch_size=2)
    assert np.allclose(small.token_logprobs, steered.token_logprobs, atol=1e-5)
    # nothing before the token, and nothing at the token before the read point, moves
    pos = np.stack([op - 1, op, op + 1], axis=1)
    h0 = toy.capture_sites(prompts, pos).numpy()
    h1 = toy.capture_sites(prompts, pos, interventions=[iv]).numpy()
    assert np.array_equal(h0[:, :, 0], h1[:, :, 0]) and np.array_equal(h0[:, :2, 1], h1[:, :2, 1])
    assert not np.allclose(h0[:, 2, 1], h1[:, 2, 1]) and not np.allclose(h0[:, 3:, 2], h1[:, 3:, 2])
    with pytest.raises(ValueError, match="one entry per prompt"):
        toy.run(prompts, interventions=[Intervention(2, vec, 1.0, op[:3])], token_ids=tid)


def test_the_direction_at_read_point_zero_is_the_token_swap(toy):
    """v[0] is the embedding difference: adding it at the token equals swapping the twin's token state in."""
    prompts, draw = _prompts()
    tid = np.arange(10)
    twins = [vm.render(d, b) for d, b in zip(draw["demos"], draw["bases"])]
    op = np.array([max(j for j, lab in zip(*vm.query_tokens(toy, p, shown(b), "op")[::2]) if lab == "op")
                   for p, b in zip(prompts, draw["bases"])])
    h_max = toy.capture_sites(prompts, op[:, None])[:, 0, 0].float()
    h_min = toy.capture_sites(twins, op[:, None])[:, 0, 0].float()
    v0 = (h_min - h_max).mean(dim=0).numpy()
    assert np.allclose((h_min - h_max).numpy(), v0[None], atol=1e-6)  # one embedding difference for every prompt
    steer = toy.run(prompts, interventions=[Intervention(0, v0, 1.0, op)], token_ids=tid)
    swap = toy.run(prompts, site_patch=SitePatch(np.zeros(len(prompts), dtype=int), op, h_min.numpy()), token_ids=tid)
    assert np.allclose(steer.token_logprobs, swap.token_logprobs, atol=1e-5)


def test_the_unsteer_swap_undoes_the_direction_exactly(toy):
    """At the injection read point and the operator token, the site patch comes after the added vector: swapping the
    unsteered state in gives the unsteered run, bit for bit."""
    prompts, draw = _prompts()
    tid = np.arange(10)
    op = np.array([vm.query_tokens(toy, p, shown(b), "op")[0][2] for p, b in zip(prompts, draw["bases"])])
    vec = 3 * np.random.default_rng(2).standard_normal(toy.hidden_size).astype(np.float32)
    plain = toy.run(prompts, token_ids=tid)
    own = toy.capture_sites(prompts, op[:, None])[:, 2, 0]
    back = toy.run(prompts, interventions=[Intervention(2, vec, 1.0, op)], token_ids=tid,
                   site_patch=SitePatch(np.full(len(prompts), 2), op, own))
    assert np.array_equal(back.token_logprobs, plain.token_logprobs)


def test_sample_opsteer():
    d = sample_opsteer(20260916, 64, 40, 8, 96)
    assert d["info"]["n_dir"] == 64 and d["info"]["n_bases"] == 40
    assert all(e.op == "max" for e in d["dir_set"]) and all(b.op == "min" for b in d["bases"])
    used = {(e.a, e.b, e.c, e.d) for e in d["dir_set"]}
    assert not any((b.a, b.b, b.c, b.d) in used for b in d["bases"])
    for b, srcs, demos in zip(d["bases"], d["sources"], d["demos"]):
        assert set(srcs) == {"S1", "S2", "S1_same", "S2_same", "both"} and all(vm.valid_source(k, b, s) for k, s in srcs.items())
        assert len(demos) == 8 and sum(e.op == "min" for e in demos) == 4
        assert shown(b).op == "max" and twin(shown(b)) == b and shown(b).Y == max(b.S1, b.S2) != b.Y
    assert sample_opsteer(20260916, 64, 40, 8, 96)["bases"] == d["bases"]


def test_window():
    assert window(np.array([0.9, 0.8, 0.6, 0.4, 0.7])) == 2
    assert window(np.array([0.4, 0.9])) is None
    assert window(np.array([0.6, 0.6])) == 1


def test_mean_readouts():
    labels = ["op", "(", "a", ",", "␣"]
    M = 8
    op = np.zeros((M, 5))
    op[:5, 0] = 1.0  # at the operator token through read point 4
    op[6:, 4] = 0.7  # at the answer token from 6
    s = np.zeros((M, 5))
    s[5:, 4] = 0.6
    rates = {"op": {"flip": {"mean": op}}, "S1": {"flip": {"mean": s}}, "S2": {"flip": {"mean": s}}}
    assert mean_readouts(rates, labels, list(range(M)), 2) == {"R_O": 6, "O_out": 4, "G": 5}
    assert mean_readouts(rates, labels, list(range(M)), 5)["O_out"] is None  # nothing at the token from 5 on


def test_toy_run(tmp_path):
    cfg = load_opsteer_config("configs/smoke_toy_opsteer.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_opsteer(cfg, run_id="smoke")
    r = json.loads((root / "core" / "opsteer.json").read_text())
    meta = json.loads((root / "metadata.json").read_text())
    assert set(r) == {"direction", "curve", "map", "labels", "inject_read_point"} and r["labels"][0] == "op"
    assert len(r["direction"]["norm"]) == len(r["curve"]["read_points"]) and r["direction"]["n_pairs"] == cfg.n_dir
    assert set(r["curve"]["arms"]) == {"steer", "swap", "null0"}
    m = r["map"]
    assert set(m["rates"]) == set(vm.SOURCE_TYPES["maxmin"]) and set(m["operator"]) == {"R_O", "G", "O_out", "route", "reading"}
    rps = m["read_points"]
    before = [i for i, x in enumerate(rps) if x < cfg.inject_read_point]
    assert np.asarray(m["rates"]["op"]["change"]["mean"])[before].max() == 0  # the unsteer swap is a no-op before it
    assert all(meta["identity_check"].values()) and meta["determinism_check"]["swap"]["identical"]
    arrays = np.load(root / "core" / "opsteer_arrays.npz")
    assert arrays["direction"].shape == (len(r["curve"]["read_points"]), ToyModelConfig().hidden_size)
    assert (root / "figures" / "curve.png").exists() and (root / "figures" / "map.png").exists()


def test_d58_reading_script(tmp_path, monkeypatch, capsys):
    """scripts/d58_opsteer.py reads an opsteer run against a step-1 map of the same model (toy runs)."""
    import importlib.util
    import sys
    from pathlib import Path

    from directions.config import VarmapConfig

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("d58_opsteer", scripts / "d58_opsteer.py")
    d58 = importlib.util.module_from_spec(spec)
    sys.modules["d58_opsteer"] = d58
    spec.loader.exec_module(d58)
    cfg = load_opsteer_config("configs/smoke_toy_opsteer.yaml")
    cfg.output_dir = str(tmp_path / "steer")
    run_opsteer(cfg, run_id="opsteer_qwen3_4b_toy")
    vcfg = VarmapConfig(output_dir=str(tmp_path / "d57"), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                        expressions=["maxmin"], n_bases=8, n_shots=2, n_demo_pool=8, read_points=[0, 1, 2, 4], n_boot=30,
                        min_accuracy=0.0, gate=False)
    vm.run_varmap(vcfg, run_id="varmap_maxmin_qwen3_4b_toy")
    assert d58.main(["--opsteer", str(tmp_path / "steer"), "--d57", str(tmp_path / "d57"), "--out", str(tmp_path / "o.json")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())["qwen3_4b"]
    assert set(s["comparison"]) == {"lower_bounds", "means", "route"} and "natural_min_only" in s
    assert d58.compare(28, 26)["reading"] == "as natural" and d58.compare(31, 26)["reading"] == "later"
    assert d58.compare(20, 26)["reading"] == "earlier" and d58.compare(None, 26)["reading"] == "undecided"
    assert "qwen3_4b: direction" in capsys.readouterr().out
