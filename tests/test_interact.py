"""D61's interaction test: the factor pairs' sources, the additive state at read point 0, the readings, a toy run."""

from __future__ import annotations

import json

import numpy as np
import pytest

import directions.varmap as vm
from directions.config import ModelConfig, ToyModelConfig, load_interact_config
from directions.interact import (
    CATS,
    PATCHES,
    conversion_depth,
    draw_interact,
    reading,
    run_interact,
)
from directions.model import ModelBackend
from directions.opsteer import op_position, twin
from directions.seeds import rng_for


@pytest.fixture(scope="module")
def draw():
    return draw_interact(20260916, 60, 8, 96, ["OS", "S1S2", "OD"])


def test_sources_change_exactly_their_factors(draw):
    exprs = vm.all_exprs("maxmin")
    pool = {exprs[i] for i in rng_for(20260916, "varmap", "maxmin", "split").permutation(len(exprs))[:96]}
    assert len(draw["bases"]) == 60 and not any(b in pool for b in draw["bases"])
    for b, srcs, demos in zip(draw["bases"], draw["sources"], draw["demos"]):
        assert sum(e.op == "max" for e in demos) == 4
        e10, e01, e11 = srcs["OS"]
        assert e10 == twin(b) and e01.op == b.op and e01.S1 != b.S1 and e01.S2 != b.S2 and e11 == twin(e01)
        assert len({b.Y, e10.Y, e01.Y, e11.Y}) == 4 and e10 not in pool and e11 not in pool
        e10, e01, e11 = srcs["S1S2"]
        assert (e10.c, e10.d) == (b.c, b.d) and e10.S1 != b.S1 and (e01.a, e01.b) == (b.a, b.b) and e01.S2 != b.S2
        assert (e11.a, e11.b, e11.c, e11.d) == (e10.a, e10.b, e01.c, e01.d) and len({e.op for e in srcs["S1S2"]} | {b.op}) == 1
        assert e11.Y not in {b.Y, e10.Y, e01.Y} and e11.S1 != e11.S2 and e11 not in pool
        e10, e01, e11 = srcs["OD"]
        assert e10 == twin(b) and (e01.S1, e01.S2) == (b.S1, b.S2) and e11 == twin(e01)
        assert (e01.a, e01.b) != (b.a, b.b) and (e01.c, e01.d) != (b.c, b.d)
        assert e01.Y == b.Y and e11.Y == e10.Y  # the digits do not move the answer
    assert draw_interact(20260916, 60, 8, 96, ["OS", "S1S2", "OD"])["sources"] == draw["sources"]


def test_the_additive_state_is_the_joint_one_at_read_point_0(draw):
    """At read point 0 a token's state is its embedding alone, so h10 + h01 - h00 = h11 at every token."""
    toy = ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16,
                                   toy=ToyModelConfig(family="qwen3")), run_seed=0)
    for pair in ("OS", "S1S2", "OD"):
        quads = [(b, *draw["sources"][i][pair]) for i, b in enumerate(draw["bases"][:6])]
        prompts = [[vm.render(draw["demos"][i], e) for i, e in enumerate(col)] for col in zip(*quads)]
        pos = np.array([op_position(toy, p, q[0])[1] for p, q in zip(prompts[0], quads)])
        h = [toy.capture_sites(prompts[k], pos)[:, 0].numpy() for k in range(4)]
        assert np.allclose(h[1] + h[2] - h[0], h[3], atol=1e-6), pair
        assert not np.allclose(h[1], h[0]) and not np.allclose(h[2], h[0])


def test_conversion_depth_and_reading():
    rps = [0, 1, 2, 3, 4]
    assert conversion_depth(np.array([0.9, 0.8, 0.6, 0.3, 0.1]), np.array([1, 1, 0.8, 0.45, 0.2]), rps, 0.5) == \
        {"K": 3, "holds_at_first": True}
    assert conversion_depth(np.array([0.9] * 5), np.array([1.0] * 5), rps, 0.5)["K"] is None
    assert reading({"OS": 24, "S1S2": 23, "OD": None})["control_combined"] == "where the operands are combined"
    assert reading({"OS": 18, "S1S2": 23, "OD": None})["control_combined"] == "earlier than the operands"
    assert reading({"OS": 28, "S1S2": 23, "OD": None})["control_combined"] == "later than the operands"
    assert reading({"OS": None, "S1S2": 23, "OD": None})["control_combined"].startswith("never")
    assert reading({"OS": 24, "S1S2": 23, "OD": 25})["null_clear"] is False  # the null fails within 2 of K_OS
    assert reading({"OS": 24, "S1S2": 23, "OD": 30})["null_clear"] is True


def test_toy_run(tmp_path):
    cfg = load_interact_config("configs/smoke_toy_interact.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_interact(cfg, run_id="smoke")
    r = json.loads((root / "core" / "interact.json").read_text())
    meta = json.loads((root / "metadata.json").read_text())
    assert set(r["pairs"]) == {"OS", "S1S2", "OD"} and r["patches"] == list(PATCHES) and r["categories"] == list(CATS)
    for p in r["pairs"].values():
        assert set(p["rates"]) == set(PATCHES) and p["additive_equals_joint_at_0"] == 1.0
        assert np.asarray(p["ratio_mean"]).shape == (len(cfg.read_points), len(p["labels"]))
    assert set(r["reading"]) == {"K", "control_combined", "null_clear", "localization"}
    assert all(meta["identity_check"].values()) and meta["determinism_check"]["patch"]["identical"]
    arrays = np.load(root / "core" / "interact_arrays.npz")
    assert arrays["OS__argmax"].shape == (len(PATCHES), cfg.n_bases, len(cfg.read_points))
    assert (root / "figures" / "interact.png").exists()
