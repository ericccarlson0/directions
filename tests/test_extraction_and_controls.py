"""Extraction, stability and matched-control construction against the model."""

from __future__ import annotations

import numpy as np
import pytest

from directions import mathx
from directions.controls import build_random_controls, control_rng
from directions.extraction import (
    DirectionSpec,
    alpha_from_rho,
    extract_directions,
    extraction_rng,
    median_residual_norms,
)
from directions.tasks import load_task, split_task


@pytest.fixture(scope="module")
def extracted(tiny_cfg, tiny_model):
    splits = split_task(load_task("antonym"), 6, 4, 6, seed=0)
    layers = [1, 2]
    return extract_directions(
        tiny_model, tiny_cfg, "antonym", list(splits.extraction), list(splits.extraction), layers
    ), layers


def test_directions_are_unit_norm_and_per_layer(extracted):
    ext, layers = extracted
    assert sorted(ext.layers) == layers
    for l in layers:
        le = ext.layers[l]
        assert np.isclose(np.linalg.norm(le.consensus), 1.0)
        assert le.seed_directions.shape[0] == 2
        for v in le.seed_directions:
            assert np.isclose(np.linalg.norm(v), 1.0)


def test_stability_is_a_cosine_in_range(extracted):
    ext, layers = extracted
    for l in layers:
        le = ext.layers[l]
        assert 0.0 <= le.stability_min <= le.stability_mean <= 1.0 + 1e-9
        assert len(le.pairwise_abs_cosine) == 1  # n_seeds=2 -> one pair


def test_directions_are_oriented_toward_the_mean_difference(extracted):
    ext, layers = extracted
    for l in layers:
        assert all(c >= 0 for c in ext.layers[l].mean_diff_cosine)


def test_extraction_is_reproducible(tiny_cfg, tiny_model):
    splits = split_task(load_task("antonym"), 6, 4, 6, seed=0)
    args = (tiny_model, tiny_cfg, "antonym", list(splits.extraction), list(splits.extraction), [2])
    a = extract_directions(*args)
    b = extract_directions(*args)
    assert np.allclose(a.layers[2].consensus, b.layers[2].consensus)


def test_extraction_rng_is_deterministic_and_seed_dependent():
    a = extraction_rng(0, "antonym", 0).standard_normal(4)
    assert np.allclose(a, extraction_rng(0, "antonym", 0).standard_normal(4))
    assert not np.allclose(a, extraction_rng(0, "antonym", 1).standard_normal(4))
    assert not np.allclose(a, extraction_rng(1, "antonym", 0).standard_normal(4))


def test_median_residual_norms_shape_and_positivity(tiny_model, tiny_cfg):
    from directions.config import PromptTemplate
    from directions.prompts import build_eval_prompts

    splits = split_task(load_task("antonym"), 6, 4, 6, seed=0)
    prompts = build_eval_prompts(
        np.random.default_rng(0), PromptTemplate(), list(splits.evaluation),
        list(splits.evaluation), 0,
    )
    h = tiny_model.capture(prompts, batch_size=3, max_seq_len=512)
    norms = median_residual_norms(h)
    assert norms.shape == (tiny_model.n_layers + 1,)
    assert (norms > 0).all()


def test_alpha_from_rho_is_the_relative_norm():
    assert alpha_from_rho(0.1, 20.0) == pytest.approx(2.0)
    assert alpha_from_rho(0.0, 20.0) == 0.0


# -- controls --------------------------------------------------------------


@pytest.fixture
def direction():
    v = np.zeros(32, dtype=np.float32)
    v[3] = 1.0
    return DirectionSpec(layer=2, vector=v, alpha=1.5)


def test_controls_match_layer_and_norm(direction):
    ctrls = build_random_controls(control_rng(0, "t", 2), direction, 6, ["isotropic", "orthogonal"])
    assert len(ctrls) == 6
    for c in ctrls:
        assert c.layer == direction.layer
        assert c.alpha == direction.alpha
        assert np.isclose(np.linalg.norm(c.vector), 1.0, atol=1e-6)


def test_orthogonal_controls_are_orthogonal(direction):
    ctrls = build_random_controls(control_rng(0, "t", 2), direction, 8, ["isotropic", "orthogonal"])
    orth = [c for c in ctrls if c.label.startswith("orthogonal")]
    iso = [c for c in ctrls if c.label.startswith("isotropic")]
    assert len(orth) == len(iso) == 4
    for c in orth:
        assert abs(mathx.cosine(c.vector, direction.vector)) < 1e-6


def test_controls_are_distinct_and_reproducible(direction):
    a = build_random_controls(control_rng(0, "t", 2), direction, 4, ["isotropic"])
    b = build_random_controls(control_rng(0, "t", 2), direction, 4, ["isotropic"])
    for x, y in zip(a, b):
        assert np.allclose(x.vector, y.vector)
    assert not np.allclose(a[0].vector, a[1].vector)


def test_controls_differ_across_layers_and_tasks(direction):
    a = build_random_controls(control_rng(0, "t", 2), direction, 2, ["isotropic"])
    b = build_random_controls(control_rng(0, "t", 3), direction, 2, ["isotropic"])
    c = build_random_controls(control_rng(0, "u", 2), direction, 2, ["isotropic"])
    assert not np.allclose(a[0].vector, b[0].vector)
    assert not np.allclose(a[0].vector, c[0].vector)


def test_unknown_control_kind_is_rejected(direction):
    with pytest.raises(ValueError, match="unknown control kind"):
        build_random_controls(control_rng(0, "t", 2), direction, 1, ["adversarial"])
