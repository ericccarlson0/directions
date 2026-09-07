"""Layerwise metric assembly, on synthetic residual streams with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from directions import layerwise, profiles
from directions.config import Config, build


@pytest.fixture
def cfg():
    return build(Config, {"bootstrap": {"n_boot": 50}})


def _streams(n_layers=6, n_ex=12, d=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_layers + 1, n_ex, d)) + 5.0
    return base


def test_upstream_layers_are_nan(cfg):
    base = _streams()
    steer = base.copy()
    v = np.zeros(8)
    v[0] = 1.0
    steer[3:] += v
    m = layerwise.measure(cfg, "t", 3, 1.0, v, base, steer)
    assert np.isnan(m.S[:3]).all()
    assert np.isnan(m.d_eff[:3]).all()
    assert np.isfinite(m.S[3:]).all()


def test_pure_propagation_gives_unit_gain_and_zero_conversion(cfg):
    """A perturbation that is carried unchanged: log G = 0, C = 0, N undefined."""
    base = _streams()
    v = np.zeros(8)
    v[2] = 1.0
    steer = base.copy()
    steer[2:] += 0.5 * v
    m = layerwise.measure(cfg, "t", 2, 0.5, v, base, steer)
    assert np.allclose(m.G[2:6], 1.0)
    assert np.allclose(m.C[2:6], 0.0)
    assert np.allclose(m.control_alignment[2:], 1.0)
    # every delta is identical, so the centered D_l has no variance anywhere
    assert np.allclose(np.nan_to_num(m.centered_variance[2:]), 0.0, atol=1e-20)
    assert np.allclose(np.nan_to_num(m.d_eff[3:]), 0.0)


def test_pure_amplification_gives_the_expected_log_gain(cfg):
    base = _streams()
    v = np.zeros(8)
    v[1] = 1.0
    steer = base.copy()
    g = 2.0
    for i, l in enumerate(range(2, 7)):
        steer[l] += (g**i) * v
    m = layerwise.measure(cfg, "t", 2, 1.0, v, base, steer)
    assert np.allclose(np.log(m.G[2:6]), np.log(g))
    assert np.allclose(m.C[2:6], g - 1.0)


def test_effective_rank_tracks_planted_dimensionality(cfg):
    rng = np.random.default_rng(1)
    n_ex, d = 60, 12
    base = rng.standard_normal((5, n_ex, d)) + 3.0
    steer = base.copy()
    v = np.zeros(d)
    v[0] = 1.0
    steer[1] += v  # rank 0 after centering
    # layer 2 onwards: perturbation varies in a k-dimensional subspace
    for l, k in ((2, 1), (3, 4)):
        coeff = rng.standard_normal((n_ex, k))
        steer[l] += v + coeff @ np.eye(d)[1 : 1 + k]
    m = layerwise.measure(cfg, "t", 1, 1.0, v, base, steer)
    # the intervention layer is degenerate by construction and reported as NaN
    assert np.isnan(m.d_eff[1])
    assert m.numerical_noise["intervention_layer_d_eff_observed"] == pytest.approx(0.0, abs=1e-6)
    assert m.d_eff[2] == pytest.approx(1.0, abs=1e-6)
    assert 3.0 < m.d_eff[3] <= 4.0
    assert m.d90[3] == 4


def test_new_subspace_fraction_detects_a_created_direction(cfg):
    rng = np.random.default_rng(2)
    n_ex, d = 40, 10
    base = rng.standard_normal((4, n_ex, d)) + 2.0
    steer = base.copy()
    e = np.eye(d)
    coeff = rng.standard_normal((n_ex, 1))
    steer[1] += coeff @ e[0:1]                        # D_1 spans e0
    steer[2] += coeff @ e[0:1] + coeff @ e[5:6]       # block 1 adds an orthogonal direction
    steer[3] += coeff @ e[0:1] + coeff @ e[5:6]       # block 2 adds nothing
    m = layerwise.measure(cfg, "t", 1, 1.0, e[0], base, steer)
    assert m.N[1] == pytest.approx(1.0, abs=1e-6)
    assert m.N[2] == pytest.approx(0.0, abs=1e-6)  # block 2 does nothing


def test_new_subspace_is_zero_when_the_block_only_rescales(cfg):
    rng = np.random.default_rng(3)
    base = rng.standard_normal((4, 30, 9)) + 1.0
    steer = base.copy()
    pert = rng.standard_normal((30, 9))
    steer[1] += pert
    steer[2] += 1.7 * pert
    steer[3] += 1.7 * pert
    m = layerwise.measure(cfg, "t", 1, 1.0, np.eye(9)[0], base, steer)
    assert m.N[1] < 0.15  # 90% subspace of D_1 already contains most of B_1


def test_measure_rejects_shape_mismatch(cfg):
    with pytest.raises(ValueError, match="shape mismatch"):
        layerwise.measure(cfg, "t", 0, 1.0, np.eye(4)[0],
                          np.zeros((3, 5, 4)), np.zeros((3, 6, 4)))


def test_to_dict_is_json_safe(cfg):
    base = _streams()
    steer = base + 0.1
    m = layerwise.measure(cfg, "t", 1, 1.0, np.eye(8)[0], base, steer)
    d = m.to_dict()
    assert d["S_median"][0] is None  # upstream of the intervention
    assert all(v is None or np.isfinite(v) for v in d["log_G_median"])
    assert len(d["d_eff"]) == m.n_layers + 1


def test_aggregate_controls_shapes(cfg):
    base = _streams()
    ms = []
    for s in range(4):
        rng = np.random.default_rng(s)
        steer = base.copy()
        steer[2:] += rng.standard_normal((5, 12, 8)) * 0.1
        ms.append(layerwise.measure(cfg, f"c{s}", 2, 1.0, np.eye(8)[0], base, steer))
    null = layerwise.aggregate_controls(ms)
    assert set(null) >= {"S", "log_G", "C", "d_eff", "N"}
    assert null["S"]["n_controls"] == 4
    assert len(null["S"]["samples"]) == 4
    assert len(null["S"]["q50"]) == ms[0].n_layers + 1


def test_profile_classification_is_deterministic_and_labelled(cfg):
    base = _streams(n_layers=10, n_ex=20)
    v = np.eye(8)[0]
    steer = base.copy()
    steer[3:] += v  # conserved: no gain, no conversion
    m = layerwise.measure(cfg, "t", 3, 1.0, v, base, steer)
    out = profiles.classify(m)
    assert out["label"] in profiles.LABELS
    assert out == profiles.classify(m)
    assert out["features"]["max_conversion"] == pytest.approx(0.0, abs=1e-9)
    assert out["label"] == "conserved_transmission"


def test_numerical_noise_diagnostic_is_recorded(cfg):
    """The intervention layer's centered variance bounds the numerical noise floor."""
    rng = np.random.default_rng(30)
    base = rng.standard_normal((5, 20, 8)) + 4.0
    v = np.eye(8)[0]
    steer = base.copy()
    steer[2:] += 2.0 * v
    steer[4:] += 0.5 * rng.standard_normal((1, 20, 8))
    m = layerwise.measure(cfg, "t", 2, 2.0, v, base, steer)
    noise = m.numerical_noise
    assert np.isnan(m.d_eff[2]) and np.isnan(m.d90[2])
    assert noise["intervention_layer_centered_variance"] == pytest.approx(0.0, abs=1e-18)
    assert noise["next_layer_centered_variance"] == pytest.approx(0.0, abs=1e-18)
    assert m.centered_variance[4] > 1.0


def test_centered_variance_is_reported_per_layer(cfg):
    base = _streams()
    steer = base + 0.3
    m = layerwise.measure(cfg, "t", 1, 1.0, np.eye(8)[0], base, steer)
    assert m.centered_variance.shape == (m.n_layers + 1,)
    d = m.to_dict()
    assert len(d["centered_variance"]) == m.n_layers + 1
    assert d["numerical_noise"]["note"]
