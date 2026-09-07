"""Unit tests for the numerical/statistical primitives."""

from __future__ import annotations

import numpy as np
import pytest

from directions import mathx


# -- normalization ---------------------------------------------------------


def test_normalize_gives_unit_norm_and_preserves_direction():
    v = np.array([3.0, 4.0, 0.0])
    u = mathx.normalize(v)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.allclose(u, v / 5.0)


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        mathx.normalize(np.zeros(8))


def test_normalize_is_scale_invariant():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(32)
    assert np.allclose(mathx.normalize(v), mathx.normalize(1e6 * v))


def test_cosine_and_orient_like():
    a = np.array([1.0, 0.0])
    assert np.isclose(mathx.cosine(a, np.array([2.0, 0.0])), 1.0)
    assert np.isclose(mathx.cosine(a, np.array([0.0, 5.0])), 0.0)
    assert np.isclose(mathx.abs_cosine(a, -a), 1.0)
    assert np.allclose(mathx.orient_like(-a, a), a)
    assert np.allclose(mathx.orient_like(a, a), a)


def test_project_out_removes_component():
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((3, 16))
    v = rng.standard_normal(16)
    r = mathx.project_out(v, basis)
    assert np.allclose(basis @ r, 0.0, atol=1e-10)
    # the removed part is exactly the component in the basis span
    assert np.allclose(mathx.project_out(r, basis), r, atol=1e-10)


# -- effective rank --------------------------------------------------------


def test_effective_rank_of_rank_one_is_one():
    assert np.isclose(mathx.effective_rank(np.array([5.0, 0.0, 0.0])), 1.0)


def test_effective_rank_of_k_equal_values_is_k():
    for k in (1, 3, 7):
        assert np.isclose(mathx.effective_rank(np.full(k, 2.0)), k)


def test_effective_rank_is_scale_invariant():
    sv = np.array([4.0, 2.0, 1.0, 0.5])
    assert np.isclose(mathx.effective_rank(sv), mathx.effective_rank(1e3 * sv))


def test_effective_rank_is_between_one_and_n():
    rng = np.random.default_rng(2)
    for _ in range(20):
        sv = np.abs(rng.standard_normal(9)) + 1e-6
        assert 1.0 - 1e-9 <= mathx.effective_rank(sv) <= 9.0 + 1e-9


def test_effective_rank_zero_matrix():
    assert mathx.effective_rank(np.zeros(5)) == 0.0


def test_effective_rank_matches_definition_on_a_known_matrix():
    # D with exactly two orthogonal directions of variance 9 and 1.
    D = np.zeros((4, 3))
    D[:, 0] = [3.0, -3.0, 3.0, -3.0]
    D[:, 1] = [1.0, 1.0, -1.0, -1.0]
    sv = mathx.singular_values(D, center=True)
    s2 = np.sort(sv**2)[::-1][:2]
    expected = (s2.sum() ** 2) / (s2**2).sum()
    assert np.isclose(mathx.effective_rank(sv), expected)


def test_d_frac_counts_components():
    # variances 89, 6, 5 -> 90% needs 2 components
    sv = np.sqrt(np.array([89.0, 6.0, 5.0]))
    assert mathx.d_frac(sv, 0.9) == 2
    assert mathx.d_frac(sv, 0.5) == 1
    assert mathx.d_frac(sv, 0.999) == 3


def test_d_frac_boundary_is_inclusive():
    """A component explaining exactly 90% satisfies the 90% criterion."""
    sv = np.sqrt(np.array([90.0, 5.0, 5.0]))
    assert mathx.d_frac(sv, 0.9) == 1


def test_d_frac_rank_one_needs_one_component():
    assert mathx.d_frac(np.array([7.0, 0.0, 0.0]), 0.9) == 1


def test_spectrum_metrics_on_isotropic_data_approaches_full_rank():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((400, 5))
    m = mathx.spectrum_metrics(D)
    assert 4.0 < m.effective_rank <= 5.0
    assert m.d90 in (4, 5)


def test_centering_is_applied():
    D = np.ones((10, 4)) * 3.0  # constant rows: zero variance after centering
    assert mathx.spectrum_metrics(D, center=True).total_variance < 1e-20
    assert mathx.spectrum_metrics(D, center=False).total_variance > 1.0


# -- PCA direction extraction ---------------------------------------------


def test_pc1_recovers_a_planted_direction():
    rng = np.random.default_rng(4)
    d = 24
    v = mathx.normalize(rng.standard_normal(d))
    coeffs = rng.standard_normal(200)[:, None]
    D = coeffs * v[None, :] + 0.01 * rng.standard_normal((200, d))
    got, ratio = mathx.pc1(D, center=True)
    assert mathx.abs_cosine(got, v) > 0.99
    assert ratio > 0.9


def test_pc1_uncentered_recovers_the_shared_mean_direction():
    """The key property behind decision D1: with a large shared offset, the
    uncentered PC1 follows the mean while the centered PC1 does not."""
    rng = np.random.default_rng(5)
    d = 32
    shared = mathx.normalize(rng.standard_normal(d))
    varying = mathx.normalize(rng.standard_normal(d))
    D = 10.0 * shared[None, :] + rng.standard_normal((300, 1)) * varying[None, :]
    v_unc, _ = mathx.pc1(D, center=False)
    v_cen, _ = mathx.pc1(D, center=True)
    assert mathx.abs_cosine(v_unc, shared) > 0.98
    assert mathx.abs_cosine(v_cen, varying) > 0.98


def test_pc1_is_unit_norm_and_sign_orientable():
    rng = np.random.default_rng(6)
    D = rng.standard_normal((50, 10)) + 5.0
    v, _ = mathx.pc1(D)
    assert np.isclose(np.linalg.norm(v), 1.0)
    mu = D.mean(axis=0)
    assert mathx.cosine(mathx.orient_like(v, mu), mu) > 0


def test_pc1_rejects_non_matrix():
    with pytest.raises(ValueError):
        mathx.pc1(np.zeros(5))


# -- orthogonal random controls -------------------------------------------


def test_random_unit_is_unit_norm():
    rng = np.random.default_rng(7)
    for _ in range(10):
        assert np.isclose(np.linalg.norm(mathx.random_unit(rng, 64)), 1.0)


def test_random_unit_orthogonal_is_exactly_orthogonal_and_unit():
    rng = np.random.default_rng(8)
    v = mathx.random_unit(rng, 128)
    for _ in range(20):
        w = mathx.random_unit_orthogonal(rng, v)
        assert np.isclose(np.linalg.norm(w), 1.0)
        assert abs(float(np.dot(w, v))) < 1e-12


def test_random_controls_are_seed_reproducible():
    a = mathx.random_unit(np.random.default_rng(9), 32)
    b = mathx.random_unit(np.random.default_rng(9), 32)
    assert np.allclose(a, b)


def test_random_unit_orthogonal_spans_the_complement():
    """Sampling should not collapse onto a single direction in the complement."""
    rng = np.random.default_rng(10)
    v = mathx.random_unit(rng, 16)
    W = np.stack([mathx.random_unit_orthogonal(rng, v) for _ in range(64)])
    assert mathx.effective_rank(mathx.singular_values(W, center=False)) > 10


# -- blockwise response decomposition -------------------------------------


def test_blockwise_response_is_the_layer_difference():
    rng = np.random.default_rng(11)
    deltas = rng.standard_normal((6, 4, 3))
    b = mathx.blockwise_response(deltas)
    assert b.shape == (5, 4, 3)
    assert np.allclose(b[2], deltas[3] - deltas[2])


def test_blockwise_response_telescopes():
    rng = np.random.default_rng(12)
    deltas = rng.standard_normal((7, 5, 8))
    b = mathx.blockwise_response(deltas)
    assert np.allclose(b.sum(axis=0), deltas[-1] - deltas[0])


def test_blockwise_response_rejects_wrong_rank():
    with pytest.raises(ValueError):
        mathx.blockwise_response(np.zeros((3, 4)))


def test_new_subspace_fraction_is_zero_inside_the_inherited_subspace():
    rng = np.random.default_rng(13)
    basis = np.linalg.qr(rng.standard_normal((16, 2)))[0].T  # (2, 16)
    D = rng.standard_normal((40, 2)) @ basis
    B = rng.standard_normal((40, 2)) @ basis  # lives in the same 2-D span
    assert mathx.new_subspace_fraction(D, B, frac=0.999) < 1e-10


def test_new_subspace_fraction_is_one_when_fully_orthogonal():
    rng = np.random.default_rng(14)
    Q = np.linalg.qr(rng.standard_normal((16, 4)))[0]
    D = rng.standard_normal((40, 2)) @ Q[:, :2].T
    B = rng.standard_normal((40, 2)) @ Q[:, 2:].T
    assert mathx.new_subspace_fraction(D, B, frac=0.999) > 1.0 - 1e-10


def test_new_subspace_fraction_is_a_fraction():
    rng = np.random.default_rng(15)
    for _ in range(20):
        D = rng.standard_normal((30, 12))
        B = rng.standard_normal((30, 12))
        n = mathx.new_subspace_fraction(D, B)
        assert 0.0 <= n <= 1.0


def test_new_subspace_fraction_zero_block_response():
    D = np.random.default_rng(16).standard_normal((10, 5))
    assert mathx.new_subspace_fraction(D, np.zeros((10, 5))) == 0.0


# -- layerwise gain / conversion / magnitude -------------------------------


def test_perturbation_magnitude_is_the_norm_ratio():
    delta = np.array([[3.0, 4.0], [0.0, 1.0]])
    base = np.array([[10.0, 0.0], [0.0, 4.0]])
    assert np.allclose(mathx.perturbation_magnitude(delta, base), [0.5, 0.25])


def test_layer_gain_matches_definition():
    d0 = np.array([[2.0, 0.0]])
    d1 = np.array([[6.0, 0.0]])
    assert np.allclose(mathx.layer_gain(d0, d1), [3.0])
    assert np.allclose(np.log(mathx.layer_gain(d0, d0)), [0.0])


def test_block_conversion_matches_definition():
    delta = np.array([[4.0, 0.0]])
    block = np.array([[0.0, 2.0]])
    assert np.allclose(mathx.block_conversion(delta, block), [0.5])


def test_gain_and_conversion_are_consistent_for_pure_scaling():
    """If a block only rescales delta by g, then C = |g - 1| and G = g."""
    rng = np.random.default_rng(17)
    delta = rng.standard_normal((25, 9))
    for g in (0.5, 1.0, 2.0):
        nxt = g * delta
        block = nxt - delta
        assert np.allclose(mathx.layer_gain(delta, nxt), g)
        assert np.allclose(mathx.block_conversion(delta, block), abs(g - 1.0))


def test_gain_of_zero_perturbation_is_finite():
    z = np.zeros((3, 4))
    assert np.all(np.isfinite(mathx.layer_gain(z, z)))


# -- uncertainty and null comparison --------------------------------------


def test_bootstrap_ci_brackets_the_median_and_is_seed_stable():
    rng = np.random.default_rng(18)
    x = rng.normal(5.0, 1.0, size=400)
    med, lo, hi = mathx.bootstrap_ci(x, seed=0)
    assert lo <= med <= hi
    assert abs(med - 5.0) < 0.3
    assert mathx.bootstrap_ci(x, seed=0) == mathx.bootstrap_ci(x, seed=0)


def test_summarize_ignores_non_finite_values():
    s = mathx.summarize([1.0, 2.0, np.nan, np.inf, 3.0], n_boot=100)
    assert s["n"] == 3
    assert np.isclose(s["median"], 2.0)


def test_empirical_p_value_bounds():
    null = np.arange(10, dtype=float)
    assert mathx.empirical_p_value(100.0, null) == pytest.approx(1 / 11)
    assert mathx.empirical_p_value(-1.0, null) == pytest.approx(11 / 11)


def test_z_against_null():
    null = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    z = mathx.z_against_null(2.0 + 2 * np.std(null, ddof=1), null)
    assert np.isclose(z, 2.0)
