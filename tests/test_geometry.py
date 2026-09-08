import numpy as np
import pytest

from directions.geometry import (
    block_responses,
    cosine,
    d90,
    effective_rank,
    layerwise_example_metrics,
    matched_random_controls,
    new_subspace_fraction,
    normalize,
    pairwise_abs_cosines,
    pca_direction,
    random_orthogonal_unit_vector,
    random_unit_vector,
    spectrum,
    stability,
)


def test_normalize_unit_norm_and_zero_error():
    v = np.array([3.0, 4.0])
    assert np.allclose(normalize(v), [0.6, 0.8])
    assert np.allclose(np.linalg.norm(normalize(np.random.default_rng(0).standard_normal((5, 7)), axis=1), axis=1), 1)
    with pytest.raises(ValueError):
        normalize(np.zeros(3))


def test_effective_rank_known_cases():
    rng = np.random.default_rng(0)
    # rank-1 (uncentered) matrix -> 1
    u = rng.standard_normal((20, 1))
    v = rng.standard_normal((1, 10))
    assert effective_rank(u @ v, center=False) == pytest.approx(1.0)
    # k orthonormal rows with equal singular values -> k
    q, _ = np.linalg.qr(rng.standard_normal((10, 4)))
    assert effective_rank(q.T, center=False) == pytest.approx(4.0)
    # known spectrum: s^2 = [4, 1] -> (5)^2 / (16 + 1) = 25/17
    X = np.diag([2.0, 1.0])
    assert effective_rank(X, center=False) == pytest.approx(25 / 17)
    # zero centered matrix (identical rows) -> nan
    assert np.isnan(effective_rank(np.ones((5, 3)), center=True))


def test_d90_and_spectrum():
    X = np.diag([3.0, 1.0, 0.1])  # s^2 = 9, 1, 0.01 -> 90% needs 2
    assert d90(X, center=False) == 2
    assert d90(np.ones((4, 3))) is None
    sp = spectrum(X, frac=0.9, center=False)
    assert sp.d90 == 2 and sp.top_subspace.shape == (2, 3)
    assert sp.total_variance == pytest.approx(10.01)
    assert sp.effective_rank == pytest.approx(effective_rank(X, center=False))


def test_pca_direction_recovers_planted_direction_uncentered():
    rng = np.random.default_rng(1)
    d = 50
    v = normalize(rng.standard_normal(d))
    # shared component + small per-example noise
    diffs = 5.0 * v[None, :] + 0.3 * rng.standard_normal((40, d))
    res = pca_direction(diffs, center=False)
    assert abs(cosine(res.direction, v)) > 0.99
    assert res.direction @ diffs.mean(axis=0) > 0  # sign aligned with the mean
    assert res.cos_with_mean > 0.99
    assert 0.8 < res.explained_variance_ratio <= 1.0
    assert np.linalg.norm(res.direction) == pytest.approx(1.0)


def test_pca_direction_centering_discards_shared_component():
    rng = np.random.default_rng(2)
    d = 30
    v = normalize(rng.standard_normal(d))
    w = normalize(rng.standard_normal(d))
    w = normalize(w - (w @ v) * v)  # orthogonal to v
    # large shared v component, variation along w between examples
    diffs = 10.0 * v[None, :] + rng.standard_normal((60, 1)) * w[None, :] + 0.05 * rng.standard_normal((60, d))
    unc = pca_direction(diffs, center=False).direction
    cen = pca_direction(diffs, center=True).direction
    assert abs(cosine(unc, v)) > 0.99
    assert abs(cosine(cen, w)) > 0.99


def test_stability_and_pairwise():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pc = pairwise_abs_cosines(a)
    assert pc.shape == (3,)
    assert stability(a) == pytest.approx(0.0)
    assert stability(np.array([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(1.0)  # sign-invariant
    assert stability(np.array([[1.0, 0.0]])) == 1.0


def test_random_controls_norm_and_orthogonality():
    rng = np.random.default_rng(3)
    v = normalize(rng.standard_normal(64))
    u = random_unit_vector(rng, 64)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    o = random_orthogonal_unit_vector(rng, v)
    assert np.linalg.norm(o) == pytest.approx(1.0)
    assert abs(o @ v) < 1e-12
    ctrls = matched_random_controls(rng, v, 6)
    assert [k for k, _ in ctrls] == ["isotropic", "orthogonal"] * 3
    for kind, c in ctrls:
        assert np.linalg.norm(c) == pytest.approx(1.0)
        if kind == "orthogonal":
            assert abs(c @ v) < 1e-12
    # isotropic controls have the expected |cos| with v: ~1/sqrt(d) scale
    iso = np.array([abs(random_unit_vector(rng, 64) @ v) for _ in range(500)])
    assert iso.mean() < 3 / np.sqrt(64)


def test_block_responses_and_new_subspace_fraction():
    rng = np.random.default_rng(4)
    delta = rng.standard_normal((4, 5, 6))  # (L+1, n, d)
    b = block_responses(delta)
    assert b.shape == (3, 5, 6)
    assert np.allclose(b[1], delta[2] - delta[1])
    # B inside the subspace -> 0; B orthogonal to it -> 1
    basis = np.eye(6)[:2]
    inside = rng.standard_normal((5, 2)) @ basis
    outside = rng.standard_normal((5, 3)) @ np.eye(6)[3:]
    assert new_subspace_fraction(inside, basis) == pytest.approx(0.0, abs=1e-12)
    assert new_subspace_fraction(outside, basis) == pytest.approx(1.0)
    mixed = inside + outside
    expected = np.sum(outside**2) / np.sum(mixed**2)
    assert new_subspace_fraction(mixed, basis) == pytest.approx(expected)
    assert np.isnan(new_subspace_fraction(np.zeros((5, 6)), basis))
    assert new_subspace_fraction(mixed, np.zeros((0, 6))) == 1.0


def test_layerwise_gain_conversion_hand_computed():
    # one example, d = 2, three read points
    v = np.array([1.0, 0.0])
    base = np.zeros((3, 1, 2))
    base[0, 0] = [2.0, 0.0]
    base[1, 0] = [0.0, 4.0]
    base[2, 0] = [1.0, 0.0]
    delta = np.zeros((3, 1, 2))
    delta[0, 0] = [1.0, 0.0]  # ||delta_0|| = 1, aligned with v
    delta[1, 0] = [0.0, 2.0]  # ||delta_1|| = 2, orthogonal to v
    delta[2, 0] = [3.0, 4.0]  # ||delta_2|| = 5
    m = layerwise_example_metrics(delta, base, v)
    assert np.allclose(m.magnitude[:, 0], [1 / 2, 2 / 4, 5 / 1])
    assert np.allclose(m.gain[:, 0], [2 / 1, 5 / 2])
    assert np.allclose(m.log_gain[:, 0], np.log([2.0, 2.5]))
    # b_0 = delta_1 - delta_0 = (-1, 2) -> norm sqrt(5); C_0 = sqrt(5)/1
    # b_1 = delta_2 - delta_1 = (3, 2) -> norm sqrt(13); C_1 = sqrt(13)/2
    assert np.allclose(m.conversion[:, 0], [np.sqrt(5), np.sqrt(13) / 2])
    assert np.allclose(m.alignment[:, 0], [1.0, 0.0, 3 / 5])
    # zero perturbation -> nan gain/conversion/alignment, not an exception
    z = layerwise_example_metrics(np.zeros((3, 1, 2)), base, v)
    assert np.all(np.isnan(z.gain)) and np.all(np.isnan(z.alignment))
    assert np.allclose(z.magnitude, 0.0)


def test_layerwise_metrics_batched_matches_loop():
    rng = np.random.default_rng(5)
    L1, n, d = 6, 7, 9
    delta = rng.standard_normal((L1, n, d))
    base = rng.standard_normal((L1, n, d))
    v = normalize(rng.standard_normal(d))
    m = layerwise_example_metrics(delta, base, v)
    for l in range(L1):
        for i in range(n):
            assert m.magnitude[l, i] == pytest.approx(np.linalg.norm(delta[l, i]) / np.linalg.norm(base[l, i]))
            assert m.alignment[l, i] == pytest.approx(abs(cosine(delta[l, i], v)))
            if l < L1 - 1:
                assert m.gain[l, i] == pytest.approx(np.linalg.norm(delta[l + 1, i]) / np.linalg.norm(delta[l, i]))
                assert m.conversion[l, i] == pytest.approx(
                    np.linalg.norm(delta[l + 1, i] - delta[l, i]) / np.linalg.norm(delta[l, i])
                )


def test_direction_readouts_hand_computed():
    from directions.geometry import direction_readouts

    rng = np.random.default_rng(3)
    L1, n, d = 3, 4, 5
    delta = rng.standard_normal((L1, n, d))
    V = rng.standard_normal((L1, d))
    G = rng.standard_normal((L1, n, d))
    out = direction_readouts(delta, V, G)
    for l in range(L1):
        for x in range(n):
            dv = delta[l, x]
            v = V[l] / np.linalg.norm(V[l])
            assert out["task_alignment"][l, x] == pytest.approx(dv @ v / np.linalg.norm(dv))
            g = G[l, x]
            assert out["gradient_alignment"][l, x] == pytest.approx(dv @ g / (np.linalg.norm(dv) * np.linalg.norm(g)))
            assert out["gradient_projection"][l, x] == pytest.approx(dv @ g)
    # the sign is kept (a direction pointing against v gives -1) and missing inputs give nan
    delta2 = -np.broadcast_to(V[:, None, :], (L1, n, d)).copy()
    assert np.allclose(direction_readouts(delta2, V, None)["task_alignment"], -1.0)
    partial = direction_readouts(delta, None, G)
    assert np.all(np.isnan(partial["task_alignment"])) and not np.any(np.isnan(partial["gradient_alignment"]))
    assert np.all(np.isnan(direction_readouts(delta, None, None)["gradient_projection"]))
    # a zero perturbation is undefined, not an error
    zero = np.zeros((L1, n, d))
    assert np.all(np.isnan(direction_readouts(zero, V, G)["task_alignment"]))
