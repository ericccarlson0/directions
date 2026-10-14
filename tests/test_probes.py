"""The conditional probe and its nulls (docs/DECISIONS.md D46) on synthetic states with known content."""

import numpy as np
import pytest
import torch

from directions.probes import (
    _standardise,
    conditional_probe,
    fold_ids,
    match_rates,
    permute_within,
    probe_statistics,
    ridge_cv_accuracy,
)


def _one_lane_like(n=360, d=48, seed=1):
    rng = np.random.default_rng(seed)
    Y, c = rng.integers(0, 10, n), rng.integers(0, 10, n)
    S1 = Y + c
    allowed = np.zeros((n, 19), dtype=bool)
    for i in range(n):
        allowed[i, Y[i] : Y[i] + 10] = True
    ey, es = rng.normal(size=(10, d)), rng.normal(size=(19, d))
    feats = np.stack([rng.normal(size=(n, d)),                          # nothing
                      3 * ey[Y] + rng.normal(size=(n, d)),              # the answer only
                      3 * ey[Y] + es[S1] + rng.normal(size=(n, d))])    # the answer and S1
    return feats, Y, S1, allowed


def test_conditional_probe_sees_information_beyond_the_condition_and_only_that():
    feats, Y, S1, allowed = _one_lane_like()
    r = conditional_probe(feats, S1, Y, allowed, n_perm=40, seed=0)
    assert r["carried"].tolist() == [False, False, True]
    assert r["accuracy"][2] > 0.9 and abs(r["excess"][1]) < 0.06
    # the answer itself is carried wherever it is encoded
    ry = conditional_probe(feats, Y, np.zeros_like(Y), np.ones((len(Y), 10), dtype=bool), n_perm=40, seed=0)
    assert ry["carried"].tolist() == [False, True, True]


def test_probe_is_deterministic_and_a_single_read_point_matches_the_full_run():
    feats, Y, S1, allowed = _one_lane_like(n=120)
    a = conditional_probe(feats, S1, Y, allowed, n_perm=10, seed=5)
    b = conditional_probe(feats, S1, Y, allowed, n_perm=10, seed=5, read_points=[2])
    np.testing.assert_array_equal(a["null"][2:], b["null"])
    np.testing.assert_array_equal(a["accuracy"][2:], b["accuracy"])


def test_leave_one_out_closed_form_matches_explicit_refits():
    rng = np.random.default_rng(0)
    n, d, C = 40, 12, 5
    x = torch.as_tensor(rng.normal(size=(n, d)))
    y = rng.integers(0, C, n)
    xs, _ = _standardise(x, x)
    K = xs @ xs.T / xs.shape[1]
    T = torch.nn.functional.one_hot(torch.as_tensor(y), C).to(torch.float64)
    tc = T - T.mean(0)
    lam = 0.7
    s, U = torch.linalg.eigh(K)
    w = s.clamp_min(0) / (s.clamp_min(0) + lam)
    fit = U @ (w[:, None] * (U.T @ tc))
    h = ((U * U) @ w)[:, None]
    loo = tc - (tc - fit) / (1 - h)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        alpha = torch.linalg.solve(K[tr][:, tr] + lam * torch.eye(n - 1, dtype=torch.float64), tc[tr])
        torch.testing.assert_close(K[i, tr] @ alpha, loo[i], rtol=0, atol=1e-10)


def test_restricted_prediction_never_leaves_the_allowed_classes():
    rng = np.random.default_rng(3)
    n = 60
    x = torch.as_tensor(rng.normal(size=(n, 8)))
    y = rng.integers(0, 2, n)  # the true class is always 0 or 1; class 2 is never allowed
    allowed = torch.zeros(n, 3, dtype=torch.bool)
    allowed[:, :2] = True
    acc, _ = ridge_cv_accuracy(x, torch.as_tensor(y)[None], allowed, fold_ids(n, 3, rng), [1.0])
    assert 0 <= float(acc[0]) <= 1
    with pytest.raises(ValueError, match="allowed"):
        conditional_probe(np.zeros((1, n, 4)), np.full(n, 2), np.zeros(n), allowed.numpy(), n_perm=2)


def test_permute_within_keeps_each_group_s_values():
    rng = np.random.default_rng(0)
    y, g = rng.integers(0, 9, 200), rng.integers(0, 4, 200)
    p = permute_within(y, g, rng)
    for k in range(4):
        assert sorted(p[g == k]) == sorted(y[g == k])
    assert (p != y).any()


def test_constant_states_are_never_carried():
    n = 50
    y = np.arange(n) % 5
    r = conditional_probe(np.ones((2, n, 6)), y, np.zeros(n), np.ones((n, 5), dtype=bool), n_perm=10, seed=0)
    assert not r["carried"].any()


def test_family_wise_threshold_and_match_rates():
    null = np.random.default_rng(0).normal(0.1, 0.01, size=(5, 200))
    st = probe_statistics(np.array([0.1, 0.1, 0.2, 0.1, 0.1]), null)
    assert st["carried"].tolist() == [False, False, True, False, False] and st["p_fwer"][2] < 0.01
    pred = np.array([[1, 2, 3, 4, 5, 6] * 10, [0] * 60])
    cand = np.array([1, 2, 3, 4, 5, 6] * 10)
    r = match_rates(pred, cand, np.ones(60, dtype=bool), n_perm=50, seed=0)
    assert r["rate"].tolist() == [1.0, 0.0] and r["carried"].tolist() == [True, False]
