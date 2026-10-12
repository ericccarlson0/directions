"""scripts/single_control_summary.py: the statistical helpers of the single-control close-out."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # a dependency on fable; main's environment has none

_SPEC = importlib.util.spec_from_file_location("single_control_summary",
                                               Path(__file__).resolve().parents[1] / "scripts" / "single_control_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_first_crossing_interpolates_between_grid_points():
    assert summary.first_crossing([10, 13, 16], [0.5, 0.8, 1.0]) == pytest.approx(14.5)
    assert summary.first_crossing([10, 13, 16], [0.95, 0.8, 1.0]) == 10.0  # already past the share at the first point
    assert summary.first_crossing([10, 13, 16], [0.1, None, 0.95]) == 16.0  # a missing value breaks the interpolation
    assert summary.first_crossing([10, 13], [0.1, 0.2]) is None
    assert summary.first_reaching([10, 13, 16], [0.5, 0.8, 1.0]) == 16


def test_pooled_within_slope_ignores_group_intercepts():
    groups = {g: [(x, 3.0 * g + 0.5 * x) for x in (6, 8, 11, 14)] for g in range(4)}
    groups["single"] = [(5, 100.0)]  # a group with one point carries no within-group information
    assert summary.pooled_within_slope(groups) == pytest.approx(0.5)


def test_cluster_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(0)
    groups = {g: [(x, g + 0.05 * x + rng.normal(0, 0.1)) for x in (6, 8, 11, 14)] for g in range(12)}
    out = summary.cluster_bootstrap_slope(groups, np.random.default_rng(1))
    assert out["ci95"][0] <= out["slope"] <= out["ci95"][1]
    assert out["n_groups"] == 12 and out["n_points"] == 48


def test_spearman_exact_and_sign_test():
    sp = summary.spearman_exact([1, 2, 3, 4, 5], [2, 4, 5, 7, 9])
    assert sp["rho"] == pytest.approx(1.0)
    assert sp["p_one_sided"] == pytest.approx(1 / 120)  # only the identity permutation reaches rho = 1
    assert sp["p_two_sided"] == pytest.approx(2 / 120)
    assert summary.sign_test(10, 10) == pytest.approx(2 / 1024)
    assert summary.sign_test(5, 10) == 1.0


def test_kendall_tau_perm_sign():
    rng = np.random.default_rng(0)
    up = summary.kendall_tau_perm([0, 0, 1, 1, 2, 2], [0.1, 0.2, 0.3, 0.5, 0.8, 0.9], rng)
    assert up["tau_b"] > 0.8
    down = summary.kendall_tau_perm([0, 1, 2, 3], [4.0, 3.0, 2.0, 1.0], rng)
    assert down["tau_b"] == pytest.approx(-1.0)
