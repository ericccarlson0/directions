import numpy as np
import pytest

from directions.seeds import derive_seed, rng_for
from directions.stats import (
    bootstrap_median_ci,
    bootstrap_median_ci_rows,
    centre_of_mass,
    compare_to_null,
    entropy_ratio,
    paired_bootstrap_test,
    rankdata,
    spearman,
)


def test_derive_seed_stable_and_distinct():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2)
    assert derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
    assert derive_seed(1, "a") != derive_seed(2, "a")
    assert rng_for(7, "x").integers(0, 10**9) == rng_for(7, "x").integers(0, 10**9)


def test_paired_bootstrap_detects_effect_and_respects_null():
    rng = np.random.default_rng(0)
    control = rng.standard_normal(64)
    treated = control + 0.5 + 0.2 * rng.standard_normal(64)
    t = paired_bootstrap_test(treated, control, np.random.default_rng(1), n_boot=1000)
    assert t.p_value < 0.01 and t.mean_diff > 0.3 and t.n == 64 and t.ci_low > 0
    null = paired_bootstrap_test(control + 0.2 * rng.standard_normal(64), control, np.random.default_rng(2), n_boot=1000)
    assert null.p_value > 0.05
    # under the null, p is roughly uniform: a modest fraction of tests pass at 0.05
    ps = []
    for i in range(200):
        c = rng.standard_normal(30)
        ps.append(paired_bootstrap_test(c + rng.standard_normal(30), c, np.random.default_rng(i), n_boot=200).p_value)
    assert 0.0 <= np.mean(np.array(ps) <= 0.05) <= 0.15


def test_bootstrap_median_ci_contains_median():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100) + 3
    ci = bootstrap_median_ci(x, np.random.default_rng(1), n_boot=500)
    assert ci.low <= ci.median <= ci.high and ci.n == 100
    rows = bootstrap_median_ci_rows(np.stack([x, np.full(100, np.nan)]), np.random.default_rng(2), n_boot=100)
    assert rows["n"] == [100, 0] and np.isnan(rows["median"][1])


def test_compare_to_null():
    c = compare_to_null(3.0, np.array([0.0, 1.0, 2.0, -1.0]))
    assert c.p_upper == pytest.approx(1 / 5)
    assert c.p_lower == pytest.approx(5 / 5)
    assert c.z > 1
    assert c.n_null == 4
    assert np.isnan(compare_to_null(1.0, np.array([])).z)


def test_rankdata_and_spearman():
    assert np.allclose(rankdata(np.array([10, 20, 20, 5])), [2, 3.5, 3.5, 1])
    x = np.arange(10, dtype=float)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.array([1.0, np.nan]), np.array([1.0, 2.0])))


def test_entropy_ratio_and_centre_of_mass():
    assert entropy_ratio(np.ones(8)) == pytest.approx(1.0)
    assert entropy_ratio(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert centre_of_mass(np.array([1.0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(0.5)
    assert centre_of_mass(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
