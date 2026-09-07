"""Statistical utilities: bootstraps, paired tests, null comparisons, profile summaries.

Everything is seeded explicitly; nothing uses global random state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmedian(x))


@dataclass(frozen=True)
class MedianCI:
    median: float
    low: float
    high: float
    n: int


def bootstrap_median_ci(
    x: np.ndarray, rng: np.random.Generator, n_boot: int = 1000, alpha: float = 0.05
) -> MedianCI:
    """Median with a percentile bootstrap CI (nan entries dropped)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = x.shape[0]
    if n == 0:
        return MedianCI(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return MedianCI(float(x[0]), float(x[0]), float(x[0]), 1)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    return MedianCI(
        median=float(np.median(x)),
        low=float(np.quantile(meds, alpha / 2)),
        high=float(np.quantile(meds, 1 - alpha / 2)),
        n=int(n),
    )


def bootstrap_median_ci_rows(
    X: np.ndarray, rng: np.random.Generator, n_boot: int = 1000, alpha: float = 0.05
) -> dict[str, list[float]]:
    """Row-wise :func:`bootstrap_median_ci` for ``X`` shaped (rows, n)."""
    out = {"median": [], "low": [], "high": [], "n": []}
    for row in np.asarray(X, dtype=np.float64):
        ci = bootstrap_median_ci(row, rng, n_boot=n_boot, alpha=alpha)
        out["median"].append(ci.median)
        out["low"].append(ci.low)
        out["high"].append(ci.high)
        out["n"].append(ci.n)
    return out


@dataclass(frozen=True)
class PairedTest:
    mean_diff: float
    median_diff: float
    p_value: float  # one-sided bootstrap P(mean diff <= 0)
    ci_low: float
    ci_high: float
    n: int
    frac_improved: float


def paired_bootstrap_test(
    treated: np.ndarray,
    control: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> PairedTest:
    """One-sided paired bootstrap for ``mean(treated - control) > 0``.

    The p-value is the fraction of bootstrap resamples (over examples) whose
    mean difference is ``<= 0``, with a +1 continuity correction so that p is
    never exactly zero.
    """
    d = np.asarray(treated, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = d.shape[0]
    if n == 0:
        return PairedTest(*(float("nan"),) * 5, 0, float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    p = (1.0 + float(np.sum(means <= 0))) / (n_boot + 1.0)
    return PairedTest(
        mean_diff=float(d.mean()),
        median_diff=float(np.median(d)),
        p_value=float(p),
        ci_low=float(np.quantile(means, alpha / 2)),
        ci_high=float(np.quantile(means, 1 - alpha / 2)),
        n=int(n),
        frac_improved=float(np.mean(d > 0)),
    )


@dataclass(frozen=True)
class NullComparison:
    value: float
    null_mean: float
    null_std: float
    z: float
    p_upper: float  # (1 + #{null >= value}) / (1 + n_null)
    p_lower: float  # (1 + #{null <= value}) / (1 + n_null)
    n_null: int


def compare_to_null(value: float, null_values: np.ndarray) -> NullComparison:
    """Compare a real-control statistic to a matched random-control distribution."""
    null = np.asarray(null_values, dtype=np.float64)
    null = null[~np.isnan(null)]
    n = null.shape[0]
    if n == 0 or np.isnan(value):
        return NullComparison(float(value), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n)
    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if n > 1 else 0.0
    z = (value - mu) / sd if sd > 0 else (0.0 if value == mu else float(np.sign(value - mu)) * np.inf)
    return NullComparison(
        value=float(value),
        null_mean=mu,
        null_std=sd,
        z=float(z),
        p_upper=(1.0 + float(np.sum(null >= value))) / (1.0 + n),
        p_lower=(1.0 + float(np.sum(null <= value))) / (1.0 + n),
        n_null=int(n),
    )


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties handled by averaging."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ignoring positions where either input is nan."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def entropy_ratio(mass: np.ndarray) -> float:
    """Shannon entropy of a normalised non-negative mass vector over ``log(len)``."""
    m = np.asarray(mass, dtype=np.float64)
    m = m[~np.isnan(m)]
    if m.shape[0] < 2 or m.sum() <= 0:
        return float("nan")
    p = m / m.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(m.shape[0]))


def centre_of_mass(mass: np.ndarray, positions: np.ndarray) -> float:
    """Mass-weighted mean of ``positions``."""
    m = np.asarray(mass, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    ok = ~np.isnan(m)
    if ok.sum() == 0 or m[ok].sum() <= 0:
        return float("nan")
    return float((m[ok] * pos[ok]).sum() / m[ok].sum())
