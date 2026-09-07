"""Layerwise perturbation measurements downstream of an intervention.

Inputs are baseline and steered residuals at the intervened token for every
read point ``0..L`` and every held-out example. Per-example quantities (S, log G,
C, A) are summarised by medians with bootstrap CIs; dimensionality quantities
(d_eff, d90, N_l) are computed from the full example matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EvaluationConfig
from .geometry import (
    LayerwiseExampleMetrics,
    block_responses,
    effective_rank,
    layerwise_example_metrics,
    new_subspace_fraction,
    spectrum,
)
from .stats import bootstrap_median_ci, bootstrap_median_ci_rows, compare_to_null

PER_EXAMPLE_METRICS = ("magnitude", "log_gain", "conversion", "alignment")
MATRIX_METRICS = ("d_eff", "d90", "total_centered_variance", "d_eff_uncentered", "new_subspace", "new_subspace_uncentered")


@dataclass
class LayerwiseProfile:
    intervention_layer: int
    n_layers: int
    n_examples: int
    per_example: LayerwiseExampleMetrics
    summaries: dict[str, dict[str, list[float]]]  # metric -> {median, low, high, n} over read points/blocks
    d_eff: np.ndarray  # (L+1,), nan where undefined (below and at the intervention layer)
    d90: np.ndarray  # (L+1,), nan where undefined
    total_centered_variance: np.ndarray  # (L+1,)
    d_eff_uncentered: np.ndarray  # (L+1,)
    new_subspace: np.ndarray  # (L,), centred P_l; nan at/below the intervention layer
    new_subspace_uncentered: np.ndarray  # (L,), uncentred P_l; defined from the intervention layer on
    cumulative_log_gain: dict[str, float]  # median + CI of log(||delta_L|| / ||delta_l*||)
    noise_floor: dict[str, float | None]
    pre_intervention_max_abs_delta: float

    def as_dict(self) -> dict[str, Any]:
        def lst(a: np.ndarray) -> list[float | None]:
            return [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in np.asarray(a, dtype=np.float64)]

        return {
            "intervention_layer": self.intervention_layer,
            "n_layers": self.n_layers,
            "n_examples": self.n_examples,
            "summaries": self.summaries,
            "d_eff": lst(self.d_eff),
            "d90": lst(self.d90),
            "total_centered_variance": lst(self.total_centered_variance),
            "d_eff_uncentered": lst(self.d_eff_uncentered),
            "new_subspace": lst(self.new_subspace),
            "new_subspace_uncentered": lst(self.new_subspace_uncentered),
            "cumulative_log_gain": self.cumulative_log_gain,
            "noise_floor": self.noise_floor,
            "pre_intervention_max_abs_delta": self.pre_intervention_max_abs_delta,
        }

    def metric_curve(self, name: str) -> np.ndarray:
        """A single per-layer curve for ``name`` (medians for per-example metrics)."""
        if name in PER_EXAMPLE_METRICS:
            return np.asarray(self.summaries[name]["median"], dtype=np.float64)
        return np.asarray(getattr(self, name), dtype=np.float64)


def _median_rows(X: np.ndarray) -> dict[str, list[float]]:
    """Row medians without bootstrap CIs (used for the many control profiles)."""
    out = {"median": [], "low": [], "high": [], "n": []}
    for row in np.asarray(X, dtype=np.float64):
        ok = row[~np.isnan(row)]
        out["median"].append(float(np.median(ok)) if ok.size else float("nan"))
        out["low"].append(float("nan"))
        out["high"].append(float("nan"))
        out["n"].append(int(ok.size))
    return out


def compute_profile(
    base: np.ndarray,
    steered: np.ndarray,
    v: np.ndarray,
    intervention_layer: int,
    cfg: EvaluationConfig,
    rng: np.random.Generator,
    with_ci: bool = True,
) -> LayerwiseProfile:
    """Full layerwise profile from residuals shaped (L+1, n, d).

    ``with_ci=False`` skips the bootstrap confidence intervals (medians only).
    """
    base = np.asarray(base, dtype=np.float64)
    steered = np.asarray(steered, dtype=np.float64)
    L1, n, _ = base.shape
    L = L1 - 1
    ls = intervention_layer
    delta = steered - base
    pre_max = float(np.abs(delta[:ls]).max()) if ls > 0 else 0.0

    ex = layerwise_example_metrics(delta, base, v)
    # Quantities are only defined from the intervention layer on.
    for arr in (ex.magnitude, ex.alignment):
        arr[:ls] = np.nan
    for arr in (ex.gain, ex.log_gain, ex.conversion):
        arr[:ls] = np.nan

    if with_ci:
        summaries = {
            name: bootstrap_median_ci_rows(getattr(ex, name), rng, n_boot=cfg.n_boot, alpha=cfg.ci_alpha)
            for name in PER_EXAMPLE_METRICS
        }
    else:
        summaries = {name: _median_rows(getattr(ex, name)) for name in PER_EXAMPLE_METRICS}
    with np.errstate(divide="ignore", invalid="ignore"):
        cum = np.log(ex.delta_norm[L] / ex.delta_norm[ls])
    if with_ci:
        cum_ci = bootstrap_median_ci(cum, rng, n_boot=cfg.n_boot, alpha=cfg.ci_alpha)
    else:
        from .stats import MedianCI

        ok = cum[~np.isnan(cum)]
        cum_ci = MedianCI(float(np.median(ok)) if ok.size else float("nan"), float("nan"), float("nan"), int(ok.size))

    d_eff = np.full(L1, np.nan)
    d90 = np.full(L1, np.nan)
    tot = np.full(L1, np.nan)
    d_eff_unc = np.full(L1, np.nan)
    N = np.full(L, np.nan)
    N_unc = np.full(L, np.nan)
    b = block_responses(delta)
    observed_at_ls: dict[str, float | None] = {"d_eff": None, "total_centered_variance": None, "d90": None}
    for l in range(ls, L1):
        D = delta[l]
        spec_c = spectrum(D, frac=cfg.variance_fraction, center=True)
        spec_u = spectrum(D, frac=cfg.variance_fraction, center=False)
        d_eff_unc[l] = spec_u.effective_rank
        if l == ls:
            # exact arithmetic: centered D is zero; the observed value is the bf16 noise floor
            observed_at_ls = {
                "d_eff": None if np.isnan(spec_c.effective_rank) else float(spec_c.effective_rank),
                "d90": None if spec_c.d90 is None else float(spec_c.d90),
                "total_centered_variance": float(spec_c.total_variance),
            }
            tot[l] = spec_c.total_variance
        else:
            d_eff[l] = spec_c.effective_rank
            d90[l] = np.nan if spec_c.d90 is None else spec_c.d90
            tot[l] = spec_c.total_variance
        if l < L:
            if l > ls:
                N[l] = new_subspace_fraction(b[l], spec_c.top_subspace)
            N_unc[l] = new_subspace_fraction(b[l], spec_u.top_subspace)

    next_var = tot[ls + 1] if ls + 1 < L1 else float("nan")
    var_ls = observed_at_ls["total_centered_variance"]
    ratio = None
    if var_ls is not None and not np.isnan(next_var) and next_var > 0:
        ratio = float(var_ls / next_var)
    noise_floor = {
        "d_eff_observed_at_intervention_layer": observed_at_ls["d_eff"],
        "d90_observed_at_intervention_layer": observed_at_ls["d90"],
        "centered_variance_at_intervention_layer": var_ls,
        "centered_variance_at_next_layer": None if np.isnan(next_var) else float(next_var),
        "variance_ratio_intervention_over_next": ratio,
    }
    return LayerwiseProfile(
        intervention_layer=ls,
        n_layers=L,
        n_examples=n,
        per_example=ex,
        summaries=summaries,
        d_eff=d_eff,
        d90=d90,
        total_centered_variance=tot,
        d_eff_uncentered=d_eff_unc,
        new_subspace=N,
        new_subspace_uncentered=N_unc,
        cumulative_log_gain={"median": cum_ci.median, "low": cum_ci.low, "high": cum_ci.high, "n": cum_ci.n},
        noise_floor=noise_floor,
        pre_intervention_max_abs_delta=pre_max,
    )


COMPARED_METRICS = PER_EXAMPLE_METRICS + MATRIX_METRICS


def _nanmedian_cols(M: np.ndarray, n_cols: int) -> np.ndarray:
    """Column-wise nanmedian that returns nan (without warnings) for all-nan columns."""
    out = np.full(n_cols, np.nan)
    if M.shape[0] == 0:
        return out
    for j in range(n_cols):
        col = M[:, j]
        col = col[~np.isnan(col)]
        if col.size:
            out[j] = float(np.median(col))
    return out


def compare_to_random(real: LayerwiseProfile, randoms: list[LayerwiseProfile]) -> dict[str, Any]:
    """Per-layer z and empirical p of every primary metric vs the random-control profiles."""
    out: dict[str, Any] = {"n_random": len(randoms), "metrics": {}}
    for name in COMPARED_METRICS:
        rv = real.metric_curve(name)
        null = np.stack([r.metric_curve(name) for r in randoms]) if randoms else np.zeros((0, rv.shape[0]))
        rows = []
        for l in range(rv.shape[0]):
            c = compare_to_null(float(rv[l]), null[:, l] if randoms else np.array([]))
            rows.append(c.__dict__)
        out["metrics"][name] = {
            "real": [None if np.isnan(x) else float(x) for x in rv],
            "null_median": [None if np.isnan(x) else float(x) for x in _nanmedian_cols(null, rv.shape[0])],
            "per_layer": rows,
        }
    cum_real = real.cumulative_log_gain["median"]
    cum_null = np.array([r.cumulative_log_gain["median"] for r in randoms])
    out["cumulative_log_gain"] = compare_to_null(cum_real, cum_null).__dict__
    return out


def profile_arrays(profile: LayerwiseProfile) -> dict[str, np.ndarray]:
    """Per-example arrays for compact ``.npz`` serialisation."""
    ex = profile.per_example
    return {
        "magnitude": ex.magnitude,
        "gain": ex.gain,
        "log_gain": ex.log_gain,
        "conversion": ex.conversion,
        "alignment": ex.alignment,
        "delta_norm": ex.delta_norm,
        "block_norm": ex.block_norm,
        "d_eff": profile.d_eff,
        "d90": profile.d90,
        "total_centered_variance": profile.total_centered_variance,
        "d_eff_uncentered": profile.d_eff_uncentered,
        "new_subspace": profile.new_subspace,
        "new_subspace_uncentered": profile.new_subspace_uncentered,
    }


def metric_curves(profile: LayerwiseProfile) -> dict[str, list[float | None]]:
    """Compact per-layer curves of every compared metric (for storage and figures)."""
    out: dict[str, list[float | None]] = {}
    for name in COMPARED_METRICS:
        out[name] = [None if np.isnan(x) else float(x) for x in profile.metric_curve(name)]
    out["cumulative_log_gain"] = [profile.cumulative_log_gain["median"]]
    return out


def null_summary(profiles: list[LayerwiseProfile]) -> dict[str, Any]:
    """Median and 5/95 % quantiles per metric and layer over a set of control profiles."""
    out: dict[str, Any] = {"n": len(profiles)}
    if not profiles:
        return out
    for name in COMPARED_METRICS:
        M = np.stack([p.metric_curve(name) for p in profiles])
        cols = M.shape[1]
        med, lo, hi = np.full(cols, np.nan), np.full(cols, np.nan), np.full(cols, np.nan)
        for j in range(cols):
            c = M[:, j]
            c = c[~np.isnan(c)]
            if c.size:
                med[j], lo[j], hi[j] = np.median(c), np.quantile(c, 0.05), np.quantile(c, 0.95)
        out[name] = {k: [None if np.isnan(x) else float(x) for x in a] for k, a in (("median", med), ("q05", lo), ("q95", hi))}
    return out


def compare_by_kind(
    real: LayerwiseProfile, by_kind: dict[str, list[LayerwiseProfile]], gate_kinds: list[str]
) -> dict[str, Any]:
    """Primary comparison against the pooled ``gate_kinds`` null plus one per control kind."""
    pooled = [p for k in gate_kinds for p in by_kind.get(k, [])]
    return {
        "gate_kinds": list(gate_kinds),
        "primary": compare_to_random(real, pooled),
        "by_kind": {k: compare_to_random(real, ps) for k, ps in by_kind.items() if ps},
    }
