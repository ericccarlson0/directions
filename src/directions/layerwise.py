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
    direction_readouts,
    effective_rank,
    layerwise_example_metrics,
    linalg_device,
    new_subspace_fraction,
    spectra,
)
from .stats import bootstrap_median_ci, bootstrap_median_ci_rows, compare_to_null

READOUT_METRICS = ("task_alignment", "gradient_alignment")  # direction-specific (docs/DECISIONS.md D17)
PER_EXAMPLE_METRICS = ("magnitude", "log_gain", "conversion", "alignment") + READOUT_METRICS
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
    readouts: dict[str, np.ndarray] = None  # type: ignore[assignment]  # per-example (L+1, n) arrays, nan if inputs absent
    readout_diagnostics: dict[str, Any] = None  # type: ignore[assignment]

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
            "readout_diagnostics": self.readout_diagnostics,
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


def _profile_arrays_numpy(
    base: np.ndarray, steered: np.ndarray, v: np.ndarray, ls: int, frac: float,
    task_directions: np.ndarray | None, gradients: np.ndarray | None,
) -> dict[str, Any]:
    """The per-example and per-layer arrays of a profile, computed with the NumPy geometry (the reference)."""
    base = np.asarray(base, dtype=np.float64)
    steered = np.asarray(steered, dtype=np.float64)
    L1 = base.shape[0]
    L = L1 - 1
    delta = steered - base
    ex = layerwise_example_metrics(delta, base, v)
    readouts = direction_readouts(delta, task_directions, gradients)
    out: dict[str, Any] = {
        "pre_max": float(np.abs(delta[:ls]).max()) if ls > 0 else 0.0,
        "example": ex,
        "readouts": readouts,
        "gradient_norm_median": None,
        "injected_direction_gradient_cosine_median": None,
    }
    if gradients is not None:
        gnorm = np.linalg.norm(np.asarray(gradients, dtype=np.float64), axis=2)  # (L+1, n)
        out["gradient_norm_median"] = [float(x) for x in np.median(gnorm, axis=1)]
        # cos(v, g_l*): does the *injected* direction itself point along the target gradient?
        vv = v / np.linalg.norm(v)
        g_ls = np.asarray(gradients[ls], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = (g_ls @ vv) / np.linalg.norm(g_ls, axis=1)
        out["injected_direction_gradient_cosine_median"] = float(np.nanmedian(c))
    spec_c = spectra(delta[ls:], frac=frac, center=True)
    spec_u = spectra(delta[ls:], frac=frac, center=False)
    out["ranks_c"] = [(s.effective_rank, s.d90, s.total_variance) for s in spec_c]
    out["ranks_u"] = [(s.effective_rank, s.d90, s.total_variance) for s in spec_u]
    b = block_responses(delta)
    N = np.full(L, np.nan)
    N_unc = np.full(L, np.nan)
    for l in range(ls, L):
        if l > ls:
            N[l] = new_subspace_fraction(b[l], spec_c[l - ls].top_subspace)
        N_unc[l] = new_subspace_fraction(b[l], spec_u[l - ls].top_subspace)
    out["new_subspace"], out["new_subspace_uncentered"] = N, N_unc
    return out


def _median_like_numpy(t: Any, dim: int) -> Any:
    """``np.median`` along ``dim`` for a torch tensor (the mean of the two middle values when even)."""
    s, _ = t.sort(dim=dim)
    n = s.shape[dim]
    if n % 2:
        return s.narrow(dim, n // 2, 1).squeeze(dim)
    return 0.5 * (s.narrow(dim, n // 2 - 1, 1) + s.narrow(dim, n // 2, 1)).squeeze(dim)


def _profile_arrays_torch(
    base: np.ndarray, steered: np.ndarray, v: np.ndarray, ls: int, frac: float,
    task_directions: np.ndarray | None, gradients: np.ndarray | None, device: Any,
) -> dict[str, Any]:
    """:func:`_profile_arrays_numpy` in float64 torch on ``device`` (the GPU path, docs/DECISIONS.md D22):
    the residuals go to the device once, every array comes back once. ``tests/test_geometry.py`` checks
    it against the NumPy path."""
    import torch

    from .geometry import torch_spectra

    f64 = torch.float64
    B = torch.as_tensor(np.asarray(base), dtype=f64, device=device)
    D = torch.as_tensor(np.asarray(steered), dtype=f64, device=device) - B
    L1, n, _ = D.shape
    L = L1 - 1
    nan = torch.full((), float("nan"), dtype=f64, device=device)
    dn = D.norm(dim=2)
    bn = B.norm(dim=2)
    b = D[1:] - D[:-1]
    b_norm = b.norm(dim=2)
    vv = torch.as_tensor(np.asarray(v, dtype=np.float64), device=device)
    vv = vv / vv.norm()
    magnitude = torch.where(bn > 0, dn / bn, nan)
    gain = torch.where(dn[:-1] > 0, dn[1:] / dn[:-1], nan)
    conversion = torch.where(dn[:-1] > 0, b_norm / dn[:-1], nan)
    alignment = torch.where(dn > 0, (D @ vv).abs() / dn, nan)
    readouts = {k: torch.full((L1, n), float("nan"), dtype=f64, device=device)
                for k in ("task_alignment", "gradient_alignment", "gradient_projection")}
    out: dict[str, Any] = {"gradient_norm_median": None, "injected_direction_gradient_cosine_median": None}
    if task_directions is not None:
        V = torch.as_tensor(np.asarray(task_directions, dtype=np.float64), device=device)
        V = V / V.norm(dim=1, keepdim=True)
        readouts["task_alignment"] = torch.where(dn > 0, torch.einsum("lnd,ld->ln", D, V) / dn, nan)
    if gradients is not None:
        G = torch.as_tensor(np.asarray(gradients), dtype=f64, device=device)
        gn = G.norm(dim=2)
        dot = (D * G).sum(dim=2)
        readouts["gradient_projection"] = dot
        readouts["gradient_alignment"] = torch.where((dn > 0) & (gn > 0), dot / (dn * gn), nan)
        out["gradient_norm_median"] = [float(x) for x in _median_like_numpy(gn, 1).cpu().numpy()]
        c = torch.where(gn[ls] > 0, (G[ls] @ vv) / gn[ls], nan)
        c = c[~torch.isnan(c)]
        out["injected_direction_gradient_cosine_median"] = float(_median_like_numpy(c, 0).cpu()) if c.numel() else float("nan")
    ranks_c, tops_c = torch_spectra(D[ls:], frac, True)
    ranks_u, tops_u = torch_spectra(D[ls:], frac, False)
    N = torch.full((L,), float("nan"), dtype=f64, device=device)
    N_unc = torch.full((L,), float("nan"), dtype=f64, device=device)

    def fraction(Bl: Any, top: Any) -> Any:
        total = (Bl**2).sum()
        if top.shape[0] == 0:
            return torch.where(total > 0, torch.ones_like(total), nan)
        resid = Bl - (Bl @ top.transpose(0, 1)) @ top
        return torch.where(total > 0, (resid**2).sum() / total, nan)

    for l in range(ls, L):
        if l > ls:
            N[l] = fraction(b[l], tops_c[l - ls])
        N_unc[l] = fraction(b[l], tops_u[l - ls])
    # everything to the host in one transfer
    stack = torch.cat([t.reshape(-1) for t in (dn, bn, magnitude, alignment, gain, conversion, b_norm,
                                                readouts["task_alignment"], readouts["gradient_alignment"],
                                                readouts["gradient_projection"], N, N_unc)]).cpu().numpy()
    sizes = [L1 * n] * 4 + [L * n] * 3 + [L1 * n] * 3 + [L, L]
    parts = np.split(stack, np.cumsum(sizes)[:-1])
    dn_np, bn_np, mag, ali, gain_np, conv, bnorm, ta, ga, gp, N_np, Nu_np = parts
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gain = np.log(gain_np.reshape(L, n))
    out.update({
        "pre_max": float(D[:ls].abs().max().cpu()) if ls > 0 else 0.0,
        "example": LayerwiseExampleMetrics(
            magnitude=mag.reshape(L1, n), gain=gain_np.reshape(L, n), log_gain=log_gain, conversion=conv.reshape(L, n),
            alignment=ali.reshape(L1, n), delta_norm=dn_np.reshape(L1, n), block_norm=bnorm.reshape(L, n)),
        "readouts": {"task_alignment": ta.reshape(L1, n), "gradient_alignment": ga.reshape(L1, n),
                     "gradient_projection": gp.reshape(L1, n)},
        "ranks_c": ranks_c,
        "ranks_u": ranks_u,
        "new_subspace": N_np,
        "new_subspace_uncentered": Nu_np,
    })
    return out


def compute_profile(
    base: np.ndarray,
    steered: np.ndarray,
    v: np.ndarray,
    intervention_layer: int,
    cfg: EvaluationConfig,
    rng: np.random.Generator,
    with_ci: bool = True,
    task_directions: np.ndarray | None = None,
    gradients: np.ndarray | None = None,
) -> LayerwiseProfile:
    """Full layerwise profile from residuals shaped (L+1, n, d).

    ``with_ci=False`` skips the bootstrap confidence intervals (medians only).
    ``task_directions`` (L+1, d) and ``gradients`` (L+1, n, d) feed the
    direction-specific readouts (:func:`direction_readouts`); when absent those
    metrics are all ``nan``. The arrays are computed on the linalg device when
    one is set (:func:`directions.geometry.set_linalg_device`), else in NumPy.
    """
    L1, n, _ = np.shape(base)
    L = L1 - 1
    ls = intervention_layer
    device = linalg_device()
    if device is None:
        arrays = _profile_arrays_numpy(base, steered, v, ls, cfg.variance_fraction, task_directions, gradients)
    else:
        arrays = _profile_arrays_torch(base, steered, v, ls, cfg.variance_fraction, task_directions, gradients, device)
    ex: LayerwiseExampleMetrics = arrays["example"]
    readouts: dict[str, np.ndarray] = arrays["readouts"]
    # Quantities are only defined from the intervention layer on.
    for arr in (ex.magnitude, ex.alignment, *readouts.values()):
        arr[:ls] = np.nan
    for arr in (ex.gain, ex.log_gain, ex.conversion):
        arr[:ls] = np.nan

    def per_example(name: str) -> np.ndarray:
        return readouts[name] if name in readouts else getattr(ex, name)

    if with_ci:
        summaries = {
            name: bootstrap_median_ci_rows(per_example(name), rng, n_boot=cfg.n_boot, alpha=cfg.ci_alpha)
            for name in PER_EXAMPLE_METRICS
        }
    else:
        summaries = {name: _median_rows(per_example(name)) for name in PER_EXAMPLE_METRICS}
    readout_diagnostics: dict[str, Any] = {
        "task_directions_available": task_directions is not None,
        "gradients_available": gradients is not None,
    }
    if gradients is not None:
        readout_diagnostics["gradient_norm_median"] = arrays["gradient_norm_median"]
        readout_diagnostics["injected_direction_gradient_cosine_median"] = arrays["injected_direction_gradient_cosine_median"]
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
    observed_at_ls: dict[str, float | None] = {"d_eff": None, "total_centered_variance": None, "d90": None}
    for l in range(ls, L1):
        eff_c, k_c, tot_c = arrays["ranks_c"][l - ls]
        d_eff_unc[l] = arrays["ranks_u"][l - ls][0]
        if l == ls:
            # exact arithmetic: centered D is zero; the observed value is the bf16 noise floor
            observed_at_ls = {
                "d_eff": None if np.isnan(eff_c) else float(eff_c),
                "d90": None if k_c is None else float(k_c),
                "total_centered_variance": float(tot_c),
            }
            tot[l] = tot_c
        else:
            d_eff[l] = eff_c
            d90[l] = np.nan if k_c is None else k_c
            tot[l] = tot_c
    N = np.asarray(arrays["new_subspace"], dtype=np.float64)
    N_unc = np.asarray(arrays["new_subspace_uncentered"], dtype=np.float64)

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
        pre_intervention_max_abs_delta=arrays["pre_max"],
        readouts=readouts,
        readout_diagnostics=readout_diagnostics,
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
        **{k: v for k, v in (profile.readouts or {}).items()},
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
