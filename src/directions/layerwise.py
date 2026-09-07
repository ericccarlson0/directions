"""Layerwise perturbation measurement.

Given baseline and steered residual streams at every layer for a set of
held-out inputs, computes the primary metrics of ``docs/EXPERIMENT.md``:

============ =====================================================
``S_l(x)``   perturbation magnitude ``||d_l|| / ||h_l^base||``
``G_l(x)``   layerwise gain ``||d_{l+1}|| / ||d_l||``
``C_l(x)``   block conversion ``||b_l|| / ||d_l||``
``d_eff``    participation-ratio effective rank of the stacked ``D_l``
``d90``      PCs explaining 90% of the centered variance of ``D_l``
``N_l``      fraction of ``B_l`` outside the inherited subspace ``P_l``
============ =====================================================

Layers strictly upstream of the intervention have ``d_l = 0`` by construction
(a causal transformer cannot propagate backwards), so all ratios there are
reported as NaN rather than as spurious zeros.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import mathx
from .config import Config


@dataclass
class LayerwiseMeasurement:
    """All primary layerwise quantities for one (task, direction) condition."""

    label: str
    layer: int  # intervention layer
    alpha: float
    n_layers: int
    n_examples: int
    # per-example arrays, indexed [layer, example]; NaN upstream of the intervention
    S: np.ndarray
    G: np.ndarray
    C: np.ndarray
    # per-layer scalars
    d_eff: np.ndarray
    d90: np.ndarray
    d_eff_uncentered: np.ndarray
    N: np.ndarray
    N_uncentered: np.ndarray
    control_alignment: np.ndarray  # exploratory: median |cos(delta_l, v)|
    centered_variance: np.ndarray  # total centered variance of D_l, per layer
    numerical_noise: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)

    @property
    def layers(self) -> np.ndarray:
        return np.arange(self.n_layers + 1)

    def to_dict(self) -> dict:
        f = lambda a: [None if not np.isfinite(x) else float(x) for x in np.asarray(a, float)]  # noqa: E731
        with quiet_nan_reductions():
            return {
                "label": self.label,
                "intervention_layer": self.layer,
                "alpha": self.alpha,
                "n_layers": self.n_layers,
                "n_examples": self.n_examples,
                "residual_index": list(range(self.n_layers + 1)),
                "block_index": list(range(self.n_layers)),
                "S_median": f(np.nanmedian(self.S, axis=1)),
                "log_G_median": f(np.nanmedian(np.log(self.G), axis=1)),
                "C_median": f(np.nanmedian(self.C, axis=1)),
                "d_eff": f(self.d_eff),
                "d90": f(self.d90),
                "d_eff_uncentered": f(self.d_eff_uncentered),
                "N": f(self.N),
                "N_uncentered": f(self.N_uncentered),
                "control_alignment": f(self.control_alignment),
                "centered_variance": f(self.centered_variance),
                "numerical_noise": self.numerical_noise,
                "summaries": self.summaries,
            }


@contextlib.contextmanager
def quiet_nan_reductions():
    """Upstream-of-intervention layers are NaN by construction (see D5), so
    all-NaN slices in the aggregation reductions are expected, not a problem."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        with np.errstate(divide="ignore", invalid="ignore"):
            yield


def _nan(shape) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float64)


def measure(
    cfg: Config,
    label: str,
    intervention_layer: int,
    alpha: float,
    direction: np.ndarray,
    baseline: np.ndarray,
    steered: np.ndarray,
) -> LayerwiseMeasurement:
    """Compute all layerwise metrics from captured residual streams.

    ``baseline`` and ``steered`` both have shape ``(n_layers + 1, n_examples,
    d_model)``.
    """
    baseline = mathx.as_f64(baseline)
    steered = mathx.as_f64(steered)
    if baseline.shape != steered.shape:
        raise ValueError(f"shape mismatch {baseline.shape} vs {steered.shape}")
    n_resid, n_ex, _ = baseline.shape
    n_layers = n_resid - 1
    frac = cfg.layerwise.variance_fraction

    deltas = steered - baseline  # (L+1, N, d)
    blocks = mathx.blockwise_response(deltas)  # (L, N, d)
    v = mathx.normalize(direction)

    S, G, C = _nan((n_resid, n_ex)), _nan((n_resid, n_ex)), _nan((n_resid, n_ex))
    d_eff, d90 = _nan(n_resid), _nan(n_resid)
    d_eff_unc = _nan(n_resid)
    N, N_unc = _nan(n_resid), _nan(n_resid)
    align = _nan(n_resid)
    variance = _nan(n_resid)

    for l in range(intervention_layer, n_resid):
        D = deltas[l]
        S[l] = mathx.perturbation_magnitude(D, baseline[l])
        norms = mathx.row_norms(D)
        align[l] = float(np.median(np.abs(D @ v) / np.maximum(norms, mathx.EPS)))
        sm = mathx.spectrum_metrics(D, frac=frac, center=True)
        d_eff[l], d90[l] = sm.effective_rank, sm.d90
        variance[l] = sm.total_variance
        d_eff_unc[l] = mathx.effective_rank(mathx.singular_values(D, center=False))
        if l < n_layers:
            B = blocks[l]
            G[l] = mathx.layer_gain(D, deltas[l + 1])
            C[l] = mathx.block_conversion(D, B)
            # Spec-faithful N_l uses the centered D_l. At l == intervention_layer
            # every delta is identical (= alpha*v), so the centered D_l vanishes
            # and N_l is undefined; see docs/DECISIONS.md.
            if sm.total_variance > mathx.EPS:
                N[l] = mathx.new_subspace_fraction(D, B, frac=frac)
            N_unc[l] = _new_subspace_uncentered(D, B, frac)

    # At the intervention layer every delta_l(x) equals alpha*v by construction,
    # so the centered D_l is zero and its effective dimensionality is undefined.
    # In bf16 execution the observed value is the arithmetic rounding of
    # h + alpha*v, which is isotropic and therefore looks high-dimensional; it is
    # recorded as an explicit noise-floor diagnostic instead of a measurement.
    # See docs/DECISIONS.md (D15).
    noise = {
        "intervention_layer_d_eff_observed": _f(d_eff[intervention_layer]),
        "intervention_layer_centered_variance": _f(variance[intervention_layer]),
        "next_layer_centered_variance": _f(variance[intervention_layer + 1])
        if intervention_layer + 1 < n_resid
        else None,
        "note": (
            "delta is identical across examples at the intervention layer; any "
            "centered variance there is numerical rounding, and bounds the noise "
            "floor of the downstream dimensionality metrics"
        ),
    }
    if (
        noise["next_layer_centered_variance"]
        and noise["intervention_layer_centered_variance"] is not None
    ):
        noise["noise_to_signal_ratio"] = (
            noise["intervention_layer_centered_variance"] / noise["next_layer_centered_variance"]
        )
    d_eff[intervention_layer] = np.nan
    d90[intervention_layer] = np.nan

    m = LayerwiseMeasurement(
        label=label,
        layer=intervention_layer,
        alpha=alpha,
        n_layers=n_layers,
        n_examples=n_ex,
        S=S,
        G=G,
        C=C,
        d_eff=d_eff,
        d90=d90,
        d_eff_uncentered=d_eff_unc,
        N=N,
        N_uncentered=N_unc,
        control_alignment=align,
        centered_variance=variance,
        numerical_noise=noise,
    )
    m.summaries = _summaries(cfg, m)
    return m


def _f(x) -> float | None:
    return None if not np.isfinite(x) else float(x)


def _new_subspace_uncentered(D: np.ndarray, B: np.ndarray, frac: float) -> float:
    """``N_l`` with the inherited subspace taken from the *uncentered* ``D_l``.

    Reported alongside the spec-faithful centered version because centering
    removes the shared (mean) perturbation component -- which is exactly the
    control direction whose propagation we are trying to tell apart from newly
    created directions.
    """
    denom = float(np.sum(B**2))
    if denom <= mathx.EPS:
        return 0.0
    basis = mathx.top_subspace_basis(D, frac=frac, center=False)
    if basis.size == 0:
        return 1.0
    residual = B - (B @ basis.T) @ basis
    return float(np.sum(residual**2) / denom)


def _summaries(cfg: Config, m: LayerwiseMeasurement) -> dict:
    boot = cfg.bootstrap
    out: dict[str, dict] = {"S": {}, "log_G": {}, "C": {}}
    for l in range(m.layer, m.n_layers + 1):
        out["S"][str(l)] = mathx.summarize(
            m.S[l], n_boot=boot.n_boot, seed=cfg.run.seed + l, alpha=boot.alpha
        )
        if l < m.n_layers:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["log_G"][str(l)] = mathx.summarize(
                    np.log(m.G[l]), n_boot=boot.n_boot, seed=cfg.run.seed + l, alpha=boot.alpha
                )
            out["C"][str(l)] = mathx.summarize(
                m.C[l], n_boot=boot.n_boot, seed=cfg.run.seed + l, alpha=boot.alpha
            )
    return out


def aggregate_controls(measurements: list[LayerwiseMeasurement]) -> dict:
    """Null distribution over matched random controls, per layer and metric."""
    if not measurements:
        return {}
    n_resid = measurements[0].n_layers + 1
    out: dict[str, dict] = {}
    getters = {
        "S": lambda m: np.nanmedian(m.S, axis=1),
        "log_G": lambda m: np.nanmedian(np.log(m.G), axis=1),
        "C": lambda m: np.nanmedian(m.C, axis=1),
        "d_eff": lambda m: m.d_eff,
        "d90": lambda m: m.d90,
        "N": lambda m: m.N,
        "N_uncentered": lambda m: m.N_uncentered,
    }
    for name, get in getters.items():
        with quiet_nan_reductions():
            stack = np.stack([get(m) for m in measurements])  # (n_controls, n_resid)
            out[name] = {
                "n_controls": len(measurements),
                "mean": _l(np.nanmean(stack, axis=0), n_resid),
                "std": _l(np.nanstd(stack, axis=0), n_resid),
                "q05": _l(np.nanquantile(stack, 0.05, axis=0), n_resid),
                "q50": _l(np.nanquantile(stack, 0.50, axis=0), n_resid),
                "q95": _l(np.nanquantile(stack, 0.95, axis=0), n_resid),
                "samples": [_l(row, n_resid) for row in stack],
            }
    return out


def _l(arr, n) -> list:
    a = np.asarray(arr, dtype=float).ravel()
    return [None if not np.isfinite(x) else float(x) for x in a[:n]]
