"""Cross-task analysis: reduce each layerwise profile to mode-discriminating quantities.

Signatures (docs/PROJECT.md):

* conserved transmission: alignment stays high, cumulative log G ~ 0, conversion ~ 0, d_eff flat
* delayed activation: conversion mass centred on late block(s), low early conversion
* cascaded activation: conversion mass spread across blocks (entropy ratio near 1)
* localized transformation: one block carries a large share of the conversion mass
* amplification: cumulative log G > 0
* dimensional expansion: d_eff / d90 rise with depth
* new-direction creation: N_l high relative to matched random controls

Labels are assigned by fixed, configurable rules *after* the quantities are
computed; thresholds live in ``AnalysisConfig``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import AnalysisConfig
from .layerwise import LayerwiseProfile
from .stats import centre_of_mass, entropy_ratio


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[~np.isnan(a)]


# Per-metric read points excluded from the z-mean because the comparison there is
# degenerate by construction: task_alignment at l* is 1 for the real direction while
# orthogonal controls are exactly 0 and isotropic ones ~1/sqrt(d) with near-zero spread.
_Z_MEAN_EXCLUDE_AT_INTERVENTION = ("task_alignment",)


def _z_means(comparison: dict[str, Any], intervention_layer: int | None = None) -> dict[str, Any]:
    """Mean per-layer z-score of the real direction against one null, per metric."""
    out: dict[str, Any] = {"n": comparison.get("n_random", 0)}
    for metric, key in (("log_gain", "log_gain_z_mean"), ("d_eff", "d_eff_z_mean"),
                        ("new_subspace", "new_subspace_z_mean"), ("new_subspace_uncentered", "new_subspace_uncentered_z_mean"),
                        ("alignment", "alignment_z_mean"), ("conversion", "conversion_z_mean"),
                        ("task_alignment", "task_alignment_z_mean"), ("gradient_alignment", "gradient_alignment_z_mean"),
                        ("gradient_projection", "gradient_projection_z_mean"),
                        ("first_order_increment", "first_order_increment_z_mean")):
        rows = comparison["metrics"][metric]["per_layer"]
        if metric in _Z_MEAN_EXCLUDE_AT_INTERVENTION and intervention_layer is not None:
            rows = [r for l, r in enumerate(rows) if l != intervention_layer]
        z = _finite(np.array([r["z"] for r in rows], dtype=np.float64))
        out[key] = float(np.mean(z)) if len(z) else None
        p = _finite(np.array([r["p_upper"] for r in rows], dtype=np.float64))
        out[key.replace("_z_mean", "_frac_layers_p_upper_le_0.05")] = float(np.mean(p <= 0.05)) if len(p) else None
    cg = comparison.get("cumulative_log_gain", {})
    out["cumulative_log_gain_z"] = cg.get("z")
    out["cumulative_log_gain_p_upper"] = cg.get("p_upper")
    out["cumulative_log_gain_p_lower"] = cg.get("p_lower")
    return out


def profile_signature(
    profile: LayerwiseProfile,
    comparison: dict[str, Any] | None,
    cfg: AnalysisConfig,
    by_kind: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mode-discriminating scalar quantities for one task/control pair.

    ``comparison`` is the primary (gate-kind) random-control comparison;
    ``by_kind`` holds one comparison per control kind (structured nulls).
    """
    ls, L = profile.intervention_layer, profile.n_layers
    blocks = np.arange(ls, L)
    # conversion mass per block: median ||b_l(x)|| over examples, normalised over downstream blocks
    block_norm_med = np.nanmedian(profile.per_example.block_norm[ls:], axis=1)
    mass = block_norm_med / block_norm_med.sum() if block_norm_med.sum() > 0 else np.full_like(block_norm_med, np.nan)
    depth = (blocks - ls) / max(1, (L - 1 - ls))  # 0 = intervention block, 1 = last block
    align = profile.metric_curve("alignment")
    conv = profile.metric_curve("conversion")
    task_al = profile.metric_curve("task_alignment")
    grad_al = profile.metric_curve("gradient_alignment")
    first_order = profile.metric_curve("gradient_projection")  # median delta_l . g_l per read point (D25)
    increment = profile.metric_curve("first_order_increment")  # its change per block

    def downstream_mean(curve: np.ndarray) -> float | None:
        vals = _finite(curve[ls + 1 :])
        return float(np.mean(vals)) if len(vals) else None

    d_eff = profile.d_eff
    d_eff_def = _finite(d_eff)
    d90_def = _finite(profile.d90)
    n_half = max(1, len(mass) // 2)

    sig: dict[str, Any] = {
        "intervention_layer": ls,
        "n_downstream_blocks": int(L - ls),
        "alignment_final": float(align[L]) if not np.isnan(align[L]) else None,
        "alignment_min": float(np.nanmin(align[ls:])),
        "alignment_mean": float(np.nanmean(align[ls:])),
        "cumulative_log_gain": profile.cumulative_log_gain["median"],
        "cumulative_log_gain_ci": [profile.cumulative_log_gain["low"], profile.cumulative_log_gain["high"]],
        "mean_conversion": float(np.nanmean(conv[ls:])),
        "max_conversion": float(np.nanmax(conv[ls:])),
        "argmax_conversion_block": int(ls + np.nanargmax(conv[ls:])),
        "conversion_mass": [float(m) for m in mass],
        "conversion_mass_blocks": [int(b) for b in blocks],
        "conversion_entropy_ratio": entropy_ratio(mass),
        "conversion_dominant_block": int(blocks[int(np.nanargmax(mass))]) if len(mass) else None,
        "conversion_dominant_share": float(np.nanmax(mass)) if len(mass) else None,
        "conversion_centre_of_mass": centre_of_mass(mass, depth),
        "early_conversion_share": float(np.nansum(mass[:n_half])),
        "d_eff_first": float(d_eff_def[0]) if len(d_eff_def) else None,
        "d_eff_final": float(d_eff_def[-1]) if len(d_eff_def) else None,
        "d_eff_ratio_final_over_first": float(d_eff_def[-1] / d_eff_def[0]) if len(d_eff_def) > 1 and d_eff_def[0] > 0 else None,
        "d_eff_trend": _trend(d_eff_def),
        "d90_first": float(d90_def[0]) if len(d90_def) else None,
        "d90_final": float(d90_def[-1]) if len(d90_def) else None,
        "new_subspace_mean": float(np.nanmean(profile.new_subspace)) if np.any(~np.isnan(profile.new_subspace)) else None,
        "new_subspace_uncentered_mean": float(np.nanmean(profile.new_subspace_uncentered)),
        "new_subspace_uncentered_max_block": int(np.nanargmax(profile.new_subspace_uncentered)),
        "noise_floor_variance_ratio": profile.noise_floor["variance_ratio_intervention_over_next"],
        # direction-specific readouts (D17): median curves summarised downstream of the intervention
        "task_alignment_downstream_mean": downstream_mean(task_al),
        "task_alignment_final": float(task_al[L]) if not np.isnan(task_al[L]) else None,
        "gradient_alignment_at_intervention": float(grad_al[ls]) if not np.isnan(grad_al[ls]) else None,
        "gradient_alignment_downstream_mean": downstream_mean(grad_al),
        "gradient_alignment_final": float(grad_al[L]) if not np.isnan(grad_al[L]) else None,
        # the functional depth profile (D25): where the first-order predicted effect is created
        "first_order_at_intervention": float(first_order[ls]) if not np.isnan(first_order[ls]) else None,
        "first_order_final": float(first_order[L]) if not np.isnan(first_order[L]) else None,
        "first_order_increment_argmax_block": (int(ls + np.nanargmax(increment[ls:]))
                                              if np.any(~np.isnan(increment[ls:])) else None),
        "first_order_increment_centre_of_mass": _positive_centre_of_mass(increment[ls:], depth),
    }
    if comparison is not None:
        zm = _z_means(comparison, ls)
        sig.update({k: v for k, v in zm.items() if k != "n"})
        sig["n_primary_controls"] = zm["n"]
        sig["cumulative_log_gain_vs_random"] = comparison["cumulative_log_gain"]
    sig["by_kind"] = {k: _z_means(c, ls) for k, c in (by_kind or {}).items()}
    sig["labels"] = assign_labels(sig, cfg)
    return sig


def _positive_centre_of_mass(values: np.ndarray, depth: np.ndarray) -> float | None:
    """Centre of mass (in normalised depth) of the positive part of ``values``; None without positive mass."""
    pos = np.where(np.isnan(values), 0.0, np.clip(values, 0.0, None))
    total = float(pos.sum())
    if total <= 0.0:
        return None
    return float((pos * depth).sum() / total)


def _trend(values: np.ndarray) -> float | None:
    """Slope of a least-squares line through ``values`` vs index (None if < 2 points)."""
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=np.float64)
    x = x - x.mean()
    return float((x * (values - values.mean())).sum() / (x**2).sum())


def assign_labels(sig: dict[str, Any], cfg: AnalysisConfig) -> list[str]:
    """Rule-based qualitative labels; several may apply, or none."""
    labels: list[str] = []
    af = sig.get("alignment_final")
    cum = sig.get("cumulative_log_gain")
    if (
        af is not None
        and cum is not None
        and af >= cfg.conserved_alignment_min
        and abs(cum) <= cfg.conserved_abs_cumlog_gain_max
    ):
        labels.append("conserved_transmission")
    share = sig.get("conversion_dominant_share")
    ent = sig.get("conversion_entropy_ratio")
    com = sig.get("conversion_centre_of_mass")
    if share is not None and share >= cfg.dominant_block_share_min:
        labels.append("localized_transformation")
        if com is not None and com >= cfg.delayed_centre_of_mass_min:
            labels.append("delayed_activation")
    elif ent is not None and ent >= cfg.cascade_entropy_ratio_min:
        labels.append("cascade")
    elif com is not None and com >= cfg.delayed_centre_of_mass_min:
        labels.append("delayed_activation")
    if cum is not None:
        ci = sig.get("cumulative_log_gain_ci") or [None, None]
        if ci[0] is not None and ci[0] > 0:
            labels.append("amplification")
        elif ci[1] is not None and ci[1] < 0:
            labels.append("attenuation")
    ratio = sig.get("d_eff_ratio_final_over_first")
    if ratio is not None and ratio >= cfg.expansion_ratio_min:
        labels.append("dimensional_expansion")
        com_d = sig.get("d_eff_trend")
        if com_d is not None and com_d > 0 and sig.get("early_conversion_share", 0) > 0.5:
            labels.append("early_expansion")
    z = sig.get("new_subspace_uncentered_z_mean")
    if z is not None and z >= cfg.new_direction_z_min:
        labels.append("new_direction_creation")
    return labels


def cross_task_table(signatures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compact cross-task table of the discriminating quantities."""
    keys = [
        "intervention_layer",
        "alignment_final",
        "cumulative_log_gain",
        "mean_conversion",
        "conversion_entropy_ratio",
        "conversion_dominant_block",
        "conversion_dominant_share",
        "conversion_centre_of_mass",
        "d_eff_first",
        "d_eff_final",
        "d_eff_ratio_final_over_first",
        "new_subspace_uncentered_mean",
        "new_subspace_uncentered_z_mean",
        "new_subspace_z_mean",
        "log_gain_z_mean",
        "d_eff_z_mean",
        "alignment_z_mean",
        "cumulative_log_gain_z",
        "task_alignment_downstream_mean",
        "task_alignment_z_mean",
        "gradient_alignment_at_intervention",
        "gradient_alignment_downstream_mean",
        "gradient_alignment_z_mean",
        "first_order_at_intervention",
        "first_order_final",
        "first_order_increment_argmax_block",
        "first_order_increment_centre_of_mass",
        "gradient_projection_z_mean",
        "first_order_increment_z_mean",
        "n_primary_controls",
        "noise_floor_variance_ratio",
        "labels",
    ]
    table: dict[str, Any] = {}
    for task, sig in signatures.items():
        row = {k: sig.get(k) for k in keys}
        row["by_kind"] = {
            kind: {k: d.get(k) for k in ("n", "log_gain_z_mean", "d_eff_z_mean", "new_subspace_uncentered_z_mean",
                                          "alignment_z_mean", "cumulative_log_gain_z",
                                          "task_alignment_z_mean", "gradient_alignment_z_mean",
                                          "gradient_projection_z_mean", "first_order_increment_z_mean")}
            for kind, d in sig.get("by_kind", {}).items()
        }
        table[task] = row
    return table
