"""Automatic qualitative labelling of amplification profiles (exploratory).

These labels are a deterministic post-hoc summary of the quantitative
layerwise metrics; they are *not* preregistered measurements and must never be
used in place of the numbers they summarize. Every threshold is reported
alongside the label so the classification can be audited or recomputed.
"""

from __future__ import annotations

import numpy as np

from .layerwise import LayerwiseMeasurement

LABELS = (
    "conserved_transmission",
    "delayed_activation",
    "cascade",
    "early_expansion",
    "localized_transformation",
)

THRESHOLDS = {
    "conserved_log_gain_abs_max": float(np.log(1.5)),
    "conserved_conversion_max": 0.30,
    "conserved_deff_max": 2.0,
    "localized_block_share": 0.40,
    "cascade_entropy_ratio": 0.70,
    "delayed_centroid_depth": 0.60,
    "early_expansion_depth": 0.25,
    "early_expansion_deff_ratio": 0.50,
}


def _downstream(m: LayerwiseMeasurement):
    """Indices of the residual points and blocks downstream of the intervention."""
    resid = np.arange(m.layer, m.n_layers + 1)
    blocks = np.arange(m.layer, m.n_layers)
    return resid, blocks


def _clean(d: dict) -> dict:
    """NaN -> None, so features are JSON-safe and comparable across calls."""
    return {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()
    }


def profile_features(m: LayerwiseMeasurement) -> dict:
    resid, blocks = _downstream(m)
    if blocks.size == 0:
        return {"insufficient_depth": True}

    logG = np.nanmedian(np.log(m.G[blocks]), axis=1)
    C = np.nanmedian(m.C[blocks], axis=1)
    S = np.nanmedian(m.S[resid], axis=1)
    deff = m.d_eff[resid]
    N = m.N_uncentered[blocks]

    # relative block response mass: ||b_l|| relative to the incoming ||delta_l||
    mass = np.nan_to_num(C, nan=0.0)
    total = float(mass.sum())
    share = mass / total if total > 0 else np.zeros_like(mass)
    u = (blocks - m.layer) / max(1, (m.n_layers - 1 - m.layer))  # 0..1 over blocks

    with np.errstate(divide="ignore", invalid="ignore"):
        nz = share[share > 0]
        entropy = float(-np.sum(nz * np.log(nz))) if nz.size else 0.0
    max_entropy = float(np.log(len(share))) if len(share) > 1 else 1.0

    deff_finite = deff[np.isfinite(deff)]
    deff_max = float(np.nanmax(deff)) if deff_finite.size else float("nan")
    depth_at_half_deff = float("nan")
    if deff_finite.size and deff_max > 0:
        u_resid = (resid - m.layer) / max(1, (m.n_layers - m.layer))
        hit = np.where(np.nan_to_num(deff) >= 0.5 * deff_max)[0]
        if hit.size:
            depth_at_half_deff = float(u_resid[hit[0]])

    return _clean({
        "insufficient_depth": False,
        "cumulative_log_gain": float(np.nansum(logG)),
        "max_abs_log_gain": float(np.nanmax(np.abs(logG))),
        "max_conversion": float(np.nanmax(C)),
        "mean_conversion": float(np.nanmean(C)),
        "max_block_share": float(np.nanmax(share)) if share.size else float("nan"),
        "argmax_block": int(blocks[int(np.argmax(share))]) if share.size else -1,
        "conversion_centroid_depth": float(np.sum(share * u)) if share.size else float("nan"),
        "conversion_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0.0,
        "final_S": float(S[-1]) if S.size else float("nan"),
        "max_d_eff": deff_max,
        "final_d_eff": float(deff[-1]) if deff.size else float("nan"),
        "depth_at_half_max_d_eff": depth_at_half_deff,
        "mean_new_subspace": float(np.nanmean(N)) if N.size else float("nan"),
    })


def classify(m: LayerwiseMeasurement) -> dict:
    """Score each qualitative label; return scores, the argmax and the features."""
    f = profile_features(m)
    if f.get("insufficient_depth"):
        return {"label": None, "reason": "no downstream blocks", "features": f, "scores": {}}
    t = THRESHOLDS
    scores = {
        "conserved_transmission": float(
            (f["max_abs_log_gain"] <= t["conserved_log_gain_abs_max"])
            + (f["max_conversion"] <= t["conserved_conversion_max"])
            + ((f["max_d_eff"] if f["max_d_eff"] is not None else 1e9) <= t["conserved_deff_max"])
        )
        / 3.0,
        "localized_transformation": float(
            (f["max_block_share"] or 0.0) >= t["localized_block_share"]
        ),
        "cascade": float(
            (f["conversion_entropy_ratio"] >= t["cascade_entropy_ratio"])
            and ((f["max_block_share"] or 0.0) < t["localized_block_share"])
        ),
        "delayed_activation": float(
            (f["conversion_centroid_depth"] or 0.0) >= t["delayed_centroid_depth"]
        ),
        "early_expansion": float(
            (f["depth_at_half_max_d_eff"] or 1.0) <= t["early_expansion_depth"]
            and (f["max_d_eff"] or 0.0) > t["conserved_deff_max"]
        ),
    }
    # Deterministic tie-break: LABELS order.
    best = max(LABELS, key=lambda k: (scores[k], -LABELS.index(k)))
    return {
        "label": best if scores[best] > 0 else None,
        "scores": scores,
        "features": f,
        "thresholds": t,
        "note": "exploratory: deterministic rule-based summary of the primary metrics",
    }
