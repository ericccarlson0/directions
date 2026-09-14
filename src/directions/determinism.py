"""In-run determinism check (docs/DECISIONS.md D23).

Once per run, the pipeline repeats the steered forward pass, the gradient pass
and one layerwise profile of the first task that reaches held-out steering and
requires the repeats to be bit-identical to the originals. This is what a full
replicate of the run used to establish, at the cost of one forward pass, one
backward pass and one profile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .layerwise import LayerwiseProfile, profile_arrays
from .model import ForwardResult


def forward_arrays(r: ForwardResult) -> dict[str, np.ndarray | None]:
    return {
        "logprob_sum": r.logprob_sum,
        "logprob_per_token": r.logprob_per_token,
        "exact_match": r.exact_match,
        "first_token_margin": r.first_token_margin,
        "first_token_logprob": r.first_token_logprob,
        "residuals": r.residuals,
        "kl_from_reference": r.kl_from_reference,
        "argmax_changed": r.argmax_changed,
    }


def compare_arrays(a: dict[str, np.ndarray | None], b: dict[str, np.ndarray | None]) -> dict[str, Any]:
    """Per key: whether the two arrays are bit-identical (``nan`` equal to ``nan``) and their largest
    absolute difference; plus the overall verdict."""
    keys: dict[str, Any] = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            keys[k] = {"identical": x is None and y is None, "max_abs_diff": None}
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            keys[k] = {"identical": False, "max_abs_diff": None, "shapes": [list(x.shape), list(y.shape)]}
            continue
        same = bool(np.array_equal(x, y, equal_nan=True)) if x.dtype.kind in "fc" else bool(np.array_equal(x, y))
        diff = None
        if x.dtype.kind in "fc" and x.size:
            with np.errstate(invalid="ignore"):
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            diff = float(np.nanmax(d)) if np.any(~np.isnan(d)) else 0.0
        keys[k] = {"identical": same, "max_abs_diff": diff}
    return {"identical": all(v["identical"] for v in keys.values()), "arrays": keys}


def profile_check_arrays(p: LayerwiseProfile) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = dict(profile_arrays(p))
    out["cumulative_log_gain_median"] = np.array([p.cumulative_log_gain["median"]])
    for m in sorted(p.summaries):
        out[f"summary_median:{m}"] = np.asarray(p.summaries[m]["median"], dtype=np.float64)
    return out


def determinism_report(forward: dict[str, Any], gradients: dict[str, Any] | None, profile: dict[str, Any] | None,
                       task: str) -> dict[str, Any]:
    parts = {"forward": forward, "gradients": gradients, "profile": profile}
    return {
        "task": task,
        "identical": all(p["identical"] for p in parts.values() if p is not None),
        **{k: v for k, v in parts.items()},
    }
