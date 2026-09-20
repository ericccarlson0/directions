"""Depth of commitment: where the injected direction stops being needed as a direction (docs/DECISIONS.md D29).

The steered run injects ``alpha * v`` at read point ``l*``. At every later read point ``m`` the perturbation
``delta_m = h_m^steer - h_m^base`` is edited at the query token, on top of the original injection, and the
held-out effect that survives is measured:

* ``remove``: ``delta_m -> delta_m - (delta_m . u) u``, the component along ``u`` taken out. If the effect
  survives, the direction ``u`` is no longer carrying it at ``m``: it has been converted into other features.
* ``keep``: ``delta_m -> (delta_m . u) u``, everything the perturbation has acquired except its component
  along ``u`` taken out. If the effect survives, the direction itself still carries it.

``u`` is the injected direction (``v``) or the task's PC1 at ``m`` (the contrast direction the task itself
uses at that depth). The matched null applies the same edit to the same steered run along random unit
directions: removing a random component changes nothing, keeping only a random component keeps nothing, so
the paired excess of the real edit over the random edits says whether the direction matters at ``m`` beyond
the noise of editing. A second null (D35) applies an edit of the real edit's *size* along the random
directions (``random_matched_*``): where the model is sensitive to an edit of that size whatever its direction
the real edit says nothing about what carries the effect, and the read point is left out of the hand-over
summary (``readable_read_points``). The edits are exact because the forward pass is deterministic: the steered residual at
``m`` in the edited run equals the captured one up to ``m``, so an additive per-example intervention lands the
residual on the intended vector.

Outputs per variant and read point: the surviving effect (log p per token over the baseline), its share of
the full effect, a paired bootstrap CI, and the excess test against the random edits. Summary, per direction:
the hand-over depths as effect sizes (core; ``handover_shares``): the first read point from which removing the
direction leaves at least half (90 %) of the effect (``handed_over_50``, ``handed_over_90``) and the last read
point up to which the direction alone carries at least half (90 %) of it (``carried_alone_until_50``,
``carried_alone_until_90``), each also as a fraction of the downstream depth; then, secondary, the last read
point at which removing ``v`` still costs effect beyond random removal (``needed_until``), the first read
point after it (``commitment_layer``), and the share of the effect the direction alone carries at the end
(``carried_by_direction_final``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CommitmentConfig
from .model import ForwardResult, Intervention, ModelBackend
from .prompts import Prompt
from .stats import paired_bootstrap_test, paired_excess_test

VARIANTS = ("remove", "keep")


def edit_vectors(delta_m: np.ndarray, u: np.ndarray, variant: str, magnitude: np.ndarray | None = None) -> np.ndarray:
    """The per-example additive vectors that turn ``delta_m`` into its edited form: ``(n, d)`` float32.

    With ``magnitude`` (``(n,)``, signed) the component along ``u`` is not the perturbation's own but the given
    one: the norm-matched random control (D35), the same edit size along a random direction.
    """
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    d = np.asarray(delta_m, dtype=np.float64)
    coef = (d @ u) if magnitude is None else np.asarray(magnitude, dtype=np.float64)
    along = coef[:, None] * u[None, :]
    if variant == "remove":
        return (-along).astype(np.float32)
    if variant == "keep":
        return (along - d).astype(np.float32)
    raise ValueError(f"unknown variant {variant!r}")


def random_unit_directions(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    x = rng.standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def depth_of_commitment(
    backend: ModelBackend,
    prompts: list[Prompt],
    base: ForwardResult,
    steered: ForwardResult,
    layer: int,
    v: np.ndarray,
    alpha: float,
    task_directions: np.ndarray | None,
    cfg: CommitmentConfig,
    control_rng: np.random.Generator,
    boot_rng: np.random.Generator,
) -> dict[str, Any]:
    """The remove/keep sweep over the read points after ``layer`` (see the module docstring).

    ``base`` and ``steered`` must be captures (``residuals`` of shape ``(L+1, n, d)``) of exactly ``prompts``,
    run with the same batching as the edited runs (the pipeline runs them on the same prompt list).
    """
    assert base.residuals is not None and steered.residuals is not None
    n_read = base.residuals.shape[0]
    n = len(prompts)
    delta = steered.residuals.astype(np.float64) - base.residuals.astype(np.float64)  # (L+1, n, d)
    full_diff = steered.logprob_per_token - base.logprob_per_token
    full = float(np.mean(full_diff))
    dim = base.residuals.shape[2]
    controls = random_unit_directions(control_rng, cfg.n_controls, dim)
    read_points = list(range(layer + 1, n_read))
    inject = Intervention(layer, v, alpha)

    def run_edit(m: int, u: np.ndarray, variant: str, magnitude: np.ndarray | None = None) -> np.ndarray:
        vec = edit_vectors(delta[m], u, variant, magnitude)
        r = backend.run(prompts, interventions=[inject, Intervention(m, vec, 1.0)])
        return r.logprob_per_token - base.logprob_per_token

    directions: dict[str, Any] = {"injected": v}
    if task_directions is not None:
        directions["task_pc1"] = task_directions  # (L+1, d): the direction depends on m
    out: dict[str, Any] = {
        "intervention_layer": layer, "n_read_points": n_read, "read_points": read_points, "n_prompts": n,
        "n_controls": cfg.n_controls, "full_effect": full, "directions": sorted(directions), "variants": list(VARIANTS),
        "curves": {}, "summary": {},
    }
    rows_by: dict[str, list[dict[str, Any]]] = {f"{variant}:{dname}": [] for dname in directions for variant in VARIANTS}
    for m in read_points:
        # the random-direction edits do not depend on which real direction they are matched against: once per m
        ctl_by_variant = {variant: np.stack([run_edit(m, c, variant) for c in controls]) for variant in VARIANTS}
        for dname, dvec in directions.items():
            u = dvec[m] if dname == "task_pc1" else dvec
            u = np.asarray(u, dtype=np.float64)
            u = u / np.linalg.norm(u)
            magnitude = delta[m] @ u  # (n,) the real edit's size per prompt: the component along the direction
            for variant in VARIANTS:
                real = run_edit(m, u, variant)
                ctl = ctl_by_variant[variant]
                # the norm-matched control (D35): the same edit size along the random directions. A model that
                # is sensitive to an edit of that size, whatever its direction, cannot be read at this m
                matched = np.stack([run_edit(m, c, variant, magnitude) for c in controls]) if cfg.matched_controls else None
                t = paired_bootstrap_test(real + base.logprob_per_token, base.logprob_per_token, boot_rng, n_boot=cfg.n_boot)
                row: dict[str, Any] = {
                    "read_point": m, "effect": float(np.mean(real)), "retained": None if full == 0 else float(np.mean(real) / full),
                    "ci_low": t.ci_low, "ci_high": t.ci_high,
                    "cost_vs_full": float(np.mean(full_diff - real)),  # effect lost to the edit
                    "cost_test": paired_bootstrap_test(steered.logprob_per_token, real + base.logprob_per_token, boot_rng,
                                                       n_boot=cfg.n_boot).__dict__,
                    "edit_norm_median": float(np.median(np.abs(magnitude))),
                }
                row["random_effect_mean"] = float(np.mean(ctl))
                row["random_retained_mean"] = None if full == 0 else float(np.mean(ctl) / full)
                if matched is not None:
                    row["random_matched_effect_mean"] = float(np.mean(matched))
                    row["random_matched_retained_mean"] = None if full == 0 else float(np.mean(matched) / full)
                # remove: the real edit costs *more* than a random one when the direction is still needed;
                # keep: the real edit retains *more* than a random one when the direction still carries the effect
                if variant == "remove":
                    row["excess_vs_random"] = paired_excess_test(-real, -ctl, boot_rng, n_boot=cfg.n_boot).__dict__
                else:
                    row["excess_vs_random"] = paired_excess_test(real, ctl, boot_rng, n_boot=cfg.n_boot).__dict__
                rows_by[f"{variant}:{dname}"].append(row)
    out["curves"] = rows_by
    out["handover_shares"] = [float(s) for s in cfg.handover_shares]
    out["summary"] = summarise(out, cfg.alpha, out["handover_shares"], cfg.readability_tolerance)
    return out


def handed_over_from(rows: list[dict[str, Any]], share: float) -> int | None:
    """The first read point from which the share retained after removal stays at or above ``share``."""
    start = None
    for r in rows:
        if r["retained"] is not None and r["retained"] >= share:
            start = r["read_point"] if start is None else start
        else:
            start = None
    return start


def carried_alone_until(rows: list[dict[str, Any]], share: float) -> int | None:
    """The last read point up to which the direction alone retains at least ``share`` at every read point."""
    last = None
    for r in rows:
        if r["retained"] is None or r["retained"] < share:
            break
        last = r["read_point"]
    return last


def readable_read_points(rem: list[dict[str, Any]], keep: list[dict[str, Any]], tolerance: float) -> set[int]:
    """The read points at which the random edits keep their premise (D35): removing a random component
    retains the effect within ``tolerance`` of 1 and keeping only a random component retains within
    ``tolerance`` of 0. Elsewhere the model is too sensitive to an edit of that size for the real edit to say
    what carries the effect (Gemma 4's deep half), and the hand-over summary skips the read point. Rows
    without the random columns count as readable."""
    keep_by = {r["read_point"]: r for r in keep}
    out = set()
    for r in rem:
        k = keep_by.get(r["read_point"]) or {}
        ok = True
        for key in ("random_retained_mean", "random_matched_retained_mean"):  # shape-matched, and norm-matched (D35)
            rr, kr = r.get(key), k.get(key)
            ok = ok and (rr is None or abs(rr - 1.0) <= tolerance) and (kr is None or abs(kr) <= tolerance)
        if ok:
            out.add(r["read_point"])
    return out


def summarise(out: dict[str, Any], alpha: float, handover_shares: list[float] = (0.5, 0.9),
              readability_tolerance: float = 0.1) -> dict[str, Any]:
    """Where the effect is handed over (effect sizes, the core summary), then where the direction stops being
    needed against the random edits (significance, secondary) and what it carries at the end. The hand-over
    depths are read over the readable read points only (``readable_read_points``); the unreadable ones are
    listed."""
    layer, n_read = out["intervention_layer"], out["n_read_points"]
    downstream = max(1, (n_read - 1) - layer)

    def frac(m: int | None) -> float | None:
        return None if m is None else float((m - layer) / downstream)

    summary: dict[str, Any] = {}
    for dname in out["directions"]:
        rem = out["curves"].get(f"remove:{dname}", [])
        keep = out["curves"].get(f"keep:{dname}", [])
        readable = readable_read_points(rem, keep, readability_tolerance)
        rem_r = [r for r in rem if r["read_point"] in readable]
        keep_r = [r for r in keep if r["read_point"] in readable]
        handover: dict[str, Any] = {
            "unreadable_read_points": sorted({r["read_point"] for r in rem} - readable),
            "n_readable": len(readable), "readability_tolerance": readability_tolerance,
            "last_readable_read_point": max(readable) if readable else None,
            "retained_after_removal_last_readable": rem_r[-1]["retained"] if rem_r else None,
            "carried_by_direction_last_readable": keep_r[-1]["retained"] if keep_r else None,
        }
        for share in handover_shares:
            pct = f"{round(100 * share):d}"
            m_rem, m_keep = handed_over_from(rem_r, share), carried_alone_until(keep_r, share)
            handover[f"handed_over_{pct}"] = m_rem
            handover[f"handed_over_{pct}_fraction"] = frac(m_rem)
            handover[f"carried_alone_until_{pct}"] = m_keep
            handover[f"carried_alone_until_{pct}_fraction"] = frac(m_keep)
        needed = [r["read_point"] for r in rem
                  if "excess_vs_random" in r and r["excess_vs_random"]["excess_mean"] > 0 and r["excess_vs_random"]["p_value"] <= alpha]
        needed_until = max(needed) if needed else None
        commitment = None
        if needed_until is not None and needed_until < rem[-1]["read_point"]:
            commitment = needed_until + 1
        elif needed_until is None and rem:
            commitment = rem[0]["read_point"]  # never needed beyond the injection point
        carried = [r["read_point"] for r in keep
                   if "excess_vs_random" in r and r["excess_vs_random"]["excess_mean"] > 0 and r["excess_vs_random"]["p_value"] <= alpha]
        summary[dname] = {
            **handover,
            "needed_until": needed_until, "needed_until_fraction": frac(needed_until),
            "commitment_layer": commitment, "commitment_fraction": frac(commitment),
            "needed_at_end": bool(needed and needed_until == rem[-1]["read_point"]),
            "retained_after_removal_final": rem[-1]["retained"] if rem else None,
            "retained_after_removal_min": min((r["retained"] for r in rem if r["retained"] is not None), default=None),
            "carried_by_direction_final": keep[-1]["retained"] if keep else None,
            "carried_by_direction_max": max((r["retained"] for r in keep if r["retained"] is not None), default=None),
            "carried_until": max(carried) if carried else None, "carried_until_fraction": frac(max(carried) if carried else None),
        }
    return summary
