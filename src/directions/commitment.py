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
the noise of editing. The edits are exact because the forward pass is deterministic: the steered residual at
``m`` in the edited run equals the captured one up to ``m``, so an additive per-example intervention lands the
residual on the intended vector.

Outputs per variant and read point: the surviving effect (log p per token over the baseline), its share of
the full effect, a paired bootstrap CI, and the excess test against the random edits. Summary: the last read
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


def edit_vectors(delta_m: np.ndarray, u: np.ndarray, variant: str) -> np.ndarray:
    """The per-example additive vectors that turn ``delta_m`` into its edited form: ``(n, d)`` float32."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    d = np.asarray(delta_m, dtype=np.float64)
    along = (d @ u)[:, None] * u[None, :]
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

    def run_edit(m: int, u: np.ndarray, variant: str) -> np.ndarray:
        vec = edit_vectors(delta[m], u, variant)
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
            for variant in VARIANTS:
                u = dvec[m] if dname == "task_pc1" else dvec
                real = run_edit(m, u, variant)
                ctl = ctl_by_variant[variant]
                t = paired_bootstrap_test(real + base.logprob_per_token, base.logprob_per_token, boot_rng, n_boot=cfg.n_boot)
                row: dict[str, Any] = {
                    "read_point": m, "effect": float(np.mean(real)), "retained": None if full == 0 else float(np.mean(real) / full),
                    "ci_low": t.ci_low, "ci_high": t.ci_high,
                    "cost_vs_full": float(np.mean(full_diff - real)),  # effect lost to the edit
                    "cost_test": paired_bootstrap_test(steered.logprob_per_token, real + base.logprob_per_token, boot_rng,
                                                       n_boot=cfg.n_boot).__dict__,
                }
                row["random_effect_mean"] = float(np.mean(ctl))
                row["random_retained_mean"] = None if full == 0 else float(np.mean(ctl) / full)
                # remove: the real edit costs *more* than a random one when the direction is still needed;
                # keep: the real edit retains *more* than a random one when the direction still carries the effect
                if variant == "remove":
                    row["excess_vs_random"] = paired_excess_test(-real, -ctl, boot_rng, n_boot=cfg.n_boot).__dict__
                else:
                    row["excess_vs_random"] = paired_excess_test(real, ctl, boot_rng, n_boot=cfg.n_boot).__dict__
                rows_by[f"{variant}:{dname}"].append(row)
    out["curves"] = rows_by
    out["summary"] = summarise(out, cfg.alpha)
    return out


def summarise(out: dict[str, Any], alpha: float) -> dict[str, Any]:
    """The depth at which the injected direction stops being needed, and what it carries at the end."""
    layer, n_read = out["intervention_layer"], out["n_read_points"]
    downstream = max(1, (n_read - 1) - layer)

    def frac(m: int | None) -> float | None:
        return None if m is None else float((m - layer) / downstream)

    summary: dict[str, Any] = {}
    for dname in out["directions"]:
        rem = out["curves"].get(f"remove:{dname}", [])
        keep = out["curves"].get(f"keep:{dname}", [])
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
