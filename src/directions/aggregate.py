"""Aggregate several runs of the same configuration (typically different seeds)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .runinfo import read_json

SCALARS = (
    "cumulative_log_gain",
    "cumulative_log_gain_z",
    "alignment_final",
    "alignment_z_mean",
    "d_eff_ratio_final_over_first",
    "d_eff_z_mean",
    "log_gain_z_mean",
    "new_subspace_uncentered_mean",
    "new_subspace_uncentered_z_mean",
    "conversion_entropy_ratio",
    "conversion_dominant_share",
    "task_alignment_downstream_mean",
    "task_alignment_z_mean",
    "gradient_alignment_at_intervention",
    "gradient_alignment_downstream_mean",
    "gradient_alignment_z_mean",
    "first_order_final",
    "first_order_increment_centre_of_mass",
    "gradient_projection_z_mean",
    "first_order_increment_z_mean",
)
BY_KIND_KEYS = ("new_subspace_uncentered_z_mean", "log_gain_z_mean", "alignment_z_mean", "d_eff_z_mean",
                "cumulative_log_gain_z", "task_alignment_z_mean", "gradient_alignment_z_mean",
                "gradient_projection_z_mean", "first_order_increment_z_mean")


def _mean_sd(values: list[float | None]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


# Core quantities of a qualified task (docs/EXPERIMENT.md): read from the task's files when present, so runs
# written before a quantity existed contribute nothing to it rather than failing.
CORE = (
    "layer",                          # selected injection layer
    "rho_layer_norm",                 # selected strength as a multiple of the median residual norm there
    "held_out_effect",                # d(log p / token) over the baseline on the held-out pool
    "gate_excess",                    # excess of the effect over the gate controls
    "gap_fraction",                   # share of the few-shot gap (few-shot minus zero-shot log p / token) recovered
    "steered_accuracy",
    "neutral_kl",                     # KL(base || steered) on neutral prose
    "neutral_gate_excess",
    "cos_with_common",                # cosine of the function vector with the leave-one-out common direction
    "common_part_effect",
    "residual_part_effect",
    "handed_over_50_fraction",        # D29 (amended): hand-over depths as fractions of the downstream depth
    "handed_over_90_fraction",
    "carried_alone_until_50_fraction",
    "retained_after_removal_final",
    "carried_by_direction_final",
    "task_pc1_removal_min_retained",  # least share retained when the task PC1's component is removed
    "task_pc1_alone_max_retained",    # most the task PC1's component alone retains
)


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def core_quantities(root: Path, task: str) -> dict[str, Any]:
    """The task's core quantities of one run (``None`` where the run does not have them)."""
    tdir = root / "core" / "tasks" / task
    q = read_json(tdir / "qualification.json") if (tdir / "qualification.json").exists() else {}
    ev = read_json(tdir / "evaluation.json") if (tdir / "evaluation.json").exists() else {}
    out: dict[str, Any] = {k: None for k in CORE}
    out["layer"] = _get(q, "selection", "layer")
    out["rho_layer_norm"] = _get(q, "selection", "rho_layer_norm")
    out["held_out_effect"] = _get(ev, "steering_test", "mean_diff")
    out["gate_excess"] = _get(ev, "excess_test", "excess_mean")
    out["steered_accuracy"] = _get(ev, "steered", "accuracy")
    few, zero = _get(q, "fewshot", "logprob_per_token_mean"), _get(q, "zeroshot_baseline", "logprob_per_token_mean")
    if out["held_out_effect"] is not None and few is not None and zero is not None and few > zero:
        out["gap_fraction"] = float(out["held_out_effect"] / (few - zero))
    out["neutral_kl"] = _get(q, "damage", "neutral_kl_mean")
    out["neutral_gate_excess"] = _get(q, "damage", "neutral_gate_excess_mean")
    out["cos_with_common"] = _get(q, "decomposition", "cos_with_common")
    out["common_part_effect"] = _get(q, "decomposition", "common_part_mean_diff")
    out["residual_part_effect"] = _get(q, "decomposition", "residual_part_mean_diff")
    if (tdir / "commitment.json").exists():
        from .commitment import summarise

        com = read_json(tdir / "commitment.json")
        # re-summarised from the saved curves, so runs written under the original D29 summary get the amended one
        s = summarise(com, _get(com, "alpha") or 0.05, com.get("handover_shares") or [0.5, 0.9])
        inj, pc1 = s.get("injected", {}), s.get("task_pc1", {})
        for k in ("handed_over_50_fraction", "handed_over_90_fraction", "carried_alone_until_50_fraction",
                  "retained_after_removal_final", "carried_by_direction_final"):
            out[k] = inj.get(k)
        out["task_pc1_removal_min_retained"] = pc1.get("retained_after_removal_min")
        out["task_pc1_alone_max_retained"] = pc1.get("carried_by_direction_max")
    return out


def head_count(root: Path) -> dict[str, Any] | None:
    meta = read_json(root / "metadata.json") if (root / "metadata.json").exists() else {}
    hc = _get(meta, "function_vector", "head_count")
    return None if not isinstance(hc, dict) else {"chosen": hc.get("chosen"), "mean_effect_by_k": hc.get("mean_effect_by_k")}


def aggregate_runs(runs: list[Path]) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    run_ids, seeds, models, head_counts = [], [], set(), []
    for root in runs:
        summary = read_json(root / "core" / "summary.json")
        run_ids.append(summary["run_id"])
        seeds.append(summary.get("seed"))
        models.add(summary.get("model"))
        head_counts.append(head_count(root))
        cross = read_json(root / "core" / "cross_task.json") if (root / "core" / "cross_task.json").exists() else {"signatures": {}}
        for task, info in summary["tasks"].items():
            t = per_task.setdefault(task, {"n_runs": 0, "n_qualified": 0, "selections": [], "labels": {},
                                            "scalars": {k: [] for k in SCALARS}, "by_kind": {}, "gates_failed": {},
                                            "core": {k: [] for k in CORE}})
            t["n_runs"] += 1
            t["n_qualified"] += int(bool(info["qualified"]))
            if info["qualified"]:
                for k, v in core_quantities(root, task).items():
                    t["core"][k].append(v)
            for g, ok in (info.get("gates") or {}).items():
                if not ok:
                    t["gates_failed"][g] = t["gates_failed"].get(g, 0) + 1
            if info.get("selection"):
                t["selections"].append({"seed": summary.get("seed"), "layer": info["selection"]["layer"], "rho": info["selection"]["rho"]})
            sig = cross["signatures"].get(task)
            if sig:
                for lab in sig.get("labels", []):
                    t["labels"][lab] = t["labels"].get(lab, 0) + 1
                for k in SCALARS:
                    t["scalars"][k].append(sig.get(k))
                for kind, d in sig.get("by_kind", {}).items():
                    bk = t["by_kind"].setdefault(kind, {k: [] for k in BY_KIND_KEYS})
                    for k in bk:
                        bk[k].append(d.get(k))
    for t in per_task.values():
        t["scalars"] = {k: _mean_sd(v) for k, v in t["scalars"].items()}
        t["by_kind"] = {kind: {k: _mean_sd(v) for k, v in d.items()} for kind, d in t["by_kind"].items()}
        t["core"] = {k: {**_mean_sd(v), "values": v} for k, v in t["core"].items()}
    return {"runs": run_ids, "seeds": seeds, "models": sorted(m for m in models if m), "head_counts": head_counts,
            "tasks": per_task}


def _ms(d: dict[str, Any], digits: int = 2) -> str:
    if d["mean"] is None:
        return "-"
    return f"{d['mean']:.{digits}f}" if d["n"] < 2 else f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def format_core_table(result: dict[str, Any]) -> str:
    """Mean ± sd over the qualified runs of the core quantities (``None`` values skipped)."""
    hcs = [h["chosen"] for h in result.get("head_counts") or [] if h]
    lines = [f"head count chosen per run: {hcs or '-'}", ""]
    lines.append("| task | qualified | layer | held-out effect | gap fraction | accuracy | neutral KL | neutral excess | "
                 "cos common | common part | residual part | handed over 50 % | handed over 90 % | alone until 50 % | "
                 "retained at end | alone at end | PC1 removal min | PC1 alone max |")
    lines.append("|" + "---|" * 18)
    for task, t in sorted(result["tasks"].items()):
        c = t["core"]
        cells = [task, f"{t['n_qualified']}/{t['n_runs']}",
                 ", ".join(f"{x:g}" for x in c["layer"]["values"] if x is not None) or "-",
                 _ms(c["held_out_effect"]), _ms(c["gap_fraction"]), _ms(c["steered_accuracy"]), _ms(c["neutral_kl"]),
                 _ms(c["neutral_gate_excess"]), _ms(c["cos_with_common"]), _ms(c["common_part_effect"]),
                 _ms(c["residual_part_effect"]), _ms(c["handed_over_50_fraction"]), _ms(c["handed_over_90_fraction"]),
                 _ms(c["carried_alone_until_50_fraction"]), _ms(c["retained_after_removal_final"]),
                 _ms(c["carried_by_direction_final"]), _ms(c["task_pc1_removal_min_retained"]),
                 _ms(c["task_pc1_alone_max_retained"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_table(result: dict[str, Any]) -> str:
    lines = [f"runs: {len(result['runs'])}  seeds: {result['seeds']}  models: {result['models']}", ""]
    lines.append(format_core_table(result))
    lines.append("")
    lines.append("| task | qualified | selections (layer, rho) | cum log G | cum log G z | align z | N_unc z | "
                 "task-align z | grad-align z | labels |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for task, t in sorted(result["tasks"].items()):
        s = t["scalars"]

        def ms(k: str) -> str:
            d = s[k]
            return "-" if d["mean"] is None else f"{d['mean']:.2f} ± {d['sd']:.2f}"

        sel = ", ".join(f"({x['layer']}, {x['rho']:g})" for x in t["selections"]) or "-"
        labels = ", ".join(f"{k}×{v}" for k, v in sorted(t["labels"].items())) or "-"
        lines.append(f"| {task} | {t['n_qualified']}/{t['n_runs']} | {sel} | {ms('cumulative_log_gain')} | "
                     f"{ms('cumulative_log_gain_z')} | {ms('alignment_z_mean')} | {ms('new_subspace_uncentered_z_mean')} | "
                     f"{ms('task_alignment_z_mean')} | {ms('gradient_alignment_z_mean')} | {labels} |")
    return "\n".join(lines)
