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
)
BY_KIND_KEYS = ("new_subspace_uncentered_z_mean", "log_gain_z_mean", "alignment_z_mean", "d_eff_z_mean",
                "cumulative_log_gain_z", "task_alignment_z_mean", "gradient_alignment_z_mean")


def _mean_sd(values: list[float | None]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def aggregate_runs(runs: list[Path]) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    run_ids, seeds, models = [], [], set()
    for root in runs:
        summary = read_json(root / "core" / "summary.json")
        run_ids.append(summary["run_id"])
        seeds.append(summary.get("seed"))
        models.add(summary.get("model"))
        cross = read_json(root / "core" / "cross_task.json") if (root / "core" / "cross_task.json").exists() else {"signatures": {}}
        for task, info in summary["tasks"].items():
            t = per_task.setdefault(task, {"n_runs": 0, "n_qualified": 0, "selections": [], "labels": {},
                                            "scalars": {k: [] for k in SCALARS}, "by_kind": {}, "gates_failed": {}})
            t["n_runs"] += 1
            t["n_qualified"] += int(bool(info["qualified"]))
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
    return {"runs": run_ids, "seeds": seeds, "models": sorted(m for m in models if m), "tasks": per_task}


def format_table(result: dict[str, Any]) -> str:
    lines = [f"runs: {len(result['runs'])}  seeds: {result['seeds']}  models: {result['models']}", ""]
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
