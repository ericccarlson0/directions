"""The composition test across models (docs/DECISIONS.md D41): one table from the landmark runs of the composition
family (``<run>/landmarks/landmarks.json``, whose per-task ``composition`` entries hold the verdicts), plus the
preregistered rule per construction (a verdict holds when at least three of the four Qwen3 sizes read so).

usage: uv run python scripts/composition_summary.py qwen3_0.6b=<landmark run> ... [--out results/composition_summary.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

QWEN3 = ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b")


def _fmt(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def summarise(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per model, composition and construction: the verdict per component, the hand-overs of the composition and
    its components, the excess peaks and the causal maxima; then the rule over the Qwen3 sizes."""
    rows: list[dict[str, Any]] = []
    for model, lm in models.items():
        tasks = lm["tasks"]
        for task, t in tasks.items():
            comp = t.get("composition")
            if not comp:
                continue
            for c, entry in comp["constructions"].items():
                row: dict[str, Any] = {"model": model, "task": task, "construction": c, "handover": entry.get("handover"),
                                       "components": {}, "peaks_in_step_order": entry.get("peaks_in_step_order")}
                for r in comp["references"]:
                    v = entry["per_reference"].get(r)
                    if v is None:
                        continue
                    row["components"][r] = {"verdict": v["verdict"], "first_positive": v.get("first_positive"),
                                            "first_negative_after": v.get("first_negative_after"),
                                            "excess_peak": v.get("excess_peak"), "excess_peak_read_point": v.get("excess_peak_read_point"),
                                            "natural_cos_peak": v.get("natural_cos_peak"), "natural_cos_peak_read_point": v.get("natural_cos_peak_read_point"),
                                            "keep_along_component_max": v.get("keep_along_component_max"),
                                            "component_handover": ((tasks.get(r) or {}).get("handover", {}).get(c) or {}).get("read_point")}
                rows.append(row)
    # the rule: per composition, construction and (innermost) component, the verdict on >= 3 of the 4 Qwen3 sizes
    rule: dict[str, Any] = {}
    for row in rows:
        if row["model"] not in QWEN3:
            continue
        for r, v in row["components"].items():
            rule.setdefault(row["task"], {}).setdefault(row["construction"], {}).setdefault(r, []).append(v["verdict"])
    verdicts: dict[str, Any] = {}
    for task, per_c in rule.items():
        for c, per_r in per_c.items():
            for r, vs in per_r.items():
                count = Counter(vs)
                top, n = count.most_common(1)[0]
                verdicts.setdefault(task, {}).setdefault(c, {})[r] = {"verdict": top if n >= 3 else "undecided", "counts": dict(count), "n_sizes": len(vs)}
    return {"rows": rows, "rule": verdicts}


def table(summary: dict[str, Any]) -> str:
    lines = ["| model | composition | construction | hand-over (composition; components) | per component: verdict, first + / first − after, e peak (read point), keep-along max |",
             "|---|---|---|---|---|"]
    for row in summary["rows"]:
        comps = "; ".join(f"{r}: {v['verdict']}, {_fmt(v['first_positive'])}/{_fmt(v['first_negative_after'])}, {_fmt(v['excess_peak'])} ({_fmt(v['excess_peak_read_point'])}), {_fmt(v['keep_along_component_max'])}"
                          for r, v in row["components"].items())
        hand = f"{_fmt(row['handover'])}; " + ", ".join(f"{r} {_fmt(v['component_handover'])}" for r, v in row["components"].items())
        lines.append(f"| {row['model']} | {row['task']} | {row['construction']} | {hand} | {comps} |")
    lines.append("")
    lines.append("Rule (>= 3 of 4 Qwen3 sizes):")
    for task, per_c in summary["rule"].items():
        for c, per_r in per_c.items():
            lines.append(f"  {task:20s} {c:8s} " + "; ".join(f"{r}: {v['verdict']} {v['counts']}" for r, v in per_r.items()))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="model=<landmark run directory> (with landmarks/landmarks.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    models: dict[str, dict[str, Any]] = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        with open(Path(path) / "landmarks" / "landmarks.json") as f:
            models[name] = json.load(f)
    summary = summarise(models)
    print(table(summary))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
