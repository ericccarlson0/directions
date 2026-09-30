"""The staging test (docs/DECISIONS.md D42) across checkpoints: the per-checkpoint readings of every composition and
the cross-checkpoint rule (a reading holds when at least ``--min-agree`` checkpoints read so).

usage: uv run python scripts/staging_summary.py qwen3_4b=<staging run> qwen3_8b=<staging run> ... --out results/staging_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY = ("upper_antonym", "last_antonym", "upper_last_antonym")


def load(runs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """model -> task -> the task's staging.json."""
    out: dict[str, dict[str, Any]] = {}
    for model, root in runs.items():
        summary = json.loads((root / "core" / "summary.json").read_text())
        out[model] = {c["task"]: json.loads((root / "core" / c["task"] / "staging.json").read_text()) for c in summary["compositions"]}
    return out


def masking_peak(res: dict[str, Any]) -> dict[str, Any] | None:
    """The masking layer at which the intermediate's top-1 rate peaks, with the rate there and at both ends."""
    m = res.get("masking", {}).get("per_intermediate", {}).get(next(iter(res["intermediates"])))
    if not m or "top1_int" not in m:
        return None
    t = m["top1_int"]
    i = max(range(len(t)), key=lambda k: t[k])
    return {"layer": i, "top1_int": t[i], "top1_int_ends": [t[0], t[-1]], "top1_fin_at_peak": m["top1_fin"][i],
            "top1_fin_unmasked": m["top1_fin"][-1]}


def rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model, tasks in data.items():
        for task, res in tasks.items():
            v = res["verdicts"]
            row = {"model": model, "task": task, "natural": "staged" if v["natural"]["staged"] else "one_step",
                   "natural_masking": v["natural"].get("masking_staged"), "natural_lens": v["natural"]["lens_staged"],
                   "lens_power": v["natural"]["lens_power"], "masking_peak": masking_peak(res)}
            for c, x in v["controls"].items():
                row[c] = x
            out.append(row)
    return out


def verdicts(rs: list[dict[str, Any]], min_agree: int) -> dict[str, Any]:
    """Per composition and reading (the model's own computation, each construction's control): the counts of
    checkpoints per reading, and the reading that holds (at least ``min_agree`` checkpoints) or None."""
    out: dict[str, Any] = {}
    for task in dict.fromkeys(r["task"] for r in rs):
        entry: dict[str, Any] = {}
        for key in ("natural", "fv", "learned"):
            counts: dict[str, int] = {}
            for r in rs:
                if r["task"] != task or key not in r:
                    continue
                reading = r[key] if key == "natural" else r[key]["reading"]
                counts[reading] = counts.get(reading, 0) + 1
            holds = [k for k, n in counts.items() if n >= min_agree and k != "undecided"]
            entry[key] = {"counts": counts, "holds": holds[0] if holds else None}
        out[task] = entry
    return out


def table(rs: list[dict[str, Any]]) -> str:
    lines = ["| model | composition | own computation (masking; lens) | intermediate top-1 at the masking peak (L) | head mean: e_perp; lens | learned: e_perp; lens |",
             "|---|---|---|---|---|---|"]
    for r in rs:
        mp = r["masking_peak"]
        peak = "n/a" if mp is None else f"{mp['top1_int']:.2f} ({mp['layer']}; ends {mp['top1_int_ends'][0]:.2f}, {mp['top1_int_ends'][1]:.2f})"
        cells = []
        for c in ("fv", "learned"):
            x = r.get(c)
            cells.append("–" if x is None else f"{x['reading']}: {x['reference_verdict']}{'' if x['reference_power'] else ' (no power)'}; "
                                               f"{'staged' if x['lens_staged'] else 'not staged'}{'' if x['lens_power'] else ' (no power)'}")
        own = f"{r['natural']}: {'staged' if r['natural_masking'] else 'not staged'}; {'staged' if r['natural_lens'] else 'not staged'}"
        lines.append(f"| {r['model']} | {r['task']} | {own} | {peak} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="model=<staging run directory>")
    p.add_argument("--min-agree", type=int, default=3, help="checkpoints that must agree for a reading to hold (D42: 3 of 4)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    runs = {k: Path(v) for k, v in (a.split("=", 1) for a in args.runs)}
    rs = rows(load(runs))
    result = {"rows": rs, "verdicts": verdicts(rs, args.min_agree), "min_agree": args.min_agree, "primary": list(PRIMARY)}
    print(table(rs))
    print(json.dumps(result["verdicts"], indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
