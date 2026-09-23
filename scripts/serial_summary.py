"""The serial injection test across models (docs/DECISIONS.md D41): one table from the serial runs' summaries
(``<run>/core/summary.json``): per model, composition and construction, the composed task's own control, the sum of
the two components' vectors at the first layer, the serial curve's best layer and value against the null, each
control alone, and the reading.

usage: uv run python scripts/serial_summary.py qwen3_0.6b=<serial run> ... [--out results/serial_summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(x: Any) -> str:
    return "–" if x is None else (f"{x:+.2f}" if isinstance(x, float) else str(x))


def summarise(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, s in models.items():
        for c in s["compositions"]:
            grid = c["grid"]
            best = max(grid, key=lambda r: r["serial"]) if grid else None
            composed = (c.get("composed_control") or {}).get("effect")
            rows.append({"model": model, "task": c["task"], "construction": c["construction"], "steps": c["steps"],
                         "first_layer": c["first"]["layer"], "first_alone": c["first_alone"]["effect"],
                         "composed_control": composed, "composed_layer": (c.get("composed_control") or {}).get("layer"),
                         "superposition": c["superposition_at_first_layer"]["effect"], "angle_deg": c["superposition_at_first_layer"]["angle_deg"],
                         "grid_layers": [r["layer"] for r in grid], "bright_layers": c["bright_layers"], "reading": c["reading"],
                         "best_layer": None if best is None else best["layer"], "best_serial": None if best is None else best["serial"],
                         "best_null": None if best is None else best["null_mean"], "best_second_alone": None if best is None else best["second_alone"],
                         "best_p": None if best is None else best["serial_over_null"]["p_value"],
                         "serial_by_layer": {str(r["layer"]): r["serial"] for r in grid},
                         # the serial effect as a share of the composed control's, at the best layer
                         "best_share": None if best is None or not composed else best["serial"] / composed})
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | composition | construction | first alone (layer) | composed control (layer) | sum at first layer (angle) | best serial: layer, effect (null; second alone) | bright layers / grid | reading |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['task']} | {r['construction']} | {_fmt(r['first_alone'])} ({r['first_layer']}) | {_fmt(r['composed_control'])} ({_fmt(r['composed_layer'])}) | "
                     f"{_fmt(r['superposition'])} ({r['angle_deg']:.0f}°) | {_fmt(r['best_layer'])}, {_fmt(r['best_serial'])} ({_fmt(r['best_null'])}; {_fmt(r['best_second_alone'])}) | "
                     f"{len(r['bright_layers'])}/{len(r['grid_layers'])} {r['bright_layers']} | {r['reading']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="model=<serial run directory>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    models: dict[str, dict[str, Any]] = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        with open(Path(path) / "core" / "summary.json") as f:
            models[name] = json.load(f)
    rows = summarise(models)
    print(table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
