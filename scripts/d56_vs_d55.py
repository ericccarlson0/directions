"""D56 against D55, cell by cell: the same bases, Cause pairs, sites and k, with Isolate added to the training objective.

Per model, read point and variable, at each run's k*: test Cause with its 95% interval, the Isolate lower bound
(the smallest over the variable's Isolate kinds) and each kind's rate, under D55 (Cause only) and D56 (Cause and
Isolate). Each cell is read by D56's preregistered readings: held in both; held under D56 only (a clean subspace
that Cause-only training missed); Isolate passes and Cause fails (the variable is not separable from the answer
in a subspace of this size); Cause passes and Isolate fails; both fail. "Held" is D48's criterion with D54's
control rule, as `scripts/d54_compare.py` computes it.

    uv run python scripts/d56_vs_d55.py --d55 <dirs holding das_d55_* runs> --d56 <dirs holding das_d56_* runs> \
        --out docs/das/d56_vs_d55.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import paired_over_control, size_of  # noqa: E402

SEED = 20260916


def read_run(run: Path) -> dict:
    """Per cell at k*: Cause, its interval, the Isolate lower bound and per-kind rates, and held under D48 + D54."""
    r = json.loads((run / "core" / "das.json").read_text())
    pred = dict(np.load(run / "core" / "das_predictions.npz"))
    ctrl = r["control_site"]
    th = r["thresholds"]
    rng = np.random.default_rng(SEED)
    out = {}
    for s in r["sites"]:
        for v in r["variables"]:
            c = r["cells"][f"{s}/{v}"]
            g = c["grid"][str(c["k_star"])]
            cell = {"k_star": c["k_star"], "cause": round(g["cause"]["mean"], 3), "cause_lo": round(g["cause"]["lo"], 3),
                    "cause_hi": round(g["cause"]["hi"], 3), "isolate_lo": round(g["isolate_lo"], 3),
                    "isolate": {k: round(x["mean"], 3) for k, x in g["isolate"].items()}, "held_d48": c["held"]}
            over = None
            if s != ctrl:
                cc = r["cells"][f"{ctrl}/{v}"]
                over = paired_over_control(pred, f"{s}/{v}/{c['k_star']}", f"{ctrl}/{v}/{cc['k_star']}", rng)
                cell["over_control"] = over
            cell["held"] = bool(c["held"] and (s == ctrl or over["lo"] > 0))
            cell["cause_passes"] = g["cause"]["lo"] >= th["cause_lower"]
            cell["isolate_passes"] = g["isolate_lo"] >= th["isolate_lower"]
            out[f"{s}/{v}"] = cell
    return {"control_site": ctrl, "sites": r["sites"], "variables": r["variables"], "cells": out}


def reading(old: dict, new: dict) -> str:
    if new["held"] and old["held"]:
        return "held in both"
    if new["held"]:
        return "held under D56 only"
    if old["held"]:
        return "held under D55 only"
    if new["isolate_passes"] and not new["cause_passes"]:
        return "Isolate passes, Cause fails"
    if new["cause_passes"] and not new["isolate_passes"]:
        return "Cause passes, Isolate fails"
    return "both fail"


def runs_in(roots: list[Path], prefix: str) -> dict[str, Path]:
    found = {}
    for root in roots:
        for run in sorted(root.glob(f"{prefix}*")):
            if (run / "core" / "das.json").exists() and size_of(run.name):
                found[size_of(run.name)] = run
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d55", type=Path, nargs="+", required=True)
    ap.add_argument("--d56", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    d55, d56 = runs_in(args.d55, "das_d55_"), runs_in(args.d56, "das_d56_")
    sizes = sorted(set(d55) & set(d56), key=lambda x: float(x[:-1]))
    if not sizes:
        print("no model has both a D55 and a D56 run", file=sys.stderr)
        return 1
    summary = {}
    for size in sizes:
        a, b = read_run(d55[size]), read_run(d56[size])
        if a["sites"] != b["sites"] or a["variables"] != b["variables"]:
            raise ValueError(f"{size}: D55 and D56 differ in sites or variables")
        cells = {}
        print(f"qwen3_{size} (control {b['control_site']}): Cause [lo] / Isolate lo, D55 -> D56; reading")
        for key in a["cells"]:
            old, new = a["cells"][key], b["cells"][key]
            cells[key] = {"d55": old, "d56": new, "reading": reading(old, new)}
            print(f"  {key:5s} {old['cause']:.2f} [{old['cause_lo']:.2f}] / {old['isolate_lo']:.2f}  ->  "
                  f"{new['cause']:.2f} [{new['cause_lo']:.2f}] / {new['isolate_lo']:.2f}   {cells[key]['reading']}")
        sums = [v for v in ("S1", "S2", "S3") if v in b["variables"]]
        G = [s for s in b["sites"] if s != b["control_site"]][0]
        summary[f"qwen3_{size}"] = {"control_site": b["control_site"], "gather": G, "cells": cells,
                                    "sums_held_at_G": {"d55": sum(a["cells"][f"{G}/{v}"]["held"] for v in sums),
                                                       "d56": sum(b["cells"][f"{G}/{v}"]["held"] for v in sums)},
                                    "C_held_anywhere": {"d55": any(a["cells"][f"{s}/C"]["held"] for s in a["sites"])
                                                        if "C" in a["variables"] else None,
                                                        "d56": any(b["cells"][f"{s}/C"]["held"] for s in b["sites"])
                                                        if "C" in b["variables"] else None}}
        print(f"  sums held at G = {G}: D55 {summary[f'qwen3_{size}']['sums_held_at_G']['d55']}, "
              f"D56 {summary[f'qwen3_{size}']['sums_held_at_G']['d56']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
