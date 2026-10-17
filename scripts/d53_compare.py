"""The D53 joint DAS read against D51's answer-token DAS and D52's ceilings (descriptive; the preregistered readouts
are each run's das.json).

Per model, site and variable: the joint fit's test Cause at k* with its interval, Isolate's lower bound, k*, the
shuffled-target null and whether it is held; D51's answer-token cell at the same site when that site was fitted
there; and D52's whole-state rate for the non-digit set at the site (the source changing the variable's bracket for
S1-S3, the source changing every sum otherwise), the ceiling a subspace of those tokens can reach. A model's two
parts (D53 splits each run by sites) are read together.

    uv run python scripts/d53_compare.py --d53 <dirs holding das_d53_* runs> --d51 <dirs holding das_d51_* runs> \
        --d52 <dirs holding posset_* runs> --out docs/das/d53_compare.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

CEILING_SOURCE = {"S1": "S1", "S2": "S2", "S3": "S3", "T": "all", "U": "all", "Y": "all"}


def size_of(name: str) -> str | None:
    m = re.search(r"qwen3_(\d+(?:\.\d+)?b)", name)
    return m.group(1) if m else None


def cell(c: dict) -> dict:
    g = c["grid"][str(c["k_star"])] if str(c["k_star"]) in c["grid"] else c["grid"][c["k_star"]]
    return {"cause": round(g["cause"]["mean"], 3), "cause_ci": [round(g["cause"]["lo"], 3), round(g["cause"]["hi"], 3)],
            "isolate_lo": round(g["isolate_lo"], 3), "k_star": c["k_star"], "held": c["held"],
            "null": round(c["null_cause"]["mean"], 3), "decodability": round(c["projection_decodability"]["accuracy"], 3)}


def load_runs(roots: list[Path], prefix: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for root in roots:
        for run in sorted(root.glob(f"{prefix}*")):
            f = run / "core" / "das.json"
            if f.exists() and size_of(run.name):
                out.setdefault(size_of(run.name), []).append(json.loads(f.read_text()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d53", type=Path, nargs="+", required=True)
    ap.add_argument("--d51", type=Path, nargs="+", default=[])
    ap.add_argument("--d52", type=Path, nargs="+", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    d53 = load_runs(args.d53, "das_d53_")
    d51 = {k: v[0] for k, v in load_runs(args.d51, "das_d51_").items()}
    d52 = {}
    for root in args.d52:
        for run in sorted(root.glob("posset_*")):
            f = run / "core" / "tasks" / "three_lane" / "posset.json"
            if f.exists() and size_of(run.name):
                d52[size_of(run.name)] = json.loads(f.read_text())
    if not d53:
        print("no D53 runs found", file=sys.stderr)
        return 1
    summary: dict[str, dict] = {}
    for size in sorted(d53, key=lambda s: float(s[:-1])):
        parts = d53[size]
        control = next((p["control_site"] for p in parts if p["control_site"] is not None), None)
        model: dict[str, dict] = {"control_site": control, "sites": {}}
        for p in parts:
            for s in p["sites"]:
                per = {"held": p["per_site"][str(s)]["held"] if str(s) in p["per_site"] else p["per_site"][s]["held"],
                       "variables": {}}
                for v in p["variables"]:
                    row = {"joint": cell(p["cells"][f"{s}/{v}"])}
                    old = d51.get(size)
                    if old and s in old["sites"] and f"{s}/{v}" in old["cells"]:
                        row["answer_token_d51"] = cell(old["cells"][f"{s}/{v}"])
                    pz = d52.get(size)
                    if pz and s in pz["read_points"] and v in CEILING_SOURCE:
                        i = pz["read_points"].index(s)
                        src = CEILING_SOURCE[v]
                        row["ceiling_d52"] = {"non_digit": round(float(pz["rates"][src]["non_digit"]["flip"]["mean"][i]), 3),
                                              "answer": round(float(pz["rates"][src]["answer"]["flip"]["mean"][i]), 3)}
                    per["variables"][v] = row
                model["sites"][s] = per
        model["sites"] = dict(sorted(model["sites"].items()))
        summary[f"qwen3_{size}"] = model
        print(f"qwen3_{size} (control {control})")
        for s, per in model["sites"].items():
            print(f"  read point {s}{' (control)' if s == control else ''}: held {per['held']}")
            for v, row in per["variables"].items():
                j = row["joint"]
                line = (f"    {v:2s} joint {j['cause']:.2f} [{j['cause_ci'][0]:.2f}, {j['cause_ci'][1]:.2f}] iso_lo {j['isolate_lo']:.2f}"
                        f" k {j['k_star']} null {j['null']:.2f} dec {j['decodability']:.2f}{' HELD' if j['held'] else ''}")
                if "answer_token_d51" in row:
                    a = row["answer_token_d51"]
                    line += f" | D51 {a['cause']:.2f} iso_lo {a['isolate_lo']:.2f}{' HELD' if a['held'] else ''}"
                if "ceiling_d52" in row:
                    line += f" | ceiling {row['ceiling_d52']['non_digit']:.2f} (answer token {row['ceiling_d52']['answer']:.2f})"
                print(line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
