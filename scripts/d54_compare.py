"""The D54 two-token DAS read with its control rule, beside D51's answer-token cells and D52's ceilings.

Per model, variant (joint or shared basis), site and variable: the fit's test Cause at k* with its interval,
Isolate's lower bound, the shuffled-target null and D48's held criterion; then D54's control rule: the Cause hits at
the site minus the same variable's hits at the control, paired by test base, with a 95% percentile-bootstrap
interval (1000 resamples). A variable is read as held when D48's criterion holds and that lower bound is above 0;
a variable held at the control voids the variant on that model. Beside each cell, D51's answer-token result at the
site and D52's whole-state rate for answer + = (the bracket's own source for S1-S3, every sum otherwise).
The same rule reads D55's max DAS at the answer token (`--prefix das_d55_`; no D51 or D52 runs beside it).

    uv run python scripts/d54_compare.py --d54 <dirs holding das_d54_* runs> --d51 <dirs holding das_d51_* runs> \
        --d52 <dirs holding posset_* runs> --out docs/das/d54_compare.json
    uv run python scripts/d54_compare.py --prefix das_d55_ --d54 <dirs holding das_d55_* runs> --out docs/das/d55_compare.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

CEILING_SOURCE = {"S1": "S1", "S2": "S2", "S3": "S3", "T": "all", "U": "all", "Y": "all"}
SEED, N_BOOT = 20260916, 1000


def size_of(name: str) -> str | None:
    m = re.search(r"qwen3_(\d+(?:\.\d+)?b)", name)
    return m.group(1) if m else None


def cell(c: dict) -> dict:
    g = c["grid"][str(c["k_star"])]
    return {"cause": round(g["cause"]["mean"], 3), "cause_ci": [round(g["cause"]["lo"], 3), round(g["cause"]["hi"], 3)],
            "isolate_lo": round(g["isolate_lo"], 3), "k_star": c["k_star"], "held_d48": c["held"],
            "null": round(c["null_cause"]["mean"], 3), "decodability": round(c["projection_decodability"]["accuracy"], 3)}


def paired_over_control(pred: dict, site_key: str, control_key: str, rng: np.random.Generator) -> dict:
    """Cause hits at the site minus at the control, on the test bases both were scored on."""
    a, b = pred[f"cause_hit__{site_key}"], pred[f"cause_hit__{control_key}"]
    rows = np.intersect1d(a[:, 0], b[:, 0])
    da = dict(zip(a[:, 0].tolist(), a[:, 1].tolist()))
    db = dict(zip(b[:, 0].tolist(), b[:, 1].tolist()))
    d = np.array([da[r] - db[r] for r in rows.tolist()], dtype=float)
    n = len(d)
    boots = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT) @ d / n
    return {"n": int(n), "mean": round(float(d.mean()), 3), "lo": round(float(np.quantile(boots, 0.025)), 3),
            "hi": round(float(np.quantile(boots, 0.975)), 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d54", type=Path, nargs="+", required=True)
    ap.add_argument("--d51", type=Path, nargs="+", default=[])
    ap.add_argument("--d52", type=Path, nargs="+", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prefix", default="das_d54_", help="the run-name prefix of the DAS runs read with the control rule")
    args = ap.parse_args(argv)
    runs = []
    for root in args.d54:
        for run in sorted(root.glob(f"{args.prefix}*")):
            if (run / "core" / "das.json").exists() and size_of(run.name):
                runs.append(run)
    d51 = {}
    for root in args.d51:
        for run in sorted(root.glob("das_d51_*")):
            if (run / "core" / "das.json").exists() and size_of(run.name):
                d51[size_of(run.name)] = json.loads((run / "core" / "das.json").read_text())
    d52 = {}
    for root in args.d52:
        for run in sorted(root.glob("posset_*")):
            f = run / "core" / "tasks" / "three_lane" / "posset.json"
            if f.exists() and size_of(run.name):
                d52[size_of(run.name)] = json.loads(f.read_text())
    if not runs:
        print(f"no {args.prefix}* runs found", file=sys.stderr)
        return 1
    summary: dict[str, dict] = {}
    for run in sorted(runs, key=lambda r: (float(size_of(r.name)[:-1]), r.name)):
        r = json.loads((run / "core" / "das.json").read_text())
        pred = dict(np.load(run / "core" / "das_predictions.npz"))
        size = size_of(run.name)
        variant = "shared" if r.get("shared_basis") else "joint" if r.get("positions", "answer") != "answer" else "answer"
        ctrl = r["control_site"]
        rng = np.random.default_rng(SEED)
        out = {"control_site": ctrl, "control_held": r["per_site"][str(ctrl)]["held"], "sites": {}}
        out["void"] = bool(out["control_held"])
        for s in r["sites"]:
            per = {"held_d48": r["per_site"][str(s)]["held"], "held": [], "variables": {}}
            for v in r["variables"]:
                c = r["cells"][f"{s}/{v}"]
                row = {"fit": cell(c)}
                if s != ctrl:
                    cc = r["cells"][f"{ctrl}/{v}"]
                    row["over_control"] = paired_over_control(pred, f"{s}/{v}/{c['k_star']}", f"{ctrl}/{v}/{cc['k_star']}", rng)
                    if c["held"] and row["over_control"]["lo"] > 0 and not out["void"]:
                        per["held"].append(v)
                old = d51.get(size)
                if old and f"{s}/{v}" in old["cells"]:
                    row["answer_token_d51"] = cell(old["cells"][f"{s}/{v}"])
                pz = d52.get(size)
                if pz and s in pz["read_points"] and v in CEILING_SOURCE:
                    i = pz["read_points"].index(s)
                    src = CEILING_SOURCE[v]
                    row["ceiling_d52"] = {"answer_eq": round(float(pz["rates"][src]["answer_eq"]["flip"]["mean"][i]), 3),
                                          "answer": round(float(pz["rates"][src]["answer"]["flip"]["mean"][i]), 3)}
                per["variables"][v] = row
            per["sums_held"] = sum(v in per["held"] for v in ("S1", "S2", "S3"))
            out["sites"][s] = per
        summary[f"qwen3_{size}/{variant}"] = out
        print(f"qwen3_{size} {variant}: control {ctrl} held {out['control_held']}{' (VOID)' if out['void'] else ''}")
        for s, per in out["sites"].items():
            print(f"  read point {s}{' (control)' if s == ctrl else ''}: held (D48) {per['held_d48']}; held (D54) {per['held']}; sums held {per['sums_held']}")
            for v, row in per["variables"].items():
                f = row["fit"]
                line = (f"    {v:2s} {f['cause']:.2f} [{f['cause_ci'][0]:.2f}, {f['cause_ci'][1]:.2f}] iso_lo {f['isolate_lo']:.2f}"
                        f" k {f['k_star']} null {f['null']:.2f} dec {f['decodability']:.2f}")
                if "over_control" in row:
                    o = row["over_control"]
                    line += f" | over control {o['mean']:+.2f} [{o['lo']:+.2f}, {o['hi']:+.2f}]"
                if "answer_token_d51" in row:
                    a = row["answer_token_d51"]
                    line += f" | D51 {a['cause']:.2f} iso_lo {a['isolate_lo']:.2f}{' HELD' if a['held_d48'] else ''}"
                if "ceiling_d52" in row:
                    line += f" | ceiling {row['ceiling_d52']['answer_eq']:.2f} (answer token {row['ceiling_d52']['answer']:.2f})"
                print(line)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
