"""The D48 DAS runs, summarised (descriptive; the preregistered readouts are in each run's das.json).

Per model and site: the variables held (D48: Cause lower bound >= 0.5 and Isolate lower bound >= 0.75 at k*), and per
variable the test Cause and Isolate at k*, k*, the shuffled-target null, the untrained subspace and the projection
decodability. Then whether the control site is clean. Where the run stored its fitted bases (`das_bases.npz` keyed
`<site>/<var>/<k>`, from D49 on), the overlap of the S1, S2 and Y subspaces at k* per site (D49, descriptive): the
share of the smaller subspace inside the larger (mean squared cosine of the principal angles) beside a random pair's
expected share (the larger k over d).

    uv run python scripts/das_summary.py results/remote/run-<id> --out docs/das/summary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def _r(x, nd: int = 2):
    return None if x is None else round(x, nd)


def overlap(B1: np.ndarray, B2: np.ndarray) -> dict:
    """Two subspaces given by orthonormal rows (k x d): the cosines of their principal angles, the share of the
    smaller inside the larger (mean squared cosine) and a random pair's expected share (larger k / d)."""
    cos = np.linalg.svd(B1.astype(np.float64) @ B2.astype(np.float64).T, compute_uv=False)
    k_small, k_large = sorted((B1.shape[0], B2.shape[0]))
    return {"share": float((cos**2).sum() / k_small), "max_cos": float(cos.max()), "chance": k_large / B1.shape[1]}


def summarise(r: dict, bases=None) -> dict:
    out = {"control_site": r["control_site"], "control_clean": r["control_clean"], "n": r["n"], "sites": {}}
    for s in r["sites"]:
        ps = r["per_site"][str(s)]
        row = {"held": ps["held"], "count": ps["count"], "digits_in_use": ps["digits_in_use"], "variables": {}}
        for v in r["variables"]:
            c = r["cells"][f"{s}/{v}"]
            g = c["grid"][str(c["k_star"])]
            row["variables"][v] = {
                "k_star": c["k_star"], "cause": _r(g["cause"]["mean"]), "cause_ci": [_r(g["cause"]["lo"]), _r(g["cause"]["hi"])],
                "isolate": _r(g["isolate_mean"]), "isolate_lo": _r(g["isolate_lo"]),
                "isolate_by_kind": {k: _r(x["mean"]) for k, x in g["isolate"].items()},
                "untrained": _r(g["untrained_cause"]), "null": _r(c["null_cause"]["mean"]),
                "decodability": _r(c["projection_decodability"]["accuracy"]),
                "decodability_chance": _r(c["projection_decodability"]["chance"]),
                "cause_by_k": {k: _r(x["cause"]["mean"]) for k, x in sorted(c["grid"].items(), key=lambda kx: int(kx[0]))},
            }
        if bases is not None:
            keys = {v: f"{s}/{v}/{r['cells'][f'{s}/{v}']['k_star']}" for v in ("S1", "S2", "Y") if v in r["variables"]}
            row["overlap"] = {f"{u}-{v}": {k: _r(x, 3) for k, x in overlap(bases[keys[u]], bases[keys[v]]).items()}
                              for u, v in itertools.combinations(keys, 2) if keys[u] in bases.files and keys[v] in bases.files}
        out["sites"][s] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    summary = {}
    for run in sorted(args.root.glob("das_*")):
        f = run / "core" / "das.json"
        if f.exists():
            b = run / "core" / "das_bases.npz"
            bases = np.load(b) if b.exists() else None
            if bases is not None and not any(k.count("/") == 2 and "__" not in k for k in bases.files):
                bases = None  # D48's runs stored the test predictions under this name
            summary[run.name] = summarise(json.loads(f.read_text()), bases)
    if not summary:
        print(f"no DAS runs under {args.root}", file=sys.stderr)
        return 1
    for run, s in summary.items():
        print(f"{run}: control site {s['control_site']} clean: {s['control_clean']}; kept {s['n']}")
        for site, row in s["sites"].items():
            print(f"   read point {site}: held {row['held']} (count {row['count']}, digits in use {row['digits_in_use']})")
            for v, x in row["variables"].items():
                print(f"      {v:2s} k*={x['k_star']:<2} Cause {x['cause']} {x['cause_ci']}  Isolate {x['isolate']} (lo {x['isolate_lo']}) "
                      f"{x['isolate_by_kind']}  null {x['null']}  untrained {x['untrained']}  decod {x['decodability']} "
                      f"(chance {x['decodability_chance']})  by k {x['cause_by_k']}")
            for pair, o in row.get("overlap", {}).items():
                print(f"      overlap {pair}: share {o['share']} (random {o['chance']}), max cos {o['max_cos']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
