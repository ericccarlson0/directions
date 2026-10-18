"""The D51 DAS sites of each model, from its three-bracket job 1 (docs/DECISIONS.md D51).

G3 is the first read point with all three sums present at the answer token (D45 reading of the stored rates). The
control is the latest read point at or below G3 - 2 at which all three bracket swaps change the answer token's answer
in at most 0.10 of the bases (D50's rule). F is the read point at which the three-bracket Y probe (D46) reaches 0.9
of its final accuracy. The other sites run from G3 to F: G3..max(G3 + 2, F) when F <= G3 + 3, otherwise G3,
G3 + (F - G3) // 3, G3 + 2 (F - G3) // 3 and F.

    uv run python scripts/d51_sites.py results/remote/run-<id> [results/remote/run-<id> ...] --out <file.json>

The same rule picks the sites of the max task (D55; ``--task max``: G is the first read point with both sums present).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

CONTROL_MAX = 0.10


def sites_for(G: int, F: int) -> list[int]:
    if F <= G + 3:
        return list(range(G, max(G + 2, F) + 1))
    return sorted({G, G + (F - G) // 3, G + 2 * (F - G) // 3, F})


def control_for(flips: dict[str, np.ndarray], G: int) -> int | None:
    """The latest read point m <= G - 2 at which every bracket swap's mean flip at the answer token is <= 0.10."""
    for m in range(G - 2, -1, -1):
        if all(f[m] <= CONTROL_MAX for f in flips.values()):
            return m
    return None


def model_sites(varmap_json: Path, varprobe_json: Path) -> dict:
    r = json.loads(varmap_json.read_text())
    ans = len(r["labels"]) - 1
    brackets = [k for k in ("S1", "S2", "S3") if k in r["rates"]]
    count = np.asarray(r["reading"]["count"])[:, ans]
    hits = np.flatnonzero(count == len(brackets))
    if not len(hits):
        return {"G3": None, "reason": "the sums are never all present together at the answer token"}
    G = int(r["read_points"][hits[0]])
    flips = {k: np.asarray(r["rates"][k]["flip"]["mean"])[:, ans] for k in brackets}
    control = control_for(flips, G)
    p = json.loads(varprobe_json.read_text())
    y = np.asarray(p["probes"]["Y"]["accuracy"])
    F = int(np.flatnonzero(y >= 0.9 * y[-1])[0])
    out = {"G3": G, "F": F, "control": control, "sites": sites_for(G, F), "accuracy": r["accuracy"],
           "flips_at_control": {k: round(float(v[control]), 3) for k, v in flips.items()} if control is not None else None,
           "count_at_answer_token": count.tolist()}
    out["all_sites"] = sorted({control, *out["sites"]}) if control is not None else out["sites"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--task", default="three_lane", choices=("three_lane", "max"))
    args = ap.parse_args(argv)
    fam = {"three_lane": "three", "max": "max"}[args.task]
    found: dict[str, dict] = {}
    for root in args.roots:
        for run in sorted(root.glob(f"varmap_{fam}_qwen3_*")):
            size = re.search(r"qwen3_(\d+(?:\.\d+)?b)", run.name).group(1)
            vm = run / "core" / "tasks" / args.task / "varmap.json"
            vp = next(root.glob(f"varprobe_{fam}_qwen3_{size}_*/core/tasks/{args.task}/varprobe.json"), None)
            if not vm.exists() or vp is None:
                found[size] = {"skipped": "no three-bracket map or probe (gate, or not run)"}
                continue
            found[size] = model_sites(vm, vp)
    if not found:
        print("no three-bracket runs found", file=sys.stderr)
        return 1
    for size, s in found.items():
        print(size, {k: v for k, v in s.items() if k != "count_at_answer_token"})
    if args.out:
        args.out.write_text(json.dumps(found, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
