"""The D47 block split, summarised (descriptive; the preregistered readouts are in each run's varsplit.json).

Per model and task: for S1 given Y and for Y, each block's gain in probe accuracy, attention's share of it (with
its bootstrap interval) and the D47 reading; then, per swapped block, the head reading (lanes, one channel, mixed,
no carrier), the carriers of each bracket, and the all-heads transfer beside the sum of the single-head transfers.

    uv run python scripts/varsplit_summary.py results/remote/run-<id> --out docs/varsplit/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _r(x, nd: int = 2):
    """Rounded, or None where the bootstrap had no gain to divide by (stored as null)."""
    return None if x is None else round(x, nd)


def summarise(r: dict) -> dict:
    out: dict = {"gather": r.get("gather")}
    p = r["probe"]
    if "skipped" in p and "skipped" in r["heads"]:
        return out | {"skipped": p}
    out["probe"] = {}
    if "skipped" not in p:
        out["accuracy"], out["n_kept"], out["additivity"] = p["accuracy"], p["n_kept"], p["additivity"]
    for t in ("S1|Y", "Y") if "skipped" not in p else ():
        if t not in p["targets"]:
            continue
        st = p["targets"][t]["states"]
        out["probe"][t] = {
            "accuracy": dict(zip(p["states"], [round(x, 3) for x in st["accuracy"]])),
            "blocks": {b: {"gain": _r(s["gain"], 3), "share": _r(s["share"]), "ci": [_r(s["lo"]), _r(s["hi"])],
                           "reading": s["reading"]} for b, s in p["targets"][t]["shares"].items()},
        }
    h = r["heads"]
    if "skipped" in h:
        return out | {"heads": h}
    out["heads"] = {}
    for b, blk in h["blocks"].items():
        rd = blk["reading"]
        out["heads"][b] = {"reading": rd.get("reading"), "carries_S1": rd.get("carries_S1"), "carries_S2": rd.get("carries_S2"),
                           "S1_only": rd.get("S1_only"), "S2_only": rd.get("S2_only"),
                           "all_heads": {k: round(v, 2) for k, v in blk["all_heads"].items()},
                           "sum_of_heads": {k: round(v, 2) for k, v in blk["sum_of_heads"].items()},
                           "n": {k: v.get("n") for k, v in blk["types"].items()}}
        for k in ("S1", "S2"):
            if f"{k}_carriers_holding_a_digit" in rd:
                out["heads"][b][f"{k}_carriers_holding_a_digit"] = rd[f"{k}_carriers_holding_a_digit"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    summary: dict = {}
    for run in sorted(args.root.glob("varsplit_*")):
        for task_dir in sorted((run / "core" / "tasks").glob("*")):
            summary.setdefault(run.name, {})[task_dir.name] = summarise(json.loads((task_dir / "varsplit.json").read_text()))
    if not summary:
        print(f"no block-split runs under {args.root}", file=sys.stderr)
        return 1
    for run, tasks in summary.items():
        for task, s in tasks.items():
            if "skipped" in s:
                print(f"{run} {task}: skipped")
                continue
            if "accuracy" in s:
                print(f"{run} {task}: accuracy {s['accuracy']:.3f}, n {s['n_kept']}, gather {s['gather']}, "
                      f"additivity max {s['additivity']['max']:.4f}")
            else:
                print(f"{run} {task}: heads only, gather {s['gather']}")
            for t, v in s["probe"].items():
                print(f"   {t:5s} " + "  ".join(f"{k} {a:.2f}" for k, a in v["accuracy"].items()))
                for b, x in v["blocks"].items():
                    if x["reading"] != "no gain":
                        print(f"         block {b}: gain {x['gain']:.2f}, attention share {x['share']:.2f} {x['ci']} -> {x['reading']}")
            for b, x in s.get("heads", {}).items():
                print(f"   heads of block {b}: {x['reading']}; S1 {x['carries_S1']} (only {x['S1_only']}), S2 {x['carries_S2']} "
                      f"(only {x['S2_only']}); all heads {x['all_heads']}, sum of heads {x['sum_of_heads']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
