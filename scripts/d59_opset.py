"""D59: where the operator sits between its token and the answer token, read by D59's preregistered rules.

Per model and arm (natural: D57's operator swap; added: D58's direction):
- the bases kept and the whole-query check;
- the gap (read points where neither the operator token nor the answer token alone is sufficient), the class at
  each gap read point (operands / structure / either side / only together / undecided) and the gap's reading
  (the majority class);
- beside them: where O+A is sufficient, each group's necessity within the gap, and the minimal sufficient sets at
  each gap read point;
- natural against added (exploratory): the gap read points where both arms have a class, and whether the classes
  agree.

    uv run python scripts/d59_opset.py results/remote/run-<4B> results/remote/run-<8B> --out docs/varmap/d59_opset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import size_of  # noqa: E402


def read_run(run: Path) -> dict:
    r = json.loads((run / "core" / "opset.json").read_text())
    out = {}
    for arm, a in r["arms"].items():
        gap = a["gap"]
        out[arm] = {"n": a["n"], "accuracy": a["accuracy"], "query_check": a.get("query_check"), "gap": gap,
                    "classes": {int(k): v for k, v in a["classes"].items()}, "reading": a["reading"],
                    "counts": a["counts"], "ends_sufficient": a["ends_sufficient"],
                    "necessary_in_gap": {g: [m for m in ms if m in gap] for g, ms in a["necessity"].items()},
                    "minimal_sufficient_in_gap": {int(m): s for m, s in a["minimal_sufficient"].items() if int(m) in gap}}
    if "natural" in out and "added" in out:
        both = sorted(set(out["natural"]["classes"]) & set(out["added"]["classes"]))
        out["natural_vs_added"] = {"read_points": both,
                                   "agree": [m for m in both if out["natural"]["classes"][m] == out["added"]["classes"][m]]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="directories holding opset_* runs")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = {}
    for root in args.roots:
        for run in sorted(root.glob("opset_qwen3_*")):  # D59's runs, not D60's opset_sums_* (one arm only)
            if (run / "core" / "opset.json").exists() and size_of(run.name):
                runs[size_of(run.name)] = run
    if not runs:
        print("no opset runs found", file=sys.stderr)
        return 1
    summary = {}
    for size in sorted(runs, key=lambda x: float(x[:-1])):
        s = read_run(runs[size])
        summary[f"qwen3_{size}"] = s
        for arm in ("natural", "added"):
            if arm not in s:
                continue
            a = s[arm]
            print(f"qwen3_{size} {arm} ({a['n']} bases, whole-query check {a['query_check']}): gap {a['gap']} -> {a['reading']} "
                  f"{a['counts']}")
            print(f"  classes {a['classes']}; O+A sufficient at {a['ends_sufficient']}")
            print(f"  necessary in the gap: { {g: ms for g, ms in a['necessary_in_gap'].items() if ms} }")
            print(f"  minimal sufficient sets: {a['minimal_sufficient_in_gap']}")
        if "natural_vs_added" in s:
            c = s["natural_vs_added"]
            print(f"  natural vs added: classes agree at {len(c['agree'])} of {len(c['read_points'])} shared gap read points")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
