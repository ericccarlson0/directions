"""The D52 position-set checks, summarised (descriptive; the preregistered readouts are each run's posset.json).

Per model: per source, the rate of each set at the read points before the gather and from the gather to F (their
ranges), the non-digit set's and answer + ='s paired gains over the answer token, and D52's decision (the joint
DAS's positions and sites).

    uv run python scripts/posset_summary.py results/remote/run-<id> [results/remote/run-<id> ...] \
        --out docs/posset/summary.json --figures docs/posset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

SETS = ("answer", "answer_eq", "non_digit", "answer_digits", "query")


def _range(x: np.ndarray) -> list[float]:
    return [round(float(np.min(x)), 3), round(float(np.max(x)), 3)] if len(x) else []


def summarise(r: dict) -> dict:
    rps = np.asarray(r["read_points"])
    G3 = r["marks"].get("G3")
    pre = rps < G3 if G3 is not None else np.zeros(len(rps), dtype=bool)
    out = {"accuracy": r["accuracy"], "n_kept": r["n_kept"], "read_points": r["read_points"], "marks": r["marks"],
           "decision": {k: v for k, v in r["decision"].items() if k != "passing"},
           "n_passing": len(r["decision"].get("passing", [])), "sources": {}}
    for k, rates in r["rates"].items():
        s = {"n": rates["n"]}
        for name in SETS:
            m = np.asarray(rates[name]["flip"]["mean"])
            s[name] = {"before_G3": _range(m[pre]), "G3_to_F": _range(m[~pre])}
        for name in ("non_digit", "answer_eq", "answer_digits"):
            g = r["gains"][k][name]
            s[f"gain_{name}"] = {"mean": np.round(g["mean"], 3).tolist(), "lo": np.round(g["lo"], 3).tolist()}
        out["sources"][k] = s
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--figures", type=Path)
    args = ap.parse_args(argv)
    found: dict[str, dict] = {}
    for root in args.roots:
        for run in sorted(root.glob("posset_*")):
            for task_dir in sorted((run / "core" / "tasks").glob("*")):
                r = json.loads((task_dir / "posset.json").read_text())
                size = re.search(r"qwen3_(\d+(?:\.\d+)?b)", run.name)
                key = f"qwen3_{size.group(1)}" if size else run.name
                found[f"{key}/{task_dir.name}"] = summarise(r)
                fig = run / "figures" / f"{task_dir.name}.png"
                if args.figures and fig.exists():
                    args.figures.mkdir(parents=True, exist_ok=True)
                    shutil.copy(fig, args.figures / f"{key}_{task_dir.name}.png")
    if not found:
        print("no position-set runs found", file=sys.stderr)
        return 1
    order = sorted(found, key=lambda k: float(re.search(r"(\d+(?:\.\d+)?)b", k).group(1)) if re.search(r"(\d+(?:\.\d+)?)b", k) else 0)
    for key in order:
        s = found[key]
        print(f"{key}: accuracy {s['accuracy']:.3f}, kept {s['n_kept']}, marks {s['marks']}; decision {s['decision']} "
              f"({s['n_passing']} passing pairs)")
        for k, v in s["sources"].items():
            print(f"   {k:4s} n {v['n']}: " + "; ".join(f"{name} {v[name]['before_G3']} -> {v[name]['G3_to_F']}" for name in SETS))
            print(f"        non-digit gain {v['gain_non_digit']['mean']} (lo {v['gain_non_digit']['lo']}); "
                  f"answer + = gain {v['gain_answer_eq']['mean']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({k: found[k] for k in order}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
