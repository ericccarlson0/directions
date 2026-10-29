"""D62: the task screen read by its preregistered rule.

Per model, form and number of demonstrations: accuracy with its interval, the gaps, the errors and whether the form
passes. Then the rule: the full test's task is the first of if_mixed, if_gt, ternary, python that passes on both the 8B
and the 14B at 8 demonstrations (else at 16; else none). Beside it, the models it passes on (the 4B joins where it
passes), the given-condition arm (if_given), the token-order arm (python, where it passes besides the selected form),
and the harder rung (if_sum, reported, not selected).

    uv run python scripts/d62_screen.py results/remote/run-<4B> results/remote/run-<8B> results/remote/run-<14B> \
        --out docs/screen/d62_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import size_of  # noqa: E402

SELECTION_ORDER = ("if_mixed", "if_gt", "ternary", "python")  # as src/directions/screen.py (tests/test_d62.py)
SHOTS = ("8", "16")


def select(passed: dict[str, dict[str, dict[str, bool]]]) -> dict:
    """D62's rule on ``passed[size][form][shots]``."""
    def ok(size: str, form: str, k: str) -> bool:
        return bool(passed.get(size, {}).get(form, {}).get(k, False))

    for k in SHOTS:
        for form in SELECTION_ORDER:
            if ok("8b", form, k) and ok("14b", form, k):
                sizes = [s for s in sorted(passed, key=lambda x: float(x[:-1])) if ok(s, form, k)]
                return {"shots": int(k), "task": form, "models": sizes,
                        "given_arm": [s for s in sizes if ok(s, "if_given", k)],
                        "order_arm": [] if form == "python" else [s for s in sizes if ok(s, "python", k)],
                        "if_sum_passes": [s for s in sorted(passed, key=lambda x: float(x[:-1])) if ok(s, "if_sum", k)]}
    return {"shots": None, "task": None, "models": [], "given_arm": [], "order_arm": [], "if_sum_passes": []}


def read_run(run: Path) -> dict:
    s = json.loads((run / "core" / "screen.json").read_text())
    out = {}
    for form, f in s["forms"].items():
        for k, m in f.items():
            if not k.isdigit():
                continue
            out.setdefault(form, {})[k] = {
                "accuracy": [round(m["accuracy"][x], 3) for x in ("mean", "lo", "hi")],
                "gaps": {g: [round(v[x], 3) for x in ("mean", "lo", "hi")] for g, v in m["gaps"].items()},
                "cells": {c: round(v["mean"], 3) for c, v in m["cells"].items()},
                "errors": m["errors"], "aligned": m["aligned"], "passes": m["passes"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="directories holding screen_qwen3_* runs")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = {}
    for root in args.roots:
        for run in sorted(root.glob("screen_qwen3_*")):
            if (run / "core" / "screen.json").exists() and size_of(run.name):
                runs[size_of(run.name)] = run
    if not runs:
        print("no screen runs found", file=sys.stderr)
        return 1
    table = {size: read_run(runs[size]) for size in sorted(runs, key=lambda x: float(x[:-1]))}
    for size, forms in table.items():
        for form, by_k in forms.items():
            for k, m in by_k.items():
                print(f"qwen3_{size} {form:9s} {k:>2s} shots: accuracy {m['accuracy']}, gaps {m['gaps']}, errors "
                      f"{m['errors']}, aligned {m['aligned']} -> {'pass' if m['passes']['pass'] else 'fail'}")
    passed = {size: {form: {k: m["passes"]["pass"] for k, m in by_k.items()} for form, by_k in forms.items()}
              for size, forms in table.items()}
    sel = select(passed)
    print(f"selection: {sel}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"table": {f"qwen3_{s}": v for s, v in table.items()}, "selection": sel}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
