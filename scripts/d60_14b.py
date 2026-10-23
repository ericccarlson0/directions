"""D60: the operator line on the 14B, read by D60's preregistered tests (D57's, D58's and D59's own readings come from
scripts/d57_operator.py, d58_opsteer.py and d59_opset.py).

- From job 1's map: G* (D55's: the both swap at the answer token, mean at least 0.5), and the first read point where
  the operator swap at the answer token alone has a mean of at least 0.5.
- From job 2's token-set swaps (natural arm), D59's pattern on the 4B and 8B as predictions:
  (a) F+S's upper bound below 0.5 at every read point; (b) O+A sufficient at every gap read point; (c) E+A sufficient
  from some gap read point on.
- Does the take-up follow the gather? E+A's first sufficient read point against [G* - 6, G* - 2] (follows the gather)
  and [18, 22] (fixed depth: the 4B and 8B's 20 +/- 2); not discriminating when the two overlap (G* <= 28). Beside
  it, the answer token alone against [G* - 3, G* + 3] and [22, 27].

    uv run python scripts/d60_14b.py --d57 results/remote/run-<job 1> --opset results/remote/run-<job 2> \
        --out docs/varmap/d60_14b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d57_operator import gather_star  # noqa: E402

FIXED_EA = (18, 22)  # the 4B and 8B: E+A first sufficient at 20 (D59, both arms)
FIXED_A = (22, 27)  # the 4B and 8B: the answer token alone first at a mean of 0.5 at 22 and 27 (D57)


def first_mean_at_answer(result: dict, kind: str = "op") -> int | None:
    ans = len(result["labels"]) - 1
    x = np.asarray(result["rates"][kind]["flip"]["mean"])[:, ans]
    idx = np.flatnonzero(x >= 0.5)
    return int(result["read_points"][idx[0]]) if len(idx) else None


def within(x: int | None, lo: int, hi: int) -> bool:
    return x is not None and lo <= x <= hi


def take_up(G_star: int | None, ea: int | None, a: int | None) -> dict:
    if G_star is None:
        return {"reading": "undecided (no G*)"}
    gather, fixed = (G_star - 6, G_star - 2), FIXED_EA
    out = {"G_star": G_star, "EA_first": ea, "A_first": a, "gather_range": gather, "fixed_range": fixed,
           "A_gather_range": (G_star - 3, G_star + 3), "A_fixed_range": FIXED_A}
    if gather[0] <= fixed[1]:
        out["reading"] = "not discriminating (the ranges overlap)"
    elif within(ea, *gather):
        out["reading"] = "follows the gather"
    elif within(ea, *fixed):
        out["reading"] = "fixed depth"
    else:
        out["reading"] = "neither"
    beside = [name for name, ok in (("follows the gather", within(a, G_star - 3, G_star + 3)),
                                    ("fixed depth", within(a, *FIXED_A))) if ok]
    out["A_beside"] = " and ".join(beside) or "neither"
    return out


def predictions(arm: dict, read_points: list[int], threshold: float = 0.5) -> dict:
    rates = arm["rates"]
    lo = {k: np.asarray(r["flip"]["lo"]) for k, r in rates.items()}
    hi = {k: np.asarray(r["flip"]["hi"]) for k, r in rates.items()}
    gap = arm["gap"]
    gi = [read_points.index(m) for m in gap]
    a = bool((hi["FS"] < threshold).all())
    b = bool(gi) and all(lo["OA"][i] >= threshold for i in gi)
    ea_gap = [read_points[i] for i in gi if lo["EA"][i] >= threshold]
    ea_all = [m for i, m in enumerate(read_points) if lo["EA"][i] >= threshold]
    return {"gap": gap, "a_operands_not_involved": a, "FS_max_hi": round(float(hi["FS"].max()), 3),
            "b_two_ends_hold_it": b, "OA_missing_in_gap": [read_points[i] for i in gi if lo["OA"][i] < threshold],
            "c_eq_joins_the_answer": bool(ea_gap), "EA_first_in_gap": ea_gap[0] if ea_gap else None,
            "EA_first": ea_all[0] if ea_all else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d57", type=Path, required=True, help="the directory holding job 1's varmap_maxmin_qwen3_14b_* run")
    ap.add_argument("--opset", type=Path, required=True, help="the directory holding job 2's opset_qwen3_14b_* run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    maps = sorted(args.d57.glob("varmap_maxmin_qwen3_14b*"))
    sets = sorted(args.opset.glob("opset_qwen3_14b*"))
    if not maps or not sets:
        print("need a varmap_maxmin_qwen3_14b run and an opset_qwen3_14b run", file=sys.stderr)
        return 1
    result = json.loads((maps[0] / "core" / "tasks" / "maxmin" / "varmap.json").read_text())
    opset = json.loads((sets[0] / "core" / "opset.json").read_text())
    G_star = gather_star(result)
    a_first = first_mean_at_answer(result)
    out = {"map": {"G_star": G_star, "G": result["operator"]["G"], "R_O": result["operator"]["R_O"],
                   "O_out": result["operator"]["O_out"], "A_first_mean": a_first}, "arms": {}}
    for arm, a in opset["arms"].items():
        if not a.get("rates"):
            continue
        p = predictions(a, opset["read_points"])
        out["arms"][arm] = {**p, "take_up": take_up(G_star, p["EA_first"], a_first if arm == "natural" else None)}
    m = out["map"]
    print(f"qwen3_14b map: G* {m['G_star']} (G {m['G']}), R_O {m['R_O']}, O_out {m['O_out']}; the answer token alone "
          f"first at a mean of 0.5: {m['A_first_mean']}")
    for arm, a in out["arms"].items():
        print(f"  {arm}: gap {a['gap']}; (a) operands not involved {a['a_operands_not_involved']} (F+S upper bound "
              f"at most {a['FS_max_hi']}); (b) O+A holds the gap {a['b_two_ends_hold_it']} (missing at "
              f"{a['OA_missing_in_gap']}); (c) = joins the answer {a['c_eq_joins_the_answer']} (E+A first in the gap "
              f"{a['EA_first_in_gap']})")
        t = a["take_up"]
        print(f"    take-up: E+A first sufficient {t.get('EA_first')} against {t.get('gather_range')} (gather) and "
              f"{t.get('fixed_range')} (fixed) -> {t['reading']}"
              + (f"; the answer token alone {t.get('A_first')} -> {t.get('A_beside')}" if arm == "natural" else ""))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
