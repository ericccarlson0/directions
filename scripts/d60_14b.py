"""D60: the operator line on the 14B, read by D60's preregistered tests (D57's, D58's and D59's own readings come from
scripts/d57_operator.py, d58_opsteer.py and d59_opset.py).

- From job 1's map: G* (D55's: the both swap at the answer token, mean at least 0.5), and the first read point where
  the operator swap at the answer token alone has a mean of at least 0.5.
- From job 2's token-set swaps (natural arm), D59's pattern on the 4B and 8B as predictions:
  (a) F+S's upper bound below 0.5 at every read point; (b) O+A sufficient at every gap read point; (c) E+A sufficient
  from some gap read point on.
- Does the take-up follow the gather? As preregistered: E+A's first sufficient read point against [G* - 6, G* - 2]
  (follows the gather) and [18, 22] (fixed depth); its premise, a G* below the output, failed on max/min (D60's
  amendment). As amended: both arrivals at E+A on all three models, from the sums arm (G2, the both swap's first
  mean of 0.5) and the natural arm (T2, E+A's first sufficient read point); ΔG and ΔT the 14B's minus the mean of the
  4B's and 8B's; follows the gather if |ΔT - ΔG| <= 2, fixed depth if |ΔT| <= 2, not discriminating if ΔG < 3 or
  both hold. Beside it, the same at the answer token alone.

    uv run python scripts/d60_14b.py --d57 results/remote/run-<job 1> --opset results/remote/run-<job 2> \
        --others results/remote/run-<D59 4B> results/remote/run-<D59 8B> results/remote/run-<sums 4B> \
        results/remote/run-<sums 8B> --out docs/varmap/d60_14b.json
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


def first_half(arm: dict, key: str, read_points: list[int], bound: str = "mean") -> int | None:
    """The first read point where set ``key``'s flip rate (its mean, or its lower bound) is at least 0.5."""
    x = np.asarray(arm["rates"][key]["flip"][bound])
    idx = np.flatnonzero(x >= 0.5)
    return int(read_points[idx[0]]) if len(idx) else None


def two_site(models: dict[str, dict]) -> dict:
    """D60's amended test: per model, T2 (E+A first sufficient for the operator) and G2 (the both swap at E+A first at a
    mean of 0.5); the 14B's shifts against the mean of the 4B's and 8B's. The same at the answer token alone beside."""
    out: dict = {"per_model": models}
    for site, t_key, g_key in (("E+A", "T2", "G2"), ("answer token", "T1", "G1")):
        ref = [models[m] for m in ("4b", "8b") if m in models]
        big = models.get("14b")
        if big is None or len(ref) < 2 or any(r[t_key] is None or r[g_key] is None for r in ref + [big]):
            out[site] = {"reading": "undecided (a model's arrival is missing)"}
            continue
        dG = big[g_key] - np.mean([r[g_key] for r in ref])
        dT = big[t_key] - np.mean([r[t_key] for r in ref])
        fg, fd = abs(dT - dG) <= 2, abs(dT) <= 2
        if dG < 3:
            reading = "not discriminating (the gathers differ by less than 3)"
        elif fg and fd:
            reading = "not discriminating (both hold)"
        else:
            reading = "follows the gather" if fg else "fixed depth" if fd else "neither"
        out[site] = {"dG": float(dG), "dT": float(dT), "reading": reading}
    return out


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
    ap.add_argument("--others", type=Path, nargs="*", default=[],
                    help="directories holding the 4B's and 8B's opset_* (D59) and opset_sums_* (D60) runs")
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
        if arm == "sums" or not a.get("rates"):  # the predictions are about the operator's arms
            continue
        p = predictions(a, opset["read_points"])
        out["arms"][arm] = {**p, "take_up": take_up(G_star, p["EA_first"], a_first if arm == "natural" else None)}
    # the amended test: T2 and G2 per model, from the natural and sums arms of the token-set runs
    arms_by_model: dict[str, dict[str, tuple[dict, list[int]]]] = {"14b": {a: (r, opset["read_points"]) for a, r in opset["arms"].items()}}
    for root in args.others:
        for run in sorted(root.glob("opset*_qwen3_*")):
            size = next((x for x in ("4b", "8b") if f"_qwen3_{x}_" in f"{run.name}_"), None)
            if size and (run / "core" / "opset.json").exists():
                r = json.loads((run / "core" / "opset.json").read_text())
                for a, arm in r["arms"].items():
                    arms_by_model.setdefault(size, {})[a] = (arm, r["read_points"])
    models = {}
    for size, arms in arms_by_model.items():
        if "natural" not in arms or "sums" not in arms or not arms["natural"][0].get("rates") or not arms["sums"][0].get("rates"):
            continue
        (nat, nrp), (sm, srp) = arms["natural"], arms["sums"]
        models[size] = {"T2": first_half(nat, "EA", nrp, "lo"), "G2": first_half(sm, "EA", srp),
                        "T1": first_half(nat, "A", nrp), "G1": first_half(sm, "A", srp)}
        models[size]["lag2"] = (None if None in (models[size]["T2"], models[size]["G2"])
                                else models[size]["T2"] - models[size]["G2"])
    out["two_site"] = two_site(models)
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
    ts = out["two_site"]
    print("amended take-up test (D60): per model " + "; ".join(f"{k}: T2 {v['T2']}, G2 {v['G2']}, lag {v['lag2']}, "
                                                             f"T1 {v['T1']}, G1 {v['G1']}" for k, v in ts["per_model"].items()))
    for site in ("E+A", "answer token"):
        print(f"  at {site}: {ts[site]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
