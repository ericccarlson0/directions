"""D61: where the max/min operator is combined with the operands, read by D61's preregistered rules.

Per model:
- the bases kept per pair, the four prompts' accuracies, and the checks (the additive state equals the joint state at
  read point 0; the joint patch gives 11's answer);
- the conversion depths K_OS, K_S1S2, K_OD (the first read point where the additive patch's share of 11's answer has
  an upper bound below 0.5), the reading (where the operands are combined / earlier / later / never) and the null;
- the localization at K_OS - 2 to K_OS + 2 (the group whose additive substitution lowers the share most);
- beside them (exploratory): what the additive patch gives from K on, the interaction ratio per group at K, and K_OS
  against D60's landmarks (T2, the operator's arrival at = + answer; G2, the two-token gather).

    uv run python scripts/d61_interact.py results/remote/run-<4B> results/remote/run-<8B> results/remote/run-<14B> \
        --out docs/varmap/d61_interact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import size_of  # noqa: E402

D60 = Path(__file__).resolve().parents[1] / "docs" / "varmap" / "d60_14b.json"


def after_failure(rates: dict, cats: list[str], read_points: list[int], K: int | None) -> dict:
    """From K on: the additive patch's most frequent category at each read point, and the shares at K."""
    if K is None:
        return {}
    i0 = read_points.index(K)
    means = np.array([rates["additive"][c]["mean"] for c in cats])  # (categories, read points)
    top = [cats[j] for j in means[:, i0:].argmax(axis=0)]
    return {"at_K": {c: round(float(means[j, i0]), 3) for j, c in enumerate(cats)},
            "most_frequent_from_K": dict(Counter(top)), "by_read_point": dict(zip(read_points[i0:], top))}


def ratio_by_group(ratio_mean: list, groups: dict, read_points: list[int], m: int | None) -> dict:
    """The mean interaction ratio over each group's tokens at read point ``m``."""
    if m is None or m not in read_points:
        return {}
    r = np.asarray(ratio_mean)[read_points.index(m)]  # (tokens,)
    return {g: round(float(r[idx].mean()), 3) for g, idx in groups.items() if len(idx)}


def read_run(run: Path, landmarks: dict | None = None) -> dict:
    r = json.loads((run / "core" / "interact.json").read_text())
    rps, cats = r["read_points"], r["categories"]
    out: dict = {"read_points": [rps[0], rps[-1]], "pairs": {}}
    for p, a in r["pairs"].items():
        e: dict = {"n": a["n"], "accuracy": a["accuracy"], "K": a.get("K"), "holds_at_first": a.get("holds_at_first")}
        if "rates" in a:
            joint = np.asarray(a["rates"]["joint"]["11"]["mean"])
            add = np.asarray(a["rates"]["additive"]["11"]["mean"])
            e.update({"additive_equals_joint_at_0": a.get("additive_equals_joint_at_0"),
                      "joint_min": round(float(joint.min()), 3),
                      "additive_11": {m: round(float(x), 3) for m, x in zip(rps, add)},
                      "after_failure": after_failure(a["rates"], cats, rps, a.get("K")),
                      "ratio_at_K": ratio_by_group(a["ratio_mean"], a["groups"], rps, a.get("K"))})
        out["pairs"][p] = e
    rd = r["reading"]
    out["reading"] = {"K": rd["K"], "control_combined": rd["control_combined"], "null_clear": rd["null_clear"],
                      "localization": {m: v["lowest"] for m, v in rd["localization"].items()},
                      "localization_shares": rd["localization"]}
    k_os = rd["K"].get("OS")
    if landmarks:
        out["landmarks"] = {**landmarks, "K_OS_minus_T2": None if k_os is None or landmarks.get("T2") is None else k_os - landmarks["T2"],
                            "K_OS_minus_G2": None if k_os is None or landmarks.get("G2") is None else k_os - landmarks["G2"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="directories holding interact_* runs")
    ap.add_argument("--d60", type=Path, default=D60, help="D60's summary (the two-site landmarks)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = {}
    for root in args.roots:
        for run in sorted(root.glob("interact_qwen3_*")):
            if (run / "core" / "interact.json").exists() and size_of(run.name):
                runs[size_of(run.name)] = run
    if not runs:
        print("no interact runs found", file=sys.stderr)
        return 1
    two_site = json.loads(args.d60.read_text())["two_site"]["per_model"] if args.d60.exists() else {}
    summary = {}
    for size in sorted(runs, key=lambda x: float(x[:-1])):
        lm = two_site.get(size)
        s = read_run(runs[size], {"T2": lm["T2"], "G2": lm["G2"]} if lm else None)
        summary[f"qwen3_{size}"] = s
        rd = s["reading"]
        print(f"qwen3_{size}: K {rd['K']} -> the control is combined {rd['control_combined']}; null clear {rd['null_clear']}")
        print(f"  localization {rd['localization']}")
        if "landmarks" in s:
            print(f"  landmarks {s['landmarks']}")
        for p, e in s["pairs"].items():
            print(f"  {p}: {e['n']} bases, accuracy {e['accuracy']}; additive = joint at 0 on {e.get('additive_equals_joint_at_0')}, "
                  f"joint's lowest share {e.get('joint_min')}; held at the first read point {e['holds_at_first']}")
            if e.get("after_failure"):
                print(f"    from K: {e['after_failure']['most_frequent_from_K']}; at K {e['after_failure']['at_K']}; "
                      f"ratio at K {e['ratio_at_K']}")
    if len(summary) > 1:
        summary["across"] = {"control_combined": {k: v["reading"]["control_combined"] for k, v in summary.items()},
                             "K_OS": {k: v["reading"]["K"].get("OS") for k, v in summary.items()}}
        print(f"across: {summary['across']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
