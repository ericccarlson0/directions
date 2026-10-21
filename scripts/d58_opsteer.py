"""D58 (D57 step 3): the operator's natural direction at the operator token, read by D58's preregistered rules.

Per model:
- The direction: its norm, its ratio to the max state's norm and the pairs' mean cosine with it, at a few read points.
- The switch curve: W_steer against W_swap, the null's largest switch upper bound, the switch rates at the injection
  read point, and D58's reading of the direction.
- The steered map: accuracy, bases per type, and the added operator's R_O, O_out, G and route. Each is compared with
  step 1's natural value from the same model's maxmin map, by lower bounds (D45) and by means, one comparison at a
  time: at most 2 read points apart is "as natural", 3 or more "later" or "earlier".
- Secondary (exploratory): step 1's natural readouts on its min bases only (the unsteer swap goes from min to max).

    uv run python scripts/d58_opsteer.py --opsteer <dirs holding opsteer_* runs> --d57 <dirs holding varmap_maxmin_* runs> \
        --out docs/varmap/d58_opsteer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import size_of  # noqa: E402

from directions.opsteer import mean_readouts  # noqa: E402
from directions.varmap import bootstrap_rate, operator_readouts  # noqa: E402

SEED, N_BOOT, ALPHA, PRESENT = 20260916, 1000, 0.05, 0.5
SHOW = (0, 6, 11, 16, 20, 24, 28)  # read points of the direction's diagnostics


def compare(steered: int | None, natural: int | None) -> dict:
    if steered is None or natural is None:
        return {"steered": steered, "natural": natural, "difference": None, "reading": "undecided"}
    d = steered - natural
    return {"steered": steered, "natural": natural, "difference": d,
            "reading": "as natural" if abs(d) <= 2 else "later" if d > 0 else "earlier"}


def natural_min_only(task_dir: Path) -> dict:
    """Step 1's operator readouts on its min bases only (exploratory), by lower bounds and by means."""
    r = json.loads((task_dir / "varmap.json").read_text())
    arrays = dict(np.load(task_dir / "varmap_arrays.npz"))
    labels, rps = r["labels"], r["read_points"]
    rates, rng = {}, np.random.default_rng(SEED)
    for k in ("op", "S1", "S2"):
        bases = arrays[f"{k}__bases"]
        rows = [i for i, b in enumerate(bases) if r["examples"][int(b)]["base"].startswith("min")]
        rates[k] = {"n": len(rows), "flip": bootstrap_rate(arrays[f"{k}__flip"][rows], rng, N_BOOT, ALPHA)}
    ans = len(labels) - 1
    present = {k: np.asarray(rates[k]["flip"]["lo"]) >= PRESENT for k in rates}
    count = present["S1"].astype(int) + present["S2"].astype(int)
    read = {"present_O": present["op"], "count": count}
    out = operator_readouts(read, labels, rps)
    return {"n": {k: v["n"] for k, v in rates.items()}, "lower_bounds": out,
            "means": mean_readouts(rates, labels, rps, 0), "answer_token_op_mean": np.asarray(rates["op"]["flip"]["mean"])[:, ans].round(3).tolist()}


def read_model(run: Path, d57_task: Path | None) -> dict:
    r = json.loads((run / "core" / "opsteer.json").read_text())
    d, c, m = r["direction"], r["curve"], r["map"]
    m0 = r["inject_read_point"]
    out: dict = {"inject_read_point": m0}
    out["direction"] = {rp: {"norm": round(d["norm"][rp], 3), "ratio": round(d["ratio"][rp], 4),
                             "pair_cosine": round(d["pair_cosine"][rp], 3)} for rp in SHOW if rp < len(d["norm"])}
    arms = c["arms"]
    W_s, W_w, null_hi = c["W_steer"], c["W_swap"], c["null_switch_max_hi"]
    # D58: "the null's switch rate at most 0.05 everywhere": the rate's point estimate; its upper bound is reported beside
    nulls = [a for a in arms if a.startswith("null")]
    null_rate = max((float(np.max(arms[a]["switch"]["mean"])) for a in nulls), default=None)
    if W_s is not None and W_w is not None and W_s >= W_w - 1 and (null_rate is None or null_rate <= 0.05):
        direction_reading = "the operator at its token is one direction for as long as the token holds it"
    elif W_s is not None and W_w is not None and W_s < W_w - 1:
        direction_reading = "a direction only early"
    else:
        direction_reading = "undecided (no window, or the null switches)"
    out["curve"] = {"n": c["n"], "W_steer": W_s, "W_swap": W_w, "null_switch_max": null_rate, "null_switch_max_hi": null_hi,
                    "reading": direction_reading,
                    "switch_at_inject": {a: round(arms[a]["switch"]["mean"][m0], 3) for a in ("steer", "swap")},
                    "swap_minus_steer_at_inject": [round(c["swap_minus_steer"][q][m0], 3) for q in ("mean", "lo", "hi")]}
    out["map"] = {"accuracy": m["accuracy"], "unsteered_max": m["unsteered_max"], "natural_min": m["natural_min"],
                  "n_kept": m["n_kept"]}
    if "operator" not in m or m["operator"] is None:
        out["map"]["readouts"] = None
        return out
    op, means = m["operator"], m["operator_means"]
    out["map"]["readouts"] = {"lower_bounds": op, "means": means}
    if d57_task is not None:
        nat = json.loads((d57_task / "varmap.json").read_text())
        nat_lb = nat["operator"]
        nat_means = mean_readouts(nat["rates"], nat["labels"], nat["read_points"], 0)
        out["comparison"] = {
            "lower_bounds": {k: compare(op[k], nat_lb[k]) for k in ("R_O", "O_out", "G")},
            "means": {k: compare(means[k], nat_means[k]) for k in ("R_O", "O_out", "G")},
            "route": {"steered": op["route"], "natural": nat_lb["route"]},
        }
        out["natural_min_only"] = natural_min_only(d57_task)
    return out


def find(roots: list[Path], pattern: str, rel: str) -> dict[str, Path]:
    found = {}
    for root in roots:
        for run in sorted(root.glob(pattern)):
            if (run / rel).exists() and size_of(run.name):
                found[size_of(run.name)] = run
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opsteer", type=Path, nargs="+", required=True)
    ap.add_argument("--d57", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = find(args.opsteer, "opsteer_*", "core/opsteer.json")
    maps = {k: v / "core" / "tasks" / "maxmin" for k, v in find(args.d57, "varmap_maxmin_*", "core/tasks/maxmin/varmap.json").items()}
    if not runs:
        print("no opsteer runs found", file=sys.stderr)
        return 1
    summary = {}
    for size in sorted(runs, key=lambda x: float(x[:-1])):
        s = read_model(runs[size], maps.get(size))
        summary[f"qwen3_{size}"] = s
        c, m = s["curve"], s["map"]
        print(f"qwen3_{size}: direction at read point {s['inject_read_point']}: {s['direction'].get(s['inject_read_point'])}")
        print(f"  curve ({c['n']} bases): W_steer {c['W_steer']}, W_swap {c['W_swap']}, null switch at most {c['null_switch_max']} "
              f"(upper bound {c['null_switch_max_hi']}); "
              f"switch at injection {c['switch_at_inject']} -> {c['reading']}")
        print(f"  map: steered accuracy {m['accuracy']:.3f} (unsteered max {m['unsteered_max']:.3f}, natural min "
              f"{m['natural_min']:.3f}); kept {m['n_kept']}")
        if m["readouts"]:
            print(f"  added operator: {m['readouts']['lower_bounds']}; by the means {m['readouts']['means']}")
        for basis, comp in s.get("comparison", {}).items():
            if basis != "route":
                print(f"  against step 1 ({basis}): " + "; ".join(f"{k} {v['steered']} vs {v['natural']} ({v['reading']})"
                                                               for k, v in comp.items()))
        if "natural_min_only" in s:
            nm = s["natural_min_only"]
            print(f"  step 1 on its min bases only (exploratory, {nm['n']}): {nm['lower_bounds']['reading']}; "
                  f"R_O {nm['lower_bounds']['R_O']}, O_out {nm['lower_bounds']['O_out']}; by the means {nm['means']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
