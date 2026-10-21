"""D57 step 1: the operator map on max/min, read by D57's preregistered rules.

Per model: the few-shot accuracy, overall and per operator; R_O (the first read point with the operator present at
the answer token), G (the gather on these bases), O_out (the last read point with the operator present at the
operator token), the route (other tokens where it is present) and D57's reading, as run and with the fragile rows
dropped (`scripts/varmap_summary.py`'s re-read). Beside them, the single-control hand-over (D43, D44) and the
max-only gather (D55). From the probe: the read points at which O|Y, C|Y, S1|Y and S2|Y are carried, and whether
O|Y is carried at every read point from G on (the operator still represented once the answer has formed).

    uv run python scripts/d57_operator.py results/remote/run-<1.7B and 4B> results/remote/run-<8B> \
        --out docs/varmap/d57_operator.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d54_compare import size_of  # noqa: E402
from varmap_summary import HANDOVER, digit_token_ids, fragile_rows, rates_from_arrays  # noqa: E402

from directions.varmap import operator_readouts, reading  # noqa: E402

GATHER_MAX = {"1.7b": 21, "4b": 24, "8b": 24, "14b": 30}  # D55: runs 39178177443, 39178188546, 39252850362
PROBE_TARGETS = ("O|Y", "C|Y", "S1|Y", "S2|Y")


def read_map(task_dir: Path) -> dict:
    result = json.loads((task_dir / "varmap.json").read_text())
    arrays = dict(np.load(task_dir / "varmap_arrays.npz"))
    th, labels, rps = result["thresholds"], result["labels"], result["read_points"]
    ids = digit_token_ids(result, arrays)
    drop = {k: fragile_rows(result, arrays, k, ids) for k in result["rates"]}
    rec = rates_from_arrays(result, arrays, ids, drop)
    dropped = operator_readouts(reading(rec, "maxmin", th["present_flip"], th["value_change"]), labels, rps)
    return {"accuracy": result["accuracy"], "accuracy_by_op": result["accuracy_by_op"], "n_kept": result["n_kept"],
            "labels": labels, "as_run": result["operator"], "fragile_rows": {k: int(len(v)) for k, v in drop.items()},
            "fragile_dropped": dropped}


def read_probe(task_dir: Path, G: int | None) -> dict:
    r = json.loads((task_dir / "varprobe.json").read_text())
    rps = r["read_points"]
    out = {"accuracy": r["accuracy"], "n_kept": r["n_kept"]}
    for t in PROBE_TARGETS:
        carried = np.asarray(r["probes"][t]["carried"], dtype=bool)
        out[t] = [int(rps[m]) for m in np.flatnonzero(carried)]
    o = np.asarray(r["probes"]["O|Y"]["carried"], dtype=bool)
    out["O|Y_carried_from_G_on"] = bool(G is not None and o[rps.index(G):].all())
    return out


def find(roots: list[Path], stage: str) -> dict[str, Path]:
    found = {}
    for root in roots:
        for run in sorted(root.glob(f"{stage}_maxmin_*")):
            task_dir = run / "core" / "tasks" / "maxmin"
            if (task_dir / f"{stage}.json").exists() and size_of(run.name):
                found[size_of(run.name)] = task_dir
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="directories holding varmap_maxmin_* and varprobe_maxmin_* runs")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    maps, probes = find(args.roots, "varmap"), find(args.roots, "varprobe")
    if not maps:
        print("no maxmin variable maps found", file=sys.stderr)
        return 1
    summary = {}
    for size in sorted(maps, key=lambda x: float(x[:-1])):
        m = read_map(maps[size])
        a, f = m["as_run"], m["fragile_dropped"]
        hand = HANDOVER.get(f"qwen3_{size}")
        p = read_probe(probes[size], a["G"]) if size in probes else None
        summary[f"qwen3_{size}"] = {"map": m, "probe": p, "handover": hand[0] if hand else None,
                                    "gather_max_d55": GATHER_MAX.get(size)}
        acc = ", ".join(f"{k} {v:.2f}" for k, v in m["accuracy_by_op"].items())
        print(f"qwen3_{size}: accuracy {m['accuracy']:.2f} ({acc}); kept {m['n_kept']}")
        print(f"  as run:          R_O {a['R_O']}, G {a['G']}, O_out {a['O_out']}, route {a['route']} -> {a['reading']}")
        print(f"  fragile dropped: R_O {f['R_O']}, G {f['G']}, O_out {f['O_out']}, route {f['route']} -> {f['reading']}"
              f"  ({sum(m['fragile_rows'].values())} rows)")
        print(f"  beside: hand-over {hand[0] if hand else None}, max-only gather (D55) {GATHER_MAX.get(size)}")
        if p:
            print(f"  probe (accuracy {p['accuracy']:.2f}, kept {p['n_kept']}): carried at "
                  + "; ".join(f"{t} {p[t][:1] + (['..', p[t][-1]] if len(p[t]) > 1 else [])}" for t in PROBE_TARGETS)
                  + f"; O|Y carried from G on: {p['O|Y_carried_from_G_on']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
