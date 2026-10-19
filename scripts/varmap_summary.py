"""The D45 variable maps, summarised (descriptive; the preregistered readouts are the maps themselves).

Per model and task: the read point from which both sub-results are present at the answer token (the *gather*), the
last read point at which a same-value swap at a digit token still changes the answer in most bases, whether either
sub-result is ever present at a token other than the digits and the answer token, and how often the swap from a
source changing both brackets gives the source's answer at the gather. Then the same reading with the *fragile*
bases dropped: those whose no-op swap (the first query token, identical in base and source because both share
their demonstrations) does not reproduce the gated base answer at some read point, i.e. bases whose answer moves
with the batch's padding in bf16.

    uv run python scripts/varmap_summary.py results/remote/run-38502381471 --out docs/varmap/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from directions.varmap import Expr, bootstrap_rate, prediction, reading

N_BOOT, ALPHA, SEED = 1000, 0.05, 20260916
# the single-control hand-over (D43 vector, median read point) and the universal heads' median write depth, from the
# D44 entry of STATUS.md; the figure marks them beside the gather
HANDOVER = {"qwen3_0.6b": (20, 19.0), "qwen3_1.7b": (20, 18.5), "qwen3_4b": (24, 23.0), "qwen3_8b": (26, 23.0)}



def by_size(path: Path) -> tuple:
    """Run directories in model-size order (``..._qwen3_14b_...`` after ``..._qwen3_8b_...``), then by name."""
    m = re.search(r"_(\d+(?:\.\d+)?)b(?:_|$)", path.name)
    return (float(m.group(1)) if m else float("inf"), path.name)

def parse_expr(text: str) -> Expr:
    """``(a+b) - (c+d)``, ``(a+b) - c``, ``(a+b) - (c+d) - (e+f)`` or ``max(a+b, c+d)`` as written by :meth:`Expr.text`."""
    return Expr(*[int(x) for x in re.findall(r"\d", text)], op="max" if text.startswith("max") else "-")


def digit_token_ids(result: dict, arrays: dict[str, np.ndarray]) -> dict[int, int]:
    """digit -> token id, read off the swaps that produced their predicted digit (no tokenizer needed)."""
    ids: dict[int, int] = {}
    for k in result["rates"]:
        flip, am, bases = arrays[f"{k}__flip"], arrays[f"{k}__argmax"], arrays[f"{k}__bases"]
        for row, b in enumerate(bases):
            ex = result["examples"][int(b)]
            base, src = parse_expr(ex["base"]), parse_expr(ex["sources"][k])
            hit = np.argwhere(flip[row])
            if len(hit):
                m, j = hit[0]
                ids.setdefault(prediction(k, base, src), int(am[row, m, j]))
    return ids


def fragile_rows(result: dict, arrays: dict[str, np.ndarray], k: str, ids: dict[int, int]) -> np.ndarray:
    """Rows of type ``k`` whose no-op swap (query token 0, every read point) does not give the base answer."""
    am, bases = arrays[f"{k}__argmax"], arrays[f"{k}__bases"]
    y = np.array([ids.get(result["examples"][int(b)]["Y"], -1) for b in bases])
    return np.where((am[:, :, 0] != y[:, None]).any(axis=1))[0]


def rates_from_arrays(result: dict, arrays: dict[str, np.ndarray], ids: dict[int, int], drop: dict[str, np.ndarray]) -> dict:
    """Flip and change rates recomputed from the stored per-row arrays, without the rows in ``drop``."""
    out = {}
    for i, k in enumerate(result["rates"]):
        flip, am, bases = arrays[f"{k}__flip"], arrays[f"{k}__argmax"], arrays[f"{k}__bases"]
        keep = np.setdiff1d(np.arange(len(bases)), drop.get(k, np.array([], dtype=int)))
        y = np.array([ids.get(result["examples"][int(b)]["Y"], -1) for b in bases])
        change = am != y[:, None, None]  # the base run's answer is Y on every kept row (D45 gate)
        out[k] = {"n": int(len(keep)),
                  "flip": bootstrap_rate(flip[keep], np.random.default_rng(SEED + 2 * i), N_BOOT, ALPHA),
                  "change": bootstrap_rate(change[keep], np.random.default_rng(SEED + 2 * i + 1), N_BOOT, ALPHA)}
    return out


def describe(result: dict, rates: dict, read: dict) -> dict:
    labels, rps = result["labels"], result["read_points"]
    ans = len(labels) - 1
    digits = [j for j, lab in enumerate(labels) if lab in ("a", "b", "c", "d", "e", "f")]
    others = [j for j in range(len(labels)) if j not in digits and j != ans]
    count = np.asarray(read["count"])
    brackets = [k[len("present_"):] for k in read if k.startswith("present_")]
    all_at_ans = np.where(count[:, ans] == len(brackets))[0]  # the gather: every sub-result at the answer token
    gather = int(all_at_ans[0]) if len(all_at_ans) else None
    same = [k for k in rates if k.endswith("_same")]
    dig = np.max([np.asarray(rates[k]["change"]["mean"])[:, digits].max(axis=1) for k in same], axis=0)
    last_digit = max((m for m in range(len(rps)) if dig[m] > 0.5), default=None)
    combined = "all" if "all" in result["rates"] else "both"
    hyp = result["rates"].get(combined, {}).get("hypotheses")
    return {
        "gather_read_point": rps[gather] if gather is not None else None,
        "gather_fraction": round(rps[gather] / rps[-1], 3) if gather is not None else None,
        "gather_holds_to_the_end": bool(gather is not None and (count[gather:, ans] == len(brackets)).all()),
        "last_digit_read_point": rps[last_digit] if last_digit is not None else None,
        "present_elsewhere": {v: bool(np.asarray(read[f"present_{v}"])[:, others].any()) for v in brackets},
        "both_swap_source_answer_at_gather": (round(float(np.asarray(hyp[combined]["mean"])[gather, ans]), 3)
                                              if hyp and gather is not None else None),
    }


def summarise_task(task_dir: Path) -> dict:
    result = json.load(open(task_dir / "varmap.json"))
    arrays = dict(np.load(task_dir / "varmap_arrays.npz"))
    th = result["thresholds"]
    stored = {k: {kk: {s: np.asarray(v) for s, v in result["rates"][k][kk].items()} for kk in ("flip", "change")}
              for k in result["rates"]}
    out = {"accuracy": result["accuracy"], "n_kept": result["n_kept"], "n_read_points": len(result["read_points"]),
           "as_run": describe(result, stored, reading(stored, result["task"], th["present_flip"], th["value_change"]))}
    ids = digit_token_ids(result, arrays)
    drop = {k: fragile_rows(result, arrays, k, ids) for k in result["rates"]}
    out["fragile_rows"] = {k: int(len(v)) for k, v in drop.items()}
    rec = rates_from_arrays(result, arrays, ids, drop)
    out["fragile_dropped"] = describe(result, rec, reading(rec, result["task"], th["present_flip"], th["value_change"]))
    return out


def plot(summary: dict, path: Path) -> None:
    """Per model and task: where the operands live along the depth (the digit tokens up to the last digit read point,
    the answer token from the gather on), with the single-control hand-over and the heads' write depth marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(run, task, s) for run, tasks in summary.items() for task, s in sorted(tasks.items(), reverse=True)]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3))
    for i, (run, task, s) in enumerate(rows):
        a, L = s["as_run"], s["n_read_points"] - 1
        y = len(rows) - 1 - i
        if a["last_digit_read_point"] is not None:
            ax.barh(y, a["last_digit_read_point"] / L, left=0, height=0.5, color="#9ec5f4")
        if a["gather_read_point"] is not None:
            ax.barh(y, 1 - a["gather_read_point"] / L, left=a["gather_read_point"] / L, height=0.5, color="#184f95")
        key = next((k for k in HANDOVER if k + "_" in run), None)
        if key:
            ax.plot([HANDOVER[key][0] / L] * 2, [y - 0.33, y + 0.33], color="#eb6834", lw=2)
            ax.plot(HANDOVER[key][1] / L, y, marker="v", color="#0b0b0b", ms=4)
        ax.text(1.01, y, f"acc {s['accuracy']:.2f}", va="center", fontsize=7, color="#52514e")
    ax.set_yticks(range(len(rows)))
    sizes = [re.search(r"_(\d+(?:\.\d+)?b)_", run + "_").group(1).upper() for run, _, _ in reversed(rows)]
    ax.set_yticklabels([f"Qwen3 {size} {task.replace('_', '-')}" for size, (_, task, _) in zip(sizes, reversed(rows))],
                       fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("depth (read point / number of blocks)", fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#9ec5f4"), plt.Rectangle((0, 0), 1, 1, color="#184f95"),
               plt.Line2D([], [], color="#eb6834", lw=2), plt.Line2D([], [], color="#0b0b0b", marker="v", ls="", ms=4)]
    ax.legend(handles, ["operands at their digit tokens, as digits", "every sub-result at the answer token",
                        "single-control hand-over (D44)", "in-context heads' write depth"],
              fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False)
    tasks = {t for _, t, _ in rows}
    names = [n for t, n in (("two_lane", "(a+b) - (c+d)"), ("one_lane", "(a+b) - c"), ("three_lane", "(a+b) - (c+d) - (e+f)")) if t in tasks]
    ax.set_title(f"Where the operands of {', '.join(names)} live, by depth", fontsize=8.5, loc="left")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a directory holding varmap run directories")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--figure", type=Path)
    args = ap.parse_args(argv)
    summary = {}
    for run in sorted(args.root.glob("varmap_*"), key=by_size):
        for task_dir in sorted((run / "core" / "tasks").glob("*")):
            if (task_dir / "varmap.json").exists():
                summary.setdefault(run.name, {})[task_dir.name] = summarise_task(task_dir)
    if not summary:
        print(f"no variable maps under {args.root}", file=sys.stderr)
        return 1
    for run, tasks in summary.items():
        for task, s in tasks.items():
            a, f = s["as_run"], s["fragile_dropped"]
            print(f"{run} {task}: accuracy {s['accuracy']:.3f}, gather at read point {a['gather_read_point']} "
                  f"({a['gather_fraction']} of the stack; {f['gather_read_point']} without the {sum(s['fragile_rows'].values())} "
                  f"fragile rows), last digit read point {a['last_digit_read_point']}, present elsewhere {a['present_elsewhere']}, "
                  f"both-swap source answer at the gather {a['both_swap_source_answer_at_gather']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    if args.figure:
        plot(summary, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
