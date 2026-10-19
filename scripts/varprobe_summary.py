"""The D46 probe and lens runs, summarised against the D45 gather (descriptive; the preregistered readouts are the
per-read-point curves in each run's varprobe.json).

Per model and task: for each probe target, its mean accuracy before the gather (read points 1..G-1; read point 0 is
the bare embedding), at G-1, its peak after the gather and where, and its value at L; where the Y probe reaches 0.9 of
its accuracy at L; where the lens first reads Y in at least half the examples; and the lens candidates read above
their family-wise null within two read points of the gather, with their peak share. Then the D46 decision rules on
the two-lane "S1 given Y". The figure overlays the models on a depth axis.

    uv run python scripts/varprobe_summary.py results/remote/run-38587101804 --out docs/varprobe/summary.json \
        --figure docs/varprobe/depth.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# the D45 gather read point per model (runs 38502381471 and, for the 8B, 38738001665; the configs' marks)
GATHER = {"qwen3_0.6b": 21, "qwen3_1.7b": 21, "qwen3_4b": 24, "qwen3_8b": 24, "qwen3_14b": 30}
# the three-bracket gather G3 per model (D51; first read point with all three sums at the answer token)
GATHER3 = {"qwen3_4b": 24, "qwen3_8b": 24, "qwen3_14b": 30}  # runs 38934415363, 38934932951, 38934471957
# the max gather per model (D55; first read point with both sums at the answer token), from its job 1
GATHER_MAX = {"qwen3_1.7b": 21, "qwen3_4b": 24, "qwen3_8b": 24}  # runs 39178177443, 39178188546



def by_size(path: Path) -> tuple:
    """Run directories in model-size order (``..._qwen3_14b_...`` after ``..._qwen3_8b_...``), then by name."""
    m = re.search(r"_(\d+(?:\.\d+)?)b(?:_|$)", path.name)
    return (float(m.group(1)) if m else float("inf"), path.name)

def model_key(run: str) -> str:
    return next(k for k in GATHER if f"_{k}_" in f"_{run}_")


def summarise(r: dict, G: int) -> dict:
    L = len(r["read_points"]) - 1
    out = {"accuracy": r["accuracy"], "n_kept": r["n_kept"], "gather": G, "L": L, "probes": {}, "lens": {}}
    for t, p in r["probes"].items():
        acc = np.asarray(p["accuracy"])
        post = int(G + np.argmax(acc[G:]))
        out["probes"][t] = {"pre_mean": round(float(acc[1:G].mean()), 3), "at_G_minus_1": round(float(acc[G - 1]), 3),
                            "peak_after": round(float(acc[post]), 3), "peak_at": post, "at_L": round(float(acc[L]), 3),
                            "null_mean": round(float(np.mean(p["null_mean"])), 3),
                            "carried_after_G": bool(np.asarray(p["carried"])[G:].any()),
                            "carried_at_L": bool(p["carried"][L])}
    y = np.asarray(r["probes"]["Y"]["accuracy"])
    out["y_probe_formed_at"] = int(np.flatnonzero(y >= 0.9 * y[L])[0])
    lens_y = np.asarray(r["lens"]["candidates"]["Y"]["rate"])
    out["lens_reads_y_from"] = int(np.flatnonzero(lens_y >= 0.5)[0]) if (lens_y >= 0.5).any() else None
    for c, lc in r["lens"]["candidates"].items():
        if c == "Y" or not lc.get("n"):
            continue
        rate, carried = np.asarray(lc["rate"]), np.asarray(lc["carried"], dtype=bool)
        win = [m for m in range(max(1, G - 2), min(L + 1, G + 3)) if carried[m]]
        if win:
            m = max(win, key=lambda k: rate[k])
            out["lens"][c] = {"peak": round(float(rate[m]), 3), "at": m, "null": round(float(lc["null_mean"][m]), 3),
                              "carried_at": win}
    return out


def rules(s: dict) -> dict:
    """The D46 decision rules on the two-lane "S1 given Y" (rule 5 is read from the lens fields)."""
    p = s["probes"]["S1|Y"]
    return {"rule_1_answer_only": not p["carried_after_G"],
            "rule_2_discarded_before_L": p["carried_after_G"] and not p["carried_at_L"],
            "carried_to_L": p["carried_at_L"]}


def plot(runs: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"qwen3_1.7b": "#2a78d6", "qwen3_4b": "#eb6834"}  # categorical slots 1 and 2, in fixed order
    panels = [("probe", "Y", "probe: Y"), ("probe", "S1|Y", "probe: S1 given Y"), ("lens", "Y", "lens reads Y"),
              ("lens", "S1", "lens reads S1")]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.6), sharey=True)
    for ax, (kind, key, title) in zip(axes, panels):
        for model, r in runs.items():
            if model not in colours:
                continue
            L = len(r["read_points"]) - 1
            x = np.arange(L + 1) / L
            d = r["probes"][key] if kind == "probe" else r["lens"]["candidates"][key]
            obs = np.asarray(d.get("accuracy", d.get("rate")), dtype=float)
            # where the lens reads one digit for every example the share equals its null exactly: left blank
            obs = np.where(np.asarray(d["null_sd"]) > 1e-12, obs, np.nan)
            ax.plot(x, np.asarray(d["null_mean"]), color=colours[model], lw=0.8, ls=(0, (2, 2)), alpha=0.7)
            ax.plot(x, obs, color=colours[model], lw=1.4, label=model.replace("qwen3_", "Qwen3 ").upper().replace("QWEN3", "Qwen3"))
            ax.axvline(GATHER[model] / L, color=colours[model], lw=0.8, ls=(0, (1, 2)))
        ax.set_title(title, fontsize=8, loc="left", color="#0b0b0b")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("depth (read point / L)", fontsize=7, color="#52514e")
        ax.tick_params(labelsize=6.5, length=2, colors="#52514e")
        ax.grid(axis="y", color="#ecebe7", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
    axes[0].set_ylabel("accuracy / share", fontsize=7, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#8a8985", lw=0.8, ls=(0, (2, 2))), plt.Line2D([], [], color="#8a8985", lw=0.8, ls=(0, (1, 2)))]
    labels += ["null mean", "D45 gather (gaps: the lens reads one digit for every example)"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.08), ncol=4, fontsize=7, frameon=False)
    fig.suptitle("Answer token of (a+b) - (c+d): the sums become linearly readable at the gather, the answer a few blocks later",
                 fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--figure", type=Path)
    args = ap.parse_args(argv)
    summary, two_lane = {}, {}
    for run in sorted(args.root.glob("varprobe_*"), key=by_size):
        key = model_key(run.name)
        for task_dir in sorted((run / "core" / "tasks").glob("*")):
            r = json.loads((task_dir / "varprobe.json").read_text())
            G = {"three_lane": GATHER3, "max": GATHER_MAX}.get(task_dir.name, GATHER).get(key)
            if G is None:
                print(f"{key} {task_dir.name}: no gather recorded for this model and task; skipped", file=sys.stderr)
                continue
            s = summarise(r, G)
            if task_dir.name == "two_lane":
                s["rules"] = rules(s)
                two_lane[key] = r
            summary.setdefault(key, {})[task_dir.name] = s
        skipped = json.loads((run / "core" / "summary.json").read_text()).get("skipped", {})
        for task, why in skipped.items():
            summary.setdefault(key, {})[task] = {"skipped": why}
    if not summary:
        print(f"no probe runs under {args.root}", file=sys.stderr)
        return 1
    for model, tasks in summary.items():
        for task, s in tasks.items():
            if "skipped" in s:
                print(f"{model} {task}: skipped ({s['skipped']})")
                continue
            p = s["probes"]
            print(f"{model} {task}: acc {s['accuracy']:.3f}, n {s['n_kept']}, G {s['gather']}, L {s['L']}; Y probe formed at "
                  f"{s['y_probe_formed_at']}, lens reads Y (>= 0.5) from {s['lens_reads_y_from']}")
            for t, v in p.items():
                print(f"   {t:5s} pre {v['pre_mean']:.2f}  G-1 {v['at_G_minus_1']:.2f}  peak {v['peak_after']:.2f} @ {v['peak_at']}"
                      f"  L {v['at_L']:.2f}  (null {v['null_mean']:.2f})")
            print("   lens near the gather:", {c: (v["peak"], v["at"], v["null"]) for c, v in s["lens"].items()})
            if "rules" in s:
                print("   rules:", s["rules"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    if args.figure:
        plot(two_lane, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
