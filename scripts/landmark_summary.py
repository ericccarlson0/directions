"""The landmark test across models (docs/DECISIONS.md D37): read each model's ``landmarks/landmarks.json``
(scripts/landmarks.py), print the table of landmark depths against the hand-over, the per-task offsets, the
across-model rank correlations, and draw the scatter (landmark depth against hand-over depth, one point per
model, both as fractions of the stack).

usage: uv run python scripts/landmark_summary.py <label>=<landmark run dir> ... [--out results/landmarks_summary.json] [--figure results/landmarks.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from directions.landmarks import across_models

COLUMNS = [("handover_learned", "hand-over (learned)"), ("handover_fv", "hand-over (head mean)"),
           ("pool_rank_sustained", "pool rank one (sustained ≥ 0.7)"), ("pool_rank_fraction_of_max", "pool rank (0.9 × max)"),
           ("pool_rank_centered_sustained", "centred rank one (expl.)"), ("pool_rank_centered_fraction_of_max", "centred rank 0.9 × max (expl.)"),
           ("verbal_onset", "verbal onset (logit lens)"), ("verbal_exit", "verbal exit (logit lens)"),
           ("verbal_onset_jlens", "verbal onset (J-lens)"), ("verbal_exit_jlens", "verbal exit (J-lens)"),
           ("heads_median", "universal heads (median)")]


def fmt(v: float | None, L: int | None) -> str:
    if v is None:
        return "  –  "
    return f"{v:4.1f} ({v / L:.2f})" if L else f"{v:4.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=<landmark run dir>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    models: dict[str, dict] = {}
    for arg in args.runs:
        label, root = arg.split("=", 1)
        p = Path(root)
        p = p / "landmarks" / "landmarks.json" if p.is_dir() else p
        models[label] = json.load(open(p))

    print("landmark depths per model: median read point over tasks (fraction of the stack); the hand-over is the learned vector's")
    for label, m in models.items():
        s = m["summary"]
        L = s["n_layers"]
        print(f"\n== {label}: {m['model']}, L={L}")
        for key, name in COLUMNS:
            v = s["median"].get(key)
            n = s["n_tasks"].get(key, 0)
            off = s["offset_from_handover"].get(key) or {}
            line = f"  {name:34s} {fmt(v, L):>12}  n={n:2d}"
            if key != "handover_learned" and off.get("per_task"):
                line += f"   offset per task {off['per_task']}  median {off['median']:+.1f}  |median| {off['median_abs']:.1f}"
            print(line)
        band = m.get("lens_band")
        if band and band.get("pooled"):
            for k in ("hit@1", "hit@5", "hit@10"):
                b = band["pooled"]["band"][k]
                print(f"  workspace band ({k}, pooled sets)       onset {fmt(b['onset'], L)} peak {fmt(b['peak'], L)} exit {fmt(b['exit'], L)} (max rate {b['max']:.2f})")
            for name, sset in band["sets"].items():
                b = sset["band"]["hit@10"]
                print(f"    [{name:12s}] hit@10 onset {fmt(b['onset'], L)} peak {fmt(b['peak'], L)} (max {b['max']:.2f}; hit@1 max {max(v for v in sset['rates']['hit@1'] if v is not None):.2f}; n={sset['n_intermediates']})")

    cross = across_models(models)
    print("\nacross models (fractions of the stack; Spearman with a permutation p; |offset| in read points):")
    for lm, r in cross.items():
        c = r["spearman"]
        pts = " ".join(f"{p['model']}:{p['handover_frac']:.2f}->{p['landmark_frac']:.2f}" for p in r["points"])
        print(f"  {lm:28s} n={c['n']} rho={c['rho'] if c['rho'] is None else round(c['rho'], 2)!s:>5} p={c['p'] if c['p'] is None else round(c['p'], 3)!s:>6} "
              f"median|offset|={r['median_abs_offset_read_points']} coincides={r['coincides']}   {pts}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"models": {k: v["summary"] for k, v in models.items()}, "lens_band": {k: v.get("lens_band") for k, v in models.items()},
                       "across": cross}, f, indent=1)
        print("wrote", args.out)
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [("pool_rank_sustained", "pool rank one"), ("verbal_onset", "verbal onset"), ("verbal_exit", "verbal exit"),
                ("heads_median", "universal heads"), ("handover_fv", "hand-over, head mean")]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for key, name in keys:
            pts = cross.get(key, {}).get("points", [])
            if not pts:
                continue
            ax.scatter([p["handover_frac"] for p in pts], [p["landmark_frac"] for p in pts], label=name, s=36)
            for p in pts:
                ax.annotate(p["model"], (p["handover_frac"], p["landmark_frac"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("hand-over read point (fraction of the stack)")
        ax.set_ylabel("landmark read point (fraction of the stack)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.set_title("The landmark test (D37)", fontsize=9)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
