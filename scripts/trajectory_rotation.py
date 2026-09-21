"""Exploratory: the geometry of the population-mean perturbation trajectory after the injection (STATUS.md, D35
exploratory diagnostics). Is the downstream change of the perturbation a *rotation* (successive increments in
the same direction or plane) or a sequence of writes in new directions?

For each task of a finished `directions trajectories` run and each construction at the task's primary layer,
from the stored population-mean trajectories (``trajectories_arrays.npz``: ``mean_delta_icl`` and
``L{l}_mean_delta_{construction}``, one d-vector per read point):

* turn: cos(δ_m, δ_{m+1}), how much the direction changes per block;
* consec: cos(Δ_m, Δ_{m+1}) with Δ_m = δ_{m+1} − δ_m, whether successive increments point the same way, and the
  median pairwise |cos| among all increments after the injection (a random baseline is about 1/√d);
* ev1/2/4: the share of the increments' squared norm on their top 1/2/4 principal directions, against random
  directions with the same norms (the norm distribution alone sets a floor);
* toward-end: cos(Δ_m, δ_L − δ_m), whether the increments point toward where the trajectory ends;
* toward-natural: cos(Δ_m, n_{m+1}) with n the natural (demonstrations) mean difference at the next read point;
* inc~natural-inc: cos(Δ_m, Δn_m), whether the block writes the same thing it writes in the natural run;
* persist: cos(δ_l*, δ_m), how much of the injected direction is still there.

Models whose blocks multiply their output by a scalar (Gemma 4) need ``--scalars s_0,...,s_{L-1}``: the
increment is then δ_{m+1} − s_m δ_m.

usage: uv run python scripts/trajectory_rotation.py <label>=<run dir> [...] [--scalars s0,s1,...]
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def geometry(D: np.ndarray, l: int, L: int, scal: list[float] | None) -> dict:
    D = D.astype(np.float64)
    steps = list(range(l, L))
    inc = np.stack([D[m + 1] - (scal[m] if scal is not None else 1.0) * D[m] for m in steps])
    U = inc / np.maximum(np.linalg.norm(inc, axis=1, keepdims=True), 1e-12)
    G = U @ U.T
    s = np.linalg.svd(inc, compute_uv=False) ** 2
    ev = s / s.sum()
    rng = np.random.default_rng(0)
    R = rng.standard_normal(inc.shape)
    R *= np.linalg.norm(inc, axis=1, keepdims=True) / np.linalg.norm(R, axis=1, keepdims=True)
    sR = np.linalg.svd(R, compute_uv=False) ** 2
    return {
        "turn": [cos(D[m], D[m + 1]) for m in steps],
        "consec": [cos(inc[i], inc[i + 1]) for i in range(len(steps) - 1)],
        "pair_abs": float(np.median(np.abs(G[np.triu_indices(len(steps), 1)]))),
        "ev": [float(ev[0]), float(ev[:2].sum()), float(ev[:4].sum())],
        "ev_random": [float(sR[0] / sR.sum()), float(sR[:2].sum() / sR.sum()), float(sR[:4].sum() / sR.sum())],
        "toward_end": [cos(inc[i], D[L] - D[m]) for i, m in enumerate(steps[:-1])],
        "persist": [cos(D[l], D[m]) for m in range(l, L + 1)],
        "inc": inc, "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=<trajectories run dir>")
    ap.add_argument("--scalars", default=None, help="comma-separated per-block output scalars (Gemma 4)")
    ap.add_argument("--constructions", nargs="*", default=["learned", "fv"])
    args = ap.parse_args()
    scal = [float(x) for x in args.scalars.split(",")] if args.scalars else None
    for arg in args.runs:
        label, run = arg.split("=", 1)
        print(f"\n## {label}: {run}")
        agg: dict[str, list[float]] = {}
        for f in sorted(glob.glob(f"{run}/core/tasks/*/trajectories.json")):
            task = f.split("/")[-2]
            r = json.load(open(f))
            l, L = r["primary_layer"], r["n_read_points"] - 1
            a = np.load(f.replace("trajectories.json", "trajectories_arrays.npz"))
            nat = a["mean_delta_icl"]
            gn = geometry(nat, l, L, scal)
            print(f"  {task:19s} L*={l:2d} natural: turn {np.median(gn['turn']):.2f} consec {np.median(gn['consec']):+.2f} "
                  f"pair|cos| {gn['pair_abs']:.2f} ev1/2/4 {'/'.join(f'{v:.2f}' for v in gn['ev'])}")
            for c in args.constructions:
                key = f"L{l}_mean_delta_{c}"
                if key not in a.files:
                    continue
                g = geometry(a[key], l, L, scal)
                toward_nat = [cos(g["inc"][i], nat[m + 1].astype(np.float64)) for i, m in enumerate(g["steps"])]
                same_write = [cos(g["inc"][i], gn["inc"][i]) for i in range(len(g["steps"]))]
                print(f"     {c:8s} turn {np.median(g['turn']):.2f} (min {min(g['turn']):.2f}) | consec {np.median(g['consec']):+.2f} "
                      f"(min {min(g['consec']):+.2f}) pair|cos| {g['pair_abs']:.2f} | ev1/2/4 {'/'.join(f'{v:.2f}' for v in g['ev'])} "
                      f"(random {'/'.join(f'{v:.2f}' for v in g['ev_random'])}) | toward-end {np.median(g['toward_end']):+.2f} | "
                      f"toward-natural {np.median(toward_nat):+.2f} (first {toward_nat[0]:+.2f}, last {toward_nat[-1]:+.2f}) | "
                      f"inc~natural-inc {np.median(same_write):+.2f} | persist mid {g['persist'][len(g['persist']) // 2]:.2f} end {g['persist'][-1]:.2f}")
                for k, v in (("turn", np.median(g["turn"])), ("consec", np.median(g["consec"])), ("pair", g["pair_abs"]),
                             ("ev1", g["ev"][0]), ("ev1_random", g["ev_random"][0]), ("toward_end", np.median(g["toward_end"])),
                             ("toward_nat", np.median(toward_nat)), ("toward_nat_last", toward_nat[-1]),
                             ("same_write", np.median(same_write)), ("persist_end", g["persist"][-1])):
                    agg.setdefault(f"{c} {k}", []).append(float(v))
            for k, v in (("turn", np.median(gn["turn"])), ("consec", np.median(gn["consec"])), ("pair", gn["pair_abs"])):
                agg.setdefault(f"natural {k}", []).append(float(v))
        print("  medians over tasks:")
        for k, v in agg.items():
            print(f"    {k:26s} {np.median(v):+.2f}  (range {min(v):+.2f} .. {max(v):+.2f}, n={len(v)})")


if __name__ == "__main__":
    main()
