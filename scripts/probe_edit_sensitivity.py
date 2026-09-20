"""Probe: how sensitive is a model's held-out effect to small additive edits of the residual at a later read
point? (A diagnostic for the D29/D33/D34 edit mechanics on a new model family; docs/DECISIONS.md D35.)

The depth-of-commitment test (D29) edits the steered perturbation at a read point ``m`` and compares the real
edit with the same edit along random unit directions, on the premise that removing a random component changes
nothing. On Gemma 4 that premise failed in the deep half of the stack. This probe measures, on one task of a
finished pilot run and at chosen read points:

* the exactness of an all-zero edit at ``m`` (must reproduce the steered effect bit for bit);
* the effect of random edits ``eps * u`` (``u`` a random unit direction) of several norms, applied to the
  steered run and to the unsteered run: the mean change of the per-token log-probability and its spread across
  prompts;
* the D29 random-removal edit for reference;
* the norm of the residual and of the perturbation at ``m`` and how concentrated they are on a few coordinates.

usage: uv run python scripts/probe_edit_sensitivity.py --run <pilot run dir> --tasks antonym past_tense \
           [--read-points 26 30 34 38 42 46 47 48] [--eps 0.01 0.1 1 10] [--n-dirs 4] [--n-prompts 96] [--out-dir results/probe]

Read points at or below a task's injection layer are skipped. One JSON per task under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from directions.config import load_config
from directions.model import Intervention, ModelBackend
from directions.prompts import zero_shot_prompt
from directions.seeds import rng_for
from directions.tasks import Item


def concentration(x: np.ndarray, k: int = 5) -> dict[str, float]:
    """Per row: the share of the squared norm on the ``k`` largest coordinates; medians over rows."""
    sq = x.astype(np.float64) ** 2
    tot = sq.sum(-1, keepdims=True)
    top = -np.sort(-sq, axis=-1)[:, :k]
    return {
        "norm_median": float(np.median(np.sqrt(tot[:, 0]))),
        "max_abs_median": float(np.median(np.abs(x).max(-1))),
        f"top{k}_share_median": float(np.median(top.sum(-1) / np.maximum(tot[:, 0], 1e-30))),
        "argmax_mode": int(np.bincount(np.abs(x).argmax(-1)).argmax()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--read-points", type=int, nargs="*", default=None)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--n-dirs", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.resolved.yaml")
    t0 = time.time()
    backend = ModelBackend(cfg.model, run_seed=cfg.seed)
    print(f"model {backend.metadata()['name']} ({backend.n_layers} layers, d={backend.hidden_size}); load {time.time() - t0:.0f}s", flush=True)
    for task in args.tasks:
        probe_task(backend, cfg, run, task, args)


def probe_task(backend: ModelBackend, cfg, run: Path, task: str, args) -> None:
    task_dir = run / "core" / "tasks" / task
    splits = json.load(open(task_dir / "splits.json"))
    qual = json.load(open(task_dir / "qualification.json"))
    sel = qual["selection"]
    layer, alpha = int(sel["layer"]), float(sel["alpha"])
    dirs = np.load(task_dir / "directions.npz")
    if "fv_direction" in dirs.files:  # the control of the run: the function vector, the learned vector or PC1
        v = np.asarray(dirs["fv_direction"], dtype=np.float64)
    elif "learned" in dirs.files:
        v = np.asarray(dirs["learned"], dtype=np.float64)[list(dirs["layers"]).index(layer)]
    else:
        v = np.asarray(dirs["pooled"], dtype=np.float64)[list(dirs["layers"]).index(layer)]
    v = v / np.linalg.norm(v)

    prompt_cfg = replace(cfg.prompt, target_scoring=splits.get("target_scoring") or cfg.prompt.target_scoring)
    prompts = [zero_shot_prompt(prompt_cfg, Item(i, o)) for i, o in splits["evaluation"]][: args.n_prompts]
    n, d, L = len(prompts), backend.hidden_size, backend.n_layers
    read_points = args.read_points or sorted({layer + 1, layer + 2, layer + 4, layer + 8, (layer + L) // 2, L - 2, L - 1, L})
    read_points = [m for m in read_points if layer < m <= L]
    print(f"task {task}, layer {layer}, alpha {alpha:.2f}, {n} prompts, read points {read_points}", flush=True)

    inject = Intervention(layer, v, alpha)
    base = backend.run(prompts, capture=True)
    steered = backend.run(prompts, interventions=[inject], capture=True)
    again = backend.run(prompts, interventions=[inject])
    full_diff = steered.logprob_per_token - base.logprob_per_token
    full = float(np.mean(full_diff))
    print(f"full effect {full:+.3f} nats/token (sd over prompts {np.std(full_diff):.2f}); repeat bit-identical: "
          f"{bool(np.array_equal(again.logprob_per_token, steered.logprob_per_token))}", flush=True)
    rng = rng_for(cfg.seed, "probe_edit_sensitivity", task)
    U = rng.standard_normal((args.n_dirs, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    out = {"run": str(run), "task": task, "layer": layer, "alpha": alpha, "n_prompts": n, "full_effect": full,
           "full_sd": float(np.std(full_diff)), "eps": args.eps, "n_dirs": args.n_dirs, "read_points": []}
    for m in read_points:
        t1 = time.time()
        h = base.residuals[m].astype(np.float64)
        delta = steered.residuals[m].astype(np.float64) - h
        row: dict = {"read_point": m, "residual": concentration(h), "perturbation": concentration(delta),
                     "relative_magnitude_median": float(np.median(np.linalg.norm(delta, axis=1) / np.linalg.norm(h, axis=1)))}
        # (a) the all-zero edit must reproduce the steered run exactly
        z = backend.run(prompts, interventions=[inject, Intervention(m, np.zeros((n, d), dtype=np.float32), 1.0)])
        row["zero_edit"] = {"effect": float(np.mean(z.logprob_per_token - base.logprob_per_token)),
                            "max_abs_dev_from_steered": float(np.max(np.abs(z.logprob_per_token - steered.logprob_per_token)))}
        # (b) random edits of fixed norm on the steered and on the unsteered run
        row["random_edits"] = []
        for eps in args.eps:
            st_ch, ba_ch = [], []
            for u in U:
                e = backend.run(prompts, interventions=[inject, Intervention(m, u, eps)])
                st_ch.append(e.logprob_per_token - steered.logprob_per_token)
                b = backend.run(prompts, interventions=[Intervention(m, u, eps)])
                ba_ch.append(b.logprob_per_token - base.logprob_per_token)
            st_ch, ba_ch = np.stack(st_ch), np.stack(ba_ch)
            row["random_edits"].append({
                "eps": eps,
                "steered": {"mean_change": float(st_ch.mean()), "sd_over_prompts": float(st_ch.std(axis=1).mean()),
                            "max_abs": float(np.abs(st_ch).max()), "share_over_1nat": float(np.mean(np.abs(st_ch) > 1.0))},
                "base": {"mean_change": float(ba_ch.mean()), "sd_over_prompts": float(ba_ch.std(axis=1).mean()),
                         "max_abs": float(np.abs(ba_ch).max()), "share_over_1nat": float(np.mean(np.abs(ba_ch) > 1.0))},
            })
        # (c) the D29 random-removal edit: -(delta . u) u, per prompt
        rem = []
        for u in U:
            vec = (-(delta @ u)[:, None] * u[None, :]).astype(np.float32)
            r = backend.run(prompts, interventions=[inject, Intervention(m, vec, 1.0)])
            rem.append(r.logprob_per_token - base.logprob_per_token)
        rem = np.stack(rem)
        row["d29_random_remove"] = {"retained_mean": float(rem.mean() / full) if full else None,
                                    "edit_norm_median": float(np.median(np.abs(delta @ U.T))),
                                    "sd_over_prompts": float((rem - full_diff[None, :]).std(axis=1).mean())}
        out["read_points"].append(row)
        re = row["random_edits"]
        print(f"m={m:2d} |h| {row['residual']['norm_median']:.1f} (max coord {row['residual']['max_abs_median']:.1f}, top5 share "
              f"{row['residual']['top5_share_median']:.2f}) |delta| {row['perturbation']['norm_median']:.1f} (rel {row['relative_magnitude_median']:.2f}, "
              f"max coord {row['perturbation']['max_abs_median']:.1f}, top5 {row['perturbation']['top5_share_median']:.2f}); "
              f"zero edit dev {row['zero_edit']['max_abs_dev_from_steered']:.2e}; "
              + "; ".join(f"eps {r['eps']:g}: steered {r['steered']['mean_change']:+.2f}±{r['steered']['sd_over_prompts']:.2f} "
                          f"base {r['base']['mean_change']:+.2f}±{r['base']['sd_over_prompts']:.2f}" for r in re)
              + f"; D29 rand-remove retained {row['d29_random_remove']['retained_mean']:+.2f} (edit norm {row['d29_random_remove']['edit_norm_median']:.2f})"
              f"; {time.time() - t1:.0f}s", flush=True)
    if args.out_dir:
        path = Path(args.out_dir) / f"{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
