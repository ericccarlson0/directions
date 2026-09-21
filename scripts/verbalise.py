"""The verbalisation test (docs/DECISIONS.md D36) on a finished `directions trajectories` run: read the stored
population-mean directions through a fitted Jacobian lens and through the plain logit lens at every read point,
and ask whether they name the task.

Per task, at the primary layer, the directions read are: the mean natural difference (demonstrations minus
none) at every read point; the learned vector's and the head-mean vector's mean perturbation at every read
point; the injected vectors themselves at their injection read point; the generic response; and random
directions of the same norm as the natural difference at each read point (the floor). For each, the top
tokens, the probability mass on a fixed list of task words (``directions.verbalise.TASK_WORDS``), the best
task word and its rank, and the mass on the task's answer tokens (the first target token of every evaluation
prompt). The hand-over read point is taken from the run's subspace test where present, else from the first
patch row at which the per-prompt keep retains 0.9.

usage: uv run python scripts/verbalise.py --run <trajectories run dir> \
           [--lens-repo neuronpedia/jacobian-lens --lens-file qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt | --lens <local .pt>] \
           [--model-config <learned run>/config.resolved.yaml] [--tasks ...] [--top 12] [--n-random 8] --out-dir results/verbalise
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from directions.config import load_config
from directions.model import ModelBackend
from directions.prompts import zero_shot_prompt
from directions.seeds import rng_for
from directions.tasks import Item
from directions.verbalise import TASK_WORDS, Lens, readout_summary, transport, word_token_ids


def handover_read_point(res: dict, layer: int) -> int | None:
    info = res["per_layer"][str(layer)]
    sub = info.get("subspace")
    if sub and "learned" in sub.get("constructions", {}):
        return sub["constructions"]["learned"].get("handover_read_point")
    patch = info.get("patch", {}).get("learned") or {}
    for row in patch.get("rows", []):
        if row.get("keep", {}).get("retained", 0) >= 0.9:
            return row["read_point"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lens", default=None, help="local lens .pt")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", default=None, help="path inside the repo, e.g. qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt")
    ap.add_argument("--model-config", default=None, help="config.resolved.yaml of one of the runs read (default: the learned run's)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.load(open(run / "metadata.json"))
    learned_run = Path(meta["runs"]["learned"]["path"])
    cfg = load_config(args.model_config or learned_run / "config.resolved.yaml")
    t0 = time.time()
    backend = ModelBackend(cfg.model, run_seed=cfg.seed)
    L = backend.n_layers
    if args.lens:
        lens_path = args.lens
    else:
        from huggingface_hub import hf_hub_download

        lens_path = hf_hub_download(args.lens_repo, args.lens_file)
    lens = Lens.load(lens_path)
    if lens.d_model != backend.hidden_size:
        raise ValueError(f"lens d_model {lens.d_model} != model hidden size {backend.hidden_size}")
    print(f"model {backend.metadata()['name']} ({L} layers, d={backend.hidden_size}); lens {lens_path} fitted on "
          f"{lens.n_prompts} prompts, blocks {min(lens.jacobians)}..{max(lens.jacobians)}; load {time.time() - t0:.0f}s", flush=True)
    tok = backend.tokenizer
    decode = tok.decode
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(h: np.ndarray, m: int, use_lens: bool, task_ids: dict[str, int], answer_ids: np.ndarray) -> dict:
        z = transport(lens if use_lens else None, h, m, L)
        logits = backend.logits_from_residual(z.reshape(1, -1).to(backend.device))[0]
        lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        return readout_summary(lp, decode, args.top, task_ids, answer_ids)

    tasks = args.tasks or sorted(p.name for p in (run / "core" / "tasks").iterdir() if (p / "trajectories.json").exists())
    summary: dict[str, dict] = {}
    for task in tasks:
        tdir = run / "core" / "tasks" / task
        res = json.load(open(tdir / "trajectories.json"))
        layer = res["primary_layer"]
        arrays = np.load(tdir / "trajectories_arrays.npz")
        splits = json.load(open(learned_run / "core" / "tasks" / task / "splits.json"))
        prompt_cfg = replace(cfg.prompt, target_scoring=splits.get("target_scoring") or cfg.prompt.target_scoring)
        prompts = [zero_shot_prompt(prompt_cfg, Item(i, o)) for i, o in splits["evaluation"]]
        answer_ids = backend.first_target_token_ids(prompts)
        registry = splits.get("registry_task", task)
        task_ids = word_token_ids(tok, TASK_WORDS.get(task, TASK_WORDS.get(registry, [])))
        dirs = np.load(learned_run / "core" / "tasks" / task / "directions.npz")
        injected = {"learned": np.asarray(dirs["learned"])[list(dirs["layers"]).index(layer)]} if "learned" in dirs.files else {}
        fv_run = Path(meta["runs"]["fv"]["path"])
        fv_dirs_path = fv_run / "core" / "tasks" / task / "directions.npz"
        if fv_dirs_path.exists():
            fd = np.load(fv_dirs_path)
            if "fv_direction" in fd.files:
                injected["fv"] = np.asarray(fd["fv_direction"])
        hand = handover_read_point(res, layer)
        rng = rng_for(cfg.seed, "verbalise", task)
        rows = []
        for m in range(1, L + 1):
            nat = arrays["mean_delta_icl"][m].astype(np.float32)
            row: dict = {"read_point": m, "depth_fraction": m / L, "natural_norm": float(np.linalg.norm(nat))}
            row["natural"] = {"jlens": read(nat, m, True, task_ids, answer_ids), "logit_lens": read(nat, m, False, task_ids, answer_ids)}
            for c in ("learned", "fv"):
                key = f"L{layer}_mean_delta_{c}"
                if key in arrays.files and m >= layer:
                    row[c] = {"jlens": read(arrays[key][m].astype(np.float32), m, True, task_ids, answer_ids)}
            gkey = f"L{layer}_mean_generic"
            if gkey in arrays.files and m >= layer:
                row["generic"] = {"jlens": read(arrays[gkey][m].astype(np.float32), m, True, task_ids, answer_ids)}
            # random directions with the natural difference's norm: the floor of the masses
            R = rng.standard_normal((args.n_random, backend.hidden_size)).astype(np.float32)
            R *= np.linalg.norm(nat) / np.linalg.norm(R, axis=1, keepdims=True)
            rr = [read(r, m, True, task_ids, answer_ids) for r in R]
            row["random"] = {"jlens": {"task_mass_mean": float(np.mean([r["task_mass"] for r in rr])),
                                       "answer_mass_mean": float(np.mean([r["answer_mass"] for r in rr])),
                                       "task_best_rank_median": float(np.median([r["task_best_rank"] for r in rr if r["task_best_rank"]])) if task_ids else None}}
            rows.append(row)
        inj = {c: {"read_point": layer, "jlens": read(v.astype(np.float32), layer, True, task_ids, answer_ids),
                   "logit_lens": read(v.astype(np.float32), layer, False, task_ids, answer_ids)} for c, v in injected.items()}
        result = {"task": task, "primary_layer": layer, "n_layers": L, "handover_read_point": hand, "task_words": task_ids,
                  "n_answer_tokens": int(len(np.unique(answer_ids))), "rows": rows, "injected": inj}
        with open(out_dir / f"{task}.json", "w") as f:
            json.dump(result, f, indent=1)
        # a compact log: the natural difference through the lens at a few read points, and at the hand-over
        def fmt(r: dict) -> str:
            return (f"task-mass {r['task_mass']:.3f} (best '{r['task_best']}' rank {r['task_best_rank']}) answer-mass {r['answer_mass']:.3f} "
                    f"(rank {r['answer_best_rank']}) top {[t for t, _ in r['top'][:8]]}")
        print(f"\n== {task} L*={layer} hand-over {hand} (answers: {len(np.unique(answer_ids))} distinct first tokens; task words {list(task_ids)})", flush=True)
        for row in rows:
            m = row["read_point"]
            if m in (layer, hand, L) or m % 4 == 0:
                print(f"  m={m:2d} ({row['depth_fraction']:.2f}) natural/jlens: {fmt(row['natural']['jlens'])} | random task-mass {row['random']['jlens']['task_mass_mean']:.3f} answer-mass {row['random']['jlens']['answer_mass_mean']:.3f}", flush=True)
                if "learned" in row:
                    print(f"            learned/jlens: {fmt(row['learned']['jlens'])}", flush=True)
                if m == hand:
                    print(f"            natural/logit-lens: {fmt(row['natural']['logit_lens'])}", flush=True)
        for c, r in inj.items():
            print(f"  injected {c} at m={layer}: jlens {fmt(r['jlens'])} | logit lens {fmt(r['logit_lens'])}", flush=True)
        summary[task] = {"primary_layer": layer, "handover_read_point": hand,
                         "natural_task_mass_by_read_point": [row["natural"]["jlens"]["task_mass"] for row in rows],
                         "natural_answer_mass_by_read_point": [row["natural"]["jlens"]["answer_mass"] for row in rows],
                         "random_task_mass_by_read_point": [row["random"]["jlens"]["task_mass_mean"] for row in rows],
                         "learned_task_mass_by_read_point": [row.get("learned", {}).get("jlens", {}).get("task_mass") for row in rows]}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"run": str(run), "lens": str(lens_path), "model": backend.metadata()["name"], "tasks": summary}, f, indent=1)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
