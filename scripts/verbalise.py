"""The verbalisation test (docs/DECISIONS.md D36) on a finished `directions trajectories` run: read the stored
population-mean directions through a fitted Jacobian lens (where one exists) and through the plain logit lens
at every read point, and ask whether they name the task.

Per task, at the primary layer, the directions read are: the mean natural difference (demonstrations minus
none) at every read point; the learned vector's and the head-mean vector's mean perturbation at every read
point; the injected vectors themselves at their injection read point; the generic response; and random
directions of the same norm as the natural difference at each read point (the floor). For each, the top
tokens, the probability mass on a fixed list of task words (``directions.verbalise.TASK_WORDS``), the best
task word and its rank, and the mass on the task's answer tokens (the first target token of every evaluation
prompt). The hand-over read point is taken from the run's subspace test where present, else from the first
patch row at which the per-prompt keep retains 0.9.

Without a lens (no --lens / --lens-file) only the logit lens is read (the landmark test, D37, on the checkpoints
without a published lens); the readout keys are then ``logit_lens`` only.

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
from typing import Any

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


def load_lens(backend: ModelBackend, lens: str | None, lens_repo: str, lens_file: str | None) -> tuple[Lens | None, str | None]:
    """The lens named on the command line (a local file, or a file of a Hub repo), or none."""
    if lens:
        path = lens
    elif lens_file:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(lens_repo, lens_file)
    else:
        return None, None
    out = Lens.load(path)
    if out.d_model != backend.hidden_size:
        raise ValueError(f"lens d_model {out.d_model} != model hidden size {backend.hidden_size}")
    return out, path


def readout(run: Path, backend: ModelBackend, lens: Lens | None, out_dir: Path, tasks: list[str] | None = None,
            top: int = 12, n_random: int = 8, model_config: str | None = None, lens_path: str | None = None) -> dict[str, Any]:
    """The readouts of every task of ``run`` (a trajectories run), written to ``out_dir`` (``<task>.json`` and
    ``summary.json``); returns the summary."""
    run = Path(run)
    meta = json.load(open(run / "metadata.json"))
    learned_run = Path(meta["runs"]["learned"]["path"])
    cfg = load_config(model_config or learned_run / "config.resolved.yaml")
    L = backend.n_layers
    tok = backend.tokenizer
    decode = tok.decode
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lenses = {"logit_lens": None} if lens is None else {"jlens": lens, "logit_lens": None}
    primary = "jlens" if lens is not None else "logit_lens"

    def read(h: np.ndarray, m: int, which: str, task_ids: dict[str, int], answer_ids: np.ndarray) -> dict:
        z = transport(lenses[which], h, m, L)
        logits = backend.logits_from_residual(z.reshape(1, -1).to(backend.device))[0]
        lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        return readout_summary(lp, decode, top, task_ids, answer_ids)

    names = tasks or sorted(p.name for p in (run / "core" / "tasks").iterdir() if (p / "trajectories.json").exists())
    summary: dict[str, dict] = {}
    for task in names:
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
            row["natural"] = {k: read(nat, m, k, task_ids, answer_ids) for k in lenses}
            for c in ("learned", "fv"):
                key = f"L{layer}_mean_delta_{c}"
                if key in arrays.files and m >= layer:
                    row[c] = {primary: read(arrays[key][m].astype(np.float32), m, primary, task_ids, answer_ids)}
            gkey = f"L{layer}_mean_generic"
            if gkey in arrays.files and m >= layer:
                row["generic"] = {primary: read(arrays[gkey][m].astype(np.float32), m, primary, task_ids, answer_ids)}
            # random directions with the natural difference's norm: the floor of the masses
            R = rng.standard_normal((n_random, backend.hidden_size)).astype(np.float32)
            R *= np.linalg.norm(nat) / np.linalg.norm(R, axis=1, keepdims=True)
            rr = [read(r, m, primary, task_ids, answer_ids) for r in R]
            row["random"] = {primary: {"task_mass_mean": float(np.mean([r["task_mass"] for r in rr])),
                                       "answer_mass_mean": float(np.mean([r["answer_mass"] for r in rr])),
                                       "task_best_rank_median": float(np.median([r["task_best_rank"] for r in rr if r["task_best_rank"]])) if task_ids else None}}
            rows.append(row)
        inj = {c: {"read_point": layer, **{k: read(v.astype(np.float32), layer, k, task_ids, answer_ids) for k in lenses}}
               for c, v in injected.items()}
        result = {"task": task, "primary_layer": layer, "n_layers": L, "handover_read_point": hand, "task_words": task_ids,
                  "n_answer_tokens": int(len(np.unique(answer_ids))), "lenses": list(lenses), "rows": rows, "injected": inj}
        with open(out_dir / f"{task}.json", "w") as f:
            json.dump(result, f, indent=1)

        def fmt(r: dict) -> str:
            return (f"task-mass {r['task_mass']:.3f} (best '{r['task_best']}' rank {r['task_best_rank']}) answer-mass {r['answer_mass']:.3f} "
                    f"(rank {r['answer_best_rank']}) top {[t for t, _ in r['top'][:8]]}")
        print(f"\n== {task} L*={layer} hand-over {hand} (answers: {len(np.unique(answer_ids))} distinct first tokens; task words {list(task_ids)})", flush=True)
        for row in rows:
            m = row["read_point"]
            if m in (layer, hand, L) or m % 4 == 0:
                print(f"  m={m:2d} ({row['depth_fraction']:.2f}) natural/{primary}: {fmt(row['natural'][primary])} | random task-mass "
                      f"{row['random'][primary]['task_mass_mean']:.3f} answer-mass {row['random'][primary]['answer_mass_mean']:.3f}", flush=True)
                if "learned" in row:
                    print(f"            learned/{primary}: {fmt(row['learned'][primary])}", flush=True)
                if m == hand and lens is not None:
                    print(f"            natural/logit-lens: {fmt(row['natural']['logit_lens'])}", flush=True)
        for c, r in inj.items():
            print(f"  injected {c} at m={layer}: " + " | ".join(f"{k} {fmt(r[k])}" for k in lenses), flush=True)
        summary[task] = {"primary_layer": layer, "handover_read_point": hand, "lens": primary,
                         "natural_task_mass_by_read_point": [row["natural"][primary]["task_mass"] for row in rows],
                         "natural_task_best_rank_by_read_point": [row["natural"][primary]["task_best_rank"] for row in rows],
                         "natural_answer_mass_by_read_point": [row["natural"][primary]["answer_mass"] for row in rows],
                         "random_task_mass_by_read_point": [row["random"][primary]["task_mass_mean"] for row in rows],
                         "learned_task_mass_by_read_point": [row.get("learned", {}).get(primary, {}).get("task_mass") for row in rows]}
    out = {"run": str(run), "lens": lens_path, "model": backend.metadata()["name"], "tasks": summary}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_dir, flush=True)
    return out


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
    lens, lens_path = load_lens(backend, args.lens, args.lens_repo, args.lens_file)
    print(f"model {backend.metadata()['name']} ({backend.n_layers} layers, d={backend.hidden_size}); "
          + (f"lens {lens_path} fitted on {lens.n_prompts} prompts, blocks {min(lens.jacobians)}..{max(lens.jacobians)}" if lens else "logit lens only")
          + f"; load {time.time() - t0:.0f}s", flush=True)
    readout(run, backend, lens, Path(args.out_dir), args.tasks, args.top, args.n_random, args.model_config, lens_path)


if __name__ == "__main__":
    main()
