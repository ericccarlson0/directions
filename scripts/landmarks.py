"""The landmark test (docs/DECISIONS.md D37) on one model: run the landmark comparison
(configs/trajectories_landmarks.yaml: the patch test at every downstream read point and the pool's spectrum
at every read point), then the verbal-onset readout of the stored mean natural difference (scripts/verbalise.py,
logit lens; the Jacobian lens too when one is named), then, when a lens is named, the workspace band of the
paper's own lens-quality prompt sets (data/jlens_evaluations) read through that lens, and collect the model's
landmarks (directions.landmarks) into ``<run>/landmarks/landmarks.json``.

usage: uv run python scripts/landmarks.py --config configs/trajectories_landmarks.yaml \
           --fv-run <head-mean run> --learned-run <learned run> --run-id landmarks_<model>_seed<seed> \
           [--lens-file qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt | --lens <local .pt>] \
           [--eval-sets data/jlens_evaluations] [--run <finished comparison run: skip the comparison>]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verbalise  # noqa: E402  (scripts/verbalise.py)

from directions.config import load_config, load_trajectories_config  # noqa: E402
from directions.landmarks import band_from_rates, collect_model_landmarks, hit_rates, intermediate_forms, readout_prompt  # noqa: E402
from directions.model import ModelBackend  # noqa: E402
from directions.prompts import Prompt  # noqa: E402
from directions.tasks import Item  # noqa: E402
from directions.trajectories import run_trajectories  # noqa: E402
from directions.verbalise import Lens, transport  # noqa: E402

EVAL_SETS = ("multihop", "multilingual", "order-ops", "association", "typo", "poetry")


def first_token_ids(tokenizer: Any, forms: list[str]) -> list[int]:
    ids: list[int] = []
    for f in forms:
        enc = tokenizer.encode(f, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def lens_eval_band(backend: ModelBackend, lens: Lens, eval_dir: Path, ks: tuple[int, ...] = (1, 5, 10), batch_size: int = 16) -> dict[str, Any]:
    """The paper's lens-quality sets read through the lens at every read point: per set and pooled, the rate at
    which the intermediates rank within k, and the band those rates define (directions.landmarks.band_from_rates)."""
    L = backend.n_layers
    tok = backend.tokenizer
    out: dict[str, Any] = {"n_layers": L, "sets": {}, "pooled": {}}
    all_ranks: dict[str, list[np.ndarray]] = {}
    for name in EVAL_SETS:
        path = eval_dir / f"lens-eval-{name}.json"
        if not path.exists():
            continue
        items = json.load(open(path))["items"]
        prompts = [Prompt(prompt=readout_prompt(it["prompt"], name, "target" in it), target=" x", query=Item(it["prompt"], "x"), demos=())
                   for it in items]
        t0 = time.time()
        res = backend.run(prompts, capture=True, batch_size=batch_size)
        assert res.residuals is not None
        H = res.residuals  # (L+1, n, d) at the readout position
        ranks_set: dict[str, list[list[int]]] = {"jlens": [], "logit_lens": []}  # per intermediate: rank per read point
        names: list[str] = []
        for i, it in enumerate(items):
            for word in it["intermediates"]:
                ids = first_token_ids(tok, intermediate_forms(word, name))
                if not ids:
                    continue
                per_m: dict[str, list[int]] = {"jlens": [], "logit_lens": []}
                for m in range(1, L + 1):  # read points 1..L: the outputs of blocks 0..L-1, the lens's own layers
                    for which in per_m:
                        z = transport(lens if which == "jlens" else None, H[m, i], m, L)
                        logits = backend.logits_from_residual(z.reshape(1, -1).to(backend.device))[0].float()
                        order = torch.argsort(logits, descending=True)
                        pos = torch.empty_like(order)
                        pos[order] = torch.arange(len(order), device=order.device)
                        per_m[which].append(int(min(pos[j].item() for j in ids)) + 1)
                for which in per_m:
                    ranks_set[which].append(per_m[which])
                names.append(f"{it.get('name', i)}:{word}")
        entry: dict[str, Any] = {"n_items": len(items), "seconds": round(time.time() - t0, 1), "intermediates": names}
        for which in ranks_set:
            R = np.asarray(ranks_set[which], dtype=np.int64)
            all_ranks.setdefault(which, []).append(R)
            rates = {k: [None] + v for k, v in hit_rates(R, ks).items()}  # indexed by read point (none at 0)
            entry[which] = {"n_intermediates": int(R.shape[0]), "rates": rates, "band": {f"hit@{k}": band_from_rates(rates[f"hit@{k}"]) for k in ks},
                            "min_rank_over_read_points": [int(x) for x in R.min(1)]}
        out["sets"][name] = entry
        for which in ("jlens", "logit_lens"):
            b = entry[which]["band"]["hit@10"]
            print(f"lens band [{name}] {which:10s} {entry[which]['n_intermediates']} intermediates: hit@10 onset m={b['onset']} peak m={b['peak']} "
                  f"(max {b['max']:.2f}) exit m={b['exit']}; hit@1 max {max(v for v in entry[which]['rates']['hit@1'] if v is not None):.2f}", flush=True)
    for which, Rs in all_ranks.items():
        R = np.concatenate(Rs, 0)
        rates = {k: [None] + v for k, v in hit_rates(R, ks).items()}
        out["pooled"][which] = {"n_intermediates": int(R.shape[0]), "rates": rates, "band": {f"hit@{k}": band_from_rates(rates[f"hit@{k}"]) for k in ks}}
        b = out["pooled"][which]["band"]["hit@10"]
        print(f"lens band [pooled] {which}: {R.shape[0]} intermediates: hit@10 onset m={b['onset']} peak m={b['peak']} exit m={b['exit']} (max {b['max']:.2f})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trajectories_landmarks.yaml")
    ap.add_argument("--fv-run", required=True)
    ap.add_argument("--learned-run", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run", default=None, help="a finished landmark comparison run to read instead of running one")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", default=None)
    ap.add_argument("--eval-sets", default="data/jlens_evaluations")
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--out-dir", default=None, help="where the readouts and landmarks go (default <run>/landmarks)")
    args = ap.parse_args()

    if args.run:
        root = Path(args.run)
    else:
        cfg = load_trajectories_config(args.config)
        root = run_trajectories(cfg, args.fv_run, args.learned_run, run_id=args.run_id, config_path=args.config)
        # the comparison's model must leave the device before the readouts load theirs (a 7B model twice does not
        # fit a 24 GB card)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_dir = Path(args.out_dir) if args.out_dir else root / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_config(Path(args.learned_run) / "config.resolved.yaml")
    t0 = time.time()
    backend = ModelBackend(run_cfg.model, run_seed=run_cfg.seed)
    lens, lens_path = verbalise.load_lens(backend, args.lens, args.lens_repo, args.lens_file)
    print(f"landmarks: model {backend.metadata()['name']} ({backend.n_layers} layers); " + (f"lens {lens_path}" if lens else "logit lens only")
          + f"; load {time.time() - t0:.0f}s", flush=True)
    verbalise.readout(root, backend, lens, out_dir / "readout", n_random=args.n_random, lens_path=lens_path)
    band = None
    if lens is not None:
        band = lens_eval_band(backend, lens, Path(args.eval_sets))
        with open(out_dir / "lens_band.json", "w") as f:
            json.dump(band, f, indent=1)
    lm = collect_model_landmarks(root, out_dir / "readout", Path(args.fv_run), band, lens_key="jlens" if lens else "logit_lens")
    with open(out_dir / "landmarks.json", "w") as f:
        json.dump(lm, f, indent=1)
    s = lm["summary"]
    print(f"\nlandmarks of {lm['model']} (L={lm['n_layers']}): medians over tasks (read points)", flush=True)
    for k, v in s["median"].items():
        n = s["n_tasks"][k]
        off = s["offset_from_handover"].get(k, {})
        print(f"  {k:28s} {v if v is None else round(v, 1)!s:>6} (n={n})" + (f"  offset from hand-over median {off.get('median')} |median| {off.get('median_abs')}" if off else ""), flush=True)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
