"""Time the pieces of a layerwise control profile on the current machine (CPU and, if present, GPU).

    uv run python scripts/profile_bench.py [--model Qwen/Qwen3-0.6B-Base --task singular] [--repeats 8]

Synthetic residuals of the pilot's shape (read points x examples x width) time
``compute_profile`` on the NumPy and the torch paths, the batched spectra on
each, the raw decompositions (NumPy eigh/svd, torch eigh on the device with
each linear-algebra backend) and the remaining per-example arithmetic. With
``--model`` the same is timed on real residuals: one task's evaluation
prompts through the model with a random intervention, then a handful of
control profiles exactly as the pipeline computes them. Results go to
``results/bench_<utc>/bench.json`` (a run directory, so the RunPod path
uploads it) and to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DIRECTIONS_BLAS_THREADS", "4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from directions.config import EvaluationConfig  # noqa: E402
from directions.geometry import (  # noqa: E402
    block_responses,
    direction_readouts,
    layerwise_example_metrics,
    new_subspace_fraction,
    set_linalg_device,
    spectra,
)
from directions.layerwise import compute_profile  # noqa: E402


def timed(fn: Callable[[], Any], repeats: int, sync: bool = False) -> dict[str, float]:
    wall, cpu = [], []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        cpu.append(time.process_time() - c0)
    return {"min": min(wall), "median": statistics.median(wall), "max": max(wall), "cpu_median": statistics.median(cpu), "n": repeats}


def synthetic(L1: int, n: int, d: int, ls: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = (rng.standard_normal((L1, n, d)) * 20).astype(np.float32)
    delta = np.zeros((L1, n, d), dtype=np.float32)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    delta[ls] = 40 * v  # the injection, then a growing, rotating perturbation downstream
    for l in range(ls + 1, L1):
        delta[l] = 1.3 * delta[l - 1] + rng.standard_normal((n, d)).astype(np.float32) * (2 + l)
    return {"base": base, "steered": base + delta, "v": v,
            "task_directions": rng.standard_normal((L1, d)), "gradients": rng.standard_normal((L1, n, d)).astype(np.float32) * 1e-2}


def bench_arrays(arrs: dict[str, np.ndarray], ls: int, repeats: int, label: str) -> dict[str, Any]:
    cfg = EvaluationConfig()
    rng = np.random.default_rng(0)
    base, steered, v = arrs["base"], arrs["steered"], arrs["v"]
    kw = {"task_directions": arrs["task_directions"], "gradients": arrs["gradients"]}
    out: dict[str, Any] = {}
    delta = steered.astype(np.float64) - base.astype(np.float64)
    D = delta[ls:]
    L1, n, d = base.shape
    print(f"[{label}] shapes: read points {L1}, examples {n}, width {d}, intervention layer {ls}", flush=True)

    set_linalg_device(None)
    out["profile_numpy"] = timed(lambda: compute_profile(base, steered, v, ls, cfg, rng, with_ci=False, **kw), repeats)
    out["spectra_numpy_centered"] = timed(lambda: spectra(D, 0.9, True), repeats)
    out["spectra_numpy_uncentered"] = timed(lambda: spectra(D, 0.9, False), repeats)
    G = D @ np.transpose(D, (0, 2, 1))
    out["numpy_eigh_x%d" % (L1 - ls)] = timed(lambda: [np.linalg.eigh(g) for g in G], repeats)
    out["numpy_svd_vals_x%d" % (L1 - ls)] = timed(lambda: [np.linalg.svd(x, compute_uv=False) for x in D], repeats)
    out["numpy_gram_x%d" % (L1 - ls)] = timed(lambda: D @ np.transpose(D, (0, 2, 1)), repeats)
    ex = layerwise_example_metrics(delta, base, v)
    out["example_metrics"] = timed(lambda: layerwise_example_metrics(delta, base, v), repeats)
    out["direction_readouts"] = timed(lambda: direction_readouts(delta, kw["task_directions"], kw["gradients"]), repeats)
    b = block_responses(delta)
    tops = [s.top_subspace for s in spectra(D, 0.9, True)]
    out["new_subspace_fraction_x%d" % (L1 - ls - 1)] = timed(
        lambda: [new_subspace_fraction(b[ls + i], tops[i]) for i in range(L1 - ls - 1)], repeats)
    out["float64_conversion"] = timed(lambda: steered.astype(np.float64) - base.astype(np.float64), repeats)
    del ex

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        set_linalg_device(dev)
        out["profile_torch_cuda"] = timed(lambda: compute_profile(base, steered, v, ls, cfg, rng, with_ci=False, **kw), repeats, sync=True)
        out["spectra_torch_cuda_centered"] = timed(lambda: spectra(D, 0.9, True), repeats, sync=True)
        set_linalg_device(None)
        T = torch.as_tensor(D, dtype=torch.float64, device=dev)
        Gt = T @ T.transpose(1, 2)
        torch.cuda.synchronize()
        out["cuda_gram_f64"] = timed(lambda: T @ T.transpose(1, 2), repeats, sync=True)
        out["cuda_upload_f64"] = timed(lambda: torch.as_tensor(D, dtype=torch.float64, device=dev), repeats, sync=True)
        for backend in ("default", "cusolver", "magma"):
            try:
                torch.backends.cuda.preferred_linalg_library(backend)
                torch.linalg.eigh(Gt[:1])
                torch.cuda.synchronize()
                out[f"cuda_eigh_f64_batched_{backend}"] = timed(lambda: torch.linalg.eigh(Gt), repeats, sync=True)
                out[f"cuda_eigh_f64_loop_{backend}"] = timed(lambda: [torch.linalg.eigh(g) for g in Gt], repeats, sync=True)
                out[f"cuda_eigh_f32_batched_{backend}"] = timed(lambda: torch.linalg.eigh(Gt.float()), repeats, sync=True)
                out[f"cuda_svd_f64_{backend}"] = timed(lambda: torch.linalg.svd(T, full_matrices=False), repeats, sync=True)
            except Exception as e:  # a backend torch was not built with
                out[f"cuda_eigh_{backend}"] = {"error": f"{type(e).__name__}: {e}"}
        torch.backends.cuda.preferred_linalg_library("default")
        Tc = T.cpu()
        out["torch_cpu_eigh_f64_batched"] = timed(lambda: torch.linalg.eigh(Tc @ Tc.transpose(1, 2)), repeats)
    for k, r in out.items():
        if "median" in r:
            print(f"  {k:<40} median {r['median']*1e3:9.1f} ms  min {r['min']*1e3:9.1f}  max {r['max']*1e3:9.1f}  cpu {r['cpu_median']*1e3:9.1f}", flush=True)
        else:
            print(f"  {k:<40} {r}", flush=True)
    return out


def real_arrays(model: str, task_name: str, n: int, layer: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from directions.config import ModelConfig, PromptConfig
    from directions.geometry import random_unit_vector
    from directions.model import Intervention, ModelBackend
    from directions.prompts import zero_shot_prompt
    from directions.tasks import build_task, filter_items

    info: dict[str, Any] = {}
    t0 = time.perf_counter()
    backend = ModelBackend(ModelConfig(name=model, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=128))
    info["model_load_s"] = time.perf_counter() - t0
    pcfg = PromptConfig()
    task = build_task(task_name, {})
    kept, _ = filter_items(task, backend.target_token_count, pcfg.target_template, 4)
    prompts = [zero_shot_prompt(pcfg, it) for it in kept[:n]]
    rng = np.random.default_rng(0)
    v = random_unit_vector(rng, backend.hidden_size)
    t0 = time.perf_counter()
    base = backend.run(prompts, capture=True)
    info["baseline_forward_s"] = time.perf_counter() - t0
    alpha = 2.0 * float(np.median(np.linalg.norm(base.residuals[layer].astype(np.float64), axis=1)))
    t0 = time.perf_counter()
    steered = backend.run(prompts, interventions=[Intervention(layer, v, alpha)], capture=True)
    info["steered_forward_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    grads = backend.gradients(prompts)
    info["gradients_s"] = time.perf_counter() - t0
    info["model"] = backend.metadata()
    return {"base": base.residuals, "steered": steered.residuals, "v": v,
            "task_directions": rng.standard_normal((backend.n_layers + 1, backend.hidden_size)), "gradients": grads}, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="also time real residuals of this model")
    ap.add_argument("--task", default="singular")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--layer", type=int, default=6, help="intervention layer")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or f"results/bench_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "sched_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        env["numpy_blas"] = str(np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name"))
    except Exception:
        pass
    print(json.dumps(env, indent=1), flush=True)
    result: dict[str, Any] = {"environment": env, "args": vars(args)}
    rng = np.random.default_rng(0)
    result["synthetic"] = bench_arrays(synthetic(args.layers + 1, args.n, args.d, args.layer, rng), args.layer, args.repeats, "synthetic")
    if args.model:
        arrs, info = real_arrays(args.model, args.task, args.n, args.layer)
        print(f"[real] {json.dumps({k: v for k, v in info.items() if k != 'model'})}", flush=True)
        result["real"] = {"info": info, **bench_arrays(arrs, args.layer, args.repeats, "real")}
    with open(out_dir / "bench.json", "w") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=str)
    print(f"wrote {out_dir / 'bench.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
