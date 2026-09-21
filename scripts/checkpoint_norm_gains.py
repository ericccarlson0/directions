"""Read the normalisation weights (and per-block output scalars) of a Hugging Face checkpoint without downloading
it, through HTTP range requests into its safetensors files, and summarise the gain each block applies to the
residual coordinates it reads (docs/DECISIONS.md D35: why Gemma 4's deep half is hypersensitive to residual
edits).

For every requested layer and every norm module found in it, prints the weight's median and maximum, how many
coordinates exceed a threshold, and the weight at the given coordinates (e.g. a massive-activation coordinate).
The final norm and, where present, the per-block output scalar are printed too.

usage: uv run python scripts/checkpoint_norm_gains.py google/gemma-4-12B --layers 0 4 8 12 16 24 32 40 47 --coords 1750
       uv run python scripts/checkpoint_norm_gains.py Qwen/Qwen3-8B-Base --layers 18 30 --threshold 5
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess

import numpy as np

HUB = "https://huggingface.co"


def fetch(url: str, a: int, b: int) -> bytes:
    return subprocess.run(["curl", "-sS", "-L", "-r", f"{a}-{b}", url], capture_output=True, check=True).stdout


def header(url: str) -> tuple[dict, int]:
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    return json.loads(fetch(url, 8, 8 + n - 1)), 8 + n


def tensor(url: str, hdr: dict, base: int, key: str) -> np.ndarray:
    info = hdr[key]
    a, b = info["data_offsets"]
    raw = fetch(url, base + a, base + b - 1)
    if info["dtype"] == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if info["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if info["dtype"] == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    raise ValueError(f"unsupported dtype {info['dtype']} for {key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--coords", type=int, nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=50.0)
    args = ap.parse_args()

    root = f"{HUB}/{args.repo}/resolve/{args.revision}"
    try:
        index = json.loads(subprocess.run(["curl", "-sS", "-L", f"{root}/model.safetensors.index.json"],
                                          capture_output=True, check=True).stdout)["weight_map"]
    except (json.JSONDecodeError, KeyError):
        index = None  # a single model.safetensors
    headers: dict[str, tuple[dict, int]] = {}

    def get(key: str) -> np.ndarray | None:
        shard = index.get(key) if index else "model.safetensors"
        if shard is None:
            return None
        url = f"{root}/{shard}"
        if url not in headers:
            headers[url] = header(url)
        hdr, base = headers[url]
        return tensor(url, hdr, base, key) if key in hdr else None

    # the layer prefix (a multimodal wrapper keeps the decoder under language_model)
    probe_url = f"{root}/{index[next(iter(index))] if index else 'model.safetensors'}"
    headers[probe_url] = header(probe_url)
    keys = list(headers[probe_url][0]) if not index else list(index)
    prefix = "model.language_model.layers" if any(k.startswith("model.language_model.layers") for k in keys) else "model.layers"
    norm_names = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]
    thr = args.threshold
    print(f"{args.repo}: norm weights per layer (threshold {thr:g}; coords {args.coords})")
    for layer in args.layers:
        parts = []
        for name in norm_names:
            w = get(f"{prefix}.{layer}.{name}.weight")
            if w is None:
                continue
            at = " ".join(f"w[{c}]={w[c]:+.3f}" for c in args.coords if c < w.size)
            parts.append(f"{name}: median {np.median(w):.3f} max {w.max():.1f} n>{thr:g} {int((w > thr).sum())} {at}")
        s = get(f"{prefix}.{layer}.layer_scalar")
        if s is not None:
            parts.append(f"output scalar {float(s[0]):.3f}")
        print(f"  layer {layer:3d} | " + " | ".join(parts))
    fn = get("model.language_model.norm.weight") if prefix.startswith("model.language_model") else get("model.norm.weight")
    if fn is not None:
        top = np.argsort(-fn)[:5]
        at = " ".join(f"w[{c}]={fn[c]:+.3f}" for c in args.coords if c < fn.size)
        print(f"  final norm | median {np.median(fn):.3f} max {fn.max():.1f} n>{thr:g} {int((fn > thr).sum())} {at}; "
              f"top coords {top.tolist()} = {fn[top].round(1).tolist()}")


if __name__ == "__main__":
    main()
