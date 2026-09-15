"""Worker network probe: download throughput to the hosts a run depends on, from the image's Python.

Run as the request command (``python scripts/probe_bandwidth.py``): no environment install, so it answers
in a minute whether a worker's egress to PyPI (the wheels), Hugging Face (the model) and a generic CDN is
healthy. Prints one line per host with the bytes fetched and the rate; also the image's torch and CUDA state.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request

BYTES = 30_000_000


def fetch(url: str, n: int = BYTES) -> None:
    t = time.time()
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "directions-probe/1.0"})
    try:
        body = urllib.request.urlopen(req, timeout=240).read()
    except Exception as exc:  # noqa: BLE001
        print(f"{url.split('/')[2]:32s} FAILED after {time.time() - t:.0f} s: {exc}", flush=True)
        return
    dt = time.time() - t
    print(f"{url.split('/')[2]:32s} {len(body) / 1e6:6.1f} MB in {dt:5.1f} s = {len(body) / 1e6 / dt:6.2f} MB/s", flush=True)


def main() -> int:
    try:
        import torch  # type: ignore[import-not-found]

        print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("torch unavailable:", exc, flush=True)
    fetch("https://speed.cloudflare.com/__down?bytes=%d" % BYTES)
    try:
        html = urllib.request.urlopen("https://pypi.org/simple/torch/", timeout=60).read().decode()
        wheels = [m for m in re.findall(r'href="(https://files\.pythonhosted\.org/[^"#]+)', html) if "manylinux" in m and "cp311" in m]
        fetch(wheels[-1])
    except Exception as exc:  # noqa: BLE001
        print("pypi index FAILED:", exc, flush=True)
    fetch("https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors")
    fetch("https://download.pytorch.org/whl/cu130/torch/")  # small: the index page, latency only
    return 0


if __name__ == "__main__":
    sys.exit(main())
