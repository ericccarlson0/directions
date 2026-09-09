"""Flash entrypoint: runs an arbitrary command inside the deployed artifact and leaves the results on a network volume.

Deployed by ``flash deploy`` (see ``.github/workflows/run-gpu.yml``). The artifact root is the repository root;
``src/`` is put on ``sys.path`` and ``PYTHONPATH`` for the child process; the research CLI never imports Flash.

A network volume (mounted at ``/runpod-volume``) receives ``results/<results_name>`` and holds the Hugging Face
cache. The runner downloads results through RunPod's S3-compatible API. Without a mounted volume (local or smoke
use) the results are returned inside the job output as a base64 tarball (with ``max_output_mb``).

Build-time knobs:
* ``DIRECTIONS_GPU_TIER`` - a ``GpuGroup`` member name (default ``ADA_24``); a memory class whose pool can
  mix architectures (``AMPERE_16`` holds Ampere and Ada cards; only ``ADA_24`` is a single card).
* ``DIRECTIONS_GPU_TYPE`` - an exact device name (a ``GpuType`` value, e.g. ``NVIDIA RTX A4500``); when set it
  pins that card and overrides ``DIRECTIONS_GPU_TIER``.
* ``DIRECTIONS_DATACENTER`` - where the volume lives (default ``EUR-NO-1``); the endpoint is pinned to it.
* ``DIRECTIONS_VOLUME_NAME`` / ``DIRECTIONS_VOLUME_GB`` - volume name (default ``directions``) and size (50).
* ``DIRECTIONS_MIN_CUDA`` - lowest driver CUDA version a worker's host may have (default ``13.0``). The
  pipeline installs torch from ``uv.lock`` at job start, and that torch is a CUDA 13 build; on a host with an
  older driver it cannot see the GPU and ``device: auto`` would silently run on the CPU (observed: 50-170x
  slower). Raise this if the lock moves to a newer CUDA build.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from runpod_flash import Endpoint, GpuGroup, GpuType, NetworkVolume

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VOLUME_ROOT = Path("/runpod-volume")

GPU = (
    GpuType(os.environ["DIRECTIONS_GPU_TYPE"])
    if os.environ.get("DIRECTIONS_GPU_TYPE")
    else GpuGroup[os.environ.get("DIRECTIONS_GPU_TIER", "ADA_24")]
)
VOLUME = NetworkVolume(
    name=os.environ.get("DIRECTIONS_VOLUME_NAME", "directions"),
    size=int(os.environ.get("DIRECTIONS_VOLUME_GB", "50")),
    datacenter=os.environ.get("DIRECTIONS_DATACENTER", "EUR-NO-1"),
)


@Endpoint(
    name="directions-runner",
    gpu=GPU,
    volume=VOLUME,
    workers=(0, 2),  # scale to zero; one experiment at a time, plus room for a short probe job alongside it
    min_cuda_version=os.environ.get("DIRECTIONS_MIN_CUDA", "13.0"),
    idle_timeout=30,  # seconds a warm worker lingers
    execution_timeout_ms=0,  # unlimited; the GitHub job enforces the wall clock
    # FlashBoot restores workers from a snapshot of an earlier boot on the same image, /app included, so
    # after a redeploy workers ran the previous build (observed 2026-09-09). Off: every worker unpacks the
    # current artifact at start.
    flashboot=False,
)
async def run_experiment(
    argv: list[str],
    results_dir: str = "results",
    max_output_mb: float = 8.0,
    git_commit: str | None = None,
    results_name: str | None = None,
) -> dict:
    """Run ``argv`` at the artifact root; results go to the volume: ``results/<results_name>``."""
    from directions.remote import artifact_fingerprints, execute

    extra_env = {"PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}
    if git_commit:
        extra_env["DIRECTIONS_GIT_COMMIT"] = git_commit
    if VOLUME_ROOT.is_dir():
        extra_env["HF_HOME"] = str(VOLUME_ROOT / "hf")
    result = execute(
        argv=argv,
        cwd=ROOT,
        results_dir=results_dir,
        max_output_mb=max_output_mb,
        extra_env=extra_env,
        volume_root=VOLUME_ROOT,
        results_name=results_name,
    )
    result.update(artifact_fingerprints(ROOT))
    return result
