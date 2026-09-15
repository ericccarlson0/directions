"""Flash entrypoint: runs an arbitrary command inside the source tree a job ships and leaves the results on a
network volume.

Deployed by ``flash deploy`` from a staging directory that holds only this file and the ``directions.remote``
helpers (``scripts/runpod_stage.py``; see ``.github/workflows/run-gpu.yml``), so the artifact root ``/app``
contains no experiment code. The research CLI never imports Flash.

The runner ships the checked-out commit inside every job (``source_tar_b64`` + ``source_sha256``,
``directions.remote.pack_source``); the handler verifies the digest, unpacks it under ``/app/jobs/<digest>`` and
runs the command there with that tree's ``src/`` on ``PYTHONPATH``.

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
    # off: every worker unpacks the current artifact at start
    flashboot=False,
)
async def run_experiment(
    argv: list[str],
    results_dir: str = "results",
    max_output_mb: float = 8.0,
    git_commit: str | None = None,
    results_name: str | None = None,
    source_tar_b64: str | None = None,
    source_sha256: str | None = None,
    **unknown_inputs: object,
) -> dict:
    """Run ``argv`` at the root of the shipped tree (else the artifact); results go to ``results/<results_name>``.

    Unknown inputs are reported, not rejected: a worker may hold an older handler than the runner expects
    (docs/INFRA.md), and a signature error would surface as a FAILED job instead of a diagnosable output.
    """
    from directions.remote import (
        SOURCE_JOBS_DIR,
        WHEELHOUSE_SUBDIR,
        artifact_fingerprints,
        execute,
        offline_argv,
        stale_worker,
        unpack_source,
    )

    fingerprints = artifact_fingerprints(ROOT)
    info: dict = {"handler_stale": stale_worker(fingerprints), **fingerprints}
    if unknown_inputs:
        info["unknown_inputs"] = sorted(unknown_inputs)
        print(f"ignoring unknown job inputs {info['unknown_inputs']} (handler older than the runner?)")
    if not source_tar_b64:
        # The artifact holds only this handler; there is nothing else to run.
        raise ValueError("the job did not ship its source tree (scripts/runpod_job.py always does)")
    root = unpack_source(source_tar_b64, source_sha256, ROOT / SOURCE_JOBS_DIR)
    info.update({"source_sha256": source_sha256, "source_dir": str(root)})
    print(f"running shipped source {str(source_sha256)[:12]} in {root}")

    extra_env = {"PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"}
    if git_commit:
        extra_env["DIRECTIONS_GIT_COMMIT"] = git_commit
    if VOLUME_ROOT.is_dir():
        extra_env["HF_HOME"] = str(VOLUME_ROOT / "hf")
    wheelhouse = VOLUME_ROOT / WHEELHOUSE_SUBDIR
    info["wheelhouse"] = str(wheelhouse) if wheelhouse.is_dir() else None
    if wheelhouse.is_dir():
        # the locked wheels are on the volume: build the environment from them instead of from PyPI
        argv = offline_argv(argv, wheelhouse)
        print(f"wheelhouse {wheelhouse}: building the environment offline")
    result = execute(
        argv=argv,
        cwd=root,
        results_dir=results_dir,
        max_output_mb=max_output_mb,
        extra_env=extra_env,
        volume_root=VOLUME_ROOT,
        results_name=results_name,
    )
    result.update(info)
    return result
