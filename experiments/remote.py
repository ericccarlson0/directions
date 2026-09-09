"""Flash entrypoint: runs an arbitrary command inside the deployed artifact and returns the results directory.

Deployed by ``flash deploy`` (see ``.github/workflows/run-gpu.yml``). The artifact root is the repository root;
``src/`` is put on ``sys.path`` and ``PYTHONPATH`` for the child process; the research CLI never imports Flash.

Build-time knobs:
* ``DIRECTIONS_GPU_TIER`` - a ``GpuGroup`` member name (default ``ADA_24``); a memory class whose pool can
  mix architectures (``AMPERE_16`` holds Ampere and Ada cards; only ``ADA_24`` is a single card).
* ``DIRECTIONS_GPU_TYPE`` - an exact device name (a ``GpuType`` value, e.g. ``NVIDIA RTX A4500``); when set it 
  pins that card and overrides ``DIRECTIONS_GPU_TIER``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from runpod_flash import Endpoint, GpuGroup, GpuType

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GPU = (
    GpuType(os.environ["DIRECTIONS_GPU_TYPE"])
    if os.environ.get("DIRECTIONS_GPU_TYPE")
    else GpuGroup[os.environ.get("DIRECTIONS_GPU_TIER", "ADA_24")]
)


@Endpoint(
    name="directions-runner",
    gpu=GPU,
    workers=(0, 1),  # scale to zero; one experiment at a time
    idle_timeout=30,  # seconds a warm worker lingers
    execution_timeout_ms=0,  # unlimited; the GitHub job enforces the wall clock
    flashboot=True,
)
async def run_experiment(
    argv: list[str],
    results_dir: str = "results",
    max_output_mb: float = 8.0,
    git_commit: str | None = None,
) -> dict:
    """Run ``argv`` at the artifact root; return log tail + base64 results tarball."""
    from directions.remote import execute

    extra_env = {"PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}
    if git_commit:
        extra_env["DIRECTIONS_GIT_COMMIT"] = git_commit
    return execute(
        argv=argv,
        cwd=ROOT,
        results_dir=results_dir,
        max_output_mb=max_output_mb,
        extra_env=extra_env,
    )
