"""Remote-execution helpers for running experiments on RunPod.

Two halves:

* the worker side (``experiments/remote.py``) calls :func:`execute` to run one command inside the deployed
  artifact and return the results directory as a base64 tarball in the job output;
* the runner side (``scripts/runpod_job.py``) calls :func:`unpack_results` to turn that tarball back into files.

Everything here is standard-library only (testable without the Flash SDK or a GPU).
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

LOG_TAIL_BYTES = 64 * 1024
VOLUME_RESULTS_SUBDIR = "results"
RUNNER_LOG_NAME = "runner.log"
METADATA_NAME = "run_metadata.json"


class ResultsTooLarge(RuntimeError):
    """The results directory does not fit within the job-output budget."""


def _tar_directory(results_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(p for p in results_dir.rglob("*") if p.is_file()):
            tar.add(path, arcname=str(path.relative_to(results_dir)))
    return buf.getvalue()


def _largest_files(results_dir: Path, n: int = 5) -> list[tuple[str, int]]:
    sizes = [
        (str(p.relative_to(results_dir)), p.stat().st_size)
        for p in results_dir.rglob("*")
        if p.is_file()
    ]
    return sorted(sizes, key=lambda item: item[1], reverse=True)[:n]


def pack_results(results_dir: Path, max_output_mb: float) -> dict[str, Any]:
    """tar+gzip+base64 ``results_dir`` for transport inside a JSON job output.

    Raises :class:`ResultsTooLarge` if the encoded payload exceeds ``max_output_mb``. A missing directory yields 
    an empty bundle rather than an error so a command that produced no output still reports.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return {
            "results_tar_b64": None,
            "results_size_bytes": 0,
            "results_file_count": 0,
        }

    files = [p for p in results_dir.rglob("*") if p.is_file()]
    encoded = base64.b64encode(_tar_directory(results_dir)).decode("ascii")
    limit = int(max_output_mb * 1024 * 1024)
    if len(encoded) > limit:
        biggest = ", ".join(f"{name} ({size / 1e6:.1f} MB)" for name, size in _largest_files(results_dir))
        raise ResultsTooLarge(
            f"encoded results are {len(encoded) / 1e6:.1f} MB, over the {max_output_mb} MB budget; "
            f"largest files: {biggest}"
        )
    return {
        "results_tar_b64": encoded,
        "results_size_bytes": len(encoded),
        "results_file_count": len(files),
    }


def unpack_results(results_tar_b64: str, dest: Path) -> None:
    """Inverse of :func:`pack_results`; refuses archive members that escape ``dest``."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(results_tar_b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if target != dest and dest not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.name!r}")
        tar.extractall(dest)


def _environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:  # torch is provided by the worker base image; absent locally is fine
        import torch  # type: ignore[import-not-found]

        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on host
        meta["torch"] = f"unavailable: {exc}"
    return meta


def artifact_fingerprints(root: Path) -> dict[str, str | None]:
    """The source fingerprint of the code on disk versus the one the endpoint was deployed with.

    Flash writes ``source_fingerprint`` into ``flash_manifest.json`` at build time and sets
    ``_FLASH_SOURCE_FINGERPRINT`` in the endpoint's env at deploy time; they differ when a worker runs a
    stale unpacked artifact.
    """
    manifest = Path(root) / "flash_manifest.json"
    on_disk = None
    if manifest.is_file():
        try:
            on_disk = json.loads(manifest.read_text()).get("source_fingerprint")
        except (OSError, ValueError):
            on_disk = None
    return {"code_fingerprint": on_disk, "expected_fingerprint": os.environ.get("_FLASH_SOURCE_FINGERPRINT")}


def _run_streaming(
    argv: list[str], cwd: Path, env: dict[str, str], log_path: Path, live_copy: Path | None = None
) -> tuple[int, str]:
    """Run ``argv``, relaying its combined output line by line to our stdout and to ``log_path`` as it arrives.

    Relaying to stdout puts the child's progress in the worker's container log, so a long run can be watched 
    from outside; ``live_copy`` (where the logs live on the network volume) receives every line, flushed, so the 
    runner can tail it over S3 throughout the run.
    Returns the exit code and the last ``LOG_TAIL_BYTES`` of output.
    """
    tail: list[bytes] = []
    tail_size = 0
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout is not None
    with log_path.open("wb") as log_file, (live_copy.open("wb") if live_copy else open(os.devnull, "wb")) as live:
        for line in proc.stdout:
            log_file.write(line)
            live.write(line)
            live.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            tail.append(line)
            tail_size += len(line)
            while tail_size > 2 * LOG_TAIL_BYTES and len(tail) > 1:
                tail_size -= len(tail.pop(0))
    returncode = proc.wait()
    return returncode, b"".join(tail).decode("utf-8", errors="replace")


def execute(
    argv: list[str],
    cwd: Path,
    results_dir: str = "results",
    max_output_mb: float = 8.0,
    extra_env: dict[str, str] | None = None,
    volume_root: Path | None = None,
    results_name: str | None = None,
) -> dict[str, Any]:
    """Run ``argv`` in ``cwd``; keep ``results_dir`` on the volume, or bundle it for the job output.

    The full combined stdout/stderr is written to ``<results_dir>/runner.log``.
    A ``run_metadata.json`` (argv, timings, exit code, environment, git commit) is placed beside the results;
    the artifact is self-describing.
    """
    cwd = Path(cwd)
    on_volume = bool(results_name) and volume_root is not None and Path(volume_root).is_dir()
    out_dir = cwd / results_dir  # local disk during the run; a network volume is slow for many small writes
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **(extra_env or {})}

    started = time.time()
    log_path = out_dir / RUNNER_LOG_NAME
    live_copy = None
    if on_volume:
        target = Path(volume_root) / VOLUME_RESULTS_SUBDIR / results_name
        target.mkdir(parents=True, exist_ok=True)
        live_copy = target / RUNNER_LOG_NAME
    try:
        returncode, log = _run_streaming(argv, cwd, env, log_path, live_copy)
    except OSError as exc:
        returncode = 127
        log = f"failed to start {argv!r}: {exc}\n"
        log_path.write_text(log)
    duration = time.time() - started

    metadata = {
        "argv": argv,
        "returncode": returncode,
        "started_unix": started,
        "duration_s": duration,
        "git_commit": env.get("DIRECTIONS_GIT_COMMIT"),
        "results_on_volume": on_volume,
        "environment": _environment_metadata(),
    }
    (out_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2))

    result: dict[str, Any] = {
        "returncode": returncode,
        "duration_s": duration,
        "log_tail": log[-LOG_TAIL_BYTES:],
    }
    if on_volume:
        shutil.copytree(out_dir, target, dirs_exist_ok=True)
        files = [p for p in target.rglob("*") if p.is_file()]
        result.update(
            {
                "results_tar_b64": None,
                "results_path": f"{VOLUME_RESULTS_SUBDIR}/{results_name}",
                "results_file_count": len(files),
                "results_size_bytes": sum(p.stat().st_size for p in files),
            }
        )
        return result
    try:
        result.update(pack_results(out_dir, max_output_mb))
    except ResultsTooLarge as exc:
        result.update({"results_tar_b64": None, "results_error": str(exc)})
    return result
