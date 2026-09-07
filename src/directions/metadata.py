"""Provenance capture: git state, environment, package versions, seeds."""

from __future__ import annotations

import importlib.metadata as im
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "numpy",
    "accelerate",
    "safetensors",
    "matplotlib",
    "pyyaml",
    "typer",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_metadata() -> dict:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(status) if status is not None else None,
        "uncommitted_files": status.splitlines() if status else [],
    }


def package_versions() -> dict:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            out[name] = None
    return out


def environment_metadata() -> dict:
    info = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "env": {
            k: os.environ[k]
            for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "PYTORCH_CUDA_ALLOC_CONF")
            if k in os.environ
        },
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
        }
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        info["torch"] = {"error": repr(exc)}
    return info


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_metadata(cfg, extra: dict | None = None) -> dict:
    return {
        "created_utc": utc_now(),
        "git": git_metadata(),
        "environment": environment_metadata(),
        "config_fingerprint": cfg.fingerprint(),
        "seeds": {
            "run_seed": cfg.run.seed,
            "derivation": (
                "all sub-generators are numpy SeedSequence children of run.seed, keyed by "
                "(purpose, task, layer/seed index); torch is seeded once from run.seed"
            ),
        },
        **(extra or {}),
    }
