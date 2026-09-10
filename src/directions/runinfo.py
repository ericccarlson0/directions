"""Run directories, metadata and JSON serialisation helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import Config, config_to_dict


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    if commit is None and os.environ.get("DIRECTIONS_GIT_COMMIT"):
        # A shipped `git archive` tree has no .git; the RunPod handler passes the commit it was given.
        return {"commit": os.environ["DIRECTIONS_GIT_COMMIT"], "dirty": False, "branch": None, "source": "shipped"}
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _model_slug(cfg: Config) -> str:
    if cfg.model.backend == "toy":
        return "toy"
    return cfg.model.name.split("/")[-1].replace(".", "_").lower()


def config_hash(cfg: Config) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def make_run_dir(cfg: Config, command: str, run_id: str | None = None) -> Path:
    """Create ``<output_dir>/<run_id>`` with a unique, informative id."""
    if run_id is None:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{command}_{cfg.name}_{_model_slug(cfg)}_{config_hash(cfg)[:8]}"
    root = Path(cfg.output_dir) / run_id
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")
    for sub in ("core/tasks", "exploratory/tasks", "figures"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_resolved_config(path: Path, cfg: Config) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


class RejectionLog:
    """Append-only JSONL log of every automatic filtering decision."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        path.write_text("")

    def add(self, stage: str, task: str, reason: str, details: dict[str, Any] | None = None) -> None:
        entry = {"stage": stage, "task": task, "reason": reason, "details": _jsonable(details or {})}
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("directions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
