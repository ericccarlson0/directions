"""Run directories and structured, JSON-serializable result output.

Layout of a run directory::

    results/<timestamp>__<run-name>__<config-fingerprint>/
        config.resolved.yaml     fully resolved configuration
        metadata.json            git commit, environment, packages, seeds
        summary.json             top-level outcome of the run
        core/                    preregistered measurements
        exploratory/             secondary / diagnostic analyses
        figures/                 generated plots
        rejections.jsonl         every automatic filtering decision
        log.txt                  human-readable progress log
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from .metadata import utc_now


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


def _sanitize(obj):
    """Replace non-finite floats with ``None`` so the JSON stays standard."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class RunDirectory:
    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("", "core", "exploratory", "figures"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._log_path = self.root / "log.txt"
        self._rejections = self.root / "rejections.jsonl"

    @classmethod
    def create(cls, cfg, command: str) -> "RunDirectory":
        stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
        name = f"{stamp}__{command}__{cfg.run.name}__{cfg.fingerprint()}"
        return cls(Path(cfg.run.output_root) / name)

    # -- writers --------------------------------------------------------
    def write_json(self, relpath: str, payload) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_sanitize(payload), indent=2, cls=NumpyJSONEncoder, allow_nan=False) + "\n"
        )
        return path

    def write_yaml(self, relpath: str, payload) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_sanitize(payload), sort_keys=False, allow_unicode=True))
        return path

    def write_npz(self, relpath: str, **arrays) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def log(self, message: str, echo: bool = True) -> None:
        line = f"[{utc_now()}] {message}"
        with self._log_path.open("a") as fh:
            fh.write(line + "\n")
        if echo:
            print(line, flush=True)

    def reject(self, stage: str, subject: str, reason: str, detail: dict | None = None) -> None:
        """Record an automatic filtering decision or a recoverable error."""
        record = {
            "time": utc_now(),
            "stage": stage,
            "subject": subject,
            "reason": reason,
            "detail": _sanitize(detail or {}),
        }
        with self._rejections.open("a") as fh:
            fh.write(json.dumps(record, cls=NumpyJSONEncoder, allow_nan=False) + "\n")
        self.log(f"REJECT [{stage}] {subject}: {reason}")

    def figure_path(self, name: str, ext: str) -> Path:
        return self.root / "figures" / f"{name}.{ext}"
