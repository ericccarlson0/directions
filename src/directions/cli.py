"""Command-line interface.

    uv run directions validate --config configs/pilot_qwen3_0.6b.yaml
    uv run directions pilot    --config configs/pilot_qwen3_0.6b.yaml
    uv run directions check    --config configs/pilot_qwen3_0.6b.yaml
    uv run directions compare  results/<run_a> results/<run_b>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import config_to_dict, load_config

# top-level fields that legitimately differ between two otherwise identical runs
_VOLATILE = {"run_id", "argv", "started_utc", "finished_utc", "timings_seconds", "config_path"}


def _cmd_run(args: argparse.Namespace, command: str) -> int:
    from .pipeline import run_pipeline

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    root = run_pipeline(cfg, command, config_path=str(args.config), run_id=args.run_id)
    print(root)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(json.dumps(config_to_dict(cfg), indent=2))
    return 0


def _load_tree(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in sorted(root.rglob("*.json")):
        rel = str(p.relative_to(root))
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in _VOLATILE}
            if rel == "metadata.json" and isinstance(data.get("config"), dict):
                data["config"] = {k: v for k, v in data["config"].items() if k != "output_dir"}
        out[rel] = data
    with open(root / "rejections.jsonl") as f:
        out["rejections.jsonl"] = [json.loads(line) for line in f if line.strip()]
    return out


def _diff(a: Any, b: Any, path: str, out: list[str], atol: float) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'A' if k in a else 'B'}")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, atol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > atol:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare two run directories; exit 0 when every JSON output matches."""
    a, b = _load_tree(Path(args.run_a)), _load_tree(Path(args.run_b))
    diffs: list[str] = []
    _diff(a, b, "", diffs, args.atol)
    if diffs:
        print(f"{len(diffs)} difference(s):")
        for d in diffs[: args.max_lines]:
            print("  " + d)
        if len(diffs) > args.max_lines:
            print(f"  ... ({len(diffs) - args.max_lines} more)")
        return 1
    print(f"identical ({len(a)} files compared, atol={args.atol})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="directions", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("validate", "task/control validation only (stages 1-5)"),
        ("pilot", "full measurement-validation pilot (validation + layerwise + exploratory + figures)"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True, help="YAML config file")
        p.add_argument("--output-dir", default=None, help="override output_dir from the config")
        p.add_argument("--run-id", default=None, help="explicit run directory name (default: timestamp + config hash)")
    p = sub.add_parser("check", help="parse and print the resolved config")
    p.add_argument("--config", required=True)
    p = sub.add_parser("compare", help="diff the JSON outputs of two runs (reproducibility check)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--max-lines", type=int, default=50)
    args = parser.parse_args(argv)
    if args.command in ("validate", "pilot"):
        return _cmd_run(args, args.command)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "compare":
        return _cmd_compare(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
