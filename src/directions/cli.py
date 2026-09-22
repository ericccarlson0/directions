"""Command-line interface.

    uv run directions validate --config configs/pilot_qwen3_0.6b.yaml
    uv run directions pilot    --config configs/pilot_qwen3_0.6b.yaml
    uv run directions check    --config configs/pilot_qwen3_0.6b.yaml
    uv run directions compare  results/<run_a> results/<run_b>
    uv run directions aggregate results/<run_1> results/<run_2> ... --out results/aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# The layerwise statistics are thousands of small (n x n, n x d) linear-algebra
# calls; with one BLAS thread per core they spend their time in thread
# synchronisation (a 192 x 1024 profile: ~1 s with 4 threads, ~70 s with 32).
# Cap the BLAS pool unless the user has set it explicitly. Must run before numpy
# is imported anywhere in this process.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DIRECTIONS_BLAS_THREADS", "4"))

from .config import config_to_dict, load_config  # noqa: E402

# top-level fields that legitimately differ between two otherwise identical runs
_VOLATILE = {"run_id", "argv", "started_utc", "finished_utc", "timings_seconds", "profile", "config_path"}


def _cmd_run(args: argparse.Namespace, command: str) -> int:
    from .pipeline import run_pipeline

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)
    root = run_pipeline(cfg, command, config_path=str(args.config), run_id=args.run_id, stop_after=args.stop_after)
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


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Summarise several runs of the same config (different seeds) per task."""
    from .aggregate import aggregate_runs, format_table

    result = aggregate_runs([Path(p) for p in args.runs])
    print(format_table(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


def _cmd_trajectories(args: argparse.Namespace) -> int:
    """All-to-all downstream-trajectory comparison from two finished runs (docs/DECISIONS.md D32)."""
    from .config import load_trajectories_config
    from .trajectories import run_trajectories

    cfg = load_trajectories_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)
    root = run_trajectories(cfg, args.fv_run, args.learned_run, run_id=args.run_id, config_path=str(args.config))
    print(root)
    return 0


def _cmd_mixing(args: argparse.Namespace) -> int:
    """The geometry test (docs/DECISIONS.md D40): a control mixed between two labels of a family."""
    from .config import load_mixing_config
    from .mixing import run_mixing

    cfg = load_mixing_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)
    runs: dict[str, tuple[str, str | None]] = {}
    for spec in args.runs:
        family, _, rest = spec.partition("=")
        learned, _, fv = rest.partition(":")
        if not family or not learned:
            raise SystemExit(f"--runs entries are family=<learned run>[:<head-mean run>], got {spec!r}")
        runs[family] = (learned, fv or None)
    root = run_mixing(cfg, runs, run_id=args.run_id, config_path=str(args.config))
    print(root)
    return 0


def _cmd_source(args: argparse.Namespace) -> int:
    """The source test (docs/DECISIONS.md D38): attention against MLP writes of the aligning increments."""
    from .config import load_source_config
    from .source import run_source

    cfg = load_source_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)
    root = run_source(cfg, args.fv_run, args.learned_run, landmark_run=args.landmark_run, run_id=args.run_id, config_path=str(args.config))
    print(root)
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
        p.add_argument("--seed", type=int, default=None, help="override the run seed from the config (recorded in the resolved config)")
        p.add_argument("--stop-after", default=None, choices=["fewshot", "extraction"],
                       help="stop every task after this stage (a smoke run of a new model: the load, the tokenisation of the "
                            "targets and the few-shot gates, or also the extraction); the run is marked incomplete")
    p = sub.add_parser("check", help="parse and print the resolved config")
    p.add_argument("--config", required=True)
    p = sub.add_parser("compare", help="diff the JSON outputs of two runs (reproducibility check)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--max-lines", type=int, default=50)
    p = sub.add_parser("aggregate", help="summarise several runs (e.g. different seeds) per task")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None, help="write the aggregate as JSON")
    p = sub.add_parser("trajectories", help="all-to-all downstream-trajectory comparison of the three constructions "
                                            "and the natural few-shot trajectory, from two finished runs of one model")
    p.add_argument("--config", required=True, help="YAML config of the comparison (configs/trajectories.yaml)")
    p.add_argument("--fv-run", required=True, help="finished run with extraction.control: function_vector")
    p.add_argument("--learned-run", required=True, help="finished run with extraction.control: learned_vector (same seed)")
    p.add_argument("--output-dir", default=None, help="override output_dir from the config")
    p.add_argument("--run-id", default=None, help="explicit run directory name")
    p.add_argument("--seed", type=int, default=None, help="override the comparison's own seed from the config (its draws: "
                                                          "second demonstration sample, derangements, isotropic directions, bootstraps)")
    p = sub.add_parser("source", help="the source test (D38): which sublayer writes the increments that turn the injected "
                                      "direction into the natural one, and whether it is necessary")
    p.add_argument("--config", required=True, help="YAML config of the test (configs/source.yaml)")
    p.add_argument("--fv-run", required=True, help="finished run with extraction.control: function_vector (the selected heads)")
    p.add_argument("--learned-run", required=True, help="finished run with extraction.control: learned_vector (same seed)")
    p.add_argument("--landmark-run", default=None, help="finished landmark run (D37): the hand-over read point per task")
    p.add_argument("--output-dir", default=None, help="override output_dir from the config")
    p.add_argument("--run-id", default=None, help="explicit run directory name")
    p.add_argument("--seed", type=int, default=None, help="override the test's own seed from the config")
    p = sub.add_parser("mixing", help="the geometry test (D40): a control mixed between two labels of a family, read as the "
                                      "masses of the family's candidate continuations against the dilution null")
    p.add_argument("--config", required=True, help="YAML config of the test (configs/mixing.yaml)")
    p.add_argument("--runs", nargs="+", required=True,
                   help="family=<learned run>[:<head-mean run>] for every family to run (others are skipped)")
    p.add_argument("--output-dir", default=None, help="override output_dir from the config")
    p.add_argument("--run-id", default=None, help="explicit run directory name")
    p.add_argument("--seed", type=int, default=None, help="override the test's own seed from the config")
    args = parser.parse_args(argv)
    if args.command in ("validate", "pilot"):
        return _cmd_run(args, args.command)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "aggregate":
        return _cmd_aggregate(args)
    if args.command == "trajectories":
        return _cmd_trajectories(args)
    if args.command == "source":
        return _cmd_source(args)
    if args.command == "mixing":
        return _cmd_mixing(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
