"""Command-line interface.

All scientifically meaningful parameters come from a version-controlled YAML
config; ``--set`` exists only for sweeps and debugging, and every override is
recorded in the run directory's resolved config.

    uv run directions validate --config configs/qwen3_0.6b.yaml
    uv run directions pilot    --config configs/qwen3_0.6b.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .config import load_config
from .pipeline import run_pilot, run_validation
from .tasks import available_tasks, load_task

app = typer.Typer(add_completion=False, help="Directions: control-propagation measurement pilot.")

ConfigOpt = typer.Option(..., "--config", "-c", help="Path to a YAML config file.")
SetOpt = typer.Option(None, "--set", "-s", help="Override, e.g. -s run.seed=1 -s data.n_eval=8.")
TasksOpt = typer.Option(None, "--tasks", help="Comma-separated task list overriding data.tasks.")
OutOpt = typer.Option(None, "--output-root", help="Override run.output_root.")


def _parse_overrides(pairs: list[str] | None, tasks: str | None, out: str | None) -> dict:
    overrides: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise typer.BadParameter(f"--set expects key.path=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    if tasks:
        overrides.setdefault("data", {})["tasks"] = [t.strip() for t in tasks.split(",") if t.strip()]
    if out:
        overrides.setdefault("run", {})["output_root"] = out
    return overrides


def _run(fn, config: str, set_: list[str] | None, tasks: str | None, output_root: str | None):
    cfg = load_config(config, _parse_overrides(set_, tasks, output_root))
    summary = fn(cfg)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "figures"}, indent=2))
    return summary


@app.command()
def validate(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    tasks: str = TasksOpt,
    output_root: str = OutOpt,
):
    """Task qualification only: ICL, extraction stability, calibration, steering, controls."""
    _run(run_validation, config, set_, tasks, output_root)


@app.command()
def pilot(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    tasks: str = TasksOpt,
    output_root: str = OutOpt,
):
    """Full pilot: validation plus the downstream layerwise measurement and figures."""
    _run(run_pilot, config, set_, tasks, output_root)


@app.command("tasks")
def list_tasks():
    """List the registered task datasets and their sizes."""
    for name in available_tasks():
        task = load_task(name)
        typer.echo(f"{name:20s} n={len(task):4d}  {task.description}")


@app.command("show-config")
def show_config(config: str = ConfigOpt, set_: list[str] = SetOpt):
    """Print the fully resolved configuration and its fingerprint."""
    cfg = load_config(config, _parse_overrides(set_, None, None))
    typer.echo(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    typer.echo(f"# fingerprint: {cfg.fingerprint()}")


@app.command("inspect")
def inspect_run(path: str = typer.Argument(..., help="Path to a results/ run directory.")):
    """Print the summary of a completed run."""
    summary = Path(path) / "summary.json"
    if not summary.exists():
        raise typer.BadParameter(f"no summary.json under {path}")
    typer.echo(summary.read_text())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
