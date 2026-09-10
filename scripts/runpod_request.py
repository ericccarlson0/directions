"""Resolve the parameters of a `run-gpu` job (``.github/workflows/run-gpu.yml``).

A run can be requested in two ways (README, "Run on GPUs"):

* **push**: commit a change to the request file ``.github/gpu-run.yaml`` and push it.
  The workflow triggers on pushes that touch that file and reads the parameters from
  it, so an agent or a script with only git push access can start a run, and the
  request is version-controlled next to the code it ran.
* **workflow_dispatch**: the form's inputs are used and the request file is ignored.

Prints ``key=value`` lines for ``$GITHUB_OUTPUT`` and a summary to stderr. Standard
library plus PyYAML (a pipeline dependency; the workflow runs this from its venv).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REQUEST_FILE = Path(".github/gpu-run.yaml")
GPU_TIERS = ("AMPERE_16", "AMPERE_24", "ADA_24", "AMPERE_48", "ADA_48_PRO", "AMPERE_80", "ADA_80_PRO")
DEFAULTS = {
    "gpu_tier": "ADA_24",
    "gpu_type": "",
    "datacenter": "EUR-NO-1",
    "flash_env": "ci",
    "timeout_minutes": "300",
    "max_output_mb": "8",
}
FIELDS = ("request", "command", *DEFAULTS)
_FLASH_ENV_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RequestError(ValueError):
    pass


def load_request(path: Path) -> dict[str, str]:
    """Read and validate the request file; every value is returned as a string."""
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise RequestError(f"request file {path} not found") from None
    except yaml.YAMLError as e:
        raise RequestError(f"request file {path} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise RequestError(f"request file {path} must be a mapping")
    unknown = sorted(set(raw) - set(FIELDS))
    if unknown:
        raise RequestError(f"request file {path}: unknown keys {unknown}; allowed: {list(FIELDS)}")
    params = dict(DEFAULTS)
    for key, value in raw.items():
        if value is None:
            value = ""
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise RequestError(f"request file {path}: {key} must be a scalar, got {type(value).__name__}")
        params[key] = str(value).strip()
    for key in ("request", "command"):
        if not params.get(key):
            raise RequestError(f"request file {path}: `{key}` is required and must be non-empty")
    return validate(params)


def resolve(event: str, inputs: dict[str, object] | None, request_file: Path) -> dict[str, str]:
    """Parameters for this workflow run: the dispatch form on workflow_dispatch, else the file."""
    if event == "workflow_dispatch":
        inputs = inputs or {}
        params = {"request": "manual"}
        for key in ("command", *DEFAULTS):
            value = inputs.get(key)
            params[key] = DEFAULTS.get(key, "") if value is None else str(value).strip()
        if not params["command"]:
            raise RequestError("workflow_dispatch: `command` input is required")
        return validate(params)
    return load_request(request_file)


def validate(params: dict[str, str]) -> dict[str, str]:
    for key, value in params.items():
        if "\n" in value or "\r" in value:
            raise RequestError(f"`{key}` must be a single line")
    if params["gpu_tier"] not in GPU_TIERS:
        raise RequestError(f"gpu_tier {params['gpu_tier']!r} is not one of {GPU_TIERS}")
    if not params["datacenter"]:
        raise RequestError("`datacenter` must be non-empty")
    if not _FLASH_ENV_RE.match(params["flash_env"]):
        raise RequestError(f"flash_env {params['flash_env']!r} must match {_FLASH_ENV_RE.pattern}")
    for key in ("timeout_minutes", "max_output_mb"):
        try:
            number = float(params[key])
        except ValueError:
            raise RequestError(f"`{key}` must be a number, got {params[key]!r}") from None
        if not number > 0:
            raise RequestError(f"`{key}` must be positive, got {params[key]!r}")
    return params


def format_outputs(params: dict[str, str]) -> str:
    return "".join(f"{key}={params[key]}\n" for key in FIELDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=REQUEST_FILE, help="request file (push trigger)")
    parser.add_argument("--event", required=True, help="$GITHUB_EVENT_NAME")
    parser.add_argument("--inputs-json", default="{}", help="toJSON(inputs) of a workflow_dispatch event")
    args = parser.parse_args(argv)
    try:
        inputs = json.loads(args.inputs_json) if args.inputs_json else {}
        if inputs is not None and not isinstance(inputs, dict):
            raise RequestError("--inputs-json must be a JSON object")
        params = resolve(args.event, inputs, args.file)
    except (RequestError, json.JSONDecodeError) as e:
        print(f"run-gpu request error: {e}", file=sys.stderr)
        return 2
    source = "workflow_dispatch inputs" if args.event == "workflow_dispatch" else f"{args.file} ({args.event})"
    print(f"run-gpu parameters from {source}:", file=sys.stderr)
    for key in FIELDS:
        print(f"  {key}: {params[key]}", file=sys.stderr)
    sys.stdout.write(format_outputs(params))
    return 0


if __name__ == "__main__":
    sys.exit(main())
