"""Resolve the parameters of a `run-gpu` job (``.github/workflows/run-gpu.yml``).

A run can be requested in two ways (README, "Run on GPUs"):

* **push**: commit a change to the request file ``.github/gpu-run.yaml`` and push it.
  The workflow triggers on pushes that touch that file and reads the parameters from
  it, so an agent or a script with only git push access can start a run, and the
  request is version-controlled next to the code it ran.
* **workflow_dispatch**: the form's inputs are used and the request file is ignored.

The request file is a flat mapping of scalars, one ``key: value`` per line with ``#``
comments and optional quotes; that subset is parsed here without PyYAML so the script
runs with the system interpreter on the CI runner. Prints ``key=value`` lines for
``$GITHUB_OUTPUT`` and a summary to stderr. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUEST_FILE = Path(".github/gpu-run.yaml")
GPU_TIERS = ("AMPERE_16", "AMPERE_24", "ADA_24", "ADA_32_PRO", "AMPERE_48", "ADA_48_PRO", "AMPERE_80", "ADA_80_PRO",
             "BLACKWELL_96", "HOPPER_141")
# Data centers a run can use: those Flash 1.19 accepts for a network volume (runpod_flash DataCenter enum) that
# RunPod also lets a network volume be created in (the REST error of run 34858770757 lists them). docs/INFRA.md.
DATACENTERS = ("US-CA-2", "US-IL-1", "US-MO-2", "US-NC-2", "EU-RO-1", "EUR-NO-1")
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
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RequestError(ValueError):
    pass


def _strip_comment(value: str) -> str:
    """Drop a trailing ``# comment`` that is outside quotes."""
    quote = None
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].strip()
    return value.strip()


def _unquote(value: str, where: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    if value and value[0] in "'\"":
        raise RequestError(f"{where}: unbalanced quote in {value!r}")
    return value


def parse_flat_mapping(text: str, where: str = "request file") -> dict[str, str]:
    """Parse ``key: value`` lines (flat YAML subset: scalars only, ``#`` comments, optional quotes)."""
    result: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise RequestError(f"{where}:{lineno}: must be a mapping of `key: value` lines, not a list")
        if ":" not in line:
            raise RequestError(f"{where}:{lineno}: expected `key: value`, got {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not _KEY_RE.match(key):
            raise RequestError(f"{where}:{lineno}: invalid key {key!r}")
        if key in result:
            raise RequestError(f"{where}:{lineno}: duplicate key {key!r}")
        value = _unquote(_strip_comment(value), f"{where}:{lineno}")
        if value[:1] in ("[", "{", "|", ">", "&", "*"):
            raise RequestError(f"{where}:{lineno}: {key} must be a scalar on one line, got {value!r}")
        result[key] = value.strip()
    return result


def load_request(path: Path) -> dict[str, str]:
    """Read and validate the request file; every value is returned as a string."""
    try:
        raw = parse_flat_mapping(path.read_text(), str(path))
    except FileNotFoundError:
        raise RequestError(f"request file {path} not found") from None
    unknown = sorted(set(raw) - set(FIELDS))
    if unknown:
        raise RequestError(f"request file {path}: unknown keys {unknown}; allowed: {list(FIELDS)}")
    params = {**DEFAULTS, **raw}
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
    if params["datacenter"] not in DATACENTERS:
        raise RequestError(f"datacenter {params['datacenter']!r} is not one of {DATACENTERS} (the data centers both Flash "
                           "and RunPod network volumes accept; docs/INFRA.md)")
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
