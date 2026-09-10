#!/usr/bin/env python
"""Stage the handler-only Flash app and print its digest.

The deployed Flash artifact contains nothing but the job handler (``experiments/remote.py``) and the helpers it
imports (``src/directions/remote.py`` and the package's ``__init__``/``_version``); the experiment code travels
inside each job (``directions.remote.pack_source``). ``flash deploy`` builds whatever directory it runs in, so the
workflow copies these files into a staging directory and deploys from there.

The digest covers the staged files and the endpoint's configuration (GPU tier/type, data center, Flash version,
...): when it equals the digest recorded for the live endpoint, the workflow reuses the endpoint instead of
redeploying (docs/INFRA.md). Standard library only.

Usage::

    python scripts/runpod_stage.py --dest .flash-app --param gpu_tier=ADA_24 --param flash_version=1.19.0
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REQUIRED = ("experiments/remote.py", "src/directions/remote.py")
OPTIONAL = ("src/directions/__init__.py", "src/directions/_version.py")


def staged_files(root: Path) -> list[Path]:
    """Relative paths to stage; missing required files are an error."""
    files = []
    for rel in REQUIRED:
        if not (root / rel).is_file():
            raise FileNotFoundError(f"handler file {rel} not found under {root}")
        files.append(Path(rel))
    files.extend(Path(rel) for rel in OPTIONAL if (root / rel).is_file())
    return files


def handler_digest(root: Path, params: dict[str, str]) -> str:
    """SHA-256 over the staged files (path + content) and the sorted ``params``."""
    h = hashlib.sha256()
    for rel in sorted(staged_files(root)):
        h.update(str(rel).encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    for key in sorted(params):
        h.update(f"{key}={params[key]}\n".encode())
    return h.hexdigest()


def stage(root: Path, dest: Path) -> list[Path]:
    """Copy the handler files into ``dest`` (recreated), keeping their relative layout."""
    if dest.exists():
        shutil.rmtree(dest)
    files = staged_files(root)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--dest", type=Path, required=True, help="staging directory (recreated)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="endpoint configuration that enters the digest (repeatable)")
    args = parser.parse_args(argv)
    params = {}
    for item in args.param:
        if "=" not in item:
            print(f"--param expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        params[key] = value
    try:
        files = stage(args.root, args.dest)
        digest = handler_digest(args.root, params)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"staged {', '.join(map(str, files))} into {args.dest}; params {params}", file=sys.stderr)
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
