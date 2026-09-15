"""Stage the locked wheels on a RunPod network volume so workers install without PyPI (docs/INFRA.md).

    uv run --with boto3 python scripts/runpod_wheelhouse.py build  --dest /tmp/wheels        # download the wheels here
    uv run --with boto3 python scripts/runpod_wheelhouse.py upload --src /tmp/wheels --datacenter EUR-NO-1
    uv run --with boto3 python scripts/runpod_wheelhouse.py list   --datacenter EUR-NO-1     # what the volume holds

``build`` exports the lock (``uv export --frozen --no-dev``) and downloads every wheel with pip for this
machine's platform, which must match the worker's (linux x86_64, CPython 3.11; the worker's glibc is 2.35, so
run this on a host whose glibc is not newer than that or check the tags with ``list``). The project's build
backend (``uv_build``) is included so the worker can build the project itself. ``upload`` puts the files under
``wheels/`` on the volume (the handler installs from there when the directory exists) and skips files already
present with the same size. Needs ``RUNPOD_API_KEY`` and the S3 keys, as ``scripts/runpod_log.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("runpod_job", Path(__file__).with_name("runpod_job.py"))
runpod_job = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(runpod_job)  # type: ignore[union-attr]

PREFIX = "wheels/"
BUILD_REQUIREMENT = "uv_build>=0.12.10,<0.13.0"  # pyproject's build-system requires


def build(dest: Path, root: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    req = dest / "requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-dev", "-q", "-o", str(req)],
                   cwd=root, check=True)
    (dest / "requirements-build.txt").write_text(BUILD_REQUIREMENT + "\n")
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--retries", "10", "--timeout", "120",
                    "-r", str(req), "-r", str(dest / "requirements-build.txt"), "-d", str(dest), "--only-binary=:all:"],
                   check=True)
    return wheel_files(dest)


def wheel_files(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).iterdir() if p.suffix == ".whl")


def platform_tags(files: list[Path]) -> dict[str, int]:
    """How many wheels carry each platform tag (``manylinux_2_28_x86_64``, ``any`` ...)."""
    tags: dict[str, int] = {}
    for p in files:
        tag = p.stem.rsplit("-", 1)[-1]
        tags[tag] = tags.get(tag, 0) + 1
    return tags


def _client(datacenter: str, volume_name: str):
    for k in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY"):
        if not os.environ.get(k):
            raise SystemExit(f"missing environment variable {k}")
    volume_id = runpod_job.resolve_volume_id(runpod_job.list_volumes(os.environ["RUNPOD_API_KEY"]), volume_name, datacenter)
    return runpod_job.s3_client(datacenter, os.environ["RUNPOD_S3_ACCESS_KEY"], os.environ["RUNPOD_S3_SECRET_KEY"]), volume_id


def on_volume(client, volume_id: str) -> dict[str, int]:
    present: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=volume_id, Prefix=PREFIX):
        for obj in page.get("Contents") or []:
            present[obj["Key"][len(PREFIX):]] = obj["Size"]
    return present


def upload(src: Path, datacenter: str, volume_name: str) -> int:
    client, volume_id = _client(datacenter, volume_name)
    present = on_volume(client, volume_id)
    files = wheel_files(src)
    if not files:
        raise SystemExit(f"no wheels in {src}")
    n = 0
    for i, p in enumerate(files, 1):
        size = p.stat().st_size
        if present.get(p.name) == size:
            continue
        client.upload_file(str(p), volume_id, PREFIX + p.name)
        n += 1
        print(f"[{i}/{len(files)}] {p.name} ({size / 1e6:.1f} MB)", flush=True)
    print(f"{n} uploaded, {len(files) - n} already present, {len(files)} wheels under {PREFIX} on {volume_id} ({datacenter})")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "upload", "list"])
    parser.add_argument("--dest", type=Path, default=Path("/tmp/wheels"), help="build: where to download")
    parser.add_argument("--src", type=Path, default=Path("/tmp/wheels"), help="upload: the wheelhouse to upload")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="the repository (its uv.lock)")
    parser.add_argument("--datacenter", default="EUR-NO-1")
    parser.add_argument("--volume-name", default="directions")
    args = parser.parse_args(argv)
    if args.action == "build":
        files = build(args.dest, args.root)
        print(f"{len(files)} wheels, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB in {args.dest}; tags {platform_tags(files)}")
        return 0
    if args.action == "upload":
        upload(args.src, args.datacenter, args.volume_name)
        return 0
    client, volume_id = _client(args.datacenter, args.volume_name)
    present = on_volume(client, volume_id)
    for name, size in sorted(present.items()):
        print(f"{size:>12}  {name}")
    print(f"{len(present)} wheels, {sum(present.values()) / 1e9:.2f} GB under {PREFIX} on {volume_id} ({args.datacenter})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
