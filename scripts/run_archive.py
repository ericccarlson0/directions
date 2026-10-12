"""The run archive: every GitHub Actions results artifact of the run-gpu workflow, copied to a RunPod network volume
before GitHub's 90-day retention deleted it (docs/INFRA.md, "The run archive"; docs/archive_manifest.jsonl).

    uv run --with boto3 python scripts/run_archive.py list                       # archived runs (from the manifest)
    uv run --with boto3 python scripts/run_archive.py fetch 35682027963 --out results/archive
    uv run --with boto3 python scripts/run_archive.py verify                     # every manifest entry is on the volume

``fetch`` downloads the run's artifact zip(s), checks the SHA-256 recorded in the manifest (GitHub's own digest of
the artifact), and unpacks them into ``<out>/run-<id>/``, the layout ``scripts/runpod_log.py --download`` produces.
Needs ``RUNPOD_API_KEY`` and ``RUNPOD_S3_ACCESS_KEY`` / ``RUNPOD_S3_SECRET_KEY`` (docs/INFRA.md); standard library plus
boto3, through ``scripts/runpod_job.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "archive_manifest.jsonl"


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def archived(entries: list[dict], run_id: str | None = None) -> list[dict]:
    """The entries whose artifact is on the volume, optionally for one workflow run id."""
    out = [e for e in entries if e.get("status") in ("archived", "already_archived")]
    if run_id is not None:
        out = [e for e in out if str(e["run_id"]) == str(run_id).removeprefix("run-")]
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runpod_job():
    spec = importlib.util.spec_from_file_location("runpod_job", Path(__file__).with_name("runpod_job.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _client(entry: dict):
    rj = _runpod_job()
    missing = [k for k in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)}")
    return rj.s3_client(entry["datacenter"], os.environ["RUNPOD_S3_ACCESS_KEY"], os.environ["RUNPOD_S3_SECRET_KEY"])


def fetch(entry: dict, out: Path, client=None) -> Path:
    """Download one archived artifact, verify its digest and unpack it under ``out/run-<id>/``."""
    client = client or _client(entry)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{entry['name']}.zip"
    client.download_file(entry["volume"], entry["key"], str(zpath))
    digest = sha256_file(zpath)
    if digest != entry["sha256"]:
        zpath.unlink()
        raise ValueError(f"{entry['key']}: sha256 {digest} does not match the manifest's {entry['sha256']}")
    dest = out / f"run-{entry['run_id']}"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink()
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("run", help="workflow run id (or run-<id>)")
    f.add_argument("--out", type=Path, default=Path("results/archive"))
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    entries = load_manifest()

    if args.cmd == "list":
        for e in archived(entries):
            print(f"{e['run_id']}  {e['size'] / 1e6:8.1f} MB  {e['name']}")
        n = archived(entries)
        print(f"{len(n)} artifact(s), {sum(e['size'] for e in n) / 1e9:.2f} GB, on {sorted({e['volume'] for e in n})}")
        return 0
    if args.cmd == "fetch":
        found = archived(entries, args.run)
        if not found:
            print(f"no archived artifact for run {args.run}", file=sys.stderr)
            return 1
        for e in found:
            print("unpacked into", fetch(e, args.out))
        return 0
    # verify: every archived entry exists on the volume with the recorded size
    bad = 0
    clients: dict[str, object] = {}
    for e in archived(entries):
        c = clients.setdefault(e["datacenter"], _client(e))
        size = c.head_object(Bucket=e["volume"], Key=e["key"])["ContentLength"]
        if size != e["size"]:
            print("SIZE MISMATCH", e["key"], size, e["size"])
            bad += 1
    print(f"{len(archived(entries)) - bad} of {len(archived(entries))} archived artifacts present with the recorded size")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
