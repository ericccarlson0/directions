"""Read a run's live log or results from the RunPod network volume, outside the workflow.

    uv run --with boto3 python scripts/runpod_log.py 34794257716              # runner.log so far
    uv run --with boto3 python scripts/runpod_log.py 34794257716 --follow     # keep tailing (Ctrl-C to stop)
    uv run --with boto3 python scripts/runpod_log.py 34794257716 --list       # objects under results/run-<id>/
    uv run --with boto3 python scripts/runpod_log.py 34794257716 --download results/remote/<name>

The argument is the GitHub workflow run id (the worker writes under ``results/run-<id>/``) or that
directory name itself. Needs ``RUNPOD_API_KEY`` (to resolve the volume id) and
``RUNPOD_S3_ACCESS_KEY`` / ``RUNPOD_S3_SECRET_KEY`` (docs/INFRA.md); reuses ``scripts/runpod_job.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("runpod_job", Path(__file__).with_name("runpod_job.py"))
runpod_job = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(runpod_job)  # type: ignore[union-attr]


def results_name(run: str) -> str:
    """``results/<name>`` on the volume for a workflow run id or an explicit directory name."""
    run = run.strip().rstrip("/")
    if run.startswith("results/"):
        run = run[len("results/") :]
    return run if run.startswith("run-") else f"run-{run}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="GitHub workflow run id, or the results directory name (run-<id>)")
    parser.add_argument("--follow", action="store_true", help="keep tailing runner.log until interrupted")
    parser.add_argument("--list", action="store_true", help="list the objects under the run's results directory")
    parser.add_argument("--download", type=Path, default=None, help="download the run's results directory here")
    parser.add_argument("--volume-name", default="directions")
    parser.add_argument("--datacenter", default="EUR-NO-1")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)

    missing = [k for k in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        print(f"missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    name = results_name(args.run)
    volume_id = runpod_job.resolve_volume_id(runpod_job.list_volumes(os.environ["RUNPOD_API_KEY"]), args.volume_name, args.datacenter)
    client = runpod_job.s3_client(args.datacenter, os.environ["RUNPOD_S3_ACCESS_KEY"], os.environ["RUNPOD_S3_SECRET_KEY"])
    prefix = f"results/{name}/"

    if args.list:
        n = 0
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=volume_id, Prefix=prefix):
            for obj in page.get("Contents") or []:
                print(f"{obj['Size']:>12}  {obj['LastModified']:%Y-%m-%dT%H:%M:%SZ}  {obj['Key'][len(prefix):]}")
                n += 1
        print(f"{n} object(s) under {prefix} on volume {volume_id}")
        return 0
    if args.download is not None:
        n = runpod_job.download_results(client, volume_id, prefix, args.download)
        print(f"downloaded {n} file(s) from {prefix} into {args.download}")
        return 0

    tail = runpod_job.LogTail(client, volume_id, f"{prefix}runner.log")
    tail()
    if tail.offset == 0:
        print(f"(no runner.log under {prefix} yet: the job has not started on a worker)", file=sys.stderr)
    if not args.follow:
        return 0
    try:
        while True:
            time.sleep(args.poll_seconds)
            tail()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
