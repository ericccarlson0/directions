#!/usr/bin/env python
"""Submit a command to the deployed ``directions-runner`` endpoint and collect results.

Usage (after ``flash deploy``)::

    uv run scripts/runpod_job.py --endpoint-url https://api.runpod.io/v2/<id> \
        --argv-json '["python", "-c", "print(1)"]' --out results/remote

Reads ``RUNPOD_API_KEY`` from the env.
Exit code mirrors the remote command's exit code (or 1 for transport/timeout failures).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from directions.remote import unpack_results  # noqa: E402

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def _request(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def submit(endpoint_url: str, api_key: str, job_input: dict) -> str:
    body = _request(f"{endpoint_url}/run", api_key, {"input": job_input})
    job_id = body.get("id")
    if not job_id:
        raise RuntimeError(f"no job id in /run response: {body}")
    return job_id


def cancel(endpoint_url: str, api_key: str, job_id: str) -> None:
    try:
        _request(f"{endpoint_url}/cancel/{job_id}", api_key, {})
    except (urllib.error.URLError, RuntimeError) as exc:  # best effort
        print(f"::warning::cancel {job_id} failed: {exc}", file=sys.stderr)


def wait(endpoint_url: str, api_key: str, job_id: str, deadline_s: float, poll_s: float) -> dict:
    """Poll ``/status``; cancel on deadline."""
    last = None
    start = time.time()
    while True:
        try:
            body = _request(f"{endpoint_url}/status/{job_id}", api_key)
        except urllib.error.URLError as exc:
            print(f"status poll failed ({exc}); retrying", file=sys.stderr)
            body = {"status": last or "UNKNOWN"}
        status = body.get("status")
        if status != last:
            print(f"[{time.time() - start:7.0f}s] {status}", flush=True)
            last = status
        if status in TERMINAL:
            return body
        if time.time() - start > deadline_s:
            print(f"deadline of {deadline_s:.0f}s exceeded; cancelling job {job_id}", file=sys.stderr)
            cancel(endpoint_url, api_key, job_id)
            return {"status": "TIMED_OUT", "error": "local deadline exceeded"}
        time.sleep(poll_s)


def collect(final: dict, out: Path) -> int:
    """Write logs/results; return the process exit code."""
    status = final.get("status")
    if status != "COMPLETED":
        print(f"job ended with status {status}: {final.get('error')}", file=sys.stderr)
        return 1
    output = final.get("output") or {}
    if output.get("success") is False:  # Flash handler wrapper caught an exception
        print(f"handler error: {output.get('error')}\n{output.get('traceback', '')}", file=sys.stderr)
        return 1

    print("----- remote log tail -----")
    print(output.get("log_tail", ""))
    print("---------------------------")
    if output.get("results_error"):
        print(f"::error::{output['results_error']}", file=sys.stderr)
    if output.get("results_tar_b64"):
        unpack_results(output["results_tar_b64"], out)
        print(f"unpacked {output.get('results_file_count')} files into {out}")
    returncode = int(output.get("returncode", 1))
    if output.get("results_error") and returncode == 0:
        returncode = 1
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint-url", required=True, help="e.g. https://api.runpod.io/v2/<endpoint-id>")
    parser.add_argument("--argv-json", required=True, help="JSON list: the command to run at the artifact root")
    parser.add_argument("--out", type=Path, default=Path("results/remote"), help="where to unpack results")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-output-mb", type=float, default=8.0)
    parser.add_argument("--timeout-minutes", type=float, default=300)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--job-id-file", type=Path, default=None, help="write the job id here right after submit")
    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 1
    command = json.loads(args.argv_json)
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command) or not command:
        print("--argv-json must be a non-empty JSON list of strings", file=sys.stderr)
        return 1

    job_input = {
        "argv": command,
        "results_dir": args.results_dir,
        "max_output_mb": args.max_output_mb,
        "git_commit": args.git_commit,
    }
    job_id = submit(args.endpoint_url.rstrip("/"), api_key, job_input)
    print(f"submitted job {job_id}")
    if args.job_id_file:
        args.job_id_file.write_text(job_id)

    final = wait(args.endpoint_url.rstrip("/"), api_key, job_id, args.timeout_minutes * 60, args.poll_seconds)
    return collect(final, args.out)


if __name__ == "__main__":
    sys.exit(main())
