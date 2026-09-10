#!/usr/bin/env python
"""Submit a command to the deployed ``directions-runner`` endpoint and collect results.

Usage (after ``flash deploy``)::

    uv run scripts/runpod_job.py --endpoint-url https://api.runpod.io/v2/<id> \
        --argv-json '["python", "-c", "print(1)"]' --out results/remote \
        --results-name run-123 --volume-name directions --datacenter EUR-NO-1

With ``--results-name`` the worker keeps the results on its network volume and this script downloads them
through RunPod's S3-compatible API. Without it the results come back inside the job output as a base64 tarball.

Reads ``RUNPOD_API_KEY`` from the env.
Exit code mirrors the remote command's exit code (or 1 for transport/timeout failures).
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
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from directions.remote import pack_source, unpack_results  # noqa: E402

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
USER_AGENT = "directions-runpod-job/1.0"  # RunPod's edge rejects the default urllib agent with HTTP 403
REST_URL = "https://api.runpod.io/v2"

Downloader = Callable[[str, Path], int]


def _request(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
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


def wait(
    endpoint_url: str,
    api_key: str,
    job_id: str,
    deadline_s: float,
    poll_s: float,
    progress: Callable[[], None] | None = None,
) -> dict:
    """Poll ``/status``; call ``progress`` after each poll; cancel on deadline."""
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
        if progress is not None:
            progress()
        if status in TERMINAL:
            return body
        if time.time() - start > deadline_s:
            print(f"deadline of {deadline_s:.0f}s exceeded; cancelling job {job_id}", file=sys.stderr)
            cancel(endpoint_url, api_key, job_id)
            return {"status": "TIMED_OUT", "error": "local deadline exceeded"}
        time.sleep(poll_s)


def is_stale(final: dict, expected_source: str | None) -> bool:
    """The job was answered by a worker that did not run the shipped source ``expected_source``.

    Either the worker completed without reporting that digest, or its handler predates shipped source and
    rejected the inputs (the platform reports that as FAILED). ``expected_source`` None disables the check
    (tests, ad-hoc use); the CLI always ships source. The artifact's own fingerprint is only reported.
    """
    if not expected_source:
        return False
    if final.get("status") == "FAILED":
        return "unexpected keyword argument 'source_" in str(final.get("error", ""))
    if final.get("status") != "COMPLETED":
        return False
    return (final.get("output") or {}).get("source_sha256") != expected_source


GRAPHQL_URL = "https://api.runpod.io/graphql"
TERMINATE_MUTATION = "mutation podTerminate($input: PodTerminateInput!) { podTerminate(input: $input) }"


def terminate_worker(api_key: str, worker_id: str) -> bool:
    """Best-effort: terminate a serverless worker (a pod) so the next job gets a fresh container.

    Stale containers were seen answering jobs for 25+ min despite the 30 s idle timeout (docs/INFRA.md); RunPod
    reuses the stopped container, /app included. Returns True when the API accepted the request.
    """
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": TERMINATE_MUTATION, "variables": {"input": {"podId": worker_id}}}).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print(f"::warning::terminating worker {worker_id} failed: {exc}", file=sys.stderr)
        return False
    if body.get("errors"):
        print(f"::warning::terminating worker {worker_id} rejected: {body['errors']}", file=sys.stderr)
        return False
    print(f"terminated stale worker {worker_id}", flush=True)
    return True


def run(
    endpoint_url: str,
    api_key: str,
    job_input: dict,
    deadline_s: float,
    poll_s: float,
    progress: Callable[[], None] | None = None,
    job_id_file: Path | None = None,
    stale_retries: int = 3,
    stale_wait_s: float = 90,
) -> dict:
    """Submit and wait; when a stale worker answers, let it scale down and resubmit (up to ``stale_retries``).

    A worker that unpacked the previous build refuses the job at once (``experiments/remote.py``), so a retry
    costs ``stale_wait_s`` (longer than the endpoint's idle timeout, after which the stale worker is gone) plus
    a cold start. Observed when a run starts seconds after the previous one ended (docs/INFRA.md).
    """
    start = time.time()
    final: dict = {}
    expected_source = job_input.get("source_sha256")
    for attempt in range(stale_retries + 1):
        job_id = submit(endpoint_url, api_key, job_input)
        print(f"submitted job {job_id}" + (f" (resubmission {attempt} of {stale_retries})" if attempt else ""), flush=True)
        if job_id_file:
            job_id_file.write_text(job_id)
        final = wait(endpoint_url, api_key, job_id, deadline_s - (time.time() - start), poll_s, progress)
        if not is_stale(final, expected_source) or attempt == stale_retries:
            return final
        output = final.get("output") or {}
        print(
            f"::warning::stale worker answered job {job_id} (status {final.get('status')}, code "
            f"{str(output.get('code_fingerprint'))[:12]}, deploy {str(output.get('expected_fingerprint'))[:12]}, shipped "
            f"source {str(output.get('source_sha256'))[:12]}, error {str(final.get('error', ''))[:80]!r}); "
            f"terminating it if known, waiting {stale_wait_s:.0f}s, then resubmitting",
            file=sys.stderr,
            flush=True,
        )
        if final.get("workerId"):
            terminate_worker(api_key, str(final["workerId"]))
        time.sleep(stale_wait_s)
    return final


def resolve_volume_id(volumes: list[dict], name: str, datacenter: str) -> str:
    """Id of the network volume called ``name`` in ``datacenter``."""
    for volume in volumes:
        location = volume.get("dataCenter") or volume.get("dataCenterId")  # REST v2 uses the former
        if volume.get("name") == name and location == datacenter:
            return volume["id"]
    raise RuntimeError(f"no network volume named {name!r} in {datacenter}")


def list_volumes(api_key: str) -> list[dict]:
    body = _request(f"{REST_URL}/network-volumes", api_key)
    return body if isinstance(body, list) else body.get("items") or body.get("networkVolumes") or []


def s3_client(datacenter: str, access_key: str, secret_key: str):
    import boto3  # installed in the deploy environment only

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=datacenter,
        endpoint_url=f"https://s3api-{datacenter.lower()}.runpod.io/",
    )


def download_results(client, bucket: str, prefix: str, out: Path) -> int:
    """Download every object under ``prefix/`` into ``out``, keeping relative paths; return the file count."""
    prefix = prefix.rstrip("/") + "/"
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            dest = out / key[len(prefix) :]
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    return count


class LogTail:
    def __init__(self, client, bucket: str, key: str) -> None:
        self.client, self.bucket, self.key = client, bucket, key
        self.offset = 0

    def __call__(self) -> None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.offset}-")["Body"].read()
        except Exception as exc:  # not written yet, or nothing new (416): both are normal mid-run
            name = type(exc).__name__
            if "NoSuchKey" in name or "InvalidRange" in name or "416" in str(exc) or "NoSuchKey" in str(exc):
                return
            print(f"::warning::log tail failed: {exc}", file=sys.stderr)
            return
        if body:
            sys.stdout.write(body.decode("utf-8", errors="replace"))
            sys.stdout.flush()
            self.offset += len(body)


def collect(final: dict, out: Path, download: Downloader | None = None, expected_source: str | None = None) -> int:
    """Write logs/results; return the process exit code. ``expected_source`` is the shipped tree's digest."""
    status = final.get("status")
    if status != "COMPLETED":
        hint = " (a worker holding a handler from before shipped source; see docs/INFRA.md)" if is_stale(final, expected_source) else ""
        print(f"job ended with status {status}: {final.get('error')}{hint}", file=sys.stderr)
        return 1
    output = final.get("output") or {}
    if output.get("success") is False:  # Flash handler wrapper caught an exception
        print(f"handler error: {output.get('error')}\n{output.get('traceback', '')}", file=sys.stderr)
        return 1

    print("----- remote log tail -----")
    print(output.get("log_tail", ""))
    print("---------------------------")
    code, expected = output.get("code_fingerprint"), output.get("expected_fingerprint")
    if is_stale(final, expected_source):
        print(
            f"::error::the worker did not run the shipped source {str(expected_source)[:12]} (it reported "
            f"{str(output.get('source_sha256'))[:12]}; its handler predates shipped source or is stale: code "
            f"{str(code)[:12]}, deploy {str(expected)[:12]})",
            file=sys.stderr,
        )
        return 1
    if expected_source:
        print(f"worker ran the shipped source {expected_source[:12]} in {output.get('source_dir')}")
    if output.get("handler_stale"):
        print(f"::warning::the worker's handler is from a previous build (code {str(code)[:12]}, deploy {str(expected)[:12]}); "
              "harmless with shipped source", file=sys.stderr)
    returncode = int(output.get("returncode", 1))
    if output.get("results_error"):
        print(f"::error::{output['results_error']}", file=sys.stderr)
        returncode = returncode or 1
    if output.get("results_path"):
        if download is None:
            print(f"::error::results are on the volume at {output['results_path']} but no S3 download is configured")
            return 1
        n = download(output["results_path"], out)
        print(f"downloaded {n} files from {output['results_path']} into {out}")
    elif output.get("results_tar_b64"):
        unpack_results(output["results_tar_b64"], out)
        print(f"unpacked {output.get('results_file_count')} files into {out}")
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint-url", required=True, help="e.g. https://api.runpod.io/v2/<endpoint-id>")
    parser.add_argument("--argv-json", required=True, help="JSON list: the command to run at the artifact root")
    parser.add_argument("--out", type=Path, default=Path("results/remote"), help="where to put the results")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--results-name", default=None, help="results/<name> on the worker's network volume")
    parser.add_argument("--volume-name", default="directions")
    parser.add_argument("--datacenter", default="EUR-NO-1")
    parser.add_argument("--max-output-mb", type=float, default=8.0)
    parser.add_argument("--timeout-minutes", type=float, default=300)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--job-id-file", type=Path, default=None, help="write the job id here right after submit")
    parser.add_argument("--stale-retries", type=int, default=3, help="resubmissions after a stale worker answers")
    parser.add_argument("--stale-wait-seconds", type=float, default=90, help="pause before each resubmission")
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="repository whose --git-commit (default HEAD) is shipped inside the job as `git archive`; "
                             "the worker runs that tree (the deployed artifact only holds the handler)")
    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 1
    command = json.loads(args.argv_json)
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command) or not command:
        print("--argv-json must be a non-empty JSON list of strings", file=sys.stderr)
        return 1

    download: Downloader | None = None
    progress: Callable[[], None] | None = None
    if args.results_name:
        access_key, secret_key = os.environ.get("RUNPOD_S3_ACCESS_KEY"), os.environ.get("RUNPOD_S3_SECRET_KEY")
        if not access_key or not secret_key:
            print("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY are required with --results-name", file=sys.stderr)
            return 1
        volume_id = resolve_volume_id(list_volumes(api_key), args.volume_name, args.datacenter)
        client = s3_client(args.datacenter, access_key, secret_key)
        download = lambda results_path, out: download_results(client, volume_id, results_path, out)  # noqa: E731
        progress = LogTail(client, volume_id, f"results/{args.results_name}/runner.log")

    job_input = {
        "argv": command,
        "results_dir": args.results_dir,
        "max_output_mb": args.max_output_mb,
        "git_commit": args.git_commit,
        "results_name": args.results_name,
    }
    source = pack_source(args.source_root, args.git_commit or "HEAD")
    job_input.update(source)
    print(f"shipping source {source['source_sha256'][:12]} ({source['source_size_bytes'] / 1e6:.2f} MB gzipped) of "
          f"{args.git_commit or 'HEAD'}")
    final = run(
        args.endpoint_url.rstrip("/"),
        api_key,
        job_input,
        args.timeout_minutes * 60,
        args.poll_seconds,
        progress,
        job_id_file=args.job_id_file,
        stale_retries=args.stale_retries,
        stale_wait_s=args.stale_wait_seconds,
    )
    return collect(final, args.out, download, expected_source=source["source_sha256"])


if __name__ == "__main__":
    sys.exit(main())
