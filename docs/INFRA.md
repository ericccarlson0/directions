# Infrastructure Notes

Facts about the RunPod path that are perhaps not obvious and perhaps had to be learned from a failed run. Read this before changing e.g. `.github/workflows/run-gpu.yml`, `experiments/remote.py`, `scripts/runpod_*.py` or `src/directions/remote.py`. How to dispatch a run is in `README.md`.

## The flow of a run

0. The workflow starts either from a push that modifies the request file `.github/gpu-run.yaml` (its fields are the run parameters; `scripts/runpod_request.py` validates them inside the job) or from a `workflow_dispatch` form (whose inputs are used instead). The push path exists because a session token with git push access may still lack the Actions write permission that `workflow_dispatch` needs (Claude Code web sessions get a narrowed token: pushes succeed, dispatches return HTTP 403 "Resource not accessible by integration"). The concurrency group of a push run is always `run-gpu-ci`, since the file is only read inside the job.
1. The workflow builds a `.venv` on the runner with the pipeline's pinned deps (CPU torch) and `runpod-flash`, deletes the previous endpoint in the Flash environment, and runs `flash deploy`, which uploads the repo as an artifact and creates a fresh endpoint attached to the network volume `directions`.
2. The runner submits one job. The worker unpacks the artifact to `/app`, builds a venv from `uv.lock` (torch from the lock, not from the image), and runs the command with `results/` on local disk.
3. The child's output streams to the container log, to `results/runner.log`, and to the same file on the volume; the runner tails that file over S3 as it polls (progress shows in the GitHub job log).
4. The results directory is copied to the volume on exit, in `results/run-<GITHUB_RUN_ID>`, the runner downloads it over S3 into `results/remote`, and the workflow uploads as an artifact.

## Flash (runpod-flash)

- `flash deploy` is only idempotent through the pickle in `.flash/`, which a fresh runner does not have; a second deploy re-creates the endpoint and RunPod rejects the duplicate template name. Hence the cleanup step, which deletes the environment's endpoint through GraphQL first. `flash undeploy` reads the same pickle.
- The build imports every shipped `.py` file (without `__init__.py`) to find the endpoint and errors on any import error, so the pipeline's deps must be importable where `flash deploy` runs, and package code must not `from . import name` for a name defined in `__init__.py`.
- torch, torchvision, torchaudio and triton are stripped from the bundled deps; the lock's `nvidia-*` and `cuda-*` Linux pins would be bundled (gigabytes), so the workflow filters them out of `requirements.txt`. The bundled deps are built for the `--python-version` (3.11) while the image runs 3.12 (pipeline does not use them; it runs from the uv venv).
- FlashBoot restores workers from a snapshot of an earlier boot on the same image, `/app` included, so after a redeploy workers ran the previous build. It is off. Also, the handler returns the on-disk source fingerprint next to the one the endpoint was deployed with, and the runner fails on a mismatch.
- A Flash environment name is part of the endpoint template's env, so two environments never collide on template names; run branches in parallel by giving each its own `flash_env`.

## RunPod

- `api.runpod.io` (GraphQL and REST v2) answers HTTP 403 to Python's default urllib user agent; the scripts send their own. The API on `api.runpod.ai` does not care.
- REST v2 has no endpoint-management routes; endpoints are managed through GraphQL. A missing Flash app or environment is a GraphQL "not found" error, not a null result.
- GPU tiers are memory classes whose pools mix architectures (`AMPERE_16` holds Ampere and Ada cards; only `ADA_24` is a single card). A host's driver can be older than the lock's torch build: the lock's torch is a CUDA 13 build, a 12.8 host ran the whole pilot on the CPU, 50-170x slower, with `device: auto`. The endpoint requires CUDA >= 13 (`DIRECTIONS_MIN_CUDA`) and the pilot configs pin `device: cuda`.
- The output of a job is not a results channel: one 0.6B pilot was 24 MB encoded and RunPod documents no ceiling. Results arrive through the volume and its S3 API (`https://s3api-<dc>.runpod.io`, region = data center id, bucket = volume id, separate S3 API key). The volume pins the endpoint to its data center.
- For reference, a cold start is ~30-90 s; the uv venv build on the worker ~30 s; a 0.6B fable pilot 42 min on an RTX 4090 (before optimizing and lowering sample size).
- Workers see the host's CPU count (120 on one host) with no cgroup quota; BLAS and torch threads are capped by the CLI (`DIRECTIONS_BLAS_THREADS`).

## Reading a run from outside

- Live: the GitHub job log (tailed from the volume), or the worker's container log (`stream-worker-logs` with the id from `list-endpoint-workers`).
- After: the workflow artifact, or `results/run-<id>/` on the volume over S3.
- A second worker is allowed on the endpoint; a probe job can run beside a pilot; the handler runs any `argv`, so a probe can read the volume, the log, or the host's state.
