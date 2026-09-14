# Directions

Research code for studying how low-dimensional control directions propagate through, and are transformed by, neural networks.

## Run on GPUs

Experiments can run on RunPod serverless GPUs through the `run-gpu` GitHub workflow. The workflow deploys a handler-only Runpod Flash app (`experiments/remote.py` and the `directions.remote` helpers), submits a job which has a `git archive` of the commit, and uploads the worker's `results/` directory as a workflow artifact.

Requirements: repository secrets `RUNPOD_API_KEY` and, for fetching results, `RUNPOD_S3_ACCESS_KEY` + `RUNPOD_S3_SECRET_KEY` (an S3 API key created in the RunPod console under Settings), and a pushed commit.

There are two ways to trigger a run. The primary way needs nothing but git push access (no Actions permission):
edit the request file `.github/gpu-run.yaml`, commit, and push. The workflow triggers on every push that changes that file and reads it; the request is versioned with the exact code it ran. An unchanged file does not trigger (we check the file's content across the push, so not every force-push or rebase triggers a run); re-requesting an identical command needs a new `request` label. `scripts/runpod_request.py` validates the file.

```bash
# request a run: edit the request file, commit, and push (command runs at the repository root on the worker)
$EDITOR .github/gpu-run.yaml
git commit -am "request: pilot3b_qwen3_0.6b_<SEED>" && git push -u origin <branch>

# or dispatch manually (the request file is ignored)
gh workflow run run-gpu.yml --ref <branch> \
  -f command='uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml' \
  -f gpu_tier=ADA_24

# follow it, then fetch the results artifact into results/remote
gh run list --workflow=run-gpu.yml --branch <branch> --limit 1
gh run watch <RUN_ID>
gh run download <RUN_ID> --dir results/remote      # artifact results-<flash_env>-<RUN_ID>-<SHA>
uv run --with boto3 python scripts/runpod_log.py <RUN_ID> --follow   # live worker log from the volume (needs the RunPod keys)
uv run python scripts/runpod_availability.py                          # which data centers have each tier's cards right now
```

Results are written to a network volume named `directions` (mounted at `/runpod-volume` on the worker, under `results/run-<GITHUB_RUN_ID>`) and downloaded through RunPod's S3-compatible API; the volume holds the Hugging Face cache as well. The volume lives in the `datacenter` input's data center (default `EUR-NO-1`), which pins the workers there. The GPU tier is a `runpod_flash.GpuGroup` name (16 to 80 GB options are listed in the workflow); a tier is a memory class (whose pool can contain different architectures), and the optional `gpu_type` input pins an exact device name instead (see the Compute section of `AGENTS.md`). Each run writes `runner.log` and `run_metadata.json` next to the results, the runner tails the log into the GitHub job log, and the workflow summary shows the metadata (`docs/INFRA.md` has the operational details). A run redeploys the endpoint only when the handler or the GPU/data-center settings changed (otherwise the recorded endpoint is reused; a redeploy first deletes the previous endpoint, `scripts/runpod_cleanup.py`), and one run executes at a time per Flash environment.

Helpers are standard library only: `src/directions/remote.py` ships and unpacks the source tree and the results, `scripts/runpod_job.py` submits and polls a job, `scripts/runpod_stage.py` stages the handler and computes its digest, `scripts/runpod_request.py` validates the request file. All are covered by `uv run pytest`. `runpod-flash` is installed only in CI, into a throwaway `.venv` on the runner (an editor could flag its import in `experiments/remote.py` locally; that is expected).
