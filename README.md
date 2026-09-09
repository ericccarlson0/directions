# Directions

Research code for studying how low-dimensional control directions propagate through, and are transformed by, neural networks.

## Run on GPUs

Experiments can run on RunPod serverless GPUs through the `run-gpu` GitHub workflow. The workflow deploys the current commit as a Runpod Flash app (`experiments/remote.py`), runs one command at the repository root on a worker, and uploads the worker's `results/` directory as a workflow artifact.

Requirements: repository secrets `RUNPOD_API_KEY` and, for fetching results, `RUNPOD_S3_ACCESS_KEY` + `RUNPOD_S3_SECRET_KEY` (an S3 API key created in the RunPod console under Settings), and a pushed commit.

```bash
# dispatch a run (command runs at the repository root on the worker)
gh workflow run run-gpu.yml \
  -f command='uv run directions pilot --config configs/pilot_<MODEL_NAME>.yaml' \
  -f gpu_tier=ADA_24

# follow it, then fetch the results artifact into results/remote
gh run watch
gh run download --name "results-ci-<RUN_ID>-<SHA>" --dir results/remote
```

Results are written to a network volume named `directions` (mounted at `/runpod-volume` on the worker, under `results/run-<GITHUB_RUN_ID>`) and downloaded through RunPod's S3-compatible API; the volume holds the Hugging Face cache as well. The volume lives in the `datacenter` input's data center (default `EUR-NO-1`), which pins the workers there. The GPU tier is a `runpod_flash.GpuGroup` name (16 to 80 GB options are listed in the workflow); a tier is a memory class (whose pool can contain different architectures), and the optional `gpu_type` input pins an exact device name instead (see the Compute section of `AGENTS.md`). Each run writes `runner.log` and `run_metadata.json` next to the results, and the workflow summary shows the metadata. Each dispatch deletes the previous endpoint and deploys a fresh one (`scripts/runpod_cleanup.py`), and *one* deploy-and-run executes at a time per Flash environment.

Helpers are standard library only: `src/directions/remote.py` packs and unpacks results, `scripts/runpod_job.py` submits and polls a job. Both are covered by `uv run pytest`. Note that `runpod-flash` is installed only in CI, into a throwaway `.venv` on the runner together with the pipeline's deps and CPU torch, because Flash's build imports every module to find the endpoint (an editor could flag its import locally; that is expected).
