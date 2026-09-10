# Directions

Research code for measuring how a low-dimensional causal control direction
propagates through, and is transformed by, the layers of a transformer.

* The scientific question: `docs/PROJECT.md`
* The experiment specification: `docs/EXPERIMENT.md`
* Methodological choices and their justifications: `docs/DECISIONS.md`
* What has actually been implemented and run: `STATUS.md`

## Quick start

```bash
uv sync --extra dev
uv run pytest                                        # ~90 s, no GPU, no download
uv run directions pilot --config configs/smoke.yaml  # ~50 s, offline end-to-end run
```

The real pilots need a GPU and will download the checkpoint on first use:

```bash
uv run directions validate --config configs/qwen3_0.6b.yaml   # task qualification only
uv run directions pilot    --config configs/qwen3_0.6b.yaml   # + layerwise measurement
uv run directions pilot    --config configs/qwen3_1.7b.yaml   # identical pipeline, larger model
```

## Commands

| Command | Purpose |
|---|---|
| `directions validate --config C` | Few-shot qualification, direction extraction and stability, intervention calibration, held-out steering, matched random controls. |
| `directions pilot --config C` | Everything in `validate`, plus the downstream layerwise measurement, control nulls, exploratory analyses and figures. |
| `directions tasks` | List the registered task datasets. |
| `directions show-config --config C` | Print the fully resolved config and its fingerprint. |
| `directions inspect PATH` | Print the summary of a completed run. |

`--set key.path=value` overrides any config field (recorded in the run's
resolved config); `--tasks a,b` and `--output-root DIR` are shorthands.

## What a run produces

```
results/<timestamp>__<command>__<name>__<config-fingerprint>/
  config.resolved.yaml     every parameter, after `extends:` and overrides
  metadata.json            git commit, environment, package versions, seeds, model metadata
  summary.json             qualified tasks, selected operating points, cross-task summary
  core/                    preregistered measurements
    validation__<task>.json      all five qualification stages with their statistics
    layerwise__<task>.json       S, log G, C, d_eff, d90, N, control nulls, z/p vs nulls
    layerwise__<task>.npz        the per-example arrays behind those metrics
    cross_task_summary.json      the quantities that discriminate the propagation modes
  exploratory/             profile labels, block ablation, strength-robustness replicate
  figures/                 the six primary figure families plus diagnostics
  rejections.jsonl         every automatic filtering decision, with the numbers behind it
  log.txt                  human-readable progress log
```

## Package layout

```
src/directions/
  config.py         typed configuration; unknown keys are a hard error
  tasks.py          deterministic task datasets and seeded disjoint splits
  prompts.py        few-shot prompts, positive/permuted demonstration pairs
  model.py          HF loading, metadata, batched teacher-forced scoring, residual capture
  hooks.py          residual-stream capture and intervention (see the indexing note)
  extraction.py     PC1 of paired activation differences, multi-seed stability
  controls.py       matched isotropic and orthogonal random directions
  calibration.py    layer x strength sweep and the selection rule
  layerwise.py      the primary geometric metrics
  mathx.py          effective rank, PCA, projections, bootstrap, stable seeding
  profiles.py       automatic qualitative labels (exploratory)
  figures.py        automatic figure generation
  pipeline.py       orchestration
  serialization.py  run directories and JSON/NPZ output
  metadata.py       git, environment and package provenance
```

## Residual-stream indexing

For `L` transformer blocks there are `L + 1` read points: `resid[0]` is the
embedding output, `resid[l]` is the output of block `l-1`, and `resid[L]` is the
final pre-norm residual. Block `l` maps `resid[l] -> resid[l+1]`, so
`b_l = delta_{l+1} - delta_l` is exactly block `l`'s contribution. An
intervention "at layer `l`" adds to `resid[l]` — the input of block `l` — at the
final query token.

## Testing

`uv run pytest` runs unit tests for the numerical utilities (effective rank,
PCA extraction, normalization, orthogonal control construction, blockwise
decomposition, gain/conversion, bootstrap), exactness tests for the hook
machinery, and a full end-to-end integration run on a randomly-initialised tiny
Qwen3 with a byte-level tokenizer — no download, no network, no GPU.

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
```

Results are written to a network volume named `directions` (mounted at `/runpod-volume` on the worker, under `results/run-<GITHUB_RUN_ID>`) and downloaded through RunPod's S3-compatible API; the volume holds the Hugging Face cache as well. The volume lives in the `datacenter` input's data center (default `EUR-NO-1`), which pins the workers there. The GPU tier is a `runpod_flash.GpuGroup` name (16 to 80 GB options are listed in the workflow); a tier is a memory class (whose pool can contain different architectures), and the optional `gpu_type` input pins an exact device name instead (see the Compute section of `AGENTS.md`). Each run writes `runner.log` and `run_metadata.json` next to the results, the runner tails the log into the GitHub job log, and the workflow summary shows the metadata (`docs/INFRA.md` has the operational details). A run redeploys the endpoint only when the handler or the GPU/data-center settings changed (otherwise the recorded endpoint is reused; a redeploy first deletes the previous endpoint, `scripts/runpod_cleanup.py`), and one run executes at a time per Flash environment.

Helpers are standard library only: `src/directions/remote.py` ships and unpacks the source tree and the results, `scripts/runpod_job.py` submits and polls a job, `scripts/runpod_stage.py` stages the handler and computes its digest, `scripts/runpod_request.py` validates the request file. All are covered by `uv run pytest`. `runpod-flash` is installed only in CI, into a throwaway `.venv` on the runner (an editor could flag its import in `experiments/remote.py` locally; that is expected).
