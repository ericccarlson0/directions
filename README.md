# Directions

Research code for measuring how a low-dimensional causal control direction
propagates through, and is transformed by, the layers of a transformer.
See `docs/PROJECT.md` (question), `docs/EXPERIMENT.md` (protocol),
`docs/DECISIONS.md` (implementation choices) and `STATUS.md` (what has run).

## Setup

```bash
uv sync            # Python 3.11, torch, transformers, numpy, matplotlib, pyyaml
uv run pytest      # unit tests + the CPU-only integration path (~2 min, dominated by imports)
```

## Commands

```bash
uv run directions check    --config configs/pilot_qwen3_0.6b.yaml   # print the resolved config
uv run directions validate --config configs/pilot_qwen3_0.6b.yaml   # task/control validation only
uv run directions pilot    --config configs/pilot_qwen3_0.6b.yaml   # full pilot (+ layerwise, exploratory, figures)
uv run directions pilot    --config configs/pilot_qwen3_1.7b.yaml   # same protocol, second model
uv run directions compare  results/<run_a> results/<run_b>          # reproducibility diff of two runs
uv run directions pilot    --config configs/smoke_toy.yaml          # tiny random model, CPU, no downloads
uv run directions pilot    --config configs/pilot_qwen3_0.6b.yaml --seed 1 --run-id r1   # seed override
uv run directions aggregate results/r1 results/r2 results/r3 --out results/aggregate.json # multi-seed summary
```

Every scientifically meaningful parameter lives in the YAML config. Each run
writes a uniquely named directory under `results/`:

```
results/<run_id>/
  metadata.json            run id, command, git commit, model, environment, seed table, timings, the
                           run profile (per-stage seconds, forward counts, peak memory) and the
                           in-run determinism check (a repeated pass and profile, bit-identical or not)
  config.resolved.yaml     the exact configuration used
  rejections.jsonl         every automatic filtering decision, with the numbers behind it
  log.txt
  core/                    preregistered outputs
    summary.json           per-task gates, selections, labels
    cross_task.json        mode-discriminating quantities per task (docs/PROJECT.md)
    tasks/<task>/
      splits.json          the three disjoint query pools
      qualification.json   few-shot gate, stability, selection, held-out steering vs random controls
      extraction.json      per-layer, per-seed PC1 statistics and cross-seed stability; with the learned-vector
                           control (D31) also the fits' loss trajectories, radii and stability per layer
      directions.npz       pooled and per-seed unit directions at the candidate layers (PC1; the function vector
                           and its heads, or the learned vectors and their radii, when they are the control)
      calibration.json     the layer x strength grid with bootstrap tests and random screens
      evaluation.json      held-out behavioural metrics per condition, every control by kind,
                           the gate comparison (isotropic + orthogonal), per-kind comparisons and the
                           damage measure (KL from the unsteered distribution at the query token and on
                           neutral prose, D24) with its excess tests
      layerwise.json       S_l, log G_l, C_l, A_l (median + CI), d_eff, d90, N_l, noise floor;
                           every control's metric curves; null summaries and per-layer z /
                           empirical p for the primary null and for each structured null
                           (covariance, other_task, demo_variation)
      layerwise_arrays.npz per-example arrays
      decomposition.json   the vector's common and task-specific parts (D28): effect, damage, profile, labels of each
      commitment.json      depth of commitment (D29): the effect surviving the removal (or sole keeping) of the
                           injected direction at every later read point, against random directions
  exploratory/tasks/<task>/
      strength_robustness.json   profiles at the other reliable strengths (weakest / middle / strongest)
                                 + rank correlations with the selected one
      block_ablation.json        necessity / sufficiency of the largest-conversion blocks
  figures/                 per-task primary figures, diagnostics, cross-task heatmaps
```

## Layout

- `src/directions/` reusable analysis code (model backend, geometry, statistics,
  extraction, calibration, layerwise metrics, analysis, figures, pipeline, CLI).
- `configs/` version-controlled experiment specifications.
- `tests/` unit tests for the numerical/statistical utilities, backend checks
  against native forward passes, and the integration path.

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

Results are written to a network volume named `directions` (mounted at `/runpod-volume` on the worker, under `results/run-<GITHUB_RUN_ID>`) and downloaded through RunPod's S3-compatible API; the volume holds the Hugging Face cache as well. The volume lives in the `datacenter` input's data center (default `EUR-NO-1`), which pins the workers there. The GPU tier is a `runpod_flash.GpuGroup` name (16 to 80 GB options are listed in the workflow); a tier is a memory class (whose pool can contain different architectures), and the optional `gpu_type` input pins an exact device name instead (see the Compute section of `AGENTS.md`). Each run writes `runner.log` and `run_metadata.json` next to the results, the runner tails the log into the GitHub job log, and the workflow summary shows the metadata (`docs/INFRA.md` has the operational details). A run redeploys the endpoint only when the handler or the GPU/data-center settings changed (otherwise the recorded endpoint is reused; a redeploy first deletes the previous endpoint, `scripts/runpod_cleanup.py`), and one run executes at a time per Flash environment (`flash_env`; requests with different environments run in parallel on their own endpoints).

Helpers are standard library only: `src/directions/remote.py` ships and unpacks the source tree and the results, `scripts/runpod_job.py` submits and polls a job, `scripts/runpod_stage.py` stages the handler and computes its digest, `scripts/runpod_request.py` validates the request file. All are covered by `uv run pytest`. `runpod-flash` is installed only in CI, into a throwaway `.venv` on the runner (an editor could flag its import in `experiments/remote.py` locally; that is expected).
