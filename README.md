# Directions

Research code for measuring how a low-dimensional causal control direction
propagates through, and is transformed by, the layers of a transformer.
See `docs/PROJECT.md` (question), `docs/EXPERIMENT.md` (protocol),
`docs/DECISIONS.md` (implementation choices), `STATUS.md` (what has run) and
`docs/GENERIC_RESPONSE.md` (the closed line on the residual stream's generic response to any injected vector).

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
uv run directions pilot    --config configs/pilot_olmo3_7b.yaml     # same protocol, another family (OLMo 3; Gemma 4: pilot_gemma4_12b.yaml)
uv run directions validate --config configs/pilot_olmo3_7b.yaml --stop-after fewshot   # smoke run of a new model: load, targets, few-shot gates
uv run python scripts/probe_edit_sensitivity.py --run results/<pilot run> --tasks antonym   # how a model reacts to residual edits at later read points (D35)
uv run python scripts/checkpoint_norm_gains.py google/gemma-4-12B --layers 12 24 47 --coords 1750   # a checkpoint's norm gains, without downloading it (D35)
uv run directions compare  results/<run_a> results/<run_b>          # reproducibility diff of two runs
uv run directions pilot    --config configs/smoke_toy.yaml          # tiny random model, CPU, no downloads
uv run directions pilot    --config configs/pilot_qwen3_0.6b.yaml --seed 1 --run-id r1   # seed override
uv run directions aggregate results/r1 results/r2 results/r3 --out results/aggregate.json # multi-seed summary
uv run directions trajectories --config configs/trajectories.yaml --fv-run results/<head-mean run> \
    --learned-run results/<learned-vector run> --run-id t1   # downstream trajectories compared all-to-all (D32)
uv run directions trajectories --config configs/trajectories_subspace.yaml --fv-run results/<head-mean run> \
    --learned-run results/<learned-vector run> --run-id s1   # the causal dimensionality of the shared component only (D34)
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

A `trajectories` run (docs/DECISIONS.md D32) reads two finished runs and writes its own directory:

```
results/<run_id>/
  metadata.json            the comparison's config, the two runs read (paths, ids, commits), model, determinism check
  core/summary.json        per task, layer and construction: strength (and its source), held-out effect, the alignment
                           summary against each natural trajectory and each other construction, the ceiling
  core/tasks/<task>/
      trajectories.json    per layer: strengths, condition metrics, every pair's median-cosine curve with CIs and floor,
                           the variants without the answer direction and without the generic response, the
                           summaries and labels, the ceilings; the same at the other strength factors, the
                           cross-strength cosines, the generic-response diagnostics and the patch test (D33);
                           the role of every compared layer (primary, the head-mean run's, a neighbouring
                           candidate) when `neighbour_layers` adds the candidates around the primary one;
                           the per-block writing of the shared component per trajectory (D33 amended);
                           the subspace test (D34): the effect retained within k-dimensional subspaces of the natural
                           differences (own task, other tasks, background, random) at the primary layer
      trajectories_arrays.npz  per-example cosines of every pair at every read point (raw and both variants),
                           projection fractions, per-block writing of the shared component, floor medians,
                           population-mean trajectories, the generic response
  figures/                 per task and layer: cosine with the natural trajectory, the same without the answer
                           direction and without the generic response, the constructions against one another;
                           alignment by strength, coherence and location of the generic response, the patch test,
                           the per-block writing of the shared component
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
uv run python scripts/runpod_balance.py --need 6                      # exit 1 unless the RunPod balance covers 6 USD plus a margin
uv run python scripts/landmarks.py --config configs/trajectories_landmarks.yaml --fv-run <pilot> --learned-run <learned> --run-id landmarks_<model> [--lens-file <hub path>]   # the landmark test (D37) on one model
uv run python scripts/landmark_summary.py qwen3_0.6b=results/landmarks_qwen3_0.6b_seed20260916 ... --figure results/landmarks.png   # across models
uv run directions source --config configs/source.yaml --fv-run <pilot> --learned-run <learned> --landmark-run <landmark run> --run-id source_<model>   # the source test (D38)
uv run python scripts/family_geometry.py --learned-run <learned family run> --fv-run <head-mean family run> --labels kth_1,kth_2,kth_3 --params 1,2,3   # the within-family geometry (D39)
uv run directions mixing --config configs/mixing.yaml --runs kth_word=<learned family run>:<head-mean family run> add_k=<learned add-k run>: lexical=<learned run>:<pilot run> --run-id mixing_<model>   # the geometry test (D40): mixed controls read as distributions
uv run python scripts/landmarks.py --config configs/trajectories_composition.yaml --fv-run <comp run> --learned-run <comp_learned run> --run-id comp_landmarks_<model>   # the composition test (D41): the landmark comparison with the components as references
uv run python scripts/composition_summary.py qwen3_0.6b=<comp landmark run> ... --out results/composition_summary.json   # across models
```

Results are written to a network volume named `directions` (mounted at `/runpod-volume` on the worker, under `results/run-<GITHUB_RUN_ID>`) and downloaded through RunPod's S3-compatible API; the volume holds the Hugging Face cache as well. The volume lives in the `datacenter` input's data center (default `EUR-NO-1`), which pins the workers there. The GPU tier is a `runpod_flash.GpuGroup` name (16 to 80 GB options are listed in the workflow); a tier is a memory class (whose pool can contain different architectures), and the optional `gpu_type` input pins an exact device name instead (see the Compute section of `AGENTS.md`). Each run writes `runner.log` and `run_metadata.json` next to the results, the runner tails the log into the GitHub job log, and the workflow summary shows the metadata (`docs/INFRA.md` has the operational details). A run redeploys the endpoint only when the handler or the GPU/data-center settings changed (otherwise the recorded endpoint is reused; a redeploy first deletes the previous endpoint, `scripts/runpod_cleanup.py`), and one run executes at a time per Flash environment (`flash_env`; requests with different environments run in parallel on their own endpoints).

Helpers are standard library only: `src/directions/remote.py` ships and unpacks the source tree and the results, `scripts/runpod_job.py` submits and polls a job, `scripts/runpod_stage.py` stages the handler and computes its digest, `scripts/runpod_request.py` validates the request file. All are covered by `uv run pytest`. `runpod-flash` is installed only in CI, into a throwaway `.venv` on the runner (an editor could flag its import in `experiments/remote.py` locally; that is expected).
