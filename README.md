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
