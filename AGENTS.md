# Directions

Research code for studying the propagation and transformation (from control into computation) of low-dimensional control directions through neural networks.

## Reference

1. `docs/PROJECT.md` for the scientific question and the core measurements (e.g. how to distinguish between propagation modes).
2. `docs/EXPERIMENT.md` for the current experiment specification.
3. `STATUS.md` for what has and has not been implemented/run.
4. `docs/DECISIONS.md` before changing an established methodological choice.

## Principles

- Scientific correctness and reproducibility take priority over convenience.
- Separate reusable analysis code (`src/directions/`) from experiments.
- Experiments must be configurable; every scientifically meaningful parameter in a version-controlled config file.
- Set and record random seeds. (Note: sub-seeds from the run seed through a stable digest, not through Python's per-process-salted `hash()`.) Prove reproducibility; run the same command twice and diff.
- Save sufficient metadata to reproduce every result.
- Prefer statistical criteria (paired tests, matched-control nulls) over fixed effect-size thresholds wherever sampling noise could be comparable to the effect.
- Keep exploratory diagnostics distinct from preregistered/core metrics.
- Add tests for e.g. numerical/statistical utilities where practical. Verify batched code paths against native references.

## Workflow

Before implementing a substantial methodological change:
- state what is changing and why;
- update `docs/DECISIONS.md`.

After a successful experimental milestone:
- update `STATUS.md`;
- record the exact command/config used.

## Compute

GPU runs go through the `run-gpu` workflow (see `README.md`). Be sure to think through the GPU tier yourself (the following heuristics are standard):

- Weights: bf16 needs 2 bytes per parameter (e.g. 8B ≈ 16 GB, 32B ≈ 64 GB). Add 25% or so for activations, the KV cache and captured residuals (prompts here are brief and no gradients are stored, so activations should not occupy too much). Pick the smallest tier whose memory exceeds that total. Do not quantize to fit a smaller tier (the spec forbids it, as well).
- A tier is a memory class, not an architecture: `AMPERE_24` holds the A5000, the L4 and the 3090, for instance (whereas `ADA_24` holds only RTX 4090). Two runs on the same tier can land on different architectures, which can have different numerics. When comparability (bit-identical diff) with earlier runs matters, pin the exact card with the workflow's `gpu_type` input (the device name in the earlier run's `run_metadata.json`) and record it with the run.
- Note that oversizing is not free; only go up a tier for a reason (OOM failure, a batch size the analysis needs) and note it.
- Flash's build imports every module under `src/` in isolation (without `__init__.py`) to find the endpoint, so the package must import module by module: no `from . import name` for a name defined in `__init__.py`, and no import-time side effects that need a GPU or a download.
