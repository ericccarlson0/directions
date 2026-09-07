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
