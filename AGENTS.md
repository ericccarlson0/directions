# Directions

Research code for studying the propagation and transformation (from control into computation) of low-dimensional control directions through neural networks.

## Reference

1. `docs/PROJECT.md` for the scientific question.
2. `docs/EXPERIMENT.md` for the current experiment specification.
3. `STATUS.md` for what has and has not been implemented/run.
4. `docs/DECISIONS.md` before changing an established methodological choice.

## Principles

- Scientific correctness and reproducibility take priority over convenience.
- Separate reusable analysis code (`src/directions/`) from experiments.
- Experiments must be configurable.
- Set and record random seeds.
- Save sufficient metadata to reproduce every result.
- Add tests for e.g. numerical/statistical utilities where practical.

## Workflow

Before implementing a substantial methodological change:
- state what is changing and why;
- update `docs/DECISIONS.md`.

After a successful experimental milestone:
- update `STATUS.md`;
- record the exact command/config used.
