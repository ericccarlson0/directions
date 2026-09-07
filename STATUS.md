# Status

Last updated: 2026-09-07 (see `git log` for the authoritative history).

## Summary

| Component | Implemented | Run |
|---|---|---|
| Package, config system, CLI | yes | yes |
| Offline integration path (`configs/smoke.yaml`) | yes | yes, passing |
| Unit + integration test suite | yes | yes, 130 passing |
| Qwen3-0.6B-Base pilot | yes | not yet |
| Qwen3-1.7B-Base pilot | yes (config only differs) | not yet |

## Implemented

All components required by `docs/EXPERIMENT.md` are in `src/directions/`:

* `model.py` — HF Transformers loading, model metadata, batched teacher-forced
  scoring, residual capture. Second backend `tiny_random` (randomly-initialised
  Qwen3 + byte tokenizer) for offline testing.
* `hooks.py` — residual-stream capture at all `L+1` read points and additive
  intervention at a chosen layer/token; block-output patching for ablation.
* `tasks.py`, `prompts.py` — five deterministic tasks (229-240 items each),
  seeded disjoint splits, few-shot prompts, positive/permuted demonstration
  pairs via derangement.
* `extraction.py` — PC1 of paired activation differences, multi-seed extraction,
  cross-seed stability, `rho -> alpha` conversion.
* `controls.py` — matched isotropic and orthogonal random controls.
* `calibration.py` — layer x strength sweep and selection rule.
* `layerwise.py` — `S_l`, `G_l`, `C_l`, `d_eff`, `d90`, `N_l` (+ uncentered
  companion), control alignment, bootstrap summaries, control null aggregation.
* `profiles.py` — automatic qualitative labelling (exploratory).
* `figures.py` — the six primary figure families plus calibration/stability
  diagnostics.
* `serialization.py`, `metadata.py` — run directories, resolved config, git
  commit, environment, seeds, rejection log.
* `pipeline.py`, `cli.py` — orchestration and the `validate` / `pilot` commands.

## Verified so far

* `uv run pytest` — **130 passed** (unit tests for effective rank, PCA
  extraction, normalization, orthogonal controls, blockwise decomposition,
  gain/conversion, bootstrap; hook-exactness tests; end-to-end integration).
* Batched, padded, length-sorted scoring is numerically identical to naive
  unbatched scoring (`test_scoring_matches_an_unbatched_unpadded_reference`).
* The intervention is exact: `delta_l = 0` upstream, `delta_l = alpha*v` at the
  intervention layer (`test_intervention_is_exact_at_its_own_layer`).
* The offline pilot (`configs/smoke.yaml`) completes end to end in ~50 s on CPU
  and produces every required artefact, including all 17 figures.
* Two identical smoke runs produce bit-identical reported numbers.

## Not yet run

* The Qwen3-0.6B-Base pilot (weights are cached locally).
* The Qwen3-1.7B-Base pilot.

## Next commands

```bash
uv run pytest
uv run directions validate --config configs/qwen3_0.6b.yaml
uv run directions pilot    --config configs/qwen3_0.6b.yaml
```
