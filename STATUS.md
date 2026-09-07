# Status

Last updated: 2026-09-07.

## Implemented (all exercised by `uv run pytest`, 39 tests)

`src/directions/`:

- `model.py` — Hugging Face model loading in bf16 with metadata; forward
  pre/post hooks capturing `resid[0..L]` at the final query token (float32,
  gathered inside the hook); interventions at any read point with a shared or
  per-example vector; teacher-forced scoring (log p per token, log p sum,
  exact match, first-token margin) verified against a native forward pass.
  A tiny random Qwen3-architecture model with a character tokenizer provides
  the no-download integration path.
- `tasks.py`, `data.py`, `prompts.py` — five deterministic tasks (antonym,
  plural, past tense, English→French, arithmetic `n → n+k`), token-length
  item filter with rejection logging, three disjoint pools per task,
  few-shot / zero-shot prompts, positive vs deranged demonstration pairs.
- `extraction.py`, `geometry.py` — uncentered PC1 of paired differences at
  every candidate layer, per-seed explained variance and `cos(PC1, mean)`,
  min-pairwise-|cos| stability, pooled direction; matched isotropic and
  orthogonal random controls.
- `calibration.py` — layer × strength sweep with paired bootstrap, minimum
  improvement, matched random-control screen and automatic grid extension.
- `layerwise.py`, `stats.py` — `S_l`, `log G_l`, `C_l`, `A_l` (median +
  bootstrap CI), centered/uncentered `d_eff`, `d90`, centered variance,
  `N_l` and `N_l^unc`, noise-floor diagnostic, per-layer z / empirical p
  against the random-control profiles.
- `analysis.py` — mode-discriminating quantities per task (alignment,
  cumulative log gain, conversion mass entropy / dominant block / centre of
  mass, `d_eff` ratio and trend, `N_l` z-scores) and rule-based labels.
- `ablation.py` — exploratory necessity/sufficiency of high-conversion blocks.
- `figures.py`, `pipeline.py`, `runinfo.py`, `cli.py` — figures, orchestration,
  run directories with resolved config / git commit / model / environment /
  seed table / rejections, and the `validate | pilot | check | compare` CLI.

Methodological choices are recorded in `docs/DECISIONS.md` (D1–D13).

## Runs completed

### Integration path (CPU, no downloads)

`uv run directions pilot --config configs/smoke_toy.yaml` runs every stage on
a random 4-layer model in ~2 s of compute (plus ~1 min of torch/transformers
import on this filesystem). Gates are computed but not enforced (D11). Two
runs are bit-identical (`tests/test_integration.py`).

### Qwen3-0.6B-Base — `configs/pilot_qwen3_0.6b.yaml`

```
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot_qwen3_0.6b_run1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot_qwen3_0.6b_run2
uv run directions compare results/pilot_qwen3_0.6b_run1 results/pilot_qwen3_0.6b_run2
```

Code commit `1a831e0`; model revision `da87bfb6`; torch 2.14.0 (cu130),
transformers 5.16.1; RTX 4090. Wall time 7.6 min (1.4 min of it model
import/load). The two runs are bit-identical in every JSON output; the only
diff reported by `compare` is `metadata.json/git/dirty` (docs were edited
between the runs). Results live in `results/` (git-ignored).

Candidate layers 6, 8, 11, 14, 17 (of 28). Per task:

| task | few-shot acc | stability (min over candidates) | selected (layer, ρ) | held-out Δ log p/token, p | vs 16 random (z, p) | outcome |
|---|---|---|---|---|---|---|
| antonym | 0.78 | 0.976 | 8, 0.5 | +0.286, 0.0005 | 1.04, 0.176 | rejected: not above random controls |
| plural | 1.00 | 0.990 | 8, 1.0 | +0.315, 0.0005 | 1.44, 0.059 | **qualified** |
| past_tense | 0.88 | 0.985 | 8, 0.2 | +0.044, 0.001 | 0.74, 0.235 | rejected: not above random controls |
| en_fr | 0.81 | 0.794 (layer 6 excluded) | none | – | – | rejected: no reliable calibration point |
| arithmetic | 0.98 | 0.395 at layer 6 (excluded); ≥0.836 elsewhere | 8, 0.8 | +0.204, 0.0005 | 1.56, 0.059 | **qualified** |

Every rejection and its numbers are in `results/pilot_qwen3_0.6b_run1/rejections.jsonl`.

What the layerwise measurements say for the two qualified tasks (`core/cross_task.json`, `core/tasks/*/layerwise.json`):

- **Alignment decays quickly.** `A_l` falls from 1 at layer 8 to ≈0.5 by
  layer 11 and to 0.09 (plural) / 0.14 (arithmetic) at the final read point.
  It is nevertheless *higher* than for random directions from layer ~12 on
  (z ≈ 1.5–3.4): the control direction is slightly better conserved than a
  random one, but it is not conserved transmission.
- **Amplification is generic.** Cumulative `log G` is 1.97 (plural) and 1.59
  (arithmetic), but the 16 matched random controls give 2.07 ± 0.19 and
  1.97 ± 0.22: a random direction of the same norm is amplified at least as
  much. Per-block `log G_l` is ≈0 for blocks 8–14 and ≈0.1–0.2 for blocks
  15–27 for both the real and random directions.
- **Conversion is distributed.** `C_l` ≈ 0.4–0.75 across all downstream
  blocks (entropy ratio 0.93–0.96, dominant block share 0.12–0.18); centre of
  mass 0.62–0.67 of the downstream depth. Labels: `cascade`, `amplification`
  (both), plus `dimensional_expansion` for plural (`d_eff` 8.0 → 14.7, peaking
  at 18.9 at layer 27) but not arithmetic (8.1 → 8.5).
- **Dimensionality and new-subspace creation are within the random null.**
  `N_l` and `N_l^unc` are ≈0.9–1.0 for every block (the block response lies
  almost entirely outside the incoming perturbation subspace) — but so they
  are for random directions (z ≈ 0, mean z ≤ 0). The same holds for `d_eff`
  and `d90`.
- **Noise floor is negligible.** At the intervention layer the observed
  centered variance is 2·10⁻⁴ (plural) and 1.2·10⁻³ (arithmetic) of the
  next layer's, with observed `d_eff` ≈ 48 (isotropic rounding, as
  predicted). `max |δ|` below the intervention layer is exactly 0.
- **Exploratory.** Strength robustness: rank correlations of the `log G_l`
  profile between the weakest and strongest reliable strengths are 0.96
  (plural) and 0.89 (arithmetic); `d_eff` 0.98 vs 0.05 (arithmetic's profile
  is flat, so its ranks are noise). Block ablation (rank by value, D13):
  plural block 27 — 34 % of the steering improvement lost, 25 % reproduced;
  arithmetic block 9 — 49 % lost, 37 % reproduced; removing the
  intervention block's own response (block 8) *increased* the improvement
  for plural.

Observations that matter for the next iteration (no methodology has been
changed; see `docs/DECISIONS.md` before changing any of it):

1. At the strengths the protocol selects (ρ = 0.5–1.0 of the median residual
   norm at layer 8), matched random directions move the held-out decision
   metric by up to ±1 nat/token in *either* direction. The random-control gate
   is therefore the binding one: antonym's +0.29 and past_tense's +0.04 are
   inside that spread.
2. Every selected layer is 8; layer 6 directions never passed the screens
   (for plural they *decrease* log p at every strength), and the two later
   layers 11–17 mostly decrease it.
3. Accuracy stays ≈0 under steering: the direction raises the probability of
   the correct target but does not make it the argmax.
4. "Smallest reliable strength" selects a weak effect for past_tense
   (ρ = 0.2, calibration Δ = +0.075) although the calibration pool shows
   +0.87 at ρ = 1.0 with all random screens passed. `calibration.min_improvement`
   is the knob that controls this; it was left at the preregistered 0.05.

### Qwen3-1.7B-Base — `configs/pilot_qwen3_1.7b.yaml`

```
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot_qwen3_1.7b_run1
```

Code commit `dcdaa45` (run1) / `8a6fcf2` (run2, docs-only change); model
revision `ea980cb0`; same environment. Wall time 14.5 / 15.4 min. The config
is identical to the 0.6B one except `model.name` (enforced by
`tests/test_configs.py`); the pipeline needed no changes. Run2
(`--run-id pilot_qwen3_1.7b_run2`) is bit-identical to run1 in every JSON
output; `compare` reports only the differing `metadata.json/git/commit`.

| task | few-shot acc | stability (min over candidates) | selected (layer, ρ) | held-out Δ log p/token, p | vs 16 random (z, p) | outcome |
|---|---|---|---|---|---|---|
| antonym | 0.88 | 0.977 | 11, 1.56 | +1.037, 0.0005 | 2.14, 0.059 | **qualified** |
| plural | 1.00 | 0.993 | 8, 0.05 | +0.098, 0.0005 | 2.58, 0.059 | **qualified** |
| past_tense | 0.92 | 0.992 | 6, 1.0 | +0.749, 0.0005 | 0.60, 0.412 | rejected: not above random controls |
| en_fr | 0.97 | 0.653 at layer 6 (excluded); ≥0.899 elsewhere | 8, 0.8 | +0.251, 0.0035 | 0.66, 0.412 | rejected: not above random controls |
| arithmetic | 1.00 | 0.808 | 11, 0.8 | +0.073, 0.0005 | 1.05, 0.235 | rejected: not above random controls |

Layerwise, qualified tasks:

- **plural (layer 8, ρ = 0.05)** is the cleanest profile of the pilot. A
  perturbation of 5 % of the residual norm is amplified by cumulative
  `log G` = 3.47 versus 3.20 ± 0.13 for random directions (z = 2.2, p = 0.059),
  with positive `log G_l` z-scores at most downstream blocks (up to 3.6), and
  is better aligned than random from layer 14 to 21 (z = 2–3.6). `d_eff`
  6.0 → 13.8; `N_l^unc` mean z = 1.2. Block ablation: blocks 9 and 19 each
  account for ≈50 % of the improvement lost and ≈60 % reproduced.
  Strength-robustness rank correlations ≥ 0.75 for every metric.
- **antonym (layer 11, ρ = 1.56)**: cumulative `log G` = 1.81, *less* than
  random (2.29 ± 0.24, z = −2.0); alignment decays from 1 to 0.13;
  `d_eff` 2.9 → 14.6 (ratio 5.1, the strongest expansion observed); `N_l`
  within the null. Blocks 18/19: 45 % / 25 % of the improvement lost.
- Labels for both: `cascade`, `amplification`, `dimensional_expansion`.

Additional observation: for past_tense at layer 6, ρ = 1.0, *all 16* random
directions improve the held-out decision metric (+0.04 to +0.96 nats/token).
A large perturbation at an early layer raises the probability of an
initially very unlikely target (zero-shot log p/token ≈ −7.5) regardless of
its direction, presumably by flattening the output distribution. The
random-control gate is what protects the protocol from this, and it is
binding for three of five tasks on this model too. A comparative metric
(the first-token logit margin is already recorded per condition) would be
less exposed to this; changing the decision metric is a methodological
change and has not been made.

### Across models

Two tasks qualified on each model (0.6B: plural, arithmetic; 1.7B: antonym,
plural). Plural qualified on both, at layer 8 on both, but at very different
strengths (ρ = 1.0 vs 0.05). Every qualified profile is `cascade` +
`amplification` (distributed conversion, `C_l` ≈ 0.4–0.9 everywhere, no
dominant block), never `conserved_transmission` or `delayed_activation`:
alignment to the injected direction is lost within a few blocks. Whether the
amplification and dimensional expansion are specific to the control
direction is decided by the random null, and the answer differs by case
(plural-1.7B yes for gain and alignment; the other three no, or less than
random). `cross_task.json` in each run holds the full table.

## Not yet run / known limitations

- `directions validate` on a real model has only been exercised through the
  pilot runs (which include the validation stages) and the toy path.
- The exploratory stages (strength robustness, block ablation) run only for
  tasks that qualified; rejected tasks have calibration grids and held-out
  steering results but no layerwise profile.
- Only two of five tasks qualified on each model. The pilot success
  criterion ("a few tasks") is met, but the block-level heatmaps hold two
  rows per model.
- Rejected items: one per run, the en_fr item "s'il vous plaît" (5 tokens > 4).
- Not started (follow-ups in `docs/EXPERIMENT.md`): canonical function-vector
  extraction, a second model family, more task families, component-level
  routing.
- Sensitivity of the qualification outcome to `calibration.min_improvement`,
  `random_control_max_p` and the number of random controls has not been
  explored; these are config parameters.

## Next commands

```bash
uv run pytest                                                            # 39 tests, ~3 min
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml           # 7–8 min on a 4090
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml           # ~15 min
uv run directions compare results/<run_a> results/<run_b>                # reproducibility diff
```

Suggested next steps, in order:

1. Decide whether to keep the preregistered "smallest reliable strength"
   rule or raise `calibration.min_improvement` (e.g. to 0.2 nats/token) so
   that the held-out test is run at a strength where the effect is not
   dominated by random-direction variance; record the decision in
   `docs/DECISIONS.md` and re-run both configs.
2. Consider adding a comparative decision metric (first-token margin is
   already recorded) as a second gate, given that random directions can
   raise log p of any unlikely target at large ρ (see above).
3. Extend candidate layers upward (`extraction.candidate_depth_fractions`)
   only if a task fails at every layer ≤ 60 % depth.
