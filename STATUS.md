# Status

Last updated: 2026-09-08.

## Implemented (all exercised by `uv run pytest`, 44 tests)

`src/directions/`:

- `model.py` — Hugging Face model loading in bf16 with metadata; forward
  pre/post hooks capturing `resid[0..L]` at the final query token (float32,
  gathered inside the hook); interventions at any read point with a shared or
  per-example vector; teacher-forced scoring (log p per token, log p sum,
  exact match, first-token margin) verified against a native forward pass.
  A tiny random Qwen3-architecture model with a character tokenizer provides
  the no-download integration path.
- `tasks.py`, `data.py`, `data_extra.py`, `prompts.py` — ten deterministic
  tasks (antonym, plural, past tense, English→French, `n → n+3`, present
  participle, singular, uppercase, number→words, two-operand addition; all
  with ≥ 320 usable items, D14), token-length item filter with a per-task
  override and rejection logging, three disjoint pools, few-shot / zero-shot
  prompts, positive vs deranged demonstration pairs.
- `extraction.py`, `geometry.py` — uncentered PC1 of paired differences at
  every candidate layer, per-seed explained variance and `cos(PC1, mean)`,
  min-pairwise-|cos| stability, pooled direction; demonstration-variation
  null directions; isotropic, orthogonal and residual-covariance-matched
  random controls; Gram-trick spectra.
- `calibration.py` — layer × strength sweep with paired bootstrap, minimum
  improvement, 16-control matched random screen and automatic grid extension.
- `layerwise.py`, `stats.py` — `S_l`, `log G_l`, `C_l`, `A_l` (median +
  bootstrap CI), centered/uncentered `d_eff`, `d90`, centered variance,
  `N_l` and `N_l^unc`, noise-floor diagnostic; per-layer z / empirical p
  against the primary (isotropic + orthogonal) null and against each
  structured null (covariance, other-task, demonstration-variation) (D15).
- `analysis.py`, `aggregate.py` — mode-discriminating quantities per task,
  rule-based labels, per-kind z summaries; multi-seed aggregation (D16).
- `ablation.py` — exploratory necessity/sufficiency of high-conversion blocks.
- `figures.py`, `pipeline.py` (two phases, D15), `runinfo.py`, `cli.py`
  (`validate | pilot | check | compare | aggregate`, `--seed`).

Methodological choices are recorded in `docs/DECISIONS.md` (D1–D16).
`docs/AGENT_COMPARISON.md` compares this branch with the `opus` branch.

## Runs completed

### Integration path (CPU, no downloads)

`uv run directions pilot --config configs/smoke_toy.yaml` runs every stage
(including the calibration screen and all five control kinds) on a random
4-layer model in ~25 s. Gates are computed but not enforced (D11). Two runs
are bit-identical (`tests/test_integration.py`).

### Iteration 1 (superseded protocol; commits `1a831e0`–`8a6fcf2`)

64 held-out examples, five tasks, 16 random controls. Two tasks qualified per
model (0.6B: plural, arithmetic; 1.7B: antonym, plural); every qualified
profile was cascade + amplification; gain, `d_eff` and `N_l` sat inside the
isotropic null except plural-1.7B. Reruns were bit-identical. Superseded by
iteration 2 because the 16-control null could not be decisive (minimum
p = 1/17) and `n = 64` bounded the dimensionality metrics (D14, D15).

### Iteration 2 (current protocol; commit `fe052ed`, configs unchanged since)

```
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot2_qwen3_0.6b_seed20260907
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 1 --run-id pilot2_qwen3_0.6b_seed1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 2 --run-id pilot2_qwen3_0.6b_seed2
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot2_qwen3_1.7b_seed20260907
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --seed 1 --run-id pilot2_qwen3_1.7b_seed1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --seed 2 --run-id pilot2_qwen3_1.7b_seed2
uv run directions aggregate results/pilot2_qwen3_0.6b_seed* --out results/aggregate_qwen3_0.6b.json
uv run directions aggregate results/pilot2_qwen3_1.7b_seed* --out results/aggregate_qwen3_1.7b.json
```

Per run: 192 held-out examples, ten tasks, candidate layers 6/8/11/14/17,
64 preregistered random controls (32 isotropic + 32 orthogonal; the gate
requires beating ≥ 61 of them, p ≤ 0.05), plus 32 covariance-matched, up to 9
other-task and 8 demonstration-variation directions. Wall time 7.6–9.6 min
(0.6B) and 13.4 min (1.7B) on the RTX 4090; torch 2.14, transformers 5.16.
Model revisions `da87bfb6` (0.6B), `ea980cb0` (1.7B). Results in `results/`
(git-ignored).

#### Qualification (gate = 64 random controls, p ≤ 0.05)

Qwen3-0.6B-Base, three seeds (20260907, 1, 2):

| task | qualified | selections (layer, ρ) | held-out Δ log p/token (seed 20260907) | gate z (seed 20260907) |
|---|---|---|---|---|
| arithmetic (n+3) | 3/3 | (8, 1), (11, 1), (17, 0.2) | +0.25 | 2.35 |
| uppercase | 3/3 | (6, 0.1), (6, 0.05), (6, 0.05) | +0.09 | 2.26 |
| past_tense | 2/3 | (8, 1.56), (8, 0.8), (8, 1.25) | +1.08 | 1.51 |
| singular | 2/3 | (8, 1.56), (8, 1.25), (8, 1.25) | +1.01 | 1.47 |
| number_to_words | 2/3 | (8, 0.1), (8, 0.4), (8, 0.1) | +0.06 | 1.69 |
| plural | 1/3 | (8, 1.25), (8, 0.6), none | +0.36 | 0.89 |
| add_two | 1/3 | (17, 0.3), (17, 0.2), (17, 0.2) | +0.11 | 1.66 |
| antonym | 0/3 | (8, 0.4), (8, 0.4), (8, 1.56) | +0.17 | 0.82 |
| present_participle | 0/3 | (8, 1.56), (17, 0.05), (6, 0.1) | +0.10 | 1.15 |
| en_fr | 0/3 | no reliable calibration point | – | – |

Qwen3-1.7B-Base, three seeds (20260907, 1, 2); wall time 13.4 / 13.2 / 11.7 min:

| task | qualified | selections (layer, ρ) | held-out Δ log p/token (seed 20260907) | gate z (seed 20260907) |
|---|---|---|---|---|
| past_tense | 3/3 | (11, 0.1), (11, 0.05), (11, 0.1) | +0.11 | 2.46 |
| present_participle | 3/3 | (11, 0.05), (11, 0.05), (11, 0.1) | +0.05 | 1.98 |
| antonym | 2/3 | (11, 1.56), (14, 0.05), (14, 0.1) | +1.16 | 1.64 |
| singular | 2/3 | (17, 0.02), (6, 0.05), (17, 0.02) | +0.06 | 4.56 |
| number_to_words | 2/3 | (11, 0.1), (8, 0.8), (8, 0.2) | +0.07 | 2.43 |
| en_fr | 2/3 | (11, 0.6), (17, 0.1), (17, 0.1) | +0.23 | 1.56 |
| add_two | 2/3 | (17, 0.3), (17, 0.3), no calibration point | – | – |
| arithmetic (n+3) | 1/3 | (11, 0.3), (14, 0.4), (14, 0.3) | +0.06 | 1.47 |
| plural | 0/3 | (6, 0.1), (6, 1.56), (6, 0.05) | +0.09 | 1.12 |
| uppercase | 0/3 | (6, 1.25), (6, 1.25), (6, 0.1) | +0.43 | 1.02 |

add_two on 1.7B is nearly solved zero-shot (log p/token −0.30), so its
calibration has little headroom.

Every held-out steering effect is statistically supported on its own
(p ≤ 0.003); the gate is binding because the real direction typically beats
60–62 of the 64 random controls (z ≈ 1.2–2.5). Which tasks clear p ≤ 0.05
therefore varies with the seed: over six runs, 31 of 55 task/seed pairs that
reached the gate passed it, and only arithmetic and uppercase (0.6B) and
past_tense and present_participle (1.7B) passed in every seed. Demonstration-variation directions are
behaviourally *inert* on 1.7B (the real direction beats them by z = 4–9),
but on 0.6B for antonym seven of eight of them steer *better* than the
antonym direction (z = −0.4).

#### Layerwise profiles against the five nulls

Every qualified profile on both models is cascade + amplification
(conversion entropy ratio 0.89–0.96, no block above 20 % of the conversion
mass, final alignment 0.01–0.15, cumulative log G 1.5–3.3); 1.7B tasks
injected at layer 11 add dimensional expansion (`d_eff` 2–3 → 15–17).

What the nulls say (per-layer z of the real direction; `core/cross_task.json`
→ `by_kind`, aggregated in `results/aggregate_*.json`):

- **Isotropic/orthogonal null.** Cumulative `log G` and `N_l^unc` of the real
  direction are *above* this null, consistently over seeds, for
  uppercase-0.6B (cum log G z 5.8 ± 0.9; `N_l^unc` z 2.5 ± 0.6),
  add_two (0.6B 4.3; 1.7B 8.9 ± 2.6), and on 1.7B for past_tense
  (5.4 ± 2.2; `N_l^unc` 1.9 ± 0.7), present_participle (5.6 ± 0.7;
  `N_l^unc` 2.5 ± 0.7, p ≤ 0.05 at every downstream block in seed
  20260907), singular (5.0 ± 1.2) and en_fr (5.0 ± 0.3). For the other
  qualified pairs they are inside or below it. Alignment `A_l` is above
  this null for most 0.6B tasks (z 1.4–1.7, p ≤ 0.05 at 40–80 % of layers).
- **Covariance-matched null.** The same quantities fall to z ≈ 0
  (1.7B: past_tense −0.1 ± 0.3, present_participle −0.3 ± 0.2, singular
  +0.1 ± 0.2, en_fr −0.1 ± 0.2; 0.6B uppercase 2.9 ± 0.6 and 1.7B add_two
  3.2 ± 0.1 are the only residual excesses). The alignment excess disappears
  too (z −1.8 to +0.9). `d_eff` of the real direction sits *between* the
  covariance null (lower) and the isotropic null (higher).
- **Other-task null.** Same picture: cum log G z −1.4 to +1.4, `N_l^unc` z
  −1.0 to +0.9, alignment z −1.4 to +0.8.
- **Demonstration-variation null.** Noisy with n = 8; no consistent excess.

Conclusion for the primary question: the amplification and new-subspace
profile of a control direction is reproduced by *any in-distribution
direction of the same norm at the same layer* (a covariance-matched random
direction, or another task's direction). The excess over isotropic random
directions reported in iteration 1 (and in the `opus` branch's summary) is
explained by the isotropic null being out of distribution, not by the
task content of the direction. The behavioural effect *is* direction-specific
(z 1.2–4.6 vs random, and other-task directions steer less), so the
task-specific part of the control lives in something the geometric profile
does not measure — which motivates direction-specific readouts (alignment
with the task's own downstream directions, or with the target's logit
gradient) as the next step.

Other observations:

- `d_eff` no longer rises monotonically with depth on 0.6B at n = 192
  (e.g. number_to_words 21 → 12): with 64 examples the early-layer estimate
  was bounded by n. Expansion is robust on 1.7B (2–3 → 15–17).
- Noise floor: centered variance at the intervention layer is 10⁻³–10⁻⁴ of
  the next layer's on every run; `max |δ|` below the intervention layer is 0.
- Wall time per run is dominated by the 113 control profiles; capping the
  BLAS thread pool (`DIRECTIONS_BLAS_THREADS`, default 4) is what makes them
  cheap — 70 s → 0.23 s per profile.

## Not yet run / known limitations

- No bit-identity rerun of the iteration-2 configs yet (iteration 1 was
  verified twice per model; nothing in the changed code touches determinism).
- Exploratory stages (strength robustness, block ablation) run only for
  qualified tasks.
- Not started: direction-specific readouts (cross-layer alignment with the
  task's own extracted directions, gradient alignment), a selection rule
  based on excess over the random spread, canonical function-vector
  extraction, a second model family.

## Next commands

```bash
uv run pytest                                                           # 44 tests
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 3  # another replicate (~9 min)
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --seed 3  # (~14 min)
uv run directions aggregate results/pilot2_qwen3_0.6b_seed* --out results/aggregate_qwen3_0.6b.json
uv run directions compare results/<run_a> results/<run_b>                # reproducibility diff
```

Suggested next steps, in order:

1. Add direction-specific readouts to the layerwise stage: cos(δ_l(x),
   v_{t,l}) against the task's own direction extracted at each later read
   point (already saved in `directions.npz` as `all_layers`), and cos with
   the per-example gradient of log p(target); compare against the same
   five nulls. This is where the behavioural specificity must show up if it
   is geometric at all.
2. Replace the fixed p ≤ 0.05 gate on the random null by a preregistered
   excess-over-null rule (e.g. z ≥ 2 on both pools) and record it in
   `docs/DECISIONS.md`; the current gate sits on the boundary of the
   random spread and flips with the seed.
3. Drop or fix tasks the models do zero-shot (add_two on 1.7B) and tasks
   whose direction never calibrates (en_fr on 0.6B).
