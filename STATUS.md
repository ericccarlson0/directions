# Status

Last updated: 2026-09-11.

## Implemented (all exercised by `uv run pytest`, 49 tests)

`src/directions/`:

- `model.py` — Hugging Face model loading in bf16 with metadata; forward
  pre/post hooks capturing `resid[0..L]` at the final query token (float32,
  gathered inside the hook); interventions at any read point with a shared or
  per-example vector; teacher-forced scoring (log p per token, log p sum,
  exact match, first-token margin) verified against a native forward pass;
  a gradient pass (`gradients`) returning ∂ log p(target)/∂ resid[l] at the
  query token for every read point, verified against finite differences
  (D17). A tiny random Qwen3-architecture model with a character tokenizer
  provides the no-download integration path.
- `tasks.py`, `data.py`, `data_extra.py`, `prompts.py` — twelve deterministic
  task builders, ten in the pilot configs (antonym, plural, past tense,
  `n → n+3`, present participle, singular, uppercase, number→words, and the
  composite `last_antonym` and `arithmetic_words`, D19; `en_fr`, `add_two`
  and `alphabetically_first` remain available), all with ≥ 320 usable
  items, token-length item filter with a per-task override and rejection
  logging, three disjoint pools, few-shot / zero-shot prompts, positive vs
  deranged demonstration pairs.
- `extraction.py`, `geometry.py` — uncentered PC1 of paired differences at
  every candidate layer, per-seed explained variance and `cos(PC1, mean)`,
  min-pairwise-|cos| stability, pooled direction; demonstration-variation
  null directions; isotropic, orthogonal and residual-covariance-matched
  random controls; Gram-trick spectra.
- `calibration.py` — layer × strength sweep with paired bootstrap, minimum
  improvement, 16-control matched random screen (paired excess test, D18)
  and automatic grid extension.
- `layerwise.py`, `stats.py` — `S_l`, `log G_l`, `C_l`, `A_l` (median +
  bootstrap CI), centered/uncentered `d_eff`, `d90`, centered variance,
  `N_l` and `N_l^unc`, noise-floor diagnostic; the direction-specific
  readouts `T_l = cos(δ_l, v_{t,l})` and `Γ_l = cos(δ_l, ∇ log p)` (D17);
  per-layer z / empirical p against the primary (isotropic + orthogonal)
  null and against each structured null (covariance, other-task,
  demonstration-variation) (D15); `paired_excess_test`, the hierarchical
  paired bootstrap behind the qualification gate (D18).
- `analysis.py`, `aggregate.py` — mode-discriminating quantities per task,
  rule-based labels, per-kind z summaries; multi-seed aggregation (D16).
- `ablation.py` — exploratory necessity/sufficiency of high-conversion blocks.
- `function_vector.py`, `model.py` head hooks — the canonical function vector
  (D21): per-head output capture and per-example head patching at the
  query token, average indirect effects on deranged-label prompts, a
  universal or per-task head ranking, the vector as the sum of the
  selected heads' mean outputs through `o_proj`, per-seed stability;
  `extraction.control` selects it as the control (PC1 stays the readout
  direction and is reported beside it).
- `figures.py`, `pipeline.py` (two phases, D15), `runinfo.py`, `cli.py`
  (`validate | pilot | check | compare | aggregate`, `--seed`).

Methodological choices are recorded in `docs/DECISIONS.md` (D1–D21).
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

### Iteration 2, models 3 and 4 (Qwen3-4B / 8B; same protocol, configs `pilot_qwen3_4b.yaml`, `pilot_qwen3_8b.yaml`)

The two larger Qwen3 base models were run with the iteration-2 protocol
unchanged (configs identical to the 0.6B/1.7B ones except `model.name`;
enforced by `tests/test_configs.py`). Both have 36 layers, so the candidate
layers are 7/11/14/18/22.

```
for seed in 20260907 1 2; do
  uv run directions pilot --config configs/pilot_qwen3_4b.yaml --seed $seed --run-id pilot2_qwen3_4b_seed$seed
  uv run directions pilot --config configs/pilot_qwen3_8b.yaml --seed $seed --run-id pilot2_qwen3_8b_seed$seed
done
uv run directions aggregate results/pilot2_qwen3_4b_seed* --out results/aggregate_qwen3_4b.json
uv run directions aggregate results/pilot2_qwen3_8b_seed* --out results/aggregate_qwen3_8b.json
```

Code at commit `fe052ed` plus the two configs; torch 2.14, transformers
5.16.1, RTX 4090. Model revisions `906bfd4b` (4B), `49e3418f` (8B).

#### Qwen3-4B-Base (d = 2560), three seeds; wall time 21.2 / 20.8 / 18.7 min

All ten tasks calibrate in every seed (few-shot accuracy 0.89–1.00). 12 of
30 task/seed pairs pass the 64-control gate; no task passes in all three seeds.

| task | qualified | selections (layer, ρ) | held-out Δ log p/token (seed 20260907) | gate z (seed 20260907) |
|---|---|---|---|---|
| en_fr | 2/3 | (18, 0.05), (14, 0.1), (14, 0.2) | +0.06 | 2.93 |
| singular | 2/3 | (18, 0.05), (11, 1.95), (11, 1.25) | +0.08 | 2.60 |
| number_to_words | 2/3 | (18, 0.2), (22, 0.05), (22, 0.05) | +0.08 | 1.81 |
| uppercase | 2/3 | (7, 1.56), (14, 1.25), (7, 1.56) | +0.31 | 1.84 |
| past_tense | 1/3 | (18, 0.1), (11, 1.56), (18, 0.1) | +0.09 | 1.98 |
| present_participle | 1/3 | (11, 0.5), (7, 1.95), (7, 1.95) | +0.23 | 1.17 |
| plural | 1/3 | (11, 0.4), (7, 1.56), (11, 0.4) | +0.12 | 1.38 |
| antonym | 1/3 | (11, 0.5), (11, 1.56), (11, 1.56) | +0.38 | 0.74 |
| arithmetic (n+3) | 0/3 | (11, 0.8), (7, 1.95), (11, 0.6) | +0.31 | 1.23 |
| add_two | 0/3 | (18, 0.5), (14, 0.4), (18, 0.4) | +0.15 | 1.03 |

Observations:

- As on the smaller models every held-out steering effect is significant on
  its own (p = 0.0005) while the real direction beats the random controls
  only by z ≈ 0.7–3 (number_to_words in seeds 1 and 2: z 5.2–5.6). The gate
  is on the boundary of the random spread; qualification flips with the seed.
- Behavioural specificity is weaker than on 1.7B: against the covariance-
  matched and other-task directions the real direction's steering z is
  0.5–2.7 and −0.2–2.6 respectively; demonstration-variation directions
  steer nearly as well (z −0.5–3.5, typically 1–2).
- Calibration often selects a *strong* injection (ρ ≥ 1.25, α 44–77) at an
  early layer (7 or 11), where the 1.7B selections were mostly ρ ≤ 0.1.
- Every qualified profile is cascade + amplification (cum log G 1.1–2.0;
  final alignment 0.02–0.08); 9 of 12 also show dimensional expansion
  (`d_eff` 3–5 → 15–18 when injected at layer 11/18). number_to_words at
  layer 22 contracts (26 → 9.5) and has a variance ratio at the intervention
  layer of 1.6 × 10⁻² (noise floor; elsewhere 10⁻⁴–3 × 10⁻³).
- Nulls, aggregated over qualified seeds (`results/aggregate_qwen3_4b.json`):
  cum log G is above the isotropic null only for number_to_words
  (z 5.9 ± 0.2), en_fr (2.9 ± 1.9) and past_tense (3.0), and against the
  covariance-matched null these fall to +1.3 ± 0.4, −0.7 ± 0.5 and −1.6;
  the other-task null gives 0.0 ± 0.2, +1.6 ± 0.1, +0.5. Every other
  qualified task is inside or *below* all nulls (antonym and uppercase cum
  log G z −2). `N_l^unc` z is −1 to +1 against isotropic and −2.5 to −0.2
  against covariance-matched. Alignment z is −0.8 to +1.3 against isotropic
  and −2.5 to −0.3 against covariance-matched. The 0.6B/1.7B conclusion
  therefore holds on 4B: the geometric profile of the task direction is
  reproduced by any in-distribution direction of the same norm at the same
  layer, and the real direction's `N_l^unc` and alignment are if anything
  *lower* than covariance-matched random directions'.

#### Qwen3-8B-Base (d = 4096), three seeds; wall time 26.2 / 25.5 / 25.2 min

The 8B weights (16.4 GB) do not fit in the `/workspace` volume's ~40 GB quota
next to the existing caches; they were downloaded to the local disk with
`HF_HUB_CACHE=/root/hf-cache` (ephemeral; the model revision is in
`metadata.json`). GPU memory in use: 18.8 GB of 24 GB.

Nine tasks calibrate in every seed; add_two never finds a calibration point
(zero-shot log p/token −0.79, few-shot −0.003: no headroom). 14 of 27
task/seed pairs pass the gate; arithmetic, number_to_words and singular pass
in all three seeds.

| task | qualified | selections (layer, ρ) | held-out Δ log p/token (seed 20260907) | gate z (seed 20260907) |
|---|---|---|---|---|
| arithmetic (n+3) | 3/3 | (18, 0.2), (18, 0.2), (14, 1.25) | +0.07 | 2.67 |
| number_to_words | 3/3 | (18, 0.3), (18, 0.8), (18, 1.0) | +0.14 | 2.22 |
| singular | 3/3 | (7, 0.1), (7, 0.05), (7, 0.02) | +0.15 | 2.44 |
| en_fr | 2/3 | (11, 1.0), (11, 0.6), (11, 0.3) | +0.97 | 2.21 |
| antonym | 1/3 | (7, 1.25), (7, 1.56), (7, 1.25) | +1.45 | 1.97 |
| past_tense | 1/3 | (11, 0.05), (7, 0.1), (7, 0.5) | +0.05 | 2.66 |
| present_participle | 1/3 | (7, 0.05), (7, 0.1), (7, 0.6) | +0.04 | 1.44 |
| plural | 0/3 | (7, 1.0), (18, 0.05), (11, 0.05) | +0.63 | 1.67 |
| uppercase | 0/3 | (11, 0.1), (14, 1.0), (14, 1.0) | +0.03 | 1.17 |
| add_two | 0/3 | no calibration point | – | – |

Observations:

- Behavioural specificity vs covariance-matched directions: z 0.7–3.5,
  highest and most consistent for singular (3.0, 3.5, 3.5); vs other-task
  directions −0.5–4.8 (arithmetic and number_to_words 2–5 in two seeds).
  Demonstration-variation directions steer with z −0.4–2.3 (one outlier 4.4).
- 14 qualified profiles: 13 cascade + amplification (conversion entropy ratio
  0.86–0.97, dominant block ≤ 21 %), one (singular, seed 2) *delayed
  activation* + amplification. Only 5 of 14 show dimensional expansion
  (`d_eff` 4 → 8–12 for layer-7/11 injections); arithmetic and en_fr do not
  expand (ratio 1.06, 1.24), and singular contracts (17–34 → 9–11, ratio
  0.43, injected at layer 7 where the raw `d_eff` estimate is high).
- Noise floor: variance ratio at the intervention layer 10⁻⁴–8 × 10⁻³
  (largest for the layer-7 singular injections, up to 8 × 10⁻³).
- Nulls (`results/aggregate_qwen3_8b.json`), qualified seeds only:
  - arithmetic (n = 3): cum log G z iso −0.2 ± 0.1, cov −1.4 ± 0.5, other
    −0.2 ± 0.7; `N_l^unc` z −0.1 / −0.7 / −1.2; alignment z −1.3 / −2.2 / −0.7.
  - number_to_words (n = 3): cum log G z −0.9 / −1.8 / −0.2; `N_l^unc`
    +0.1 / −0.3 / +0.2; alignment −1.5 / −2.1 / −0.9.
  - en_fr (n = 2), antonym, past_tense, present_participle (n = 1): inside
    every null (|z| ≤ 1.9) except alignment vs isotropic for antonym (+3.9)
    and present_participle (+4.3), which falls to +0.5 / +1.3 vs covariance.
  - **singular (n = 3) is the one task where an excess survives the
    structured nulls**: cum log G 3.56 ± 0.20 (the largest of any
    model/task), z iso +3.5 ± 1.1, other-task +3.0 ± 1.4, covariance
    +1.6 ± 0.6, demo-variation +2.7 ± 1.6; alignment z iso +4.6 ± 0.7,
    covariance +1.2 ± 0.5, other-task −0.2 ± 0.1. `N_l^unc` is at the null
    (+1.1 / −0.1 / +0.7). So singular-8B amplifies more than any other
    in-distribution direction injected at layer 7 with the same norm, but
    does not open a new subspace, and its alignment excess is not
    task-specific (other tasks' directions align equally well).

Across the four models (0.6B, 1.7B, 4B, 8B; 12 runs), the iteration-2
conclusion stands: the isotropic-null excess is an out-of-distribution
artefact, and against covariance-matched or other-task directions the
geometric profile of the task direction is unremarkable, with singular-8B
(cum log G) and the two zero-shot-solved arithmetic tasks on the small models
as the only residuals. The gate remains seed-dependent (12/30 pairs pass on
4B, 14/27 on 8B), and larger models increasingly select strong (ρ ≥ 1,
α 40–100) early-layer injections for the lexical tasks.

### Iteration 3 (current protocol; commit `8928088`; D17–D19)

Changes from iteration 2, each in its own commit: direction-specific
readouts `T_l = cos(δ_l, v_{t,l})` and `Γ_l = cos(δ_l, ∇ log p)` (D17); the
random-control gate and calibration screen are a paired hierarchical
bootstrap of the real direction's excess over the *mean* matched control,
now including the covariance-matched pool (96 gate controls, p ≤ 0.05;
D18); `add_two` and `en_fr` replaced by the composite `last_antonym` and
`arithmetic_words` (D19). Ten tasks per model as before; everything else
unchanged (192 held-out examples, 64 + 32 + 9 + 8 controls, candidate
layers by depth fraction).

```
for size in 0.6b 1.7b 4b 8b; do
  for rep in a b; do
    HF_HUB_CACHE=/root/hf-cache uv run directions pilot --config configs/pilot_qwen3_$size.yaml --seed 20260907 --run-id pilot3_qwen3_${size}_seed20260907_$rep
  done
  uv run directions compare results/pilot3_qwen3_${size}_seed20260907_a results/pilot3_qwen3_${size}_seed20260907_b
done
```

#### Reproducibility

Each model's seed-20260907 run was executed twice from the same commit;
`directions compare` reports every JSON output (72–75 files per pair,
including all per-example arrays, bootstrap p-values and the new gradient
readouts) **bit-identical** (`atol = 0`) for 0.6B (72 files), 1.7B (75), 4B (81) and 8B (84).
Wall time per run: 7.1 / 11.6 / 19.5 / 29.9 min.

#### Qualification with the paired-excess gate (seed 20260907, one run per model)

| model | few-shot pass | qualified | fails | selections (layer, ρ) of qualified tasks |
|---|---|---|---|---|
| 0.6B | 8/10 (composites fail few-shot) | 8/8 | – | all layer 8, ρ 0.1–0.8; uppercase (6, 0.1) |
| 1.7B | 9/10 (`arithmetic_words` fails few-shot) | 8/9 | present_participle (excess +0.012, p = 0.34) | layer 6 ρ 0.1–0.8; antonym (11, 1.0), arithmetic (11, 0.3), last_antonym (14, 0.5) |
| 4B | 10/10 | 9/10 | uppercase (Δ +0.006, steering p > 0.05) | layer 7 ρ 0.2–0.3 for the lexical tasks; arithmetic (7, 1.95), plural / arithmetic_words (7, 1.25), last_antonym (7, 1.56), antonym (11, 0.1), number_to_words (18, 0.2) |
| 8B | 10/10 | 10/10 | – | layer 7 ρ 0.05–0.6 (antonym, last_antonym, past_tense, plural, present_participle, singular); layer 11 (arithmetic_words 0.8, number_to_words 0.8, uppercase 0.1); arithmetic (14, 0.5) |

Every qualified pair has excess p = 0.0005 (the minimum for 2000 replicates)
against the 96 gate controls, while the iteration-2 rank statistic of the
same runs is z 0.5–1.5 (p 0.08–0.32) — i.e. the previous gate was rejecting
directions whose excess over the *average* random direction is beyond doubt.
The gate now fails only where there is no excess at all (present_participle
1.7B: +0.012; uppercase 4B: +0.006, whose steering effect itself is not
significant). The composites qualify where they pass few-shot:
`last_antonym` on 1.7B (Δ log p/token +0.24), 4B (+0.82) and 8B (+0.05),
`arithmetic_words` on 4B (+0.12) and 8B (+0.13). On 8B every task
qualifies, at small strengths (ρ ≤ 0.8, six tasks at layer 7 with
ρ 0.05–0.6).

Against the structured behavioural nulls (`by_kind_excess`): the real
direction beats the covariance-matched directions everywhere (excess
+0.05–+1.9, p ≤ 0.001), but beats the *other tasks'* directions in only
about half of the qualified pairs (0.6B: antonym, arithmetic,
number_to_words, plural, uppercase; 1.7B: antonym, arithmetic,
last_antonym, number_to_words, singular; 4B: antonym, arithmetic_words,
last_antonym; 8B: last_antonym, present_participle, singular). For
past_tense, present_participle, singular and plural on the smaller models
the other tasks' directions steer as well as the task's own (other-task
excess −0.02 to +0.09, p 0.13–0.88), and for the numeric/format tasks on
the larger models the task's own direction steers *worse* than the other
tasks' directions (4B arithmetic −0.025, p = 0.995; 8B arithmetic −0.05,
arithmetic_words −0.11, number_to_words −0.16, uppercase −0.02, p ≥ 0.97).
Demonstration-variation directions are beaten in most pairs, but not for
antonym/present_participle/singular on 0.6B, antonym/last_antonym on 1.7B,
or antonym/arithmetic_words/last_antonym/past_tense/plural on 8B.

Together with the readouts below this says that a large part of the
behavioural effect of a "task direction" is a *shared* in-context/answer
component that every task's direction carries (and that a covariance-matched
random direction does not), not content specific to the task: another
task's direction is an equally good or better steering vector for most
tasks on 8B. The task-specific residue is largest for the composites
(`last_antonym` beats the other tasks' directions on 1.7B, 4B and 8B) and
for antonym.

#### Direction-specific readouts (D17), qualified pairs

- **`T_l` (alignment with the task's own direction at `l`).** Downstream
  median 0.03–0.31, final-layer −0.38 to +0.58 (negative for several 4B/8B
  pairs). Against isotropic and covariance-matched directions it is
  *above* the null in every pair (z +2.3 to +8.3 and +1.8 to +6.9), against
  demonstration-variation directions in most (+1.0 to +5.3), but against
  the **other tasks' directions z is −2.8 to +3.0, typically ≤ +1.3**
  (8B: +0.1 to +2.0): other tasks'
  directions, injected at the same layer, end up just as aligned with this
  task's later-layer direction. The other-task directions already start at
  cos ≈ 0.3–0.5 with the injected vector at `l*` (figure
  `<task>_readouts.png`) — the per-task directions share a large common
  component, and `v_{t,l}` at later layers is largely that common
  component. (The per-layer z of `T_l` at `l*` itself is degenerate by
  construction — real = 1, orthogonal controls = 0 — and is excluded from
  the z-mean; it still shows in the `_vs_random.png` bar charts.)
- **`Γ_l` (alignment with the target-log-prob gradient).** At the
  intervention layer the injected direction is nearly orthogonal to the
  gradient: `cos(v, ∇ log p)` = −0.02 to +0.06 in every pair. Downstream
  the perturbation rotates toward the gradient (median `Γ_l` rising to
  0.01–0.12 by the late layers, largest for `arithmetic`: 0.11–0.12 on
  0.6B/1.7B/4B, 0.06 on 8B, e.g. 0.24 at layers 28–34 of 4B), and the real
  direction is only modestly more gradient-aligned than a random one of the
  same norm (z vs isotropic/covariance +0.4 to +3.1; vs other-task −1.2 to
  +3.5, above +2 for antonym-0.6B, number_to_words-0.6B, arithmetic-1.7B,
  last_antonym-1.7B, antonym-4B, last_antonym-8B). So the behavioural excess of the real
  direction is not carried by a first-order (gradient-aligned) component of
  the perturbation at any layer; whatever makes it steer better is
  nonlinear in `δ`, or lives at the target positions rather than the query
  token.
- Geometric profile unchanged: every qualified pair is cascade +
  amplification (cum log G 1.1–3.4; expansion on 1.7B for all but
  number_to_words, on 4B for antonym, arithmetic and singular, on 8B for
  antonym and uppercase), and cum log G against the covariance-matched null
  is z −3.6 to +2.3 (uppercase-0.6B +2.3 and singular-8B +2.0, the latter
  again above the other-task null too at +2.8, are the only excesses), as
  in iteration 2.

### Iteration 3b (wall-time reduction; D20; configs changed 2026-09-10)

The four pilot configs now use 16 isotropic + 16 orthogonal + 16
covariance-matched controls (48 gate controls, 32 for the primary layerwise
null; previously 32 + 32 + 32) and `batch_size: 128` (previously 32).
Everything else is unchanged. Runs go through the `run-gpu` workflow, now
requested by pushing a change to `.github/gpu-run.yaml` (the workflow's
push trigger; `workflow_dispatch` remains as a manual fallback).

```
# workflow run 34434269088 (manual dispatch), commit fbab4d4, RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot3b_qwen3_0.6b_seed20260907
```

Wall time 8.4 min for the pipeline (12.5 min for the GitHub job including the
deploy), against 41.8 min (46 min) for the same config at 32 + 32 + 32
controls and batch size 32 the day before (run 34409519341, commit `29a3e01`,
same card): the per-task measurement stage went from 106–1019 s to a flat
49–55 s. Same seed, same 8/10 tasks past the few-shot gate (the composites
fail on 0.6B as before), all 8 qualified with excess p = 0.0005 against the
48 gate controls:

| task | sel (layer, ρ) | Δ log p/token | excess vs cov / other-task / demo (p) | labels | cum log G (z iso/cov/other) | d_eff |
|---|---|---|---|---|---|---|
| antonym | (8, 0.1) | +0.053 | +0.05 (0.04) / +0.12 (0.000) / −0.05 (0.90) | cascade+amp | 2.02 (−0.8/−2.0/−0.8) | 16.5 → 8.3 |
| arithmetic | (8, 0.4) | +0.067 | +0.11 / +0.04 / +0.08 (all ≤ 0.001) | cascade+amp | 1.61 (−2.0/−2.9/−1.7) | 11.1 → 10.6 |
| number_to_words | (8, 0.1) | +0.057 | +0.05 / +0.06 / +0.07 (all ≤ 0.001) | cascade+amp | 1.91 (−1.1/−2.0/+0.3) | 21.9 → 12.6 |
| past_tense | (8, 0.2) | +0.045 | +0.08 (0.000) / −0.00 (0.50) / +0.03 (0.33) | cascade+amp | 2.09 (−0.2/−1.6/+1.4) | 13.4 → 13.0 |
| plural | (8, 0.5) | +0.182 | +0.31 / +0.19 / +0.36 (all ≤ 0.001) | cascade+amp | 2.09 (−0.2/−1.3/+2.0) | 11.0 → 14.0 |
| present_participle | (6, 0.1) | +0.057 | +0.09 (0.000) / +0.02 (0.09) / +0.05 (0.01) | cascade+amp | 3.10 (+4.7/+1.9/+0.8) | 23.0 → 11.9 |
| singular | (8, 1.0) | +0.265 | +0.76 (0.002) / +0.16 (0.06) / −0.00 (0.51) | cascade+amp+expansion | 1.68 (−1.6/−1.6/−1.5) | 11.8 → 22.5 |
| uppercase | (6, 0.1) | +0.096 | +0.11 / +0.03 / +0.12 (all ≤ 0.001) | cascade+amp | 3.20 (+5.1/+2.1/+1.0) | 15.6 → 11.3 |

Compared with the 96-control run of the previous day, six of eight
selections are identical; present_participle moved from (8, 0.4) to (6, 0.1)
and singular from ρ 0.8 to 1.0 (the calibration screen draws new random
controls, so the first reliable point can move). Every held-out effect,
every per-kind excess conclusion (the real direction beats covariance-matched
directions everywhere; other-task directions steer past_tense, singular and
present_participle as well as their own; demonstration-variation directions
beat antonym) and every profile label is the same. Cumulative log G and its
z-scores agree to within 0.1–0.3 z, as expected from the same direction
against a half-sized null.

#### Bit-identity check (required after the batch-size change)

```
# workflow run 34438110479 (push-triggered request), commit 82171db (since squashed into ac33360, whose src/ and configs are unchanged; only the RunPod path differs), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot3b_qwen3_0.6b_seed20260907_b
uv run directions compare results/remote/run1/.../pilot3b_qwen3_0.6b_seed20260907 results/remote/run2/.../pilot3b_qwen3_0.6b_seed20260907_b
```

`directions compare` on the two runs (71 JSON files and 16 `.npz` arrays
each, `atol = 0`): **every result file is identical**; the only three
differences are `metadata.json/git/{commit, dirty, source}`, which the
replicate records (`82171db`, later squashed into `ac33360`, shipped
tree) and run 1 does not (`null`,
before the run metadata learned the shipped commit). The two runs come
from commits `fbab4d4` and `82171db` (since squashed into `ac33360` together
with later RunPod-path changes), whose `src/` numerical path and configs
are identical (the intervening commits touch the RunPod workflow, scripts,
`directions/remote.py`, `runinfo.py` and docs only).
Wall time 8.4 min for both.

Three earlier attempts at the replicate failed on the RunPod path, not on
the science, and led to the changes recorded in `docs/INFRA.md`: workers
were running the *previous* deploy's artifact when a run started within
~15 min of the previous one (runs 34434569577 and 34436082660; in the
latter the same container answered every retry for 16 min), and that
container then rejected the shipped-source job inputs with a FAILED status
(run 34437755152). The job now ships `git archive` of its commit and the
worker runs that, so results no longer depend on which build a worker
holds; a stale handler is only a warning, and the runner terminates and
resubmits when a worker predates shipped source.

### Iteration 4 (D21: the canonical function vector as the control; commit `07ee329`)

```
# workflow run 34545400854 (push-triggered request), commit 07ee329, RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot4_qwen3_0.6b_seed20260907
```

Protocol as in iteration 3b (D20 counts, batch 128) with `extraction.control:
function_vector`: ten universal heads ranked by the mean recovered
first-token probability over the eight tasks that reach extraction, the
vector as the sum of their mean outputs through `o_proj`, calibrated, gated
and measured exactly like the PCA direction was. PC1 is still extracted and
is the `T_l` readout direction. A first run of the same commit family (run
34541635889, commit `2488b92`) ranked heads by the log-probability change
and is superseded: that scale let the lexical tasks dictate the head set
(D21). Wall time 34.2 min, of which 863 s were the CPU-side control
profiles of one task (plural) against 30 s for the same work on arithmetic
(see "known limitations").

Universal heads (block, head; mean effect over tasks): (15, 6; 0.134),
(18, 5; 0.047), (16, 11; 0.031), (18, 4; 0.025), (19, 6; 0.025),
(18, 1; 0.019), (19, 2; 0.013), (18, 13; 0.011), (16, 10; 0.010),
(16, 7; 0.009): all in blocks 15–19 of 28, dominated by one head. Per task
the effects of these heads are 0.16–0.33 (best head) for antonym,
number_to_words, past_tense and present_participle, 0.05–0.07 for plural,
singular and uppercase, and 0.000 for arithmetic. The arithmetic zero is a
tokenisation artefact of the metric, not a causal fact (found afterwards,
D22): its targets tokenise as `' '`, `'4'`, `'5'`, so the "first answer
token" is a lone space that the deranged prompts already predict with
probability 0.991, and the first-token probability cannot move. Iteration
4b ranks heads by the whole-target probability instead.

8 of 8 few-shot-passing tasks qualify (excess p = 0.0005 against 48 gate
controls). Every function vector has cross-seed stability ≥ 0.999 and a
natural norm of 34–49, i.e. 1.7–2.5 × the median residual norm at layer 6
(`natural_rho`); calibration selects **layer 6 for every task** (the vector
is layer-independent and the rule takes the earliest reliable layer) at
ρ 0.05–0.4, 5–50 × weaker than the canonical injection.

| task | sel (layer, ρ) | natural ρ | cos(FV, PC1@l*) | Δ log p/token | excess vs cov / other-task / demo (p) | cum log G (z iso/cov/other) | d_eff |
|---|---|---|---|---|---|---|---|
| antonym | (6, 0.05) | 1.75 | +0.03 | +0.062 | +0.05 (0.001) / +0.04 (0.009) / +0.04 (0.000) | 2.33 (−0.9/−1.8/−0.5) | 23 → 14 |
| arithmetic | (6, 0.4) | 2.06 | +0.04 | +0.064 | +0.12 (0.000) / −0.01 (0.68) / −0.01 (1.00) | 1.76 (−1.7/−2.4/−0.8) | 11 → 8 |
| number_to_words | (6, 0.1) | 2.21 | −0.04 | +0.057 | +0.05 (0.000) / −0.00 (0.54) / +0.01 (0.000) | 2.27 (0.0/−0.9/+1.2) | 21 → 12 |
| past_tense | (6, 0.05) | 2.53 | −0.01 | +0.050 | +0.06 (0.000) / +0.03 (0.003) / −0.00 (0.90) | 2.44 (−0.6/−1.6/+0.8) | 27 → 14 |
| plural | (6, 0.05) | 2.35 | −0.02 | +0.073 | +0.07 (0.000) / +0.05 (0.000) / +0.03 (0.000) | 2.35 (−1.0/−1.3/−0.3) | 19 → 14 |
| present_participle | (6, 0.1) | 2.52 | +0.04 | +0.076 | +0.11 (0.000) / +0.06 (0.013) / +0.04 (0.000) | 1.99 (−1.7/−2.2/−1.1) | 22 → 14 |
| singular | (6, 0.05) | 2.37 | −0.01 | +0.036 | +0.03 (0.001) / +0.01 (0.08) / +0.03 (0.000) | 2.31 (−1.2/−1.5/−0.7) | 28 → 16 |
| uppercase | (6, 0.1) | 2.45 | +0.03 | +0.092 | +0.10 (0.000) / +0.09 (0.000) / +0.01 (0.031) | 2.19 (−0.4/−1.4/+0.5) | 17 → 13 |

What the run says, against the iteration-3b PCA run of the same seed:

- **The function vector and PC1 are orthogonal at the injection layer**
  (|cos| ≤ 0.04 at layer 6; the largest |cos| with PC1 at any layer is
  0.3–0.4, around blocks 16–19 where the heads live), and the function
  vector's perturbation never rotates toward PC1 downstream (`T_l` downstream
  mean −0.04 to +0.09; final-layer −0.22 to +0.37; z against every null
  |z| ≤ 1.6). Two directions that share no component at layer 6 both steer
  the task, with held-out effects of the same size at the calibrated
  strengths (function vector +0.04 to +0.09, PCA +0.05 to +0.10 for six
  tasks; PCA larger for plural and singular, whose PCA strengths were
  10–20 × higher).
- **Most of the function vector is not task content.** The tasks' vectors
  have pairwise cosines 0.30–0.83 (median 0.58), and each is 0.5–0.8
  aligned with the vector built the same way from deranged-label prompts
  (the `demo_variation` null). Consistently, the task's own vector beats the
  other tasks' vectors by only +0.01 to +0.09 (significant for 5 of 8) and
  the deranged-prompt vectors by −0.01 to +0.04 (not at all for
  arithmetic, past_tense). This is the iteration-3 picture (a shared
  in-context/answer component carries most of the behavioural effect) seen
  from the head-level construction. (Arithmetic's vector steers by +0.064
  although its head effects were all zero under the first-token metric;
  see the tokenisation note above.)
- **Geometric profile.** Every task is cascade + amplification (cumulative
  log G 1.8–2.4), and `d_eff` *contracts* (17–28 → 8–16) where the PCA
  direction expanded for plural and singular. Against the nulls the
  function vector amplifies *less* than covariance-matched random
  directions (z −0.9 to −2.4) and sits inside the isotropic and other-task
  nulls; the gradient readout `Γ_l` is +0.02 to +0.05 at `l*` and
  downstream, as for the PCA direction. The propagation profile of the
  canonical direction is therefore as unremarkable as the PCA direction's
  was: nothing in the layerwise geometry distinguishes it from an
  in-distribution random direction of the same norm.

Bit-identity (required after the change of the numerical path):
the replicate `pilot4_qwen3_0.6b_seed20260907_b` (run 34548153518, commit
`7bde911`, same `src/` and configs) is **bit-identical** to the run above
in every result file (`directions compare`: 71 JSON files and 16 `.npz`
arrays, `atol = 0`; the only difference is `metadata.json/git/commit`).
Its wall time was 84 min against 34 min for the first run, entirely in the
CPU-side control profiles (70–836 s per task against 29–863 s the first
time, for identical work; see "known limitations").

### Iteration 4b (D22: the answer probability, the canonical strength, device-side profiles, a run profile; commit `9e82c9b`)

```
# workflow run 34624145202 (push-triggered request), commit 9e82c9b, RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot4b_qwen3_0.6b_seed20260907
```

Iteration 4 with the four D22 changes: heads ranked by the recovered
probability of the whole target (`aie_metric: target_probability`); the
function-vector calibration in units of the vector's own norm
(`strength_unit: natural`, grid 0.05–2.0 extended to 4.0) with the
canonical injection ρ = 1 as the reference strength (`reference_rho: 1.0`,
earliest reliable layer as before); the whole layerwise profile computed in
float64 torch on the GPU; and every stage instrumented
(`metadata.json/profile`). Wall time **11.3 min** (iteration 4: 34 and 84
min): the control profiles take 6–7 s per task, the head effects 33–62 s
per task, the largest stages are now the extraction passes. An earlier run
of the same protocol with only the spectra on the GPU (run 34615062827,
commit `53541ac`) agrees with this run within 1e-6 in every result file
(`directions compare --atol 1e-6`: only the commit and the host kernel
differ) and took 56 min, 2600 s of them in the control profiles with CPU
seconds equal to wall seconds; `scripts/profile_bench.py` on the same
worker class (run 34623168957) then measured 0.6 s per full profile in
NumPy and 0.2 s with the spectra on the GPU, on synthetic and on real
residuals alike, and the host's load average during this run was 168 on
120 CPUs: the slow phases were host contention, not the decompositions
(D22). The container's cgroup CPU statistics are not readable on the
worker (`throttled_seconds` absent), so the profiler cannot yet say
whether a quota was binding.

Universal heads: the same ten as iteration 4, in the same order but for
the 4th/5th (block, head; mean whole-target effect): (15, 6; 0.130),
(18, 5; 0.038), (16, 11; 0.027), (19, 6; 0.020), (18, 4; 0.020),
(18, 1; 0.015), (19, 2; 0.010), (18, 13; 0.008), (16, 10; 0.008),
(16, 7; 0.007). The whole-target metric equals the first-token one for the
one-token targets (antonym, plural, singular, past_tense, uppercase,
present_participle; deranged baselines identical to four decimals) and
changes arithmetic's picture only in what it measures: the deranged-prompt
baseline is now 0.007 instead of 0.991, and the largest effect of *any*
head on arithmetic is 0.0002 (the selected heads: 0.000). So the
iteration-4 zero was a saturated metric, but the answer is the same: no
single head restores arithmetic on deranged prompts in this model, and its
"function vector" is the selected heads' mean outputs on arithmetic
prompts (norm 42, in the range of the others' 34–49).

Selection: **layer 6, ρ = 1 (natural units) for every task**, i.e. the
paper's unscaled injection, α = 34–49 = 1.7–2.5 × the median residual norm
at layer 6 (iteration 4 had selected 0.05–0.4 × that norm). The reliable
range at layer 6 is 0.05–2.0 (extended to 2.5 for plural, present_participle,
singular and to 3.9 for number_to_words; arithmetic from 0.2), and the best
grid point sits at 0.75–3.9, so ρ = 1 is inside the reliable range for
all tasks and near the peak for most.

| task | Δ log p/token (held-out) | acc base → steered | gap recovered | excess vs cov / other-task / demo (p) | cum log G (z iso/cov/other) | d_eff | T_L final | ρ_S(log G) weakest / middle / strongest |
|---|---|---|---|---|---|---|---|---|
| antonym | +0.75 | 0.00 → 0.00 | 0.12 | +2.51 (0.000) / +0.14 (0.39) / −0.87 (1.00) | 2.17 (0.0/−0.1/+1.0) | 24 → 7 | +0.42 | 0.66 / 0.86 / 0.88 |
| arithmetic | +0.77 | 0.00 → 0.00 | 0.43 | +0.93 (0.000) / +0.75 (0.000) / +0.29 (0.000) | 1.52 (−1.2/−1.4/−0.9) | 14 → 6 | −0.08 | 0.75 / 0.96 / 0.79 |
| number_to_words | +0.69 | 0.00 → 0.00 | 0.21 | +1.37 (0.000) / +0.21 (0.22) / +0.57 (0.000) | 2.21 (+1.4/+1.0/+1.0) | 15 → 9 | +0.77 | 0.68 / 0.96 / 0.49 |
| past_tense | +2.15 | 0.01 → 0.07 | 0.31 | +5.05 (0.000) / +2.48 (0.000) / +0.65 (0.000) | 1.52 (−1.8/−1.1/−1.1) | 39 → 12 | −0.31 | 0.66 / 0.85 / 0.79 |
| plural | +2.46 | 0.00 → 0.04 | 0.45 | +6.24 (0.000) / +4.02 (0.000) / +1.89 (0.000) | 1.52 (−2.2/−2.6/−1.0) | 31 → 8 | +0.03 | 0.72 / 0.91 / 0.75 |
| present_participle | +1.84 | 0.00 → 0.01 | 0.31 | +6.12 (0.000) / +2.68 (0.000) / +1.08 (0.000) | 1.43 (−2.2/−2.5/−1.5) | 38 → 8 | +0.26 | 0.79 / 0.89 / 0.78 |
| singular | +3.43 | 0.01 → 0.84 | 0.71 | +6.20 (0.000) / +3.26 (0.000) / +2.60 (0.000) | 2.15 (+0.3/+0.3/+0.7) | 39 → 17 | +0.59 | 0.84 / 0.94 / 0.86 |
| uppercase | +1.96 | 0.00 → 0.00 | 0.39 | +4.08 (0.000) / +2.02 (0.000) / −0.18 (0.99) | 2.36 (+1.4/+0.8/+1.5) | 35 → 20 | +0.76 | 0.72 / 0.84 / 0.85 |

("gap recovered": the steered improvement over the zero-shot baseline as a
fraction of the few-shot-minus-zero-shot gap in log p per token; ρ_S: the
Spearman rank correlation of the selected profile's log G curve with the
profile at the weakest (0.05), middle and strongest reliable strengths.)

What the run says, against iteration 4 (same directions, 5–50 × weaker):

- **At the canonical strength the function vector does a large part of
  the task.** Held-out improvements are 0.7–3.4 nats per token (iteration
  4: 0.04–0.09), recovering 12–71 % of the few-shot-minus-zero-shot gap;
  the improvement is in probability mass rather than argmax except for
  singular, whose accuracy goes from 0.01 to 0.84. 8 of 8 tasks qualify
  (excess over the 48 gate controls p = 0.0005).
- **Task specificity appears at this strength.** The task's own vector
  beats the other tasks' vectors by +0.75 to +4.0 nats for six tasks
  (iteration 4: +0.01 to +0.09), not for antonym and number_to_words,
  and beats the deranged-prompt vectors by +0.3 to +2.6 for six; for
  antonym the deranged-prompt vector is *better* than the task's own by
  0.87 nats and for uppercase the two are equal. The vectors themselves
  are unchanged (pairwise |cos| 0.30–0.83, median 0.58; 0.5–0.8 aligned
  with the deranged-prompt vectors), so the shared component still carries
  a large part of every task's effect, and for antonym all of it.
- **The geometric profile is the same as at the weak strength.** Every
  task is cascade + amplification; cumulative log G 1.4–2.4 with |z| ≤ 2.6
  against every null (negative for five tasks against the covariance
  null: the canonical injection amplifies *less* than in-distribution
  random directions of the same norm); `d_eff` contracts (14–39 → 6–20; z
  against covariance −1.0 to +0.8) and `N_l` sits inside the nulls (z −0.3
  to −1.8). The perturbation ends aligned with PC1 at the final layer for
  five tasks (`T_L` 0.42–0.77; at the weak strength: −0.22 to +0.37), the
  gradient readout `Γ_l` is 0.02–0.05 at `l*` and 0.05–0.13 downstream.
  Against the profile at the weakest reliable strength (0.05, close to
  iteration 4's selections) the log G curve rank-correlates 0.66–0.84 and
  the `d_eff` curve −0.02 to 0.70 (five tasks below 0.15: the
  dimensionality curve of a barely-effective injection is not the
  canonical one), while the middle strength (0.3–0.75) correlates 0.84–0.96
  and 0.61–0.96 with the selected profile.

## Not yet run / known limitations

- Iteration 4b has run only on 0.6B, seed 20260907. The 1.7B/4B/8B
  configs carry the same protocol but have not been run with it. Iteration
  4 (weakest-reliable calibration, first-token ranking) is superseded by 4b
  for the head ranking and the strength.
- The worker's host was heavily loaded during every iteration-4/4b run
  (load average 168 on 120 CPUs in the last one); the device-side profile
  keeps the pipeline's host arithmetic small, but a stage that is still
  host-bound (figures, bootstraps) can slow down again. The container's
  cgroup CPU statistics are not readable there, so `throttled_seconds`
  is absent from the profile.
- Iteration 3b (D20: 16 controls per random kind, batch 128) has run only
  on 0.6B, seed 20260907 (twice, bit-identical). The 1.7B/4B/8B configs
  carry the same change but have not been run with it; 8B at batch 128 has
  not been memory-checked (18.8 GB of 24 GB at batch 32; the batch only adds
  transformer activations for ~200-token prompts, so it should fit, but
  verify `run_metadata`/log on the first run).
- Iteration 3 has one seed per model (run twice for bit-identity); seeds
  1 and 2 are not run yet, so the iteration-3 tables above have no
  cross-seed spread. The iteration-2 configs were never rerun for
  bit-identity (their code path is the iteration-3 one minus the readouts;
  iteration 1 was verified twice per model).
- Exploratory stages (strength robustness, block ablation) run only for
  qualified tasks.
- Not started: canonical function-vector extraction, a second model family,
  a readout at the *target* positions (the gradient readout is taken at the
  query token only), a task-specific version of `T_l` (the task's direction
  with the cross-task common component removed).
- 8B weights live in `/root/hf-cache` (local disk), not in `HF_HOME`: the
  `/workspace` volume has a ~40 GB hard quota and the 16.4 GB do not fit
  next to the other three models and the uv cache (pruned to 7.6 GB). A
  rerun after a container restart re-downloads them. Moving them back needs
  either a larger quota or dropping the 0.6B/1.7B weights (4.5 GB), which
  would still leave too little headroom; left as is.

## Next commands

```bash
uv run pytest                                                           # 148 tests
# GPU runs: edit .github/gpu-run.yaml (command + a new `request` label), commit, push; the run-gpu
# workflow triggers on the push (README, "Run on GPUs"). One run per push; the ci environment runs
# them one at a time. Then:
gh run list --workflow=run-gpu.yml --branch fable --limit 1 && gh run watch <RUN_ID>
gh run download <RUN_ID> --dir results/remote/<name>
# iteration 4b (D21 + D22: function-vector control at the canonical strength; the configs default to it)
# on the other models and seeds, as request-file commands (the PCA-control protocol is `extraction.control: pca` with `calibration.strength_unit: layer_norm`):
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot4b_qwen3_1.7b_seed20260907   # ADA_24
uv run directions pilot --config configs/pilot_qwen3_4b.yaml   --run-id pilot4b_qwen3_4b_seed20260907     # ADA_24
uv run directions pilot --config configs/pilot_qwen3_8b.yaml   --run-id pilot4b_qwen3_8b_seed20260907     # ADA_24 (18.8 GB at batch 32; check `profile.totals` at 128)
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 1 --run-id pilot4b_qwen3_0.6b_seed1
uv run directions aggregate results/remote/<a>/... results/remote/<b>/... --out results/aggregate4b_qwen3_0.6b.json
uv run directions compare results/<run_a> results/<run_b>                # reproducibility diff
```

Suggested next steps, in order:

1. Run seeds 1 and 2 of iteration 3 on all four models (commands above) so
   the gate stability and the readout z-scores have a cross-seed spread.
2. Separate the shared component from the task-specific one: extract the
   cross-task common direction at each layer (mean or PC1 of the per-task
   directions), inject it as a sixth control kind, and re-express `T_l`
   against the task direction with that component projected out. The
   other-task results above predict that the common direction steers most
   tasks as well as their own direction.
3. Add a gradient readout at the target positions (the perturbation is
   measured at the query token; the behavioural effect is read at the
   target tokens), and the first-order prediction `δ_l · ∇ log p` as a
   compared metric, to test whether the excess over random is first-order
   anywhere.
4. Iteration-2 conclusions that still hold and need no more runs: the
   geometric profile (cascade + amplification, expansion in the larger
   models) is not direction-specific; singular-8B is the only cumulative-gain
   excess over the structured nulls.
