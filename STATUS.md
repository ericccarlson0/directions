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

### Iteration 1 (superseded protocol)

64 held-out examples, five tasks, 16 random controls. Two tasks qualified per
model (0.6B: plural, arithmetic; 1.7B: antonym, plural); every qualified
profile was cascade + amplification; gain, `d_eff` and `N_l` sat inside the
isotropic null except plural-1.7B. Reruns were bit-identical. Superseded by
iteration 2 because the 16-control null could not be decisive (minimum
p = 1/17) and `n = 64` bounded the dimensionality metrics (D14, D15).

### Iteration 2 (current protocol; configs unchanged since)

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

The iteration-2 code plus the two configs; torch 2.14, transformers
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

### Iteration 3 (current protocol; D17–D19)

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
# workflow run 34434269088 (manual dispatch), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot3b_qwen3_0.6b_seed20260907
```

Wall time 8.4 min for the pipeline (12.5 min for the GitHub job including the
deploy), against 41.8 min (46 min) for the same config at 32 + 32 + 32
controls and batch size 32 the day before (run 34409519341,
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
# workflow run 34438110479 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot3b_qwen3_0.6b_seed20260907_b
uv run directions compare results/remote/run1/.../pilot3b_qwen3_0.6b_seed20260907 results/remote/run2/.../pilot3b_qwen3_0.6b_seed20260907_b
```

`directions compare` on the two runs (71 JSON files and 16 `.npz` arrays
each, `atol = 0`): **every result file is identical**; the only three
differences are `metadata.json/git/{commit, dirty, source}`, which the
replicate records and run 1 does not (`null`, before the run metadata
learned the shipped commit). The two runs come from different commits
whose `src/` numerical path and configs are identical (the intervening
commits touch the RunPod workflow, scripts, `directions/remote.py`,
`runinfo.py` and docs only).
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

### Iteration 4 (D21: the canonical function vector as the control)

```
# workflow run 34545400854 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot4_qwen3_0.6b_seed20260907
```

Protocol as in iteration 3b (D20 counts, batch 128) with `extraction.control:
function_vector`: ten universal heads ranked by the mean recovered
first-token probability over the eight tasks that reach extraction, the
vector as the sum of their mean outputs through `o_proj`, calibrated, gated
and measured exactly like the PCA direction was. PC1 is still extracted and
is the `T_l` readout direction. A first run of the same commit family (run
34541635889) ranked heads by the log-probability change
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
the replicate `pilot4_qwen3_0.6b_seed20260907_b` (run 34548153518, same
`src/` and configs) is **bit-identical** to the run above
in every result file (`directions compare`: 71 JSON files and 16 `.npz`
arrays, `atol = 0`; the only difference is `metadata.json/git/commit`).
Its wall time was 84 min against 34 min for the first run, entirely in the
CPU-side control profiles (70–836 s per task against 29–863 s the first
time, for identical work; see "known limitations").

### Iteration 4b (D22: the answer probability, the canonical strength, device-side profiles, a run profile)

```
# workflow run 34624145202 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
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
of the same protocol with only the spectra on the GPU (run 34615062827)
agrees with this run within 1e-6 in every result file
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

Bit-identity (required after the change of the numerical path): the
replicate `pilot4b_qwen3_0.6b_seed20260907_b` (run 34624148329, same
`src/` and configs, another RTX 4090 worker) is
**bit-identical** to the run above in every result file (`directions
compare`, `atol = 0`, and all 200 `.npz` arrays equal; the only difference
is `metadata.json/git/commit`), so the batched device-side decompositions
are deterministic across workers. Its wall time was 10.4 min (control
profiles 5–7 s per task) at a host load average of 29.

### Iteration 4b, model 2 (Qwen3-1.7B)

```
# workflow run 34636424201 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot4b_qwen3_1.7b_seed20260907
```

Same protocol and seed as the 0.6B run (config identical but for
`model.name`; 28 layers, d = 2048, candidate layers 6/8/11/14/17). Wall
time 26.9 min (control profiles 10–14 s per task, head effects 67–118 s per
task, peak 8.7 GB reserved of 24). The in-run determinism check (D23, its
first GPU run) passed on antonym in 1.1 s: the repeated steered pass, the
gradients and the profile are bit-identical.

**9 of 9** few-shot-passing tasks qualify: last_antonym now passes the
few-shot gate (0.59; 0.29 on 0.6B), arithmetic_words still does not
(0.25). Universal heads (block, head; mean whole-target effect): (15, 6;
0.086), (18, 5; 0.023), (16, 11; 0.018), (18, 4; 0.011), (19, 6; 0.010),
(16, 0; 0.008), (19, 2; 0.007), (17, 6; 0.006), (18, 1; 0.005),
(16, 10; 0.004): eight of the ten (layer, head) indices are the 0.6B set,
again in blocks 15–19 of 28 and again dominated by (15, 6). Arithmetic's
head effects are all ≤ 0.001 here too (deranged baseline 0.006).

Selection: layer 6, ρ = 1 (natural) for every task, α = 149–198 =
2.9–3.9 × the median residual norm at layer 6 (0.6B: 1.7–2.5 ×). The
reliable range is 0.05–3.91 for every task (last_antonym from 0.5), and
the best grid point sits at the grid's ceiling (3.91 = `max_rho` 4.0 after
three extensions) for seven of nine tasks, i.e. the effect is still
growing at four times the canonical norm on this model; ρ = 1 is inside
the reliable range everywhere, as the rule requires, but is not near the
peak here.

| task | Δ log p/token | acc base → steered | gap recovered | excess vs cov / other-task / demo (p) | labels | cum log G (z iso/cov) | d_eff | T_L final | ρ_S(log G) weakest / middle / strongest |
|---|---|---|---|---|---|---|---|---|---|
| antonym | +4.57 | 0.00 → 0.56 | 0.67 | +7.67 / +3.79 / +1.91 (all 0.000) | cascade + amplification + expansion | 2.16 (−2.2/−1.4) | 9 → 21 | +0.31 | 0.71 / 0.92 / 0.73 |
| arithmetic | +0.20 | 0.00 → 0.00 | 0.13 | +0.67 / +0.59 / +0.01 (0.000/0.000/0.005) | delayed activation + amplification | 2.30 (−1.4/−0.7) | 8 → 5 | +0.41 | 0.53 / 0.95 / 0.56 |
| last_antonym | +2.58 | 0.00 → 0.06 | 0.43 | +6.30 / +2.59 / +2.08 (all 0.000) | cascade + amplification | 2.18 (−2.2/−1.3) | 15 → 8 | +0.57 | 0.96 / 0.94 / 0.51 |
| number_to_words | +1.18 | 0.00 → 0.00 | 0.29 | +1.19 / +0.44 / +0.43 (all 0.000) | delayed activation + amplification | 2.71 (+0.2/+0.5) | 9 → 5 | +0.76 | 0.59 / 0.88 / 0.67 |
| past_tense | +5.02 | 0.01 → 0.32 | 0.69 | +7.99 / +3.64 / +3.58 (all 0.000) | delayed activation + amplification | 2.32 (−1.5/−0.6) | 10 → 10 | +0.52 | 0.77 / 0.93 / 0.52 |
| plural | +2.71 | 0.01 → 0.18 | 0.48 | +7.32 / +2.70 / +1.80 (all 0.000) | delayed activation + amplification | 2.30 (−1.2/−1.1) | 12 → 13 | +0.24 | 0.72 / 0.93 / 0.59 |
| present_participle | +3.85 | 0.00 → 0.39 | 0.63 | +7.15 / +3.43 / +3.30 (all 0.000) | cascade + amplification | 2.08 (−2.7/−1.8) | 13 → 10 | +0.47 | 0.76 / 0.95 / 0.46 |
| singular | +2.68 | 0.03 → 0.64 | 0.60 | +8.90 / +2.83 / +1.95 (all 0.000) | cascade + amplification + expansion | 1.95 (−3.6/−2.6) | 14 → 21 | +0.01 | 0.88 / 0.95 / 0.46 |
| uppercase | +4.17 | 0.00 → 0.58 | 0.85 | +6.10 / +4.07 / +2.00 (all 0.000) | delayed activation + amplification + expansion | 2.43 (−1.5/−0.6) | 11 → 20 | +0.31 | 0.74 / 0.97 / 0.57 |

Against the 0.6B run of the same protocol:

- **Behaviour.** The canonical injection recovers 43–85 % of the few-shot
  gap for the seven lexical tasks (0.6B: 12–71 %) and now changes the
  argmax: accuracy 0.00–0.03 → 0.18–0.64 for six tasks. Arithmetic
  (+0.20 nats, 13 % of a small gap) and number_to_words (29 %) are the
  weak ones, as before. Task specificity is uniform here: every task beats
  the other tasks' vectors (+0.44 to +4.07) and the deranged-prompt
  vectors (+0.01 to +3.58, all p ≤ 0.005); the antonym reversal of 0.6B
  (deranged vector better than the task's own) is gone. The vectors are
  as shared as on 0.6B (pairwise |cos| 0.28–0.89, median 0.54; 0.5–0.8
  aligned with the deranged-prompt vectors).
- **Geometry.** Cumulative log G 2.0–2.7 with z ≤ 0.5 against every null
  (negative for eight tasks against the isotropic null, −1.2 to −3.6),
  `N_l` inside the nulls (z −0.1 to −1.5 vs covariance), and `T_L` 0.24–0.76
  for eight tasks (singular 0.01): the perturbation again ends aligned
  with PC1 at the final layer. The labels split where 0.6B was uniform:
  four tasks cascade, five delayed activation (the conversion mass sits
  later in depth), and `d_eff` *expands* for antonym, singular and
  uppercase (9–14 → 20–21; on 0.6B every task contracted) while it
  contracts or holds for the other six. The expansion tasks are the three
  with the largest accuracy gains, which is consistent with the
  iteration-2 observation that expansion appears on the larger models.
  Against the weakest reliable strength (0.05) the log G curve
  rank-correlates 0.53–0.96 and `d_eff` −0.65 to 0.93; against the
  strongest (3.1–3.9) log G only 0.46–0.73: the profile shape at the grid
  ceiling differs from the canonical one more than the weak-injection
  profile does.

### Iteration 4b, model 3 (Qwen3-4B)

```
# workflow run 34641432381 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_4b.yaml --run-id pilot4b_qwen3_4b_seed20260907
```

Same protocol and seed (36 layers, d = 2560, 32 heads; candidate layers
7/11/14/18/22). Wall time 103 min, 82 % of it the head effects (1152 head
patches per task on a 4B model: 390–670 s per task); control profiles
15–21 s per task; peak allocated 11.2 GB, reserved 17.8 GB of 24. The
determinism check passed on antonym (45 s).

**10 of 10** tasks qualify: arithmetic_words passes the few-shot gate here
(0.81; 0.02 and 0.25 on the smaller models). Universal heads (block, head;
mean effect): (20, 24; 0.015), (20, 26; 0.013), (22, 30; 0.010),
(22, 10; 0.008), (20, 30; 0.008), (19, 11; 0.007), (22, 13; 0.007),
(25, 28; 0.007), (20, 25; 0.004), (20, 21; 0.004): blocks 19–25 of 36,
the same 55–70 % depth band as blocks 15–19 of 28, but the effects are
an order of magnitude smaller than on 0.6B/1.7B (the best head moves
antonym's deranged prompts from 0.02 to 0.06, against 0.06 → 0.39 on
0.6B): no single head carries much of any task on 4B, and for arithmetic
and arithmetic_words none carries anything (≤ 0.001).

Selection: layer 7 for every task; the canonical strength ρ = 1 for six
tasks, but ρ = 1.5 (last_antonym, plural, uppercase) and ρ = 2.0
(arithmetic) where ρ = 1 was not reliable (plural, uppercase: improvement
of 0.16/0.11 nats not significant; arithmetic, last_antonym: significant
but not above the 16 matched random directions). The vector is *smaller*
than the residual stream here: `natural_rho` 0.63–0.94 (0.6B: 1.7–2.5,
1.7B: 2.9–3.9), so the paper's unscaled injection is a milder relative
perturbation on this model, and the best grid point is again at the
ceiling (3.91, i.e. 2.5–3.7 × the residual norm) for nine of ten tasks.

| task | sel ρ | Δ log p/token | acc base → steered | gap recovered | excess vs cov / other-task / demo (p) | cum log G (z iso/cov) | d_eff | T_L final |
|---|---|---|---|---|---|---|---|---|
| antonym | 1.0 | +0.54 | 0.00 → 0.00 | 0.08 | +0.49 (0.009) / −0.01 (0.55) / +0.07 (0.015) | 1.80 (+0.3/−0.4) | 11 → 8 | −0.01 |
| arithmetic | 2.0 | +0.45 | 0.00 → 0.00 | 0.25 | +0.11 (0.000) / +0.04 (0.20) / +0.06 (0.000) | 1.25 (−2.3/−1.1) | 7 → 7 | −0.06 |
| arithmetic_words | 1.0 | +0.19 | 0.00 → 0.00 | 0.04 | +0.14 (0.012) / −0.09 (0.97) / +0.16 (0.000) | 1.55 (−0.5/−0.9) | 8 → 4 | +0.03 |
| last_antonym | 1.5 | +1.27 | 0.00 → 0.00 | 0.19 | +0.96 (0.000) / +0.42 (0.000) / +1.07 (0.000) | 1.53 (−0.5/−0.6) | 35 → 20 | −0.33 |
| number_to_words | 1.0 | +0.11 | 0.00 → 0.00 | 0.03 | +0.04 (0.25) / −0.21 (1.00) / +0.08 (0.000) | 1.99 (+1.2/+0.2) | 6 → 3 | +0.44 |
| past_tense | 1.0 | +0.39 | 0.01 → 0.02 | 0.06 | +0.38 (0.000) / +0.10 (0.005) / +0.02 (0.16) | 1.62 (−0.2/−0.7) | 12 → 8 | 0.00 |
| plural | 1.5 | +0.21 | 0.01 → 0.01 | 0.04 | +0.47 (0.001) / +0.06 (0.13) / −0.21 (1.00) | 1.66 (+0.6/−0.3) | 16 → 17 | +0.05 |
| present_participle | 1.0 | +0.44 | 0.00 → 0.00 | 0.07 | +0.62 (0.000) / −0.04 (0.81) / −0.04 (0.96) | 1.55 (−0.8/−1.3) | 12 → 8 | +0.06 |
| singular | 1.0 | +0.53 | 0.03 → 0.05 | 0.12 | +0.60 (0.000) / +0.33 (0.000) / +0.40 (0.000) | 1.55 (0.0/−0.6) | 11 → 8 | +0.10 |
| uppercase | 1.5 | +0.23 | 0.00 → 0.00 | 0.05 | +0.30 (0.000) / +0.07 (0.04) / +0.07 (0.000) | 1.37 (−1.3/−2.2) | 10 → 6 | +0.18 |

Against the smaller models:

- **At the canonical strength the vector barely does the task on 4B.**
  Δ log p per token +0.1 to +1.3 (1.7B: +0.2 to +5.0), 3–25 % of the
  few-shot gap, and no change of accuracy for any task. The grid shows why:
  every task's improvement keeps rising to the ceiling (plural +3.8, antonym
  +4.6 nats at 3.9 × ‖FV‖ ≈ 3.4 × the residual norm), so the strength that
  moves this model is, in units of the residual stream, about the same
  2–4 × as on 0.6B and 1.7B, while the paper's vector itself is only
  0.6–0.9 × the stream here.
- **Task specificity is mostly absent at this strength.** The task's own
  vector beats the other tasks' vectors for only four of ten tasks
  (last_antonym, past_tense, singular, uppercase; +0.07 to +0.42), and the
  deranged-prompt vectors for seven (plural and present_participle are
  *worse* than their deranged-prompt vector). This is the weak-strength
  picture of iteration 4 on 0.6B again: at 0.6–2 × the vector's norm, the
  shared component carries most of what the injection does. The vectors
  are as shared as before (pairwise |cos| 0.28–0.92, median 0.55; 0.44–0.72
  aligned with the deranged-prompt vectors).
- **Geometry.** Every task is cascade + amplification (cumulative log G
  1.25–1.99, |z| ≤ 2.3 against every null), `d_eff` contracts or holds
  (no expansion, unlike 1.7B), `N_l` inside the nulls, and `T_L` is ≈ 0
  for nine tasks (number_to_words 0.44): the perturbation of a weak
  injection does not end up aligned with PC1. Where the middle strength
  exists, the log G curve rank-correlates 0.85–0.98 with the selected
  profile, and 0.90–0.95 with the strongest (grid-ceiling) one.

### Iteration 4b, model 4 (Qwen3-8B)

```
# workflow run 34859201216 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2
uv run directions pilot --config configs/pilot_qwen3_8b.yaml --run-id pilot4b_qwen3_8b_seed20260907
```

Same protocol and seed (36 layers, d = 4096, 32 heads; candidate layers
7/11/14/18/22). Wall time 52.7 min on an H100 (head effects 77 %,
190–320 s per task; control profiles 13 s per task); peak allocated
26.6 GB, reserved 27.5 GB, so the 24 GB tier would not have held batch 128.
Four requests preceded this one (`docs/INFRA.md`): two sat in the queue
for the whole timeout because EUR-NO-1 has no 48/80 GB card (runs
34778958703, 34794257716), two failed at deploy because the data center
was not one Flash or RunPod volumes accept (runs 34858497382, 34858770757).
The determinism check passed on antonym (1.7 s).

**10 of 10** tasks qualify. Universal heads (block, head; mean effect):
(20, 26; 0.026), (20, 24; 0.018), (22, 10; 0.017), (20, 30; 0.016),
(22, 30; 0.012), (25, 28; 0.012), (20, 25; 0.010), (21, 26; 0.010),
(22, 13; 0.010), (23, 21; 0.009): blocks 20–25 of 36, seven of the ten
(block, head) indices shared with 4B, effects as small as on 4B (the best
single head moves any task's deranged prompts by at most 0.08; arithmetic
and arithmetic_words by 0.000/0.001).

Selection: layer 7 for every task; ρ = 1 (natural, 0.68–1.03 × the median
residual norm) for seven tasks and ρ = 2 for arithmetic, last_antonym and
number_to_words, where the canonical injection is *not* reliable: at ρ = 1
it changes the calibration metric by −0.05, −1.25 and −0.02 nats
respectively (last_antonym is hurt by it) and only 2 × the vector's norm
improves. The best grid point is at the ceiling (3.91) for seven tasks.

| task | sel ρ | Δ log p/token | acc base → steered | gap recovered | excess vs cov / other-task / demo (p) | cum log G (z iso/cov) | d_eff | T_L final |
|---|---|---|---|---|---|---|---|---|
| antonym | 1.0 | +0.52 | 0.00 → 0.01 | 0.07 | +0.49 (0.002) / +0.16 (0.003) / +0.48 (0.000) | 2.59 (−0.5/−0.9) | 8 → 5 | +0.26 |
| arithmetic | 2.0 | +0.27 | 0.00 → 0.00 | 0.16 | +0.21 (0.000) / +0.15 (0.000) / +0.26 (0.000) | 1.65 (−1.1/−1.3) | 6 → 4 | −0.31 |
| arithmetic_words | 1.0 | +0.08 | 0.00 → 0.00 | 0.02 | +0.07 (0.017) / +0.02 (0.29) / +0.14 (0.000) | 1.63 (−3.0/−4.4) | 10 → 3 | −0.17 |
| last_antonym | 2.0 | +0.78 | 0.00 → 0.01 | 0.12 | +1.55 (0.000) / +0.09 (0.32) / +0.65 (0.000) | 2.04 (+0.9/−0.8) | 35 → 5 | +0.25 |
| number_to_words | 2.0 | +0.28 | 0.00 → 0.00 | 0.08 | +0.49 (0.000) / +0.14 (0.08) / +0.34 (0.000) | 1.59 (−3.1/−2.0) | 15 → 6 | −0.42 |
| past_tense | 1.0 | +0.58 | 0.00 → 0.02 | 0.08 | +0.88 (0.000) / +0.25 (0.001) / +0.48 (0.000) | 2.27 (−1.3/−1.9) | 10 → 10 | +0.09 |
| plural | 1.0 | +0.76 | 0.00 → 0.01 | 0.13 | +1.06 (0.000) / +0.40 (0.000) / +0.21 (0.000) | 2.53 (−0.3/−0.8) | 12 → 12 | +0.12 |
| present_participle | 1.0 | +0.38 | 0.00 → 0.00 | 0.06 | +0.85 (0.000) / +0.00 (0.49) / +0.28 (0.000) | 2.42 (+0.4/−0.8) | 8 → 12 | +0.05 |
| singular | 1.0 | +0.56 | 0.01 → 0.02 | 0.12 | +0.93 (0.000) / +0.29 (0.001) / +0.75 (0.000) | 2.47 (−0.1/−1.0) | 14 → 15 | +0.28 |
| uppercase | 1.0 | +0.57 | 0.00 → 0.00 | 0.11 | +0.44 (0.000) / +0.13 (0.000) / +0.32 (0.000) | 2.60 (+1.2/−0.7) | 10 → 9 | −0.24 |

8B repeats the 4B picture rather than the 1.7B one: at the canonical
strength the vector recovers 2–16 % of the few-shot gap with no change of
accuracy, the improvement keeps rising to the grid ceiling (+4 to +6 nats
at 3.9 × ‖FV‖), the task's own vector beats the other tasks' vectors by
small margins (+0.13 to +0.40 for seven tasks, not for arithmetic_words,
last_antonym, present_participle) and beats the deranged-prompt vectors
for all ten (+0.14 to +0.75), and the profile is cascade + amplification
inside the nulls for every task, with `d_eff` contracting for six tasks and
`T_L` between −0.42 and +0.28. Across the four models the vector's norm
relative to the residual stream at the injection layer falls from 1.7–3.9
(0.6B, 1.7B) to 0.7–1.0 (4B, 8B), and the behavioural effect of the
canonical injection falls with it, while the strength that moves each
model stays at 2–4 × the stream.

### Iteration 5, batch A (D24 damage, D25 first-order depth profile, D26 head count and head support)

```
# workflow run 34870190577 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot5_qwen3_0.6b_seed20260907
```

Iteration 4b with the three batch-A additions and nothing else changed
(`n_heads: null` with candidates 1–32, `head_support_min_restored: 0.1`;
the damage measure; the two first-order metrics). Wall time 15.1 min
(host load average 137 on 120 CPUs); the determinism check passed on
antonym. A first run of the same request without the restored-fraction
requirement (run 34868012639) was **bit-identical to the iteration-4b
run** in every array the additions do not touch (PC1 directions, head
effects, extraction: 40 arrays and blocks equal), so the regression is
clean; it also admitted arithmetic's head set on the statistical tests
alone (+0.003 on the deranged prompts, 0.4 % of its gap), which is why
D26 gained the restored fraction before this run.

- **Head count.** The pooled joint effect of the top-*k* universal heads
  on the deranged prompts rises 0.13 → 0.28 → 0.50 → 0.63 → 0.67 for
  k = 1, 2, 4, 8, 16 and stalls at 32 (0.66); every k < 16 is
  significantly below 16 (p = 0.0005), so **16 heads** are used (iteration
  4b: 10 by fiat). The six added heads are (16, 0), (16, 14), (14, 13),
  (17, 6), (19, 9), (12, 0), effects 0.007 → 0.004, so the set now spans
  blocks 12–19. The 16-head set restores 77–99 % of the
  deranged-to-positive gap for the seven lexical tasks and 0.35 % of
  arithmetic's, so **arithmetic is rejected by the head-support gate**
  and seven tasks qualify (4b: eight).
- **Damage at the canonical strength.** KL(base‖steered) at the query
  token is 0.8–5.5 nats and the most likely next token changes for 94–98 %
  of the held-out prompts (number_to_words: 0 %, although its first-token
  margin improves from −8.1 to −2.6), which is the steering working; the
  collateral KL (answer token removed) is within 0–20 % of the raw KL, so
  it does not separate the
  intended change from the rest. Against the 48 gate controls at the same
  norm the vector's KL is *lower* for past_tense, plural and
  present_participle (−0.9 to −2.1 nats, p = 1), not different for antonym
  (+0.08, p = 0.14), and higher for number_to_words, singular and
  uppercase (+0.8 to +1.7, p ≤ 0.001). Along the grid at layer 6 the KL
  is 0.01–0.1 at ρ = 0.05–0.1, 0.4–0.85 at ρ = 0.5, 0.8–3.1 at ρ = 0.75
  and 5.5–10 at ρ = 2. The neutral-prose probe (D24 amendment) is not in
  this run.
- **First-order depth profile.** The first-order predicted effect
  `δ·g` at the injection layer is already +1.8 to +5.9 nats (the realised
  held-out effect is +0.7 to +4.3) and ends at +1.3 to +6.5 at the final
  read point: the direction points along the target's gradient where it
  is injected, and the blocks downstream neither build nor destroy that
  projection on net. The per-block increment is a large negative at block
  6 (−1 to −7) followed by a large positive at block 7 (+1.6 to +5.0) for
  five of seven tasks, then noise of ±1 with a centre of mass at 24–60 % of
  the downstream depth (conversion: 68–73 %). Against the nulls the
  *level* is distinct (gradient-projection z +1.0 to +5.3 against the
  isotropic, orthogonal and covariance kinds, whose directions project to
  zero) while the *increments* are not (z −0.07 to +0.56): the blocks
  transform the function vector's perturbation toward the answer no more
  than they transform a random direction of the same norm. The
  deranged-prompt vectors are the exception (projection z 21–44 for six
  tasks, −8 for antonym): they carry a first-order effect of their own.
- **The 16-head vector is a stronger vector.** Its norm is 47–65 (ten
  heads: 34–49), 2.9–3.3 × the median residual norm at layer 6, and at
  ρ = 1 (selected for six tasks; antonym ρ = 0.5) it recovers 43–89 % of
  the few-shot gap (4b: 12–71 %) with accuracy 0.29–0.94 for four tasks
  (4b: singular only); held-out +0.7 to +4.3 nats. The profile is
  unchanged: cascade + amplification inside the nulls, `d_eff`
  contracting (17–39 → 6–21).

```
# workflow run 34870249769 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot5_qwen3_1.7b_seed20260907
```

Same protocol on 1.7B. Wall time 35.2 min (4b: 26.9; load average 109;
the plural measurement and its control forwards took 300 and 240 s
against 120–190 s for the other tasks, host contention again). PC1
directions, head effects and extraction are **bit-identical to the
iteration-4b 1.7B run** (54 arrays and blocks); the determinism check
passed on antonym.

- **Head count: the sweep hit the ceiling.** The pooled joint effect
  rises 0.32 → 0.36 → 0.51 → 0.55 → 0.60 → 0.63 for k = 1 … 32 and every
  k < 32 is significantly below 32 (p = 0.0005), so **32 heads** are used,
  the largest candidate: whether 64 would do better is not known on this
  model. Heads 11–32 have effects 0.004 → 0.001 and spread over blocks
  12–26 (the first ten are the 4b ten). The set restores 77–98 % of the
  gap for seven lexical tasks, 42 % for last_antonym and 0.34 % for
  arithmetic, which is again rejected by the head-support gate;
  arithmetic_words fails the few-shot gate (0.25) as before, so **8 of 10**
  qualify (4b: 9 of 9 few-shot-passing tasks).
- **The 32-head vector is stronger again.** Norm 240–337 (ten heads:
  149–198), 4.6–6.6 × the residual norm at layer 6; ρ = 1 selected for
  every task and reliable everywhere; held-out +3.1 to +5.9 nats, 68–92 %
  of the few-shot gap (4b: 43–85 %), accuracy 0.41–0.92 for seven tasks
  (number_to_words 0.10). Every task beats the other tasks' vectors
  (+2.5 to +5.3) and the deranged-prompt vectors (+1.9 to +4.4).
- **Damage.** KL at the query token 1.7–4.5 nats at ρ = 1, argmax changed
  for 94–99 % (number_to_words 12 %); collateral 5–20 % below the raw KL.
  The vector's KL is **below the 48 gate controls' for seven of eight
  tasks** (−0.2 to −2.9 nats, p ≥ 0.76) and not different for
  number_to_words (+0.2, p = 0.15); against the deranged-prompt vectors it
  is higher for five tasks (+0.2 to +2.5, p = 0). Along the grid the KL reaches
  1 nat at ρ = 0.5–0.75 (uppercase at 0.3) and 2.8–8.8 at ρ = 2.
- **First-order depth profile.** `δ·g` at the injection layer is +5.5 to
  +15 nats for seven tasks (the realised effect is +3.1 to +5.9) and ends
  at +3.6 to +6.9; increments z −0.8 to +0.7 against the gate kinds
  (projection level z +4.7 to +9.0). last_antonym is the one task whose
  projection is built downstream: −1.7 at layer 6, +8.9 across block 7,
  +6.6 at the end. Blocks 6–7 carry the largest increments for six tasks
  (here mostly negative, −1 to −7, after a projection that starts far above
  the realised effect: the first block after the injection removes the
  part of the perturbation the model will not act on), with the
  increment's centre of mass at 18–50 % of the downstream depth
  (conversion: 73–79 %).
- Profile: cascade or delayed activation + amplification, cumulative
  log G 1.6–2.4 with z −0.6 to −4.9 against the isotropic null (the vector
  amplifies less than random directions of its norm), `d_eff` expanding
  for five tasks (6–15 → 22–47) and contracting for present_participle.

Batch A on both models, in one line: the head-count sweep asks for more
heads than the paper's ten (16 on 0.6B, ≥ 32 on 1.7B), the resulting
vectors are 1.4–1.7 × larger and recover most of the few-shot gap at the
canonical strength, the damage at that strength is no larger than a random
direction's of the same norm for most tasks, and the first-order effect is
present at the injection layer and not built downstream except for
last_antonym.

### Iteration 5, batch B (D27 half-depth candidates with the best-layer rule, D28 common direction and decomposition, D24 neutral-prose damage)

```
# workflow run 34873923702 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot5b_qwen3_0.6b_seed20260907
```

Batch A plus: candidate layers at 20/30/40/50 % depth (6/8/11/14 of 28)
with `layer_rule: best`; the leave-one-out common direction as a control
kind (`common`) and the additive split of the vector into its common and
residual parts (`decomposition.json`); the neutral-prose damage probe.
The pipeline took 19.5 min (batch A: 15.1; the probe, the decomposition
and the extra control) once it started: this is the run whose worker
spent four hours on the environment install (`docs/INFRA.md`). The function
vectors, head effects, per-layer PC1s, head count (16) and head support
are bit-identical to batch A (40 arrays; the per-seed PC1 array differs
only in its candidate-layer set), as they must be up to the selection;
the determinism check passed on antonym. A second run of the same
request (run 34895927702, another RTX 4090, pipeline 19.0 min) is
**bit-identical** in every result file (`directions compare`: only the
recorded commit differs).

- **The best layer is the deepest allowed.** Layer 14 (50 %) for six
  tasks, layer 8 for singular; ρ = 1 everywhere. The improvement of the
  canonical injection rises with depth (at layers 6/8/11/14: antonym
  −1.7 (not reliable) / +2.4 / +2.9 / +5.0 nats; uppercase +2.4 → +4.9;
  past_tense +4.7 → +6.5; present_participle +3.9 → +4.7; plural
  +3.2/+2.7/+3.3/+4.3) and is within 0.3 nats across layers for
  number_to_words and singular, so the half-depth cap binds for five
  tasks and the layer-8 choice for singular is a toss-up. At layer 14 the
  vector is 1.06–1.48 × the median residual norm (at layer 6: 2.9–3.3;
  the stream has grown), at layer 8 for singular 3.0 ×.
- **Effects at 50 % depth.** Held-out +2.5 to +5.9 nats (layer 6 in batch
  A: +0.7 to +4.3), 74–97 % of the few-shot gap, accuracy 0.57–0.96 for
  six tasks (batch A: 0.29–0.94 for four; antonym 0.01 → 0.57,
  number_to_words stays at 0). The task's own vector beats the other
  tasks' by +1.7 to +5.2 and the deranged-prompt vectors by +0.9 to +3.7
  (all p = 0).
- **The shared direction carries part of every effect, the residual
  most of it for four tasks.** cos(FV, common of the other six) is
  0.58–0.79. Injected at the vector's norm, the common direction alone
  steers +0.4 to +2.8 nats (the `common` control; the task's vector beats
  it by +0.8 to +4.3, p = 0). Split additively, the common part gives +0.5
  to +2.4 and the residual +0.85 to +3.9, both significant against the
  gate controls for every task; the residual carries most of antonym
  (+3.9 vs +0.6), uppercase (+3.2 vs +0.5), plural, past_tense, the common
  part more of present_participle (+2.0 vs +0.85) and singular. The parts
  interact: the vector's effect exceeds the sum of the parts by +0.1 to
  +1.7 nats for six tasks. Both parts show the vector's profile shape
  (log G rank correlation 0.72–0.95 common, 0.70–0.96 residual); the
  common part is labelled cascade + amplification + expansion for every
  task, the residual for four.
- **Damage.** Query-token KL 1.4–8.2 nats at ρ = 1, above the 48 gate
  controls' for all seven tasks (+0.2 to +4.4; at layer 6 three were
  below). On neutral prose the vector moves the next-token distribution
  by 0.46–1.28 nats, against 0.34–1.9 for random directions of the same
  norm: excess +0.11, +0.17, +0.21 (p ≤ 0.03) for antonym, number_to_words
  and uppercase, none for past_tense, plural and present_participle, and
  −0.41 for singular (p = 0.98). Along the grid at the selected layer the
  neutral KL is ≤ 0.07 up to ρ = 0.3, 0.10–0.18 at 0.5, 0.25–0.53 at
  0.75, then 1.0–3.5 at 1.5 and 1.6–7 at 2: the canonical injection is
  at the knee. The common part disturbs prose less than the residual for
  four tasks (0.11–0.38 vs 0.24–0.47).
- **At 50 % depth the blocks build the first-order effect.** `δ·g` is
  +2.7 to +6.9 at the injection layer and +6.3 to +10.6 at the end, and
  the per-block increments now sit *above* the nulls (z +1.1 to +2.6
  against the gate kinds; at layer 6 in batch A: −0.1 to +0.6), with the
  centre of mass at 38–61 % of the downstream depth. The geometric profile
  is cascade + amplification with dimensional expansion for five tasks
  (`d_eff` 10–17 → 23–35; at layer 6 every task contracted), cumulative
  log G 1.3–1.9 with |z| ≤ 0.7 against the isotropic null, `T_L` −0.19 to
  +0.45.

```
# workflow run 34901506341 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot5b_qwen3_1.7b_seed20260907
```

Same protocol on 1.7B: pipeline 34.6 min
(batch A: 35.2; load average 10), determinism check passed on antonym;
32 heads again, arithmetic rejected by head support, arithmetic_words by
the few-shot gate, 8 of 10 qualify.

- **On 1.7B the layer barely matters between 20 and 50 % depth.** The
  canonical injection's improvement differs by at most 0.6 nats across
  the four candidate layers for six tasks (antonym +6.0/+5.6/+6.5/+6.4 at
  6/8/11/14; past_tense +6.3/+6.3/+5.9/+6.1; present_participle
  +5.3/+5.2/+5.4/+5.3; uppercase +5.5/+5.6/+5.5/+5.4; last_antonym +4.2 →
  +4.0; number_to_words +3.0/+3.2/+3.1/+2.9) and rises with depth only for
  plural (+3.4 → +4.3) and singular (+3.5 → +4.0). The rule therefore
  picks layer 6 for two tasks, 8 for two, 11 for two and 14 for two, all
  at ρ = 1; the vector is 4.1–6.1 × the stream at layers 6–8 and 2.0–3.0 ×
  at 11–14. Held-out +3.3 to +5.9 nats (70–95 % of the gap, accuracy
  0.41–0.99), the same as batch A's layer-6 numbers; the task's vector
  beats the other tasks' by +2.5 to +5.2 and the deranged-prompt vectors
  by +1.8 to +4.4.
- **The residual carries most of the effect for five tasks; last_antonym
  is non-additive.** cos(FV, common of the other seven) is 0.56–0.79.
  The common direction alone (the `common` control) steers +0.04
  (uppercase) to +3.1 (past_tense); the task's vector beats it by +1.5 to
  +4.7. Split additively, the common part gives +0.06 to +2.9 and the
  residual −2.3 to +5.4: the residual carries antonym (+5.4 vs +1.1),
  uppercase (+4.3 vs +0.06), number_to_words, plural and singular, the
  common part carries past_tense (+2.9 vs +2.0) and present_participle
  (+1.5 vs +0.8), and for last_antonym the residual alone *hurts* (−2.3)
  while the whole vector gives +4.2, an interaction of +5.5 nats. Both
  parts keep the vector's profile shape (log G rank correlation
  0.86–0.96 common, 0.79–0.96 residual).
- **Damage.** Query-token KL 1.4–4.6 nats. On neutral prose the vector
  moves the next-token distribution by 0.6–3.4 nats, the larger values
  where the vector is 5–6 × the stream (past_tense 3.4, uppercase 3.3 at
  layers 6–8; present_participle 2.6 at layer 11), yet *below* random
  directions of the same norm for seven of eight tasks (excess −0.2 to
  −3.7, p ≥ 0.83; singular +0.12, p = 0.08). Against the common direction
  at the same norm the vector disturbs prose more for seven tasks (+0.12
  to +2.2, p ≤ 0.001; last_antonym +0.16, p = 0.09): the shared component
  is the gentler half. Along the grid the neutral KL reaches 1 nat at
  ρ ≈ 0.75–1.2 for the deep selections and at ρ ≈ 0.6–0.9 for the shallow
  ones.
- **First-order profile.** `δ·g` −1.7 to +8.3 at the injection layer and
  +6.1 to +9.0 at the end; the increments sit inside the nulls here (z
  −0.6 to +1.0 against the gate kinds; on 0.6B at layer 14 they were
  above, +1.1 to +2.6), including for the two layer-14 tasks. Geometric
  profile: cascade or delayed activation + amplification with expansion
  for seven of eight tasks (`d_eff` 3–9 → 10–47), cumulative log G 1.5–2.3
  at z −0.2 to −4.8 against the isotropic null, `T_L` +0.02 to +0.67.

```
# workflow run 34925153861 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_4b.yaml --run-id pilot5b_qwen3_4b_seed20260907
```

Same protocol on 4B (36 layers; candidate layers 7/11/14/18). Pipeline
110.9 min on an RTX 4090 (4b: 103.2; load average 7), determinism check
passed on antonym. **8 of 10** qualify: arithmetic_words passes the
few-shot gate (0.81) and, like arithmetic, is rejected by the
head-support gate (3.2 % and 0.8 % of the gap restored).

- **Head count.** Pooled joint effect 0.04 / 0.10 / 0.16 / 0.43 / 0.56 /
  0.61 for k = 1 … 32; 32 chosen (the ceiling; the last doubling adds
  10 %). The first ten heads are the iteration-4b ten and 29 of the 32
  sit in blocks 19–25 (53–69 % depth). The 32-head vector's norm is
  1.8–2.4 × the ten-head vector's.
- **Layer 18 (50 %) for every task**, ρ = 1, the vector 0.69–1.17 × the
  median residual norm there. At ρ = 1 the improvement rises with depth
  for every task and the deepest layer wins by a wide margin (antonym
  +2.3 / +2.6 / +1.9 / +4.5 at 7/11/14/18; uppercase +1.1 → +3.9;
  number_to_words +0.5 → +2.2; past_tense +2.2 → +4.6), while each
  layer's best grid point is again the same to within 0.2 nats: as on 8B,
  the layer sets how far the canonical strength is from the ceiling.
- **Effects.** Held-out +2.3 to +4.5 nats (4b at layer 7: +0.1 to +1.3),
  61–84 % of the few-shot gap, accuracy 0.27–0.94 for seven tasks
  (number_to_words 0.01). Own vector over the other tasks' +1.3 to +3.9,
  over the deranged-prompt vectors +1.6 to +3.3.
- **Common versus residual.** cos 0.62–0.85. The common direction alone
  +0.2 to +3.1; common part +0.2 to +2.7, residual +0.85 to +3.5: the
  residual carries antonym (+3.5 vs +0.2), uppercase (+3.3 vs +0.6),
  last_antonym (+2.7 vs +0.5), plural and number_to_words, the common part
  past_tense (+2.7 vs +1.1), present_participle and singular; additivity
  gaps −0.8 to +1.1. Profile rank correlations with the vector 0.60–0.98
  (common) and 0.76–0.95 (residual).
- **Damage.** Query-token KL 0.55–2.5 nats, above the gate controls' for
  every task (+0.3 to +2.0). Neutral prose: 0.12–0.23 nats, **below** the
  random directions' for all eight tasks (excess −0.01 to −0.08,
  p ≥ 0.92); the neutral KL passes 1 nat only at ρ ≈ 2–2.5.
- **First-order and geometric profile.** `δ·g` +2.7 to +5.2 at the
  injection layer, +5.1 to +6.6 at the end, increments above the nulls for
  every task (z +1.2 to +5.2; centre of mass 37–60 %). Cascade +
  amplification + dimensional expansion for all eight tasks (`d_eff` 3–15
  → 6–49), cumulative log G 1.6–2.1 at z +0.7 to +3.7 against the
  isotropic null.

```
# workflow run 34925203393 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions pilot --config configs/pilot_qwen3_8b.yaml --run-id pilot5b_qwen3_8b_seed20260907
```

Same protocol on 8B (36 layers; candidate layers 7/11/14/18), run in
parallel with the 4B run above in a second Flash environment. Pipeline
62.6 min on an H100 (4b: 52.7; load average 21), determinism check passed
on antonym. **8 of 10** qualify: arithmetic_words now passes the few-shot
gate (0.70; 0.6B and 1.7B: 0.02 and 0.25) and, like arithmetic, is
rejected by the head-support gate (2.8 % and 1.3 % of the gap restored).

- **Head count: the ceiling binds hardest here.** The pooled joint effect
  is 0.03 / 0.06 / 0.14 / 0.25 / 0.46 / 0.60 for k = 1 … 32: it nearly
  doubles from 8 to 16 heads and grows by a third more from 16 to 32,
  with no plateau in sight, so the chosen 32 is censored by the candidate
  list. The 32 heads sit in blocks 16–27 of 36, thirty of them in blocks
  19–25 (53–69 % depth); the first ten are the iteration-4b ten.
- **The 32-head vector changes the 8B picture.** Its norm is 2.1–2.6 ×
  the ten-head vector's, and at ρ = 1 it recovers 65–94 % of the few-shot
  gap (4b at layer 7: 2–16 %), held-out +2.2 to +5.4 nats (4b: +0.08 to
  +0.78), accuracy 0.39–1.00 for seven tasks (4b: no change from 0). The
  own vector beats the other tasks' by +1.3 to +4.3 and the deranged-prompt
  vectors by +1.3 to +3.9.
- **Layer.** Layer 18 (50 %) for seven tasks, layer 7 for plural (a tie,
  +5.06 at both). At ρ = 1 the improvement rises with depth for every task
  but dips at layer 11 for all of them (antonym +4.2 / +2.9 / +4.0 / +5.3
  at 7/11/14/18; past_tense +3.4 / +3.9 / +5.6 / +5.8); the best point of
  each layer's grid is the same to within 0.1 nats (antonym +6.9 at every
  layer), so the layer decides how far ρ = 1 is from the ceiling, not the
  ceiling. At layer 18 the vector is 0.76–1.10 × the median residual norm,
  at layer 7 2.1 ×.
- **Common versus residual.** cos(FV, common of the other seven)
  0.64–0.84 (the highest of the four models). The common direction alone
  steers +0.5 to +3.7 nats; the common part +0.55 to +3.4 and the residual
  +1.3 to +4.4, the residual carrying antonym (+4.4 vs +1.2), uppercase
  (+4.1 vs +0.8), last_antonym, number_to_words, the common part
  past_tense (+3.4 vs +2.1) and present_participle; additivity gaps −2.4
  to +2.0. Both parts carry the vector's profile shape (log G rank
  correlation 0.80–0.97) except uppercase's common part (0.12).
- **Damage.** Query-token KL 0.5–2.8 nats, above the gate controls' for
  every task (+0.3 to +1.9). On neutral prose the vector moves the
  distribution by only 0.15–0.39 nats, the same as random directions of its
  norm (excess −0.03 to +0.04, p ≥ 0.09) except uppercase (+0.16,
  p = 0); the neutral KL passes 1 nat only at ρ ≈ 1.5–2.5. At ρ = 1 the
  8B injection is, on this measure, the gentlest of the four models.
- **At half depth on 8B the blocks build the effect, strongly.** `δ·g`
  +0.3 to +5.1 at the injection layer and +5.3 to +10.7 at the end, with
  the per-block increments above the nulls for every task (z +0.9 to
  +5.8 against the gate kinds; 0.6B at layer 14: +1.1 to +2.6; 1.7B:
  inside), centre of mass at 26–53 % of the downstream depth. The
  geometric profile also leaves the nulls for the first time: cumulative
  log G 1.8–2.8 with z +0.8 to +4.0 against the isotropic null (4b: −0.5
  to +1.2), cascade + amplification for six tasks, delayed activation for
  plural and uppercase, `d_eff` expanding for three tasks and contracting
  for plural (37 → 6).

Batch B on the four models, in one paragraph. With the injection
confined to the first half of the depth, the canonical strength does best
at the deepest layer allowed on 0.6B, 4B and 8B and anywhere on 1.7B;
each layer's grid reaches the same ceiling, so the layer decides how far
ρ = 1 sits from it. The head-count sweep reaches its candidate ceiling
(32) on the three larger models, still rising on 8B, and the resulting
vectors are 1.4–2.6 × the ten-head ones; with them the canonical injection
recovers 61–97 % of the few-shot gap on every model (iteration 4b on 4B
and 8B: 2–16 %). The leave-one-out common direction alone steers 0.04–3.7
nats, the task-specific residual carries most of the effect for most
tasks, and the two interact (additivity gaps of −2.4 to +5.5). At the
canonical strength the vector disturbs unrelated prose by 0.12–0.39 nats
on 4B and 8B and 0.5–3.4 on the small models, no more than a random
direction of its norm for 27 of 31 task-model pairs and more than its own
shared component. The downstream blocks build the first-order effect above
the nulls at half depth on 0.6B, 4B and 8B, not on 1.7B, and on 4B and 8B
the cumulative gain also exceeds the isotropic null for the first time.

### Iteration 6 (D29 depth of commitment; D26 amended: candidates to 64, a doubling must gain 10 %)

```
# workflow run 35036855067 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1; environment installed from the wheelhouse
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --run-id pilot6_qwen3_0.6b_seed20260907
```

Batch B plus the commitment sweep and the bounded head count. Pipeline
20.0 min (batch B: 19.5; the commitment stage 11–16 s per task, 27–39k
forward examples), determinism check passed. Three earlier requests of
this run were cancelled during their environment install: PyPI's CDN was
serving RunPod at 0.3 MB/s in two data centers (`docs/INFRA.md`), so the
locked wheels now live on each volume and the worker installs from them
in seconds.

- **Head count: 8.** Pooled joint effect 0.13 / 0.28 / 0.50 / 0.63 /
  0.67 / 0.66 / 0.67 for k = 1 … 64: the gains per doubling are 314 %,
  80 %, 27 %, then 6.7 % (8 → 16) and nothing beyond, so the rule stops
  at 8 (batch B took 16). The 8-head vector is 0.66–0.76 × the 16-head
  one and at ρ = 1 recovers 53–82 % of the few-shot gap (batch B: 74–97 %),
  held-out +1.9 to +4.6 nats (batch B +2.5 to +5.9), accuracy 0.19–0.77
  for five tasks; own vector over the other tasks' +1.3 to +4.0. Layer 14
  for five tasks, 11 for number_to_words, 8 for singular. Neutral-prose
  damage 0.22–0.54 nats, excess over the random controls ≤ +0.09 (antonym
  significant, the rest not).
- **Depth of commitment.** Removing the injected direction's component
  from the perturbation right after the injection (read point l* + 1)
  leaves −0.20 to +0.12 of the effect: the direction is the whole effect
  there. The retained share then rises smoothly with depth, stays at or
  above 50 % from 21–57 % of the downstream depth on (read points 17–22
  for the layer-14 injections; 15 for singular's layer-8 injection; the
  amended D29 summary, applied to all iteration-6 runs) and above 90 %
  from 50–93 % on, and reaches 0.93–1.00 at the last read point. Keeping only the
  direction's component mirrors it: 0.68–0.97 of the effect right after
  the injection, 50 % until 10–29 % of the downstream depth, −0.04 to
  +0.12 at the end. Against the random-direction edits the direction is
  still "needed" (a significant cost of removing it) until the last or
  second-to-last read point for every task, but that cost is 0–7 % of the
  effect there. The task's own PC1 at each depth is a different matter:
  removing its component never costs more than 16–31 % (singular's last
  read point excepted, an artefact of PC1 at the pre-norm output), and
  keeping only it retains at most 18–31 %. So the control direction is
  converted, gradually and over most of the downstream depth, into
  features that are not the task's contrast direction at that depth; the
  conversion is neither localised nor delayed.

```
# workflow run 35037075704 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --run-id pilot6_qwen3_1.7b_seed20260907
```

Same protocol on 1.7B: pipeline 32.7 min (batch B: 34.6; commitment
12–27 s per task), determinism check passed, 8 of 10 qualify.

- **Head count: 16.** Pooled effect 0.09 / 0.19 / 0.46 / 0.59 / 0.64 /
  0.67 / 0.68 for k = 1 … 64; the doublings gain 676 %, 261 %, 45 %,
  14 %, then 5.1 % and 1.2 %, so the rule stops at 16 (batch B took 32).
  The 16-head vector is 0.86–0.91 × the 32-head one and does the same:
  held-out +3.1 to +6.4 nats (batch B +3.3 to +5.9), 70–95 % of the gap,
  accuracy 0.37–0.98 for seven tasks. Layers 14, 11, 8 and 6 for two,
  one, two and three tasks: the layer still does not matter on 1.7B
  (per-layer improvements within 0.5 nats for six tasks). Neutral-prose
  damage 0.4–4.3 nats, below the random controls' for six tasks and above
  for singular (+0.15, p = 0.007).
- **Depth of commitment.** As on 0.6B, removing the direction's component
  right after the injection leaves −0.12 to +0.08 of the effect (and
  −0.50 for past_tense two read points on, where it reverses it), the
  retained share stays above 50 % from 21–55 % of the downstream depth
  on and above 90 % from 43–79 % on for six tasks (number_to_words and
  plural end at 0.88–0.89 and never stay above 90 %), and reaches
  0.88–1.00 at the end; the task's PC1 at each depth is dispensable (removal never costs
  more than 27 %, keeping it alone retains at most 39 %). The difference
  from 0.6B is how long the direction alone suffices: keeping only its
  component retains 50 % of the effect until 29–65 % of the downstream
  depth on 1.7B (0.6B: 10–29 %), with 0.93–1.03 right after the injection
  and 0.00–0.17 at the end. The direction persists longer as the carrier
  on the model whose blocks did not build the first-order effect above
  the nulls (batch B), and where the injection layer does not matter.

```
# workflow run 35041155731 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions pilot --config configs/pilot_qwen3_8b.yaml --run-id pilot6_qwen3_8b_seed20260907
```

Same protocol on 8B, run in parallel with the 4B run below: pipeline
65.0 min (batch B: 62.6; commitment 17–32 s per task), determinism check
passed, 8 of 10 qualify.

- **Head count: 32, now uncensored.** Pooled effect 0.03 / 0.06 / 0.14 /
  0.25 / 0.46 / 0.60 / 0.62 for k = 1 … 64: the doublings gain 137 %,
  126 %, 78 %, 85 %, 32 % and then 3.3 % (32 → 64), so the rule keeps
  32, the batch-B choice that was censored by the candidate list. With
  the same 32 heads every core number of batch B repeats bit for bit
  (layer 18 for seven tasks and 7 for plural, held-out +2.2 to +5.4
  nats, neutral-prose damage 0.15–0.39 nats, only uppercase above the
  random controls).
- **Depth of commitment.** Removing the direction's component right
  after the injection leaves −0.14 to +0.14 of the effect for seven
  tasks (uppercase +0.44); the retained share passes 50 % at 11–33 % of
  the downstream depth for the layer-18 injections (read points 20–24;
  plural's layer-7 injection: 59 %), 90 % at 50–79 % for seven tasks
  (uppercase ends at 0.87 and never stays above 90 %), and reaches
  0.87–0.99 at the end. Keeping only the direction's component retains
  0.88–1.04 right after the injection, 50 % until 22–39 % of the
  downstream depth (plural 55 %), 0.03–0.28 at the end. The direction is
  "needed" against the random edits until the last read point for every
  task, at a cost of 1–13 % of the effect there. The task's PC1 at each
  depth is again dispensable, with one transient: two to three blocks
  after the injection (read point 21) keeping only the PC1 component
  retains 0.40–0.61 for antonym and last_antonym and removing it costs
  up to 51 %; elsewhere removal costs at most 10–31 % and keeping it
  alone retains at most 16–42 %. Compared with the small models the
  hand-over starts earlier: half of the effect is carried by other
  features by 11–33 % of the downstream depth (0.6B: 21–57 %, 1.7B:
  21–55 %), and the direction alone suffices for 50 % until 22–39 %
  (0.6B 10–29 %, 1.7B 29–65 %), consistent with the blocks building the
  first-order effect strongly on 8B (batch B).

```
# workflow run 35038665604 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_4b.yaml --run-id pilot6_qwen3_4b_seed20260907
```

Same protocol on 4B: pipeline 116.3 min (batch B: 110.9; commitment
26–34 s per task), determinism check passed, 8 of 10 qualify.

- **Head count: 32, now uncensored.** Pooled effect 0.01 / 0.05 / 0.14 /
  0.36 / 0.53 / 0.61 / 0.64 for k = 1 … 64: the doublings gain 229 %,
  181 %, 168 %, 47 %, 16 % and then 3.6 % (32 → 64), so the rule keeps
  the batch-B 32 and, as on 8B, every core number of batch B repeats bit
  for bit (layer 18 for all eight tasks, held-out +2.3 to +4.5 nats,
  neutral-prose damage 0.12–0.23 nats, below the random controls' for
  every task).
- **Depth of commitment.** Removing the direction's component right
  after the injection leaves 0.00–0.20 of the effect (uppercase 0.27);
  the retained share passes 50 % at 17–33 % of the downstream depth
  (read points 21–24), 90 % at 56–89 %, and reaches 0.95–0.99 at the
  end. Keeping only the direction's component retains 0.86–0.92 right
  after the injection, 50 % until 22–33 % of the downstream depth,
  0.00–0.09 at the end. The direction is needed against the random edits
  until the last read point for seven tasks, at a cost of 1–5 % of the
  effect there, and until the second-to-last for uppercase (a commitment
  layer in the D29 sense at read point 36, where the cost of removal,
  1 %, is inside the random edits'; the fourth such pair of the 31, after
  antonym and uppercase on 0.6B and antonym on 1.7B). The task's PC1 at
  each depth is dispensable throughout:
  removing it costs at most 15–29 %, keeping only it retains at most
  0.23–0.41. The 8B transient (the task PC1 carrying up to 61 % three
  blocks after the injection) does not appear on 4B.

Iteration 6 on the four models, in one paragraph. The bounded head-count
rule chooses 8 / 16 / 32 / 32 heads for 0.6B / 1.7B / 4B / 8B: the last
doubling that is taken gains 27 %, 14 %, 16 % and 32 %, the first that is
refused 6.7 %, 5.1 %, 3.6 % and 3.3 %, so the 64-head candidate settles
the censoring left by batch B on the two larger models (their batch-B
results stand unchanged) and the two smaller models drop to half the
heads at a small cost in effect (0.6B: 53–82 % of the few-shot gap
instead of 74–97 %; 1.7B: within the batch-B range). The depth of
commitment is the same shape on every model. Right after the injection
the injected direction is the effect: removing its component from the
perturbation leaves −0.20 to +0.27 of the effect across the 31
task-model pairs (uppercase on 8B +0.44), keeping only it leaves
0.68–1.04. The hand-over to other features is then gradual and
monotone, without a layer at which it happens: the share that survives
removal stays at or above 50 % from 21–57 % of the downstream depth on
for 0.6B and 1.7B and from 11–33 % on for 4B and 8B (8B's plural,
injected at layer 7: 59 %), above 90 % from 43–93 % on for 28 of the 31
pairs (the other three end at 0.87–0.89), and reaches 0.87–1.00 at the
last read point, where the direction alone carries −0.04 to +0.28 (the
amended D29 summary). Against random directions of
the same norm the direction is still "needed" at the last read point
for 27 of the 31 pairs (the other four at the second-to-last), but at a
cost of 0–13 % of the effect, so the D29 commitment layer is undefined
almost everywhere because the hand-over is never quite complete rather
than because it is late. The features the effect is handed to are not
the task's own contrast direction at that depth: removing the task
PC1's component costs at most 10–51 % (the 51 % a transient on 8B three
blocks after the injection; elsewhere ≤ 31 %) and keeping only it
retains at most 16–61 %. The one model-size effect is where the
hand-over sits:
earlier and more uniform on 4B and 8B (the models whose blocks build the
first-order effect above the nulls at half depth, batch B), later and
more spread on 0.6B, and latest on 1.7B, where the direction alone
still carries half of the effect until 29–65 % of the downstream depth
and where the injection layer does not matter.

### Iteration 6, second seed (20260916; D29 amended: hand-over depths as the core summary)

One more seed of the iteration-6 protocol on each model, with the run
seed overridden on the command line (the seed drives the prompt splits,
the demo draws, the control directions and the bootstraps; the config is
otherwise unchanged). Two seeds per model are aggregated with
`directions aggregate` (now leading with the core quantities, the
commitment curves re-summarised under the amended D29).

```
# workflow run 35050574186 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 20260916 --run-id pilot6_qwen3_0.6b_seed20260916
uv run directions aggregate <run of seed 20260907> <run of seed 20260916> --out results/aggregate6_qwen3_0.6b.json
```

0.6B: pipeline 18.0 min, determinism check passed, the same 7 of 10
qualify. The second seed reproduces the first almost to the digit.

- **Head count 8 again** (pooled 0.11 / 0.27 / 0.54 / 0.63 / 0.65 /
  0.67 / 0.67; the 8 → 16 doubling gains 4.2 %, seed 1: 6.7 %).
- **Layer 14 for five tasks and 11 for number_to_words in both seeds;
  singular 8 then 6** (its per-layer improvements are within 0.3 nats).
- **Effects.** Held-out +1.5 to +4.9 nats, seed differences 0.04–0.45
  (number_to_words 1.92 vs 1.47 the largest, singular 3.93 vs 3.51);
  gap fractions within 0.03 for five tasks (number_to_words 0.58 / 0.46,
  singular 0.82 / 0.72); accuracy within 0.13. Neutral-prose damage
  0.21–0.46 nats (seed 1: 0.22–0.54), the excess over the random controls
  +0.05 for antonym (p = 0.03; seed 1 +0.09) and ≤ +0.03 otherwise.
- **Common versus residual.** cos with the common direction within 0.04
  of seed 1 for every task; the common part within 0.2 nats and the
  residual within 0.25 for six tasks. The exception is singular's
  residual part: +1.56 in seed 1, −1.68 in seed 2 (its common part +1.6
  / +1.3), so for singular the split of the effect between the two parts
  is not stable, while the whole vector's effect is.
- **Depth of commitment, the most stable quantity of all.** The hand-over
  depth at 50 % is identical in both seeds for antonym, plural,
  present_participle and uppercase (0.21, 0.43, 0.57, 0.29 of the
  downstream depth) and within one read point for the rest
  (number_to_words 0.41 / 0.35, past_tense 0.36 / 0.29, singular 0.35 /
  0.27 with a different injection layer); at 90 % identical for four
  tasks and within two read points otherwise; the share carried by the
  direction alone stays ≥ 50 % until the same read point in both seeds
  for six tasks. Retained after removal at the end 0.94–1.00 in both
  seeds. The task PC1 is again dispensable (removal costs at most 15–31 %;
  the seed-1 singular artefact at the last read point, 0.05, does not
  recur: 0.85).

```
# workflow run 35050661831 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_1.7b.yaml --seed 20260916 --run-id pilot6_qwen3_1.7b_seed20260916
uv run directions aggregate <run of seed 20260907> <run of seed 20260916> --out results/aggregate6_qwen3_1.7b.json
```

1.7B: pipeline 29.7 min, determinism check passed, the same 8 of 10
qualify. The vector and its effect reproduce; the layer does not, and
the quantities that depend on the layer move with it.

- **Head count 16 again** (pooled 0.08 / 0.17 / 0.45 / 0.59 / 0.64 /
  0.66 / 0.66; 16 → 32 gains 3.2 %, seed 1: 5.1 %).
- **The layer is seed-dependent on 1.7B**, as batch B predicted from
  the flat per-layer improvements: antonym and singular keep 14 and
  uppercase 8, but last_antonym and number_to_words move from 6 to 11,
  past_tense from 6 to 8, plural from 11 to 14 and present_participle
  from 8 to 11 (one candidate step deeper in every case).
- **Effects.** Held-out within 0.3 nats of seed 1 for six tasks
  (number_to_words 3.11 → 2.37, past_tense 6.37 → 5.61), gap fractions
  within 0.07 for six (0.78 → 0.62 and 0.88 → 0.74 for those two),
  accuracy within 0.1 for six (past_tense 0.81 → 0.48, present_participle
  0.89 → 0.69). Neutral-prose damage 0.36–2.48 nats (seed 1: 0.42–4.26):
  the four tasks whose injection moved deeper lost most of it
  (past_tense 3.83 → 1.33, present_participle 4.26 → 1.30, number_to_words
  1.80 → 0.54, plural 1.25 → 0.62), so the large neutral damage of 1.7B
  in iterations 5–6 was a property of the shallow injections at 2–5 ×
  the residual norm, not of the model. Still below the random controls'
  for six tasks; antonym and singular +0.03 and +0.07 (p = 0.17, 0.09).
- **Common versus residual.** cos within 0.09 and the common part within
  0.25 nats of seed 1 for every task; the residual within 0.5 for six
  and not for last_antonym (−0.6 → +2.3) and present_participle (+3.6 →
  +0.7), both with a moved layer.
- **Depth of commitment.** For the three tasks with the same layer the
  hand-over fractions repeat (antonym 0.21 / 0.43, singular 0.36 / 0.79,
  uppercase 0.30–0.35 / 0.50; the direction alone suffices for 50 %
  until 0.29, 0.36–0.57 and 0.50). For the five whose layer moved the
  fractions shift by up to 0.15, but the read point does not: **the 50 %
  hand-over sits at read point 17–19 in both seeds for seven of the eight
  tasks (uppercase 14–15), whatever the injection layer (6–14)**. On 1.7B
  the conversion happens at a depth of the network, blocks 17–19 of 28,
  not at a distance from the injection; the batch-B finding that the
  layer does not matter there has the same shape. (On 0.6B the one task
  whose layer moved, singular, moved with it: 8 → 15 and 6 → 12; the other
  six all inject at 14, so the model gives no test.) Retained after
  removal at the end 0.90–1.00 in both seeds; the 90 % hand-over is never
  reached for number_to_words in either seed (ends 0.88–0.90). The task
  PC1 is dispensable (removal costs at most 12–26 %, keeping only it
  retains at most 0.43).

## Not yet run / known limitations

- Iteration 4b has run once on each of the four models, seed 20260907
  (8B on an H100 in US-CA-2, the others on RTX 4090s in EUR-NO-1).
  Iteration 4 (weakest-reliable calibration, first-token ranking) is
  superseded by 4b for the head ranking and the strength. Iteration 5
  batch B and iteration 6 have run once on all four models (batch A on
  0.6B and 1.7B). With candidates up to 64 and the 10 % marginal-gain
  rule the head count is no longer censored on any model (8 / 16 / 32 /
  32; the refused doubling gains 3–7 %); the ceiling is not extended to
  128.
- Data centers: EUR-NO-1 (the volume with the 0.6B–4B cache) currently
  offers nothing above 24 GB, and only US-CA-2, US-IL-1, US-MO-2, US-NC-2,
  EU-RO-1 and EUR-NO-1 can host a run at all (`docs/INFRA.md`;
  `scripts/runpod_availability.py` shows the current stock per tier). The
  8B run created a second volume in US-CA-2. The default data center of
  the request file has not been changed.
- On 1.7B, 4B and 8B the calibration effect was still growing at the grid
  ceiling (`max_rho` 4.0 in natural units) for seven to nine of the tasks; the reference rule still selects the reliable point nearest
  ρ = 1, but the strongest-strength robustness profile is taken at the
  ceiling rather than at a peak. Raising `max_rho` costs one screened grid
  point per extension and would be a config change (D22). The pipeline
  has no damage check (perplexity, KL to the unsteered distribution) to
  say whether those strengths are still perturbations; planned after 8B.
- Full replicate runs are no longer requested (D23): every run now repeats
  one steered pass, one gradient pass and one profile and records whether
  they are bit-identical (`metadata.json/determinism_check`,
  `summary.json/deterministic`); passed on the 1.7B run (1.1 s).
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
uv run pytest                                                           # 150 tests
# GPU runs: edit .github/gpu-run.yaml (command + a new `request` label), commit, push; the run-gpu
# workflow triggers on the push (README, "Run on GPUs"). One run per push; the ci environment runs
# them one at a time. Then:
gh run list --workflow=run-gpu.yml --branch fable --limit 1 && gh run watch <RUN_ID>
gh run download <RUN_ID> --dir results/remote/<name>
# iteration 4b (D21 + D22: function-vector control at the canonical strength; the configs default to it)
# on the other models and seeds, as request-file commands (the PCA-control protocol is `extraction.control: pca` with `calibration.strength_unit: layer_norm`):
uv run directions pilot --config configs/pilot_qwen3_0.6b.yaml --seed 1 --run-id pilot4b_qwen3_0.6b_seed1
uv run directions aggregate results/remote/<a>/... results/remote/<b>/... --out results/aggregate4b_qwen3_0.6b.json
uv run directions compare results/<run_a> results/<run_b>                # diff two runs (only after a change of the numerical path, D23)
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
