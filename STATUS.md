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

```
# workflow run 35050604985 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions pilot --config configs/pilot_qwen3_8b.yaml --seed 20260916 --run-id pilot6_qwen3_8b_seed20260916
uv run directions aggregate <run of seed 20260907> <run of seed 20260916> --out results/aggregate6_qwen3_8b.json
```

8B: pipeline 65.8 min, determinism check passed, the same 8 of 10
qualify. Everything reproduces.

- **Head count 32 again** (pooled 0.02 / 0.06 / 0.13 / 0.23 / 0.46 /
  0.59 / 0.63; 32 → 64 gains 6.1 %, seed 1: 3.3 %, both under the 10 %
  rule).
- **Layer 18 for seven tasks and 7 for plural in both seeds.**
- **Effects.** Held-out within 0.17 nats of seed 1 for every task
  (+2.0 to +5.4), gap fractions within 0.04, accuracy within 0.06.
  Neutral-prose damage 0.14–0.38 nats, within 0.03 of seed 1 for every
  task; again the same as the random controls' except uppercase (+0.18,
  p = 0; seed 1 +0.16).
- **Common versus residual.** cos within 0.01, the common part within
  0.15 nats and the residual within 0.45 of seed 1 for every task.
- **Depth of commitment.** The 50 % hand-over fraction is identical in
  both seeds for six tasks and one read point apart for past_tense
  (0.28 / 0.33) and plural (0.59 / 0.55); the 90 % hand-over identical
  for five, one read point apart for two, and never reached for
  uppercase in either seed (ends at 0.87–0.88); the direction alone
  suffices for 50 % until the same read point for seven tasks (antonym
  0.28 / 0.33). Retained after removal at the end within 0.01 of seed 1
  for every task. The seed-1 transient (the task PC1 carrying up to 0.61
  three blocks after the injection for last_antonym) recurs at 0.58.

```
# workflow run 35051925394 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/pilot_qwen3_4b.yaml --seed 20260916 --run-id pilot6_qwen3_4b_seed20260916
uv run directions aggregate <run of seed 20260907> <run of seed 20260916> --out results/aggregate6_qwen3_4b.json
```

4B: pipeline 115.9 min, determinism check passed, the same 8 of 10
qualify.

- **Head count 32 again** (pooled 0.01 / 0.05 / 0.14 / 0.37 / 0.54 /
  0.61 / 0.64; 32 → 64 gains 4.8 %, seed 1: 3.6 %).
- **Layer 18 for all eight tasks in both seeds.**
- **Effects.** Held-out +1.9 to +4.1 nats, lower than seed 1 for every
  task by 0.1–0.8 (past_tense 4.23 → 3.45, present_participle 3.84 →
  3.28, the rest ≤ 0.45); gap fractions lower by 0.01–0.14 and accuracy
  by 0.03–0.19, in the same direction for every task, so the second
  seed's held-out pool is harder rather than its vector weaker (the
  vector's cosine with the common direction and its neutral-prose
  damage are unchanged). Neutral-prose damage 0.11–0.20 nats, within
  0.04 of seed 1, below the random controls' for every task in both
  seeds.
- **Common versus residual.** cos within 0.03, the common part within
  0.3 nats and the residual within 0.45 of seed 1 for every task.
- **Depth of commitment.** The 50 % hand-over fraction is identical in
  both seeds for all eight tasks (0.17–0.33); the 90 % hand-over
  identical for five and one read point apart for three; the direction
  alone suffices for 50 % until the same read point for six tasks and
  one apart for two. Retained after removal at the end within 0.02 of
  seed 1. The task PC1 is dispensable (removal costs at most 15–31 %,
  keeping only it retains at most 0.45).

Two seeds on the four models, in one paragraph. The head count is the
same in both seeds on every model (8 / 16 / 32 / 32; the refused
doubling gains 3–6 %). The injection layer repeats on 4B and 8B
(all tasks) and on 0.6B (six of seven), and moves by one candidate
step for five of eight tasks on 1.7B, where the per-layer improvements
are flat; the large neutral-prose damage of 1.7B in the first seed
went with its shallow, 2–5 × norm injections and dropped to 0.5–1.3
nats where the layer moved deeper. Held-out effects agree within 0.5
nats for 27 of 31 task-model pairs (the four exceptions 0.5–0.8),
neutral damage within 0.1 nats for every pair on 0.6B, 4B and 8B (on
1.7B within 0.4 where the layer repeated and 0.6–3 nats where it
moved), the cosine with the common direction within 0.09, and the
common/residual split within 0.5 nats except for three pairs with a
moved layer or a sign change (singular on 0.6B, last_antonym and
present_participle on 1.7B). The depth of commitment is the most
reproducible quantity measured: with the same injection layer the 50 %
hand-over fraction repeats exactly for 20 of 25 pairs and within one
read point for the other five, and on 1.7B, where the layer moves, the
hand-over sits at the same absolute read point (17–19 of 28) in both
seeds. One seed of the commitment
sweep is a stable measurement; the layer choice on 1.7B and the
common/residual split for a few tasks are not, and the held-out effect
carries a seed-to-seed spread of up to 0.8 nats that intervals must
cover.

### Iteration 7a (D30: the add-k family, scored on the changed digits)

The pilot protocol on the arithmetic tasks alone: `arithmetic` (n → n + k,
n in 0–400) and `arithmetic_words` (n → words(n + k)) for k = 1, 2, 3,
5, 10, ten labelled tasks per run. The numeric targets are scored on the
digits the operation changes (1.1–1.6 of their 3.7 tokens; the leading
space and the untouched digits no longer dilute the metric), so the
deranged-prompt baseline of the head-effect metric is a one-in-ten
guess (0.04–0.13) rather than the 0.005 of the earlier runs, and a
restored fraction means what it says. Universal heads, other-task
controls and the common direction are taken within the family.

```
# workflow run 35127046369 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/arith_qwen3_0.6b.yaml --run-id arith_qwen3_0.6b_seed20260907
```

0.6B: pipeline 6.4 min. **0 of 10 qualify.** The five word tasks fail the
few-shot gate (accuracy 0.48 for add-1 words, ≤ 0.04 for the rest). The
five numeric tasks pass it (0.93–0.99) and all fail the head-support
gate: the best single head restores 0.0008–0.009 of a 0.86–0.93 gap in
answer probability and the 32-head vector 0.5 % (add-3), 0.5 % (add-5),
0.9 % (add-2), 2.5 % (add-10) and 4.2 % (add-1), against the 10 %
threshold and the lexical tasks' 78–82 % under the same construction.
The successor task add-1 is the best of the family but is not carried
by any head either. The five vectors are nearly the same vector: pairwise
cosines 0.94–0.99, norms 55–57, and the operand explains 0.46 of the
small residual variation with successive differences uncorrelated (cos
−0.35 to +0.05): the head mean keeps the shared numeric content and
carries nothing about k. The head-count sweep chose 32 on pooled effects
of 0.003–0.015, i.e. the rule ran on noise.

```
# workflow run 35127207193 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/arith_qwen3_1.7b.yaml --run-id arith_qwen3_1.7b_seed20260907
```

1.7B: pipeline 14.5 min. **0 of 10 qualify**, but the successor task
comes closest. Add-1 words now passes the few-shot gate (0.71; the other
word tasks 0.00–0.31 and rejected), so six tasks reach the head sweep.
The 16-head vector restores 9.9 % of add-1's gap (0.18 → 0.99 in answer
probability; the threshold is 10 %), 5.7 % of add-1 words', and 0.7–1.9 %
for k = 2, 3, 5, 10; the best single head 0.011 for add-1 and
0.003–0.004 for the rest. The two universal heads of the lexical tasks
on this model, (15, 6) and (18, 4), top the add-1, add-2, add-3 and add-5
rankings, so what little the family has is the lexical tasks' task-format
signal, not an operand. The numeric vectors are less alike than on 0.6B
(pairwise cosines 0.87–0.94; add-1 words at 0.55–0.58 to them), the
operand explains 0.34 of their variation, and successive differences are
anticorrelated (−0.34 to −0.09): still no operand direction.

```
# workflow run 35127116164 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions pilot --config configs/arith_qwen3_8b.yaml --run-id arith_qwen3_8b_seed20260907
```

8B: pipeline 62 min, determinism check passed. **1 of 10 qualifies: the
successor task add-1.** Nine tasks pass the few-shot gate (add-10 words
0.39 fails), and the head sweep chose 64 heads on a pooled effect of
0.058, its top heads (22, 13), (20, 30), (20, 26) being the lexical
tasks' universal heads.

- **Add-1 has head support: the 64-head vector restores 33 % of the gap**
  (deranged 0.24 → 0.99 in the changed digit's probability; the best
  single head 0.008). It then does the whole task: layer 14, ρ = 1 at
  2.0 × the residual norm, held-out +1.47 nats on the changed digit,
  90 % of the few-shot gap, steered accuracy 1.00 (all digits) from 0.005
  zero-shot; neutral-prose damage 0.50 nats, 0.26 below the random
  controls'. Its vector beats the other operands' vectors, injected at
  the same layer and norm, by +0.86 nats (p = 0.000), although those
  vectors have cosines 0.95–0.96 with it: the small operand-specific
  residue is what does the work.
- **The parameterised operands do not:** k = 2, 3, 5, 10 restore 2.7 %,
  2.2 %, 7.0 % and 9.1 % (add-10 the closest, its last digit unchanged,
  the tens digit carrying the operation), the word tasks 2.0–4.5 %.
  Across the five numeric vectors the operand explains 0.38 of a small
  variation and successive differences are anticorrelated (−0.34 to
  −0.15); the five word vectors are one vector (cosines 0.98–0.99) at
  0.67–0.71 to the numeric ones.
- **Add-1's commitment curves are censored by the ceiling.** The steered
  log-probability sits at −0.1 on the changed digit, so any edit that
  keeps the answer near certain reads as retaining 1.0: removing the
  direction leaves −0.17 to +0.58 of the effect until read point 21
  (0.32 of the downstream depth) and 1.00 from read point 26 on, while
  keeping only the direction retains ≥ 0.73 at every read point and 1.00
  at the end. Either route alone suffices to hit the ceiling, so unlike
  the lexical tasks this sweep cannot say whether the direction is
  handed over; a lower strength (ρ = 0.5, still reliable) would be needed
  to read it.

```
# workflow run 35128391515 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/arith_qwen3_4b.yaml --run-id arith_qwen3_4b_seed20260907
```

4B: pipeline 115.5 min, determinism check passed. **1 of 10 qualifies,
add-1 again.** Nine tasks pass the few-shot gate (add-10 words 0.12
fails); 64 heads on a pooled effect of 0.043.

- **Add-1: 26 % head support** (deranged 0.21 → 1.00; best single head
  0.013), layer 7 at ρ = 1.5 (ρ = 1 was not reliable on this model,
  the reliable range is 1.5–3.9), 3.0 × the residual norm; held-out
  +1.48 nats on the changed digit, 74 % of the gap, accuracy 0.93 from
  zero; neutral damage 0.44 nats, 0.15 below the random controls'; its
  vector beats the other operands' by +0.44 nats (p = 0.000) at cosines
  0.88–0.91 to them.
- **k = 2, 3, 5, 10 restore 1.2–4.1 %**, the word tasks 2.4–5.8 %. The
  operand explains 0.38 of the numeric vectors' variation with
  anticorrelated successive differences (−0.29 to −0.12); the word
  vectors are one vector (0.96–0.99) at 0.62–0.74 to the numeric ones.
- **Commitment, again at the ceiling** (steered log-probability −0.5 on
  the changed digit): removing the direction leaves 0.30–0.50 of the
  effect for the first five read points after the layer-7 injection and
  ≥ 0.82 from read point 13 on (0.21 of the downstream depth); keeping
  only it retains ≥ 0.41 everywhere and 1.00 at the end.

The add-k family on the four models, in one paragraph. With the metric
restricted to the digits the operation changes, the head-support test
gives a clean verdict: **the successor task add-1 has a function vector
on 4B and 8B (26 % and 33 % of the gap restored by the universal heads;
74 % and 90 % of the few-shot gap recovered by the injection, accuracy
0.93 and 1.00), almost one on 1.7B (9.9 % against a 10 % threshold) and
none on 0.6B (4.2 %); no parameterised operand (k = 2, 3, 5, 10) has one
on any model (0.5–9.1 % restored, the best being add-10 on 8B, whose
last digit is unchanged and whose tens digit carries the operation), and
neither does any word task (2–6 %).** The head-mean vectors of the five
operands are nearly the same vector on every model (cosines 0.87–0.99),
their differences across k are not a consistent direction (the operand
explains 0.34–0.46 of a small variation, successive differences
anticorrelated), yet where add-1 works its own vector beats the other
operands' by 0.4–0.9 nats: the vector for "next" is a specific vector,
and the vectors for "add k" are the same numeric-format vector with
nothing about k in it. This is the boundary the literature predicted
without testing it: a fixed relation, even a numeric one, is a function
vector; an operation whose parameter is read from the demonstrations is
not carried by the mean output of any set of heads, on any of the four
models. The add-1 vectors also differ from the lexical ones in kind: they
need 2–3 × the residual norm, and at that strength their effect
saturates, so the commitment sweep reads 1.0 for both removing and
keeping the direction from mid-depth on and cannot say how the effect is
carried. Whether a rank-k operator or a learned vector can carry an
operand (the next rung) is open; the head-mean construction cannot.

### Iteration 7b (D31: the learned single vector)

The pilot protocol with `extraction.control: learned_vector`: at each
candidate layer one vector, fitted by projected Adam (100 steps, step
0.05 of the norm) with the model frozen on the extraction pool's 64
zero-shot prompts, held at the median residual norm of the layer; the
control is the first seed's fit, two more seeds measure how unique it
is; the `demo_variation` controls are vectors fitted the same way to the
targets permuted across items. `arithmetic` runs as `add_3` on the
changed digits (D30). The first request was cancelled on its first fit
lines and D31 amended (stability reported, not gated; one deterministic
fit as the control) because three seeds reached a training loss near
zero at every layer with pairwise cosines of 0.1–0.2.

```
# workflow run 35181427862 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/learned_qwen3_0.6b.yaml --run-id learned_qwen3_0.6b_seed20260907
```

0.6B: pipeline 40 min (fits 2 min per task, the permuted null 1.5 min),
determinism check passed with the new gradient path. **8 of 10 qualify**,
the same eight as the head-mean run plus add-3 in place of nothing: the
two rejections are the few-shot failures (arithmetic_words, last_antonym).

- **The fit reaches zero training loss at every candidate layer** for
  every task (from 4.7–10.7 nats per example; 0.15 for uppercase at layer
  8), within 20–30 steps of the 100; the solutions are not unique
  (stability 0.01–0.19) and are nearly orthogonal to PC1 (|cos| ≤ 0.06)
  and to the head-mean function vector (cos 0.05–0.09).
- **Held out, a single learned vector recovers the whole few-shot gap.**
  Layer 6–14 at ρ = 1 (one residual norm; the reliable range 0.05–2.0 for
  every task), held-out +3.4 to +6.6 nats and 0.81–1.03 of the gap
  (the head-mean vector: +1.9 to +4.6, 0.53–0.82), accuracy 0.63–0.99
  from 0.00–0.01 zero-shot (head mean: 0.19–0.77). Every task beats its
  permuted-target null by +1.9 to +7.5 nats (p = 0.000): the fit finds
  the task, not the answer format. The other tasks' learned vectors at
  the same layer and norm make a task 5–10 nats worse, so the excess
  over them is +5.6 to +15.7.
- **And rewrites the distribution to do it.** The query-token KL from
  the unsteered model is 15–25 nats (head mean: 0.5–2.0) with the
  argmax changed for 96–100 % of prompts and the collateral KL (answer
  token removed) 12–15 nats: the learned vector does not raise the
  answer, it replaces the next-token distribution. On neutral prose the
  damage is 0.18–0.68 nats (head mean 0.22–0.54), but unlike the head
  mean it exceeds the random controls' for every task (+0.05 to +0.56,
  p ≤ 0.02).
- **Add-3 qualifies with a learned vector**: +3.9 nats on the changed
  digit (the whole gap: log p ≈ 0 for carry and non-carry items alike),
  +2.5 over its permuted-target null (p = 0.000), exact-match accuracy
  0.58 from 0.00 (few-shot 0.96). The 42 % that fail are wrong in the
  digits the operation does not change: accuracy is 0.71 on the
  carrying items (all digits scored) and 0.53 on the rest, where the
  fit was never scored on the copied hundreds and tens. So a single
  fitted vector does carry "add 3" on 0.6B, at 2.2 nats of neutral-prose
  damage (+1.9 over the controls), and the changed-digit objective lets
  it corrupt what it is not scored on; a fit on all digits is the
  obvious follow-up.
- **Depth of commitment.** The same shape as for the head-mean vector
  and earlier: removal leaves ≥ 50 % from 0.14–0.41 of the downstream
  depth and ≥ 90 % from 0.18–0.55, the direction alone suffices for 50 %
  until 0.14–0.30, retained after removal 1.00 at the end and carried
  alone 0.01–0.14. One difference: the direction stops being needed
  against the random edits before the end for every task (needed until
  read points 15–22 of 28, a D29 commitment layer at 16–23), whereas the
  head-mean vector was needed to the last read point on 27 of 31 pairs.
  The learned vectors' effects sit at the ceiling (accuracy near 1), so
  this reads as the hand-over completing rather than as a cleaner one.

```
# workflow run 35181465402 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions pilot --config configs/learned_qwen3_8b.yaml --run-id learned_qwen3_8b_seed20260907
```

8B: pipeline 52 min (fits 1.5–2 min per task), determinism check passed.
**10 of 10 qualify**, including add-3 and arithmetic_words, which no
head-mean vector has ever carried.

- **Zero training loss at every layer for every task** (from 3.9–17.9
  nats per example), stability 0.03–0.25, |cos| with PC1 ≤ 0.07 and
  with the head-mean function vector 0.02–0.12: the learned direction is
  not the function vector's direction.
- **Held out, the whole gap, with accuracy at ceiling.** Layers 7–18 at
  ρ = 1, held-out +3.3 to +7.1 nats, 0.83–1.04 of the few-shot gap (head
  mean: 0.65–0.94), accuracy 0.81–1.00 (head mean 0.39–1.00). Every
  task beats its permuted-target null by +1.9 to +12.8 nats (p = 0.000).
- **Add-3 is carried by one vector: accuracy 0.99 from 0.005**, on
  carrying and non-carrying items alike (0.98 / 0.99), +2.7 nats over
  the permuted-target null; arithmetic_words (words of n + 3) accuracy
  0.95 from 0.00, +2.2 over its null. The demonstration-read operand
  that no set of heads' mean output carries (iteration 7a) is carried by
  a single fitted direction of one residual norm at layer 14. The
  rank-one argument does not bind here: on this model "add 3" is
  reachable by an additive shift.
- **The price is the distribution.** Query-token KL 12–30 nats with the
  argmax changed for 99–100 % of prompts (head mean 0.5–2.8); on neutral
  prose the learned vectors of antonym, past_tense, plural, singular and
  arithmetic_words cost 0.13–0.40 nats (+0.05 to +0.28 over the random
  controls, p = 0.000), number_to_words and last_antonym 0.9–1.4, and
  add-3, present_participle and uppercase 4.6, 5.9 and 9.3 nats, 4.5–9.1
  above the controls: those three make the model unusable on unrelated
  text at the strength that solves the task. The head-mean vectors sat
  at 0.15–0.39 nats, at the random controls' level.
- **Depth of commitment.** Hand-over earlier than the head mean's:
  removal leaves ≥ 50 % from 0.06–0.24 of the downstream depth and
  ≥ 90 % from 0.11–0.33 (head mean 0.11–0.33 and 0.50–0.78); the
  direction alone suffices for 50 % until 0.07–0.50, and for add-3, at
  the ceiling, to the end. The direction stops being needed against the
  random edits before the end for eight of ten tasks (commitment layers
  at read points 20–36).

```
# workflow run 35181498809 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/learned_qwen3_1.7b.yaml --run-id learned_qwen3_1.7b_seed20260907
```

1.7B: pipeline 50 min, determinism check passed. **9 of 10 qualify**
(arithmetic_words fails the few-shot gate, 0.25), one more than the
head-mean run: add-3 again.

- Zero training loss at every layer (from 3.4–12.7 nats), stability
  0.02–0.31, |cos| with PC1 ≤ 0.05, with the head-mean vector 0.03–0.12.
- **Held out: 0.81–1.01 of the gap** (head mean 0.70–0.95), accuracy
  0.72–0.99 (head mean 0.36–0.98); for antonym and uppercase the learned
  and head-mean vectors tie (+5.47 vs +5.51, +4.69 vs +4.69), the only
  ties in the four models. Every task beats its permuted-target null by
  +1.6 to +8.6 nats.
- **Add-3: accuracy 0.91 from 0.00** (0.98 on carrying items, 0.88 on
  the rest), +2.4 over its permuted null, at 1.9 nats of neutral damage.
- Query-token KL 19–28 nats; neutral-prose damage 0.10–0.60 nats for
  six tasks (+0.02 to +0.51 over the controls), 1.1 for last_antonym,
  and 3.3 and 4.5 for number_to_words and past_tense.
- Commitment: removal leaves ≥ 50 % from 0.05–0.50 of the downstream
  depth, ≥ 90 % from 0.21–0.55; the direction alone suffices for 50 %
  until 0.09–0.35 (add-3 at the ceiling, to the end); the direction
  stops being needed before the end for every task (commitment layers
  15–26 of 28).

```
# workflow run 35184285964 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions pilot --config configs/learned_qwen3_4b.yaml --run-id learned_qwen3_4b_seed20260907
```

4B: pipeline 78 min (fits 2.5–3 min per task, the null 2 min; the
head-mean run took 116), determinism check passed. **10 of 10 qualify.**

- Zero training loss at every layer (from 4.1–18.3 nats), stability
  0.04–0.33, |cos| with PC1 ≤ 0.09, with the head-mean vector
  0.04–0.13.
- **Held out: 0.75–1.03 of the gap** (head mean 0.61–0.84), accuracy
  0.76–1.00 (head mean 0.10–0.94); +1.6 to +8.0 over the permuted-target
  nulls.
- **Add-3: accuracy 0.98 from 0.00** (0.99 / 0.95 without / with carry),
  +2.6 over its null, at only 0.30 nats of neutral damage (+0.15 over
  the controls); arithmetic_words accuracy 0.94, +2.1 over its null, at
  5.2 nats.
- Query-token KL 12–27 nats; neutral-prose damage 0.29–0.52 nats for
  five tasks (+0.15 to +0.38 over the controls), 0.8–1.4 for three, and
  4.7, 5.2 and 6.7 for present_participle, arithmetic_words and
  number_to_words.
- Commitment: removal leaves ≥ 50 % from 0.12–0.36 of the downstream
  depth, ≥ 90 % from 0.16–0.45; the direction alone suffices for 50 %
  until 0.08–0.78; the direction stops being needed before the end for
  every task (commitment layers 22–36 of 36).

The learned vector on the four models, in one paragraph. Fitting one
vector at one residual norm on 64 zero-shot prompts, with the model
frozen, reaches zero training loss at every candidate layer on every
task and model in under 30 steps, from a large solution set (cross-seed
cosines 0.01–0.33) that is orthogonal to PC1 and to the head-mean
function vector (|cos| ≤ 0.13). Held out, that one vector recovers
0.75–1.04 of the few-shot gap on all 37 task-model pairs (the
head-mean vector: 0.53–0.95) with accuracy 0.58–1.00, and beats a vector
fitted the same way to permuted targets by 1.6–12.8 nats everywhere: it
is a task vector, not a format vector. **It carries add-3 on every
model** (accuracy 0.58, 0.91, 0.98, 0.99 on 0.6B, 1.7B, 4B, 8B; with the
carry as well as without on the three larger models) and
arithmetic_words on 4B and 8B (0.94, 0.95), which no set of heads' mean
output carries: the rank-one limit predicted by the task-vector theory
does not bind for these tasks in these models, and the head-mean
construction, not the rank, was what failed in iteration 7a. The cost is
in the distribution: the query-token KL from the unsteered model is
12–30 nats (head mean 0.5–2.8) with the argmax changed for 96–100 % of
prompts, and on neutral prose the learned vectors exceed the random
controls' damage on every pair, by 0.02–0.6 nats for most tasks and by
1–9 nats for a task or three per model (add-3 on 0.6B and 8B,
number_to_words on 1.7B and 4B, present_participle and uppercase on 8B,
past_tense on 1.7B, arithmetic_words on 4B), where the head-mean vectors
sat at the controls' level. The commitment sweep gives the same
qualitative picture as for the head-mean vectors, with an earlier
hand-over (50 % by 0.05–0.50 of the downstream depth, 90 % by
0.11–0.55) and, for the first time, the direction no longer needed
against the random edits before the end on 35 of 37 pairs. The fit was
scored on the changed digits for add-3, and on 0.6B that let it corrupt
the unchanged ones (accuracy 0.53 without carry vs 0.71 with); a fit on
all tokens is the obvious follow-up, as is a damage-penalised fit, since
the vectors that exist are stronger than the model can bear on other
text.

### Iteration 8 (D32: the downstream trajectories compared all-to-all)

The question left by iteration 7b: the learned vector recovers the whole
few-shot gap while orthogonal to PC1 and to the head-mean function
vector, so what happens between the injection and the answer? A new
command, `directions trajectories`, reads the head-mean run and the
learned run of one model (iteration 6 and 7b, seed 20260907: same
splits and held-out prompts, checked) and re-captures the held-out
residuals under the natural few-shot context (the pipeline's own
few-shot prompts, a second demonstration sample, the deranged-label
contrast) and under each construction, PC1, the head-mean vector and
the learned vector, injected at common layers (the learned run's
selected layer as the primary, the head-mean run's when different),
each at the strength its own run's calibration gives it there (PC1,
calibrated in neither run, at the best point of the D1 grid on the
calibration pool). Every pair among the six trajectories is compared
per example and read point by the signed cosine, against a floor of
four isotropic directions per construction at its strength and the
ceiling of two natural trajectories, in three variants: raw, with the
answer's unembedding direction projected out, and with the *generic
response* projected out (D32, amended after the first 0.6B and 8B
runs: the isotropic floor itself rises with depth, to 0.14–0.30 at the
last read point in the median over pairs and up to 0.5–0.6, because any
perturbation of these norms drives the late residual along a shared
direction; with the mean isotropic trajectory removed from both vectors
the floors sit at 0.01–0.04 in the median). The generic response is
orthogonal to the answer direction (|cos| ≤ 0.14 at every read point),
so it is a residual-stream phenomenon, not the answer's; the answer
variant changes nothing anywhere (every number below within 0.03 of the
raw one), and the numbers quoted are the generic-removed ones unless
said otherwise. The first-version runs (before the amendment;
0.6B 35245908998, 1.7B 35246910274, 8B 35245885629) reproduce the
re-runs bit for bit in every raw array (903, 1026 and 909 arrays), a
determinism check across code versions on top of the repeated pass of
each run.

```
# workflow run 35248187883 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35036855067/pilot6_qwen3_0.6b_seed20260907 --learned-run /runpod-volume/results/run-35181427862/learned_qwen3_0.6b_seed20260907 --run-id trajectories_qwen3_0.6b_seed20260907
```

0.6B: 14 min (20 s to 2.3 min per task), determinism check passed; 8
tasks (the two few-shot rejects have no learned selection and are
skipped; add-3 has no head-mean vector, its run never had the label).

- **The learned vector's trajectory converges onto the natural one on
  all 8 tasks**, gradually: from a cosine of −0.01 to 0.13 with the
  few-shot difference at the injection layer to 0.50–0.92 at the last
  read point (raw 0.71–0.93; the ceiling, two demonstration samples,
  0.96–0.99; the floor −0.50 to 0.18). Half of the final alignment is
  reached at 0.41–0.70 of the downstream depth and 90 % at 0.64–1.00:
  a steady climb over the second half of the stack, with no jump.
- **The head-mean vector aligns less and earlier and then partly lets
  go**: peak 0.39–0.71 at 0.55–1.00 of the depth, final 0.12–0.71,
  with a significant decline from the peak on 3 of 7 tasks
  (past_tense 0.48 → 0.12, plural 0.54 → 0.33, present_participle
  0.41 → 0.22).
- **PC1's own component dies within two blocks.** At injection PC1 is
  the direction of the correct-versus-deranged contrast (cosine
  0.56–0.82 with it), and that cosine halves within 1–3 blocks and ends
  at −0.05 to 0.35; against the few-shot difference PC1 ends at −0.01 to
  0.31, carrying 0.00–0.18 of the gap at its calibrated strength.
- **The constructions do not converge onto one another.** Head mean
  versus learned ends at −0.15 to 0.71 (median 0.23) and PC1 versus
  learned at −0.10 to 0.44: what the two effective constructions share
  downstream is the natural component, not a common form of their own.

```
# workflow run 35253499155 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35037075704/pilot6_qwen3_1.7b_seed20260907 --learned-run /runpod-volume/results/run-35181498809/learned_qwen3_1.7b_seed20260907 --run-id trajectories_qwen3_1.7b_seed20260907
```

1.7B: 26 min (2–4 min per task), determinism check passed; 9 tasks
(arithmetic_words is the few-shot reject).

- **Learned: peaks and then lets go, mostly in the last block.** Peak
  0.51–0.79 at 0.82–1.00 of the depth, a significant decline on 8 of 9
  tasks to a final 0.28–0.79 (median 0.45; raw 0.52–0.91). The last
  block alone takes off 0.06–0.33 on seven tasks (plural 0.61 → 0.28,
  singular 0.66 → 0.41, present_participle 0.63 → 0.45), where the
  head-mean trajectory loses 0.00–0.04 on six of eight; add-3 goes the
  other way (+0.51 in the last block, to 0.79). Half of the peak at
  0.14–0.75 of the depth (median 0.55), 90 % at 0.71–1.00.
- **Head mean: converges on 6 of 8**, peak 0.44–0.79 at 0.59–1.00 of
  the depth, final 0.31–0.77 (median 0.53): the one model on which the
  head-mean vector ends more aligned with the natural trajectory than
  the learned vector does.
- **PC1 ends at −0.17 to 0.19**, at the floor on 5 of 9 and at
  0.12–0.19 on the other four, carrying 0.00–0.25 of the gap; its
  contrast component halves within 1–3 blocks from 0.65–0.87 at injection.
- Head mean versus learned ends at −0.06 to 0.66 (median 0.27), but on
  this model random directions at the head mean's strength reach
  0.16–0.69 with the learned trajectory even after the generic removal
  (the two strengths differ most here, the head mean injected at
  210–300 against the learned vector's 50–160, and the generic response
  is pooled over both), so there is no excess; PC1 versus learned −0.56
  to 0.21.

```
# workflow run 35247747527 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35038665604/pilot6_qwen3_4b_seed20260907 --learned-run /runpod-volume/results/run-35184285964/learned_qwen3_4b_seed20260907 --run-id trajectories_qwen3_4b_seed20260907
```

4B: 50 min (2–9 min per task; the 4B pipeline was the slow one before
too), determinism check passed; 10 tasks.

- **Learned: converges on 10 of 10**, final 0.67–0.83 (raw 0.72–0.90)
  from −0.02 to 0.12 at injection, floors −0.25 to 0.19; half of the
  final alignment at 0.33–0.55 of the depth (median 0.45), 90 % at
  0.73–1.00 (median 0.93). Against the correct-versus-deranged contrast
  the final cosine is 0.06–0.69 (median 0.53).
- **Head mean: peaks at 0.35–0.78 at 0.64–1.00 of the depth, ends at
  0.18–0.76**, declining significantly from the peak on 6 of 9 tasks
  (antonym 0.55 → 0.25, past_tense 0.50 → 0.22, singular 0.35 → 0.18
  among them).
- **PC1**: its contrast component halves within 2–3 blocks from
  0.81–0.94 at injection; against the few-shot difference it ends at
  −0.01 to 0.69, the high values where it also steers (last_antonym
  0.69 at 0.47 of the gap, add-3 0.47 at 0.08).
- Head mean versus learned ends at −0.14 to 0.62 (median 0.03).

```
# workflow run 35247750874 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35041155731/pilot6_qwen3_8b_seed20260907 --learned-run /runpod-volume/results/run-35181465402/learned_qwen3_8b_seed20260907 --run-id trajectories_qwen3_8b_seed20260907
```

8B: 19 min (1–3 min per task), determinism check passed; 10 tasks.

- **Learned: converges on 4, aligns then partly diverges on 6.** Peak
  0.56–0.81 at 0.80–1.00 of the depth, final 0.49–0.81 (raw 0.56–0.83),
  floors −0.38 to 0.24; the six declines are 0.07–0.15 from a peak at
  read points 31–35 of 36. Half of the peak at 0.28–0.66 of the depth
  (median 0.51), 90 % at 0.68–1.00 (median 0.81). Against the contrast
  the final cosine is 0.15–0.69 (median 0.56).
- **Head mean: converges on 5, partly diverges on 4**; peak 0.47–0.80
  at 0.66–1.00 of the depth, final 0.35–0.75. Its half-rise comes
  earlier than the learned vector's (0.17–0.48 of the depth, median
  0.34, against 0.51).
- **PC1 ends at −0.26 to 0.65 (median −0.03)**, above the floor at the
  end on 5 of 10 tasks, at −0.08 to 0.11 on four of them and at 0.65 on
  last_antonym, the one task where it steers (0.69 of the gap); its
  contrast component halves within 2–4 blocks from 0.43–0.92.
- Head mean versus learned ends at −0.16 to 0.72 (median 0.07), PC1
  versus learned at −0.43 to 0.38 (median −0.21): on this model the two
  effective constructions end unrelated to each other on most tasks
  while each is aligned with the natural trajectory.

The trajectories on the four models, in one paragraph, against the
three readings the comparison was built to separate. **The fitted
vector's trajectory collapses onto the natural one, progressively.**
On every task and model it starts orthogonal to the few-shot difference
at the injection layer (cosine −0.02 to 0.13) and climbs to a peak of
0.5–0.9 against a ceiling of 0.95–0.99 by the last read points, with the generic
response of any perturbation removed and floors at the origin; half of
the way is reached at about half of the downstream depth (medians 0.60,
0.55, 0.45, 0.51 on 0.6B, 1.7B, 4B, 8B) and 90 % at 0.8–0.9 of it, and
no task on any model shows a jump. "Different all the way through" is
false for the residual stream as a whole, with two qualifications that
keep part of it: the alignment saturates well below the ceiling, so a
component the natural trajectory does not have survives to the end, and
the two effective constructions do not converge onto each other (head
mean versus learned ends at a median of 0.23, 0.27, 0.03 and 0.07 on
the four models, against floors), so what the downstream layers
canonicalise is the component shared with the demonstrations, not a
form of the control signal of their own. **"Aligns then diverges"
holds in a weak form**: for the head-mean vector on about half the
pairs (a peak at 0.6–0.9 of the depth, then a significant decline of
0.1–0.4), and for the learned vector on the two models where the last
block pulls it away (1.7B, 8 of 9 tasks, by 0.06–0.33 in that block;
8B, 6 of 10, by 0.07–0.15), where on 0.6B and 4B it climbs to the end.
**PC1's own direction is dropped at once**: the correct-versus-deranged
contrast it is the top direction of (cosine 0.4–0.9 at injection)
halves within 1–4 blocks on every task and model, and PC1 ends at the
floor wherever it does not steer; the natural trajectory does not carry
that direction forward. The head-mean vector, which starts partly
aligned (−0.03 to 0.17), aligns earlier than the learned vector (half-rise
at a median of 0.35, 0.40, 0.17, 0.34 of the depth against 0.60, 0.55,
0.45, 0.51) and less far, except on 1.7B. Two things the numbers do
not yet say: whether the shared late component is what carries the
effect (a patch of it at the read point where the alignment arrives),
and what the generic response is; both are one captured-pass experiment
on the same runs (limitations below).

### Iteration 9 (D33: the geometry on the device, two strengths, the generic response, the patch test)

The `trajectories` command with its geometry moved onto the model's
device (the NumPy functions stay as the tested reference), run once per
model on the same head-mean and learned runs as iteration 8, with the
comparison repeated at half the canonical strength, the coherence of the
random responses, the generic-response diagnostics and the patch test
(configs/trajectories.yaml). The four runs took 3–7 min against 14–50 for
iteration 8's, doing twice the passes: the geometry is now a small part
of a run. All numbers below are at the primary layer with the generic
response removed unless said otherwise; "x1" is the canonical strength,
"x0.5" half of it. Skipped tasks and layers are as in iteration 8.

```
# workflow run 35296239774 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35036855067/pilot6_qwen3_0.6b_seed20260907 --learned-run /runpod-volume/results/run-35181427862/learned_qwen3_0.6b_seed20260907 --run-id trajectories9_qwen3_0.6b_seed20260907
```

0.6B: 2.8 min, determinism check passed; 8 tasks.

- **The patch test: the shared component carries the effect.** For the
  learned vector at three quarters of the downstream depth, keeping only
  the component along the prompt's own natural difference retains
  0.97–1.15 of the held-out effect (random per-example directions keep
  0.00–0.11), removing it leaves −0.76 to 0.25 (random removal 1.00;
  negative means worse than the unsteered model), and the natural
  difference patched into the unsteered run, with nothing injected at
  the injection layer, gives 0.89–1.25 of the effect by itself (random
  vectors of the same norm −0.27 to −0.07); every contrast p = 0.000
  (seven lexical tasks). At half depth the hand-over is under way: keep
  retains 0.18–1.03, remove leaves −0.94 to 0.99, the patch alone gives
  0.19–1.23, in step with the alignment there (0.19–0.44 against
  0.51–0.80 at three quarters). Add-3 is the exception by construction:
  its scored digits sit at later target positions whose predictions
  read the query position through the keys and values of earlier
  layers, so every edit at or after half depth is a no-op on the
  per-token score (retained 1.00, the patch 0.02), and the first-token
  ratios are undefined (a first-token effect of 0.1 nats).
- **Half strength.** The learned vector at x0.5 still recovers
  0.45–0.91 of the gap (x1 0.81–1.03) and aligns with the natural
  trajectory as much (final 0.47–0.80 against 0.50–0.92; converges on 6
  of 8, partly diverges on 2); the head mean at x0.5 keeps 0.04–0.27 of
  the gap (x1 0.20–0.71). The same construction's trajectories at the two
  strengths start identical and end at a cosine of 0.30–0.75 (learned)
  and 0.42–0.93 (head mean): the downstream change is not proportional
  to the push.
- **The random responses are coherent from the first block.** The norm
  of their mean over the mean of their norms is 0.35–0.53 one block after
  injection and 0.53–0.81 at the end at x1, 0.32–0.51 and 0.42–0.74 at
  x0.5: half of a random push's downstream change is the shared part,
  at either strength. The generic response's norm is 0.11–0.16 of the
  residual's at x1 and 0.04–0.08 at x0.5, so its size is close to
  proportional to the push while its coherence is not.
- **What the generic response is.** Its energy sits 0.29–0.64 in the
  residual's 16 largest coordinates (of 1024) and 0.39–0.70 in the
  largest 64; its cosine with the mean residual is −0.63 to −0.77 at
  mid-depth and −0.32 to −0.82 at the end; the generic responses at the
  two strengths agree at 0.35–0.88 at the end. The logit lens of the mean
  random response promotes fragments and function words ('CUR', 'number',
  'culture', 'and', '=') and demotes ' yes', ' Sure', ' options': a push
  away from the background along the massive coordinates that reads out
  as a mild move toward generic tokens (mean absolute logit change
  0.3–0.7).

```
# workflow run 35296599331 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35037075704/pilot6_qwen3_1.7b_seed20260907 --learned-run /runpod-volume/results/run-35181498809/learned_qwen3_1.7b_seed20260907 --run-id trajectories9_qwen3_1.7b_seed20260907
```

1.7B: 4.2 min, determinism check passed; 9 tasks.

- **Patch test**, learned vector, eight lexical tasks: at three quarters
  of the depth keep retains 0.87–1.19 (random −0.00 to 0.12), remove
  leaves −1.14 to −0.14 (random 1.00), the natural patch alone gives
  0.90–1.24 (random −0.15 to −0.05), all p = 0.000; at half depth keep
  0.23–1.09, remove −0.57 to 0.94, patch 0.45–1.21 (alignment 0.10–0.41
  there, 0.39–0.76 at three quarters). For add-3 the first digit's
  effect is carried by the shared component from half depth on (the
  natural patch alone gives 0.96 of it; removal costs nothing at half
  depth and everything at three quarters).
- **Half strength removes the late decline.** At x1 the learned vector
  partly diverged after its peak on 8 of 9 tasks (iteration 8: the last
  block pulled it away); at x0.5 it converges on 7 of 9, with a final
  alignment of 0.19–0.91 (median 0.68) against 0.28–0.79 (median 0.45),
  while keeping 0.31–1.07 of the gap (median 0.75; x1 0.81–1.01). The
  head mean at x0.5 keeps 0.34–0.65 of the gap (x1 0.64–0.91) and ends
  at 0.20–0.68 (x1 0.31–0.77). The two strengths' trajectories end at a
  cosine of 0.20–0.97 (median 0.64) for the learned vector, 0.75–0.95
  for the head mean.
- **Coherence** 0.35–0.58 one block after injection, 0.66–0.84 at the
  end (x0.5: 0.33–0.56, 0.46–0.76); the generic response's norm is
  0.05–0.23 of the residual's at x1 (median 0.19, the largest of the
  four models) and 0.03–0.18 at x0.5.
- **The generic response here is almost the negative of the residual
  mean**: cosine −0.42 to −0.82 at mid-depth and −0.56 to −0.97 at the
  end, with 0.17–0.49 of its energy in the residual's 16 largest
  coordinates (of 2048) and 0.36–0.76 in the largest 64; the two
  strengths' generic responses agree at 0.73–0.97. Its logit lens
  promotes punctuation (',', '.', ':') and demotes answer-opening
  adverbs (' Indeed', ' Typically', ' Essentially', ' True').

```
# workflow run 35296680682 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35038665604/pilot6_qwen3_4b_seed20260907 --learned-run /runpod-volume/results/run-35184285964/learned_qwen3_4b_seed20260907 --run-id trajectories9_qwen3_4b_seed20260907
```

4B: 8.8 min (50 in iteration 8), determinism check passed; 10 tasks.

- **Patch test**, learned vector, nine lexical tasks: the hand-over is
  complete by half depth here. Keep retains 0.94–1.20 at half depth and
  0.98–1.29 at three quarters (random 0.00–0.47, the 0.47 on
  arithmetic_words), remove leaves −0.19 to 0.84 and −0.73 to 0.52
  (random 1.00), the natural patch alone gives 0.53–1.31 and 0.52–1.33
  (random −0.11 to −0.01), all p = 0.000; the alignment is 0.35–0.55 at
  half depth and 0.54–0.73 at three quarters. Add-3's first digit: the
  natural patch alone gives 0.98 of its effect at half depth.
- **Half strength**: the learned vector keeps 0.66–1.05 of the gap
  (median 0.91; x1 0.75–1.03) and converges on 10 of 10 at both
  strengths, final 0.51–0.86 (median 0.82) against 0.67–0.83 (median
  0.76); the head mean at x0.5 keeps 0.05–0.35 (x1 0.33–0.69). The two
  strengths' trajectories end at 0.57–0.93 (learned), 0.67–0.93 (head
  mean).
- **Coherence** 0.34–0.45 one block after injection, 0.56–0.67 at the
  end (x0.5: 0.33–0.47, 0.44–0.56); the generic response's norm 0.05–0.14
  of the residual's at x1, 0.02–0.04 at x0.5.
- **Generic response**: 0.03–0.19 of its energy in the residual's 16
  largest coordinates (of 2560), 0.08–0.36 in the largest 64; cosine
  with the mean residual −0.48 to −0.76 at mid-depth and −0.84 to 0.45
  at the end; the two strengths agree at 0.67–0.93. The logit lens is
  the least regular of the four models (fragments, a few nouns, ':' and
  ','; demotes ' option', ' typically', markup tokens).

```
# workflow run 35296238273 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-35041155731/pilot6_qwen3_8b_seed20260907 --learned-run /runpod-volume/results/run-35181465402/learned_qwen3_8b_seed20260907 --run-id trajectories9_qwen3_8b_seed20260907
```

8B: 4.1 min, determinism check passed; 10 tasks.

- **Patch test**, learned vector, nine lexical tasks: at three quarters
  of the depth keep retains 0.98–1.15 (random 0.00–0.45, the 0.45 on
  arithmetic_words), remove leaves −0.48 to 0.76 (random 1.00), the
  natural patch alone gives 0.52–1.20 (random −0.10 to −0.02), all
  p = 0.000; at half depth keep 0.45–0.99, remove 0.68–1.00, patch
  0.21–1.13 (alignment 0.16–0.49 there, 0.46–0.72 at three quarters).
  Add-3's per-token rows are no-ops as on 0.6B, but its first-token
  effect (1.0 nats on the first digit) reads normally: at half depth
  the natural patch alone gives 0.98 of it and keeping the shared
  component 0.95, removing it 0.77; at three quarters removal leaves
  0.08.
- **Half strength suits the learned vector better here.** At x0.5 it
  recovers 0.34–1.01 of the gap (median 0.95; x1 0.83–1.04) and aligns
  with the natural trajectory *more* than at x1: final 0.59–0.93
  (median 0.74) against 0.49–0.81 (median 0.63), converging on 9 of 10
  where x1 partly diverged on 6. The head mean at x0.5 keeps 0.04–0.75
  of the gap (median 0.15). Trajectories at the two strengths end at a
  cosine of 0.14–0.94 (median 0.85) for the learned vector, lowest for
  arithmetic_words (0.14) and past_tense (0.26), the two tasks where
  half strength loses most of the effect.
- **Coherence** 0.35–0.52 one block after injection, 0.48–0.76 at the
  end (x0.5: 0.33–0.51, 0.41–0.76); the generic response's norm
  0.04–0.14 of the residual's at x1, 0.02–0.09 at x0.5.
- **The generic response on 8B is less a matter of the massive
  coordinates**: 0.06–0.40 of its energy in the residual's 16 largest
  coordinates (of 4096) and 0.11–0.56 in the largest 64, most of it
  put there by the last block (0.04–0.36 before it, 0.11–0.56 after;
  past_tense 0.14 to 0.49);
  cosine with the mean residual −0.29 to −0.61 at mid-depth, −0.59 to
  0.59 at the end (positive only for arithmetic_words). Its
  logit lens promotes punctuation and Chinese function words (',', '.',
  '不', '和', '当然') and demotes content tokens of the task's kind
  (' produces', ' performs' on plural and singular; ' islands',
  ' hurricane' on the antonym tasks): a move toward the model's
  unconditional defaults.

Iteration 9 on the four models, in one paragraph. **The shared component
is what carries the effect.** On 33 lexical task-model pairs, by three
quarters of the downstream depth, keeping only the component of the
fitted vector's perturbation along the prompt's own natural difference
retains 0.87–1.33 of the held-out effect (random per-example directions
0.00–0.12, except 0.45–0.47 on two arithmetic_words pairs), removing that
component leaves −1.14 to 0.76 (random removal 1.00), and the natural
difference patched into the unsteered run, with nothing injected, gives
0.52–1.34 of the effect by itself (random vectors of its norm −0.27 to
−0.01), every contrast at p = 0.000. At half depth the same edits are
partial on 0.6B, 1.7B and 8B (keep 0.18–1.09, patch 0.19–1.23) and
already complete on 4B, in step with the alignment measured there. So the
cosines of iteration 8 were reading a causal quantity: where the fitted
trajectory has come to agree with the natural one, that agreement is
necessary and sufficient for the task, and the natural difference alone
often does slightly more than the fitted vector. For add-3, whose scored
digits are predicted at later positions, edits at the query token from
half depth on are no-ops on the per-token score (those digits read the
query position through earlier layers' keys and values) while the first
digit behaves like the lexical tasks. **Half strength.** At half the
canonical strength the learned vector keeps most of its effect (medians
0.75–0.95 of the gap; the head mean 0.15–0.59) and aligns with the
natural trajectory at least as well (final medians 0.68–0.82 against
0.45–0.76 at full strength), and the late decline seen at full strength
on 1.7B and 8B largely disappears (converging on 7 of 9 and 9 of 10 where
full strength converged on 1 and 4): the canonical strength overshoots on
those models. The same construction's trajectories at the two strengths
start identical and end at cosines of 0.14–0.97 (learned) and 0.42–0.96
(head mean): the downstream change is not proportional to the push.
**The generic response.** Random pushes produce coherent downstream
changes from the first block on: the norm of their mean over the mean of
their norms is 0.34–0.58 one block after injection and 0.48–0.84 at the
end on every model, and the same at half strength (0.32–0.56, 0.41–0.76),
so the shared response is not a large-push nonlinearity, while its size
scales with the push (0.04–0.23 of the residual's norm at full strength,
0.02–0.18 at half). What it is: at mid-depth it points against the mean
residual on every model (cosine −0.29 to −0.82), and at the end still on
the two small ones (0.6B −0.32 to −0.82, 1.7B −0.56 to −0.97) and less so
on 4B and 8B (−0.84 to 0.59); its energy sits in the residual's largest
coordinates in proportion to their weight, 0.29–0.64 in the 16 largest on
0.6B and 0.03–0.40 on 4B and 8B, where the last block puts most of it
there; its readout promotes punctuation, function words and the model's
default tokens and demotes answer-opening adverbs and task-content words.
A shrinkage of the background along the massive coordinates, with a
drift toward the unconditional defaults, at a magnitude of a few percent
of the residual: that is the component the generic-removed variant takes
out, and it is orthogonal to the answer on every model. **Cost.** The
four runs took 2.8, 4.2, 8.8 and 4.1 min, about $0.60 in total, against
$3.20 for iteration 8's; the geometry on the device makes a strength
factor cost 15 passes per layer and nothing else.

### Iteration 10 (D33 amended: quarter strength, the neighbouring candidate layers, the patch grid at eighths for both vectors, per-block writing; a second seed)

The learned-vector protocol (D31, `learned_*.yaml`) run at seed 20260916
on the four models, then the `trajectories` command with `--seed
20260916` on those runs and on iteration 6's head-mean runs of the same
seed, with `strength_factors: [1.0, 0.5, 0.25]`, `neighbour_layers: 1`
(the nearest candidate layer below and above the learned run's selected
layer, each construction at the strength its own run's calibration
gives it there), the patch test at every compared layer on eighths of
the downstream depth for the learned and the head-mean vector, and the
per-block writing of the shared component. Nothing in the seed-20260907
runs was re-run: the quantities the two iterations share (the primary
layer, canonical and half strength, the patch rows at 0.5, 0.75 and
1.0) are compared across seeds below, at the same injection layer
wherever seed 2 covers seed 1's selected layer as its primary or as a
neighbour. All alignment numbers are with the generic response removed;
"x1", "x0.5", "x0.25" are the strength factors.

**The learned vector replicates on effect, not on layer.** The "best"
layer rule (D27) picks a different candidate on 6 of 8 tasks (0.6B),
8 of 9 (1.7B), 8 of 10 (4B) and 8 of 10 (8B) between the two seeds, while
the held-out effects agree within 0.02–0.36 nats per token (the
aggregates: `directions aggregate` of the two learned runs per model)
and the gap fractions within 0.03. The candidates' effects are close
(every candidate recovers 0.8–1.0 of the gap at its natural norm), so
the selection is a coin toss among them and the comparisons below are
read at the same layer in both seeds where possible.

```
# workflow runs 35414623339 (learned, H100 80GB HBM3, ADA_80_PRO, US-CA-2, Flash environment ci-8b) and 35417242891 (comparison, same)
uv run directions pilot --config configs/learned_qwen3_8b.yaml --seed 20260916 --run-id learned_qwen3_8b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050604985/pilot6_qwen3_8b_seed20260916 --learned-run /runpod-volume/results/run-35414623339/learned_qwen3_8b_seed20260916 --run-id trajectories10_qwen3_8b_seed20260916
```

8B: learned run 51 min, comparison 17.6 min (4.1 in iteration 9: the
patch grid is now 9.9 min of it), both determinism checks passed; 10
tasks, 29 compared layers (2–4 per task).

- **The hand-over sits at a fixed read point, not at a fixed distance
  from the injection.** For each lexical task the first read point at
  which keeping only the component along the prompt's natural
  difference retains 0.9 of the learned vector's effect is the same
  whether the vector was injected at layer 7, 11, 14 or 18: past_tense
  27/25/25 (layers 11/14/18), plural 29/27/25 (7/11/14), singular,
  uppercase, number_to_words, present_participle 27/25/25, antonym
  22/25 (14/18), last_antonym 25/23/25/25 (7/11/14/18); that is 7–22
  blocks after the injection and 0.39–0.76 of the downstream depth, but
  read points 22–29 of 36 in every case. The natural difference patched
  alone reaches 0.9 of the effect at read points 20–25 and removal falls
  to 0.5 at 25–33, with the same invariance. Blocks around 20–27 of 36
  canonicalise whatever enters them; what the injection layer changes is
  only how many blocks the vector waits.
- **The shared component carries the head mean's effect as well.** At
  three quarters of the depth keeping only the component along the
  natural difference retains 0.63–2.38 of the head mean's effect
  (random per-example directions at most 0.07), removing it leaves
  −0.83 to 0.33 (random 1.00), and the natural difference alone gives
  1.13–4.71 of the head mean's effect (random vectors of its norm −0.44
  to −0.03): more than the head mean itself, which recovers 0.21–0.88 of
  the gap where the natural difference recovers all of it. Its hand-over
  read points (keep ≥ 0.9 at 20–30) coincide with the learned vector's.
- **Quarter strength is below the working range on 8B.** The learned
  vector at x0.25 keeps 0.08–0.72 of the gap (median 0.35) and the head
  mean at most 0.20; on the four task-layer pairs where x0.25 still
  recovers half the gap the final alignment is 0.47–0.74, no better
  than x1's 0.52–0.85 or x0.5's 0.76 (median). Half strength remains the
  sweet spot: gap 0.51–1.00 (median 0.91), converging on 9 of 10 at the
  primary layer where x1 partly diverges on 8 of 10 (final medians 0.76
  against 0.63), as in iteration 9's seed.
- **At a neighbouring layer the learned vector fitted there steers as
  well** (0.89–1.03 of the gap at x1 on 19 neighbour pairs, at the
  reliable grid point nearest ρ = 1) and aligns as much (final 0.29–0.83,
  median 0.62, against 0.52–0.85 at the primaries), converging on 7 and
  partly diverging on 12; the selected layer is not special.
- **Block writing.** With the reference moving with depth, the last
  block writes 0.47–1.13 of the natural difference into the learned
  trajectory on all 29 layers (the next largest block 0.20–0.42 of
  that) and 0.43–0.74 into the natural trajectory itself; the rank
  correlation of the learned and the natural block profiles is 0.12–0.69
  (median 0.47), the head mean's 0.15–0.75. Held at the hand-over read
  point m* on the population means (offline, from the saved means), the
  component present there was written by blocks m*, m*−1 and m*−2 (half
  of it by m*−1), with the same top three blocks for the steered and the
  natural run on every task and layer; held at the last read point, the
  last block writes 0.6–0.8 of it. So the shared component is not a
  fixed direction carried forward: every block re-writes it, and the
  hand-over is where the steered run's re-writing has caught up with the
  natural one. The moving reference makes the natural trajectory's own
  profile trivially peak at block 1 (its difference is zero at the
  embedding) and the entropy ratios uninformative (0.95–0.97).
- **Coherence** at the end 0.48–0.90 at x1, 0.33–0.77 at x0.5, 0.28–0.72
  at x0.25: it falls with the push here, unlike between x1 and x0.5 in
  iteration 9.
- **Cross-seed, at the same layer (7 tasks).** Seed 2 minus seed 1:
  learned gap fraction at x1 −0.01 to +0.01, final alignment −0.05 to
  +0.04, head-mean gap −0.06 to +0.02 and final −0.08 to +0.04, coherence
  −0.06 to +0.12, generic norm ratio ±0.03, keep at 0.75 ±0.01, the
  natural patch at 0.75 −0.01 to +0.03, at 0.5 ±0.02; only removal at
  0.75 (−0.50 to +0.20) and the half-strength gap on arithmetic_words
  (0.34 → 0.76) move. The trajectory results are a property of the
  model, not of the fit. Taken at each seed's own selected layer instead
  the spreads triple (final x1 −0.18 to +0.14), which is the layer
  difference, not noise.

```
# workflow runs 35414652608 (learned, RTX 4090, ADA_24, EUR-NO-1) and 35417248145 (comparison, same)
uv run directions pilot --config configs/learned_qwen3_0.6b.yaml --seed 20260916 --run-id learned_qwen3_0.6b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050574186/pilot6_qwen3_0.6b_seed20260916 --learned-run /runpod-volume/results/run-35414652608/learned_qwen3_0.6b_seed20260916 --run-id trajectories10_qwen3_0.6b_seed20260916
```

0.6B: learned run 50 min, comparison 12.6 min (2.8 in iteration 9),
determinism checks passed; 8 tasks, 24 compared layers.

- **The same fixed read point.** Keep ≥ 0.9 of the learned vector's
  effect at read points 20–23 of 28 for injections at layers 6, 8, 11
  and 14 (plural 20/20/21, present_participle 20/23/21, uppercase
  20/20/21, past_tense 20/22/21; antonym earlier, 18–20): 4–15 blocks
  after the injection, 0.29–0.75 of the downstream depth, read points
  18–23 in every case. The natural patch alone reaches 0.9 at 16–20.
- **Head mean**: keep at 0.75 retains 1.02–5.48 of its effect (random
  ≤ 0.23), the natural patch alone 1.38–9.93 (the head mean recovers
  0.09–0.72 of the gap here), removal −2.92 to 0.39. Its hand-over is
  earlier than the learned vector's on most pairs (keep ≥ 0.9 at 10–23).
- **Quarter strength**: the learned vector keeps 0.00–0.29 of the gap at
  the primary layers and 0.07–0.44 at the neighbours; nothing to align.
  **Half strength loses more on this seed's fits**: 0.17–0.91 of the gap
  at the primary layers (median 0.39; seed 1: 0.45–0.91), while at the
  deeper neighbour (layer 14) it keeps 0.83–0.93. At x1 the learned
  vector converges on 8 of 8 primaries and 15 of 16 neighbours (final
  0.52–0.89, median 0.69).
- **Block writing**: the last block writes 0.35–0.76 of the natural
  difference into the learned trajectory (dominant on 24 of 24) and
  0.22–0.48 into the natural one; for the head mean it is not the
  dominant block (last-block share −0.10 to 0.36, median 0.03). Rank
  correlation with the natural profile 0.02–0.81 (learned), 0.37–0.85
  (head mean).
- **Cross-seed, same layer (8 tasks)**: learned gap at x1 ±0.02, final
  −0.15 to +0.01, keep at 0.75 −0.01 to +0.03, natural patch ±0.02,
  coherence ±0.06; the half-strength gap −0.74 to +0.40 (uppercase 0.75
  → 0.19, singular 0.91 → 0.17): at half the canonical norm a different
  fit of the same layer has a different strength curve on this model.

```
# workflow runs 35414702823 (learned, RTX 4090, ADA_24, EUR-NO-1) and 35419811910 (comparison, same)
uv run directions pilot --config configs/learned_qwen3_1.7b.yaml --seed 20260916 --run-id learned_qwen3_1.7b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050661831/pilot6_qwen3_1.7b_seed20260916 --learned-run /runpod-volume/results/run-35414702823/learned_qwen3_1.7b_seed20260916 --run-id trajectories10_qwen3_1.7b_seed20260916
```

1.7B: learned run 52 min, comparison 16.0 min (4.2 in iteration 9),
determinism checks passed; 9 tasks, 25 compared layers.

- **The fixed read point again**: keep ≥ 0.9 of the learned vector's
  effect at read points 17–22 of 28 for injections at layers 6–14
  (uppercase 20/20/22 for layers 6/8/11, plural 20/22/21 for 8/11/14,
  present_participle 20/22/21, past_tense 20/19/21, antonym 17/18/17/18
  for 6/8/11/14), 4–14 blocks after the injection; the head mean's at
  16–22, mostly a read point or two earlier. The exception is
  last_antonym, where keeping the shared component alone reaches 0.9 of
  the learned vector's effect only at the last read point from every
  injection layer (removal still costs half of it by read points 16–19,
  and the natural patch alone reaches 0.9 at 17–22): on this task the
  learned vector keeps a route of its own beside the shared one.
- **Head mean** (the one model on which it steers as well as the learned
  vector, gap 0.61–0.94): keep at 0.75 retains 0.74–1.38 of its effect
  (random ≤ 0.10), the natural patch alone 1.04–1.71, removal −0.15 to
  0.46; converging on 7 of 8 primaries at x1.
- **Quarter strength**: the learned vector keeps 0.04–0.68 of the gap
  (median 0.27), the head mean 0.12–0.35; half strength keeps 0.43–1.00
  (median 0.89) and converges on 7 of 8 primaries where x1 partly
  diverges on 6 of 9 (final medians 0.75 against 0.52), as in seed 1.
- **Block writing**: the last block writes 0.11–1.13 of the natural
  difference into the learned trajectory (median 0.48; dominant on most
  layers, not all) and 0.26–0.62 into the natural one; the head mean's
  last-block share is 0.08–0.31 and its profile follows the natural one
  most closely of the three models (rank correlation 0.50–0.89, median
  0.71; the learned vector's 0.32–0.81).
- **Cross-seed, same layer**: only 4 of 9 tasks, since seed 2's layers
  (its selection and the neighbours) miss seed 1's selection on the
  other five. On those four: learned gap at x1 −0.02 to +0.03, final +0.01
  to +0.07, head-mean final −0.06 to +0.13, keep at 0.75 −0.07 to +0.01,
  the natural patch −0.04 to +0.01; the half-strength gap −0.33 to +0.04
  and removal at 0.75 −0.04 to +0.48. Two labels flip from
  aligns_then_partly_diverges to converges at x1 (antonym, number_to_words),
  a change in the significance of a decline of 0.05–0.10, not in the curve.

```
# workflow runs 35420419658 (learned, RTX 4090, ADA_24, EUR-NO-1) and 35421215047 (comparison, same)
uv run directions pilot --config configs/learned_qwen3_4b.yaml --seed 20260916 --run-id learned_qwen3_4b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35051925394/pilot6_qwen3_4b_seed20260916 --learned-run /runpod-volume/results/run-35420419658/learned_qwen3_4b_seed20260916 --run-id trajectories10_qwen3_4b_seed20260916
```

4B: learned run 79 min, comparison 27.9 min (8.8 in iteration 9),
determinism checks passed; 10 tasks, 29 compared layers.

- **The fixed read point**: keep ≥ 0.9 of the learned vector's effect
  at read points 22–27 of 36 for injections at layers 7, 11, 14 and 18
  (past_tense 25/27/25/25, plural 25/23/25/25, singular 25/23/25,
  uppercase 23/25/25, present_participle 27/25/25, number_to_words
  25/25, last_antonym 25/23/25), 4–18 blocks after the injection and
  0.22–0.64 of the downstream depth; the natural patch alone reaches 0.9
  at 20–27, removal falls to 0.5 at 22–29. The same read points as on 8B
  (both 36 blocks): the canonicalising blocks sit at 0.6–0.8 of the
  stack on every model (0.6B and 1.7B: 17–23 of 28).
- **The learned vector converges on every pair at x1** (10 of 10
  primaries, 19 of 19 neighbours; final 0.64–0.84, median 0.77, the
  highest of the four models) and at x0.5 (final 0.56–0.89, gap
  0.15–1.00, median 0.90); x0.25 keeps 0.03–0.69 of the gap (median
  0.17). The head mean is weak on this model (0.10–0.61 of the gap at
  x1, at most 0.29 at x0.5), and where it steers its effect is carried
  by the shared component like the others (keep at 0.75 0.73–1.74, the
  natural patch alone 1.47–5.72, removal −0.18 to 0.44).
- **Block writing**: the last block writes 0.65–1.02 of the natural
  difference into the learned trajectory (dominant on 29 of 29) and
  0.40–0.62 into the natural one; the head mean's last-block share is
  −0.03 to 0.20 (median 0.02) and its profile the least like the
  natural one (rank correlation −0.03 to 0.60; the learned vector's
  0.09–0.82).
- **Cross-seed, same layer (7 tasks)**: learned gap at x1 −0.08 to
  +0.02, final ±0.05, at x0.5 gap −0.04 to +0.06 and final −0.05 to
  +0.06, keep at 0.75 −0.02 to +0.09, the natural patch −0.02 to +0.09,
  coherence ±0.08, generic norm ratio −0.01 to +0.04. The head mean is
  the one construction that moves consistently, −0.02 to −0.13 of the
  gap and −0.00 to −0.14 of the final alignment, because seed 2's head
  mean is a different run (iteration 6's second seed: its own heads and
  strength), not a re-read of the same vector.

Iteration 10 on the four models, in one paragraph. **The hand-over is
a property of the depth, not of the vector or its injection layer.** On
every lexical task and model the read point at which the component of
the perturbation along the prompt's own natural difference carries at
least 0.9 of the learned vector's effect is the same for every candidate
injection layer, 17–23 of 28 blocks on 0.6B and 1.7B and 22–29 of 36 on
4B and 8B (0.6–0.8 of the stack), whether that is 4 or 22 blocks after
the injection; the natural difference patched alone reaches 0.9 of the
effect a read point or two earlier and removal costs half of the effect
a read point or two later, with the same invariance. A learned vector
fitted at a neighbouring candidate layer, at the strength that layer's
calibration gives it, steers as well as the selected one (0.72–1.04 of
the gap on 70 neighbour pairs) and aligns as much, so the "best" layer
rule (D27) selects among equivalents, which is why the two seeds pick
different layers on 30 of 37 tasks while the held-out effects agree
within 0.36 nats. **The shared component carries the head mean's effect
as it carries the learned vector's**: keeping only that component
retains 0.47–5.48 of the head mean's effect and the natural difference
alone gives 1.04–9.93 of it (the head mean recovers 0.07–0.94 of the
gap where the natural difference recovers all of it; random matches at
most 0.23 and −1.42 to −0.01), with a hand-over at the same read points
or a little earlier; on 1.7B's last_antonym alone the learned vector
keeps a route of its own beside the shared one (keep ≥ 0.9 only at the
last read point from every layer). **Quarter strength is below the
working range**: the learned vector keeps 0.00–0.88 of the gap at x0.25
(medians 0.14–0.35 at the primaries) and the head mean −0.01 to 0.45,
and where x0.25 still steers its alignment (0.24–0.87) is no better than
x0.5's; half strength remains the point at which the alignment is
highest (final medians 0.64–0.78 against 0.52–0.77 at x1) with most of
the effect (medians 0.39–0.91 of the gap), except that on 0.6B this
seed's fits lose more at x0.5 than seed 1's did. **Per-block writing is
a moving target.** With the reference taken at each read point, the
last block writes the largest share of the natural difference into the
learned trajectory on every layer of 0.6B, 4B and 8B and most of
1.7B's (0.11–1.13 of it), the natural
trajectory itself gets 0.22–0.74 of its final difference from its last
block, and the block profiles of the learned and the natural trajectory
rank-correlate at 0.02–0.82 (medians 0.47–0.65); held at the hand-over
read point on the population means (8B, offline) the component present
there was written by the two or three blocks just before it, identically
for the steered and the natural run. The natural difference is
re-written by every block rather than carried, and the hand-over is the
read point from which the steered run's re-writing coincides with the
natural one; the moving reference makes the entropy ratios
uninformative (0.95–0.97 for the natural trajectory) and this measure
is exploratory. **Cross-seed.** At the same injection layer the
comparison's numbers replicate to within a few hundredths on 7, 8, 4
and 7 tasks (0.6B, 1.7B, 4B, 8B): the learned vector's gap fraction at
x1 within ±0.08, its final alignment within −0.15 to +0.07, keep and the
natural patch at 0.75 within ±0.09, coherence within ±0.12; the
quantities that move are removal at 0.75 (−0.50 to +0.60, the noisiest
edit), the half-strength gap on 0.6B (−0.74 to +0.40) and the head mean
on 4B (a different head-mean run), and the labels flip on 4 of 26 pairs
where a decline of 0.05–0.10 crosses significance. Taken at each seed's
own selected layer the spreads are two to three times larger, which is
the layer difference. The trajectory results are a property of the
model; the learned vector's selected layer is not.

**Cost.** The eight runs: learned-vector runs of 50, 52, 79 and 51
min and comparisons of 12.6, 16.0, 27.9 and 17.6 min (0.6B, 1.7B, 4B,
8B), about $9.80 of worker time in total ($5.46 of it the 8B pair on
the H100), against the $2.50 of a comparison-only iteration; the second
seed of the learned vector is three quarters of it. The comparisons
took 3–4.5 × iteration 9's, as projected, the patch grid (eight read
points, two constructions, every compared layer) being half of each.

### Iteration 11 (D34: the causal dimensionality of the shared component)

The `trajectories` command with the subspace test (D34) on
`configs/trajectories_subspace.yaml`, which carries only what the test
needs from the earlier stages (the canonical strength, the primary
layer, the learned vector's patch grid at eighths for the hand-over
read point, no diagnostics), on the seed-20260916 runs of iteration
10. Per task, at the primary layer, the natural differences of the
extraction and calibration pools (128 prompts, never evaluated) are
captured once; at the hand-over read point (the first patch-grid read
point at which keeping the prompt's own direction retains 0.9 of the
learned vector's effect), at half depth and at the last read point, the
steered perturbation is kept only within (or stripped of) the top-k
uncentred principal subspace of those differences, and the prompt's
natural difference projected on that subspace is patched alone, for k
in 1, 2, 4, 8, against random per-example k-subspaces, the top-k
subspace of the pool's unsteered residuals (the background) and the
top-k subspace of the other tasks' pool differences (leave-one-task-out).
"Keep at k" below is the effect retained with only the k-dimensional
own projection of the perturbation left in place; "patch at k" the
effect of the projected natural difference alone in the unsteered run;
the per-prompt keep and patch of the patch grid are the ceilings.

```
# workflow run 35464684771 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050574186/pilot6_qwen3_0.6b_seed20260916 --learned-run /runpod-volume/results/run-35414652608/learned_qwen3_0.6b_seed20260916 --run-id trajectories11_qwen3_0.6b_seed20260916
```

0.6B: 5.1 min (the subspace stage 2.2 of it), determinism check passed;
8 tasks, the learned vector, hand-over read points 18–22 of 28.

- **Rank one.** At the hand-over read point the own subspace of rank
  one, the pool's mean natural difference (cosine 0.99–1.00 with the
  held-out prompts' mean), carries the effect: keep at k = 1 retains
  0.75–0.99 of it on the seven lexical tasks (antonym 0.75, the others
  0.90–0.99) against a per-prompt ceiling of 0.91–0.98, and the mean
  difference patched alone gives 0.89–0.97 (ceiling 0.90–1.18); random
  rank-1 subspaces keep 0.00–0.13 and patch 0.00. Ranks 2, 4 and 8 add
  0.00–0.06 (keep at k = 8: 0.81–0.99). k90 for keep is 1 on five tasks,
  2 on present_participle, and never reached on antonym (0.81 at k = 8,
  ceiling 0.91). The explained fraction of the pool differences' top
  component is 0.78–0.92 and the participation ratio 1.2–1.7.
- **The task's own, not a shared in-context subspace.** The other tasks'
  subspace keeps 0.00–0.52 of the effect at k = 1 (median 0.29) and
  0.00–0.48 at k = 8, and patched alone gives −0.08 to 0.44, though its
  top component overlaps the task's own at 0.66–0.86 of the energy at
  k = 1: the direction the tasks share is most of the mean difference by
  norm and a small part of its effect. The background subspace keeps
  −0.47 to 0.16 and patches 0.00–0.10.
- **Before the hand-over** (half depth, five tasks with a distinct read
  point) the rank-1 own subspace retains 0.14–0.46, equal to the
  per-prompt ceiling there (0.15–0.49), so the shortfall is the depth,
  not the rank. **At the last read point** the own rank-1 subspace
  retains 0.66–0.87 and rank 8 0.71–0.93 (ceilings 0.99–1.20), the
  patch alone 0.58–0.86; the other-task and background subspaces are
  destructive there (keep −0.10 to −1.46 with wide intervals, patch
  −0.58 to 0.31): what remains at the end is task-specific and slightly
  higher in rank than at the hand-over.
- Add-3 is the known no-op (the digits are predicted at later positions;
  every edit at the query token retains 1.00 and patches 0.01).

```
# workflow run 35464683643 (push-triggered request), H100 80GB HBM3 (ADA_80_PRO), US-CA-2, Flash environment ci-8b
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050604985/pilot6_qwen3_8b_seed20260916 --learned-run /runpod-volume/results/run-35414623339/learned_qwen3_8b_seed20260916 --run-id trajectories11_qwen3_8b_seed20260916
```

8B: 7.1 min (the subspace stage 2.6), determinism check passed; 10
tasks, hand-over read points 21–27 of 36.

- **Rank one for the natural difference, rank one to eight for the
  steered perturbation.** At the hand-over read point the pool's mean
  natural difference patched alone gives 0.88–0.98 of the effect on the
  eight lexical tasks (ceilings 0.93–1.17; arithmetic_words 0.58 against
  its per-prompt ceiling of 0.60), k90 for the patch being 1 on seven
  tasks and 2 on past_tense and present_participle. Keeping only the
  rank-1 projection of the steered perturbation retains 0.70–1.05
  (median 0.82) and rank 8 0.90–1.05 (median 0.95) against ceilings of
  0.92–1.12: k90 for keep is 1 on two tasks, 2 on two, 4 on two and 8 on
  three (last_antonym, past_tense, present_participle). The steered
  perturbation carries its effect in a few more directions of the pool
  spectrum than the natural difference needs; the spectrum itself is
  less concentrated than 0.6B's (top component 0.59–0.92 of the
  variance, median 0.71; participation ratio 1.2–2.8).
- **The task's own.** The other tasks' subspace keeps 0.04–0.28 at
  k = 1 (median 0.19) and patches −0.05 to 0.34; at k = 8, where nine
  tasks' pools span several task means, it keeps 0.08–0.82 and patches
  0.00–0.91, the high values on the two antonym tasks (0.61 and 0.81
  kept, 0.91 patched) and number_to_words (0.66 patched), which share a
  subspace with their relatives. The background subspace keeps −0.22 to
  0.27 and patches at most 0.26 (random 0.00–0.06; arithmetic_words'
  random keep is 0.47, its first-token effect being small).
- **At the last read point** the picture loosens: the own rank-1
  projection retains −0.09 to 0.97 (median 0.56) and rank 8 −1.34 to
  0.80 with wide intervals on the two antonym tasks, the projected
  natural difference gives 0.09–0.54 at k = 1 and 0.20–0.84 at k = 8
  against ceilings of 0.92–1.17, and the other-task and background
  subspaces are destructive (keep −0.3 to −3.0). What remains at the
  end is not captured by the pool's top eight directions.
- Add-3 is the no-op it always is at the query token.

```
# workflow run 35464728409 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050661831/pilot6_qwen3_1.7b_seed20260916 --learned-run /runpod-volume/results/run-35414702823/learned_qwen3_1.7b_seed20260916 --run-id trajectories11_qwen3_1.7b_seed20260916
```

1.7B: 11.1 min (the subspace stage 2.4), determinism check passed; 9
tasks, hand-over read points 18–22 of 28 (last_antonym: the last read
point, as in iteration 10).

- **Rank one to two.** At the hand-over read point the mean natural
  difference patched alone gives 0.86–0.95 of the effect on the seven
  lexical tasks with a hand-over before the end (ceilings 0.90–1.18),
  k90 for the patch 1 on five of them, 2 on present_participle and 4 on
  plural; keeping the rank-1 projection of the steered perturbation
  retains 0.62–1.00 (median 0.88) and rank 8 0.85–1.00 (median 0.96),
  k90 for keep 1 on two tasks, 2 on two, and not reached at 8 on
  antonym (0.85 against 0.98), plural (0.88 against 0.93) and
  present_participle (0.86 against 0.94). The top component explains
  0.68–0.93 of the pool variance (participation ratio 1.2–2.1) and has
  cosine 0.99–1.00 with the held-out mean difference.
- **The task's own**, with the relatives sharing: the other tasks'
  subspace keeps −0.01 to 0.34 at k = 1 and −0.15 to 0.50 at k = 8, and
  patches −0.12 to 0.35 at k = 1; at k = 8 antonym's 0.87 and (at half
  depth) last_antonym's 0.84 come from each other's pool, the pair being
  the two tasks whose answers coincide. The background keeps −0.41 to
  0.39 and patches at most 0.24; random subspaces 0.00–0.14.
- **last_antonym** keeps its own route: the natural difference on the
  own rank-1 subspace gives 0.81 of the effect at half depth (ceiling
  0.82) while every keep edit there retains 0.55–0.70, and at the last
  read point, where the per-prompt keep first reaches 0.9, no subspace
  of rank 8 carries anything (own keep 0.05 to −0.73, patch 0.13–0.26).
- **At the last read point** the own rank-1 projection retains
  0.55–0.87 and rank 8 0.62–0.94, the projected natural difference
  0.47–0.72 at k = 1 and 0.55–0.83 at k = 8 against ceilings of
  0.94–1.20; the other-task and background subspaces are destructive
  (keep −0.1 to −2.4).

```
# workflow run 35465035832 (push-triggered request), RTX 4090 (ADA_24), EUR-NO-1
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35051925394/pilot6_qwen3_4b_seed20260916 --learned-run /runpod-volume/results/run-35420419658/learned_qwen3_4b_seed20260916 --run-id trajectories11_qwen3_4b_seed20260916
```

4B: 21.7 min (the subspace stage 5.1), determinism check passed; 10
tasks, hand-over read points 22–27 of 36.

- **Rank one for the natural difference, one to four for the steered
  perturbation.** At the hand-over read point the mean natural
  difference patched alone gives 0.82–1.12 of the effect on the eight
  lexical tasks (ceilings 0.91–1.24; arithmetic_words 0.52 against its
  ceiling of 0.54), k90 for the patch 1 on five tasks, 2 on past_tense
  and present_participle, 8 on last_antonym. The rank-1 projection of
  the steered perturbation retains 0.70–0.99 (median 0.86), rank 8
  0.87–0.99 (median 0.94), ceilings 0.91–1.00; k90 for keep 1 on four
  tasks, 2 on two, 4 on antonym and plural, not reached on last_antonym
  (0.87 against 1.00). Top component 0.69–0.93 of the pool variance,
  participation ratio 1.2–2.1, cosine 0.99–1.00 with the held-out mean.
- **The task's own, the relatives shared.** The other tasks' subspace
  keeps at most 0.33 at k = 1 on the lexical tasks and patches at most
  0.39; at k = 8 it keeps 0.00–0.81 and patches up to 1.02 (antonym),
  0.86 (number_to_words, from arithmetic_words' pool) and 0.74
  (last_antonym): the subspace of nine other tasks contains a relative's
  mean. On arithmetic_words every subspace keeps at least the 0.45 the
  random ones keep (its first-token effect is small), and the other
  tasks' reaches 0.95 at k = 8 through number_to_words. The background
  keeps at most 0.12 at k = 1 on the lexical tasks (0.42 on
  number_to_words at k = 8) and patches at most 0.27.
- **At the last read point** the own rank-1 projection retains
  0.02–0.85 and rank 8 −0.55 to 0.94, the projected natural difference
  0.15–0.59 at k = 1 and 0.29–0.83 at k = 8 (ceilings 0.89–1.30); the
  other-task and background subspaces are destructive.

Iteration 11 on the four models, in one paragraph. **What replaces the
control is a rank-one, task-specific signal at the hand-over depth.**
On the 31 lexical task-model pairs with a hand-over before the last
read point, the mean natural difference of prompts that were never
evaluated, patched alone into the unsteered run at that read point,
gives 0.82–1.12 of the learned vector's held-out effect (per-prompt
ceilings 0.90–1.24; random rank-1 subspaces 0.00), and it is the top
principal component of the pool's differences (0.59–0.95 of their
variance, cosine 0.99–1.00 with the held-out mean). Keeping only the
projection of the steered perturbation onto that one direction retains
0.62–1.00 (medians 0.82–0.93 on the four models), and onto the top
eight 0.81–1.05 (medians 0.94–0.97), against ceilings of 0.91–1.20:
the steered perturbation spends a few more of the pool's directions
than the natural difference needs (k90 for keep 1–2 on most tasks of
0.6B and 1.7B, 1–8 on 4B and 8B, the higher ranks where the top
component explains less of the variance), but nothing beyond the top
eight, since the per-prompt ceiling is reached there. **The signal is
the task's own.** The top-k subspace of the other tasks' pools keeps at
most a third of the effect and patches at most 0.44 at rank one on
every model, although its top component overlaps the task's own at
0.6–0.9 of the energy: the direction the tasks share is most of the
mean difference by norm and little of its effect, as D28 found for the
vectors at the injection layer. At rank eight the other-task subspace
carries the effect only where a relative sits in the pool (the two
antonym tasks, number_to_words with arithmetic_words, 0.6–1.0), which
is the relative's own mean, not a shared in-context subspace. The
background's top-k subspace keeps and patches at most 0.1–0.4, mostly
nothing. **Before the hand-over** the rank-1 signal carries what the
per-prompt direction carries (the shortfall is the depth); **at the
last read point** it carries 0.5–0.9 of the effect and the projected
natural difference 0.5–0.8, with the other-task and background
subspaces destructive: what remains at the end is higher in rank than
the pool's top eight directions, and this, with add-3's no-op and
last_antonym's own route on 1.7B, is where the rank-one description
stops. So the causal dimensionality of the control's replacement is
one at the depth where the hand-over happens, in the causal sense the
geometric effective dimensionality could not deliver, and dimensional
expansion in the causal sense appears only after the hand-over.
**Sufficient before necessary, by a margin that grows with size** (the
removal rows, read after the entry above was written). Removing the
own rank-1 direction from the steered perturbation at the hand-over
read point leaves a median of −0.23 of the effect on 0.6B (−0.93 to
0.44 over the lexical tasks) and −0.09 on 1.7B (−0.72 to 0.57), a
deficit: without that direction the rest of the perturbation is useless
or harmful; 0.39 on 4B (−0.16 to 0.99); and 0.96 on 8B (0.51–1.00),
where removing the top eight directions still leaves 0.85 (0.22–0.94).
Random subspaces removed leave 1.00 everywhere. So on the small models
the rank-1 direction is the unique carrier at the hand-over, both
sufficient and necessary, while on 8B at the read point where it has
become sufficient the complement of the pool's top eight directions is
sufficient too: the effect is redundantly encoded there and the
complement loses its power only over the following blocks (the
per-prompt removal curves of iteration 10 fall 4–8 read points after
the keep curves rise on 8B, at the same read point on 0.6B). At the last
read point removal of the rank-1 direction leaves −0.11, 0.11, 0.36 and
0.66 (medians; 0.6B, 1.7B, 4B, 8B) and of the top eight 0.42, 0.38,
0.63 and 0.73. The distance between the sufficiency and the necessity
read points is the quantity that separates "a low-dimensional
replacement signal" from "a distributed re-encoding": zero on the small
models, several blocks on 8B. The subspace removal at every patch-grid
read point would locate it per task; a component attribution inside the
hand-over band (which heads and MLPs write the direction) is not done.
**Cost.** 5.1, 11.1, 21.7 and 7.1 min (0.6B, 1.7B, 4B, 8B), about $1.30
of worker time for the four models; the subspace stage was 2–5 min of
each.

### D35: two further model families (OLMo 3 7B, Gemma 4 12B)

Code: the backend reads the architecture (text config, block list,
final norm, per-layer head widths, the final-logit soft-cap, the
tokenizer's prompt prefix); tiny random models of the three families
run the backend tests; `--stop-after {fewshot,extraction}` on
`validate`/`pilot` for a smoke run; configs `pilot_olmo3_7b.yaml`,
`learned_olmo3_7b.yaml`, `pilot_gemma4_12b.yaml`, `learned_gemma4_12b.yaml`
(the 8B configs with the model name replaced; OLMo 3 at batch 32).

**OLMo 3 7B smoke run** (workflow run 35529137780, `validate
--stop-after fewshot`, RTX 4090, 69 s after a 45 s load; batch 128):
the model loads as `Olmo3ForCausalLM` with 32 layers, 32 heads of 128,
vocabulary 100278, no prompt prefix, no soft-cap. Few-shot accuracy
(zero-shot in brackets): antonym 0.885 (0.000), plural 0.995 (0.010),
past_tense 0.953 (0.057), present_participle 1.000 (0.000), singular
1.000 (0.031), number_to_words 1.000 (0.224), last_antonym 0.646
(0.000), arithmetic_words 0.000 (0.000; fails the gate, as on the
smaller Qwen3 models). `arithmetic` and `uppercase` ran out of memory
at batch 128 (the card's 23.5 GB were full after the weights' 13.6 GiB
and the multi-head attention activations of 128 eight-shot prompts);
`uppercase` also drops items whose target exceeds four tokens under
the GPT-2-style tokenizer (`phenomenon`). The batch was set to 32 for
this family (D35) and the full pilot requested.

**Gemma 4 12B smoke run** (workflow run 35529434599, `validate
--stop-after fewshot`, H100 in US-CA-2, 114 s after a 69 s load on a
cold cache; batch 128, 36 GB reserved): the checkpoint loads as
`Gemma4UnifiedForConditionalGeneration` (12.0 B parameters including
the unused vision and audio towers) with 48 layers, 16 heads of 256 on
the 40 local layers and of 512 on the 8 global ones (every sixth
layer), soft-cap 30, BOS prefix, vocabulary 262144. Few-shot accuracy
(zero-shot 0.000 on every task): antonym 0.865, plural 1.000,
past_tense 0.948, arithmetic 0.995, present_participle 1.000, singular
1.000, uppercase 0.896, number_to_words 1.000, last_antonym 0.802,
arithmetic_words 0.609; all ten pass the gate (`uppercase` drops
`phenomenon` at five tokens, as OLMo 3 does). The full pilot was
requested on the 80 GB tier.

## Not yet run / known limitations

- Iteration 4b has run once on each of the four models, seed 20260907
  (8B on an H100 in US-CA-2, the others on RTX 4090s in EUR-NO-1).
  Iteration 4 (weakest-reliable calibration, first-token ranking) is
  superseded by 4b for the head ranking and the strength. Iteration 5
  batch B has run once and iteration 6 twice (seeds 20260907 and
  20260916) on all four models (batch A on 0.6B and 1.7B). With
  candidates up to 64 and the 10 % marginal-gain rule the head count is
  no longer censored on any model (8 / 16 / 32 / 32 in both seeds; the
  refused doubling gains 3–6 %); the ceiling is not extended to 128.
  Two seeds give a range, not an interval; the quantities that vary
  between them (the 1.7B layer, a few common/residual splits, held-out
  effects by up to 0.8 nats) would need more seeds before a write-up
  states them with error bars.
- The add-k family (iteration 7a, D30) has run once per model. Its
  metric (the changed digits) is not comparable with the pilot runs'
  arithmetic numbers, and its two qualifying cases (add-1 on 4B and 8B)
  sit at the metric's ceiling at the selected strength, so their
  commitment curves are censored; a run at a lower strength would be
  needed to read them. The pilot configs still carry `arithmetic` and
  `arithmetic_words` at k = 3, which cost a third of the 4B wall time
  for two guaranteed rejections under the head-mean control.
- The learned vector (iteration 7b, D31) has run once per model with the
  first step budget tried (100 steps, 0.05 of the norm; every fit
  converged in under 30). Its fits are scored on the task's scoring
  (the changed digits for add-3), which on 0.6B let the vector corrupt
  the unchanged digits, and nothing in the objective limits the damage
  to other text, which is large for a few tasks per model; a fit on all
  tokens and a damage-penalised fit are the follow-ups, and a second
  seed would show whether the held-out numbers are as stable as the
  head-mean ones.
- The trajectory comparison (iterations 8 and 9, D32, D33) has run once
  per model, on the seed-20260907 runs only; its alignment labels are
  exploratory (the peak is chosen on the same data). The patch test ran
  for the learned vector only (the head mean is a config switch away),
  at depth fractions 0.5, 0.75 and 1.0: on 4B the hand-over is complete
  by 0.5 and for add-3's later digits earlier than any edited read
  point, so an earlier fraction (12 passes per task) would locate it;
  four isotropic controls per construction and strength are few for the
  floors' quantiles; and the generic-response diagnostics are taken on
  population means (the logit lens with the mean random response), not
  per prompt. The generic-response line is closed
  (docs/GENERIC_RESPONSE.md condenses it, with the open hypotheses and
  their tests recorded and not run).
- Iteration 10 (D33 amended) has run once per model on seed 20260916,
  with the learned-vector protocol at that seed; its shared quantities
  are replicated against iteration 9's seed at the same injection layer,
  but the additions (the quarter strength, the neighbouring layers, the
  patch grid at eighths, the head mean's patch rows, the per-block
  writing) have one seed. The per-block writing uses a reference that
  moves with depth, so its shares do not telescope and its entropy
  ratios say little; a fixed-reference version (the natural difference
  at the hand-over read point, per example) would be the measure to
  preregister if the block localisation is to be a core quantity. The
  learned vector's selected layer is not reproducible across seeds
  (30 of 37 tasks differ) because the candidates are equivalent; a
  write-up should report per layer, not per selection.
- Iteration 11 (D34) has run once per model on seed 20260916, with the
  subspace rank capped at 8 by decision and the fitting pools at 128
  prompts; the hand-over read point is read off the same run's patch
  grid (the per-prompt keep on the held-out prompts, not the subspace
  results), a mild selection on the evaluation pool that a fixed
  fraction avoids. The other-task subspace pools every other task
  equally, so at rank 8 it reflects whichever relative is in the pool;
  a per-task-pair version would separate "shared" from "the relative's
  own". The answer-content reading (a subspace fitted from the answer
  unembedding directions) was not run.
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
uv run pytest                                                           # 206 tests (the trajectories smoke run covers D32–D34; the backend tests run on the three toy families, D35)
# GPU runs: edit .github/gpu-run.yaml (command + a new `request` label), commit, push; the run-gpu
# workflow triggers on the push (README, "Run on GPUs"). One run per push; the ci environment runs
# them one at a time (8B goes through the ci-8b environment in US-CA-2). Then:
gh run list --workflow=run-gpu.yml --branch fable --limit 1 && gh run watch <RUN_ID>
gh run download <RUN_ID> --dir results/remote/<name>
uv run --with boto3 python scripts/runpod_log.py <RUN_ID> --follow      # the live worker log
# iteration 10 (D33 amended: quarter strength, the neighbouring candidate layers, the patch test at every compared
# layer on eighths of the depth for the learned and the head-mean vector, per-block writing), on a second seed,
# as run (workflow runs in the iteration-10 entry): first the learned-vector protocol (D31) at seed 20260916 on
# each model (the head-mean runs of that seed are iteration 6's), then the comparison with --seed 20260916:
uv run directions pilot --config configs/learned_qwen3_0.6b.yaml --seed 20260916 --run-id learned_qwen3_0.6b_seed20260916
uv run directions pilot --config configs/learned_qwen3_1.7b.yaml --seed 20260916 --run-id learned_qwen3_1.7b_seed20260916
uv run directions pilot --config configs/learned_qwen3_4b.yaml --seed 20260916 --run-id learned_qwen3_4b_seed20260916
uv run directions pilot --config configs/learned_qwen3_8b.yaml --seed 20260916 --run-id learned_qwen3_8b_seed20260916   # ci-8b, US-CA-2
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050574186/pilot6_qwen3_0.6b_seed20260916 --learned-run /runpod-volume/results/run-35414652608/learned_qwen3_0.6b_seed20260916 --run-id trajectories10_qwen3_0.6b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050661831/pilot6_qwen3_1.7b_seed20260916 --learned-run /runpod-volume/results/run-35414702823/learned_qwen3_1.7b_seed20260916 --run-id trajectories10_qwen3_1.7b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35051925394/pilot6_qwen3_4b_seed20260916 --learned-run /runpod-volume/results/run-35420419658/learned_qwen3_4b_seed20260916 --run-id trajectories10_qwen3_4b_seed20260916
uv run directions trajectories --config configs/trajectories.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050604985/pilot6_qwen3_8b_seed20260916 --learned-run /runpod-volume/results/run-35414623339/learned_qwen3_8b_seed20260916 --run-id trajectories10_qwen3_8b_seed20260916
# iteration 11 (D34: the causal dimensionality of the shared component), the lean config on the seed-20260916 runs of
# iteration 10 (canonical strength, primary layer, the learned vector's patch grid, the subspace test):
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050574186/pilot6_qwen3_0.6b_seed20260916 --learned-run /runpod-volume/results/run-35414652608/learned_qwen3_0.6b_seed20260916 --run-id trajectories11_qwen3_0.6b_seed20260916
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050661831/pilot6_qwen3_1.7b_seed20260916 --learned-run /runpod-volume/results/run-35414702823/learned_qwen3_1.7b_seed20260916 --run-id trajectories11_qwen3_1.7b_seed20260916
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35051925394/pilot6_qwen3_4b_seed20260916 --learned-run /runpod-volume/results/run-35420419658/learned_qwen3_4b_seed20260916 --run-id trajectories11_qwen3_4b_seed20260916
uv run directions trajectories --config configs/trajectories_subspace.yaml --seed 20260916 --fv-run /runpod-volume/results/run-35050604985/pilot6_qwen3_8b_seed20260916 --learned-run /runpod-volume/results/run-35414623339/learned_qwen3_8b_seed20260916 --run-id trajectories11_qwen3_8b_seed20260916
# D35 (two further families): a smoke run per model first (load, targets, few-shot gates; OLMo 3 on ci/ADA_24/EUR-NO-1,
# Gemma 4 on ci-8b/ADA_80_PRO/US-CA-2), then the pilot, the learned vector and the trajectory comparison:
uv run directions validate --config configs/pilot_olmo3_7b.yaml --stop-after fewshot --run-id smoke_olmo3_7b
uv run directions validate --config configs/pilot_gemma4_12b.yaml --stop-after fewshot --run-id smoke_gemma4_12b
uv run directions pilot --config configs/pilot_olmo3_7b.yaml --run-id pilot6_olmo3_7b_seed20260907
uv run directions pilot --config configs/learned_olmo3_7b.yaml --run-id learned_olmo3_7b_seed20260907
uv run directions trajectories --config configs/trajectories.yaml --fv-run /runpod-volume/results/run-<pilot>/pilot6_olmo3_7b_seed20260907 --learned-run /runpod-volume/results/run-<learned>/learned_olmo3_7b_seed20260907 --run-id trajectories_olmo3_7b_seed20260907
# the pilot protocols, for reference (head-mean control: pilot_*.yaml; learned vector: learned_*.yaml; add-k: arith_*.yaml):
uv run directions pilot --config configs/learned_qwen3_0.6b.yaml --run-id learned_qwen3_0.6b_seed20260907
uv run directions aggregate results/remote/<a>/... results/remote/<b>/... --out results/aggregate_<model>.json   # multi-seed summary
uv run directions compare results/<run_a> results/<run_b>                # diff two runs (only after a change of the numerical path, D23)
```

Suggested next steps (2026-09-18; the list of two days ago with the state of each):

1. **Strength dependence** (done for the trajectory comparison): canonical,
   half and quarter strength on all four models (iterations 9 and 10);
   quarter strength is below the working range, half strength the point
   of highest alignment. Double strength is not planned (the calibration
   grids were still rising at their ceiling; the canonical strength
   already overshoots on 1.7B and 8B).
2. **Commitment at every candidate injection layer** (done for the
   primary and its neighbouring candidates, iteration 10): the hand-over
   sits at a fixed read point of the stack for every injection layer.
   The two candidate layers farthest from the selection are not covered
   on tasks whose selection sits at an end of the candidate range; a
   `layers` list would cover all four at 15 passes per layer and factor.
3. **Useful dimensionality of the perturbation** (done, iteration 11,
   D34): at the hand-over depth the effect is carried by one direction
   across prompts, the task's mean natural difference, the task's own
   rather than a shared in-context subspace; the steered perturbation
   needs at most the pool's top eight directions. Not measured: whether
   a rank-k operator carries what one vector cannot (add-k, iteration
   7a), and the rank of what remains at the last read point beyond
   eight.
4. **Mechanism** (partial): the answer-direction variant, the logit lens
   of the generic response and the first-token readouts exist; a logit
   lens of the common direction (D28) and of the learned vector at the
   last read point, and a head attribution downstream of the injection,
   do not. The unembedding code is in the backend.
5. **Another model family** (in progress, D35): OLMo 3 7B and Gemma 4
   12B under the unchanged protocol (`pilot_olmo3_7b.yaml`,
   `pilot_gemma4_12b.yaml` and the `learned_*` counterparts); the
   backend reads the architecture (text config, block list, final norm,
   per-layer head widths, the final-logit soft-cap) and the test suite
   runs on tiny random models of all three families. GPU smoke runs
   (`--stop-after fewshot`) precede the full runs; see the D35 entry
   below once run.
6. **Hardening** (partial): the determinism check, the device geometry
   and the workflow are in place; the learned-vector run and the
   trajectory comparison now have two seeds (iterations 7b and 10) and
   replicate at the same injection layer; iteration 7a (add-k) has one
   seed, and two seeds give a range, not an interval.

The generic-response line is closed (docs/GENERIC_RESPONSE.md).
