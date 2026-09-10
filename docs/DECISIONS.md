# Decisions

Methodological and implementation choices that are not fully determined by
`docs/EXPERIMENT.md`, with the reason for each. Change an entry here before
changing the behaviour it describes.

## 2026-09-07 — Initial pipeline implementation

### D1. Prompt format and target tokenisation

`Q: {input}\nA:` for the query, ` {output}` for the target, demonstrations
separated by a blank line. The prompt and the target are tokenised
*separately* (the target with a leading space and without special tokens) and
the token ids are concatenated, so that the target tokenisation is identical
across few-shot, zero-shot, steered and unsteered conditions. This is the
tokenisation that the `max_target_tokens` item filter sees. All template
strings are config parameters (`prompt.*`).

### D2. Decision metric and the teacher-forced pass

All behavioural metrics come from one teacher-forced pass over
`prompt + target`. Logits are computed only at the target positions (the
final-norm hidden states are gathered before the unembedding), which keeps
memory small for a 152k vocabulary. Exact match requires the argmax at *every*
target position to equal the target token.

### D3. Extraction reads every layer in one pass; candidates by depth fraction

Residuals at the final query token are gathered at all `L+1` read points in a
single forward pass per prompt, so extraction at every candidate layer costs
one pass per (seed, pair). Candidate layers are
`round(fraction * L)` for the configured depth fractions (default 20–60%),
deduplicated. For Qwen3-0.6B (`L = 28`) this gives layers 6, 8, 11, 14, 17.
Stability at *every* read point is also written (exploratory diagnostic).

### D4. Stability gate filters candidate layers before calibration

"Require cross-seed direction stability at the candidate layer" is implemented
as: only candidate layers whose minimum pairwise `|cos|` across seed PC1s is
`>= qualification.min_stability` (default 0.8) enter calibration. If none
passes, the task is rejected at the stability stage. Rationale: calibration
on an unstable direction would be selecting an artefact of one demonstration
sample.

### D5. Calibration selection rule, operationalised

At each stable layer (ascending) and each grid strength (ascending) the real
direction is scored on the calibration pool. A point is *reliable* if

1. one-sided paired bootstrap `p <= calibration.bootstrap_alpha` (0.05) for
   mean Δ(log p per target token) > 0 (with +1 continuity correction);
2. mean Δ `>= calibration.min_improvement` (0.05 nats/token);
3. its mean Δ exceeds the mean Δ of `n_random_screen` (8) matched random
   directions (same layer, same absolute norm; alternating isotropic and
   orthogonal) with empirical `p_upper <= random_screen_max_p` (0.2). With 8
   controls the minimum attainable p is 1/9 ≈ 0.11, so the real direction
   must beat all eight.

The selected point is the first reliable point in this order (earliest layer,
then smallest strength). The random screen is only run at points that pass
(1)–(2) while the layer can still be, or is, the selected layer; the full real
grid is still computed for every stable layer for the calibration figure.

Grid extension: after a layer's grid is exhausted, if the best point (max mean
Δ) sits at the upper edge the next strength `rho * extension_factor` (1.25) is
appended, up to `max_rho` (2.0). Extended points are flagged in the output.

### D6. Held-out qualification thresholds

On the evaluation pool: paired bootstrap `p <= qualification.steering_alpha`
(0.05) and mean Δ > 0; and empirical `p_upper <= random_control_max_p` (0.1)
against the 16 matched random controls (minimum attainable 1/17 ≈ 0.059, so the
real direction must beat all 16). Thresholds are configuration parameters.

### D7. Per-example quantities are undefined below the intervention layer

`δ_l = 0` exactly for `l < l*` (the two passes are bit-identical up to the
intervention; the pipeline records the maximum |δ| below `l*` as a check). All
per-example and matrix metrics are reported as null there, and `d_eff`, `d90`
and the centred `N_l` are null *at* `l*` as specified. The observed centred
spectrum at `l*` is stored separately as the numerical-noise-floor diagnostic.

### D8. Conversion mass and profile labels

The cross-task analysis defines the conversion mass of block `l` as the median
over held-out examples of `||b_l(x)||`, normalised over downstream blocks. From
it: entropy ratio, dominant block and share, centre of mass in normalised
depth, and early-conversion share. Qualitative labels are assigned by fixed
rules with thresholds in `analysis.*` (see `src/directions/analysis.py`);
several labels can co-occur and the rules are deliberately simple. They are a
summary of the quantities, not a replacement for them.

### D9. Random controls in the layerwise stage reuse the steering-stage passes

The 16 matched random controls used for the held-out behavioural test are the
same forward passes (with residual capture) used for the random-control null
of every layerwise metric. The real direction's z and empirical p are
computed per layer against these 16 profiles.

### D10. Block ablation (exploratory)

Blocks are chosen by the largest median conversion `C_l` (or, by config, the
largest uncentred `N_l`). Necessity subtracts the per-example `b_l(x)` at
`resid[l+1]` inside the steered run; sufficiency adds it to an unsteered run.
Both use the per-example vectors captured in the layerwise stage. Fractions
lost/reproduced are relative to the mean steering improvement on the decision
metric. Written to `exploratory/`.

### D11. Smoke mode (`qualification.enforce: false`)

The integration path uses a tiny random model that does no task. To exercise
every stage, gates are computed and logged but do not block; when calibration
finds no reliable point the best grid point is used and flagged
`fallback: true`. Smoke runs are never scientific results. Real configs set
`enforce: true`.

### D12. Reproducibility

All sub-seeds derive from `seed` through SHA-256 (`seeds.py`); the seed table
is written to `metadata.json`. `directions compare <run_a> <run_b>` diffs every
JSON output of two runs ignoring only run id, timestamps, timings and argv.
Proven for the toy path in `tests/test_integration.py`; see `STATUS.md` for the
GPU runs.

### D13. Which blocks the causal ablation targets

"Layers with unusually large conversion or new-subspace creation" is
ambiguous between *largest value* and *largest relative to the random-control
null*. `exploratory.block_ablation.rank_by` selects: `value` (default, the
preregistered pilot configs) ranks blocks by the median metric itself, which
in practice always includes the intervention block (its `C_l` is largest
because `δ_{l*}` is the pure injected direction); `z` ranks by the per-block
z-score against the matched random controls. The ablation is exploratory, so
both are available and the choice is recorded in `block_ablation.json`.

## 2026-09-07 — Strengthening the null and the power (second iteration)

### D14. Evaluation pool tripled to 192; tasks with larger item lists; five more tasks

The initial pilot's 64 held-out examples bound the centred `D_l` to rank
≤ 63, so `d_eff`, `d90` and `N_l` were limited by `n` rather than by the
model (`N_l` ≈ 0.95 was near saturation). `data.n_evaluation` is now 192,
which needs ≥ 320 usable items per task. The four word lists were extended
(antonym 435, plural 435, past tense 439, English→French 488 unique inputs;
`src/directions/data_extra.py`) and five tasks were added: `present_participle`
(rule-based `-ing` with an explicit exception table, 439 items), `singular`
(inverse of plural, identical forms excluded, 428), `uppercase` (1209),
`number_to_words` (0–999, per-task `max_target_tokens: 6` because the targets
are longer), and `add_two` (two-operand addition over all ordered pairs in
[2, 59], 3364). A per-task `max_target_tokens` override was added to
`TaskConfig` for the same reason. The extraction and calibration pools stay
at 64.

### D15. Controls by kind: 64 preregistered random controls and three structured nulls

`evaluation.controls` is a mapping kind → count; every control is injected at
the selected layer and token with the selected absolute norm. The
*preregistered* null (the qualification gate and the primary per-layer z / p)
is `gate_kinds = [isotropic, orthogonal]` with 32 + 32 = 64 directions
(minimum attainable empirical p = 1/65; the gate threshold
`random_control_max_p` is 0.05, i.e. the real direction must beat at least
61 of 64). The calibration screen uses 16 (must beat all 16 at p ≤ 0.1).

Three structured nulls are reported alongside, each with its own per-layer z
and empirical p, and are never used for the gate:

* `covariance` (32): `u ∝ H_cᵀ g`, `g ~ N(0, I)`, with `H_c` the centred
  baseline residuals of the evaluation pool at the intervention layer, so the
  direction lies in the subspace the residual stream actually occupies;
* `other_task` (up to 9): the pooled control direction of every other task
  that reached extraction, at the same layer — a real, behaviourally
  meaningful direction with the wrong content;
* `demo_variation` (8): PC1 (same uncentred rule as the control) of
  `h(p_i^{+,a}) − h(p_i^{+,b})`, the difference between two *correct*
  demonstration samples for the same query. This keeps the prompt-difference
  structure of the extraction without any task contrast. A "shuffled-pair"
  null (re-pairing positive and permuted prompts across queries) was
  considered and rejected: the uncentred PC1 depends only on the Gram matrix
  `Σ dᵢdᵢᵀ`, which is invariant to re-pairing the mean and to row sign flips,
  so it would recover the same direction.

Because `other_task` needs every task's direction before any task is
measured, the pipeline runs in two phases (extraction for all tasks, then
calibration/evaluation/layerwise per task). Control profiles are computed
without bootstrap CIs (medians only) and their residuals are discarded
immediately; spectra use the `n × n` Gram matrix when `n < d`, which matches
the full SVD to 1e-14 (`tests/test_controls.py`). A z-score against a null
with zero spread is reported as null rather than ±∞.

### D16. Multi-seed replication and aggregation

`--seed N` overrides the run seed from the command line (recorded in the
resolved config and in `metadata.json`), and `directions aggregate <runs...>`
summarises several runs per task: qualification rate, gates failed,
selections, label counts, and mean ± sd of the cross-task scalars and of the
per-kind z-scores. Seeds change the pool partition, demonstrations,
derangements, random controls and bootstraps together.


## 2026-09-08 — Third iteration: direction-specific readouts, paired-excess gate, task set

### D17. Direction-specific readouts in the layerwise stage

Iteration 2 (four Qwen3 models, twelve runs) established that every geometric
metric of the control direction's perturbation — gain, `d_eff`, `N_l`,
alignment with the injected vector — is reproduced by any in-distribution
direction of the same norm at the same layer, while the *behavioural* effect
is direction-specific. The geometric metrics cannot see task content, so two
readouts that can were added as core per-example metrics (docs/EXPERIMENT.md,
"Direction-specific readouts"):

* `task_alignment` `T_l(x) = cos(δ_l(x), v_{t,l})`, with `v_{t,l}` the pooled
  PC1 of the same extraction differences at read point `l` (already computed
  for every layer and stored as `all_layers` in `directions.npz`);
* `gradient_alignment` `Γ_l(x) = cos(δ_l(x), ∇_{h_l} log p(target | x))`,
  the gradient taken on the baseline (unsteered) zero-shot pass at the query
  token, one forward+backward per task with frozen parameters
  (`ModelBackend.gradients`; verified against central finite differences on
  the toy model, `tests/test_model.py`). `δ_l · ∇ log p` (first-order
  predicted Δ log p) is stored as an array but not compared.

Choices: (i) signed cosines, not absolute values — `v_{t,l}` has a meaningful
sign (aligned with the mean few-shot-minus-permuted difference) and a
perturbation pointing *down* the gradient is a different finding from one
that is merely unaligned; (ii) the gradient is that of the *baseline* pass,
so every control's profile is scored against the same field and the readout
is a property of the perturbation, not of the steered state; (iii) the
gradient of the summed target log-probability (not per token) — cosines are
invariant to the per-example scale; (iv) bf16 backward: the gradient carries
the same rounding as the forward differences it is compared with, and both
are float32 at the query token. Both readouts are undefined below `l*`, are
summarised and compared per layer against all five control kinds exactly
like the existing metrics (per-layer z and empirical p; `_z_means`), enter
the cross-task table, aggregation (`task_alignment_z_mean`,
`gradient_alignment_z_mean`, downstream means) and a new figure
(`<task>_readouts.png`). No profile label depends on them yet; the
qualitative labels stay the preregistered iteration-2 set. The per-kind
z-mean of `task_alignment` excludes the intervention layer: there the real
direction is 1 by construction while orthogonal controls are exactly 0 and
isotropic ones ~1/√d with near-zero spread, so the z at `l*` is degenerate
(hundreds) and would dominate the average; the downstream layers are the
question. `gradient_alignment` at `l*` — cos between the injected direction
and the gradient — is well defined for every control and stays in.


### D18. The random-control gate is a paired excess test, not a rank

Iteration 2's gate ("beat ≥ 61 of the 64 gate controls", empirical
`p ≤ 0.05`) turned out to sit on the boundary of the random spread: every
held-out steering effect was significant on its own (`p = 0.0005`) while the
real direction typically beat 60–62 of 64 controls, so which tasks qualified
flipped with the seed (31/55 task-seed pairs on 0.6B/1.7B, 12/30 on 4B,
14/27 on 8B; no task passed in every seed on 4B). The rank rule asks the
real direction to be an *outlier among individual random directions*, which
conflates the hypothesis of interest with the sampling spread of the
controls themselves and with the example-level noise that the paired design
was introduced to remove.

The gate now tests the hypothesis directly: **does the real direction
improve the decision metric more than a matched random direction does on
average?** `paired_excess_test` (`src/directions/stats.py`) takes the
per-example improvements `Δ_real(x)` and `Δ_c(x)` of every gate control on
the same evaluation examples and computes the paired mean excess
`mean_x[Δ_real(x) − mean_c Δ_c(x)]`. Its null distribution is bootstrapped
hierarchically — every replicate resamples the examples (paired across the
real direction and all controls) *and* the controls — so both noise sources
enter the one-sided p-value (`+1` continuity correction, `n_boot = 2000`).
Qualification requires `p ≤ qualification.random_control_max_p` (0.05).

The gate controls are `qualification.gate_control_kinds =
[isotropic, orthogonal, covariance]` (32 + 32 + 32 = 96 directions).
Including the covariance-matched directions makes the behavioural null an
in-distribution one — iteration 2 showed isotropic directions are out of
distribution for the residual geometry and steer less than covariance-matched
ones — so the real direction must beat the average *in-distribution* random
direction of the same norm. The `other_task` and `demo_variation` directions
stay structured nulls (reported, never gating). The layerwise metrics keep
their preregistered primary null (`evaluation.gate_kinds =
[isotropic, orthogonal]`) so that per-layer z-scores and labels remain
comparable with iteration 2; the covariance-matched comparison is reported
per kind as before.

The calibration screen uses the same test on the calibration pool
(`calibration.screen_test: paired_excess`, 16 controls, `p ≤ 0.05`); the old
"beat all 16" rank rule (minimum attainable `p = 1/17`) is retained as
`screen_test: rank` / `gate_test: rank` for re-running the iteration-2
protocol. Per-example control improvements are now written to
`evaluation.json` (`per_example.controls_logprob_per_token_diff`) so the
gate can be recomputed offline, and both the excess test and the rank
comparison (`random_comparison`, `by_kind_comparison`) are reported for
every control kind (`by_kind_excess`).

Consequences to expect: the paired test has far more power than the rank
rule (the example-level noise is removed and the null mean is estimated from
96 directions rather than compared with each of them), so more task/seed
pairs will qualify and qualification should become stable across seeds; the
screen is also more permissive, so calibration will tend to select earlier
layers and smaller strengths (the selection rule is unchanged: first reliable
point). Both effects are intended.


### D19. Task set: drop `add_two` and `en_fr`, add two composite tasks

Dropped: `add_two` (two-operand addition) is solved zero-shot by every model
from 1.7B up (zero-shot log p/token −0.30 on 1.7B, −0.51 on 4B, −0.79 on
8B, few-shot −0.003), so calibration has no headroom and the task never
found a reliable intervention on 8B; `en_fr` never produced a reliable
calibration point on 0.6B and qualified inconsistently elsewhere. Their
builders remain in `tasks.py` (old configs still load) but the pilot configs
no longer use them.

Added, keeping ten tasks per model, two *composite* tasks in the spirit of
the function-vector composition experiments of Todd et al. (2024,
"Function Vectors in Large Language Models", ICLR; sec. 4.3 composes a
list-selection step with a lexical mapping, e.g. "Last-Antonym") and the
algorithmic list tasks of Hendel et al. (2023, "In-Context Learning Creates
Task Vectors"):

* `last_antonym`: a comma-separated list of three words → the antonym of the
  *last* one (selection + lexical mapping). The last word runs over the 435
  antonym inputs, so inputs are unique; the two distractors are drawn without
  replacement from the other antonym inputs with a fixed generator seed
  (`items_seed`, default 20260908, recorded in `params`), so the item list is
  deterministic and independent of the run seed.
* `arithmetic_words`: `n → number_to_words(n + 3)` for `n` in 0..996
  (numeric mapping followed by a lexical rendering; per-task
  `max_target_tokens: 6` as for `number_to_words`).

Both are underspecified zero-shot (zero-shot accuracy 0.000 on every model)
and harder than the single-step tasks. A feasibility screen (few-shot
accuracy on the 192-example evaluation pool of seed 20260907, one forward
pass per model, no steering) gave

| task | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| last_antonym | 0.28 | 0.59 | 0.70 | 0.80 |
| arithmetic_words | 0.02 | 0.26 | 0.80 | 0.71 |
| alphabetically_first (rejected) | 0.35 | – | – | 0.33 |

so both new tasks fail the preregistered few-shot gate (accuracy ≥ 0.5) on
0.6B, `arithmetic_words` also on 1.7B, and pass on the larger models. This is
the intended behaviour of the gate for harder tasks; the rejections are
logged. The extractive `alphabetically_first` (Todd et al.'s
"alphabetically_first") was implemented and screened but is at chance (1/3)
even on 8B, so it is not in any config (its builder is kept, tested, for
possible use with longer prompts). The screen looked only at few-shot
accuracy, never at steering or geometry, so it does not select on the
outcome variables.


## 2026-09-10 — Wall-time reduction of the pilot

### D20. Sixteen controls per random kind, batch size 128

Iteration 3 showed that the paired-excess gate (D18) has far more power than
the rank gate it replaced: every qualified task/seed pair reached the minimum
attainable `p = 0.0005` against 96 gate controls, and the gate failed only
where the excess was essentially zero. The number of random controls no
longer bounds the attainable p-value (the hierarchical bootstrap resamples
controls as well as examples, so the null mean is estimated from the pool
rather than compared with each member), and the wall time of a run is
dominated by the control profiles (STATUS.md, iteration 2). To shorten runs,
the three random-direction pools are reduced from 32 to **16 each**
(`evaluation.controls: isotropic 16, orthogonal 16, covariance 16`), so the
behavioural gate uses 48 matched directions and the primary layerwise null
32; the calibration screen (16 controls), the `other_task` (9) and
`demo_variation` (8) pools are unchanged. The forward-pass `batch_size` goes
from 32 to **128**: logits are gathered at the target positions only
(`ModelBackend._forward_scores`), so the batch's memory footprint is
dominated by the transformer activations, which fit comfortably on a 24 GB
card for every pilot model. `batch_size` does not enter the numerics of a
single example beyond the usual kernel-selection effects of a different
batch shape, so the reproducibility check (two runs of the smallest real
model from the same commit, `directions compare`) is repeated after this
change.

Consequences: the per-layer z and empirical p of the layerwise metrics
against the isotropic+orthogonal null are estimated from 32 rather than 64
directions (minimum empirical `p` 1/33 instead of 1/65); the per-kind z
against the covariance-matched null from 16 rather than 32. These are the
reported, non-gating quantities, and their standard errors grow by about
√2. The behavioural gate's p-value remains bootstrap-based and is not
bounded by the control count. Runs from this decision on are not
control-for-control comparable with iteration 3 (different random control
draws), but every gate and metric is defined identically. Runs are
recorded in `STATUS.md` with the control counts they used.


## 2026-09-10 — Fourth iteration: the canonical function vector as the control

### D21. The control direction is Todd et al.'s function vector; PC1 stays as the readout direction

`docs/EXPERIMENT.md` names, as the first follow-up after the pilot, replacing
the PCA construction of the control direction with canonical function-vector
extraction. The PCA direction (D1–D19) is this project's own construction;
the function vector of Todd et al. (2024, "Function Vectors in Large Language
Models", ICLR) is the construction the literature uses for the same object,
and the question of iteration 4 is whether *that* direction propagates the
way the PCA direction did. From this decision on, `extraction.control:
function_vector` in the pilot configs makes the function vector the
direction that is calibrated, gated, injected and measured layerwise. The
PCA direction is still extracted at every read point and reported beside it:
it remains `v_{t,l}` in the `task_alignment` readout `T_l` (D17), which now
asks how far the function vector's perturbation rotates toward the task's
contrast direction at each layer (at the injection layer `T_{l*}` is
`cos(FV, PC1_{l*})` rather than 1 by construction; the exclusion of `l*`
from the z-mean stays). The PCA direction is **not** used as a null: the
function vector's controls are the same five kinds as before, and
`function_vector.json` reports `cos(FV, PC1_l)` as a descriptive number only.

The construction, following the paper, with the choices this pipeline had
to make:

* **Mean head outputs.** For every attention head, the mean over the
  *positive* extraction prompts (correct demonstrations, the same prompts
  the PCA contrast uses) of the head's output at the final query token,
  taken before the attention output projection (`o_proj`), i.e. the head's
  own output. One mean per extraction seed, and the pooled mean over seeds.
* **Head ranking by average indirect effect.** Each head's mean output is
  patched into the *deranged-label* prompts of the extraction pairs (the
  paper's shuffled-label corruption; here the permuted prompts of the first
  `function_vector.aie_seeds` = 1 seed, 64 prompts) at the query token, one
  head at a time, and the effect is the mean change of the pipeline's
  decision metric, log p per target token. The paper measures the recovered
  probability of the correct token; the log-probability is used here so
  that the ranking speaks the same units as the calibration and the gates.
  Several heads are patched per forward pass by repeating the prompt list
  with a per-example patch (`ModelBackend.run(head_patches=...)`);
  `tests/test_function_vector.py` checks the batched path against one run
  per head, the per-head decomposition of the attention output against a
  hook on `o_proj`, and identity patches for exactness.
* **One universal head set.** `function_vector.head_selection: universal`:
  the heads are ranked by the mean indirect effect over all tasks that
  reached extraction and the top `n_heads` = 10 (the paper's default) are
  used for every task. This is the paper's construction (the heads are a
  property of the model) and keeps the per-task vectors comparable; the
  `per_task` alternative is implemented and configurable. The per-task
  effect of each selected head is reported next to its ranking effect.
* **The vector.** `FV_t = Σ_{(l,h) ∈ S} W_o^l[:, h] ā^t_{l,h}`, the selected
  heads' mean outputs mapped into the residual stream through their slices
  of the output projection (linear, no bias on Qwen3), summed. This is one
  vector per task, not one per layer; at every candidate layer the control
  is the same unit direction.
* **Strength and layer.** The paper adds the raw vector at one layer chosen
  by a sweep. Here the pipeline's calibration is kept unchanged: the unit
  direction is injected at the candidate layers with the `ρ` grid in units
  of the median residual norm, and the selection rule is the preregistered
  one. The strength the canonical (unscaled) injection would correspond to
  is reported as `natural_rho` = `||FV|| / median ||h_l||` per candidate
  layer, so a reader can see whether the calibrated strength is above or
  below the paper's.
* **Stability gate.** The cross-seed stability gate (`min_stability` 0.8)
  is applied to the function vector: the per-seed vectors use each seed's
  own mean head outputs with the shared head set. The PCA stability is
  reported (`pca_stability`) but no longer gates.
* **Structured nulls.** `other_task` controls are the other tasks' function
  vectors (same construction, same head set). The `demo_variation` null,
  which for the PCA direction is the same construction without a task
  contrast, is for the function vector the same construction without a
  task-correct signal: the function vector of deranged-label prompts (fresh
  demonstration samples and derangements, one per null seed; same heads).
  The random kinds are unchanged. No null involves the PCA direction.

Cost: the indirect effects are `L × n_heads` patched passes over 64
prompts per task (448 head patches on the 0.6B model, packed into ~230
forward batches of 128), a minute or so per task on the 0.6B model and
proportionally more on the larger ones; `timings_seconds` records it as
`head_effects:<task>`. The demonstration-variation null needs one extra
forward pass per null seed instead of two. Everything else is unchanged.

Consequences: iteration-4 results are not direction-for-direction comparable
with iteration 3 (a different control), but every gate, metric and null is
defined identically, and the PCA direction's per-layer readout gives the
bridge between the two. The change alters the numerical path, so the first
0.6B run is executed twice and diffed (`directions compare`).
