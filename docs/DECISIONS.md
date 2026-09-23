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
  head at a time, and the effect is the mean change of the **probability of
  the correct first target token**, the paper's recovered probability
  (`aie_metric: first_token_probability`; superseded by the whole-target
  probability in D22, because a target that starts with a lone space makes
  the first-token probability saturate). A first run ranked heads by the
  change of log p per target token instead (run 34541635889, superseded):
  that scale differs by task, so the lexical tasks, where the best head
  moves the deranged prompts by ~3 nats, dictated the universal set while
  arithmetic's own effects for the chosen heads were ~0.01 nats and its
  vector was 77 % aligned with the deranged-prompt null vector. The
  probability is bounded, so tasks contribute on a comparable scale, as in
  the paper; the log-probability ranking remains available as
  `aie_metric: logprob_per_token`. Several heads are patched per forward
  pass by repeating the prompt list with a per-example patch
  (`ModelBackend.run(head_patches=...)`); `tests/test_function_vector.py`
  checks the batched path against one run per head for both metrics, the
  per-head decomposition of the attention output against a hook on
  `o_proj`, and identity patches for exactness.
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

## 2026-09-11 — Iteration 4b: the answer probability, the canonical strength, GPU spectra, a run profile

### D22. Whole-target indirect effects; strength in units of the vector's norm with the canonical injection as the reference; batched spectra on the GPU; standard run instrumentation

Context (the iteration-4 0.6B run, `STATUS.md`): three things in that run
were properties of the pipeline rather than of the model.

1. **Arithmetic's head effects were all 0.000** under the first-token
   metric, and the run was read as "no head moves arithmetic". The cause is
   the tokenisation of its targets: Qwen3 tokenises ` 45` as `' '`, `'4'`,
   `'5'`, so the "first answer token" of every arithmetic target is a lone
   space, which the deranged prompts already predict with probability
   0.991. A saturated metric cannot move, whatever the heads do. (Targets
   of the lexical tasks are one token, ` cats`; `arithmetic_words` targets
   split at words, ` forty`, ` five`.)
2. **Calibration selected 5–50 × below the paper's strength.** The weakest
   reliable rule (D1) chose ρ 0.05–0.4 in units of the median residual norm
   while the canonical injection corresponds to ρ 1.7–2.5 there, and every
   grid point up to the grid's ceiling (ρ ≈ 2) was reliable with the effect
   still growing. The paper adds the raw vector; the profile the run
   measured was that of a barely-effective injection of it.
3. **The control profiles' wall time varied 30-fold** for identical work
   (30–860 s per task, 84 min for a replicate whose results were
   bit-identical): the per-layer SVDs ran in NumPy on a shared host.

Decisions:

* **Indirect-effect metric `target_probability`** (the new default of
  `extraction.function_vector.aie_metric`): the teacher-forced probability
  of the whole target, `exp Σ_j log p(y_j | ...)`. It is exactly the paper's
  first-token probability whenever the target is one token (the lexical
  tasks) and the probability of the answer, rather than of a space, when it
  is not. It stays bounded in [0, 1], so tasks contribute on a comparable
  scale to the universal ranking, which is why D21 chose a probability.
  `first_token_probability` and `logprob_per_token` remain configurable.
  Each task's `splits.json` now records how its targets tokenise
  (`target_tokenisation`: token counts, the share whose first token is
  whitespace, examples), and `extraction.json` records the deranged-prompt
  baseline of both probabilities, so the artefact is visible in the outputs
  rather than in a footnote. Consequence: the universal head set can change,
  because arithmetic now contributes to the ranking; the head ranking of
  iteration 4 is superseded.
* **Strength in units of the vector's norm, canonical reference.** For the
  function-vector configs, `calibration.strength_unit: natural` makes ρ
  multiply the direction's own norm (`‖FV_t‖`; for a PCA control it would
  be the mean-difference norm), so ρ = 1 is the paper's unscaled injection,
  with the grid 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2 (extended up to 4
  while the best point is at the edge). Within the selected layer the rule
  selects the reliable point nearest ρ = 1 in log distance
  (`calibration.reference_rho: 1.0`) instead of the weakest one: the
  canonical strength whenever it passes the same bootstrap, minimum-
  improvement and matched-random screen, and the nearest reliable strength
  otherwise. The layer rule stays the preregistered earliest reliable layer
  (`layer_rule: earliest`; `best`, the layer whose selected point improves
  most, is implemented for a paper-style layer sweep but not used). Every
  grid point and the selection also record `rho_layer_norm` = α / median
  ‖h_l‖, the D1 unit, and `qualification.json` records
  `alpha_over_natural_norm` next to `natural_rho`, so the two protocols stay
  comparable. The strength-robustness stage now profiles the **weakest**
  reliable strength beside the middle and strongest ones, so the profile at
  the D1-style strength is still measured, as an exploratory output. The
  PCA-control configs and the defaults are unchanged (`layer_norm`, weakest
  reliable), so iteration-3 runs are reproducible as they were.
* **The profile arrays on the model's device.** `compute_profile` computes
  every array of a profile in float64 torch on the CUDA device when one is
  set (`geometry.set_linalg_device`, done by the pipeline for a CUDA model):
  the residuals go to the device once, the per-example metrics, the
  direction readouts, the block responses and the spectra (one batched
  `torch.linalg.eigh` of the Gram matrices of all read points from the
  intervention layer on, per centring; right singular vectors recovered from
  the eigenvectors) are computed there, and every array comes back in one
  transfer. With no CUDA device the NumPy geometry runs unchanged, as the
  reference: `tests/test_geometry.py` checks the device path against it
  array by array (run on the CPU torch device) and the batched spectra
  against the per-matrix ones. `metadata.json/linalg_device` records which
  path ran. The first version of this change moved only the spectra; the
  run that followed (34615062827) still spent 20–826 s per task on the
  control profiles with CPU seconds equal to wall seconds, and
  `scripts/profile_bench.py` on the same worker class then measured 0.6 s
  (NumPy) and 0.2 s (spectra on the GPU) per full profile on synthetic and
  on real residuals alike, i.e. 13–40 s per task: the slow phases are not
  the decompositions but intermittent host slowness during which the
  process holds one core, which the whole-profile device path exposes to
  less (a few milliseconds of host arithmetic per profile) and which the
  profiler now records (below). Determinism on the GPU is verified the usual
  way: the first run is replicated and diffed.
* **Run instrumentation.** `directions.profiling.Profiler` replaces the
  ad-hoc timers: every stage of every task runs inside a named section
  (`model_load`, `data`/`fewshot`/`extraction`/`head_effects`/
  `demo_variation`/`prepare` per task, `function_vectors`, `calibration`/
  `controls_setup`/`controls_forward`/`controls_profiles`/`layerwise`/
  `strength_robustness`/`block_ablation`/`exploratory`/`measure` per task,
  `figures`, `total`) that records wall and process-CPU seconds, the number
  of examples and batches through the forward and gradient passes (counted
  in `ModelBackend`) and, on CUDA, the peak allocated memory during the
  stage. `metadata.json/timings_seconds` keeps the flat `{section: seconds}`
  table with the same keys as before, `metadata.json/profile` holds the
  structured version plus totals (forward counts, max RSS, peak/reserved
  GPU memory against the device's total, the load average, the cgroup CPU
  quota and the container's cgroup-throttled seconds, also per section),
  and the log ends with a table of the sections over one second. CPU
  seconds against wall seconds and the throttled seconds separate host
  contention from compute; the forward counts make the stage costs
  comparable across models and batch sizes. Both fields are
  volatile for `directions compare`. The cost is a few counters per forward
  call and one CUDA memory query per section.

Consequences: iteration-4b results are not strength-for-strength comparable
with iteration 4 (a different calibration rule and possibly a different head
set); the gates, the nulls and the layerwise metrics are unchanged. The
change alters the numerical path (GPU decompositions), so the first 0.6B run
is executed twice and diffed (`directions compare`).

## 2026-09-11 — Reproducibility checking

### D23. An in-run determinism check replaces full replicate runs

Until now every change of the numerical path was followed by a second,
identical GPU run diffed against the first (`directions compare`): iteration
1 on four models, 3b, 4 and 4b, all bit-identical, the last one across two
different workers. A replicate proves that the pipeline is deterministic on
that hardware; it carries no scientific information (same seed, same model,
same data), cannot detect a deterministic bug (iteration 4's saturated
metric replicated perfectly), and costs a full run per protocol change.
Its one incidental payoff, exposing the host-contention slowdown, is now
provided by the run profile (D22).

From this decision on, the pipeline checks determinism inside every run
(`determinism_check: true`, the default): for the first task that reaches
held-out steering it repeats the steered forward pass, the gradient pass and
one layerwise profile (medians only; the bootstrap draws are seeded
separately) and requires every array to be bit-identical to the original,
`nan` equal to `nan`. The result is written to
`metadata.json/determinism_check` (per array: identical, largest absolute
difference) and `summary.json/deterministic`; a failure is logged as an
error and as a `determinism` rejection but does not stop the run. The cost
is one forward pass, one backward pass and one profile (about ten seconds
on the 0.6B model), paid on every model and every seed rather than on a
chosen day. Full replicates are no longer requested; the check covers the
model forward with hooks, the gradient capture and the device-side linear
algebra, which is where non-determinism would enter. What it does not cover
is unchanged: reproducibility across GPU architectures is a tolerance
question (`directions compare --atol`), and a replicate on a reduced config
remains the tool if the check ever fails and the cause has to be located.

## 2026-09-15 — Iteration 5: damage, first-order depth profile, head count

### D24. A damage measure beside every steering effect

Context: every calibration on 1.7B, 4B and 8B kept improving the target
log-probability up to the grid ceiling, four times the vector's norm and up
to fifteen times the residual stream at the injection layer, and nothing in
the pipeline said whether the model was still producing a distribution at
that point. The steering literature pairs every effect with a damage
measure (perplexity on generic text, a coherence judge, or a divergence from
the unsteered model); this pipeline had none.

Decision: every steered forward pass whose unsteered counterpart exists
records the **KL divergence from the unsteered next-token distribution at
the query token**, `KL(p_base || p_steered)`, per example, whether the most
likely next token changed, and the **collateral KL**: the same divergence
with the correct first answer token removed from both distributions and
the rest renormalised. The query token is where the intervention acts and
where the first answer token is decided, so the raw KL contains the
intended change (the first 0.6B run changed the most likely token for
94–98 % of the lexical prompts, which is the steering working, since the
unsteered model never produced the answer); the collateral KL is the
change the injection makes to everything *other* than the answer, and is
the damage measure proper. A large collateral KL with a small target gain
is the signature of an injection that has stopped being a perturbation. Recorded for every calibration grid point (with the matched
random screen's controls, and a paired excess test of the real direction's
KL against theirs, on its own bootstrap stream so no existing draw moves),
for the held-out steered run and every control kind (`evaluation.json/
damage`: means, the excess against the gate controls and per kind,
per-example values), for the strength-robustness alternatives, and in the
determinism check. Cost: one `(B, V)` log-softmax that the scoring already
computes, and the baseline's `(N, V)` distribution kept per task (117 MB on
the Qwen3 vocabulary, freed after the task). Not a gate: three models' worth
of values are needed before a threshold or a matched-control criterion can
be preregistered; the by-kind excess is the candidate criterion ("no more
damage than a random direction of the same norm"). A perplexity on generic
text was considered and left out for now: it needs a second prompt set and
its own forward passes, and the query-token KL answers the immediate
question, whether the calibration's strongest points still perturb rather
than replace the stream.

Amendment (batch B, before the batch-B runs): the collateral KL did not
separate from the raw KL on 0.6B (the two agreed to within a few per cent
for every task and every control), because on a task prompt the
intervention moves probability among many task-related tokens, not only
onto the answer, and removing one token from the distribution does not
remove the intended change. The measure that does not mix with the
intended change is the one the steering literature uses: the intervention
on text that carries no task. Each run therefore also applies the
direction, at the selected layer and norm, to the last token of a fixed
set of 48 **neutral prose** sentences cut mid-sentence (`directions.neutral`;
`evaluation.neutral_prompts`, 0 disables) and records the KL from the
unsteered next-token distribution there, for every calibration grid point
with its random screen (`neutral_kl_excess_test`), for the selected
condition and every control kind on the held-out pool (`evaluation.json/
damage/neutral`, paired excess tests against the gate controls and per
kind, on the existing damage bootstrap streams), for the strength-
robustness alternatives and for the decomposition parts. The sentences
are the same for every task and model, so the neutral KL is comparable
across tasks in a way the query-token KL is not. Cost: one forward pass
of 48 short prompts per condition (grid points, screen controls, held-out
controls, alternatives, parts), under a tenth of a task's forward work on
0.6B, where calibration and the control forwards together take a quarter
of a task. Still not a
gate, for the same reason as above; the by-kind excess on neutral prose
("no more disturbance of unrelated text than a random direction of the
same norm") replaces the collateral excess as the candidate criterion.

### D25. The functional depth profile: the first-order effect per read point and per block, compared against the nulls

Context: the geometric profile (S, log G, C, A, d_eff, d90, N) has come back
indistinguishable from the random-direction nulls on every model and at
every strength, while the behavioural effect is direction-specific. The
geometry describes how a perturbation grows and rotates; it does not say
where along the depth the perturbation acquires its effect on the answer.
The per-example readout `δ_l · g_l` (D17), the first-order predicted change
of the target's log-probability at read point `l` given the baseline
gradient `g_l`, was recorded but neither summarised nor compared.

Decision: two per-example metrics join the compared set. `gradient_projection`
is `δ_l(x) · g_l(x)` at every read point, and `first_order_increment` is its
change across block `l`, `δ_{l+1}·g_{l+1} − δ_l·g_l`: the first-order
contribution of block `l` to the behavioural effect, positive where the
block moves the perturbation toward the answer and negative where it moves
it away. Both are summarised like every other per-example metric (median
with a bootstrap CI for the real direction, medians for the controls) and
compared per layer against every null (z, empirical p), with
`gradient_projection_z_mean` and `first_order_increment_z_mean` in the
signature, the by-kind table and the aggregate. The signature also reports
the first-order effect at the injection layer and at the final layer and
the centre of mass of the positive increments in normalised depth, which is
"where the effect is created" in one number. Interpretation caveat: the
gradient is the baseline's, so the first-order prediction is exact only for
small perturbations; at the canonical strengths it is a projection, not a
prediction, and the final-layer value should be read next to the measured
behavioural change. The bootstrap draws of the two new metrics come from a
child stream of the profile's generator, so every existing CI and every
later draw is unchanged. Cost: none beyond the two subtractions; the figures
gain two panels.

### D26. The head count is chosen by a joint-effect sweep, and a head-support gate replaces the fixed ten

Context: D21 took the paper's ten heads. On 0.6B and 1.7B one head carries
most of the indirect effect and the other nine add mostly generic
in-context content; on 4B and 8B no head moves any task's deranged prompts
by more than a few hundredths, and arithmetic's heads move it by nothing on
every model, yet every task received a ten-head "function vector" and could
qualify on the behavioural gates alone. Ten was the paper's saturation
point on GPT-J, not a property of function vectors, and nothing in the
pipeline checked that the selected heads carry the task they are used for.

Decisions:

* **Head count by sweep** (`function_vector.n_heads: null`,
  `head_count_candidates: [1, 2, 4, 8, 16, 32]`). With the universal
  ranking in hand, the top-`k` sets are patched *jointly* into each task's
  deranged prompts (one forward pass per `k` per task, all heads at once)
  and the per-prompt change of the answer probability is pooled over
  tasks. The count with the largest pooled mean effect is found, every
  smaller count is tested against it with the paired bootstrap, and the
  smallest count whose deficit is not significant (p > 0.05) is chosen:
  the fewest heads that do what the best set does. A fixed `n_heads` keeps
  the paper's construction available. With `head_selection: per_task` the
  choice is made per task.
* **Head-support gate.** For each task the chosen set's joint effect on its
  deranged prompts must be positive (paired bootstrap, p ≤ 0.05), exceed
  that of `head_support_null` = 16 random sets of the same size patched the
  same way (paired excess test, p ≤ 0.05; own seed), and restore at least
  `head_support_min_restored` = 10 % of the gap between the deranged
  prompts' answer probability and the positive prompts' (a fraction of the
  task's own gap, so the criterion is scale-free across models; the first
  0.6B run showed why it is needed: arithmetic's 16 heads move its
  deranged prompts by +0.003, which is significant and above random sets,
  but 0.4 % of a gap of 0.8, while every lexical task's set restores
  80–110 %). A task that fails has no function vector in the paper's sense:
  its outputs up to and including the vector, the sweep and the test are
  written, and with gates enforced it is not calibrated or measured.

Rationale for testing the set rather than single heads: the paper's causal
evidence is per head, but its object is the sum, and a task carried jointly
by several heads should pass. Rationale for the saturation rule: a fixed
tolerance on the effect would need a different number per model; "not
significantly below the best" is the same criterion at every scale. Cost:
`|candidates| + 16` patched passes over 64 prompts per task, seconds on 0.6B
and about a minute on 8B, against the ranking's 448–1152 passes.

Consequences: the head set can be smaller than ten and can differ in size
between models; arithmetic (and, if the 4B/8B effects are as small as they
look, more tasks there) will fail the gate, and a model where no task passes
is a model with no function vectors under this construction, which is a
result. Runs after this decision are not head-for-head comparable with
iteration 4b.

### D27. No injection past half depth; the injection layer is the one whose selected strength improves most

Context: on every model the function-vector heads sit at 54–69 % of depth,
and the earliest-reliable-layer rule (D1) injected the vector at 20–25 %,
where every task was reliable. Todd et al. sweep the injection layer and
take the best one, which for them is in the early-middle layers. The
candidate layers ran to 60 %, past the start of the heads' band.

Decisions: the candidate layers are 20/30/40/50 % of depth
(`candidate_depth_fractions: [0.2, 0.3, 0.4, 0.5]`); nothing is injected
past half depth, because a perturbation placed inside or after the band
whose outputs the vector is made of would not have to propagate through
the network to act, which is the thing this project measures. Among those
candidates the layer is the one whose selected strength (D22's reference
rule) improves the calibration metric most (`calibration.layer_rule:
best`), the paper's sweep in the pipeline's statistical terms. The
earliest-layer rule stays the PCA-control protocol and remains available.
Consequence: the injection layer can differ between tasks and models, and
the profiles cover a shorter depth when a later candidate wins; the
"layer-matched" case of the discussion, one injection immediately upstream
of the heads, is the 50 % candidate, and whether the sweep prefers it or
the paper's early-middle layers is a result of the run.

### D28. The vector's common and task-specific parts

Context: on every model the tasks' function vectors have pairwise cosines
of 0.3–0.9 (median ≈ 0.5) and are 0.5–0.8 aligned with vectors built from
deranged-label prompts; the other-task and deranged-prompt nulls carry
most of the behavioural effect at weak and moderate strengths. Whether the
"control direction" of this construction is a shared in-context component
plus a task-specific residue, and which of the two the geometry and the
behaviour belong to, is the question the project's measurement has to
answer before the propagation question means anything.

Decisions:

* **Leave-one-out common direction.** For task `t`, `c_{-t}` is the
  normalised mean of the *other* tasks' unit control directions at the
  layer, over the tasks that reached phase B (so a task's own vector never
  enters its null). For the function vector, which is layer-independent,
  it is one direction.
* **A sixth control kind, `common`** (one per task): `c_{-t}` injected at
  the real direction's layer and norm, so the layerwise nulls and the
  behavioural excess tests include "the shared direction alone".
* **The additive split.** `v_t = (v_t · c_{-t}) c_{-t} + r_t`; each part is
  injected at its natural share of the selected strength (`α · ‖part‖` for
  the unit vector's parts), so the two injections sum to the real one, and
  each is measured exactly like the real direction: held-out effect and
  damage, the paired excess against the gate controls and against the real
  direction, the layerwise profile against the same control profiles, its
  signature and labels, and rank correlations with the real profile. The
  bootstrap draws come from a separate stream. Written to
  `decomposition.json`; the summary carries the cosine with the common
  direction and the two parts' held-out effects.

Cost: one steered pass and one profile per part per task, plus one control.
Interpretation: the model is not linear in the injection, so the two parts'
effects need not add to the real one; `additivity` records the gap. The
distributed, layer-matched injection of each head's contribution at its own
layer was considered and set aside in favour of the single-layer rule of
D27, so that every condition stays comparable to the paper's construction.

## 2026-09-15 — Iteration 6: depth of commitment, the head count bounded

### D26, amended: a larger head count must earn its place

Context: the sweep chose the largest candidate (32) on 1.7B, 4B and 8B. Its
rule, "the smallest count not significantly below the best", cannot stop
once the pooled test has 512 prompts: on 1.7B the gain from 16 to 32 heads
was 5 % and significant at p = 0.0005, on 4B 10 %, and on 8B the effect was
still rising by a third per doubling. Heads past the tenth have individual
effects at the noise floor; jointly they still add.

Decision: the candidates are extended to 64 (`head_count_candidates:
[1, 2, 4, 8, 16, 32, 64]`), and a larger candidate replaces a smaller one
only if the best larger candidate beats it *both* significantly (paired
bootstrap, p ≤ 0.05, as before) *and* by at least `head_count_min_gain` =
10 % of the smaller one's pooled effect. The candidates are visited in
ascending order and the first one that no larger candidate beats on both
counts is chosen; every candidate's p-value and gain fraction against the
best larger one are recorded (`function_vector.json/head_count`), so the
marginal gain of each doubling is on record whether or not it was taken.
The gain fraction is an effect-size threshold, which the project's
principles reserve for cases where sampling noise is not the issue; here it
is not (every deficit is significant), and a 10 % gain for a doubling of
the heads is what "marginal" means in this context. On the batch-B data the
rule would choose 8 heads on 0.6B, 16 on 1.7B and 4B, and 32 or more on 8B.

### D29. Depth of commitment: where the injected direction stops being needed as a direction

Context: the project's question is where the control signal is turned into
computation. The geometric profile has mostly sat inside the random-direction
nulls, and the first-order readouts are correlational: they say that the
perturbation's projection on the target's gradient grows downstream on
some models, not that the original direction has stopped mattering. The
block ablation (exploratory) removes whole block contributions, which
conflates the direction with everything the block does.

Decision: a causal sweep along the depth, preregistered, on every qualified
task (`commitment` config; `commitment.json`). With the vector injected at
`l*`, at every later read point `m` the perturbation `δ_m = h_m^steer −
h_m^base` at the query token is edited on top of the injection and the
held-out effect that survives is measured on the first `n_prompts` = 96
evaluation prompts (the base and steered captures are re-run on exactly
those prompts, so the edits land on the captured residuals exactly, which
the determinism of the forward pass guarantees and a test checks):

* **remove**: `δ_m ← δ_m − (δ_m·u)u`. If the effect survives, `u` is no
  longer carrying it at `m`; it has been converted into other features.
* **keep**: `δ_m ← (δ_m·u)u`. If the effect survives, the direction itself
  still carries it.

`u` is the injected direction (the same vector at every `m`) and, as a
second direction, the task's PC1 at `m` (the contrast direction the task
itself uses at that depth). Each edit is matched against the same edit
along `n_controls` = 8 random unit directions applied to the same steered
run (own seed): removing a random component changes nothing and keeping
only a random component keeps nothing, so the paired excess of the real
edit over the random ones (D18's test) says whether the direction matters
at `m` beyond the noise of editing. Per read point the surviving effect,
its share of the full effect, its bootstrap CI, the cost against the
unedited run and the excess test are recorded. The summary gives, per
direction, the last read point at which removing it still costs effect
beyond random removal (`needed_until`), the first read point after it
(`commitment_layer`, also as a fraction of the downstream depth; `None`
when the direction is needed to the end), the share of the effect
retained after removal at the end, and the share the direction alone
carries at the end (`carried_by_direction_final`) with the last read point
at which it carries more than a random component (`carried_until`).

Reading: an early commitment layer with a small `carried_by_direction`
at the end is conversion (the direction was the trigger, the effect lives
elsewhere); a direction needed to the end with a large carried share is
conserved transmission (the effect is the direction's own projection on
the readout); a direction needed to the end with a small carried share is
the direction acting through what it has recruited at every layer. Cost:
(2 directions × 2 edits + 2 edits × 8 random directions) forward passes
over 96 prompts per read point after `l*`, about three times the
control-forward stage of a task, so the 4B and 8B runs grow by a third to
a half.

Amendment (after iteration 6 on the four models): the core summary is an
effect size, the significance readout is secondary. On 27 of the 31
task-model pairs the direction was still "needed" at the last read point,
so `commitment_layer` was undefined, but the cost of removing it there
was 0–13 % of the effect: a paired test on 96 prompts detects a 2 % cost,
and a summary that hinges on it says nothing about *where* the effect
moved. What did carry the result was read off the curves post hoc: the
depth at which removal leaves half, then 90 %, of the effect. Those are
now the preregistered summary (`commitment.handover_shares`, default
`[0.5, 0.9]`): per direction, `handed_over_50` / `handed_over_90` is the
first read point from which the share retained after removal stays at or
above the share for every later read point (a single upward blip does
not count; a curve that ends below the share gives `None`), and
`carried_alone_until_50` / `_90` is the last read point up to which the
direction alone retains at least the share at every read point from the
injection on; each also as a fraction of the downstream depth. Reading:
`handed_over_50` is where the conversion is half done, `handed_over_90`
where it is essentially complete, `carried_alone_until_50` how long the
direction by itself would have sufficed. `needed_until`,
`commitment_layer` and `carried_until` stay in the summary as the
matched-null readout of whether the direction matters at all at a depth.
Both summaries are derived from the saved curves, so earlier runs are
re-summarised without a rerun.

### D30. The add-k family, scored on the digits the operation changes

Context: `arithmetic` (n → n + 3) and `arithmetic_words` have been
rejected by the head-support gate on every model and seed, while the
lexical tasks pass it with room to spare (8B: the best single head
restores 0.0003 of the deranged-to-positive gap for arithmetic against
0.02–0.08 for the lexical tasks, and the universal vector 1–3 % against
80 %). No function-vector or task-vector paper has extracted a vector for
an operation whose parameter is read from the demonstrations; their
numeric successes are successor tasks (Todd et al.'s Next-Item, Hendel et
al.'s Next Letter), fixed relations like antonym. Yang et al. (2025) argue
a task vector acts as a single synthetic demonstration, a rank-one
predictor, and fails on mappings that need more than one degree of
freedom. Whether add-3 fails because the head-mean vector cannot carry an
operand, or because numbers are different, was not testable with one
operand.

Two things were wrong with the measurement before the question could be
asked. First, Qwen tokenises numbers digit by digit and the target
template adds a leading space, so the decision metric averaged the
log-probability of four tokens of which one carries the operation (for
add-3, the last digit in 90 % of the items; two or three when it
carries); the easy digits and the space diluted any effect, and exact
match, which is all or nothing over four tokens, could not see a partial
one. Second, the config keyed tasks by registry name, so one task could
not appear with several operands.

Decision:

* **Changed-token scoring** (`prompt.target_scoring`, per task
  `tasks[].target_scoring`; default `all`, the behaviour to date). With
  `changed_tokens` the prompt carries the input rendered through the
  target template as a scoring reference, and a target token is not
  scored when the reference has the same token at the same position
  under the left alignment (the shared leading space) or under the right
  alignment (the digits the operation leaves untouched); a target
  identical to its reference keeps every token. The target's
  log-probability sum and per-token mean, and with them calibration, the
  held-out effect, the gap fraction, the head-support metric (the product
  of the scored tokens' probabilities), the commitment sweep and the
  gradient of the first-order readouts, all use the scored tokens. Exact
  match, the first-token metrics and the token counts are unchanged.
  `splits.json` records the scoring and the mean number of scored tokens.
* **Task labels** (`tasks[].label`): the task's identity in the run
  (directory, seeds, other-task and common controls) when set, so a
  registry task can appear several times; the registry name is recorded
  as `registry_task`.
* **The add-k family**: `configs/arith_qwen3_*.yaml` are the pilot
  configs with the tasks replaced by `arithmetic` for k = 1, 2, 3, 5, 10
  under changed-token scoring and `arithmetic_words` for the same k (its
  targets are words, every token changes). The universal heads, the
  other-task controls and the common direction are then taken within the
  family, which is what the question needs: is there any head whose
  output carries "add k", and does the vector for one k transfer to the
  others.

Reading: if add-1 passes the head-support gate and larger k do not, the
vector can name the relation but not carry the operand; if the vectors
for different k differ by a consistent direction, the operand does live
in one direction and the head mean averages it against the numeric
content; if nothing passes, the boundary is numbers, not parameters. The
per-token metric of the arithmetic tasks is not comparable with the
earlier runs' (their four-token average), and the family's head count and
universal heads are the family's own, so these runs are a separate
iteration, not a re-run of the pilot.

### D31. The learned single vector: the best one direction can do

Context: the add-k family (D30) showed that the head-mean function vector
carries a fixed relation (add-1 on 4B and 8B) and not a demonstration-read
operand (k ≥ 2 on any model). That is a fact about one construction. The
rank-one argument (a task vector acts as a single synthetic demonstration)
says no single vector can carry the operand, whatever the construction;
the head mean averaging the operand away says only that this one does
not. The two are separated by fitting the best single vector directly:
if a fitted vector recovers the gap where the head mean does not, the
failure was the extraction; if it does not either, the limit is the rank,
and the next rung is an operator. The same fit on the lexical tasks says
how much of the way to the few-shot gap one direction can go at all, and
whether the fitted direction is the head-mean direction.

Decision: a third control, `extraction.control: learned_vector`, beside
`pca` and `function_vector`, with everything downstream unchanged.

* **The fit.** At each candidate layer, one vector `v` is added to the
  residual at the query token of the zero-shot prompts of the extraction
  pool (64 items; the same pool the other constructions extract from,
  so the calibration and evaluation pools stay untouched) and optimised
  to raise the summed scored log-probability of the target, averaged over
  the pool, with the model frozen. The gradient with respect to `v` is
  the activation gradient at that read point of the steered forward pass
  (`ModelBackend.gradients_with_scores`; the finite-difference check is
  a test), so a step costs one forward and one backward over 64 prompts.
  The optimiser is Adam on the vector in float64 on the host, step size
  `lr_fraction` = 0.05 of the vector's norm, `n_steps` = 100, and after
  every step the vector is projected back to a fixed norm: the median
  residual norm at the layer on the extraction prompts (the vector's
  natural strength, so ρ = 1 in the calibration grid is one residual
  norm and the grid scales it as it scales the function vector). The
  control carried forward is the first seed's fit, one deterministic
  vector. Two more seeds (random unit initialisations) measure how
  unique the solution is: the stability is the min pairwise signed
  cosine between the seeds' vectors (the sign is meaningful). The loss
  trajectory is recorded per seed and layer.
* **Stability is reported, not gated** (amended before the first full
  run, on the first fit lines of 0.6B: at every candidate layer three
  seeds reached a training loss near zero from 7.5 with pairwise cosines
  of 0.1–0.2). With d free parameters and 64 prompts the set of vectors
  that fit the pool is large, so the cross-seed criterion that guards a
  noisy PC1 would reject every learned vector for being non-unique and
  the question would go unmeasured. The mean of dissimilar solutions is
  not a solution, so the pooled mean is not used either. What judges a
  learned vector is what judges every direction, the held-out gates and
  the matched nulls, plus its own permuted-target null; the stability
  says whether "the" learned direction exists (near 1) or a subspace of
  them does (near 0), which is a result about the control, not a gate.
* **What is not there.** No head-support gate: the construction has no
  heads. The PCA direction is still extracted and reported (the `T_l`
  readout and the cosine with the learned vector).
* **The matched null of the construction** (`demo_variation` under this
  control): vectors fitted with the same budget and norm to the pool's
  targets permuted across items (a derangement), built at the selected
  layer only when the controls are, one per control. A learned vector
  that beats them steers the task rather than "the answer format the
  fit could reach with any targets"; for the arithmetic tasks this is
  the format-versus-operand question asked directly.
* **Configs.** `configs/learned_qwen3_*.yaml` are the pilot configs with
  the control switched and `arithmetic` scored on the changed digits
  under the label `add_3` (D30) so that its numbers compare with the
  add-k family.

Cost: per task, 3 seeds × 4 layers × 100 steps plus 8 null fits at one
layer, each step a gradient pass over 64 prompts: about the size of the
head-effect stage on the small models and a third of a run on 4B and 8B.
The step count and step size are the first values tried, chosen so that a
fit converges on the toy in a handful of steps; the recorded loss curves
say whether 100 steps were enough or too many, and a change is a config
change to record here.

Reading: on the lexical tasks, the learned vector's held-out effect
against the function vector's is how much the head mean leaves on the
table for one direction; its cosine with the function vector says
whether it is the same direction. On add-3, a learned vector that
qualifies and beats its permuted-target null carries the operand and
the head mean does not; one that does not qualify puts the limit at the
rank.

## 2026-09-17 — Iteration 8: the trajectories compared

### D32. Downstream trajectories compared all-to-all, against the natural one and against one another

Context: the learned vector (D31) recovers the whole few-shot gap on
every task and model while being orthogonal to PC1 and to the head-mean
function vector (|cos| ≤ 0.12 in 1024–4096 dimensions). Two orthogonal
inputs that produce the same output leave open what happens in between,
and the pipeline's saved readout answers only part of it: `task_alignment`
is the cosine of the steered perturbation with the *pooled PC1* of the
extraction contrast at each read point, a direction rather than the
per-example natural difference, taken against the correct-versus-deranged
contrast rather than demonstrations-versus-none, with an arbitrary sign,
and the residual vectors themselves are not saved (37 × 192 × 4096 floats
per pass, 67 passes per task). On 8B it already shows the learned vector
starting orthogonal to that direction and ending at 0.66–0.73 on six
tasks where the head mean starts at 0.12–0.18 and stays there, but the
last read point is where every successful intervention shares the
answer's unembedding direction, so late convergence can mean "same
answer" as well as "same computation". Three readings are to be told
apart: the fitted vector stays different all the way (different
computations implement the task); it collapses, gradually or suddenly,
onto the natural trajectory (downstream layers canonicalise control
signals); it aligns and then diverges (one of several routing signals).

Decision: a separate command, `directions trajectories`, on two finished
runs of one model (the head-mean and the learned-vector run, which share
seed, splits and held-out prompts; checked), rather than a pipeline
stage, because it needs both controls at once and only captured passes:
no extraction, fitting or calibration is repeated.

* **Trajectories.** On the held-out zero-shot prompts, per example and
  read point: the perturbation of each construction injected at a common
  layer, `h_m(x; h_l + v) − h_m(x)`, for the pooled PC1, the head-mean
  function vector and the learned vector; the natural difference
  `h_m(x; demos) − h_m(x)` with the pipeline's own few-shot prompts
  (`icl`), the same with a second demonstration sample (`icl2`), and the
  label contrast `h_m(x; demos) − h_m(x; deranged)` (`task`). All fifteen
  pairs among the six are compared by the signed per-example cosine, the
  population-mean trajectories by the cosine of their means, and each
  construction against the natural references also by the projection
  fraction (how much of the natural difference it reproduces along it).
* **The answer direction removed.** Every cosine is also taken after
  projecting out, from both vectors, the unit direction `g ⊙ W_U[t]` (the
  final RMSNorm's scale times the first target token's unembedding row):
  the direction along which any intervention that raises the answer must
  move. Alignment that survives the removal is alignment of the
  computation, not of the output.
* **Common layers and strengths.** Per task the learned run's selected
  layer (primary) and the head-mean run's when different. Each
  construction enters at what its own run's calibration gives it at that
  layer: the selected point at the selected layer, else the reliable grid
  point nearest the run's reference strength (the run's own rule at that
  layer), else the natural norm. PC1 was calibrated in neither run; it
  takes the point of the D1 grid (units of the median residual norm at
  the layer, taken from the learned run's calibration) with the largest
  mean per-token improvement on the calibration pool, and its held-out
  effect is reported like the others'. A construction's strength is
  recorded with its source.
* **Floor and ceiling.** `n_isotropic` random unit directions per
  construction and layer, injected at that construction's strength, give
  the floor of every pair the construction enters (pooled over the
  controls' examples); the cosine between the two natural trajectories
  gives the ceiling that a construction could reach at all. Medians carry
  percentile-bootstrap CIs over examples; a curve is "above the floor" at
  a read point when the CI low of its median exceeds the CI high of the
  floor's.
* **Summary and labels.** Per construction and reference: the median
  cosine at injection, its peak and the peak's depth as a fraction of the
  downstream depth, its final value (the last read point, before the
  final norm), the floor at each, and a paired bootstrap test of the
  decline from the peak to the end. The label is exploratory, the peak
  being chosen on the same data: `never_aligns` (never above the floor
  after injection), `converges` (above the floor at the end, no
  significant decline), `aligns_then_diverges` (a significant decline to
  the floor), `aligns_then_partly_diverges`, `partial`. The curves and
  the numbers are the result; the label is a reading aid.
* **The generic response removed** (amended after the first 0.6B and 8B
  runs, before the others): the isotropic floor itself rises with depth,
  to 0.3–0.5 at the last read point on most tasks, with a jump at the
  final block. Any perturbation of these norms drives the late residual
  along a shared direction, so a raw cosine of 0.7 against the natural
  trajectory over a floor of 0.5 is a smaller alignment than it reads.
  The third variant projects out, from both vectors of every pair, the
  generic response: the mean of all the isotropic controls' trajectories
  at the layer, per example and read point (the answer direction is the
  same idea for the output; this is the same idea for the residual). A
  control's own floor removes the mean of the other controls, so that
  the floor is not deflated by removing the control from itself. The
  generic response's cosine with the answer direction is recorded per
  read point, so the two removals can be told apart, and so is the
  *coherence* of the random responses (the norm of their mean over the
  mean of their norms, per example and read point; 1 when every random
  push produces the same downstream change), pooled and per
  construction: how much of a random push's effect is this shared part.
* **What is saved.** Per task: the per-example cosines of every pair at
  every read point (raw, without the answer direction, without the
  generic response), the projection fractions, the floors' medians, the
  population-mean trajectories per condition and the generic response's
  mean (float16), so that this comparison never needs the passes again;
  not the residuals.
* **Reproducibility.** One repeated steered pass must be bit-identical
  (`metadata.json/determinism_check`); the comparison's own draws (the
  second demonstration sample, the derangements, the isotropic
  directions, the bootstraps) come from its config seed through the
  stable digest; the runs it read are recorded with their commits.

Cost: per task, four captured passes over the held-out prompts, ten
short passes over the calibration pool per layer for PC1, and per layer
three steered passes plus `n_isotropic` × 3 control passes, all captured:
about half of one pipeline profile stage per task, and no gradients.

## 2026-09-18 — Iteration 9: the geometry on the device, strengths, the generic response, the patch test

### D33. The trajectories at two strengths, what the generic response is, and whether the shared component carries the effect

Context: iteration 8 (D32) found the fitted vector's trajectory collapsing
progressively onto the natural one, at cosines of 0.5–0.9 against floors
near zero once the generic response is removed, with the constructions
not converging onto each other. Three things it could not say. Whether
the alignment depends on the size of the push (every construction was
injected at its canonical strength, one point on the strength axis, and
the generic response was measured at that one norm). What the generic
response is (a fixed direction of the late residual or a prompt-dependent
one; the mean of the random pushes was saved, nothing else). And whether
the shared late component is what carries the effect, which cosines
cannot tell from a bystander. A fourth problem was cost: the trajectory
geometry ran in NumPy on the worker's CPU while the GPU sat idle, 85 % of
the runs' time (43 of 50 min on 4B), so adding a strength would have
doubled a run.

Decision, four parts.

* **The geometry runs on the model's device.** Every trajectory stays on
  the device as float32 (L+1, n, d); the cosines with and without the
  removals, the projection fractions, the coherence, the leave-one-out
  generic directions and the row-wise bootstraps of the medians are
  computed in torch in float64, the same arithmetic as the NumPy
  functions, which remain in the module as the reference the tests
  compare the device path against (cosines to 1e-9, bootstrap medians
  exactly, CIs within sampling noise). The bootstrap draws come from a
  torch generator seeded from the config seed through the stable digest,
  so a run is deterministic on a device type; the small paired tests stay
  in NumPy. The run is bounded by its forward passes again, as the
  profile stage has been since D22.
* **Strength factors.** `strength_factors: [1.0, 0.5]`: the whole
  comparison (steered passes, matched floors at that norm, generic
  response, coherence, all pairs in the three variants, summaries,
  ceilings) is repeated at each factor times each construction's
  canonical strength, and the same construction's trajectories at the
  canonical and at the weaker strength are compared per example (raw
  cosine, its median curve, and the cosine between the two factors'
  generic responses). The canonical factor remains the primary result and
  keeps the D32 output layout; the others are nested under
  `other_strengths`. Half strength is the one factor added now: it is
  where a fit at one residual norm still steers on most tasks (the
  calibration grids), so the alignment at two working strengths is the
  question, and one factor costs one more set of passes.
* **What the generic response is**, recorded per factor and layer from
  the population means: the fraction of its energy in its own top k
  coordinates and in the k largest coordinates of the unsteered residual
  at the same read point (the massive-activation coordinates), with those
  coordinates listed; its cosine with the residual mean; its norm against
  the residual's; its logit lens at the last read point (the mean random
  response added to each prompt's final residual, through the final norm
  and the unembedding in float32: the ten tokens it promotes and demotes
  and the mean absolute logit change); the coherence of the random
  responses (the norm of their mean over the mean of their norms, per
  example and read point, pooled and per construction); and its cosine
  across strengths. Together these tell a background rescaling (energy
  in the massive coordinates, aligned with the residual mean, promoting
  nothing in particular, coherent at every norm) from an off-distribution
  reaction (energy spread, specific tokens promoted, coherence growing
  with the norm).
* **The patch test**, at the primary layer and the canonical strength for
  the constructions in `patch.constructions` (the learned vector by
  default; the head mean can be added). At read points at fixed fractions
  of the downstream depth (0.5, 0.75, 1.0; fixed in advance, not chosen
  on the alignment curve), the steered perturbation at the query token is
  edited before the remaining blocks run, with the depth-of-commitment
  mechanics (D29) and a per-example reference: `remove` takes out the
  component along the prompt's own natural difference δ_m^ICL(x), `keep`
  leaves only that component, and `patch` adds δ_m^ICL(x) itself to the
  unsteered run with nothing injected at the injection layer. Each edit
  is matched against the same edit along `patch.n_controls` random
  per-example unit directions (for `patch`, random vectors of the same
  per-example norm) by the paired excess test, and reported as the
  effect retained (per-token log-probability, and the first-token
  log-probability beside it, since an edit at the last read point can
  reach only the query position's own prediction: no block follows to
  carry it to later target positions, so for a multi-token target scored
  on later tokens the depth-1.0 row is a first-token statement). The
  alignment at the edited read point is recorded with each row. The
  reading: `remove` costing more than random removal says the shared
  component is needed; `keep` retaining more than random keeping says it
  suffices; `patch` giving the effect alone says the natural component
  is sufficient by itself.

Cost: on the device the geometry is minutes per model; each strength
factor adds 15 passes per layer, the patch test 36 passes per task at
the primary layer (3 read points × 3 edits × 4 runs), the diagnostics
nothing. The whole iteration is about a fifth of the D32 runs' wall
time despite doing twice the passes.

### D33, amended (2026-09-18): a quarter strength, and the neighbouring candidate layers with their own patch test

Context: the iteration-9 runs answered the strength question at two
points (canonical and half) and the commitment question at one injection
layer per task (the learned run's selected layer). Half strength aligned
at least as well as the canonical one and removed the late decline on
1.7B and 8B, so the strength axis is still open below; and every
statement about the hand-over depth rests on the one layer the selection
rule preferred, which leaves open whether the hand-over is a property of
the layer or of the vector.

Decision, two config extensions of the D33 protocol, no change of method:

* **`strength_factors: [1.0, 0.5, 0.25]`.** The quarter strength is
  added as a third factor with everything the half strength has (its own
  steered passes, matched floors, generic response, coherence, all pairs
  in the three variants, summaries, ceilings, the cosine with the
  canonical trajectories and between the generic responses). The
  canonical factor remains the primary result. Double strength is not
  added: the calibration grids were still rising at their ceiling on
  three models, and the half-strength result says the canonical strength
  already overshoots on two of them.
* **`neighbour_layers: 1`** with `layers: selected`: per task, the nearest
  candidate layer of the learned run below and above the primary layer
  is compared as well (the head-mean run's selected layer stays in when
  it differs). Only candidate layers qualify, because the learned vector
  and PC1 exist there and nowhere else; with candidates at 0.2–0.5 of the
  depth (D27) a primary at the top or bottom candidate has one neighbour,
  which the task's notes record. At a neighbouring layer each
  construction enters at the strength its own run's calibration gives it
  there (the reliable grid point nearest the reference rho, else the
  natural norm; PC1 calibrated on the calibration pool as at every
  layer), so the comparison at a neighbour is the comparison the
  selection rule would have made had it chosen that layer. The role of
  every layer (`primary`, `fv_selected`, `neighbour_below`,
  `neighbour_above`, `listed`) is written with the task's result.
* **`patch.layers: all`**: the patch test runs at every compared layer,
  not only the primary one, at the same depth fractions of each layer's
  own downstream depth, so the hand-over can be read per injection layer.
* **The patch grid at eighths of the downstream depth**
  (`depth_fractions: [0.125, …, 1.0]`, twice as fine as iteration 9's
  quarters from 0.5, extended to the injection side). Three points could
  not tell a ramp from a step, and on 4B the hand-over was complete at
  the first one; the retained-effect curve against depth is the causal
  counterpart of the conversion-mass profile, and its shape is the
  delayed-versus-cascaded reading of PROJECT.md. Fractions that round to
  the same read point are merged.
* **The head-mean vector in the patch test** (`patch.constructions:
  [learned, fv]`): whether the shared component carries the head mean's
  effect as it carries the learned vector's, or whether the construction
  that aligned earlier and partly let go keeps a route of its own. Same
  edits, same random matches, its own canonical strength.
* **Per-block writing of the shared component**, no passes: for every
  trajectory (the three constructions from the injection layer, the
  natural trajectories from the embedding) the share of the natural
  difference that each block writes, ``((δ_{m+1} − δ_m) · δ^ICL_{m+1}) /
  ‖δ^ICL_{m+1}‖²`` per example, raw and with the generic response
  removed; summarised per block by the median with its CI, the block with
  the largest share and its share of the positive total, the entropy ratio
  of the positive shares (1 = spread over the downstream blocks, 0 = one
  block) and the rank correlation of a construction's block profile with
  the natural trajectory's own. The reference moves with depth, so the
  shares do not telescope to the projection fraction. This is the causal
  localisation to set beside the geometric conversion mass: whether the
  steered run writes the shared component in the blocks where the
  natural run writes it.
* **A second seed.** Iteration 10 runs on seed 20260916 rather than on
  the seed-20260907 runs again: the learned-vector protocol (D31) is run
  at that seed on the four models, the head-mean runs of that seed exist
  from iteration 6, and the comparison takes `--seed 20260916` for its
  own draws. The quantities the two iterations share (the primary layer,
  canonical and half strength, the patch rows at 0.5, 0.75 and 1.0) are
  then a cross-seed comparison of the trajectory results and of the
  learned vector's held-out numbers, which had one seed; the additions
  (the quarter strength, the neighbouring layers, the finer grid, the
  head mean's patch rows, the block writing) have one seed.

Cost: from the iteration-9 profiles (per layer and factor 3–11 s of
passes, per patch test at three read points 3–6 s), the runs go from
15–17 compared layers per model to 26–28, from two factors to three, and
the patch test from 36 passes per task to about 190 per compared layer
(eight read points, two constructions): about 3.5–4 × the iteration-9
wall time, roughly $2.50 of compute for the four comparisons, plus the
four learned-vector runs at the new seed (40, 50, 78 and 52 min in
iteration 7b, about $7.20).

## 2026-09-19 — Iteration 11: the causal dimensionality of the shared component

### D34. How many directions across prompts carry the effect: a rank-k keep test against random, background and other-task subspaces

Context: iterations 9 and 10 established that by a fixed read point of
the stack the effect of any working construction is carried by one
direction per prompt, the prompt's own natural difference, and that
this component is re-written block by block rather than transported.
What they do not say is how many directions that per-prompt family
spans across prompts: whether the control is canonicalised onto a
task-level signal of rank one or two (a function vector at the
hand-over depth), onto the span of the answers, or onto something
distributed and prompt-specific. The geometric effective dimensionality
(D1, D17) cannot answer this, since it was shown to be a property of
any perturbation of the norm (the generic response) rather than of the
control.

Decision: a fourth causal stage of `directions trajectories`, at the
primary layer and the canonical strength, for the constructions in
`subspace.constructions` (the learned vector by default).

* **The subspaces.** For each task, the natural differences (demos
  minus none, at the query token) of the run's extraction and
  calibration pools, prompts that are never evaluated, are captured
  once at every read point; at a read point m the *own* subspace of
  rank k is the top-k uncentred principal components of those
  differences (so the first component is the pool's mean natural
  difference and the later ones the prompt-specific spread; the pool
  size bounds the rank). Two further subspaces from the same captures:
  the *other-tasks* subspace, the top-k components of the other tasks'
  pool differences pooled together (leave-one-task-out, as D28 did for
  the vectors), which tells a shared in-context subspace from the
  task's own; and the *background* subspace, the top-k components of
  the pool's unsteered residuals at m, which tells a task subspace from
  the massive coordinates. The matched floor is `n_controls` random
  k-dimensional subspaces per example (orthonormalised Gaussian
  columns).
* **The edits**, on the held-out prompts in the steered run at m, with
  the D29 mechanics: *keep* leaves only the projection of the steered
  perturbation onto the subspace, *remove* takes that projection out,
  and *patch* adds the projection of the prompt's own natural difference
  onto the subspace to the unsteered run. Each is run at every k of
  `k_grid` (1, 2, 4, 8) for every subspace, and reported as the effect
  retained (per token and first token) with the paired excess test
  against the random subspaces of the same k. The per-prompt keep and
  patch of the patch test at the same read point are the ceilings (k
  unbounded, one direction per prompt).
* **The read points**: the fractions in `subspace.depth_fractions` (0.5
  and 1.0 of the downstream depth) and, with `at_handover`, the
  hand-over read point read off the patch grid of the same run (the
  first read point at which the per-prompt keep retains
  `handover_share` of the effect), recorded with its source.
* **The summary** per construction and read point: the retained effect
  against k per subspace and edit; k90, the smallest k in the grid at
  which keep (and patch) reaches the share, or none; the spectrum of
  the own pool differences (explained fractions of the top components,
  participation ratio); the overlap of the own subspace with the
  other-task and background subspaces at each k (the fraction of the
  own subspace's energy inside the other); and the cosine of the own
  top component with the held-out prompts' mean natural difference.
* **Reading.** keep reaching the per-prompt ceiling at k of 1 or 2 with
  the own subspace: the computation the control becomes is a task-level
  signal of that rank. The other-task subspace doing as well: the
  signal is a shared in-context subspace, not the task's own. keep
  still short of the ceiling at k = 8 while the random and background
  subspaces stay at the floor: the shared component is distributed and
  prompt-specific across prompts, and "one direction per prompt" was
  the whole story. k90 rising from the hand-over read point to the last
  one: dimensional expansion in the causal sense.

The grid stops at 8 by decision: the pool of 128 prompts bounds what a
higher rank could mean, and the question is whether the rank is small.
Cost: two captured passes over the pool per task, then per read point
`|k_grid|` × 3 edits × (3 subspaces + `n_controls` random) = 60 passes,
180 per task at three read points, about the size of one construction's
patch grid; run on `configs/trajectories_subspace.yaml`, which carries
only what the test needs from the earlier stages (canonical strength,
primary layer, the learned vector's patch grid for the hand-over read
point, no diagnostics), on the seed-20260916 runs of iteration 10.

## 2026-09-20 — Two further model families

### D35. OLMo 3 7B and Gemma 4 12B under the unchanged protocol; the backend generalised to what they need

Context: every result so far is on one family (Qwen3, four sizes), so
the hand-over depth, the rank-one replacement and the sufficiency-
before-necessity gap could be properties of that family's training or
architecture. The families chosen maximise the distance from Qwen3 on
the axes that could matter: OLMo 3 (`allenai/Olmo-3-1025-7B`, a base
model with open data, norms after the sublayers, multi-head attention,
sliding-window attention on three layers in four, an untied unembedding,
a GPT-2-style tokenizer with no BOS) and Gemma 4 (`google/gemma-4-12B`,
the text decoder of a multimodal checkpoint, pre- and post-norms,
scaled embeddings, local and global attention layers whose heads have
different widths, a soft-capped final logit, a tied unembedding, a
SentencePiece tokenizer that prefixes a BOS). Llama 3 would have been
the smallest change and is left as a later, cheap addition.

Decision:

* **The protocol is unchanged.** `configs/pilot_olmo3_7b.yaml`,
  `configs/pilot_gemma4_12b.yaml` and their `learned_*` counterparts are
  the 8B configs with `model.name` replaced (enforced by
  `tests/test_configs.py`): the same tasks, pools, gates, controls,
  candidate depth fractions (layers 6, 10, 13, 16 of OLMo 3's 32 and
  10, 14, 19, 24 of Gemma 4's 48), strengths, seed 20260907 and batch
  size. The trajectory comparison runs on `configs/trajectories.yaml`
  as on Qwen3. No per-family tuning: a task that fails its few-shot
  gate on a family is a rejection of the task on that family, not a
  reason to change the gate.
* **The backend reads the architecture instead of assuming it.** The
  decoder configuration is the model's text config (the wrapper's
  `text_config` on a multimodal checkpoint); the block list is found
  under `model.layers` or `model.language_model.layers`; the final
  norm likewise. The head width is read per layer from the block's
  attention module and checked against its output projection (Gemma
  4's global layers use twice the local width), captured head outputs
  are padded to the widest and every per-layer use slices to the
  layer's own width. The scores apply the architecture's final-logit
  soft-cap where it has one (the tanh cap of Gemma), so the
  log-probabilities, margins and lens are the model's own; the
  unembedding directions are the pre-cap ones (the cap is monotone).
  The prompt prefix the tokenizer adds (a BOS or nothing) is recorded
  in the run's metadata. Everything else (the residual hooks at the
  block inputs and after the last block, the head hooks on the input
  of the output projection, the intervention mechanics) is unchanged,
  and the Qwen3 numerical path is untouched (the soft-cap is a no-op
  where the config has none; the padding is a no-op with one width).
* **Validation before spending.** Tiny random models of all three
  families (`ToyModelConfig.family`) run the whole backend test suite:
  the log-probabilities against the native forward, the interventions
  exact at the layer and zero before it, the head decomposition of the
  attention output to 1e-6, the gradients against finite differences,
  batching invariance. On the GPU, a smoke run of each model stops
  after the few-shot stage (`--stop-after fewshot`: the download, the
  device, the tokenisation of every target, the few-shot and zero-shot
  accuracies of the ten tasks, the metadata) before the full protocol
  is requested; only a model whose smoke run is clean and whose tasks
  pass the gates as Qwen3's did is run in full.
* **Order and compute.** OLMo 3 7B first (14.6 GB of bf16 weights, the
  24 GB tier in EUR-NO-1, about three dollars for the three runs), then
  Gemma 4 12B (about 24 GB of weights, the 80 GB tier in US-CA-2 as for
  8B, about ten dollars); the three runs per model are the pilot
  (head-mean control), the learned vector and the trajectory
  comparison. The OLMo 3 configs run at batch 32: the smoke run at
  batch 128 ran out of memory on the 24 GB card during the few-shot
  stage (multi-head attention keeps four times Qwen3's key/value
  activations), and the batch enters no statistic (D20 raised it from
  32 to 128 for wall time only; the batching invariance is tested).
  Going up a tier would have meant a cold cache in US-CA-2 at four
  times the hourly price for a batch the analysis does not need.
* **The head-mean vector on post-norm blocks** (found on the pilots).
  Both families normalise the attention output before adding it to
  the residual (OLMo 3: `residual + norm(attn)`; Gemma 4 the same
  inside pre-norms), where Qwen3 and Llama add it as it is. The
  function vector of D21 is the sum of the mean head outputs through
  the output projection, so on these blocks it is the *pre-norm*
  attention output: the model's own write of it has a fixed norm
  (‖w‖ of the norm's weight, 4.6 at OLMo 3's layer 13 against the
  vector's natural norm of 18–29) and, on Gemma 4, a per-coordinate
  re-weighting (the weight's cosine with the uniform vector is 0.83;
  0.98 on OLMo 3, so there the direction is kept). The vector's
  *direction* is kept as the paper defines it (the head outputs are
  the patched and averaged quantity); the consequence is in the
  natural unit: ρ = 1 is 1.4–3.2 × the stream norm on OLMo 3 (0.2–0.4
  on Qwen3 and Gemma 4) and about four times the model's own write.
  The reference rule still selects it (the reliable ranges start at
  ρ = 0.05), so the OLMo 3 head-mean numbers at ρ = 1 are a strong
  intervention with large collateral damage; the comparison across
  families rests on the learned vector (whose unit is the stream norm
  everywhere) and on the trajectories' strength factors. The backend
  records the fact per run (`attention_output_normed`, checked on the
  toy families). A write-faithful composition (the layer's post-norm
  applied to the selected heads' sum) was not adopted: the norm acts
  on the whole attention output, so a partial sum through it is no
  more the model's write than the raw sum is.
* **Readable read points, and a norm-matched null for the edits**
  (found on the Gemma 4 pilot; D29 amended). The commitment curves of
  Gemma 4 reversed the effect from read point 33 to 46 of 48 under
  both edits, and the random-direction controls, whose removal should
  change nothing, lost most of the effect too. A probe
  (`scripts/probe_edit_sensitivity.py`) showed the mechanics exact (an
  all-zero edit reproduces the steered run bit for bit) and the model
  hypersensitive: a random edit of norm 10 on a residual of norm ~100
  at the query token costs 6–9 nats on the *unsteered* model from read
  point 30–33 on, norm 1 costs 0.6 nats at 40–44 and 5 nats at the last
  read point (residual norm 9), where Qwen3 shows no measurable change
  for the D29 edits of norm 1–8. The cause is architectural: the
  residual at the query token is dominated by one massive coordinate
  (91 % of its energy in the middle of the stack; the head-mean vector
  of antonym has its largest coordinate there, and the steering shifts
  it by 20–120) and is read through norm weights with entries in the
  hundreds (the input norm of layer 24 peaks at 244, the final norm at
  604), so edits of a percent of the residual norm move what the
  deep blocks read by units. Any residual edit of the size the D29,
  D33 and D34 tests use is therefore uninterpretable there: it does
  not say what carries the effect, only that the state was pushed off
  its manifold. Two consequences, applied from here on:
  - the D29 summary reads the hand-over depths over the *readable*
    read points only, those at which the random edits keep their
    premise (random removal retains the effect within
    `readability_tolerance` = 0.1 of 1 and random keeping alone
    within it of 0), and lists the unreadable ones; on Qwen3 and OLMo
    3 every read point is readable, so nothing changes there;
  - the random null is matched in *size* as well as in shape: beside
    the same edit along random directions (whose size is the
    perturbation's own component along them, ‖δ‖/√d), each real edit
    is repeated with its own per-prompt magnitude (δ·v) along the
    random directions (`random_matched_*`; `matched_controls`), and a
    read point is readable only if that null is inert too. The
    shape-matched null alone passed the last read point of Gemma 4
    (its edit there has norm 0.02–0.09) where a real edit of norm 1
    costs 5 nats.
  The trajectories' patch and subspace tests carry shape-matched
  isotropic controls only; on Gemma 4 their read points past 30 (0.63
  of the stack) are read against the probe's sensitivity map, not
  taken at face value.

  *The mechanism, made explicit from the checkpoint*
  (`scripts/checkpoint_norm_gains.py`, which reads the norm weights
  through HTTP range requests without downloading the model). An
  RMSNorm block reads x / rms(x) ⊙ w. On Gemma 4 the query token's
  residual is dominated by one coordinate (1750) from about layer 8
  on, so rms(x) ≈ |x₁₇₅₀| / √d ≈ 3 in the middle of the stack and the
  ordinary coordinates, of size ~1, normalise to ~0.3. The learned
  weights compensate: from layer 12 on the input norm's weight at
  coordinate 1750 is 0.000 ± 0.003 (the massive coordinate is not
  read; it only sets the scale) while the *median* weight on the
  other coordinates is 33 at layer 12, 41 at 16, 16 at 20, 6 at 24,
  1–2 from 28 on, with maxima of 180–930 and 155–1722 coordinates
  above 50 at every layer past 12 (the pre-feedforward norm at layer
  12: median 170, 2916 of 3840 coordinates above 50); the final norm
  has median 7.4, 92 coordinates above 200 and 0.005 at coordinate
  1750. Qwen3-8B's norm weights have medians 0.35–2.0 and maxima
  1.6–10.9, and no coordinate sets the scale. So on Gemma 4 an
  additive edit of ε per residual coordinate is divided by ~3 and
  multiplied by 30–170 before a middle block sees it, and by up to
  600 before the unembedding: a one-percent edit of the residual is
  a 10–50 percent edit of what the block reads. The Euclidean metric
  of the residual stream, in which every steering, patching and
  removal edit of this project (and of the literature it follows) is
  made, is far from the metric the computation uses on this family.
  Before layer 12 the picture differs (coordinate 1750 carries weight
  22–55 at layers 0–4, before the massive activation exists), and the
  block after which it is established (11) multiplies its output by
  0.005.

  *Literature.* Both halves are documented, their combination at this
  degree is not something I have seen described. Outlier dimensions
  with large norm parameters: Kovaleva et al. 2021 (BERT Busters,
  arXiv 2105.06990: the outlier dimensions coincide with outlier
  LayerNorm scaling parameters, and disabling them collapses the
  model), Timkey & van Schijndel 2021 (arXiv 2109.04404), Puccetti et
  al. 2022 (arXiv 2205.11380), Dettmers et al. 2022 (LLM.int8,
  arXiv 2208.07339: emergent outlier features concentrated in a few
  dimensions from ~6.7B parameters on), Bondarenko et al. 2023
  (arXiv 2306.12929: outliers arise where attention heads need to do
  nothing, at the norm's input), He et al. 2024 (arXiv 2405.19279:
  outlier features and normalisation in training). Massive
  activations as fixed biases and attention sinks: Sun et al. 2024
  (arXiv 2402.17762), Xiao et al. 2023 (arXiv 2309.17453). Gemma's
  numerical fragility is known in practice (Gemma 2, arXiv
  2408.00118, introduced the post-norms and the logit soft-cap partly
  for stability; float16 overflow of its activations was widely
  reported), but the per-block output scalars and the norm gains of
  this checkpoint I know only from the modelling code and the weights
  themselves. What is *not* a quirk: RMSNorm is doing exactly its
  job; the model has learned to keep its content at scale ~1 on the
  residual next to a scale-setting coordinate and to read it back
  with large gains. It is a property of this checkpoint (possibly of
  the family's training recipe), not of normalisation as such, and
  its consequence for intervention work is general: additive
  residual edits presuppose that the residual basis is roughly the
  basis the model reads, and Gemma 4 shows a model for which it is
  not. A well-conditioned edit on such a model would be made in the
  read frame (x / rms(x) ⊙ w) or would leave the scale-setting
  coordinate and the high-gain coordinates untouched; neither is
  implemented. The probe's readable window on Gemma 4 ends
  where Qwen3's hand-over sits (0.6–0.8 of the stack), so on this
  family the hand-over can be seen to begin (removal of the head-mean
  direction leaves 0.8 of antonym's effect by read point 29–30) but
  not to complete.

Reading: the core quantities to compare across families are the
hand-over read point as a fraction of the stack (0.6–0.8 on Qwen3 for
every injection layer), the rank of the replacement (one, the task's
own mean natural difference) and the sufficiency-before-necessity gap
(growing with size on Qwen3). Agreement on both families makes them
properties of in-context task execution in this class of models;
disagreement on one of them locates what the Qwen3 result depended on.

Result (2026-09-21; STATUS.md, D35): the hand-over read point and
the rank-one own-task replacement replicate on both families (OLMo 3:
0.50–0.69 of the stack, keep-own rank one median 0.93; Gemma 4:
0.69–0.79, median 0.91), for every injection layer. The
sufficiency-before-necessity gap is family-dependent: on OLMo 3 7B
necessity arrives two to five blocks after sufficiency for four of
seven tasks and not at all for three; on Gemma 4 it cannot be measured
(the deep half is unreadable by residual edits). The two families
also exposed two architectural facts the protocol must know about:
post-norm blocks (the head-mean vector's unit) and per-block output
scalars (the per-block writing measure), recorded above.

## 2026-09-21 — The verbalisation test

### D36. Is the carrier at the hand-over a verbalizable task concept? The Jacobian lens on the post-trained Qwen3-8B

Context: `docs/POSITIONING.md` names three tests that would settle
whether the phenomenon is a property of control or a restatement of
where answers appear. The verbalisation test asks whether the rank-one
direction that carries the effect at the hand-over, the task's mean
natural difference at that read point, *names the task* when read
through a lens, as the concept vectors of the Transformer Circuits
workspace paper do through the Jacobian lens.

Decision:

* **The lens.** The Jacobian lens of that paper (Anthropic's `jlens`,
  Apache 2.0): one matrix per block, the average Jacobian of the
  final residual with respect to the block's output, fitted by
  Neuronpedia on 1000 WikiText contexts and published for named
  checkpoints (`neuronpedia/jacobian-lens`). A lens linearises one
  model's downstream computation, so it is used only on the
  checkpoint it was fitted for. None of the project's six checkpoints
  has one (the Qwen3 lenses are fitted on the post-trained models,
  OLMo 3's on a forward pass transformers ≥ 5.13 no longer runs, Gemma
  4 12B has none), so the test runs on **`Qwen/Qwen3-8B`**, the
  post-trained sibling of the 8B base model, under the unchanged
  protocol (`configs/pilot_qwen3_8b_post.yaml`,
  `configs/learned_qwen3_8b_post.yaml`; the same prompts, no chat
  template), which also gives a base-versus-post-trained comparison
  of every core quantity. The lens is applied as published, not
  refitted, and read with the model's own final norm and unembedding
  (the backend's logit lens with the transported vector).
* **Index convention.** The lens is keyed by block output; read point
  `m` (the input of block `m`) uses the matrix of block `m − 1`, and
  the last read point is the identity (`directions.verbalise.Lens`,
  tested).
* **What is read** (`scripts/verbalise.py`, on the stored
  population-mean trajectories of the trajectories run): the mean
  natural difference at every read point; the learned vector's and the
  head-mean vector's mean perturbation at every read point; the
  injected vectors at their injection read point; the generic response;
  and, as the floor, random directions of the natural difference's norm
  at each read point. Each through the Jacobian lens and, for the
  natural difference and the injected vectors, through the plain logit
  lens.
* **What counts as naming the task.** Two numbers per readout beside
  the top tokens: the probability mass on a fixed list of task words
  per task (`TASK_WORDS`, written before any readout was seen: for
  antonym "opposite", "antonym", "contrary", …; for plural "plural",
  "many", …; for past_tense "past", "tense", "ago", …), with the best
  word and its rank; and the mass on the task's *answer* tokens (the
  first target token of every evaluation prompt), which is the
  competing reading. The random floor gives the masses a chance level
  at every read point. Reading: task mass and rank rising above the
  floor around the hand-over while the answer mass stays low says the
  carrier is a task concept; answer mass dominating says it is the
  answer being written; neither above the floor says the direction is
  not verbalizable in this lens.
* **Exploratory**, not preregistered: the word lists are a judgement,
  the lens is a linearisation averaged over generic text, and the
  readouts are of population means. A positive result is a lead for
  the mechanism, a negative one closes only this lens.

Result (`STATUS.md`, D36): the carrier at the hand-over is not
verbalizable in this lens. The mean natural difference names the task
at 0.50–0.61 of the stack on the tasks that have a name in the
vocabulary (antonym "opposite" rank 1, mass 0.20; singular "single"
rank 1; last_antonym and plural weakly) and reads as nothing at the
hand-over read point (0.69) and after it; the learned vector's
perturbation never names the task at any read point and reads as the
answer only on the number tasks late in the stack; the head-mean
perturbation names the output form (uppercase words, plural nouns,
past-tense verbs), as the direct decoding of function vectors does in
Todd et al. 2024; the generic response is never verbal; the plain
logit lens reads the same as the Jacobian lens wherever either reads
the task. Test C of `docs/POSITIONING.md` is therefore negative for the
direction the experiment measures, with the exploratory lead that the
verbalizable concept precedes the hand-over.

### D37. The landmark test: does the hand-over coincide with a depth defined without the control?

Context: the hand-over sits at a fixed read point of the stack for any
injection layer (D33–D35: 0.6–0.8 of the stack on Qwen3, 0.50–0.69 on
OLMo 3, 0.69–0.79 on Gemma 4), so it belongs to the stack, not to the
control. Test A of `docs/POSITIONING.md` asks what else has a knee
there, measured with no steering. Written before any landmark was
read; the criteria below are fixed so that a coincidence cannot be
found after the fact among 28–48 read points.

Decision:

* **The hand-over at read-point resolution.** The patch test of D33
  (keep only the steered perturbation's component along the prompt's
  natural difference) is run at *every* downstream read point of the
  primary layer (`configs/trajectories_landmarks.yaml`: 64 fractions of
  the downstream depth, which lands on every read point for at most 32
  downstream blocks), for the learned and the head-mean vector. The
  hand-over of a task is the first read point at which the keep
  retains 0.9 of the effect (the D33/D34 share); the model's hand-over
  is the median over tasks. The learned vector's is the primary
  quantity; the head mean's is a landmark of its own.
* **Landmark (i), the pool's rank.** The uncentred spectrum of the
  pool's natural differences (D34's pools, extraction plus calibration)
  at every read point (`subspace.pool_spectrum_all_read_points`). The
  knee: the first read point from which the top component explains
  ≥ 0.7 of the energy at every later read point (primary); the first
  read point reaching 0.9 of the profile's maximum (secondary). The
  participation ratio is recorded beside it.
* **Landmark (ii), the verbal onset.** The logit-lens readout of the
  stored mean natural difference at every read point
  (`scripts/verbalise.py` without a lens; D36's word lists, unchanged).
  The onset: the first read point at which the best task word ranks
  within the top 10; the exit: the last such read point; none where
  the task is never named. On the checkpoint with a published lens the
  Jacobian-lens readout is recorded as well, but the comparison across
  models uses the logit lens on every model, so that the instrument
  is the same.
* **Landmark (iii), the workspace band**, on `Qwen/Qwen3-8B` only,
  the one checkpoint with a published Jacobian lens; no other model
  gets a substitute for it. The paper's own six lens-quality prompt
  sets (`data/jlens_evaluations`, copied from the companion repository
  with their readout positions) are read through the Neuronpedia lens
  at every read point at the paper's readout position; the rate at
  which an intermediate ranks within the top 10 (also top 1 and 5) per
  read point is the profile; the onset is the first read point at
  which that rate reaches half of its maximum, the exit the last, the
  peak its argmax; per set and pooled.
* **Landmark (iv), the universal heads.** The selected heads of the
  head-mean run (`fv_heads`); block `l` writes to read point `l + 1`;
  the landmark is the median read point over the selected heads, with
  the band (min–max) recorded.
* **The comparison.** Per model, every landmark's offset from the
  learned vector's hand-over in read points, per task and as the
  median. Across models (the four Qwen3 Base sizes at seed 20260916,
  OLMo 3 7B and Gemma 4 12B at seed 20260907, and the post-trained
  Qwen3-8B), the scatter of landmark depth against hand-over depth,
  both as fractions of the stack, one point per model, with Spearman's
  rank correlation and a permutation p-value. A landmark is said to
  coincide with the hand-over only if p ≤ 0.05 across models *and* the
  median absolute offset over models is ≤ 2 read points. With six or
  seven models the test has little power; a landmark that fails the
  correlation but sits within two read points on every model is
  reported as "co-located, not shown to co-vary".
* **What the outcomes mean.** A landmark that coincides makes the
  fixed read point a property of the model's own computation named by
  that landmark (the pool's collapse to one direction, the task's
  verbal form, the heads' write depth, the lens's workspace band). None
  coinciding leaves the hand-over a property of how a rank-one bump is
  rotated into the natural direction, and test B remains.
* **Runs.** `scripts/landmarks.py` runs the comparison and the
  readouts in one job per model (one model load for the comparison,
  one for the readouts); `scripts/landmark_summary.py` aggregates.
  The comparison's own seed and determinism check are those of the
  trajectories runs. The prior trajectories runs are not reused for
  the hand-over (their grid is eighths); their pools are re-captured
  by the same code.
* **Exploratory**, not preregistered beyond the criteria above: the
  knee thresholds are judgements (0.7 of the energy, rank 10, half the
  maximum rate), fixed here before the profiles were seen.

Amendments made while running (before the results were read across
models): the pool's uncentred spectrum turned out to be rank one from
read point 1 on the first model (its top component is the pool mean),
so the spectrum about the mean was recorded beside it, labelled
exploratory; a readout's rank counts as a verbal reading only with
task-word mass ≥ 10⁻³ (Gemma 4's final norm makes some difference
vectors read out flat, every token tied, which the strict rank
reported as rank 1), and ranks count ties against the word; the
target-bearing eval prompts are read at the token before the target
under the paper's joint tokenisation (the first run read a trailing
space token). The plain logit lens was recorded beside the Jacobian
lens in the band.

Result (`STATUS.md`, D37): one landmark coincides with the hand-over,
the write depth of the universal heads: median offset −3 to 0 read
points on all seven checkpoints (|offset| median 1.0), the band of
head write points containing the hand-over on every model, and the
two depths co-varying across models (hand-over 0.59–0.71 of the
stack, heads 0.52–0.68; Spearman ρ = 0.85, permutation p = 0.015),
which meets the rule above. The head-mean vector hands over at the
same read point as the learned one. The verbal window, where there
is one, opens 2–6 read points before the hand-over and closes at it;
the pool's uncentred rank is degenerate (rank one from read point 1)
and its centred peak sits 8–10 read points below the hand-over on the
Qwen3 Base models and elsewhere on the others; the published lens's
band on `Qwen/Qwen3-8B` (the paper's six eval sets) lies above the
hand-over (top-10 rate rising from 0.78 of the stack to 0.17) and is
matched by the plain logit lens. The fixed read point is therefore
the depth at which the model's own in-context heads write; test B
now has its blocks named.

### D38. The source test: which sublayer writes the increments that turn the injected direction into the natural one, and is it necessary?

Context: D37 places the hand-over at the write depth of the universal
heads on seven checkpoints. Test B of `docs/POSITIONING.md` asks who
writes the steered run's aligning increments there: in the steered
zero-shot run there are no demonstrations to read, so the increments
that align the perturbation with the natural difference are produced
either by the attention sublayer (the heads, attending to the query
itself) or by the MLP sublayer reading the steered residual. Written
before any decomposition was read.

Decision:

* **What is captured.** Every block's attention write and MLP write
  at the query token (`ModelBackend.run(capture_sublayers=True)`: the
  attention and MLP module outputs on pre-norm blocks, the
  post-sublayer norm outputs on OLMo 3 and Gemma 4, so that the two
  writes sum to the residual increment exactly; tested on the three
  toy families and checked in every run as the median relative
  residual of the split), in three passes on the held-out prompts of
  every task: the unsteered zero-shot run, the steered run (the
  learned vector at the primary layer, canonical strength: the same
  vector and strength as D33–D37) and the demonstration run (the same
  few-shot construction as the trajectories runs). `directions source`
  (`configs/source.yaml`).
* **The aligning write of a sublayer** at block `l` is the component
  of its increment (steered minus unsteered, or demonstrations minus
  none) along the prompt's natural difference at read point `l + 1`,
  the read point the write lands on; the two components sum to the
  aligning increment of the block. Per block: their medians with
  bootstrap CIs, the norms of the four increments, the cosine of the
  steered attention (MLP) increment with the natural attention (MLP)
  increment.
* **The window shares.** Over the blocks from the injection layer to
  the hand-over read point of D37 (the landmark run's per-task
  hand-over of the learned vector), the aligning writes are summed per
  sublayer and prompt, and attention's share of the total is read; the
  same after the hand-over and over all downstream blocks, and the
  same for the natural run. Reading: a window is *attention-written*
  if the median share is ≥ 0.6 with its CI above 0.5, *MLP-written*
  if ≤ 0.4 with the CI below 0.5, mixed otherwise. The question of
  test B is whether the steered window to the hand-over is
  attention-written like the natural one, or MLP-written.
* **The selected heads.** From the per-head outputs (before the
  output projection), the selected heads' share of the attention
  write's aligning component over the blocks that hold selected heads:
  exact on pre-norm blocks (the write is linear in the head outputs),
  the pre-norm attention output on OLMo 3 and Gemma 4.
* **The random floor.** Random directions injected at the learned
  vector's norm (four per task) give the split of a perturbation that
  carries no task: their aligning components against the natural
  difference and the norms of their increments.
* **Necessity.** Each sublayer's aligning component is removed from the
  steered run over the window to the hand-over by subtracting it at the
  read point it lands on (the D29 edit at the query token, one
  intervention per read point, the removed vectors those of the
  unedited steered run), and the effect retained (per-token gain over
  the unsteered run, relative to the steered run's) is read against
  random per-example directions of the same norms at the same read
  points (two matched controls, the paired excess test of D29). The
  whole increment is removed the same way. Reading: a sublayer is
  *necessary* if its removal retains ≤ 0.5 of the effect while the
  matched random removal retains ≥ 0.9; the comparison of the two
  sublayers is the answer. One block at a time, each block's aligning
  component is removed alone, for the profile.
* **Runs.** The six checkpoints of D37 with their landmark runs (the
  post-trained 8B is not repeated: its core quantities equal the Base
  model's). Seeds, determinism check (a repeated steered pass with the
  capture, bit-identical) and bootstraps as in the trajectories runs.
* **Exploratory**, not preregistered beyond the criteria above: the
  thresholds (0.6/0.4 for the shares, 0.5/0.9 for necessity) are
  judgements fixed here.

Amendment made while running: Gemma 4 multiplies a block's output by
a per-block scalar after both adds, so the writes are captured in
the frame they land in and a block's increment carries a third,
rescaling term, `(scalar − 1)` times the block's input, recorded
beside the two writes; the first Gemma run, which subtracted unscaled
writes, is superseded.

Result (`STATUS.md`, D38): the aligning increments of the steered run
are MLP-written on every model and task (attention 0.10–0.29 of the
aligning write over the window to the hand-over, medians 0.14–0.17),
and the natural run's are MLP-written too with attention a third
(0.20–0.46, medians 0.21–0.37); the steered run reproduces 0.62–0.79
of the natural MLP aligning sum and 0.25–0.39 of the natural
attention aligning sum; the universal heads write 0.19–0.67 of
attention's own aligning write. Attention's aligning components are
never necessary (removal retains 0.54–1.02, median 1.00, against
random 0.99–1.02); the MLPs' are necessary by the rule on 7 of 8, 5 of
9, 3 of 10, 1 of 10 tasks from the 0.6B to the 8B and 1 of 9 on OLMo
3, and their whole increments on nearly every Qwen3 task with p ≤
0.001, where the matched random removal is itself destructive at
those norms. Gemma 4's removals are unreadable by the D29/D35 rule
(the window reaches the hypersensitive deep half). Test B is
answered: the control substitutes for the heads' delivery, and the
MLPs of the same blocks perform the transformation in both runs.

### D39. The k-th word family: a parameterised task whose parameter is a position

Context: every task run so far is a fixed map from one word to one
token, so the control is a selector of a fixed relation, and the one
parameterised family, add-k (D30), has only head-mean vectors, which
carry nothing about the operand; its learned vector exists for one
operand (add_3, D31). The next question of `docs/POSITIONING.md`
(whether a control is a value on a dial or a choice among programs)
needs a family whose parameter is ordered and whose vectors can be
compared and mixed. Add-k alone cannot answer it: its five vectors
share a numeric-format direction (cosines 0.87–0.99 between the
head means), so any structure across k would be confounded with the
format. A second family whose parameter is a position rather than a
number, and whose outputs are words rather than digits, controls that
confound, and selecting a position is an attention operation where
adding k is arithmetic in the MLPs, so a parameter that behaves the
same way in both families is a property of the control rather than of
one circuit. Written before any run of the family.

Decision:

* **The task** (`kth_word`; `src/directions/tasks.py`): a list of five
  distinct words drawn from the union of the single-word lists with a
  fixed items seed, the target its k-th word, k = 1..5 as five labels
  `kth_1`..`kth_5` of one registry task. The lists are the same for
  every k (the family shares its inputs and differs only in the
  target), the inputs are unique by construction and independent of
  the run seed, and 500 lists fill the three pools without reduced
  splits. Scoring is the protocol's (all target tokens; 47 of the 1209
  words are two tokens on the Qwen3 tokeniser and are kept). Nothing
  else in the protocol changes.
* **Family-only configs, as for add-k** (`configs/kth_<model>.yaml`
  with the head-mean control, `configs/kth_learned_<model>.yaml` with
  the learned vector; identical to the pilot and learned configs but
  for `name` and `tasks`, enforced by `tests/test_configs.py`). The
  universal heads, the other-task controls and the common direction
  are taken within the family. No other task is re-run: every later
  stage reads tasks from a run directory, so the family's runs stand
  beside the existing ones.
* **The full chain on the family**: the head-mean run and the learned
  run (seed 20260916 on Qwen3, as the runs of D37–D38; 20260907 on
  OLMo 3), then the comparison (`configs/trajectories.yaml`), the
  landmark comparison and readout (`scripts/landmarks.py`) and the
  source test (`configs/source.yaml`), each with the criteria of
  D32–D38 unchanged. Gemma 4 is configured but not run until the
  budget allows; its removals are unreadable in the deep half (D35,
  D38) in any case.
* **What is read, preregistered**: (i) qualification per k under
  both controls, and the head-support gate per k (D30's question for
  a position: does the mean output of any head set carry *which*
  position?); (ii) the hand-over read point, its landmarks and the
  sublayer shares per k, read by the D37/D38 rules (are they the
  lexical tasks' or add-k's?); (iii) the within-family structure of
  the learned vectors at the primary layer: the pairwise cosines
  across k against the seed-to-seed spread the learned run records
  (three fits per layer), and whether the cosine falls with |k − k′|
  (Spearman over the ten pairs, one-sided, permutation p ≤ 0.05
  reads "ordered"); and (iv) the other-task rows of the comparison's
  subspace test (a k control's effect kept along the other k′ pools'
  directions), which say whether the family's carriers are one
  direction or five.
* **Not in this decision**: the mixing test (a control interpolated
  between two k's, read as a distribution over positions) and the
  learned add-k run it is compared with; they are the geometry test
  proper and get their own decision once the family's backbone
  exists.

Amendment made after the first head-mean runs (Qwen3 0.6B and 8B
Base, five-word lists; the few-shot probe of shorter lists on the
0.6B on the CPU, `--stop-after fewshot`): the middle positions of a
five-word list fail the few-shot gate on both models (accuracies
1.00 / 0.55 / 0.30 / 0.32 / 0.94 on the 0.6B and 1.00 / 0.48 / 0.27 /
0.28 / 0.92 on the 8B for positions 1–5), and a four-word list loses
its second-to-last position on the 0.6B (1.00 / 0.59 / 0.37); a
three-word list qualifies at every position (1.00 / 0.70 / 0.98). The
family is therefore the three-word list, k = 1..3 (`kth_1`..`kth_3`
in the same configs), which leaves the ordered test of reading (iii)
without power (three labels give six label permutations, so the
one-sided p cannot fall below 1/6); reading (iii) is reduced to the
cross-label cosines against the seed spread and, exploratory, whether
the two neighbouring pairs' cosines exceed the end-to-end pair's. The
mixing test (the later decision) becomes the family's real test of
order. The five-word head-mean runs (0.6B: positions 1, 2 and 5
qualify with the head mean supported at 81–94 % of the gap restored
and the same selected layer; 8B: positions 1 and 5) and the 0.6B's
five-word learned run are kept as data on the five-word labels and
are not carried through the later stages; the pending five-word
learned run on the 8B was cancelled before it started.

Result (`STATUS.md`, D39; five checkpoints, the three-word family):
(i) every position qualifies under both controls where it passes the
few-shot gate (the middle position fails on the 4B, 0.47), and the
head-support gate passes at every qualifying position (0.75–1.03 of
the gap restored): unlike add-k's operand, a position *is* carried by
the mean output of the universal heads. (ii) The hand-over, landmarks
and sublayer shares of the qualifying end positions read as on the
lexical tasks (rank-one carrier, MLP-written aligning increments,
attention's components never necessary), with two differences: the
family's carriers overlap (the other positions' pool directions keep
0.50–0.83 of a position's effect at the hand-over, where other
lexical tasks kept ≤ 0.4), and the learned control's hand-over sits
above the universal heads' write depth on the larger models (offsets
+1 to +8 read points against −3 to 0 in D37); the middle position's
learned control never hands over (its keep-along-natural curve
plateaus below 0.9 on every model). (iii) The head-mean vectors are
close and ordered by position on every model with three labels (the
neighbouring pairs' cosines 0.76–0.94 exceed the end-to-end pair's
0.60–0.86), while the learned vectors are near-orthogonal across
positions (0.04–0.19) at or below their seed spread (0.08–0.23) and
anti-aligned in effect (another position's learned vector costs
12–23 nats; another position's head mean helps by 0.4–3.0 nats on
all but two rows). The mixing test of the geometry test proper
therefore has a construction with a geometry to interpolate, the
head mean, and one without, the learned vector.

### D40. The geometry test: is a control a value or a choice? Mixing two labels' controls and reading the intermediate

Context: D39 gives a parameterised family whose head means are
ordered by position and whose learned vectors are not; D30 gives the
add-k family with head means that carry no operand and, so far, a
learned vector for one operand. The positioning's second test asks
whether a control is a continuous parameter of one computation (a
value on a dial) or a selector among programs (a choice). The
vectors' geometry alone cannot answer it (the head means of both
families share a large common component; the learned fit is not
unique, D31). The behavioural version can: mix two labels' controls
and read what the model does with a control it was never given.
Written before any mixed control was run.

Decision:

* **Vectors and families.** For a model and a family, two end labels
  a and b with their unit directions u_a, u_b under one construction
  and one injection layer: the mixed control at weight t ∈ {0, 0.1,
  …, 1} is v(t) = α · normalise((1 − t) u_a + t u_b), injected at the
  query token of the zero-shot prompt, with α the calibrated strength
  of label a at that layer (the head-mean norm or the learned radius;
  the endpoints t = 0 and t = 1 are then a's control and b's
  direction at a's strength). The layer is a's selected layer; the
  head-mean vector is one vector regardless of layer, and the learned
  vector at that candidate layer is taken from b's fit there. The
  families and end labels: the k-th word family (D39), positions 1
  and 3, on Qwen3 0.6B–8B and OLMo 3 7B, under both constructions;
  the add-k family, operands 1 and 5 (and 2 and 10 where both
  qualify), under the learned vector from the family-only learned
  runs of this decision (`configs/arith_learned_<model>.yaml`: the
  learned config with the five `arithmetic` operands of D30, scored
  on the changed digits) on Qwen3 0.6B–8B, and under the head mean
  where D30's head means qualified (add-1 on the 4B and 8B only, so
  the head-mean mix of add-k is at most one pair); and the lexical
  baseline, antonym and plural, from the existing runs (the
  seed-20260916 learned and head-mean runs on Qwen3, the 20260907
  runs on OLMo 3), under both constructions.
* **Readout.** On the held-out evaluation prompts of label a (the
  k-th word lists are the same for every position; the add-k numbers
  are the same for every operand), at every t the teacher-forced
  log-probability of every candidate continuation of the family: the
  three listed words for the k-th word family (the position
  distribution, normalised over the three), the five operands' results
  n + 1, 2, 3, 5, 10 for add-k (changed-token scoring, D30). For the
  lexical baseline, whose two tasks have different inputs, the readout
  is each task's own held-out effect (d(lp/tok) of the own target on
  the task's own prompts) at every t: two effect curves.
* **Null.** Mixing dilutes the endpoint control; a rise of an
  intermediate label's mass in the middle of the path may be the
  endpoint's effect fading rather than a new control. The null is the
  same path from u_a towards a random unit direction (and from u_b
  towards another), `n_null` random directions per side at the same
  norm: the dilution curves. A candidate's mass at weight t is
  compared with its mass on the dilution curves at the same t, paired
  over prompts (the paired excess test of D18, one-sided).
* **Readings, per model, family and construction.** *Parameter*: the
  intermediate label's mass (position 2; the operands between a and
  b) exceeds, at some interior weight 0.2 ≤ t ≤ 0.8, both endpoints'
  values of it and the dilution null's at that t (p ≤ 0.05), and its
  curve over t has an interior maximum. *Switch*: the two end labels'
  masses cross, and the intermediate label never exceeds the dilution
  null (the mix is one endpoint or the other, or neither). *Neither*:
  the endpoints' masses do not cross (the mix is not even a switch;
  the path leaves the family). The lexical baseline is expected to
  read as a switch and calibrates what one looks like. A family reads
  as a parameter under a construction when at least three of the four
  Qwen3 sizes with a qualifying pair read so; the constructions are
  reported separately, as D39 found only the head means ordered.
* **Superposition.** The unnormalised sum u_a + u_b at the norm α is
  the t = 0.5 point up to a factor √2 · cos(θ/2) in norm; it is read
  from the same curve and not run separately.
* **Exploratory**: the same curves for the position pair (1, 2) and
  (2, 3) where the middle qualifies; the pair's path length (the angle
  between u_a and u_b) beside the curves.

Amendment made while running: the first 8B and 4B jobs crashed on the
add-k head-mean construction, for which no head-mean run is given
(a path was built before the skip); fixed, the constructions without
a run are skipped and noted, and the jobs re-run.

Result (`STATUS.md`, D40; five checkpoints): by the preregistered
rule neither family is a parameter under either construction. The
k-th word family reads as a parameter under the head mean on the
0.6B and 8B (and OLMo 3) and under the learned vector on the 1.7B
and 4B, two of four Qwen3 sizes each; add-k on one pair each of the
0.6B and 4B. The magnitudes decide the reading: where the head means
are strong controls of their positions (the 0.6B and OLMo 3, own
masses 0.87–0.97), their mix puts 0.30 and 0.25 of the mass on the
middle word at t = 0.6 against 0.19 and 0.09 under dilution
(p 0.001); where they are weak (the 1.7B, 4B and 8B, own masses
0.48–0.78) the path has no range. The learned mix is a
near-deterministic switch on every model (own masses 1.00 → 0.00,
the middle word never above 0.10), and add-k never produces an
intermediate operand (masses 0.00–0.05). The lexical baseline is a
switch on every row. So: the optimiser's control is a choice; the
heads' control can be a value, when it is a control at all.

### D41. The composition test: is a control converted once, or once per step? Composed tasks read against their components

Context: every task so far is one step (a fixed map from one word to
one token), so the transition the project studies, a control
converted by the MLPs before the heads' write depth into the task's
own direction (D37, D38), has only been seen as one event. The
positioning's first test asks what happens when the function has an
intermediate: whether the control is converted into the model's own
state for step one and that state into the composition's (two
hand-offs), or whether the model holds the composed relation as one
relation that the control selects (one hand-off). The two composed
tasks of the registry hand over at or after the later of their
components on every checkpoint (D37; up to nine read points after),
which says the question is measurable and nothing more. D39 and D40
showed the two constructions to be different objects (the learned
vector a fitted switch, the head mean the model's own write with a
geometry), so the test runs under both and states its prediction per
construction. Written before any composed control was read against
its components.

Decision:

* **The family.** One family-only config per model in the D39
  pattern (`configs/comp_<model>.yaml`, the pilot protocol; `comp_learned_<model>.yaml`,
  the learned protocol; the D39 stages on top): four single-step
  components, antonym, uppercase, plural and the last word of a
  three-word list (`kth_word`, k = 3), and five compositions of them
  through the `compose` registry task (the first step's items with
  their outputs mapped through the later steps): uppercase∘antonym
  (hot → COLD), uppercase∘plural (cat → CATS), antonym∘last (the
  registry's `last_antonym`: "a, b, hot" → cold), uppercase∘last
  ("a, b, c" → C) and the three-step uppercase∘antonym∘last
  ("a, b, hot" → COLD). Components and compositions sit in one run so
  that the references below share the seed, the pools and the read
  points. Qwen3 0.6B, 1.7B, 4B and 8B Base and OLMo 3 7B; Gemma 4 is
  configured and not run (its deep half is unreadable under the
  patch edits, D35 and D38, and the composed hand-over comes late).
* **The gate probe.** Before any run is requested, the family's
  few-shot gate is probed on the CPU on the 0.6B (D39 lost the
  middle positions of five-word lists to the gate after a run was
  requested). A composition that fails the few-shot gate on a
  checkpoint is excluded there and recorded as "the model cannot do
  it", which is a different finding from "no control exists", and
  the two are never pooled.
* **The measurement (the landmark comparison with references,
  `configs/trajectories_composition.yaml`).** For each composition
  and each of its components in step order, the component's natural
  difference is captured on the composition's own held-out prompts:
  the component's demonstrations (its evaluation items) before the
  composed query, minus the zero-shot run, at every read point
  ("what the model writes for antonym alone on this input"). At every
  read point of the patch grid (every downstream read point, D37) the
  composed control's perturbation is read against it: e(m), its
  cosine with the component's difference minus its cosine with the
  composition's own natural difference, paired over prompts, and the
  same difference for the matched isotropic control (the generic
  response aligns every perturbation with every natural difference
  late in the stack, D32, and the difference of two cosines does not
  remove it). Beside it, causally, the keep and remove edits of the
  patch test along the component's difference (D33 mechanics, the
  same random controls), and the cosine between the two natural
  differences themselves (the model's own computation read the same
  way).
* **Readings, per composition, component and construction.** A read
  point is positive when e(m) > 0 by the one-sided paired bootstrap
  and the paired excess over the isotropic control (both p ≤ 0.05),
  negative when the same holds for e(m) < 0; a window is two
  consecutive grid read points (the project's sustained rule, against
  the multiplicity of read points). *Staircase* (two hand-offs): a
  positive window before the composition's hand-over (the learned or
  head-mean hand-over of D37, on the composed task), then a negative
  window after it: the perturbation is closer to the component's
  write than to the composition's first, and to the composition's
  after. *Composed only* (one hand-off): no positive window. *Component
  only*: a positive window and no negative window after it: the
  control never becomes the composition's own. For the three-step
  case the two inner components are read separately and the order of
  their excess peaks is recorded. A verdict holds for a construction
  when at least three of the four Qwen3 sizes with a qualifying
  composed control read so (D40's rule); OLMo 3 is the out-of-family
  check.
* **Predictions, per construction.** The learned vector, a fitted
  switch that selects its label (D39, D40), is expected to read as
  composed only, and its hand-over is expected to come later than the
  components' (D37's hint). The head mean, the heads' own write, is
  the construction under which a staircase can appear; no direction
  is predicted for it. The lexical two-step compositions with the
  same inputs as their component (uppercase∘antonym, uppercase∘plural)
  are the cleanest reads; the list compositions carry the
  extractive step first.
* **Serial injection (a second job, after the reading above).** The
  first component's control at its own layer and strength, the second
  component's at a grid of deeper layers, read as the composed task's
  effect on its held-out prompts against the second control's
  matched random directions at the same layer; and the two controls
  added at one layer (superposition). Under the head mean first (D40:
  two learned controls added are a switch or destructive), the
  learned vector as the secondary. Bright only past the first
  control's hand-over: the second stage reads a finished
  intermediate; bright everywhere: the controls add; nowhere: the
  composed relation is not assembled from its parts as controls.
  Its design is fixed when the first reading is in, and recorded as
  an amendment here.
* **Source stage.** `directions source` runs on a composition only
  where a staircase is read, with the window split at the first
  positive window; there is nothing for it to split otherwise.
* **Exploratory**: the components' demonstrations scored on the
  composed target (the few-shot log-probability of COLD under antonym
  demonstrations), the natural-cosine curve's peak, and the
  hand-over offsets between the composition and its components.

Gate probe (the 0.6B on the CPU, few-shot accuracy on the evaluation
pool; gate 0.5): antonym 0.80, uppercase 0.91, plural 0.98, last
word 0.96, uppercase∘plural 0.73 and uppercase∘last 0.89 pass;
uppercase∘antonym 0.44, antonym∘last 0.27 (as in the main runs, where
`last_antonym` never qualified on the 0.6B) and the three-step
uppercase∘antonym∘last 0.16 fail. So on the smallest checkpoint the
family has two two-step compositions and the antonym compositions
are "the model cannot do it"; the larger checkpoints (where
`last_antonym` qualified from the 1.7B on) are expected to carry all
five, which the runs decide. The family runs as configured (the gate
excludes per checkpoint); the stages are the two family runs and the
landmark comparison with references (the all-to-all trajectories
comparison of D32 is not run: the reference readout is the
composition's version of it), the source stage as stated.
