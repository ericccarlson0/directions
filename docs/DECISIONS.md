# Methodological Decisions

Every entry records a choice that is *not* fully determined by
`docs/EXPERIMENT.md`, why it was made, and how to change it. Where a choice was
ambiguous it is exposed as a configuration parameter rather than hard-coded, so
the alternative can be run without editing code.

---

## D1. PC1 is computed on **uncentered** paired differences

**Config:** `extraction.center_pca` (default `false`)

`docs/EXPERIMENT.md` asks for `v = PC1({d_i})` where `d_i = h_l(p_i^+) -
h_l(p_i^-)`. "PCA" conventionally mean-centers, but the mean of `{d_i}` is
precisely the task-general component of the contrast — the part that is shared
across queries and that the function-vector literature identifies as the
control signal. Centering would discard it and return the direction of largest
*variation between queries*, which is a different (and query-specific) object.

We therefore take PC1 to be the top right-singular vector of the raw difference
matrix, and orient its (otherwise arbitrary) sign to have positive inner
product with the mean difference. The centered variant is available via
`extraction.center_pca: true`.

**Diagnostics recorded:** `explained_variance_ratio` and
`mean_diff_cosine` = cos(PC1, mean_i d_i) per extraction seed. A
`mean_diff_cosine` near 1 confirms that the shared component dominates.

---

## D2. The steering evaluation is **zero-shot**, extraction is few-shot

**Config:** `data.n_shot_extraction`, `data.n_shot_qualification`,
`data.n_shot_eval` (default `10 / 10 / 0`)

`docs/EXPERIMENT.md` requires "a held-out causal steering effect" and asks to
select the strength that "reliably improves held-out task performance", but
does not state the evaluation prompt format. With the same few-shot prompt used
for extraction the model is already at or near ceiling, so there is no headroom
and no steering effect could be observed regardless of the direction's quality.

The evaluation prompts are therefore zero-shot: the query alone, in the same
template. This is the standard function-vector protocol — extract from the
contrast between working and broken demonstrations, then test whether injecting
the direction makes the model perform the task without demonstrations.

Task qualification ("does this model do this task at all?") remains few-shot,
since that is what it is meant to measure.

---

## D3. Three disjoint splits: extraction / calibration / evaluation

**Config:** `data.n_extraction`, `data.n_calibration`, `data.n_eval`

The specification names an extraction set and a held-out set. Selecting the
intervention layer and strength is itself a fit to data, so doing it on the
final held-out set would contaminate the reported steering effect. Queries are
partitioned into three disjoint pools by a seeded permutation; demonstrations
for a prompt are drawn from that prompt's own pool and never include the query
itself.

The absolute strength `alpha = rho * median_x ||h_l(x)||` is computed once from
the **calibration** split and reused unchanged at evaluation, so the selected
strength transfers as a fixed vector norm rather than being re-fit. The
evaluation-split median norms are recorded as a diagnostic.

---

## D4. Behavioural metrics are teacher-forced, not generated

**Config:** `intervention.selection_metric` (default `accuracy`)

Three exact metrics are computed from a single forward pass over
`prompt + target`:

* `accuracy` — the argmax token at every target position equals the target
  (teacher-forced exact match); identical to greedy generation for
  single-token targets;
* `target_logprob` — teacher-forced summed log-probability of the target
  sequence, as `docs/EXPERIMENT.md` requests for multi-token outputs;
* `logit_margin` — first-target-token logit minus the largest competing logit.

Free-running generation is avoided because it costs one forward pass per
generated token and introduces decoding choices that are not part of the
scientific question. For multi-token targets teacher-forced exact match is an
upper bound on greedy exact match; this is documented rather than hidden.

---

## D5. Residual-stream indexing and the meaning of "intervene at layer l"

There are `L + 1` residual read points for `L` transformer blocks:
`resid[0]` is the embedding output, `resid[l]` is the output of block `l-1`,
and `resid[L]` is the final pre-norm residual. Block `l` maps
`resid[l] -> resid[l+1]`, so `b_l = delta_{l+1} - delta_l` is exactly block
`l`'s contribution.

An intervention "at layer `l`" adds `alpha * v` to `resid[l]`, i.e. to the
*input* of block `l`, at the final query token. It is defined for
`0 <= l <= L-1`. Because attention is causal, `delta_l = 0` for every layer
upstream of the intervention; the ratio metrics there are reported as `null`
rather than as spurious zeros.

---

## D6. `N_l` is reported with both a centered and an uncentered inherited subspace

**Config:** `layerwise.variance_fraction` (default `0.9`)

`docs/EXPERIMENT.md` defines `P_l` as the projector onto the top
right-singular subspace of the **centered** `D_l`. Two consequences of
centering were found during implementation, and both are handled explicitly
rather than silently:

1. At `l = l_intervention` every `delta_l(x)` is identical (`= alpha * v`), so
   the centered `D_l` is exactly zero and `P_l` is undefined. `N_l` is reported
   as `null` at that layer.
2. Centering removes the shared perturbation component — which is the control
   direction itself. A block that merely *propagates* the control direction
   would then score as if it had created a new direction, which inverts the
   metric's intended interpretation.

The spec-faithful centered `N_l` is reported as the primary metric. An
uncentered companion `N_uncentered_l`, whose `P_l` includes the shared control
component, is reported alongside it and is the one to use when distinguishing
propagation from creation. The same is done for effective rank
(`d_eff` centered, `d_eff_uncentered` alongside).

---

## D7. Random controls are matched on layer and norm, in two families

**Config:** `controls.n_random`, `controls.kinds`

Every control uses the same intervention layer, the same token, and the same
`alpha` (both vectors are unit norm, so the injected norm is identical).
`isotropic` controls are uniform random unit vectors; `orthogonal` controls are
uniform random unit vectors in the orthogonal complement of the control
direction, which removes the possibility that an isotropic control accidentally
carries a component along it. Controls alternate between the kinds.

---

## D8. Qualitative profile labels and block ablation are exploratory

**Config:** `exploratory.*`

The five qualitative labels (`conserved_transmission`, `delayed_activation`,
`cascade`, `early_expansion`, `localized_transformation`) are produced by a
deterministic rule over the primary metrics, with every threshold recorded next
to the label. They are written to `exploratory/` and never to `core/`, and are
a summary of the numbers, not evidence in their own right.

The block-ablation causal check (`docs/EXPERIMENT.md`, "Causal Validation of
Important Layers") is likewise secondary. It removes `-b_l(x)` from block `l`'s
output during a steered run (necessity) and injects `+b_l(x)` during an
unsteered run (sufficiency), at the intervened token.

---

## D9. Statistics in float64, model execution in bf16

Model weights and activations follow `run.dtype` (bf16 on GPU per the
specification, float32 on CPU where bf16 is poorly supported). Every captured
residual is immediately cast to float32, and all PCA/SVD/statistics run in
float64. Full activation tensors are never persisted; only the derived
per-layer metrics are saved.

---

## D10. The calibration/steering metric is teacher-forced log-probability per target token

**Config:** `intervention.selection_metric` (default `target_logprob_per_token`)

Measured on Qwen3-0.6B-Base: zero-shot exact-match accuracy on the four
word-mapping tasks is **0.000 at every candidate layer and every strength in
the swept range**. The model does not even adopt the answer format zero-shot
(top continuations for `"Q: sincere\nA:"` are `" S"`, `" The"`, `" A"`), so
accuracy carries no gradient with which to calibrate anything, and a threshold
on it is a coin flip. The same runs show a large, orderly effect in
log-probability (antonym at layer 14, rho 0.5: −7.64 → −5.52 nats/token).

`docs/EXPERIMENT.md` already prefers "teacher-forced target sequence
log-probability for multi-token outputs", so the default selection and
steering-qualification metric is now the **per-token** version, which is
comparable across tasks with different target lengths. Accuracy, summed
log-probability and the logit margin are still computed and reported for every
condition; only the *decision* metric changed. Thresholds are correspondingly
in nats per token.

Few-shot task qualification ("does this model do this task at all?") still uses
accuracy, which is the right metric for that question and is well away from the
floor there (0.67-1.00 across the five tasks).

---

## D11. The strength grid extends past rho = 0.3

**Config:** `intervention.strengths`

`docs/EXPERIMENT.md` sketches `rho in {0.01, ..., 0.3}` but also says to sweep
"as you see fit". On Qwen3-0.6B-Base nothing measurable happens below
rho ~ 0.3, the effect peaks around rho 0.5-0.8, and it collapses (log-prob far
below baseline) by rho ~ 1.2 as the residual stream is driven off distribution.
The grid is therefore `[0.02 ... 1.0]`, which brackets both the onset and the
peak while stopping short of the collapse.

---

## D12. "Reliably improves" is enforced by two statistical tests, not a threshold

**Config:** `intervention.min_improvement`, `intervention.max_selection_p`,
`intervention.control_screen_n`, `intervention.control_screen_max_p`

`docs/EXPERIMENT.md` says to select the earliest layer and smallest strength
that *reliably* improves held-out performance. A bare effect-size threshold does
not implement "reliably": with ~48 calibration examples the sampling noise on
the metric is comparable to the threshold, so the earliest/smallest rule
reproducibly locked onto whichever grid point happened to fluctuate upward
first. On the first Qwen3-0.6B run this selected layer 6 / rho 0.3 for
`arithmetic_add` (+0.10 accuracy, noise) while layer 11 / rho 0.3 gave +0.31;
the selected point then failed the held-out random-control comparison and the
task was discarded despite having a large real effect.

A grid point is now selectable only if all three hold:

1. its improvement clears `min_improvement`;
2. a **paired one-sided bootstrap** over calibration examples gives
   `p <= max_selection_p` (default 0.05);
3. it beats a **matched random-control screen** of `control_screen_n` (default
   8) directions at the same layer and norm, at
   `p <= control_screen_max_p` (default 0.15).

The earliest-layer / smallest-rho preference is then applied to the surviving
points, unchanged. The screen runs on the *calibration* split with its own
control seed, so the final held-out comparison against 16 fresh controls
remains an independent test.

---

## D13. Seeds are derived from a stable digest, never from `hash()`

**Bug found and fixed during the first real runs.** Sub-generators were keyed on
`abs(hash(task_name))`. CPython salts string hashing per process
(`PYTHONHASHSEED`), so two invocations of the *same config* drew different
demonstrations and different random controls — visible as 10-shot accuracy
drifting between 0.656 and 0.672 and layer-6 stability between 0.89 and 0.97
across identical commands.

All seed derivation now goes through `mathx.stable_key`, a BLAKE2b digest of
the key parts. `tests/test_extraction_and_controls.py` runs the derivation
under three different `PYTHONHASHSEED` values and asserts the results agree.

---

## D14. The layerwise profile is also measured at the strongest reliable strength

**Config:** `layerwise.also_measure_strongest` (default `true`)

The preregistered operating point is the *smallest* reliable strength, which is
the right choice for measuring propagation with minimal off-distribution
distortion but leaves open whether the measured amplification profile is a
property of the control direction or of the injection magnitude. The full
layerwise measurement is therefore repeated at the strongest reliable strength
at the *same* layer, written to `exploratory/`, together with rank correlations
between the two profiles for `log G_l`, `d_eff` and `N_l`. This is a robustness
check, not a second preregistered measurement.
