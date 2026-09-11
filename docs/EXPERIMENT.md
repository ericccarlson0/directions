# Measurement-Validation Pilot

## Goal

Primary Question:

> Where, and in what form, is a low-dimensional control signal transformed into downstream computation?

## Models

Run in order:

1. `Qwen/Qwen3-0.6B-Base`
2. `Qwen/Qwen3-1.7B-Base`
3. `Qwen/Qwen3-4B-Base`
4. `Qwen/Qwen3-8B-Base`

Use:

* Hugging Face Transformers
* PyTorch hooks
* bf16 model execution
* float32 for captured activations, float64 for PCA/SVD/statistics
* no quantization
* no training

Note that bf16 execution has a relative precision of \(2^{-8}\). Any geometric quantity computed from *differences* of residual streams can therefore be contaminated by rounding wherever the true difference is small or near-constant across examples. The pipeline should measure this noise floor rather than assume it is negligible.

## Candidate Tasks

Start with ~5 deterministic mappings.
Tasks must admit automatic scoring.

* antonym
* singular → plural (and its inverse)
* present → past, present participle
* upper-casing, number → words, simple arithmetic mappings
* composite tasks (third iteration): antonym of the last word of a list; arithmetic followed by number → words

Tasks the models solve zero-shot (two-operand addition) or whose direction never calibrates (English → French on the smallest model) were dropped in the third iteration (docs/DECISIONS.md D19).

Each task needs enough unique items for three disjoint query pools (see *Data Splits*). An item whose target does not tokenize within a configured maximum number of tokens should be dropped automatically, and every dropped item should be written to the run's rejection log.

## Prompt Regimes

Two prompt regimes:

* **few-shot** (8 demonstrations or so): for task qualification ("does this model do this task at all?") and for control-direction extraction;
* **zero-shot** (only the query, same template): for intervention calibration, held-out steering, and all layerwise measurements.

The evaluation regime must leave behavioural headroom, and with few-shot prompts the model is frequently at or near ceiling (no steering effect could be observed regardless of the direction's quality).

## Data Splits

Partition each task's items into three disjoint query pools:

| split | used for |
|---|---|
| extraction | paired prompts for direction extraction |
| calibration | layer/strength selection |
| evaluation | held-out steering, controls, layerwise measurement |

Selecting the intervention layer and strength is itself a fit to data, so it must not happen on the final evaluation pool. Demonstrations for a prompt are drawn from that prompt's own pool and never include the query itself.

## Behavioral Metrics

All metrics come from a single teacher-forced forward pass over `prompt + target`:

* **target log-probability per token** — teacher-forced log-probability of the target sequence divided by its token count. This is the decision metric for calibration and for the held-out steering test.
* target log-probability (sum);
* teacher-forced exact-match accuracy — the argmax at every target position equals the target;
* first-target-token logit margin — gold logit minus the largest competing logit.

Rationale: a small base model provided a bare zero-shot `Q:/A:` prompt could not even adopt the answer format, so zero-shot exact-match accuracy could sit at or near 0 across the entire calibration grid, producing no gradient with which to select anything. log-probability is exact, and per-token normalisation makes it comparable across tasks with different target lengths. Accuracy, log-probability-sum and the logit margin are still computed and reported for every condition.

Few-shot task qualification uses **accuracy**, which is the correct metric for "does the model do the task" and should not hit the floor.

## Task Qualification

For each model/task:

1. Evaluate few-shot ICL performance on the evaluation pool.
2. Reject tasks below a configured accuracy/log-probability threshold.
3. Extract control directions at each candidate layer using ≥ 3 seeds.
4. Require cross-seed direction stability at the candidate layer.
5. Require a held-out causal steering effect (statistically supported).
6. Require steering to outperform matched random controls: the real direction's mean held-out improvement must exceed the *mean* improvement of the matched random directions (isotropic, orthogonal and covariance-matched, same layer and norm), by a paired hierarchical bootstrap over examples and controls (one-sided \(p \le 0.05\); docs/DECISIONS.md D18). The rank of the real direction among the individual controls is reported but does not gate.

Only qualified task/control pairs proceed to the layerwise experiment. Every rejection is written to a rejection log together with the numbers that produced it.

## Control-Direction Extraction

For each task, construct paired prompts:

* positive: correct demonstrations
* negative/permuted: identical demonstration *inputs*, demonstration *outputs* deranged (a permutation with no fixed point)

At candidate layer \(l\), record the final query-token residual:

$$
d_i = h_l(p_i^+) - h_l(p_i^-)
$$

Extract the first principal component of \(\{d_i\}\):

$$
v_{t,l} = \operatorname{PC1}(\{d_i\})
$$

Do not mean-center before the PCA. The mean of \(\{d_i\}\) is the task-general component of the contrast (the part shared across queries that constitutes the control signal). To mean-center before the PCA would be to discard the task-general component and return the direction of largest variation *between* queries. Take PC1 as the top right-singular vector of the raw difference matrix. You can expose centering as a configuration option, off by default.

Normalize \(v_{t,l}\) to unit norm.

Use at least 3 independent extraction seeds (each resamples the demonstrations and the derangement). Report, per seed, the explained-variance ratio and \(\cos(\text{PC1}, \bar d)\), where stability is the minimum pairwise \(|\cos|\) between seed directions. The direction that we carry forward should be PC1 of the pooled differences across seeds.

### Canonical function vector (iteration 4, docs/DECISIONS.md D21)

With `extraction.control: function_vector` the direction carried forward is the function vector of Todd et al. (2024) instead of PC1, built from the same paired prompts:

* for every attention head \((l, j)\), the mean output at the final query token over the positive prompts, \(\bar a_{l,j}\), taken before the attention output projection;
* the average indirect effect of each head: the mean change of the probability of the correct target (teacher-forced over the whole target, which is the paper's first-token probability whenever the target is one token; docs/DECISIONS.md D22) on the deranged-label prompts when the head's output at the query token is replaced by \(\bar a_{l,j}\);
* one universal head set \(S\): the top \(k = 10\) heads by the mean indirect effect over all tasks that reached extraction;
* \(\mathrm{FV}_t = \sum_{(l,j) \in S} W_o^{l}[:, j]\, \bar a^{t}_{l,j}\), normalised to unit norm and used at every candidate layer.

Gates, controls and layerwise measurements are unchanged. The calibration grid is in units of the vector's own norm with the canonical (unscaled) injection as the reference strength (see "Intervention Calibration"). Report the natural strength \(\|\mathrm{FV}_t\| / \operatorname{median}\|h_l\|\) at each candidate layer, the cross-seed stability of the per-seed function vectors (the stability gate applies to it), and \(\cos(\mathrm{FV}_t, v_{t,l})\) as a descriptive comparison with PC1. PC1 is still extracted at every read point and remains \(v_{t,l}\) in the direction-specific readouts.

## Intervention Calibration

Candidate intervention layers:

* approximately 20%, 30%, 40%, 50%, 60% through model depth

Relative strength:

$$
\rho =
\frac{\|\alpha v\|}
{\operatorname{median}_x \|h_l(x)\|}
$$

Sweep a log-spaced grid over the following (or wider, if necessary):

$$
\rho \in [0.01,\ 1.0]
$$

(e.g. 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0). The effects of steering typically only become measurable at a sizeable fraction of the residual norm, peak, and then collapse as the stream is pushed off-distribution. The grid should bracket both the onset and the peak (so, if the best point sits at the upper edge of the grid, extend the grid). The absolute strength \(\alpha = \rho \cdot \operatorname{median}_x\|h_l(x)\|\) is computed once from the calibration pool and reused unchanged at evaluation.

For the function-vector control (docs/DECISIONS.md D22) the unit of \(\rho\) is the vector's own norm instead (`calibration.strength_unit: natural`, \(\alpha = \rho \cdot \|\mathrm{FV}_t\|\)), so that \(\rho = 1\) is the canonical unscaled injection of Todd et al.; the grid is 0.05–2.0 in those units (extended up to 4.0), and every grid point also records \(\alpha / \operatorname{median}_x\|h_l(x)\|\) for comparison with the PCA protocol.

### Selection Rule

Select the earliest layer and, within it, the smallest strength that reliably improves the decision metric on the calibration pool, where "reliably" is operationalised statistically: a paired one-sided bootstrap over calibration examples (\(p \le 0.05\)), a minimum improvement over the unsteered baseline, and a matched random-control screen (16 random directions at the same layer and norm) using the same paired excess test as the qualification gate (D18).

For the function-vector control the strength within the selected layer is instead the reliable grid point nearest the reference \(\rho = 1\) (in log distance; `calibration.reference_rho`), i.e. the paper's injection whenever it is reliable, and the weakest reliable strength is measured in the exploratory strength-robustness stage beside the middle and strongest ones. The layer rule is unchanged (`calibration.layer_rule: earliest`; `best`, the layer whose selected point improves most, is available).

Intervene at the final query token.

## Layerwise Measurements

For each held-out input \(x\), record baseline and steered residuals at the intervened token at every layer. (Gather at the measured token inside the hook so full activation tensors are never retained.)

Define:

$$
\delta_l(x)
=
h_l^{\text{steer}}(x)
-
h_l^{\text{base}}(x)
$$

and blockwise response:

$$
b_l(x)
=
\delta_{l+1}(x)
-
\delta_l(x)
$$

### Primary metrics

#### Perturbation magnitude

$$
S_l
=
\frac{\|\delta_l\|}
{\|h_l^{\text{base}}\|}
$$

#### Layerwise gain

$$
G_l
=
\frac{\|\delta_{l+1}\|}
{\|\delta_l\|}
$$

#### Block conversion

$$
C_l
=
\frac{\|b_l\|}
{\|\delta_l\|}
$$

#### Control alignment

$$
A_l(x) = \left|\cos\big(\delta_l(x),\, v\big)\right|
$$

The most direct test of conserved transmission: it equals 1 at the intervention layer by construction and stays near 1 only if the injected direction is propagated unchanged.

#### Direction-specific readouts

The metrics above are *geometric*: none of them knows what the task is, and a matched random direction of the same norm reproduces their profile (iteration 2). Two readouts tie the perturbation to the task content:

$$
T_l(x) = \cos\big(\delta_l(x),\, v_{t,l}\big), \qquad
\Gamma_l(x) = \cos\big(\delta_l(x),\, \nabla_{h_l(x)} \log p(\text{target} \mid x)\big)
$$

* \(v_{t,l}\) is the task's own control direction extracted (same rule, same pool) at read point \(l\), sign-aligned with the mean few-shot-minus-permuted difference; \(T_{l^*} = 1\) by construction. It asks whether the downstream perturbation stays inside the task's own later-layer representation of the control.
* \(\nabla_{h_l(x)} \log p(\text{target} \mid x)\) is the gradient of the summed target log-probability with respect to the *baseline* residual at the query token, from one backward pass per task (parameters frozen; same bf16 forward as everywhere else). \(\Gamma_l\) asks whether the perturbation points where the target log-probability increases; at \(l^*\) it is the cosine between the injected direction and the gradient. The signed first-order prediction \(\delta_l \cdot \nabla \log p\) is stored alongside.

Both are signed cosines, summarised like \(A_l\) (per-example, median with bootstrap CI) and compared per layer against every control kind. They are the place where behavioural specificity must appear if it is geometric at all: a covariance-matched random direction that steers less should also align less with the gradient.

#### Effective dimensionality

Across held-out inputs, stack:

$$
D_l =
[\delta_l(x_1),\ldots,\delta_l(x_N)]
$$

Compute:

* effective rank
* \(d_{90}\): number of PCs explaining 90% of variance

#### New-subspace creation

Measure the fraction of \(b_l\) lying outside the dominant perturbation subspace inherited from layer \(l\).

This distinguishes propagation from creation of new context-dependent directions.

## Random Controls

For every validated control direction:

* sample matched random directions
* match intervention layer
* match intervention norm
* include directions orthogonal to the control vector

Use two types, alternating: **isotropic** (uniform random unit vector) and **orthogonal** (uniform random unit vector in the orthogonal complement of \(v\)). Use at least ~16 random controls per validated direction in the main pilot.

Report every primary metric relative to the random-control distribution (per-layer \(z\) and empirical \(p\)). Several geometric quantities, such as \(N_l\) and \(d_{\mathrm{eff}}\), are only interpretable against this null, because a random direction of the same norm also propagates into "new" directions.

## Residual-Stream Indexing

For \(L\) transformer blocks there are \(L+1\) read points:

* `resid[0]` = embedding output (input to block 0);
* `resid[l]` = output of block \(l-1\) = input to block \(l\), \(1 \le l \le L-1\);
* `resid[L]` = output of block \(L-1\), before the final norm.

Block \(l\) maps `resid[l]` to `resid[l+1]`, so \(b_l = \delta_{l+1} - \delta_l\) is block \(l\)'s contribution. An intervention "at layer \(l\)" adds \(\alpha v\) to `resid[l]` (the *input* of block \(l\)) and is defined for \(0 \le l \le L-1\).

## Primary Figures

1. **control magnitude through depth**
   layer vs. median \(S_l(x)\), with bootstrap confidence intervals.

2. **layerwise amplification**
   layer vs. median \(\log G_l(x)\), with bootstrap confidence intervals.

3. **dimensional expansion**
   layer vs. \(d_{\mathrm{eff}}(D_l)\) and \(d_{90}(D_l)\).

4. **new-subspace creation**
   block/layer vs. \(N_l\).

5. **task × layer heatmaps**
   separate heatmaps across validated task/control pairs for:

   * median \(\log G_l\)
   * \(d_{\mathrm{eff}}(D_l)\)
   * \(N_l\)

6. **real vs. random controls**
   For each primary metric, compare the real control direction against the distribution from matched random controls.

Plus diagnostics: the calibration grid per task, and cross-seed stability per candidate layer.

Note that, when discussing a task's layerwise profile, the "amplification profile" refers collectively to its measured trajectories of \(S_l\), \(\log G_l\), \(d_{\mathrm{eff}}(D_l)\), and \(N_l\).

### Effective Dimensionality

For each downstream layer, stack perturbations across held-out inputs:

$$
D_l =
\begin{bmatrix}
\delta_l(x_1)^T\\
\vdots\\
\delta_l(x_N)^T
\end{bmatrix}
$$

Center \(D_l\) across examples before computing its SVD.

Report:

* effective rank:

$$
d_{\mathrm{eff}}(D_l)
=
\frac{(\sum_i \sigma_i^2)^2}
{\sum_i \sigma_i^4}
$$

* \(d_{90}(D_l)\): minimum number of singular directions explaining 90% of centered variance.
* the total centered variance of \(D_l\), per layer, so that a reader can tell where the spectrum is signal and where it is noise.

Also report the **uncentered** effective rank, since centering removes the shared control component.

Note that, at the intervention layer every \(\delta_l(x)\) equals \(\alpha v\) by construction, so the centered \(D_l\) is exactly zero and \(d_{\mathrm{eff}}\), \(d_{90}\) are undefined there. Report them as null.

### New-Subspace Creation

Stack the blockwise responses:

$$
B_l =
\begin{bmatrix}
b_l(x_1)^T\\
\vdots\\
b_l(x_N)^T
\end{bmatrix}
$$

Let \(P_l\) be the projector onto the top right-singular-vector subspace of centered \(D_l\) explaining 90% of its variance.

Define:

$$
N_l =
\frac{
\|B_l(I-P_l)\|_F^2
}{
\|B_l\|_F^2
}
$$

\(N_l\) is the fraction of steering-induced block response lying outside the perturbation subspace already present entering the block.

Interpretation:

* \(N_l \approx 0\): the block mostly transforms/amplifies directions already present.
* large \(N_l\): the block creates steering-induced variation in new residual-stream directions.

Two refinements are required:

1. Centering \(D_l\) removes the shared (mean) perturbation component (the control direction itself), so a block that merely propagates the control direction would score as if it created a new one. Report the centered \(N_l\) as the primary metric as well as an uncentered companion \(N_l^{\text{unc}}\) (whose \(P_l\) includes the shared component); use the uncentered one when distinguishing propagation from creation.
2. At the intervention layer the centered \(D_l\) is zero and \(N_l\) is undefined; report null.

### Numerical Noise Floor

In exact arithmetic the centered variance of \(D_l\) at the intervention layer is zero. In bf16 the observed value is the rounding of \(h + \alpha v\), which is isotropic and therefore *looks* high-dimensional. Record, for every condition:

* the observed \(d_{\mathrm{eff}}\) and centered variance at the intervention layer (as a diagnostic, never as a measurement);
* the ratio of that variance to the centered variance at the next layer.

This ratio can be used to check the numerical noise floor of every downstream dimensionality metric.

### Aggregation

Perturbation magnitude, gain, conversion, alignment, etc. are defined per example:

$$
S_l(x)=
\frac{\|\delta_l(x)\|}
{\|h_l^{\mathrm{base}}(x)\|}
\qquad
G_l(x)=
\frac{\|\delta_{l+1}(x)\|}
{\|\delta_l(x)\|}
$$

For plots, report the median across held-out examples with bootstrap confidence intervals.

For gain, plot \(\log G_l\), so:

* \(0\) = no magnitude change
* \(>0\) = amplification
* \(<0\) = attenuation

Effective dimensionality and new-subspace creation are computed from the full held-out-example matrices, not averaged per-example.

### Strength robustness (exploratory)

We preregister the *smallest* reliable strength (for the PCA control; the canonical strength for the function vector, D22), which is correct for measuring propagation with minimal off-distribution distortion but leaves open whether the measured profile is a property of the direction or of the injection magnitude. Repeat the full layerwise measurement at the other reliable strengths of the same layer that differ from the selected one, (a) the **weakest**, (b) the **strongest** and (c) a strength between the weakest and the strongest, and report rank correlations with the selected profile for \(\log G_l\), \(d_{\mathrm{eff}}\) and \(N_l\).
This should be written to the exploratory outputs (not the core outputs).

## Qualitative Profile Labels

Use these only after computing quantitative metrics:

* conserved transmission
* delayed activation
* cascade
* early expansion
* localized transformation

Any clustering/classification procedure should be automatic.

## Causal Validation of Important Layers

For layers with unusually large conversion or new-subspace creation:

1. Run the steered forward pass.
2. Remove the steering-induced block contribution \(b_l(x)\) at the intervened token (necessity), and separately inject \(+b_l(x)\) into an unsteered run (sufficiency).
3. Measure the fraction of the steering-induced behavioural improvement lost, and reproduced.

Note that this is secondary to the main geometric measurements.

## Initial Run Sizes

Per task:

* extraction examples: 64
* calibration examples: 64
* held-out evaluation examples: 64
* extraction seeds: 3
* random directions: 16 (plus ≥ 8 for the calibration screen)

These can be reduced for e.g. smoke tests.

## Outputs

Each run must save:

* resolved config
* model identifier
* git commit
* random seeds
* environment/package metadata
* task qualification results
* control-direction stability metrics
* intervention calibration results
* layerwise metrics
* random-control comparisons
* the numerical noise-floor diagnostic
* a cross-task analysis of the quantities that discriminate the propagation modes (see `docs/PROJECT.md`)
* a rejection log for automatic filtering decisions
* generated figures

Keep preregistered/core outputs and exploratory outputs in separate directories.
Avoid saving e.g. full activation tensors unless needed for debugging.

## Pilot Success Criterion

The pilot succeeds if at least a few tasks produce:

1. stable low-dimensional control directions;
2. reproducible held-out causal steering;
3. effects exceeding matched random controls;
4. interpretable layerwise amplification/transformation profiles.

And if Qwen3-0.6B fails, for instance, rerun the same protocol on Qwen3-1.7B before changing the methodology.

## Follow-up

After the pilot succeeds:

1. replace the PCA control with canonical function-vector extraction;
2. replicate on a second model family;
3. expand task families;
4. investigate component-level routing only after the layerwise effect is established.
