# Measurement-Validation Pilot

## Goal

Primary Question:

> Where, and in what form, is a low-dimensional control signal transformed into downstream computation?

## Models

Run in order:

1. `Qwen/Qwen3-0.6B-Base`
2. `Qwen/Qwen3-1.7B-Base`

Use:

* Hugging Face Transformers
* PyTorch hooks
* bf16 model execution
* float32 for PCA/SVD/statistics
* no quantization
* no training

## Candidate Tasks

Start with ~5 deterministic mappings.
Tasks must admit automatic scoring.

* antonym
* singular → plural
* present → past
* English → French
* simple arithmetic mappings

## Task Qualification

For each model/task:

1. Evaluate few-shot ICL performance.
2. Reject tasks below a configured accuracy/log-probability threshold.
3. Extract control directions using multiple seeds.
4. Require reasonable cross-seed direction stability.
5. Require a held-out causal steering effect.
6. Require steering to outperform matched random controls.

Only qualified task/control pairs proceed to the layerwise experiment.

## Initial Control-Direction Extraction

For each task, construct paired prompts:

* positive: correct demonstrations
* negative: same inputs with demonstration outputs permuted

At candidate layer \(l\), record the final query-token residual:

$$
d_i = h_l(p_i^+) - h_l(p_i^-)
$$

Extract the first Principal Component of \(\{d_i\}\):

$$
v_{t,l} = \operatorname{PC1}(\{d_i\})
$$

Normalize \(v_{t,l}\) to unit norm.

Use at least 3 independent extraction seeds.

## Intervention Calibration

Candidate intervention layers:

* approximately 20%, 30%, 40%, 50%, 60% through model depth

Sweep intervention strengths as you see fit:

$$
\rho \in \{0.01, ..., \ 0.3\}
$$

where

$$
\rho =
\frac{\|\alpha v\|}
{\operatorname{median}_x \|h_l(x)\|}
$$

Select the earliest layer and smallest strength that reliably improves held-out task performance.

Intervene at the final query token.

## Layerwise Measurements

For each held-out input \(x\), record baseline and steered residuals at every downstream layer.

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

Use at least ~16 random controls per validated direction in the main pilot.

Report primary metrics relative to the random-control distribution where appropriate.

## Behavioral Metric

Prefer exact automatic metrics:

* classification accuracy when applicable
* target-token logit margin
* teacher-forced target sequence log-probability for multi-token outputs

## Causal Validation of Important Layers

For layers with unusually large conversion or new-subspace creation:

1. Run the steered forward pass.
2. Remove or restore the steering-induced block contribution at that layer.
3. Measure loss of the steering-induced behavioral improvement.

Note that this is secondary to the main geometric measurements.

## Initial Run Sizes

Per task:

* candidate tasks: ~5
* extraction examples: 64
* held-out evaluation examples: 64
* extraction seeds: 3
* random directions: 16

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
* generated figures

Avoid saving e.g. full activation tensors unless needed for debugging.

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

### Aggregation

Perturbation magnitude and gain are defined per example:

$$
S_l(x)=
\frac{\|\delta_l(x)\|}
{\|h_l^{\mathrm{base}}(x)\|}
$$

$$
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

## Qualitative Profile Labels

Use these only after computing quantitative metrics:

* conserved transmission
* delayed activation
* cascade
* early expansion
* localized transformation

Any clustering/classification procedure should be automatic.

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
