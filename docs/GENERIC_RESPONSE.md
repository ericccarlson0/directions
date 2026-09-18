# The generic response

The line of research on the *generic response* of the residual stream to
an injected vector, condensed: what it is, how it was measured, what the
four models showed, what the literature says, and what was left open.
This line is closed as of 2026-09-18: no further tests of the generic
response are run, and the open hypotheses below are recorded so they are
not re-derived. The quantities stay in the `trajectories` command
(docs/DECISIONS.md D32 as amended, D33), because the generic-removed
variant of every alignment is the one the results quote.

## What it is

Inject any vector of one residual norm at the query token at layer l
and follow the perturbation δ_m = h_m(x; h_l + v) − h_m(x) through the
later read points m. For random isotropic directions v the downstream
perturbations are not independent: a large part of each one's δ_m is
the same vector, per prompt, whatever direction was pushed. That shared
part is the generic response, defined operationally as the mean
trajectory of the isotropic controls at the layer, per example and read
point, g_m(x) = mean_k δ_m(v_k)(x) over the K = 12 controls (4 per
construction × 3 constructions) at a strength factor.

It was found by its symptom in the first iteration-8 runs (0.6B and
8B): the floor of the trajectory comparison, the cosine between a random
direction's trajectory and the natural few-shot trajectory, rose with
depth from zero at injection to a median of 0.14–0.30 over pairs at the
last read point, and up to 0.5–0.6 on some tasks, with a jump at the
final block. A raw cosine of 0.7 with the natural trajectory over a
floor of 0.5 is a smaller alignment than it reads.

## How it is measured

All of this is in `src/directions/trajectories.py` and is written per
task to `core/tasks/<task>/trajectories.json` and
`trajectories_arrays.npz` (the mean generic response in float16).

- **Removal (D32, amended).** The third variant of every pairwise cosine
  projects g_m(x) out of both vectors, per example and read point. A
  control's own floor removes the mean of the *other* controls
  (leave-one-out), so that a floor is not deflated by removing a control
  from itself. The answer-direction variant (the final norm's scale
  times the first target token's unembedding row projected out) is the
  same idea for the output; both are kept so the two can be told apart.
- **Cosine with the answer direction**, per read point, to check that
  the two removals take out different things.
- **Coherence (D32 amendment, D33).** The norm of the random responses'
  mean over the mean of their norms, per example and read point, pooled
  and per construction: 1 when every random push produces the same
  downstream change, 1/√K when the responses are unrelated. This is the
  one per-example quantity.
- **Diagnostics (D33)**, per strength factor and layer, from the
  population mean of g: the fraction of its energy in its own k largest
  coordinates and in the k largest coordinates of the unsteered residual
  at the same read point (k = 1, 4, 16, 64; the massive-activation
  coordinates, listed); its cosine with the mean residual; its norm
  against the residual's; its logit lens at the last read point (the
  mean random response added to each prompt's final residual, through
  the final norm and the unembedding in float32: ten promoted and ten
  demoted tokens, mean absolute logit change); and its cosine across
  strength factors.
- **Two strengths (D33).** Everything above at the canonical strength
  and at half of it, to separate a nonlinearity of large pushes from a
  linear feature of the downstream map.

The figures per task and layer show the alignment with the natural
trajectory raw, without the answer direction and without the generic
response side by side, and the coherence and the energy in the
residual's largest coordinates through depth.

## What the four models showed

Runs: iteration 8 (STATUS, workflow runs 35248187883, 35253499155,
35247747527, 35247750874) and iteration 9 (35296239774, 35296599331,
35296680682, 35296238273), seed 20260907, one run per model; the numbers
are ranges over tasks at each task's primary layer.

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| Coherence, one block after injection → last read point (x1) | 0.35–0.53 → 0.53–0.81 | 0.35–0.58 → 0.66–0.84 | 0.34–0.45 → 0.56–0.67 | 0.35–0.52 → 0.48–0.76 |
| The same at half strength (x0.5) | 0.32–0.51 → 0.42–0.74 | 0.33–0.56 → 0.46–0.76 | 0.33–0.47 → 0.44–0.56 | 0.33–0.51 → 0.41–0.76 |
| ‖g‖ / ‖h‖ at the last read point, x1 (x0.5) | 0.11–0.16 (0.04–0.08) | 0.05–0.23 (0.03–0.18) | 0.05–0.14 (0.02–0.04) | 0.04–0.14 (0.02–0.09) |
| Energy in the residual's 16 largest coordinates (of d) | 0.29–0.64 (1024) | 0.17–0.49 (2048) | 0.03–0.19 (2560) | 0.06–0.40 (4096) |
| Energy in the largest 64 | 0.39–0.70 | 0.36–0.76 | 0.08–0.36 | 0.11–0.56 |
| Cosine with the mean residual, mid-depth | −0.63 to −0.77 | −0.42 to −0.82 | −0.48 to −0.76 | −0.29 to −0.61 |
| The same at the last read point | −0.32 to −0.82 | −0.56 to −0.97 | −0.84 to 0.45 | −0.59 to 0.59 |
| Cosine between the x1 and x0.5 generic responses, end | 0.35–0.88 | 0.73–0.97 | 0.67–0.93 | 0.64–0.96 |
| Cosine of the mean natural trajectory with the mean generic response, end | 0.27–0.67 | 0.50–0.83 | −0.36 to 0.78 | −0.11 to 0.63 |
| Logit lens promotes | fragments, function words ('CUR', 'number', 'and', '=') | punctuation (',', '.', ':') | fragments, ':' and ',' (least regular) | punctuation, Chinese function words (',', '.', '不', '和', '当然') |
| Logit lens demotes | ' yes', ' Sure', ' options' | ' Indeed', ' Typically', ' Essentially', ' True' | ' option', ' typically', markup | task-content words (' produces', ' performs'; ' islands', ' hurricane') |

The last row is computed from the saved population means
(`trajectories_arrays.npz`, `mean_delta_icl` against `L<l>_mean_generic`)
and is not in STATUS; the rest is quoted from the iteration-9 entries.

What these settle.

- **It is a residual-stream phenomenon, not the answer's.** The generic
  response is orthogonal to the answer's unembedding direction on every
  model (|cos| ≤ 0.14 at every read point), the answer variant changes
  no alignment by more than 0.03, and the readout is punctuation,
  function words and the model's default tokens. With g removed the
  isotropic floors sit at 0.01–0.04 in the median at every read point.
- **It is not a large-push nonlinearity.** The coherence is the same at
  half strength as at the canonical one, while the response's size
  scales with the push (0.04–0.23 of the residual's norm at x1, 0.02–0.18
  at x0.5). Many input directions are sent to one output direction by a
  map that is, at these norms, linear in the push.
- **It is present from the first block.** A third to a half of a random
  push's downstream change is already the shared part one block after
  injection; the receiving block itself responds along a common
  direction. Coherence then rises through depth to 0.5–0.85.
- **At mid-depth it points against the background on every model**
  (cosine with the mean residual −0.29 to −0.82), and still at the end
  on the two small models; on 4B and 8B the end-point cosine has either
  sign, so a prompt-dependent part is larger there.
- **Its energy lies in the massive-activation coordinates in proportion
  to their weight.** Strongly on 0.6B (up to 0.64 of the energy in 16 of
  1024 coordinates), weakly on 4B; on 8B most of it is put there by the
  last block (0.04–0.36 of the energy in the 16 largest before it,
  0.11–0.56 after), the same block that pulled the learned vector's
  alignment down at full strength on that model.
- **Its readout is a mild move toward generic tokens** (mean absolute
  logit change 0.3–0.7 on 0.6B), demoting answer-opening adverbs and
  task-content words.

In one sentence: a shrinkage of the background along the massive
coordinates, with a drift toward the unconditional defaults, at a
magnitude of a few percent of the residual, produced by any push of
these norms.

## Relation to the other quantities

- **The natural trajectory carries a share of it.** The population-mean
  few-shot difference has a cosine of 0.27–0.83 with the mean generic
  response at the last read point on the two small models, and of either
  sign on 4B and 8B (table above): demonstrations also shrink the
  background, most clearly where the generic response is most
  stereotyped. The
  generic-removed variant discards that share from the natural
  trajectory too, so its cosines compare what is *specific* to each
  trajectory. The removal does not change the iteration-8 reading (the
  fitted vector's trajectory collapses progressively onto the natural
  one); it lowers every late cosine by the floor's amount and puts the
  floors at zero, which is what makes the peaks and declines readable.
- **The patch test does not depend on it.** The causal result of
  iteration 9 (keeping only the component along the prompt's natural
  difference retains the effect, removing it destroys it, the natural
  difference alone gives it) is measured on raw perturbations and is
  not conditioned on any removal.
- **The damage measure (D24)** is the KL from the unsteered next-token
  distribution at the query token. A share of any construction's damage
  is plausibly this background shrinkage rather than the task content,
  since the generic response reads out as a shift toward default tokens;
  this was not decomposed.
- **Strength.** At half strength the learned vector aligned with the
  natural trajectory at least as well as at the canonical strength and
  the late declines on 1.7B and 8B disappeared, while the generic
  response halved in norm and kept its coherence: the canonical strength
  overshoots on those models, and the generic response is the one part of
  the perturbation that grows in proportion.

## What the literature says

Checked on 2026-09-18, after the iteration-9 diagnostics, not before them
(the diagnostics were written from memory of the anisotropy, rogue-dimension
and massive-activation papers; the mechanisms below have a literature of
their own that would have framed the norm hypothesis better from the
start).

- **RMSNorm dilution.** Every block's RMSNorm maps its input onto a
  fixed-norm sphere, so a block does not see the extra norm of h + v,
  only its direction, in which the prompt's own content is attenuated by
  1/√2 when ‖v‖ = ‖h‖ (Angular Steering, arXiv 2510.26243). Every later
  block reads a diluted prompt and writes less of what it would have
  written: this is the anti-alignment with the mean residual measured at
  mid-depth. An angle-norm decomposition of steering finds that concepts
  live mainly in the angular structure while the norm governs the
  stability and the downstream effect of an intervention (arXiv
  2606.06735), and several methods now inject on the sphere or restore
  the norm (GeoSteer, arXiv 2609.10658).
- **Attention-sink absorption.** Sinks limit mixing and make the model
  less sensitive to token perturbations, more so in larger models
  (Barbero et al., "Why do LLMs attend to the first token?", arXiv
  2504.02732); a perturbed query attends more to the sink, whose value
  contribution is one fixed vector per layer, which would give a coherent
  response from the first block on. The sink tokens carry the massive
  activations, a few fixed dimensions acting as implicit biases in keys
  and values (Sun et al., "Massive Activations in Large Language
  Models", arXiv 2402.17762), which ties this route to the coordinate
  overlap measured here.
- **Massive-activation regeneration.** Models rebuild the start-token
  outlier in a protected channel when the architecture removes its usual
  home, so the massive coordinates are functional rather than incidental
  ("Massive Activations Are Architecturally Robust", arXiv 2606.20743);
  a perturbed state being re-equipped with them is consistent with the
  8B last-block result.
- **Confidence regulation.** Entropy neurons hedge by writing onto an
  effective null space of the unembedding and scaling the final norm
  (Stolfo et al., "Confidence Regulation Neurons in Language Models",
  arXiv 2406.16254). Orthogonal to the answer direction, a weak logit
  lens, answer-opening tokens demoted: the generic response has the
  signature of a hedging response.
- **Rogue dimensions and cosine.** A few dimensions dominate cosine
  similarity in late layers and hide representational structure;
  standardising dimensions repairs it (Timkey and van Schijndel, "All
  Bark and No Bite", arXiv 2109.04404; the anisotropy of late residuals
  in Ethayarajh 2019, arXiv 1909.00512). The generic-removed variant is
  a per-prompt version of that repair.

The literature favours a combination over one cause: dilution of the
prompt's content through RMSNorm inside every later block, absorbed
partly by the attention sink, with the confidence-regulation machinery
as its readout. Nothing was found on the prompt-dependent part seen on
4B and 8B.

## Open hypotheses, not pursued

Each is decidable on the saved runs at small cost. They are recorded
here so that the tests exist on paper; none is run.

| Hypothesis | Prediction | Test | Cost |
|---|---|---|---|
| Pure norm effect: the response is what every later block stops writing when its input is diluted | Reproduced by rescaling the residual at the injection layer to the norm a random push would give, adding no direction; anti-aligned with each prompt's *own* residual, not only the mean | One pass per layer and strength; compare its downstream change with g | Trivial |
| Attention redistribution to the sink | g equals the sink token's value contribution; attention mass on position 0 rises under any push | One pass with attention capture at the query token, steered against base | Small |
| Massive-activation regeneration | g's per-coordinate profile through depth follows where those coordinates are written | Analysis of the saved mean g against the mean absolute residual per layer | Saved arrays only |
| A prompt-dependent part (4B, 8B) | Part of g erases the prompt's own deviation from the mean, the rest is global | Decompose each prompt's g into the component along its own residual deviation and a remainder | One captured pass (the saved g is a population mean) |
| Drift toward the unconditional distribution | Entropy rises and the query-token distribution moves toward the prior under any push; g's energy lies in the unembedding's effective null space | KL to the unconditional distribution and entropy under random pushes; null-space share on the saved g | One pass; saved arrays |
| Cosine artefact of rogue dimensions | Standardised cosines give the same alignment as the generic-removed variant | A standardised-cosine variant of every pair | Saved arrays only |

One design consequence stays on the table without being decided: if a
pure rescaling reproduces g, a norm-preserving injection (add the vector,
rescale the residual back to its original norm) removes the confound at
the source instead of by projection afterwards, the floors sit at zero
without any removal, and a share of the learned vector's damage would
disappear with it. That would be a new decision (docs/DECISIONS.md)
and a re-run, and it is not made here.

## Known limits of what was measured

- One seed, one run per model; four isotropic controls per construction
  and strength, so the floors' quantiles are coarse.
- The diagnostics (energy, cosine with the mean residual, logit lens)
  are taken on population means, not per prompt; the coherence is the
  only per-example quantity.
- Two strengths, not a grid; the same layer set as the trajectory
  comparison (the learned run's selected layer, and the head-mean run's
  when different).
- The generic response is defined at the query token only, where the
  vectors are injected; nothing was measured at other positions.
