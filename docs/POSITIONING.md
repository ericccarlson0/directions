# Where the experiment stands, what would settle it, and how it maps onto the big programs

Written 2026-09-21, after D35 (three families, six models). A status
assessment, not a decision record; the numbers are in `STATUS.md`.

## What is established

Across Qwen3 (0.6B–8B, two seeds), OLMo 3 7B and Gemma 4 12B (one seed
each), for a control injected at 0.2–0.5 of the stack as a head-mean
function vector, a learned single vector or a PC1:

1. **Conversion, not transport.** By the last read point the
   perturbation is orthogonal to the injected direction (cosine
   0.00–0.06 for the learned vector). What the blocks add downstream
   are writes in mutually near-orthogonal directions (pairwise |cos|
   0.02–0.07), each correlated (0.4–0.9) with the write the same block
   makes when the demonstrations are present, each with a growing
   projection on the natural difference at that depth.
2. **A fixed read point of the stack.** From 0.6–0.8 of the stack on
   Qwen3, 0.50–0.69 on OLMo 3 and 0.69–0.79 on Gemma 4, the prompt's
   own natural difference carries ≥ 0.9 of the effect, for every
   injection layer, vector type and strength tried.
3. **Rank one, the task's own.** Across prompts that carrier is one
   direction, the task's mean natural difference at that depth (keep
   0.9–1.0 of the effect, other tasks' ≤ 0.4, random 0). Since the
   answers differ across prompts, this direction is not the answer's
   unembedding direction; it is a task-level signal.
4. **Necessity is family- and size-dependent.** Removing the rank-one
   component at the hand-over leaves −0.2 / −0.1 / 0.4 / 0.96 of the
   effect on Qwen3 0.6B / 1.7B / 4B / 8B, 0.9–1.0 for four of seven
   OLMo 3 tasks and ≤ 0.5 for three (necessity arriving a few blocks
   later), and is unmeasurable on Gemma 4.
5. **Two method results.** The "generic response" of a residual
   perturbation is a property of its norm, not its content
   (`docs/GENERIC_RESPONSE.md`), and residual edits presuppose that the
   residual basis is roughly the basis the model reads, which Gemma 4
   violates (D35: a massive coordinate sets the norm, gains of 30–170
   read the rest back).

What is not established: the mechanism (which sublayer writes the
increments in the steered run, where no demonstrations exist to be
read), whether the hand-over read point coincides with an
independently defined landmark, the per-prompt geometry (population
means only), a second seed on the new families, and necessity on
Gemma 4.

## What would show potential, or its absence

Cheapest first; the first three need no new model runs beyond one
capture pass each.

- **A. The landmark test.** Does the hand-over read point coincide
  with (i) the depth at which the natural difference across prompts
  becomes rank one (the pool spectrum per read point; the runs store
  it at three read points only), (ii) the depth at which a lens
  readout of the task's mean natural difference first names the
  task, (iii) the workspace onset of the Transformer Circuits
  workspace paper (below), (iv) the layers of the universal heads? A
  coincidence with (ii) or (iii) turns "a fixed read point" into "the
  entrance of the model's verbalizable workspace", which is a claim
  the field can use. Coincidence with the depth at which the *answer*
  becomes linearly decodable instead would mean the finding is about
  where answers appear, not about control, and would be the honest
  stopping point (partly excluded already: the rank-one direction is
  task-level, and the shared component survives removal of the answer
  direction, D33).
- **B. The source test.** Decompose each block's increment at the
  query token into its attention and MLP parts in the steered
  zero-shot run and in the demonstration run. In the steered run there
  are no demonstrations to read, so increments that look like the
  natural ones must be produced by MLPs reading the steered residual,
  or by heads attending to the query token itself. If the natural
  increments are attention-written and the steered ones MLP-written,
  then the control is a substitute *input* to a fixed per-block
  computation, and the same output arises from a different source;
  that is a mechanistic claim with teeth. One capture pass per model.
- **C. The verbalisation test.** Read the rank-one own direction at
  the hand-over through a lens: the logit lens (in the backend) and,
  where fitted lenses exist for the open models, the Jacobian lens.
  If it names the task ("opposite", "plural", "past"), the canonical
  form is a verbalizable task concept; if it names nothing or the
  answer, it is not.
  *Run (D36, on the post-trained Qwen3-8B with the Neuronpedia
  Jacobian lens; `STATUS.md`): negative at the hand-over.* The
  natural difference names the task at 0.50–0.61 of the stack on
  antonym and singular (rank 1) and nothing at the hand-over (0.69)
  or after; the learned vector's perturbation never names the task;
  the head mean names the output form. The concept precedes the
  hand-over, and what is handed over is a later state the lens cannot
  name.
- **D. Robustness.** A second seed on OLMo 3 and Gemma 4; sizes within
  one family other than Qwen3 to test the size dependence of
  necessity; Llama 3 as the cheapest further family; necessity on
  Gemma 4 with edits made in the block's read frame.

## The programs

**Transformer Circuits (Anthropic).** Three of its threads bear
directly on this experiment.

- *The residual stream as a communication channel* (the 2021
  framework) predicts what the increments show: blocks write to
  distinct subspaces, so successive writes are near-orthogonal and the
  injected direction is not what later blocks operate on.
- *Circuit tracing / attribution graphs* (2025) would render the
  hand-over as the activation of task features by the injected
  direction and their propagation to output features. The May 2026
  update ("Downstream connections predict which features will steer
  model behavior") states, for features, what our sufficiency-versus-
  necessity and weak-head-mean results suggest for directions: a
  direction's steering power is set by what reads it downstream, not
  by what it looks like.
- *Verbalizable representations form a global workspace* (July 2026;
  the "J-space" thread) is the closest match. It defines the Jacobian
  lens (the average linearised effect of an activation on a token's
  likelihood), the J-space (sparse non-negative combinations of at
  most ~25 lens vectors, never more than 10 % of activation variance
  yet carrying most of a concept vector's power to redirect an answer),
  and finds a workspace band in the middle layers (about 0.38–0.92 of
  the stack, normalised) whose onset is "independent of where
  information originates", where "interpretation of ambiguous inputs
  solidifies", and into which injected lens vectors load and can be
  swapped to flip answers. Studied on Claude, replicated on Gemma 3 4B,
  with lenses for open models on Neuronpedia.

  The mapping: our hand-over read point (0.5–0.8 of the stack,
  independent of the injection) looks like the workspace onset seen
  through a different instrument; our rank-one own-task direction
  looks like a task concept loaded into the workspace; their "workspace
  loading" (cosine of the clean activation with the lens vector)
  predicting steering success is our "the effect rides on the
  projection onto the natural direction". What they did not measure
  is the dynamics of that loading when the input is a steering vector
  rather than text: that it happens at the fixed depth, in rank one,
  through the blocks' own writes, and that the injected direction is
  gone by then. What we did not measure is whether our direction is in
  the J-space; test C answers it and would join the two.

**The function-vector and task-vector line** (Todd et al. 2024,
Hendel et al. 2023, Merullo et al. 2023) anticipated that a mid-stack
vector carries a task and works when added over a range of layers,
not what the model does with it afterwards.

**The residual-geometry line** (iterative refinement, layer pruning by
adjacent-layer cosine, "transformer layers as painters", and the 2026
depth-geometry analysis at arXiv 2607.18348) describes the full state,
whose per-block turning we reproduce for the perturbation; none of it
tracks an injected perturbation.

**Outlier-dimension and massive-activation work** (Kovaleva 2021,
Dettmers 2022, Sun 2024 and others, listed in D35) anticipates Gemma
4's structure but not its consequence for intervention methodology,
which D35 records.

## The one-paragraph verdict

The experiment has a real, replicated phenomenon, a clean method, and
a natural home in the workspace thread. Whether it has *potential*
beyond a careful phenomenology turns on tests A–C: if the hand-over is
the workspace entrance, the rank-one direction is a verbalizable task
concept, and the steered increments come from a different source than
the natural ones, then the project has shown, with matched controls on
open models, how an out-of-distribution control is admitted into the
model's own computation, and where. If the hand-over is the answer
becoming readable and the direction is the answer, the result is a
well-controlled restatement of what lenses already show, and the
honest conclusion is to stop.

*After test C* (D36): the second condition fails as stated. The
rank-one direction at the hand-over is not a verbalizable task
concept, and neither is the control's perturbation at any depth; but
the hand-over is not the answer becoming readable either (for the
large-answer tasks the carrier names neither the task nor the
answer). The verbalizable concept sits earlier, at 0.50–0.61 of the
stack, on the tasks that have a name. That moves the question to
tests A and B: whether the hand-over read point is where the
workspace paper's concept vectors stop being verbal in this model
(test A, now with a concrete depth to compare), and whether the
steered increments are produced by the same blocks and components as
the natural ones (test B). If the natural concept stops being verbal
at the same depth where the control becomes rank-one along the
natural difference, the hand-over is the exit from the verbal
representation rather than the entrance to it, which is a different
and still mechanistic claim; if not, the phenomenon is a property of
the answer computation and the conclusion above stands.
