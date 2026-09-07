# Status

Last updated: 2026-09-07. Authoritative history: `git log`.

## Summary

| Component | Implemented | Run |
|---|---|---|
| Package, config system, CLI (`validate`, `pilot`) | yes | yes |
| Offline integration path (`configs/smoke.yaml`) | yes | yes, passes |
| Test suite | yes | yes — **152 passed** |
| Qwen3-0.6B-Base pilot | yes | **yes, complete** — 3/5 tasks qualified |
| Qwen3-1.7B-Base pilot | yes (config-only difference) | **yes, complete** — 2/5 tasks qualified |

Every component in the specification is implemented; nothing is stubbed. Both
model pilots have been run end to end on an RTX 4090 (bf16, no quantization).

---

## What has actually been run

Both runs are from commit `ec189fa`, seed 0, `configs/qwen3_*.yaml` unmodified.

| | Qwen3-0.6B-Base | Qwen3-1.7B-Base |
|---|---|---|
| run directory | `results/20260907T025503Z__pilot__qwen3_0.6b_base__7d7e25f3aa` | `results/20260907T030322Z__pilot__qwen3_1.7b_base__3bd6fc4751` |
| config fingerprint | `7d7e25f3aa` | `3bd6fc4751` |
| layers / d_model | 28 / 1024 | 28 / 2048 |
| wall clock | 473 s | 1329 s |
| qualified task/control pairs | antonym, past_tense, arithmetic_add | antonym, plural |
| figures generated | 24 | 19 |

Run directories are not tracked in git (`.gitignore` excludes `results/`);
re-running the commands below reproduces them exactly.

### Task qualification, Qwen3-0.6B-Base

| task | 10-shot acc | stability (min pairwise abs cos) | outcome |
|---|---|---|---|
| antonym | 0.672 | 0.97-1.00 | **qualified**, layer 14, rho 0.05 |
| plural | 0.969 | 0.99-1.00 | rejected: 7 grid points improved, none beat the control screen |
| past_tense | 0.969 | 0.98-1.00 | **qualified**, layer 8, rho 0.8 |
| en_fr | 0.828 | 0.36-1.00 | rejected: no strength improved the metric at all (best −0.008) |
| arithmetic_add | 1.000 | 0.46-0.99 | **qualified**, layer 17, rho 0.4 |

### Task qualification, Qwen3-1.7B-Base

| task | 10-shot acc | outcome |
|---|---|---|
| antonym | 0.625 | **qualified**, layer 14, rho 0.3, held-out +0.706 nats/token, z=2.64 |
| plural | 0.984 | **qualified**, layer 8, rho 0.05, held-out +0.082 nats/token, z=3.12 |
| past_tense | 0.984 | rejected at the control comparison (p=0.118, z=1.66) |
| en_fr | 0.969 | rejected at the control comparison (p=0.412, z=0.34) |
| arithmetic_add | 1.000 | rejected at calibration; PC1 is unstable mid-depth (0.01 at layer 11) |

Every qualified pair beat **all 16** matched random controls on the held-out
split (empirical p = 1/17 = 0.059).

### Cross-task layerwise result (from `core/cross_task_summary.json`)

Qwen3-0.6B-Base:

| quantity | antonym | past_tense | arithmetic_add |
|---|---|---|---|
| intervention layer | 14 | 8 | 17 |
| final `\|cos(delta_l, v)\|` | 0.129 | 0.135 | 0.147 |
| blocks until alignment < 0.5 | 4 | 4 | 3 |
| cumulative `log G` | +1.94 | +1.78 | +1.82 |
| mean conversion `C_l` | 0.60 | 0.52 | 0.59 |
| conversion centroid depth | 0.46 | 0.47 | 0.48 |
| conversion entropy ratio | 0.99 | 0.99 | 0.97 |
| max single-block share | 0.09 | 0.07 | 0.17 |
| `d_eff` first → final | 18.5 → 18.7 | 10.2 → 13.6 | 7.4 → 10.4 |
| `d90` final | 40 | 36 | 20 |
| mean `N_l` (uncentered) | 0.91 | 0.96 | 0.99 |
| numerical noise / signal | 0.007 | 0.0004 | 0.0003 |

Qwen3-1.7B-Base: antonym `d_eff` 3.0 → 14.1 (ratio 4.7), cumulative `log G`
+2.24, final alignment 0.108; plural `d_eff` 8.0 → 12.4, cumulative `log G`
+3.43, final alignment 0.018.

**What this says, for all five qualified pairs and both models:**

* **Not conserved transmission.** The injected direction's alignment with the
  perturbation decays to 0.02-0.15 within 2-4 blocks, and mean per-block
  conversion is 0.52-0.68.
* **Cascaded, not delayed and not localized.** The conversion entropy ratio is
  0.97-0.99 (near-uniform across blocks), no single block carries more than 17%
  of the conversion mass, and the centroid sits at depth 0.46-0.48.
* **Amplification.** Cumulative `log G` is +1.78 to +3.43, i.e. the absolute
  perturbation norm grows 6-31x, while `S_l` (relative to the residual norm)
  falls — the residual stream grows faster than the perturbation.
* **Dimensional expansion, modest in 0.6B and large in 1.7B.** `d_eff` ratios
  are 1.01-1.41 on 0.6B and 1.56-4.69 on 1.7B.
* **Most of each block's response is in new directions.** `N_l` (uncentered)
  averages 0.91-0.99.

The automatic qualitative label is `cascade` for all five pairs.

### Exploratory findings

* **Block ablation.** Although the *geometric* conversion is spread evenly, the
  *behaviourally relevant* conversion is concentrated: on Qwen3-0.6B, removing
  block 19's steering contribution costs 62% (antonym) and 43% (past_tense) of
  the behavioural gain, and injecting that block's contribution alone into an
  unsteered run restores 55% and 46%. For `arithmetic_add`, removing blocks 17
  and 18 *improves* the metric (−0.54, −0.59 of the gain), a negative result
  worth following up.
* **Strength robustness.** Re-measuring at the strongest reliable strength at
  the same layer gives Spearman agreement with the preregistered (smallest
  reliable) operating point of 0.99/0.99/0.92 (past_tense), 0.85/0.80/0.67
  (antonym) and 0.77/0.72/0.75 (arithmetic_add) for `log G`, `d_eff`, `N`.

---

## Failures and problems encountered

Each of these was a real defect or an under-specified choice, not a tuning
convenience. All are documented with their evidence in `docs/DECISIONS.md`.

1. **Seeds were not reproducible (D13).** Sub-generators were keyed on
   `abs(hash(task_name))`; CPython salts string hashing per process, so
   identical commands drew different demonstrations and controls. Fixed with a
   BLAKE2b-derived key; two consecutive `validate` runs are now bit-identical
   on every reported number.
2. **Zero-shot accuracy is degenerate on this model (D10).** It is 0.000 at
   every candidate layer and every strength for all four word-mapping tasks, so
   it cannot calibrate anything. The decision metric is now teacher-forced
   log-probability per target token, which the specification already prefers
   for multi-token outputs.
3. **The literal "earliest layer, smallest strength" rule selected noise
   (D12).** It picked layer 6 / rho 0.3 for `arithmetic_add` (+0.10 accuracy,
   within sampling noise) while layer 11 gave +0.31, then failed the held-out
   control test and discarded a task with a large real effect. Selection now
   requires a paired bootstrap test *and* a matched random-control screen.
4. **The rho <= 0.3 range was too narrow (D11).** Nothing measurable happens
   below rho ~ 0.3 on Qwen3-0.6B; the effect peaks at 0.5-0.8 and collapses by
   1.2. Grid extended to 1.0.
5. **bf16 noise masqueraded as dimensionality (D15).** At the intervention
   layer every `delta_l` is identical by construction, so `d_eff` is undefined;
   in bf16 the observed value was 24-42 (isotropic rounding of `h + alpha*v`)
   and it corrupted the automatic profile label. Now reported as `null`, with
   the observed value and per-layer centered variance kept as an explicit
   noise-floor diagnostic (measured noise/signal: 0.0003-0.007).
6. **Two config fields were declared but never read.**
   `layerwise.measure_token` was removed; `intervention.token` now raises for
   any value other than the implemented one.
7. **Two tasks do not survive on either model.** `en_fr` shows no steering
   improvement at all on 0.6B and no control separation on 1.7B; `plural` and
   `past_tense` each fail the control comparison on one of the two models.
   These are recorded as rejections with their numbers, not worked around.

---

## Verification

* `uv run pytest` — **152 passed** (~85 s, no GPU, no network). Covers
  effective rank, PCA extraction, direction normalization, orthogonal control
  construction, blockwise decomposition, layerwise gain/conversion, the
  bootstrap, hook exactness, and a full end-to-end pilot on a randomly
  initialised tiny Qwen3.
* Batched, padded, length-sorted scoring is numerically identical to naive
  unbatched per-prompt scoring.
* The intervention is exact: `delta_l = 0` upstream of the intervention layer
  and `delta_l = alpha*v` at it, to float tolerance.
* Two runs of the same config produce identical reported numbers (verified on
  Qwen3-0.6B, and asserted in `test_pilot_is_reproducible`).

---

## Next commands

```bash
uv run pytest
uv run directions pilot --config configs/qwen3_0.6b.yaml
uv run directions pilot --config configs/qwen3_1.7b.yaml
```

### Suggested next work, in priority order

1. **Raise the number of qualified pairs.** Only 3/5 and 2/5 tasks survive. Two
   levers are available without changing the methodology: more evaluation
   examples (the control comparison currently bottoms out at p = 1/17, so
   `controls.n_random: 32` would allow a sharper test), and additional task
   families. Run:
   `uv run directions pilot -c configs/qwen3_0.6b.yaml -s controls.n_random=32`
2. **Widen the candidate layer range.** The specification's 20-60% window
   (layers 6-17) excludes layer ~20, where the diagnostic sweep showed the
   largest log-probability gains for `past_tense` and `en_fr`. Run:
   `uv run directions pilot -c configs/qwen3_0.6b.yaml -s extraction.candidate_layer_fracs='[0.2,0.3,0.4,0.5,0.6,0.7,0.8]'`
3. **Follow up the `arithmetic_add` ablation sign.** Removing blocks 17-18's
   steering contribution *improves* the behavioural metric; understanding this
   is the most interesting loose end in the current results.
4. **Check the bf16 result against float32** on the 0.6B model (fits easily in
   24 GB): `uv run directions pilot -c configs/qwen3_0.6b.yaml -s run.dtype=float32`
5. Then the follow-ups in `docs/EXPERIMENT.md`: replace the PCA control with
   canonical function-vector extraction, and replicate on a second model family.
