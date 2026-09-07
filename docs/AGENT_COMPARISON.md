# Agent comparison: `opus` vs `fable`

Two independent agent sessions (Claude Opus, branch `opus`; Claude Fable 5.1,
branch `fable`) implemented and ran the measurement-validation pilot from the
same prompt. This note compares the committed artifacts only (code, configs,
docs, `STATUS.md`); run directories are git-ignored on both branches, so the
results comparison uses the numbers each session recorded in its `STATUS.md`.

## Framing caveat: the two sessions did not receive the same spec

`opus` forks from `0e05e32` (the original `docs/EXPERIMENT.md`). `fable` forks
from `0bc7806`, whose commit message says the docs were rewritten "following
initial test". Everything Opus recorded as a discovered defect or
under-specified choice (its decisions D2, D10–D16: zero-shot evaluation
regime, log-p-per-token decision metric, ρ grid beyond 0.3, paired bootstrap
plus random-control screen for selection, digest-derived seeds, the bf16
noise floor, the uncentered `N_l`, three disjoint pools, control alignment,
strength robustness, a rejection log) is already text in the spec Fable
started from. Several "implementation differences" therefore reflect Opus
discovering versus Fable being told. Neither branch edited the docs.

## Implementation

| | `opus` | `fable` |
|---|---|---|
| source / tests | 3756 LOC, 17 modules; 129 test functions (152 cases) | 3423 LOC, 18 modules; 39 tests |
| config mechanism | `extends:` chain (`base.yaml` + small model files), `--set key=value` overrides (typer) | flat, complete files; a test enforces that the 0.6B and 1.7B configs differ only in `model.name` |
| extra CLI | `tasks`, `show-config`, `inspect` | `check`, `compare` (JSON diff of two runs for reproducibility) |
| few-shot / calibration pool | 10-shot, 48 examples | 8-shot, 64 examples |
| min improvement / stability gate | 0.10 nats per token, 0.5 | 0.05 nats per token, 0.8 |
| ρ grid | fixed, top 1.0 | extends geometrically to 2.0 while the best point is at the edge (triggered on all four 0.6B selections) |
| arithmetic task | two-operand `a + b`, seeded random pairs | `n → n+k`, `k` configurable (default 3) |
| word lists | JSON files, 229–236 items | Python lists, 225–265 items |
| control alignment `A_l` | exploratory, median only | primary, per-example with CI and z vs random |
| qualitative labels | single argmax label over five classes | multi-label rules, including amplification, dimensional expansion, new-direction creation |
| block-ablation selection | top 3 by `C_l + N_l^unc` | top 2 by `C_l`, or by z vs random (config) |
| determinism flags | `use_deterministic_algorithms(False)` | cuDNN deterministic, TF32 off |
| seeds | BLAKE2b spawn keys; split seed shared across tasks | SHA-256 per (task, purpose); seed table written to `metadata.json` |

Both branches verify teacher-forced scoring against a native forward pass,
gather residuals at the query token inside the hook, run the model in bf16 with
float64 statistics, prove bit-identical reruns, and write resolved config, git
commit, model revision, environment, seeds, rejections and figures. Opus's test
suite is much deeper (batched vs unbatched reference scoring, `PYTHONHASHSEED`
invariance, hook removal, linearity in α). Opus computes per-layer z-scores
against the random null but does not summarise them; Fable folds them into the
cross-task table.

## Results

### Qwen3-0.6B-Base: the qualified sets disagree

| task | `opus` | `fable` |
|---|---|---|
| antonym | qualified, layer 14, ρ 0.05 | rejected at the random-control gate (selected layer 8, ρ 0.5) |
| plural | rejected at the calibration screen | qualified, layer 8, ρ 1.0 |
| past_tense | qualified, layer 8, ρ 0.8 | rejected at the random-control gate (selected layer 8, ρ 0.2) |
| en_fr | rejected: no improvement | rejected: no reliable calibration point |
| arithmetic | qualified (`a + b`), layer 17, ρ 0.4 | qualified (`n+3`), layer 8, ρ 0.8 |

The causes are visible in the configs. Fable's lower minimum improvement made
the earliest-layer / smallest-strength rule stop at layer 8 for antonym and at
ρ 0.2 for past_tense; both then failed the 16-control held-out gate. Opus's
plural failed its 8-control screen where Fable's passed. Different
demonstration samples (seeds) give different directions. Opus's `STATUS.md`
attributes the 0.6B failures to the model; Fable's flags the sensitivity to
`calibration.min_improvement` and to seed.

### Qwen3-1.7B-Base: close agreement

Both qualify exactly antonym and plural; plural at layer 8, ρ 0.05 in both.

| 1.7B | `opus` | `fable` |
|---|---|---|
| plural: held-out Δ log p/token, z vs random | +0.082, 3.12 | +0.098, 2.58 |
| plural: `d_eff` first → final, cumulative log G, final `A_l` | 8.0 → 12.4, 3.43, 0.018 | 6.0 → 13.8, 3.47, 0.021 |
| antonym: selection | layer 14, ρ 0.3 | layer 11, ρ 1.56 |
| antonym: `d_eff` first → final, cumulative log G, final `A_l` | 3.0 → 14.1, 2.24, 0.108 | 2.9 → 14.6, 1.81, 0.125 |

### Shared conclusions

Every qualified profile on both branches is cascade plus amplification:
conversion entropy ratio 0.93–0.99, no block above 18 % of the conversion
mass, alignment to the injected direction lost within a few blocks, cumulative
log G 1.6–3.5, larger dimensional expansion on 1.7B. Both find behavioural
localisation despite geometric spread (Opus: block 19 on 0.6B carries 43–62 %
of the gain; Fable: blocks 18–19 on 1.7B antonym, 45 % / 25 %), and both find
a block whose removal *improves* the metric.

### Interpretation differs

Opus reports amplification and expansion as findings. Fable reports that
cumulative log G, `d_eff` and `N_l` sit inside the matched-random null for
three of the four qualified pairs (plural-1.7B is the exception, z ≈ 2.2 for
gain and 2–3.6 for alignment), and that at large ρ random directions also
raise the target log-probability (all 16 controls positive for past_tense on
1.7B). Opus computed the same per-layer nulls, so this is a difference in what
was summarised, not in what was measured.

### Cost and reproducibility

Wall time: Opus 473 s / 1329 s; Fable 456 s / 868 s (larger batches). Both ran
both models; Fable ran each model twice and committed the `compare` check
(bit-identical apart from git metadata).
