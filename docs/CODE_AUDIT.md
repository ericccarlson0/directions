# Code audit (2026-10-20)

Three read-only audits of the repository, one per slice: the active variable-map line (D45–D56) and its shared
infrastructure, the closed single-control line (D1–D44), and the tests, scripts, configs and GPU workflow. They
were consolidated here as input to a later quality sweep. Items marked **(verified)** were checked by hand against
the code; the others are as the audits reported them, with their file and line references at the time of writing.

## Verdict

The numerical core is in good shape. The engines are tested against independent references: the DAS single-token
path against native site swaps on three toy architectures, gradients against finite differences, the probe's
closed-form leave-one-out against explicit refits, and batched code against loops. Seeding is disciplined: one
SHA-256 helper, named streams, no `hash()`, no global RNG. Runs fail hard on determinism and identity checks.

What does not hold up is the layer above the engines.
- Each task's definition is spread over about fifteen places.
- Five active stage classes, and six closed ones, copy their scaffolding.
- A preregistered statistic, D54's control rule, lives only in scripts.
- Configs are mostly copies.
- Scripts find runs by parsing directory names.

No recorded result was found to be wrong. Several places are where a quiet mistake would go unnoticed.

## Fixed during the audit

- `scripts/d56_vs_d55.py` claimed to apply D54's control rule "as `d54_compare.py` computes it" but did not. It
  counted a variable held at the control as held, and had no void. It now applies the rule as written; no D55 or
  D56 reading changed, since nothing is held at any control. **(verified)**
- `scripts/d55_decompose.py` reloaded the config file as it is now. It now refuses one that differs from the
  config recorded with the run, allowing only fields added since. **(verified)**

## Risks: correctness and provenance (fix first; most are small)

1. **D54's held rule exists only in scripts.**
   - It is in `scripts/d54_compare.py` (`paired_over_control`, `SEED`, `N_BOOT`), which `d56_vs_d55.py` imports
     through a `sys.path` hack.
   - The bootstrap draws from one generator shared across cells, so a cell's interval depends on how many cells
     came before it.
   - The `held`, `count` and `control_clean` fields in `das.json` are still D48-only, so a run's own output does
     not hold the readout used since D54.
   - **Change:** move the rule into the package (`DAS._read` or a `das_readout` module), seed each cell with
     `derive_seed(seed, site, var)`, write it to `das.json`, and unit-test it on a hand-computed case.
2. **The closed line's seed table records seeds that were not used.** **(verified)**
   - `pipeline._seed_table` (`pipeline.py:238`) keys by the registry name `t.name`; every seed actually used is
     derived from the task's key (`pipeline.py:308`).
   - For labelled tasks that share a registry name (add-1 … add-10, all "arithmetic"), the `metadata.json` table is
     wrong. Results are unaffected: the seeds used are deterministic.
   - The D30, D39 and D41 runs are affected. **Change:** an erratum in `docs/SINGLE_CONTROL.md`, and key the
     table by `t.key`.
3. **Broad `except Exception` in the closed pipeline.** **(verified; fixed 2026-10-21:** `directions pilot` and
   `validate` now exit 1 and list the errors when any task raised; the outputs are unchanged.**)**
   - `pipeline.py:210,221,233` (and `:976,984,993`) turn any bug, an out-of-memory error included, into an
     "error" rejection. `directions pilot` still exits 0.
   - **Change:** count error rejections in `summary.json` and exit non-zero when there are any.
4. **Run directories that can be reused silently.** **(verified)** `mixing.py:198` and `source.py:193` create the
   run directory with `exist_ok=True`, so a repeated `--run-id` writes into an old run. The other stages refuse.
5. **Reproducibility checks treat git state as a difference.** **(fixed 2026-10-21:** `compare` reports `git` as
   a note and does not count it; tests in `tests/test_cli.py`.**)**
   - `directions compare` does not ignore the `git` field, and `test_integration.py` reruns the pipeline and
     compares.
   - A commit during the test run fails the test; this happened on 2026-10-19. A `compare` across two commits,
     which AGENTS.md prescribes, always reports a difference.
   - **Change:** treat `git` and `environment` as informational in `compare`.
6. **The committed-request test pins `gpu_type == ""`.** **(verified; fixed 2026-10-21:** it now checks only that
   the value is parsed without quotes.**)** `tests/test_runpod_request.py:33` fails
   as soon as a request pins a card, which AGENTS.md recommends when comparability matters. The quoting regression
   it guards is covered elsewhere in the file (`:90–111`).
7. **The runner's status poll survives only `URLError`.** **(verified; fixed 2026-10-21:** the poll retries every
   transient failure and gives up after 20 in a row; the workflow cancels the remote job on failure too.**)**
   - At `scripts/runpod_job.py:93`, a dropped connection mid-response, a read timeout or a non-JSON body crashes
     the step.
   - The workflow cancels the remote job only `if: cancelled()`, so after such a crash the job runs on unwatched.
   - **Change:** catch `(OSError, ValueError)` with a cap on consecutive failures, and cancel on
     `failure() || cancelled()`.
8. **Missing values pass silently.**
   - `isolate_lo = np.nanmin(...)` (`das.py`) ignores an Isolate kind with no evaluable pairs. On max, whose
     sources are optional, a variable could pass as held with one kind never tested.
   - `write_json` writes NaN as null, and the comparison scripts then call `round()` on null.
   - **Change:** require a minimum n per Isolate kind, or mark the cell undecided.
9. **Runs are found by name.**
   - Nine scripts glob a prefix and parse the model size with a regex. When two runs share a size, the last one
     silently wins.
   - `d51_sites.py`, which picks DAS sites, takes the first of an unsorted `glob`.
   - Summaries committed under `docs/` do not record which runs, commit or seed they read.
   - **Change:** a `find_runs()` that reads `metadata.json` and raises on duplicates, and a `write_summary()` that
     stamps the inputs.
10. **Facts that runs do not save.**
    - Digit token ids: `d55_decompose.py` hard-codes `TID0 = 15`, guarded by a check.
    - Gathers: `varprobe_summary.py` has `GATHER`, `GATHER3` and `GATHER_MAX` tables, and the varsplit configs and
      varmap summaries hold the same numbers.
    - **Change:** write them into each run's metadata, and read them from there.
11. **Read-point validation.** varmap silently drops out-of-range read points, while posset raises.
    `d51_sites.control_for` indexes arrays by read-point value, which is right only while every read point is
    mapped.

## Structure (the sweep proper)

A. **One definition of a task.**
   - Code tells tasks apart three ways: the task string, `Expr.op`, and `Expr.e is None` or a digit-array width.
   - The per-task tables are spread over varmap (`SOURCE_TYPES`, `BRACKETS`, `TEMPLATES`, `VALUE_TEST`), das
     (`TASK_*` tables extended at import time, `N_CLASSES`), varprobe (`TARGETS`, `LENS_CANDIDATES`), varsplit
     (`SAME_OF`, a copy of `VALUE_TEST` **(verified)**), posset and `config.py`'s allowed-value lists.
   - There are two source-definition systems with different vocabularies (`S1`/`S1_same` and
     `only_S1`/`same_S1`).
   - A new task touches about fifteen places with nothing to catch a missed one.
   - **Change:** an `arith.py` with a frozen `TaskSpec` registry: template, expressions, predictions, source masks
     in one vocabulary, DAS variables and kinds, probe targets and lens candidates. Config validation reads its
     allowed values from it.
B. **Shared stage scaffolding.**
   - Repeated across the five active stages (eleven counting the closed line): the run directory, logging,
     rejection log, metadata header and epilogue, digit-id lookup, model slug, "a figure never fails a run",
     repeat-and-compare determinism blocks, and CLI `_cmd_*` plus parser blocks.
   - The copies have drifted: metadata written at start versus only at end, `version` versus
     `directions_version`, and six different determinism policies, one of which never raises (`mixing.py`).
   - **Change:** a `RunContext` that refuses existing directories, writes metadata at start, owns the rejection
     log and has one determinism helper with an explicit policy; a `StageCommon` config mixin; a CLI registry; a
     `load_any_config` keyed on the config's `name`. Adopt them in the active line first.
C. **Private cross-module imports.**
   - das imports `_jsonable` from varprobe **(verified)**, posset imports `_resample_counts` from varmap
     **(verified)**, and das lazily imports `set_members` from posset.
   - Bootstraps live in four places (varmap, posset, `d54_compare`, `stats.py`) with three spellings of the
     interval keys (`lo/hi`, `low/high`, `ci_low/ci_high`).
   - **Change:** expression rendering and token alignment in `arith.py`, every bootstrap in `stats.py`.
D. **Long methods with pure logic inside.**
   - `VariableMap._task` (about 170 lines) and `PositionSets._task` mix orchestration with decisions.
   - In `DAS._read`, the k\* choice, the held test and the site summary are pure functions sitting beside code
     that needs the model; they are tested only through toy end-to-end runs.
   - **Change:** extract them and unit-test them.
E. **`LastToken` reaches into internals.** It uses the backend's private methods and the Hugging Face cache's
   internals, and re-implements the hook convention. It is fast and tested against site swaps, so it should stay,
   as a public `ModelBackend.tail_session()` in `model.py` that checks the cache API it expects.
F. **The active line imports closed modules for two small names.**
   - `determinism.py` imports `layerwise`, and through it `geometry` and `stats`, only for
     `profile_check_arrays`; the active stages need only `compare_arrays`.
   - `model → prompts → tasks` pulls in the task registry and about 430 lines of word lists, for `Prompt` and
     `Item`.
   - **Change:** move `compare_arrays` and the two types to dependency-free modules; the closed line can then be
     frozen cleanly.
G. **Configs.**
   - There are 150 files and about 11,100 lines. In families with several files, about three quarters of the
     lines are copies; for example, `pilot_*` differ in the model name, and `das_d56_*` in the model, batch and
     sites.
   - `tests/test_configs.py` (27 tests) reconstructs the inheritance by diffing loaded dicts, and picks loaders
     from filename substrings.
   - **Change:** an `extends:` key that deep-merges a base, with each run's `config.resolved.yaml` already saved;
     a loader registry keyed on `name:`; and the diff guards as a declarative table. Keep the guards' intent: they
     are preregistration checks.
H. **Tests.**
   - There is no `conftest.py`. The toy backend is built 17 times in 12 files, the same `toy` fixture appears in
     three files, and 23 config loads use paths relative to the working directory.
   - Smoke pipelines are rerun about five times beyond the intentional reproducibility reruns.
   - There is no `slow` marker, and no CI workflow runs the tests.
   - **Change:** a `conftest.py` with shared fixtures and session-scoped smoke runs, `pythonpath = ["scripts"]`,
     a `slow` marker, and a cheap CPU test workflow.
I. **Scripts.**
   - Duplicated helpers: the model-size regex at nine sites, `by_size` twice, RunPod GraphQL calls four times,
     GPU tier lists three times (only one has `BLACKWELL_180`).
   - 15 of 36 scripts have no test.
   - **Change:** a stdlib-only RunPod module as the single source of tiers and data centers; one test that runs
     every script's `--help`; the closed line's scripts in `scripts/single_control/`; the site rule and run
     discovery in the package.

## The closed line (D1–D44)

- **Freeze**, do not refactor, the closed numerical modules: any change to their numerical path loses
  bit-identical reproduction of D1–D44. They are `pipeline`, `calibration`, `extraction`, `function_vector`,
  `learned`, `layerwise`, `geometry`, `analysis`, `ablation`, `commitment`, `neutral`, `trajectories`, `source`,
  `mixing`, `serial`, `staging`, `figures`, `aggregate`, `data`, `data_extra`, and the task registry.
- **Archive**, after tagging the last commit (e.g. `single-control-final`), the modules only scripts use:
  `landmarks`, `verbalise`, `family`. Later, move the frozen set into `directions/single_control/` with its tests
  (about 20 test files import closed modules).
- **Before archiving, extract** the tools `docs/SINGLE_CONTROL.md` says carry over, if they are wanted: rank-k
  keep (`trajectories._subspace_test`, D34) and context masking (`staging._masking`, D42).
- Duplicated numerics there (four Spearman variants that disagree with ties, six unit normalisations, three
  random-unit-vector generators, two identical torch medians) should be consolidated only where the active line
  uses them.
- About 900 of `config.py`'s 1,282 lines are closed-line config classes.

## Documents

- `STATUS.md` is about 5,900 lines (around 100k tokens), and 94% of it is the run log, yet AGENTS.md makes it
  required reading. **Change:** keep a short index there and move each decision's run log to `docs/runs/D<n>.md`.
- Test counts in STATUS are stale (406 tests collected on 2026-10-20). `docs/INFRA.md`'s flow list is numbered
  0–5, 7, 6, 8. `.gitignore` whitelists a `results/.gitkeep` that does not exist, and there is no ruff
  configuration although the code carries `# noqa` comments.

## Keep doing

- Engines tested against independent references, and hard-failing determinism and identity checks inside runs.
- Named seed streams through `derive_seed` and `rng_for`.
- `RejectionLog` for every filter.
- Loud failures on misalignment (`query_tokens`, padded batches, unknown config keys).
- `probes.py` as the model for pure, documented, vectorised, tested code.
- Stable output keys across decisions, so `compare` stays clean.
- The stdlib-only, heavily tested infrastructure helpers; the archive manifest with SHA-256 per run.
- Decision numbers in code comments and config headers; keep them through any refactor.

## Suggested order for the sweep

1. The small risk fixes: 1, 3, 5, 6, 7, 8, and the erratum for 2. Each is under an hour and needs no GPU run.
2. `conftest.py`, the `slow` marker and a CPU test workflow, so the refactor that follows is guarded.
3. Freeze and tag the closed line; split `compare_arrays` and `Prompt`/`Item` out (F).
4. `RunContext` and the CLI and config registries (B), then `TaskSpec` (A), then the pure-function extraction (D)
   and the stats consolidation (C).
5. Config overlays (G), run discovery and summary provenance (9), and the STATUS split.

After each step, rerun the toy reference runs and require bit-identical outputs, as for D56's changes: a refactor
of the active line must not change a number.
