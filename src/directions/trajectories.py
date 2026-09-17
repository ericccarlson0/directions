"""All-to-all comparison of downstream trajectories (docs/DECISIONS.md D32).

Two finished runs of one model (the head-mean and the learned-vector run: same seed, splits and held-out
prompts) supply three constructions of a task direction, the pooled PC1, the head-mean function vector and
the learned vector. At common injection layers each is injected into the held-out zero-shot prompts, and
the residual difference it makes at every later read point,

    delta_m(v)(x) = h_m(x; h_l + v) - h_m(x),

is compared with the natural difference the demonstrations make,

    delta_m^ICL(x) = h_m(x; demos) - h_m(x)      (and its permuted-label contrast, h(demos) - h(deranged)),

and with the other constructions' differences, pair by pair, per example and per read point. Matched
isotropic directions at each construction's norm give the floor of every alignment; a second demonstration
sample gives its ceiling. A variant with the answer's unembedding direction projected out tells "raises the
same answer" from "runs the same computation".
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import Config, TrajectoriesConfig, config_from_dict, config_to_dict
from .model import ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .profiling import Profiler
from .prompts import Prompt, deranged_prompt, few_shot_prompt, zero_shot_prompt
from .runinfo import git_info, read_json, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .stats import bootstrap_median_ci_rows, paired_bootstrap_test
from .tasks import Item

CONSTRUCTIONS = ("pca", "fv", "learned")  # the three implementations of a task direction
NATURAL = ("icl", "icl2", "task")  # the natural trajectories: two demonstration samples and the label contrast
REFERENCES = ("icl", "task")  # what a construction's alignment is summarised against
NAMED = NATURAL + CONSTRUCTIONS
ALIGN_MIN_EXCESS = 0.05  # exploratory labels only: a decline smaller than this is not a divergence


# --------------------------------------------------------------------------- #
# Geometry (pure NumPy; tested against naive loops)
# --------------------------------------------------------------------------- #


def remove_direction(X: np.ndarray, U: np.ndarray | None) -> np.ndarray:
    """``X`` (n, d) with its component along the unit rows of ``U`` (n, d) removed; unchanged when ``U`` is None.
    Rows of ``U`` that vanish remove nothing."""
    if U is None:
        return X
    return X - (np.einsum("nd,nd->n", X, U))[:, None] * U


def unit_rows(X: np.ndarray) -> np.ndarray:
    """Rows of ``X`` (..., d) normalised to unit length in float64; vanishing rows stay zero."""
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    return np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)


def _rows_at(U: np.ndarray | None, m: int) -> np.ndarray | None:
    """The unit rows to remove at read point ``m``: ``U`` is (n, d), the same at every read point, or
    (L+1, n, d), one set per read point."""
    if U is None:
        return None
    return U[m] if U.ndim == 3 else U


def pair_cosines(A: np.ndarray, B: np.ndarray, start: int = 0, U: np.ndarray | None = None) -> np.ndarray:
    """Signed per-example cosine between ``A`` and ``B``, both (L+1, n, d), at every read point: (L+1, n).

    Read points below ``start`` are nan, as is any example whose vector vanishes. With ``U``, unit rows
    shaped (n, d) or (L+1, n, d), the cosine is taken after removing that direction from both vectors.
    """
    L1, n, _ = A.shape
    out = np.full((L1, n), np.nan)
    for m in range(start, L1):
        u = _rows_at(U, m)
        a = remove_direction(np.asarray(A[m], dtype=np.float64), u)
        b = remove_direction(np.asarray(B[m], dtype=np.float64), u)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 0) & (nb > 0)
        out[m, ok] = np.einsum("nd,nd->n", a[ok], b[ok]) / (na[ok] * nb[ok])
    return out


def projection_fraction(A: np.ndarray, B: np.ndarray, start: int = 0) -> np.ndarray:
    """``(a . b) / |b|^2`` per example and read point (L+1, n): how much of the reference ``B`` the trajectory
    ``A`` reproduces along ``B``'s own direction (1 = the same vector, 0 = nothing of it)."""
    L1, n, _ = A.shape
    out = np.full((L1, n), np.nan)
    for m in range(start, L1):
        a, b = np.asarray(A[m], dtype=np.float64), np.asarray(B[m], dtype=np.float64)
        nb2 = np.einsum("nd,nd->n", b, b)
        ok = nb2 > 0
        out[m, ok] = np.einsum("nd,nd->n", a[ok], b[ok]) / nb2[ok]
    return out


def mean_cosines(mean_a: np.ndarray, mean_b: np.ndarray, start: int = 0) -> np.ndarray:
    """Cosine between two population-mean trajectories (L+1, d) at every read point (L+1,)."""
    out = np.full(mean_a.shape[0], np.nan)
    for m in range(start, mean_a.shape[0]):
        na, nb = np.linalg.norm(mean_a[m]), np.linalg.norm(mean_b[m])
        if na > 0 and nb > 0:
            out[m] = float(mean_a[m] @ mean_b[m] / (na * nb))
    return out


VARIANT_PREFIX = {"raw": "", "answer_removed": "noanswer_", "generic_removed": "nogeneric_"}


def _variant_prefix(tag: str) -> str:
    """The array-name prefix of a comparison variant (``cos_<prefix><a>~<b>``, ``floor_<prefix><c>_vs_<o>``)."""
    return VARIANT_PREFIX[tag]


def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Strengths from the finished runs
# --------------------------------------------------------------------------- #


def alpha_from_calibration(cal: dict[str, Any] | None, layer: int) -> dict[str, Any] | None:
    """The strength a finished run's calibration gives a direction at ``layer``: its selected grid point when
    the layer is the selected one, else the reliable point nearest the run's reference rho (the run's own
    rule at that layer), else the natural norm (rho = 1) when the layer was calibrated, else None."""
    if not cal:
        return None
    sel = cal.get("selected")
    if sel and int(sel["layer"]) == layer:
        return {"alpha": float(sel["alpha"]), "rho": float(sel["rho"]), "source": "selected"}
    reliable = [g for g in cal.get("grid", []) if int(g["layer"]) == layer and g.get("reliable")]
    ref = cal.get("reference_rho")
    if reliable:
        if ref is None:
            g = min(reliable, key=lambda g: g["rho"])
        else:
            g = min(reliable, key=lambda g: (abs(np.log(g["rho"]) - np.log(ref)), g["rho"]))
        return {"alpha": float(g["alpha"]), "rho": float(g["rho"]), "source": "reliable_grid_point"}
    units = cal.get("strength_units") or {}
    if str(layer) in units:
        return {"alpha": float(units[str(layer)]), "rho": 1.0, "source": "natural_norm"}
    return None


# --------------------------------------------------------------------------- #
# Curve summaries
# --------------------------------------------------------------------------- #


def summarise_alignment(
    real: np.ndarray, floor: np.ndarray, start: int, rng: np.random.Generator, n_boot: int, alpha: float
) -> dict[str, Any]:
    """Where and how much a per-example alignment curve ``real`` (L+1, n) exceeds its floor ``floor``
    (L+1, k) (the matched random directions' alignments, pooled), from read point ``start`` on.

    The core numbers: the median at injection, its peak (and its depth as a fraction of the downstream
    depth), its final value, the floor's median at the same points, and a paired test of the decline from
    the peak to the end. The label is exploratory (the peak is chosen on the same data): ``never_aligns``
    when the median never clears the floor (CI low of the median above the floor's CI high), ``converges``
    when it clears the floor at the end without a significant decline from its peak,
    ``aligns_then_diverges`` when it clears the floor somewhere but declines significantly to the end and
    ends at the floor, ``aligns_then_partly_diverges`` when it declines significantly but ends above the
    floor, and ``partial`` otherwise.
    """
    L = real.shape[0] - 1
    med = bootstrap_median_ci_rows(real, rng, n_boot=n_boot, alpha=alpha)
    fl = bootstrap_median_ci_rows(floor, rng, n_boot=n_boot, alpha=alpha)
    rm = np.asarray(med["median"], dtype=np.float64)
    above = [bool(l > h) if not (np.isnan(l) or np.isnan(h)) else False
             for l, h in zip(med["low"], fl["high"])]
    pts = [m for m in range(start, L + 1) if not np.isnan(rm[m])]
    out: dict[str, Any] = {"start": start, "n_read_points": L + 1, "curve": med, "floor": fl, "above_floor": above}
    if not pts:
        out.update({"label": "undefined"})
        return out
    peak = max(pts, key=lambda m: (rm[m], -m))
    final = pts[-1]
    decline = paired_bootstrap_test(real[peak], real[final], rng, n_boot=n_boot, alpha=alpha)
    out.update({
        "at_injection": float(rm[pts[0]]),
        "floor_at_injection": float(fl["median"][pts[0]]),
        "peak": float(rm[peak]),
        "peak_read_point": int(peak),
        "peak_depth_fraction": float((peak - start) / (L - start)) if L > start else 0.0,
        "floor_at_peak": float(fl["median"][peak]),
        "final": float(rm[final]),
        "final_read_point": int(final),
        "floor_at_final": float(fl["median"][final]),
        "above_floor_anywhere_after_injection": any(above[m] for m in pts if m > start),
        "above_floor_at_final": bool(above[final]),
        "decline_from_peak": decline.__dict__,
    })
    significant_decline = decline.p_value < 0.05 and decline.median_diff > ALIGN_MIN_EXCESS
    if not out["above_floor_anywhere_after_injection"]:
        label = "never_aligns"
    elif above[final] and not significant_decline:
        label = "converges"
    elif significant_decline and not above[final]:
        label = "aligns_then_diverges"
    elif significant_decline:
        label = "aligns_then_partly_diverges"
    else:
        label = "partial"
    out["label"] = label
    return out


# --------------------------------------------------------------------------- #
# The finished runs
# --------------------------------------------------------------------------- #


@dataclass
class TaskInputs:
    name: str
    learned_dir: Path
    fv_dir: Path | None
    prompt_cfg: Any
    evaluation: list[Item]
    calibration: list[Item]
    learned_qual: dict[str, Any]
    learned_cal: dict[str, Any] | None
    fv_qual: dict[str, Any] | None
    fv_cal: dict[str, Any] | None
    learned_arrays: dict[str, np.ndarray]
    fv_arrays: dict[str, np.ndarray] | None
    notes: list[str] = field(default_factory=list)


def _load_run(root: Path) -> tuple[Config, dict[str, Any]]:
    meta = read_json(root / "metadata.json")
    return config_from_dict(meta["config"]), meta


def _items(pairs: list[list[str]]) -> list[Item]:
    return [Item(i, o) for i, o in pairs]


def _optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _load_task(name: str, learned_root: Path, fv_root: Path, cfg: Config) -> TaskInputs | None:
    ld = learned_root / "core" / "tasks" / name
    qual = _optional_json(ld / "qualification.json")
    if not qual or not qual.get("selection") or not (ld / "directions.npz").exists():
        return None
    tcfg = next((t for t in cfg.tasks if t.key == name), None)
    if tcfg is None:
        return None
    prompt_cfg = replace(cfg.prompt, target_scoring=tcfg.target_scoring or cfg.prompt.target_scoring)
    splits = read_json(ld / "splits.json")
    inputs = TaskInputs(
        name=name, learned_dir=ld, fv_dir=None, prompt_cfg=prompt_cfg,
        evaluation=_items(splits["evaluation"]), calibration=_items(splits["calibration"]),
        learned_qual=qual, learned_cal=_optional_json(ld / "calibration.json"), fv_qual=None, fv_cal=None,
        learned_arrays=dict(np.load(ld / "directions.npz")), fv_arrays=None,
    )
    fd = fv_root / "core" / "tasks" / name
    if (fd / "directions.npz").exists() and (fd / "splits.json").exists():
        fsplits = read_json(fd / "splits.json")
        arrays = dict(np.load(fd / "directions.npz"))
        if fsplits["evaluation"] != splits["evaluation"] or fsplits["calibration"] != splits["calibration"]:
            inputs.notes.append("head-mean run has different splits for this task: its function vector is not used")
        elif "fv_direction" not in arrays:
            inputs.notes.append("head-mean run has no function vector for this task")
        else:
            inputs.fv_dir, inputs.fv_arrays = fd, arrays
            inputs.fv_qual = _optional_json(fd / "qualification.json")
            inputs.fv_cal = _optional_json(fd / "calibration.json")
    else:
        inputs.notes.append("task absent from the head-mean run: no function vector")
    return inputs


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


class Trajectories:
    def __init__(self, cfg: TrajectoriesConfig, fv_run: Path, learned_run: Path, run_id: str | None,
                 config_path: str | None) -> None:
        self.cfg = cfg
        self.fv_root, self.learned_root = Path(fv_run), Path(learned_run)
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_trajectories_{self.learned_root.name}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core" / "tasks").mkdir(parents=True)
        (self.root / "figures").mkdir()
        self.log = setup_logging(self.root / "log.txt")
        self.prof = Profiler()
        self.run_cfg, learned_meta = _load_run(self.learned_root)
        fv_cfg, fv_meta = _load_run(self.fv_root)
        problems = []
        if learned_meta.get("control") != "learned_vector":
            problems.append(f"--learned-run has control {learned_meta.get('control')!r}, not learned_vector")
        if fv_meta.get("control") != "function_vector":
            problems.append(f"--fv-run has control {fv_meta.get('control')!r}, not function_vector")
        for section in ("model", "prompt", "data"):
            if config_to_dict(self.run_cfg)[section] != config_to_dict(fv_cfg)[section]:
                problems.append(f"the runs differ in their {section} configuration")
        if self.run_cfg.seed != fv_cfg.seed:
            problems.append("the runs have different seeds")
        if problems:
            raise ValueError("; ".join(problems))
        if cfg.device:
            self.run_cfg.model = replace(self.run_cfg.model, device=cfg.device)
        self.metadata: dict[str, Any] = {
            "run_id": self.root.name, "command": "trajectories", "argv": sys.argv, "config_path": config_path,
            "directions_version": __version__, "git": git_info(Path(__file__).resolve().parents[2]),
            "environment": environment_metadata(), "seed": cfg.seed, "config": config_to_dict(cfg),
            "runs": {"learned": {"path": str(self.learned_root), "run_id": learned_meta.get("run_id"),
                                 "git": learned_meta.get("git"), "seed": learned_meta.get("seed")},
                     "fv": {"path": str(self.fv_root), "run_id": fv_meta.get("run_id"),
                            "git": fv_meta.get("git"), "seed": fv_meta.get("seed")}},
            "run_config": config_to_dict(self.run_cfg),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(self.root / "metadata.json", self.metadata)

    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        with self.prof.section("total"):
            self._run()
        self.metadata["timings_seconds"] = self.prof.timings()
        self.metadata["profile"] = self.prof.report()
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("profile (sections >= 1 s):\n%s", self.prof.table())
        self.log.info("done in %.1fs -> %s", self.metadata["timings_seconds"]["total"], self.root)
        return self.root

    def _run(self) -> None:
        with self.prof.section("model_load"):
            self.backend = ModelBackend(self.run_cfg.model, run_seed=self.run_cfg.seed)
        self.prof.backend = self.backend
        self.metadata["model"] = self.backend.metadata()
        names = sorted(p.name for p in (self.learned_root / "core" / "tasks").iterdir() if p.is_dir())
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        for name in names:
            inputs = _load_task(name, self.learned_root, self.fv_root, self.run_cfg)
            if inputs is None:
                summary["skipped"][name] = "no learned-vector selection in the learned run"
                self.log.info("[%s] skipped: %s", name, summary["skipped"][name])
                continue
            with self.prof.section("task", name):
                summary["tasks"][name] = self._task(inputs)
        write_json(self.root / "core" / "summary.json", summary)
        self.log.info("summary:\n%s", format_summary(summary))

    # ------------------------------------------------------------------ #

    def _layers(self, inputs: TaskInputs) -> tuple[list[int], int]:
        primary = int(inputs.learned_qual["selection"]["layer"])
        if isinstance(self.cfg.layers, list):
            return sorted({int(l) for l in self.cfg.layers}), primary
        layers = {primary}
        if inputs.fv_qual and inputs.fv_qual.get("selection"):
            layers.add(int(inputs.fv_qual["selection"]["layer"]))
        return sorted(layers), primary

    def _constructions(self, inputs: TaskInputs, layer: int, calibration_base: ForwardResult,
                       calibration_prompts: list[Prompt]) -> dict[str, dict[str, Any]]:
        """Unit direction and strength of every construction available at ``layer``."""
        out: dict[str, dict[str, Any]] = {}
        la = inputs.learned_arrays
        layers = [int(l) for l in la["layers"]]
        if layer in layers:
            i = layers.index(layer)
            pc1 = np.asarray(la["pooled"][i], dtype=np.float64)
            pc1 /= np.linalg.norm(pc1)
            out["pca"] = {"direction": pc1, **self._calibrate_pc1(inputs, layer, pc1, calibration_base, calibration_prompts)}
            learned = np.asarray(la["learned"][i], dtype=np.float64)
            learned /= np.linalg.norm(learned)
            strength = alpha_from_calibration(inputs.learned_cal, layer)
            if strength is None:
                strength = {"alpha": float(la["learned_radii"][i]), "rho": 1.0, "source": "radius"}
            out["learned"] = {"direction": learned, **strength,
                              "qualified_here": bool(int(inputs.learned_qual["selection"]["layer"]) == layer)}
        else:
            inputs.notes.append(f"layer {layer} is not a candidate layer of the learned run: no PC1 or learned vector")
        if inputs.fv_arrays is not None:
            fv = np.asarray(inputs.fv_arrays["fv_direction"], dtype=np.float64)
            fv /= np.linalg.norm(fv)
            strength = alpha_from_calibration(inputs.fv_cal, layer)
            if strength is None:
                strength = {"alpha": float(np.linalg.norm(inputs.fv_arrays["fv"])), "rho": 1.0, "source": "natural_norm"}
            sel = (inputs.fv_qual or {}).get("selection")
            out["fv"] = {"direction": fv, **strength, "qualified_here": bool(sel and int(sel["layer"]) == layer)}
        return out

    def _calibrate_pc1(self, inputs: TaskInputs, layer: int, direction: np.ndarray, base: ForwardResult,
                       prompts: list[Prompt]) -> dict[str, Any]:
        """PC1's strength at ``layer``: the grid point (units of the median residual norm at the layer, taken
        from the learned run's calibration) with the largest mean per-token improvement on the calibration pool."""
        norms = (inputs.learned_cal or {}).get("layer_norms") or {}
        if str(layer) not in norms:
            raise ValueError(f"[{inputs.name}] no median residual norm for layer {layer} in the learned run's calibration")
        unit = float(norms[str(layer)])
        grid = []
        for rho in self.cfg.pc1_rho_grid:
            r = self.backend.run(prompts, interventions=[Intervention(layer, direction, rho * unit)])
            grid.append({"rho": rho, "alpha": rho * unit,
                         "mean_improvement": float(np.mean(r.logprob_per_token - base.logprob_per_token)),
                         "accuracy": float(np.mean(r.exact_match))})
        best = max(grid, key=lambda g: (g["mean_improvement"], -g["rho"]))
        return {"alpha": best["alpha"], "rho": best["rho"], "source": "calibration_pool_best", "unit": unit,
                "grid": grid, "qualified_here": None}

    # ------------------------------------------------------------------ #

    def _task(self, inputs: TaskInputs) -> dict[str, Any]:
        cfg, name, log = self.cfg, inputs.name, self.log
        seed = self.run_cfg.seed
        out_dir = self.root / "core" / "tasks" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pc = inputs.prompt_cfg
        ev = inputs.evaluation
        zs = [zero_shot_prompt(pc, x) for x in ev]
        rng = rng_for(seed, "fewshot_eval", name)  # the pipeline's few-shot prompts, exactly
        fs = [few_shot_prompt(pc, ev, x, rng) for x in ev]
        rng2 = rng_for(cfg.seed, "trajectories_fewshot_2", name)
        fs2 = [few_shot_prompt(pc, ev, x, rng2) for x in ev]
        drng = rng_for(cfg.seed, "trajectories_derangement", name)
        der = [deranged_prompt(pc, p, drng) for p in fs]
        layers, primary = self._layers(inputs)
        log.info("[%s] %d held-out prompts; layers %s (primary %d); %s", name, len(ev), layers, primary,
                 "; ".join(inputs.notes) or "both runs present")

        with self.prof.section("natural", name):
            base = self.backend.run(zs, capture=True)
            icl = self.backend.run(fs, capture=True)
            icl2 = self.backend.run(fs2, capture=True) if cfg.second_demo_sample else None
            deranged = self.backend.run(der, capture=True)
        assert base.residuals is not None and icl.residuals is not None and deranged.residuals is not None
        natural: dict[str, np.ndarray] = {"icl": icl.residuals - base.residuals, "task": icl.residuals - deranged.residuals}
        if icl2 is not None:
            assert icl2.residuals is not None
            natural["icl2"] = icl2.residuals - base.residuals
        conditions: dict[str, Any] = {"base": base.metrics_dict(), "icl": icl.metrics_dict(), "deranged": deranged.metrics_dict()}
        if icl2 is not None:
            conditions["icl2"] = icl2.metrics_dict()
        gap = float(np.mean(icl.logprob_per_token) - np.mean(base.logprob_per_token))
        U = None
        if cfg.remove_answer_direction:
            U = self.backend.unembedding_directions(self.backend.first_target_token_ids(zs))
            U = U / np.linalg.norm(U, axis=1, keepdims=True)
        log.info("[%s] zero-shot lp/tok %.3f, few-shot %.3f (gap %.3f), deranged %.3f%s", name,
                 conditions["base"]["logprob_per_token_mean"], conditions["icl"]["logprob_per_token_mean"], gap,
                 conditions["deranged"]["logprob_per_token_mean"],
                 f", second sample {conditions['icl2']['logprob_per_token_mean']:.3f}" if icl2 is not None else "")

        cal_prompts = [zero_shot_prompt(pc, x) for x in inputs.calibration]
        with self.prof.section("calibration_base", name):
            cal_base = self.backend.run(cal_prompts)

        result: dict[str, Any] = {"task": name, "registry_task": inputs.learned_qual.get("registry_task"),
                                  "n_examples": len(ev), "n_read_points": self.backend.n_layers + 1,
                                  "layers": layers, "primary_layer": primary, "notes": inputs.notes,
                                  "conditions": conditions, "fewshot_gap_per_token": gap,
                                  "answer_direction_removed": cfg.remove_answer_direction, "per_layer": {}}
        arrays: dict[str, np.ndarray] = {}
        for m, D in natural.items():
            arrays[f"mean_delta_{m}"] = D.mean(axis=1).astype(np.float16)
        for layer in layers:
            with self.prof.section("layer", name, layer):
                res, arr = self._layer(inputs, layer, zs, base, natural, U, cal_base, cal_prompts)
            result["per_layer"][str(layer)] = res
            arrays.update({f"L{layer}_{k}": v for k, v in arr.items()})
        write_json(out_dir / "trajectories.json", result)
        np.savez_compressed(out_dir / "trajectories_arrays.npz", **arrays)
        if cfg.figures:
            from .figures import trajectory_figures

            trajectory_figures(self.root / "figures", result)
        return _task_summary(result)

    # ------------------------------------------------------------------ #

    def _layer(self, inputs: TaskInputs, layer: int, zs: list[Prompt], base: ForwardResult,
               natural: dict[str, np.ndarray], U: np.ndarray | None, cal_base: ForwardResult,
               cal_prompts: list[Prompt]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cfg, name, log, seed = self.cfg, inputs.name, self.log, self.cfg.seed
        with self.prof.section("strengths", name, layer):
            cons = self._constructions(inputs, layer, cal_base, cal_prompts)
        assert base.residuals is not None
        deltas: dict[str, np.ndarray] = dict(natural)
        info: dict[str, Any] = {"layer": layer, "constructions": {}, "isotropic": {"n": cfg.n_isotropic}}
        arrays: dict[str, np.ndarray] = {}
        base_lp = float(np.mean(base.logprob_per_token))
        # the constructions' steered passes
        for c in CONSTRUCTIONS:
            if c not in cons:
                continue
            spec = cons[c]
            with self.prof.section("steered", name, layer, c):
                r = self.backend.run(zs, interventions=[Intervention(layer, spec["direction"], spec["alpha"])], capture=True)
            if cfg.determinism_check and "determinism_check" not in self.metadata and c == "learned":
                again = self.backend.run(zs, interventions=[Intervention(layer, spec["direction"], spec["alpha"])], capture=True)
                assert again.residuals is not None
                identical = bool(np.array_equal(again.residuals, r.residuals) and np.array_equal(again.logprob_sum, r.logprob_sum))
                self.metadata["determinism_check"] = {"task": name, "layer": layer, "construction": c, "identical": identical}
                if not identical:
                    raise RuntimeError("determinism check failed: a repeated steered pass differs")
            assert r.residuals is not None
            deltas[c] = r.residuals - base.residuals
            m = r.metrics_dict()
            info["constructions"][c] = {k: v for k, v in spec.items() if k != "direction"}
            info["constructions"][c].update({"metrics": m, "improvement_per_token": m["logprob_per_token_mean"] - base_lp})
            arrays[f"mean_delta_{c}"] = deltas[c].mean(axis=1).astype(np.float16)
        present = [c for c in CONSTRUCTIONS if c in deltas]
        named = [n for n in NAMED if n in deltas]
        # pairwise cosines among the named trajectories (from the injection layer on when a construction is involved)
        pairs: dict[str, Any] = {}
        for i, a in enumerate(named):
            for b in named[i + 1:]:
                start = layer if (a in CONSTRUCTIONS or b in CONSTRUCTIONS) else 0
                key = f"{a}~{b}"
                cos = pair_cosines(deltas[a], deltas[b], start)
                arrays[f"cos_{key}"] = cos.astype(np.float32)
                entry: dict[str, Any] = {"a": a, "b": b, "start": start,
                                         "mean_trajectory_cosine": mean_cosines(deltas[a].mean(axis=1), deltas[b].mean(axis=1), start).tolist()}
                if U is not None:
                    cos_na = pair_cosines(deltas[a], deltas[b], start, U)
                    arrays[f"cos_noanswer_{key}"] = cos_na.astype(np.float32)
                if b in REFERENCES or a in REFERENCES:
                    ref, other = (b, a) if b in REFERENCES else (a, b)
                    frac = projection_fraction(deltas[other], deltas[ref], start)
                    arrays[f"projection_{other}_on_{ref}"] = frac.astype(np.float32)
                pairs[key] = entry
        # matched isotropic directions at each construction's norm: the floor of every pair involving it
        iso_deltas: dict[str, list[np.ndarray]] = {c: [] for c in present}
        iso_metrics: dict[str, list[dict[str, float]]] = {c: [] for c in present}
        for c in present:
            alpha = cons[c]["alpha"]
            for k in range(cfg.n_isotropic):
                v = unit_vectors(rng_for(seed, "trajectories_isotropic", name, layer, c, k), 1, self.backend.hidden_size)[0]
                with self.prof.section("isotropic", name, layer):
                    r = self.backend.run(zs, interventions=[Intervention(layer, v, alpha)], capture=True)
                assert r.residuals is not None
                iso_deltas[c].append(r.residuals - base.residuals)
                iso_metrics[c].append(r.metrics_dict())
                del r
        for c in present:
            info["isotropic"][c] = {"alpha": cons[c]["alpha"],
                                    "improvement_per_token_mean": float(np.mean([m["logprob_per_token_mean"] for m in iso_metrics[c]])
                                                                       - float(np.mean(base.logprob_per_token)))}
        # The generic response (D32, amended): what any perturbation of these norms does downstream, the mean
        # of every isotropic control's trajectory, per example and read point; removed from both vectors of a
        # pair like the answer direction. A control's own floor removes the mean of the other controls.
        all_iso = [d for c in present for d in iso_deltas[c]]
        G: np.ndarray | None = None
        if cfg.remove_generic_response and all_iso:
            total = np.zeros_like(all_iso[0], dtype=np.float64)
            for d in all_iso:
                total += d
            G = unit_rows(total / len(all_iso))
            arrays["mean_generic"] = (total / len(all_iso)).mean(axis=1).astype(np.float16)
            info["generic_response"] = {"n_controls": len(all_iso),
                                        "cos_with_answer_direction": None if U is None else
                                        [float(np.nanmedian(np.einsum("nd,nd->n", G[m], U))) if m >= layer else None for m in range(G.shape[0])]}
        variants: dict[str, np.ndarray | None] = {"raw": None}
        if U is not None:
            variants["answer_removed"] = U
        if G is not None:
            variants["generic_removed"] = G
        for tag, R in variants.items():
            if tag == "raw":
                continue
            for i, a in enumerate(named):
                for b in named[i + 1:]:
                    if tag == "answer_removed" and f"cos_noanswer_{a}~{b}" in arrays:
                        continue  # computed above
                    start = layer if (a in CONSTRUCTIONS or b in CONSTRUCTIONS) else 0
                    arrays[f"cos_{_variant_prefix(tag)}{a}~{b}"] = pair_cosines(deltas[a], deltas[b], start, R).astype(np.float32)
        floors: dict[str, dict[str, dict[str, np.ndarray]]] = {tag: {c: {} for c in present} for tag in variants}
        for c in present:
            for j_c, d_iso in enumerate(iso_deltas[c]):
                loo: np.ndarray | None = None
                if G is not None:
                    loo = unit_rows((total - d_iso) / (len(all_iso) - 1)) if len(all_iso) > 1 else G
                for other in named:
                    for tag, R in variants.items():
                        rem = loo if tag == "generic_removed" else R
                        floors[tag][c].setdefault(other, []).append(pair_cosines(d_iso, deltas[other], layer, rem))  # type: ignore[arg-type]
                del loo
        del all_iso, iso_deltas
        for tag in variants:
            for c in present:
                floors[tag][c] = {o: np.concatenate(v, axis=1) for o, v in floors[tag][c].items()}  # (L+1, k*n)
                for o in floors[tag][c]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan rows below the injection layer
                        arrays[f"floor_{_variant_prefix(tag)}{c}_vs_{o}"] = np.nanmedian(floors[tag][c][o], axis=1).astype(np.float32)
        # summaries: every construction against the natural trajectories and against the other constructions
        brng = rng_for(seed, "trajectories_bootstrap", name, layer)
        summaries: dict[str, Any] = {}
        for c in present:
            for o in named:
                if o == c:
                    continue
                key = f"{c}~{o}" if f"{c}~{o}" in pairs else f"{o}~{c}"
                for tag in variants:
                    real = arrays[f"cos_{_variant_prefix(tag)}{key}"].astype(np.float64)
                    s = summarise_alignment(real, floors[tag][c][o], layer, brng, cfg.n_boot, cfg.ci_alpha)
                    if tag == "raw":
                        summaries[f"{c}->{o}"] = s
                    else:
                        summaries[f"{c}->{o}"][tag] = s
        ceilings: dict[str, Any] = {}
        for tag in variants:
            suffix = "" if tag == "raw" else f"_{tag}"
            if "icl2" in deltas:
                ceilings[f"icl~icl2{suffix}"] = bootstrap_median_ci_rows(arrays[f"cos_{_variant_prefix(tag)}icl~icl2"].astype(np.float64), brng, n_boot=cfg.n_boot, alpha=cfg.ci_alpha)
            ceilings[f"icl~task{suffix}"] = bootstrap_median_ci_rows(arrays[f"cos_{_variant_prefix(tag)}icl~task"].astype(np.float64), brng, n_boot=cfg.n_boot, alpha=cfg.ci_alpha)
        info.update({"pairs": pairs, "summaries": summaries, "ceilings": ceilings, "variants": list(variants)})
        for c in present:
            s = summaries.get(f"{c}->icl")
            if s and "peak" in s:
                log.info("[%s] L%d %-7s alpha %.2f (%s) %+.2f nats/tok | cos with ICL: inj %.2f peak %.2f@%.2f final %.2f "
                         "(floor %.2f) %s%s", name, layer, c, cons[c]["alpha"], cons[c]["source"],
                         info["constructions"][c]["improvement_per_token"], s["at_injection"], s["peak"],
                         s["peak_depth_fraction"], s["final"], s["floor_at_final"], s["label"], _variant_note(s))
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                s = summaries.get(f"{a}->{b}")
                if s and "peak" in s:
                    log.info("[%s] L%d %s~%s: inj %.2f peak %.2f@%.2f final %.2f (floor %.2f) %s%s", name, layer, a, b,
                             s["at_injection"], s["peak"], s["peak_depth_fraction"], s["final"], s["floor_at_final"], s["label"],
                             _variant_note(s))
        if "icl~icl2" in ceilings:
            log.info("[%s] L%d ceiling cos(icl, icl2): final %.2f (generic removed %s); cos(icl, task): final %.2f", name, layer,
                     ceilings["icl~icl2"]["median"][-1],
                     f"{ceilings['icl~icl2_generic_removed']['median'][-1]:.2f}" if "icl~icl2_generic_removed" in ceilings else "n/a",
                     ceilings["icl~task"]["median"][-1])
        return info, arrays


def _variant_note(s: dict[str, Any]) -> str:
    parts = []
    for tag, label in (("answer_removed", "no answer"), ("generic_removed", "no generic")):
        v = s.get(tag)
        if v and "final" in v:
            parts.append(f"{label}: peak {v['peak']:.2f} final {v['final']:.2f} (floor {v['floor_at_final']:.2f}) {v['label']}")
    return (" | " + " | ".join(parts)) if parts else ""


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def _task_summary(result: dict[str, Any]) -> dict[str, Any]:
    """The per-task rows of ``core/summary.json``: per layer and construction, the strength, the effect and the
    alignment summary against each natural trajectory and each other construction."""
    out: dict[str, Any] = {"layers": result["layers"], "primary_layer": result["primary_layer"],
                           "fewshot_gap_per_token": result["fewshot_gap_per_token"], "per_layer": {}}
    for layer, info in result["per_layer"].items():
        rows: dict[str, Any] = {}
        for c, spec in info["constructions"].items():
            row: dict[str, Any] = {"alpha": spec["alpha"], "rho": spec["rho"], "source": spec["source"],
                                   "qualified_here": spec.get("qualified_here"),
                                   "improvement_per_token": spec["improvement_per_token"],
                                   "gap_fraction": (spec["improvement_per_token"] / result["fewshot_gap_per_token"]
                                                    if result["fewshot_gap_per_token"] else None),
                                   "accuracy": spec["metrics"]["accuracy"], "alignment": {}}
            for key, s in info["summaries"].items():
                if not key.startswith(f"{c}->") or "peak" not in s:
                    continue
                other = key.split("->")[1]
                a = {k: s[k] for k in ("at_injection", "peak", "peak_depth_fraction", "final", "floor_at_final", "label")}
                for tag in ("answer_removed", "generic_removed"):
                    if tag in s and "peak" in s[tag]:
                        a[tag] = {k: s[tag][k] for k in ("peak", "final", "floor_at_final", "label")}
                row["alignment"][other] = a
            rows[c] = row
        ceilings = {k: v["median"][-1] for k, v in info["ceilings"].items()}
        out["per_layer"][layer] = {"constructions": rows, "ceiling_final": ceilings}
    return out


def format_summary(summary: dict[str, Any]) -> str:
    """One line per task, layer and construction: effect, alignment with the natural trajectory (at injection,
    peak, final, floor), the same with the answer direction removed, and the cross-construction final cosines."""
    lines = ["| task | layer | construction | gap frac | cos ICL inj | peak (depth) | final | floor | label | no generic: peak final (floor) label | vs pca | vs fv | vs learned | ceiling (no generic) |",
             "|" + "---|" * 15]
    for task, t in summary.get("tasks", {}).items():
        for layer, info in t["per_layer"].items():
            for c, row in info["constructions"].items():
                a = row["alignment"].get("icl", {})
                cross = []
                for o in CONSTRUCTIONS:
                    x = row["alignment"].get(o)
                    cross.append("" if o == c else ("-" if not x else f"{x['final']:.2f}"))
                ng = a.get("generic_removed", {})
                ng_text = (f"{ng['peak']:.2f} {ng['final']:.2f} ({ng['floor_at_final']:.2f}) {ng['label']}" if ng else "-")
                ceil = info["ceiling_final"]
                gap = "-" if row["gap_fraction"] is None else f"{row['gap_fraction']:.2f}"
                lines.append(
                    f"| {task} | {layer}{'*' if int(layer) == t['primary_layer'] else ''} | {c} | {gap} | "
                    f"{a.get('at_injection', float('nan')):.2f} | {a.get('peak', float('nan')):.2f} ({a.get('peak_depth_fraction', float('nan')):.2f}) | "
                    f"{a.get('final', float('nan')):.2f} | {a.get('floor_at_final', float('nan')):.2f} | {a.get('label', '')} | {ng_text} | "
                    f"{cross[0]} | {cross[1]} | {cross[2]} | {ceil.get('icl~icl2', float('nan')):.2f} ({ceil.get('icl~icl2_generic_removed', float('nan')):.2f}) |")
    return "\n".join(lines)


def run_trajectories(cfg: TrajectoriesConfig, fv_run: str | Path, learned_run: str | Path,
                     run_id: str | None = None, config_path: str | None = None) -> Path:
    return Trajectories(cfg, Path(fv_run), Path(learned_run), run_id, config_path).run()
