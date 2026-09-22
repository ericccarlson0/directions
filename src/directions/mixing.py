"""The geometry test (docs/DECISIONS.md D40): is a control a value on a dial or a choice among programs?

A control is mixed between two end labels a and b of a family, v(t) = alpha * normalise((1 - t) u_a + t u_b),
injected at label a's selected layer and strength on label a's held-out zero-shot prompts. At every weight t the
teacher-forced log-probability of each of the family's candidate continuations is read and turned into a mass
(softmax over the candidates), or, for two tasks with different inputs, each task's own held-out effect. The
dilution null mixes each endpoint with random unit directions along the same path, so that a rise of an
intermediate label in the middle is read against the endpoint's effect merely fading. An intermediate label's
mass that exceeds both endpoints and the null at an interior weight, with an interior maximum, is a parameter;
endpoints that cross with no intermediate rise are a switch; endpoints that do not cross are neither.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._version import __version__
from .config import Config, MixingConfig, MixingPair, PromptConfig
from .model import Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .prompts import Prompt
from .runinfo import git_info, read_json, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .stats import paired_excess_test
from .tasks import Item
from .trajectories import _items, _load_run, alpha_from_calibration


class _Skip(Exception):
    pass


def mix(u_a: np.ndarray, u_b: np.ndarray, t: float) -> np.ndarray:
    """The unit direction (1 - t) u_a + t u_b normalised; u_a and u_b are unit vectors."""
    v = (1.0 - t) * u_a + t * u_b
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _candidate_prompts(item: Item, readout: str, pc: PromptConfig, operands: Sequence[int]) -> list[tuple[str, Prompt]]:
    """The family's candidate continuations of one held-out item, as (candidate key, scored prompt)."""
    prompt = pc.query_template.format(input=item.input)
    if readout == "list_words":
        words = [w.strip() for w in item.input.split(", ")]
        return [(f"pos_{i + 1}", Prompt(prompt=prompt, target=pc.target_template.format(output=w), query=item, demos=()))
                for i, w in enumerate(words)]
    if readout == "operands":
        n = int(item.input)
        ref = pc.target_template.format(output=item.input)
        return [(f"add_{k}", Prompt(prompt=prompt, target=pc.target_template.format(output=str(n + k)), query=item,
                                    demos=(), score_reference=ref)) for k in operands]
    raise ValueError(f"_candidate_prompts is for distribution readouts, not {readout!r}")


def _own_prompts(items: list[Item], pc: PromptConfig) -> list[Prompt]:
    return [Prompt(prompt=pc.query_template.format(input=it.input), target=pc.target_template.format(output=it.output),
                   query=it, demos=()) for it in items]


def _masses(backend: ModelBackend, per_item: list[list[tuple[str, Prompt]]], layer: int, alpha: float,
            direction: np.ndarray) -> np.ndarray:
    """Per-item softmax mass over the candidates under the control ``alpha * direction`` at ``layer``, (n_items, K)."""
    K = len(per_item[0])
    prompts = [p for row in per_item for _, p in row]
    res = backend.run(prompts, interventions=[Intervention(layer, (alpha * direction).astype(np.float32))])
    lp = res.logprob_sum.reshape(len(per_item), K)
    lp = lp - lp.max(axis=1, keepdims=True)
    m = np.exp(lp)
    m /= m.sum(axis=1, keepdims=True)
    return m


def _effect(backend: ModelBackend, prompts: list[Prompt], layer: int, alpha: float, direction: np.ndarray,
            baseline: np.ndarray) -> np.ndarray:
    """Per-item change in the own-target log-probability per token under the control, against ``baseline``."""
    res = backend.run(prompts, interventions=[Intervention(layer, (alpha * direction).astype(np.float32))])
    return res.logprob_per_token - baseline


def verdict(weights: list[float], real: np.ndarray, null_stack: np.ndarray, keys: list[str], a_key: str, b_key: str,
            intermediates: list[str], interior: tuple[float, float], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    """Read a distribution mix as parameter / switch / neither.

    ``real`` is (n_weights, n_items, K); ``null_stack`` is (n_sides, n_weights, n_items, K). ``a_key``/``b_key``
    are the candidate keys of the two endpoints' own labels (``pos_1``/``pos_3``, or ``add_1``/``add_5``).
    """
    W = np.asarray(weights, dtype=float)
    ai, bi = keys.index(a_key), keys.index(b_key)
    real_mean = real.mean(axis=1)  # (n_weights, K)
    null_mean = null_stack.mean(axis=(0, 2)) if null_stack.shape[0] else np.full_like(real_mean, np.nan)
    endpoints_own = bool(real_mean[0, ai] > real_mean[0, bi] and real_mean[-1, bi] > real_mean[-1, ai])
    diff = real_mean[:, ai] - real_mean[:, bi]
    crosses = bool(endpoints_own and np.any(np.sign(diff[:-1]) != np.sign(diff[1:])))
    interior_idx = [j for j, w in enumerate(W) if interior[0] <= w <= interior[1]]
    inter: dict[str, Any] = {}
    parameter = False
    for key in intermediates:
        ci = keys.index(key)
        curve = real_mean[:, ci]
        argmax = int(curve.argmax())
        best: dict[str, Any] = {"weight": None, "excess_over_null": None, "p": None, "over_endpoints": False,
                                "interior_max": bool(interior[0] <= W[argmax] <= interior[1]), "argmax_weight": float(W[argmax])}
        for j in interior_idx:
            over_ep = bool(curve[j] > real_mean[0, ci] and curve[j] > real_mean[-1, ci])
            if null_stack.shape[0]:
                # paired excess of the intermediate's real mass over the dilution null at this weight: real per item
                # (n_items,), the null the same candidate's mass on each side (n_sides, n_items)
                t = paired_excess_test(real[j, :, ci], null_stack[:, j, :, ci], rng, n_boot=n_boot)
                excess, p = float(t.excess_mean), float(t.p_value)
            else:
                excess = p = None
            if over_ep and excess is not None and excess > 0 and p <= 0.05 and (best["excess_over_null"] is None or excess > best["excess_over_null"]):
                best.update({"weight": float(W[j]), "excess_over_null": excess, "p": p, "over_endpoints": True})
        inter[key] = best
        if best["over_endpoints"] and best["interior_max"] and best["p"] is not None and best["p"] <= 0.05:
            parameter = True
    label = "parameter" if parameter else ("switch" if crosses else "neither")
    return {"verdict": label, "endpoints_select_own": endpoints_own, "endpoints_cross": crosses, "keys": keys,
            "intermediates": inter, "real_mean": real_mean.tolist(), "null_mean": null_mean.tolist()}


class Mixing:
    def __init__(self, cfg: MixingConfig, runs: dict[str, tuple[str, str | None]], run_id: str | None,
                 config_path: str | None) -> None:
        self.cfg = cfg
        self.runs = runs
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_mixing"
        self.root = Path(cfg.output_dir) / run_id
        (self.root / "core").mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.root / "log.txt")
        self.run_cfg, _ = _load_run(Path(next(iter(runs.values()))[0]))
        if cfg.device:
            self.run_cfg = replace(self.run_cfg, model=replace(self.run_cfg.model, device=cfg.device))
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id,
            "git": git_info(), "environment": environment_metadata(),
            "runs": {fam: {"learned": lr, "fv": fr} for fam, (lr, fr) in runs.items()},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        self.backend = ModelBackend(self.run_cfg.model, run_seed=self.run_cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        summary: dict[str, Any] = {"pairs": [], "skipped": []}
        for pair in self.cfg.pairs:
            if pair.family not in self.runs:
                summary["skipped"].append({"family": pair.family, "labels": pair.labels, "reason": "no runs given"})
                continue
            for construction in pair.constructions:
                try:
                    out = self._pair(pair, construction)
                    summary["pairs"].append(out)
                    self.log.info("%s %s [%s]: %s (angle %.0f deg)", pair.family, "->".join(pair.labels), construction,
                                  out.get("verdict") or out.get("summary", ""), out.get("angle_deg", float("nan")))
                except _Skip as exc:
                    summary["skipped"].append({"family": pair.family, "labels": pair.labels,
                                               "construction": construction, "reason": str(exc)})
                    self.log.info("%s %s [%s]: skipped (%s)", pair.family, "->".join(pair.labels), construction, exc)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    def _endpoint(self, family: str, label: str, construction: str) -> tuple[Path, dict, dict]:
        learned_root, fv_root = self.runs[family]
        root = Path(fv_root) if construction == "fv" else Path(learned_root)
        if construction == "fv" and fv_root is None:
            raise _Skip("no head-mean run given")
        td = root / "core" / "tasks" / label
        q = read_json(td / "qualification.json") if (td / "qualification.json").exists() else None
        if not q or not q.get("selection") or not (td / "directions.npz").exists():
            raise _Skip(f"{label} not qualified under {construction}")
        return td, q, dict(np.load(td / "directions.npz"))

    def _pair(self, pair: MixingPair, construction: str) -> dict[str, Any]:
        a, b = pair.labels
        ta, qa, arr_a = self._endpoint(pair.family, a, construction)
        tb, qb, arr_b = self._endpoint(pair.family, b, construction)
        layer = int(qa["selection"]["layer"])
        cal = read_json(ta / "calibration.json") if (ta / "calibration.json").exists() else None
        if construction == "fv":
            if "fv_direction" not in arr_a or "fv_direction" not in arr_b:
                raise _Skip("no head-mean vector")
            u_a, u_b = _unit(arr_a["fv_direction"]), _unit(arr_b["fv_direction"])
            strength = alpha_from_calibration(cal, layer) or {"alpha": float(np.linalg.norm(arr_a["fv"]))}
        else:
            layers = [int(l) for l in arr_a["layers"]]
            i = layers.index(layer)
            u_a, u_b = _unit(arr_a["learned"][i]), _unit(arr_b["learned"][i])
            strength = alpha_from_calibration(cal, layer) or {"alpha": float(arr_a["learned_radii"][i])}
        alpha = float(strength["alpha"])
        pc = _prompt_cfg_for(self.run_cfg, pair.readout)
        weights = list(self.cfg.weights)
        angle = float(np.degrees(np.arccos(np.clip(float(u_a @ u_b), -1, 1))))
        base = {"family": pair.family, "labels": pair.labels, "construction": construction, "layer": layer,
                "alpha": alpha, "angle_deg": angle, "weights": weights, "readout": pair.readout}
        rng = rng_for(self.cfg.seed, "mixing", pair.family, construction, a, b)
        d = u_a.shape[0]
        r_a = [_unit(rng.normal(size=d)) for _ in range(self.cfg.n_null)]
        r_b = [_unit(rng.normal(size=d)) for _ in range(self.cfg.n_null)]

        if pair.readout == "own_targets":
            items_a = _items(read_json(ta / "splits.json")["evaluation"])[: self.cfg.n_prompts]
            items_b = _items(read_json(tb / "splits.json")["evaluation"])[: self.cfg.n_prompts]
            pa, pb = _own_prompts(items_a, pc), _own_prompts(items_b, pc)
            base_a = self.backend.run(pa).logprob_per_token
            base_b = self.backend.run(pb).logprob_per_token
            eff_a = [float(_effect(self.backend, pa, layer, alpha, mix(u_a, u_b, t), base_a).mean()) for t in weights]
            eff_b = [float(_effect(self.backend, pb, layer, alpha, mix(u_a, u_b, t), base_b).mean()) for t in weights]
            diff = np.array(eff_a) - np.array(eff_b)
            crosses = bool(np.any(np.sign(diff[:-1]) != np.sign(diff[1:])))
            return {**base, "n_items_a": len(items_a), "n_items_b": len(items_b), "effect_a": eff_a, "effect_b": eff_b,
                    "endpoints_cross": crosses, "verdict": "switch" if crosses else "neither"}

        items = _items(read_json(ta / "splits.json")["evaluation"])[: self.cfg.n_prompts]
        per_item = [_candidate_prompts(it, pair.readout, pc, pair.operands) for it in items]
        keys = [k for k, _ in per_item[0]]
        if self.cfg.determinism_check:
            m1 = _masses(self.backend, per_item, layer, alpha, mix(u_a, u_b, 0.5))
            m2 = _masses(self.backend, per_item, layer, alpha, mix(u_a, u_b, 0.5))
            base["determinism_check"] = bool(np.array_equal(m1, m2))
        real = np.stack([_masses(self.backend, per_item, layer, alpha, mix(u_a, u_b, t)) for t in weights])
        null_curves = [np.stack([_masses(self.backend, per_item, layer, alpha, mix(u_a, r, t)) for t in weights]) for r in r_a]
        null_curves += [np.stack([_masses(self.backend, per_item, layer, alpha, mix(r, u_b, t)) for t in weights]) for r in r_b]
        null_stack = np.stack(null_curves) if null_curves else np.empty((0, len(weights), len(items), real.shape[2]))
        a_key = "pos_1" if pair.readout == "list_words" else a
        b_key = "pos_3" if pair.readout == "list_words" else b
        # list_words end labels kth_1/kth_3 read as pos_1/pos_3; intermediates kth_2 -> pos_2
        inter_keys = [("pos_" + k.split("_")[1]) if pair.readout == "list_words" else k for k in pair.intermediates]
        v = verdict(weights, real, null_stack, keys, a_key, b_key, inter_keys, tuple(self.cfg.interior), rng, self.cfg.n_boot)
        return {**base, "n_items": len(items), **v}


def _prompt_cfg_for(cfg: Config, readout: str) -> PromptConfig:
    return replace(cfg.prompt, target_scoring="changed_tokens" if readout == "operands" else "all")


def run_mixing(cfg: MixingConfig, runs: dict[str, tuple[str, str | None]], run_id: str | None = None,
               config_path: str | None = None) -> Path:
    return Mixing(cfg, runs, run_id, config_path).run()
