"""The composition test with readouts that can see an intermediate (docs/DECISIONS.md D42).

D41 read a composed control's perturbation against the first component's natural difference and the
composition's, which are 0.85-0.98 aligned, so its statistic had little room to read anything. This stage reads
the same compositions (the D41 family runs, their controls and prompts) three ways:

1. component-specific references: each natural difference residualised against the other per prompt, and the
   perturbation's cosine with the step-one-specific direction minus its cosine with the composition-specific one
   (``e_perp``), against the matched isotropic control;
2. the logit lens at the query token: the gain of the intermediate's and the final's first tokens over the
   zero-shot run at every read point, for the model's own few-shot computation and for the injected controls;
3. layer-wise context masking of the model's own few-shot computation (Yang et al. 2025, "Internal
   Chain-of-Thought"): from block L on the query may not read the demonstrations, and the intermediate's
   first-token log-probability is read at the output for every L.

The first component's control alone is the power check of readouts 1 and 2 (a perturbation that carries step one
and nothing else); the first component's demonstrations are the lens's own power check.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ._version import __version__
from .config import StagingComposition, StagingConfig
from .determinism import compare_arrays, forward_arrays
from .model import ContextMask, ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .prompts import Prompt, few_shot_prompt, zero_shot_prompt
from .runinfo import git_info, read_json, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .serial import _Skip, load_control
from .stats import paired_bootstrap_test, paired_excess_test
from .tasks import LIST_SEPARATOR, Item, build_task
from .trajectories import _items, _load_run, unit_vectors

# --------------------------------------------------------------------------- #
# Intermediates
# --------------------------------------------------------------------------- #

_WORD_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "uppercase": str.upper,
    "last_word": lambda s: s.split(LIST_SEPARATOR)[-1],
}


def step_function(step: str) -> Callable[[str], str | None]:
    """One step of an intermediate chain as a function of a word (None where the step does not cover it)."""
    if step in _WORD_FUNCTIONS:
        return _WORD_FUNCTIONS[step]
    mapping = {it.input: it.output for it in build_task(step).items}
    return mapping.get


def apply_steps(word: str, fns: Sequence[Callable[[str], str | None]]) -> str | None:
    out: str | None = word
    for fn in fns:
        if out is None:
            return None
        out = fn(out)
    return out


# --------------------------------------------------------------------------- #
# Geometry and readings (NumPy; the tests check them on synthetic data)
# --------------------------------------------------------------------------- #


def residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` minus its projection on ``b``, row by row (the last axis is the vector)."""
    bb = np.sum(b * b, axis=-1, keepdims=True)
    coef = np.divide(np.sum(a * b, axis=-1, keepdims=True), bb, out=np.zeros_like(bb), where=bb > 0)
    return a - coef * b


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def specific_excess(delta: np.ndarray, d_g: np.ndarray, d_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``e_perp = cos(delta, g_perp) - cos(delta, c_perp)`` per row, with ``g_perp`` = ``d_g`` residualised against
    ``d_c`` and ``c_perp`` the reverse (D42 readout 1); also the two cosines."""
    g_perp, c_perp = residualise(d_g, d_c), residualise(d_c, d_g)
    cg, cc = rowwise_cos(delta, g_perp), rowwise_cos(delta, c_perp)
    return cg - cc, cg, cc


def windows(flags: Sequence[bool]) -> list[int]:
    """Indices at which a window starts: two consecutive True flags (the project's sustained rule)."""
    return [i for i in range(len(flags) - 1) if flags[i] and flags[i + 1]]


def reference_verdict(pos: Sequence[bool], neg: Sequence[bool]) -> dict[str, Any]:
    """Readout 1 (D42): staircase (a positive window, then a negative one deeper), component only (a positive
    window and no negative one after it) or composed only (no positive window). ``pos``/``neg`` are the per
    read point flags in depth order."""
    p, n = windows(pos), windows(neg)
    if not p:
        return {"verdict": "composed_only", "positive_windows": p, "negative_windows": n}
    later = [j for j in n if j > p[0] + 1]
    return {"verdict": "staircase" if later else "component_only", "positive_windows": p, "negative_windows": n}


def lens_verdict(int_over_fin: Sequence[bool], int_up: Sequence[bool], fin_over_int: Sequence[bool]) -> dict[str, Any]:
    """Readout 2 (D42): lens-staged when a window of read points in which the intermediate's gain exceeds the
    final's and is itself positive (above the controls it is tested against) begins before the first window in
    which the final's gain exceeds the intermediate's."""
    staged_w = windows([a and b for a, b in zip(int_over_fin, int_up)])
    fin_w = windows(fin_over_int)
    staged = bool(staged_w) and (not fin_w or staged_w[0] < fin_w[0])
    return {"staged": staged, "intermediate_windows": staged_w, "final_windows": fin_w}


def masking_verdict(above_start: Sequence[bool], above_end: Sequence[bool]) -> dict[str, Any]:
    """Readout 3 (D42): masking-staged when the intermediate's log-probability exceeds both its fully masked
    (L = 0) and its unmasked (L = n) value at two consecutive masking layers."""
    w = windows([a and b for a, b in zip(above_start, above_end)])
    return {"staged": bool(w), "windows": w}


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class Staging:
    def __init__(self, cfg: StagingConfig, fv_run: str | None, learned_run: str, run_id: str | None, config_path: str | None) -> None:
        self.cfg = cfg
        self.fv_root = Path(fv_run) if fv_run else None
        self.learned_root = Path(learned_run)
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_staging_{self.learned_root.name}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core").mkdir(parents=True)
        self.log = setup_logging(self.root / "log.txt")
        self.run_cfg, learned_meta = _load_run(self.learned_root)
        if cfg.device:
            self.run_cfg = replace(self.run_cfg, model=replace(self.run_cfg.model, device=cfg.device))
        fv_meta = _load_run(self.fv_root)[1] if self.fv_root else None
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id,
            "git": git_info(), "environment": environment_metadata(),
            "runs": {"learned": {"path": str(self.learned_root), "run_id": learned_meta.get("run_id"), "seed": learned_meta.get("seed")},
                     "fv": None if fv_meta is None else {"path": str(self.fv_root), "run_id": fv_meta.get("run_id"), "seed": fv_meta.get("seed")}},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        self.backend = ModelBackend(self.run_cfg.model, run_seed=self.run_cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        summary: dict[str, Any] = {"compositions": [], "skipped": []}
        for comp in self.cfg.compositions:
            try:
                summary["compositions"].append(self._composition(comp))
            except _Skip as exc:
                summary["skipped"].append({"task": comp.task, "reason": str(exc)})
                self.log.info("%s: skipped (%s)", comp.task, exc)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    # ------------------------------------------------------------------ #

    def _control(self, label: str, construction: str) -> dict[str, Any]:
        root = self.fv_root if construction == "fv" else self.learned_root
        if root is None:
            raise _Skip("no head-mean run given")
        return load_control(root, label, construction)

    def _pool(self, label: str) -> list[Item]:
        path = self.learned_root / "core" / "tasks" / label / "splits.json"
        if not path.exists():
            raise _Skip(f"{label} absent from the learned run")
        return _items(read_json(path)["evaluation"])

    def _capture(self, prompts: list[Prompt], interventions: Sequence[Intervention] = ()) -> ForwardResult:
        r = self.backend.run(prompts, interventions=interventions, capture=True)
        assert r.residuals is not None
        return r

    def _first_token(self, text: str) -> int:
        """The first token of a target-rendered word (the token the query position predicts)."""
        return int(self.backend.tokenizer.encode(text, add_special_tokens=False)[0])

    def _check(self, what: str, a: ForwardResult, b: ForwardResult) -> None:
        cmp = compare_arrays(forward_arrays(a), forward_arrays(b))
        self.metadata.setdefault("determinism_check", {})[what] = {"identical": cmp["identical"]}
        if not cmp["identical"]:
            raise RuntimeError(f"determinism check failed: a repeated {what} pass differs")

    def _lens(self, H: np.ndarray, token_sets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Logit-lens log-probabilities (L+1, N) of each prompt's token per set, and its rank (0 = top), at read
        points 1..L of the residuals ``H`` (L+1, N, d); read point 0 is left NaN."""
        L1, N, _ = H.shape
        out = {f"lp_{k}": np.full((L1, N), np.nan) for k in token_sets}
        out.update({f"rank_{k}": np.full((L1, N), np.nan) for k in token_sets})
        idx = {k: torch.as_tensor(v, device=self.backend.device) for k, v in token_sets.items()}
        for m in range(1, L1):
            for s in range(0, N, 64):
                logits = self.backend.logits_from_residual(H[m, s : s + 64])
                lps = torch.log_softmax(logits, dim=-1)
                for k, ids in idx.items():
                    sel = ids[s : s + 64, None]
                    g = torch.gather(lps, 1, sel)
                    out[f"lp_{k}"][m, s : s + 64] = g[:, 0].double().cpu().numpy()
                    out[f"rank_{k}"][m, s : s + 64] = (lps > g).sum(1).double().cpu().numpy()
        return out

    def _composition(self, comp: StagingComposition) -> dict[str, Any]:
        cfg, t0 = self.cfg, time.time()
        name = comp.task
        pc = self.run_cfg.prompt
        pool = self._pool(name)
        ev = pool if cfg.n_prompts is None else pool[: cfg.n_prompts]
        N = len(ev)
        L = self.backend.n_layers
        brng = rng_for(cfg.seed, "staging_bootstrap", name)
        test = lambda a, b: paired_bootstrap_test(a, b, brng, n_boot=cfg.n_boot)  # noqa: E731
        excess = lambda a, c: paired_excess_test(a, c, brng, n_boot=cfg.n_boot)  # noqa: E731

        # prompts: D41's, exactly (the pipeline's few-shot prompts; the references' seeds of the landmark stage)
        zs = [zero_shot_prompt(pc, x) for x in ev]
        rng = rng_for(self.run_cfg.seed, "fewshot_eval", name)
        fs = [few_shot_prompt(pc, pool, x, rng) for x in pool][:N]
        ref_prompts: dict[str, list[Prompt]] = {}
        for r in comp.references:
            rrng = rng_for(cfg.seed, "trajectories_reference_fewshot", name, r)
            ref_prompts[r] = [few_shot_prompt(pc, self._pool(r), x, rrng) for x in pool][:N]

        # the tokens: each intermediate's and the final's first token; items where they coincide leave the lens
        fin = np.array([self._first_token(pc.target_template.format(output=x.output)) for x in ev], dtype=np.int64)
        inter: dict[str, np.ndarray] = {}
        inter_words: dict[str, list[str | None]] = {}
        for it in comp.intermediates:
            fns = [step_function(s) for s in it.steps]
            words = [apply_steps(x.input, fns) for x in ev]
            if any(w is None for w in words):
                raise _Skip(f"intermediate {it.label} undefined on {sum(w is None for w in words)} items")
            inter_words[it.label] = words
            inter[it.label] = np.array([self._first_token(pc.target_template.format(output=w)) for w in words], dtype=np.int64)
        first_int = comp.intermediates[0].label
        keep = {k: v != fin for k, v in inter.items()}
        token_sets = {"fin": fin, **{f"int_{k}": v for k, v in inter.items()}}
        out: dict[str, Any] = {"task": name, "references": comp.references, "n_items": N, "n_layers": L,
                               "intermediates": {k: {"steps": next(i.steps for i in comp.intermediates if i.label == k),
                                                     "examples": [[ev[j].input, inter_words[k][j], ev[j].output] for j in range(min(5, N))],
                                                     "n_lens_items": int(keep[k].sum())}
                                                 for k in inter}}
        self.log.info("%s: %d prompts; intermediates %s", name, N,
                      ", ".join(f"{k} ({out['intermediates'][k]['n_lens_items']} lens items)" for k in inter))

        # natural captures
        base = self._capture(zs)
        icl = self._capture(fs)
        H: dict[str, np.ndarray] = {"base": base.residuals, "icl": icl.residuals}
        for r in comp.references:
            H[f"ref_{r}"] = self._capture(ref_prompts[r]).residuals
        out["conditions"] = {"base": base.metrics_dict(), "icl": icl.metrics_dict()}
        d_c = H["icl"] - H["base"]
        d_ref = {r: H[f"ref_{r}"] - H["base"] for r in comp.references}

        # readout 3: context masking of the model's own computation
        arrays: dict[str, np.ndarray] = {}
        if cfg.masking:
            out["masking"] = self._masking(fs, ev, pc, token_sets, keep, test, arrays)

        # steered captures per construction
        conds: dict[str, dict[str, Any]] = {}
        for c in cfg.constructions:
            try:
                cc = self._control(name, c)
            except _Skip as exc:
                self.log.info("%s [%s]: no composed control (%s)", name, c, exc)
                continue
            layer, vec = int(cc["layer"]), (cc["alpha"] * cc["unit"]).astype(np.float32)
            steered = self._capture(zs, [Intervention(layer, vec)])
            if cfg.determinism_check and "steered" not in self.metadata.get("determinism_check", {}):
                self._check("steered", steered, self._capture(zs, [Intervention(layer, vec)]))
            conds[f"{c}:composed"] = {"layer": layer, "H": steered.residuals, "ivs": [Intervention(layer, vec)],
                                      "effect": steered.logprob_per_token - base.logprob_per_token}
            d = self.backend.hidden_size

            def nulls(key: str, fixed: list[Intervention], at: int, alpha: float) -> None:
                """The matched isotropic controls of condition ``key``: its last injection replaced by random unit
                directions at the same layer and strength (the earlier injections kept)."""
                conds[key]["iso"] = []
                for k in range(cfg.n_isotropic):
                    u = unit_vectors(rng_for(cfg.seed, "staging_isotropic", name, key, k), 1, d)[0]
                    iv = Intervention(at, (alpha * u).astype(np.float32))
                    conds[f"{key}~iso{k}"] = {"layer": conds[key]["layer"], "null": True, "H": self._capture(zs, [*fixed, iv]).residuals}
                    conds[key]["iso"].append(f"{key}~iso{k}")

            nulls(f"{c}:composed", [], layer, cc["alpha"])
            try:
                cg = self._control(comp.references[0], c)
                lg = int(cg["layer"])
                ivg = Intervention(lg, (cg["alpha"] * cg["unit"]).astype(np.float32))
                conds[f"{c}:first_alone"] = {"layer": lg, "H": self._capture(zs, [ivg]).residuals}
                nulls(f"{c}:first_alone", [], lg, cg["alpha"])
                if c == "fv" and len(comp.references) > 1:
                    cf = self._control(comp.references[1], c)
                    lf = min(L - 1, lg + int(round(cfg.serial_offset_fraction * L)))
                    ivf = Intervention(lf, (cf["alpha"] * cf["unit"]).astype(np.float32))
                    conds[f"{c}:serial"] = {"layer": lg, "second_layer": lf, "H": self._capture(zs, [ivg, ivf]).residuals}
                    nulls(f"{c}:serial", [ivg], lf, cf["alpha"])
            except _Skip as exc:
                self.log.info("%s [%s]: no first-component control (%s)", name, c, exc)
            out.setdefault("controls", {})[c] = {"layer": layer, "alpha": cc["alpha"], "alpha_source": cc["alpha_source"],
                                                 "effect_per_token": float(conds[f"{c}:composed"]["effect"].mean())}

        # readout 1: component-specific references
        out["references_perp"] = self._references(comp, conds, H, d_c, d_ref, test, excess, arrays)
        # readout 2: the logit lens
        out["lens"] = self._lens_readout(comp, conds, H, token_sets, keep, test, excess, arrays)
        # exploratory: the removal edit along the step-one-specific direction
        if cfg.removal_edits:
            out["removal_edits"] = self._removal(comp, conds, zs, base, H["base"], d_c, d_ref)
        out["verdicts"] = self._verdicts(out)
        out["seconds"] = time.time() - t0
        td = self.root / "core" / name
        td.mkdir(parents=True, exist_ok=True)
        write_json(td / "staging.json", out)
        np.savez_compressed(td / "staging_arrays.npz", **{k: v.astype(np.float32) for k, v in arrays.items()})
        self.log.info("%s: %s (%.0f s)", name, out["verdicts"], out["seconds"])
        return {"task": name, "verdicts": out["verdicts"]}

    # ------------------------------------------------------------------ #

    def _masking(self, fs: list[Prompt], ev: list[Item], pc: Any, token_sets: dict[str, np.ndarray],
                 keep: dict[str, np.ndarray], test: Callable, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        L = self.backend.n_layers
        spans = np.array([self.backend.context_span(p, len(p.prompt) - len(pc.query_template.format(input=x.input)))
                          for p, x in zip(fs, ev)], dtype=np.int64)
        lp = {k: np.zeros((L + 1, len(fs))) for k in token_sets}
        top1 = {k: np.zeros((L + 1, len(fs)), dtype=bool) for k in token_sets}
        for Lm in range(L + 1):
            r = self.backend.run(fs, capture_logprobs=True, context_mask=ContextMask(Lm, spans))
            if self.cfg.determinism_check and Lm == L // 2 and "masked" not in self.metadata.get("determinism_check", {}):
                self._check("masked", r, self.backend.run(fs, capture_logprobs=True, context_mask=ContextMask(Lm, spans)))
            q = r.query_logprobs
            assert q is not None
            am = q.argmax(axis=1)
            for k, ids in token_sets.items():
                lp[k][Lm] = q[np.arange(len(fs)), ids]
                top1[k][Lm] = am == ids
        for k in token_sets:
            arrays[f"masking_lp_{k}"] = lp[k]
        rows: dict[str, Any] = {"spans_median": [float(np.median(spans[:, 0])), float(np.median(spans[:, 1]))], "per_intermediate": {}}
        for k in token_sets:
            if not k.startswith("int_"):
                continue
            sel = keep[k[4:]]
            if sel.sum() < 2:
                rows["per_intermediate"][k[4:]] = {"n_items": int(sel.sum()), "verdict": {"staged": False, "windows": []}}
                continue
            li = lp[k][:, sel]
            above0 = [test(li[Lm], li[0]) for Lm in range(L + 1)]
            aboveN = [test(li[Lm], li[L]) for Lm in range(L + 1)]
            v = masking_verdict([t.p_value <= self.cfg.alpha and t.mean_diff > 0 for t in above0],
                                [t.p_value <= self.cfg.alpha and t.mean_diff > 0 for t in aboveN])
            rows["per_intermediate"][k[4:]] = {
                "n_items": int(sel.sum()), "verdict": v,
                "lp_int_mean": li.mean(1).tolist(), "lp_fin_mean": lp["fin"][:, sel].mean(1).tolist(),
                "top1_int": top1[k][:, sel].mean(1).tolist(), "top1_fin": top1["fin"][:, sel].mean(1).tolist(),
                "p_above_masked": [t.p_value for t in above0], "p_above_unmasked": [t.p_value for t in aboveN]}
            self.log.info("masking %s: intermediate lp at L=0 %.2f, peak %.2f at L=%d, L=n %.2f; %s", k[4:], li[0].mean(),
                          li.mean(1).max(), int(li.mean(1).argmax()), li[L].mean(), v)
        return rows

    def _references(self, comp: StagingComposition, conds: dict[str, dict[str, Any]], H: dict[str, np.ndarray], d_c: np.ndarray,
                    d_ref: dict[str, np.ndarray], test: Callable, excess: Callable, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        cfg, L = self.cfg, self.backend.n_layers
        out: dict[str, Any] = {}
        for r in comp.references:
            d_g = d_ref[r]
            room = [float(np.median(np.linalg.norm(residualise(d_g[m], d_c[m]), axis=1) / np.maximum(np.linalg.norm(d_g[m], axis=1), 1e-30)))
                    for m in range(L + 1)]
            e_all: dict[str, np.ndarray] = {}
            for key, cd in conds.items():
                delta = cd["H"] - H["base"]
                e = np.full((L + 1, delta.shape[1]), np.nan)
                for m in range(cd["layer"] + 1, L + 1):
                    e[m] = specific_excess(delta[m].astype(np.float64), d_g[m].astype(np.float64), d_c[m].astype(np.float64))[0]
                e_all[key] = e
                arrays[f"eperp_{r}_{key.replace(':', '_')}"] = e
            rr: dict[str, Any] = {"room": room}
            for key, cd in conds.items():
                if cd.get("null"):
                    continue
                iso = np.stack([e_all[k2] for k2 in cd["iso"]])
                pts = list(range(cd["layer"] + 1, L + 1))
                rows, pos, neg = [], [], []
                for m in pts:
                    e = e_all[key][m]
                    tp, tn = test(e, np.zeros_like(e)), test(-e, np.zeros_like(e))
                    xp, xn = excess(e, iso[:, m]), excess(-e, -iso[:, m])
                    pos.append(tp.p_value <= cfg.alpha and xp.p_value <= cfg.alpha and tp.mean_diff > 0)
                    neg.append(tn.p_value <= cfg.alpha and xn.p_value <= cfg.alpha and tn.mean_diff > 0)
                    rows.append({"read_point": m, "e_mean": float(np.nanmean(e)), "iso_mean": float(np.nanmean(iso[:, m])),
                                 "p_pos": tp.p_value, "p_neg": tn.p_value, "p_pos_over_iso": xp.p_value, "p_neg_over_iso": xn.p_value})
                v = reference_verdict(pos, neg)
                v["positive_windows"] = [pts[i] for i in v["positive_windows"]]
                v["negative_windows"] = [pts[i] for i in v["negative_windows"]]
                rr[key] = {"verdict": v, "rows": rows}
                self.log.info("%s ref %s, %s: %s (max e %+.3f)", comp.task, r, key, v["verdict"],
                              max((row["e_mean"] for row in rows), default=float("nan")))
            out[r] = rr
        return out

    def _lens_readout(self, comp: StagingComposition, conds: dict[str, dict[str, Any]], H: dict[str, np.ndarray],
                      token_sets: dict[str, np.ndarray], keep: dict[str, np.ndarray], test: Callable, excess: Callable,
                      arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        cfg, L = self.cfg, self.backend.n_layers
        first_ref = f"ref_{comp.references[0]}"
        sources = {"base": H["base"], "icl": H["icl"], first_ref: H[first_ref], **{k: v["H"] for k, v in conds.items()}}
        lens = {k: self._lens(h, token_sets) for k, h in sources.items()}
        for k, v in lens.items():
            for kk, arr in v.items():
                arrays[f"lens_{k.replace(':', '_')}_{kk}"] = arr
        out: dict[str, Any] = {}
        for it in comp.intermediates:
            ik, sel = f"int_{it.label}", keep[it.label]
            res: dict[str, Any] = {"n_items": int(sel.sum())}
            if sel.sum() < 2:
                empty = {"staged": False, "intermediate_windows": [], "final_windows": [], "intermediate_up_windows": []}
                res.update({key: {"verdict": dict(empty), "rows": []}
                            for key in ["icl", first_ref, *[k for k in conds if not conds[k].get("null")]]})
                out[it.label] = res
                continue
            for key in ["icl", first_ref, *[k for k in conds if not conds[k].get("null")]]:
                start = conds[key]["layer"] + 1 if key in conds else 1
                iso_keys = conds[key]["iso"] if key in conds else []
                pts = list(range(start, L + 1))
                gi = lens[key][f"lp_{ik}"][:, sel] - lens["base"][f"lp_{ik}"][:, sel]
                gf = lens[key]["lp_fin"][:, sel] - lens["base"]["lp_fin"][:, sel]
                gi_iso = [lens[k]["lp_" + ik][:, sel] - lens["base"]["lp_" + ik][:, sel] for k in iso_keys]
                a, b, f, rows = [], [], [], []
                for m in pts:
                    t_if, t_fi = test(gi[m], gf[m]), test(gf[m], gi[m])
                    t_up = test(gi[m], np.zeros_like(gi[m]))
                    up = t_up.p_value <= cfg.alpha and t_up.mean_diff > 0
                    p_iso = None
                    if gi_iso:
                        x = excess(gi[m], np.stack([g[m] for g in gi_iso]))
                        p_iso = x.p_value
                        up = up and x.p_value <= cfg.alpha and x.excess_mean > 0
                    a.append(t_if.p_value <= cfg.alpha and t_if.mean_diff > 0)
                    b.append(up)
                    f.append(t_fi.p_value <= cfg.alpha and t_fi.mean_diff > 0)
                    rows.append({"read_point": m, "gain_int": float(gi[m].mean()), "gain_fin": float(gf[m].mean()),
                                 "median_rank_int": float(np.median(lens[key][f"rank_{ik}"][m, sel])),
                                 "median_rank_fin": float(np.median(lens[key]["rank_fin"][m, sel])),
                                 "p_int_over_fin": t_if.p_value, "p_fin_over_int": t_fi.p_value, "p_int_up": t_up.p_value,
                                 "p_int_over_iso": p_iso})
                v = lens_verdict(a, b, f)
                v["intermediate_windows"] = [pts[i] for i in v["intermediate_windows"]]
                v["final_windows"] = [pts[i] for i in v["final_windows"]]
                # the power check's own reading: the intermediate above the controls in a window (D42)
                v["intermediate_up_windows"] = [pts[i] for i in windows(b)]
                res[key] = {"verdict": v, "rows": rows}
                self.log.info("%s lens %s, %s: %s", comp.task, it.label, key, v)
            out[it.label] = res
        return out

    def _removal(self, comp: StagingComposition, conds: dict[str, dict[str, Any]], zs: list[Prompt], base: ForwardResult,
                 base_H: np.ndarray, d_c: np.ndarray, d_ref: dict[str, np.ndarray]) -> dict[str, Any]:
        """Exploratory: the composed control's perturbation with its component along the step-one-specific
        direction removed at each read point, against the same removal along random directions (D33 mechanics)."""
        cfg, L, d = self.cfg, self.backend.n_layers, self.backend.hidden_size
        r = comp.references[0]
        out: dict[str, Any] = {}
        for key, cd in conds.items():
            if not key.endswith(":composed"):
                continue
            full = float(cd["effect"].mean())
            delta = cd["H"] - base_H
            rows = []
            for m in range(cd["layer"] + 1, L + 1):
                g_perp = residualise(d_ref[r][m].astype(np.float64), d_c[m].astype(np.float64))
                u = g_perp / np.maximum(np.linalg.norm(g_perp, axis=1, keepdims=True), 1e-30)
                along = np.sum(delta[m] * u, axis=1, keepdims=True) * u
                real = self.backend.run(zs, interventions=[*cd["ivs"], Intervention(m, (-along).astype(np.float32))]).logprob_per_token - base.logprob_per_token
                ctl = []
                for k in range(cfg.n_edit_controls):
                    R = unit_vectors(rng_for(cfg.seed, "staging_removal", comp.task, key, m, k), len(zs), d)
                    along_r = np.sum(delta[m] * R, axis=1, keepdims=True) * R
                    ctl.append(self.backend.run(zs, interventions=[*cd["ivs"], Intervention(m, (-along_r).astype(np.float32))]).logprob_per_token
                               - base.logprob_per_token)
                rows.append({"read_point": m, "retained": None if full == 0 else float(real.mean() / full),
                             "random_retained": None if full == 0 else float(np.mean(ctl) / full)})
            out[key] = {"reference": r, "full_effect": full, "rows": rows}
        return out

    def _verdicts(self, out: dict[str, Any]) -> dict[str, Any]:
        """The D42 readings of one composition on this checkpoint (the cross-checkpoint rule is applied by the
        summary): the model's own computation and each construction's control, with the power checks."""
        refs, first = out["references"], out["references"][0]
        first_int = next(iter(out["intermediates"]))
        lens = out["lens"][first_int]
        lens_power = bool(lens[f"ref_{first}"]["verdict"]["intermediate_up_windows"])
        natural = {"lens_staged": lens["icl"]["verdict"]["staged"], "lens_power": lens_power}
        if "masking" in out:
            natural["masking_staged"] = out["masking"]["per_intermediate"][first_int]["verdict"]["staged"]
        natural["staged"] = (natural["lens_staged"] and lens_power) or natural.get("masking_staged", False)
        res: dict[str, Any] = {"natural": natural, "controls": {}}
        for c in self.cfg.constructions:
            key = f"{c}:composed"
            if key not in lens:
                continue
            ref = out["references_perp"][first]
            fa = f"{c}:first_alone"
            p1 = fa in ref and ref[fa]["verdict"]["verdict"] in ("staircase", "component_only")
            p2 = fa in lens and bool(lens[fa]["verdict"]["intermediate_up_windows"]) and lens_power
            r1 = ref[key]["verdict"]["verdict"]
            r2 = lens[key]["verdict"]["staged"]
            staged = (p1 and r1 == "staircase") or (p2 and r2)
            one_step = p1 and p2 and r1 != "staircase" and not r2
            res["controls"][c] = {"reference_verdict": r1, "reference_power": p1, "lens_staged": r2, "lens_power": p2,
                                  "reading": "staged" if staged else ("one_step" if one_step else "undecided")}
        if len(refs) > 1:
            res["inner_references"] = {r: {k: v["verdict"]["verdict"] for k, v in out["references_perp"][r].items()
                                           if isinstance(v, dict) and "verdict" in v} for r in refs[1:]}
        return res


def run_staging(cfg: StagingConfig, fv_run: str | None, learned_run: str, run_id: str | None = None,
                config_path: str | None = None) -> Path:
    return Staging(cfg, fv_run, learned_run, run_id, config_path).run()
