"""The serial injection test (docs/DECISIONS.md D41): can a composition be assembled from its components' controls?

For a composed task f∘g, the first component's control u_g is injected at its own selected layer and strength and
the second's, u_f, at a grid of deeper layers, on the composed task's held-out zero-shot prompts; what is read is
the composed target's log-probability per token against the unsteered run. The null keeps u_g and replaces u_f by
random unit directions at the same layer and norm (paired excess test, D18). Beside the serial curve: each control
alone (u_f at every layer of the grid), the two added at the first layer (superposition) and the composed task's
own control (the ceiling). A serial effect above the null only past the first control's hand-over (D37, read from
the landmark run afterwards) says the second stage reads a finished intermediate; above it everywhere, the
controls add; nowhere, the composition is not assembled from its parts as controls.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import SerialComposition, SerialConfig
from .model import Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .mixing import _own_prompts, _unit
from .prompts import Prompt
from .runinfo import git_info, read_json, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .stats import paired_bootstrap_test, paired_excess_test
from .trajectories import _items, _load_run, alpha_from_calibration


class _Skip(Exception):
    pass


class Serial:
    def __init__(self, cfg: SerialConfig, fv_run: str | None, learned_run: str, run_id: str | None, config_path: str | None) -> None:
        self.cfg = cfg
        self.fv_root = Path(fv_run) if fv_run else None
        self.learned_root = Path(learned_run)
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_serial_{self.learned_root.name}"
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
            for construction in self.cfg.constructions:
                try:
                    out = self._composition(comp, construction)
                    summary["compositions"].append(out)
                except _Skip as exc:
                    summary["skipped"].append({"task": comp.task, "steps": comp.steps, "construction": construction, "reason": str(exc)})
                    self.log.info("%s = %s∘%s [%s]: skipped (%s)", comp.task, comp.steps[1], comp.steps[0], construction, exc)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    # ------------------------------------------------------------------ #

    def _control(self, label: str, construction: str) -> dict[str, Any]:
        """A task's control under a construction: its unit direction(s), selected layer and strength; the learned
        vector per candidate layer with the calibration's strength there."""
        if construction == "fv" and self.fv_root is None:
            raise _Skip("no head-mean run given")
        root = self.fv_root if construction == "fv" else self.learned_root
        assert root is not None
        td = root / "core" / "tasks" / label
        q = read_json(td / "qualification.json") if (td / "qualification.json").exists() else None
        if not q or not q.get("selection") or not (td / "directions.npz").exists():
            raise _Skip(f"{label} not qualified under {construction}")
        arr = dict(np.load(td / "directions.npz"))
        layer = int(q["selection"]["layer"])
        cal = read_json(td / "calibration.json") if (td / "calibration.json").exists() else None
        if construction == "fv":
            if "fv_direction" not in arr:
                raise _Skip(f"{label}: no head-mean vector")
            u = _unit(arr["fv_direction"])
            strength = alpha_from_calibration(cal, layer) or {"alpha": float(np.linalg.norm(arr["fv"])), "source": "natural_norm"}
            return {"label": label, "layer": layer, "alpha": float(strength["alpha"]), "alpha_source": strength.get("source"),
                    "by_layer": None, "unit": u, "task_dir": td}
        layers = [int(l) for l in arr["layers"]]
        by_layer: dict[int, tuple[np.ndarray, float, str]] = {}
        for i, l in enumerate(layers):
            s = alpha_from_calibration(cal, l) or {"alpha": float(arr["learned_radii"][i]), "source": "radius"}
            by_layer[l] = (_unit(arr["learned"][i]), float(s["alpha"]), str(s.get("source")))
        u, alpha, src = by_layer[layer]
        return {"label": label, "layer": layer, "alpha": alpha, "alpha_source": src, "by_layer": by_layer, "unit": u, "task_dir": td}

    def _second(self, ctl: dict[str, Any], layer: int) -> tuple[np.ndarray, float, str] | None:
        """The second control's direction and strength at ``layer``: the head mean is one vector at its selected
        strength everywhere; the learned vector exists at its candidate layers only."""
        if ctl["by_layer"] is None:
            return ctl["unit"], ctl["alpha"], "selected_strength"
        if layer in ctl["by_layer"]:
            return ctl["by_layer"][layer]
        return None

    def _effect(self, prompts: list[Prompt], interventions: list[Intervention], baseline: np.ndarray) -> np.ndarray:
        return self.backend.run(prompts, interventions=interventions).logprob_per_token - baseline

    def _composition(self, comp: SerialComposition, construction: str) -> dict[str, Any]:
        g, f = comp.steps
        cg, cf = self._control(g, construction), self._control(f, construction)
        try:
            cc: dict[str, Any] | None = self._control(comp.task, construction)
        except _Skip as exc:
            cc = None
            self.log.info("%s [%s]: the composed control is %s; its ceiling is not read", comp.task, construction, exc)
        td = self.learned_root / "core" / "tasks" / comp.task
        if not (td / "splits.json").exists():
            raise _Skip(f"{comp.task} absent from the learned run")
        items = _items(read_json(td / "splits.json")["evaluation"])[: self.cfg.n_prompts]
        prompts = _own_prompts(items, self.run_cfg.prompt)
        L = self.backend.n_layers
        l_g = int(cg["layer"])
        inject_g = Intervention(l_g, (cg["alpha"] * cg["unit"]).astype(np.float32))
        base = self.backend.run(prompts).logprob_per_token
        eff = lambda ivs: self._effect(prompts, ivs, base)  # noqa: E731
        brng = rng_for(self.cfg.seed, "serial_bootstrap", comp.task, construction)
        out: dict[str, Any] = {"task": comp.task, "steps": comp.steps, "construction": construction, "n_items": len(items),
                               "first": {"label": g, "layer": l_g, "alpha": cg["alpha"], "alpha_source": cg["alpha_source"]},
                               "second": {"label": f, "selected_layer": int(cf["layer"]), "selected_alpha": cf["alpha"]},
                               "n_layers": L, "baseline_logprob_per_token": float(base.mean())}
        g_alone = eff([inject_g])
        out["first_alone"] = {"effect": float(g_alone.mean()), "test": paired_bootstrap_test(g_alone + base, base, brng, n_boot=self.cfg.n_boot).__dict__}
        if cc is not None:
            own = eff([Intervention(int(cc["layer"]), (cc["alpha"] * cc["unit"]).astype(np.float32))])
            out["composed_control"] = {"layer": int(cc["layer"]), "alpha": cc["alpha"], "effect": float(own.mean()),
                                       "test": paired_bootstrap_test(own + base, base, brng, n_boot=self.cfg.n_boot).__dict__}
        # superposition: both controls added at the first layer (u_f at its selected strength)
        v_sum = (cg["alpha"] * cg["unit"] + cf["alpha"] * cf["unit"]).astype(np.float32)
        sup = eff([Intervention(l_g, v_sum)])
        out["superposition_at_first_layer"] = {"effect": float(sup.mean()), "norm": float(np.linalg.norm(v_sum)),
                                                "test": paired_bootstrap_test(sup + base, base, brng, n_boot=self.cfg.n_boot).__dict__,
                                                "angle_deg": float(np.degrees(np.arccos(np.clip(float(cg["unit"] @ cf["unit"]), -1, 1))))}
        # the grid of second layers: the head mean at the configured fractions of the stack (at or past the first
        # layer), the learned vector at its candidate layers at or past the first layer
        if cf["by_layer"] is None:
            grid = sorted({max(l_g, min(L, int(round(fr * L)))) for fr in self.cfg.second_layer_fractions})
        else:
            grid = sorted(l for l in cf["by_layer"] if l >= l_g)
        rng = rng_for(self.cfg.seed, "serial_null", comp.task, construction)
        d = self.backend.hidden_size
        rows = []
        for l in grid:
            second = self._second(cf, l)
            if second is None:
                continue
            u_f, alpha_f, src = second
            inject_f = Intervention(l, (alpha_f * u_f).astype(np.float32))
            serial = eff([inject_g, inject_f])
            if self.cfg.determinism_check and "determinism_check" not in self.metadata:
                again = eff([inject_g, inject_f])
                identical = bool(np.array_equal(again, serial))
                self.metadata["determinism_check"] = {"task": comp.task, "construction": construction, "layer": l, "identical": identical}
                if not identical:
                    raise RuntimeError("determinism check failed: a repeated serial pass differs")
            f_alone = eff([inject_f])
            nulls = np.stack([eff([inject_g, Intervention(l, (alpha_f * _unit(rng.normal(size=d))).astype(np.float32))])
                              for _ in range(self.cfg.n_null)])
            ex = paired_excess_test(serial, nulls, brng, n_boot=self.cfg.n_boot)
            row = {"layer": l, "fraction": l / L, "alpha": alpha_f, "alpha_source": src,
                   "serial": float(serial.mean()), "second_alone": float(f_alone.mean()), "null_mean": float(nulls.mean()),
                   "serial_over_null": ex.__dict__,
                   "serial_over_first_alone": paired_bootstrap_test(serial, g_alone, brng, n_boot=self.cfg.n_boot).__dict__,
                   "serial_test": paired_bootstrap_test(serial + base, base, brng, n_boot=self.cfg.n_boot).__dict__}
            rows.append(row)
            self.log.info("%s = %s∘%s [%s] u_%s@%d + u_%s@%d: serial %+.3f (null %+.3f, p %.3f; first alone %+.3f, p %.3f), second alone %+.3f",
                          comp.task, f, g, construction, g, l_g, f, l, row["serial"], row["null_mean"], ex.p_value,
                          out["first_alone"]["effect"], row["serial_over_first_alone"]["p_value"], row["second_alone"])
        out["grid"] = rows
        bright = [r["layer"] for r in rows if r["serial_over_null"]["p_value"] <= 0.05 and r["serial_over_first_alone"]["p_value"] <= 0.05
                  and r["serial_over_null"]["excess_mean"] > 0]
        out["bright_layers"] = bright
        out["reading"] = "nowhere" if not bright else ("everywhere" if len(bright) == len(rows) else "partial")
        self.log.info("%s [%s]: serial effect above the null and above the first control alone at layers %s (%s); superposition %+.3f; composed control %s",
                      comp.task, construction, bright, out["reading"], out["superposition_at_first_layer"]["effect"],
                      "n/a" if cc is None else f"{out['composed_control']['effect']:+.3f}")
        return out


def run_serial(cfg: SerialConfig, fv_run: str | None, learned_run: str, run_id: str | None = None,
               config_path: str | None = None) -> Path:
    return Serial(cfg, fv_run, learned_run, run_id, config_path).run()
