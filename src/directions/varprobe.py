"""The conditional probe and the logit lens at the answer token (docs/DECISIONS.md D46).

The variable map (D45) found both sub-results of ``(a+b) - (c+d) = `` at the answer token from one block on (the
gather), and whole-vector swaps cannot tell "holds S1 and S2" from "holds the answer Y" there. This stage reads the
answer token's state at every read point in two ways: a conditional linear probe (does a linear readout recover S1
among the values consistent with Y, a digit among those consistent with its bracket's sum, against a permutation of
the target within the conditioning classes) and the logit lens (which digit the model's own final norm and
unembedding read off the state: the answer, a sub-result, an operand digit, against shuffled candidates).

    uv run directions varprobe --config configs/varprobe_qwen3_4b.yaml --run-id varprobe_qwen3_4b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import VarprobeConfig
from .determinism import compare_arrays
from .model import ModelBackend, environment_metadata, gpu_peak_memory, set_torch_determinism
from .probes import conditional_probe, match_rates
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .varmap import TEMPLATES, Expr, all_exprs, query_tokens, render

# the probe targets per task: name -> (target, conditioning variable); allowed classes follow from both
TARGETS = {
    "two_lane": ("Y", "S1|Y", "a|S1", "c|S2"),
    "one_lane": ("Y", "S1|Y", "a|S1"),
    "three_lane": ("Y", "S1|Y", "S2|Y", "S3|Y", "a|S1", "c|S2", "e|S3"),  # D51 (T|Y is S3|Y, U|Y is S1|Y relabelled)
    "max": ("Y", "S1|Y", "S2|Y", "C|Y", "L|Y", "a|S1", "c|S2"),  # D55: the comparison bit C and the smaller sum L
}
# the logit lens's candidates: the answer, the sub-results (three brackets: and the intermediates T, U), the digits
LENS_CANDIDATES = {"two_lane": ("Y", "S1", "S2", "a", "b", "c", "d"), "one_lane": ("Y", "S1", "S2", "a", "b", "c"),
                   "three_lane": ("Y", "S1", "S2", "S3", "T", "U", "a", "b", "c", "d", "e", "f"),
                   "max": ("Y", "S1", "S2", "L", "a", "b", "c", "d")}


def variables(exprs: list[Expr]) -> dict[str, np.ndarray]:
    """Y, S1, S2 and the operand digits of each expression (d = -1 on the one-lane task); on three brackets also
    S3, T, U and the digits e, f (D51)."""
    out = {
        "Y": np.array([e.Y for e in exprs]), "S1": np.array([e.S1 for e in exprs]), "S2": np.array([e.S2 for e in exprs]),
        "a": np.array([e.a for e in exprs]), "b": np.array([e.b for e in exprs]), "c": np.array([e.c for e in exprs]),
        "d": np.array([-1 if e.d is None else e.d for e in exprs]),
    }
    if exprs and exprs[0].op == "max":  # D55: the comparison bit and the smaller sum
        out.update({"C": np.array([e.C for e in exprs]), "L": np.array([e.L for e in exprs])})
    if exprs and exprs[0].e is not None:
        out.update({"S3": np.array([e.S3 for e in exprs]), "T": np.array([e.T for e in exprs]),
                    "U": np.array([e.U for e in exprs]), "e": np.array([e.e for e in exprs]), "f": np.array([e.f for e in exprs])})
    return out


def target_spec(name: str, task: str, v: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target, the conditioning variable and the allowed classes (N, C) of one probe (D46)."""
    n = len(v["Y"])
    if name == "Y":
        return v["Y"], np.zeros(n, dtype=np.int64), np.ones((n, 10), dtype=bool)
    if task == "max" and name in ("S1|Y", "S2|Y"):  # D55: each sum is at most the answer, Y = max(S1, S2)
        k = np.arange(10)[None, :]
        return v[name[:2]], v["Y"], k <= v["Y"][:, None]
    if name == "C|Y":  # D55: which sum is the larger, given the answer
        return v["C"], v["Y"], np.ones((n, 2), dtype=bool)
    if name == "L|Y":  # D55: the smaller sum, below the answer
        k = np.arange(10)[None, :]
        return v["L"], v["Y"], k < v["Y"][:, None]
    if name == "S1|Y":
        hi = 9 if task == "one_lane" else 18  # S1 - Y ranges over 0..18 (two, three brackets) or 0..9 (one-lane, S2 = c)
        k = np.arange(19)[None, :]
        return v["S1"], v["Y"], (k >= v["Y"][:, None]) & (k <= np.minimum(18, v["Y"][:, None] + hi))
    if name in ("S2|Y", "S3|Y"):  # three brackets: S2 + S3 = S1 - Y <= 18 - Y
        k = np.arange(19)[None, :]
        return v[name[:2]], v["Y"], k <= 18 - v["Y"][:, None]
    if name in ("a|S1", "c|S2", "e|S3"):
        digit, total = {"a|S1": ("a", "S1"), "c|S2": ("c", "S2"), "e|S3": ("e", "S3")}[name]
        k = np.arange(10)[None, :]
        s = v[total][:, None]
        return v[digit], v[total], (k >= np.maximum(0, s - 9)) & (k <= np.minimum(9, s))
    raise ValueError(f"unknown probe target {name!r}")


def lens_eligible(name: str, v: dict[str, np.ndarray]) -> np.ndarray:
    """The examples on which a lens candidate is a digit distinct from the answer (all of them for Y itself)."""
    if name == "Y":
        return np.ones(len(v["Y"]), dtype=bool)
    return (v[name] >= 0) & (v[name] <= 9) & (v[name] != v["Y"])


def sample_probe_set(seed: int, task: str, n_examples: int, n_shots: int, n_demo_pool: int
                     ) -> tuple[list[Expr], list[list[Expr]], dict[str, int]]:
    """The D46 draw: a demonstration pool, then up to ``n_examples`` expressions in a seeded order, each with its
    demonstrations."""
    exprs = all_exprs(task)
    order = rng_for(seed, "varprobe", task, "split").permutation(len(exprs))
    pool = [exprs[i] for i in order[:n_demo_pool]]
    chosen = [exprs[i] for i in order[n_demo_pool:]][:n_examples]
    demos = []
    for i in range(len(chosen)):
        r = rng_for(seed, "varprobe", task, "demos", i)
        demos.append([pool[j] for j in r.choice(len(pool), size=n_shots, replace=False)])
    return chosen, demos, {"n_expressions": len(exprs), "n_examples": len(chosen)}


class _Rows:
    """``features[m]`` = the kept examples' states at read point m, without copying the whole array."""

    def __init__(self, states: np.ndarray, keep: np.ndarray) -> None:
        self.states, self.keep = states, keep

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, m: int) -> np.ndarray:
        return self.states[m][self.keep]


class VarProbe:
    def __init__(self, cfg: VarprobeConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_varprobe_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core" / "tasks").mkdir(parents=True)
        (self.root / "figures").mkdir()
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        self.backend = ModelBackend(self.cfg.model, run_seed=self.cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        self.digit_ids = np.array([self._single_token(str(k)) for k in range(10)], dtype=np.int64)
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        for task in self.cfg.expressions:
            t0 = time.time()
            res = self._task(task)
            if "skipped" in res:
                summary["skipped"][task] = res["skipped"]
            else:
                summary["tasks"][task] = res["summary"]
            self.metadata.setdefault("timings_seconds", {})[task] = round(time.time() - t0, 1)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()  # the run's peak: the headroom a change can use
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    def _single_token(self, text: str) -> int:
        enc = self.backend.tokenizer.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{text!r} is not a single token on this tokenizer: {enc}")
        return enc[0]

    def _sample(self, task: str) -> tuple[list[Expr], list[list[Expr]]]:
        cfg = self.cfg
        chosen, demos, info = sample_probe_set(cfg.seed, task, cfg.n_examples, cfg.n_shots, cfg.n_demo_pool)
        self.metadata.setdefault("sampling", {})[task] = info
        return chosen, demos

    def _task(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        exprs, demos = self._sample(task)
        prompts = [render(demos[i], e) for i, e in enumerate(exprs)]
        pos, _ids, labels = zip(*(query_tokens(backend, p, e) for p, e in zip(prompts, exprs)))
        if len({tuple(lab) for lab in labels}) != 1:
            raise ValueError(f"{task}: the query tokenises differently across expressions")
        labels = list(labels[0])
        a_pos = np.array([pp[labels.index("a")] for pp in pos], dtype=np.int64)

        res = backend.run(prompts, capture=True, token_ids=self.digit_ids)
        states = res.residuals  # (L+1, N, d) float32 at the answer token
        L = states.shape[0] - 1
        v = variables(exprs)
        correct = res.query_argmax == self.digit_ids[v["Y"]]
        accuracy = float(correct.mean())
        log.info("%s: few-shot accuracy %.3f on %d expressions; query tokens %s", task, accuracy, len(exprs), labels)
        if accuracy < cfg.min_accuracy:
            self.rejections.add("gate", task, "few-shot accuracy below the minimum", {"accuracy": accuracy})
            return {"skipped": {"reason": "few-shot accuracy below the minimum", "accuracy": accuracy}}
        keep = correct if cfg.gate else np.ones(len(exprs), dtype=bool)
        vk = {k: x[keep] for k, x in v.items()}
        n = int(keep.sum())

        if cfg.determinism_check and "capture" not in self.metadata.get("determinism_check", {}):
            bs = backend.cfg.batch_size
            again = backend.run(prompts[:bs], capture=True, token_ids=self.digit_ids)
            cmp = compare_arrays({"r": states[:, :bs]}, {"r": again.residuals})
            self.metadata.setdefault("determinism_check", {})["capture"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("determinism check failed: a repeated capture batch differs")

        lens = self._lens(task, states, keep, vk, res.query_argmax)
        log.info("%s: lens at read point %d agrees with the model's answer on %.3f; lens reads Y above its null from "
                 "read point %s", task, L, lens["final_agreement"], _first(lens["candidates"]["Y"]["carried"]))

        feats = _Rows(states, keep)
        probes: dict[str, dict[str, Any]] = {}
        for name in TARGETS[task]:
            t0 = time.time()
            y, cond, allowed = target_spec(name, task, vk)
            r = conditional_probe(feats, y, cond, allowed, n_folds=cfg.n_folds, lambdas=cfg.lambdas, n_perm=cfg.n_perm,
                                  alpha=cfg.alpha, seed=derive_seed(cfg.seed, "varprobe", task, name), device=backend.device)
            probes[name] = r
            log.info("%s/%s: %d examples, %.0fs; carried at read points %s (max excess %.3f)", task, name, n,
                     time.time() - t0, np.flatnonzero(r["carried"]).tolist(), float(r["excess"].max()))
        if cfg.determinism_check and "probe" not in self.metadata.get("determinism_check", {}):
            name = TARGETS[task][1]
            y, cond, allowed = target_spec(name, task, vk)
            again = conditional_probe(feats, y, cond, allowed, n_folds=cfg.n_folds, lambdas=cfg.lambdas, n_perm=cfg.n_perm,
                                      alpha=cfg.alpha, seed=derive_seed(cfg.seed, "varprobe", task, name),
                                      device=backend.device, read_points=[L])
            cmp = compare_arrays({"acc": probes[name]["null"][L:], "obs": probes[name]["accuracy"][L:]},
                                 {"acc": again["null"], "obs": again["accuracy"]})
            self.metadata.setdefault("determinism_check", {})["probe"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("determinism check failed: a repeated probe differs")

        control = None
        if cfg.positive_control:
            kp = np.flatnonzero(keep)
            at_a = backend.capture_sites([prompts[i] for i in kp], a_pos[kp][:, None])  # (n, L+1, 1, d)
            x = at_a[:, L // 2, 0].float().cpu().numpy()
            del at_a
            y, cond, allowed = target_spec("a|S1", task, vk)
            control = conditional_probe([x], y, cond, allowed, n_folds=cfg.n_folds, lambdas=cfg.lambdas, n_perm=cfg.n_perm,
                                        alpha=cfg.alpha, seed=derive_seed(cfg.seed, "varprobe", task, "control"),
                                        device=backend.device)
            log.info("%s: positive control (a given S1 at the a token, read point %d): accuracy %.3f against %.3f, carried %s",
                     task, L // 2, float(control["accuracy"][0]), float(control["null_mean"][0]), bool(control["carried"][0]))

        result = {
            "task": task, "labels": labels, "read_points": list(range(L + 1)), "n_expressions": len(exprs), "n_kept": n,
            "accuracy": accuracy, "targets": list(TARGETS[task]),
            "probes": {k: _jsonable({kk: vv for kk, vv in r.items() if kk != "null"}) for k, r in probes.items()},
            "lens": _jsonable(lens),
            "positive_control": None if control is None else _jsonable(
                {"read_point": L // 2, **{kk: vv for kk, vv in control.items() if kk != "null"}}),
            "settings": {"n_folds": cfg.n_folds, "lambdas": list(cfg.lambdas), "n_perm": cfg.n_perm, "alpha": cfg.alpha},
        }
        td = self.root / "core" / "tasks" / task
        write_json(td / "varprobe.json", result)
        arrays = {f"null__{k.replace('|', '_given_')}": r["null"].astype(np.float32) for k, r in probes.items()}
        arrays.update({"lens_digit": lens.pop("_digit").astype(np.int8), "kept": keep,
                       **{f"var__{k}": x.astype(np.int8) for k, x in v.items()}})
        np.savez_compressed(td / "varprobe_arrays.npz", **arrays)
        if cfg.figures:
            try:
                mc = cfg.model
                plot_task(result, self.root / "figures" / f"{task}.png",
                          mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}", cfg.marks)
            except Exception as exc:  # a figure never fails a run
                log.warning("%s: figure failed: %s", task, exc)
        yacc = probes["Y"]["accuracy"]
        formed = int(np.flatnonzero(yacc >= 0.9 * yacc[-1])[0]) if yacc[-1] > 0 else None
        summ = {"accuracy": accuracy, "n_kept": n, "y_probe_formed_at": formed,
                "carried_from": {k: _first(r["carried"]) for k, r in probes.items()},
                "lens_from": {k: _first(np.asarray(c["carried"])) for k, c in result["lens"]["candidates"].items()
                              if c.get("n", 0)}}
        return {"summary": summ}

    def _lens(self, task: str, states: np.ndarray, keep: np.ndarray, vk: dict[str, np.ndarray],
              model_argmax: np.ndarray) -> dict[str, Any]:
        """The logit lens at the answer token at every read point, over the kept examples."""
        cfg, backend = self.cfg, self.backend
        M = states.shape[0]
        ids = torch.as_tensor(self.digit_ids, device=backend.device)
        digit = np.zeros((M, int(keep.sum())), dtype=np.int64)
        top_is_digit = np.zeros(M)
        logp_y = np.zeros(M)
        y = torch.as_tensor(vk["Y"], device=backend.device)
        for m in range(M):
            x = states[m][keep]
            dg, td, lp = [], 0, 0.0
            for s in range(0, len(x), 256):
                logits = backend.logits_from_residual(x[s : s + 256])
                dg.append(logits[:, ids].argmax(dim=1).cpu().numpy())
                td += int(torch.isin(logits.argmax(dim=1), ids).sum())
                lsm = torch.log_softmax(logits.to(torch.float64), dim=1)
                lp += float(lsm[torch.arange(len(lsm), device=lsm.device), ids[y[s : s + 256]]].sum())
            digit[m] = np.concatenate(dg)
            top_is_digit[m] = td / len(x)
            logp_y[m] = lp / len(x)
        # the lens at the last read point against the model's own answer (every expression, not only the kept)
        final = backend.logits_from_residual(states[-1])[:, ids].argmax(dim=1).cpu().numpy()
        model_digit = np.full(len(model_argmax), -1)
        for k, t in enumerate(self.digit_ids):
            model_digit[model_argmax == t] = k
        agree = float((final == model_digit)[model_digit >= 0].mean()) if (model_digit >= 0).any() else float("nan")
        cands = {}
        for name in LENS_CANDIDATES[task]:
            r = match_rates(digit, vk[name], lens_eligible(name, vk), n_perm=cfg.n_perm, alpha=cfg.alpha,
                            seed=derive_seed(cfg.seed, "varprobe", task, "lens", name))
            cands[name] = r
        return {"top_is_digit": top_is_digit, "logp_Y": logp_y, "final_agreement": agree, "candidates": cands,
                "_digit": digit}


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(np.asarray(mask))
    return int(idx[0]) if len(idx) else None


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

SERIES = "#2a78d6"  # one series per panel: the observed rate
NULL_FILL, NULL_LINE = "#e4e3df", "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
TITLES = {"Y": "Y (the answer)", "S1|Y": "S1 given Y", "S2|Y": "S2 given Y", "S3|Y": "S3 given Y", "a|S1": "a given S1",
          "c|S2": "c given S2", "e|S3": "e given S3", "C|Y": "C (S1 > S2) given Y", "L|Y": "the smaller sum given Y"}


def plot_task(result: dict[str, Any], path: Path, model_name: str, marks: dict[str, int] | None = None) -> None:
    """Small multiples over read points. Row 1: the conditional probe's accuracy per target; rows 2-3: how often
    the logit lens reads each candidate. Each panel: the observed rate (blue; filled where the target is carried at
    the family-wise level), the null's mean (grey dashes) and the family-wise band (grey: mean + threshold x s.d.)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rps = np.asarray(result["read_points"])
    titles = dict(TITLES, **({"S1|Y": "S1 given Y (= c given Y)"} if result["task"] == "one_lane" else {}))
    rows = [[(titles[t], result["probes"][t]) for t in result["targets"]],
            [(f"lens reads {c}", result["lens"]["candidates"][c]) for c in ("Y", "S1", "S2", "S3", "T", "U")
             if c in result["lens"]["candidates"]],
            [(f"lens reads {c}", result["lens"]["candidates"][c]) for c in ("a", "b", "c", "d", "e", "f")
             if c in result["lens"]["candidates"]]]
    ncol = max(len(r) for r in rows)
    fig, axes = plt.subplots(3, ncol, figsize=(2.3 * ncol + 0.6, 6.6), sharex=True)
    ylabels = ["probe accuracy\n(cross-validated)", "share of examples", "share of examples"]
    for i, row in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(row) or not row[j][1].get("n", 1):
                ax.axis("off")
                continue
            title, r = row[j]
            obs = np.asarray(r.get("accuracy", r.get("rate")), dtype=float)
            mean, sd = np.asarray(r["null_mean"], dtype=float), np.asarray(r["null_sd"], dtype=float)
            ax.fill_between(rps, mean, np.minimum(1, mean + r["threshold"] * sd), color=NULL_FILL, lw=0)
            ax.plot(rps, mean, color=NULL_LINE, lw=0.9, ls=(0, (3, 2)))
            ax.plot(rps, obs, color=SERIES, lw=1.4)
            c = np.asarray(r["carried"], dtype=bool)
            ax.plot(rps[c], obs[c], ls="", marker="o", ms=3.2, color=SERIES)
            for label, rp in (marks or {}).items():
                ax.axvline(rp, color=INK2, lw=0.7, ls=(0, (1, 2)))
                if i == 0 and j == 0:
                    ax.text(rp, 1.02, f" {label}", fontsize=6.5, color=INK2, ha="left", va="bottom",
                            transform=ax.get_xaxis_transform())
            ax.set_ylim(0, 1.0)
            ax.set_title(title + ("" if "n" not in r else f"  (n={r['n']})"), fontsize=7.5, loc="left", color=INK)
            ax.tick_params(labelsize=6.5, length=2, colors=INK2)
            ax.grid(axis="y", color="#ecebe7", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#c3c2b7")
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize=7, color=INK2)
            if i == 2 or (i == 1 and j >= len(rows[2])):
                ax.set_xlabel("read point (answer token)", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=SERIES, lw=1.4, marker="o", ms=3.2),
               plt.Line2D([], [], color=NULL_LINE, lw=0.9, ls=(0, (3, 2))), plt.Rectangle((0, 0), 1, 1, color=NULL_FILL)]
    fig.legend(handles, ["observed (dots: above the family-wise null)", "null mean", "family-wise null band"],
               loc="lower left", bbox_to_anchor=(0.01, -0.035), ncol=3, fontsize=7, frameon=False)
    task = TEMPLATES[result["task"]]
    fig.suptitle(f"Answer token of {task.strip()}  ({model_name}): conditional probe and logit lens; "
                 f"{result['n_kept']} of {result['n_expressions']} answered", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def run_varprobe(cfg: VarprobeConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return VarProbe(cfg, run_id, config_path).run()
