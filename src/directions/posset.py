"""The position-set check (docs/DECISIONS.md D52): where the rest of the answer lives after the gather.

D45's whole-state swap between a base expression's run and a source's run, with a *set* of query tokens swapped
together at one read point, each token from the source run's state at the same token and read point. Read against
the answer token alone: the gain of a set is the share of bases whose answer it moves to the source's (or to a
bracket's prediction) beyond what the answer token does by itself, paired by base. A joint subspace across positions
(DAS at several tokens at once) can only reach what the tokens it spans carry, so the non-digit set's gain decides
whether that DAS is built (D52's rule).

    uv run directions posset --config configs/posset_qwen3_4b.yaml --run-id posset_qwen3_4b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import PossetConfig
from .determinism import compare_arrays
from .model import ModelBackend, SitePatch, environment_metadata, gpu_peak_memory, set_torch_determinism
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed
from .varmap import DIGITS, TEMPLATES, _resample_counts, bootstrap_rate, prediction, query_tokens, render, sample_bases

# the sets, in order; "answer" is the reference every gain is taken against, "query" the check (the source's run)
SETS = ("answer", "answer_eq", "non_digit", "answer_digits", "query")
SET_LABELS = {"answer": "answer token", "answer_eq": "answer + =", "non_digit": "non-digit tokens",
              "answer_digits": "answer + digits", "query": "every query token"}
# the source whose prediction is the source's own answer, per task
COMBINED = {"two_lane": "both", "three_lane": "all"}


def set_members(labels: list[str]) -> dict[str, list[int]]:
    """The query-token indices of each set, from the token labels of :func:`directions.varmap.query_tokens` (digits
    named by their slot, the answer position ``␣``)."""
    ans = len(labels) - 1
    if labels[ans] != "␣":
        raise ValueError(f"the last query token is not the answer position: {labels}")
    eq = [j for j, lab in enumerate(labels) if lab.strip() == "="]
    if len(eq) != 1:
        raise ValueError(f"expected one '=' token among {labels}")
    digits = [j for j, lab in enumerate(labels) if lab in DIGITS]
    return {"answer": [ans], "answer_eq": sorted({eq[0], ans}),
            "non_digit": [j for j in range(len(labels)) if j not in digits],
            "answer_digits": sorted(digits + [ans]), "query": list(range(len(labels)))}


def paired_gain(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, np.ndarray]:
    """The mean of ``x - ref`` over the first axis (bases, paired) with its one-sided percentile lower bound at
    ``alpha``: ``x`` and ``ref`` are ``(n, M)`` indicators of the same bases."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    n = d.shape[0]
    boots = _resample_counts(n, rng, n_boot) @ d.reshape(n, -1) / n
    return {"mean": d.mean(axis=0), "lo": np.quantile(boots, alpha, axis=0).reshape(d.shape[1:])}


def decide(gains: dict[str, dict[str, dict[str, np.ndarray]]], read_points: list[int], min_gain: float,
           jointly_enough: float, max_sites: int, control: int | None) -> dict[str, Any]:
    """D52's rule. ``gains[source][set]`` holds the paired gain over the answer token per read point (``mean``, and
    ``lo`` at the Bonferroni-corrected level). Passes where the non-digit set's gain is at least ``min_gain`` with
    its lower bound above 0, for some source and read point."""
    passing = []
    for src, by_set in gains.items():
        g = by_set["non_digit"]
        for i, m in enumerate(read_points):
            if g["mean"][i] >= min_gain and g["lo"][i] > 0:
                passing.append((src, m, float(g["mean"][i]), float(by_set["answer_eq"]["mean"][i])))
    if not passing:
        return {"passes": False, "passing": []}
    eq_enough = all(eq >= jointly_enough * nd for _, _, nd, eq in passing)
    best: dict[int, float] = {}
    for _, m, nd, _ in passing:
        best[m] = max(best.get(m, -1.0), nd)
    sites = sorted(sorted(best, key=lambda m: (-best[m], m))[:max_sites])
    return {"passes": True,
            "passing": [{"source": s, "read_point": m, "gain_non_digit": round(nd, 4), "gain_answer_eq": round(eq, 4)}
                        for s, m, nd, eq in passing],
            "positions": "answer_eq" if eq_enough else "non_digit",
            "sites": sorted(set(sites) | ({control} if control is not None else set()))}


class PositionSets:
    def __init__(self, cfg: PossetConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_posset_{slug}_seed{cfg.seed}"
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
        self.digit_ids = self._digit_ids()
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        for task in self.cfg.expressions:
            t0 = time.time()
            res = self._task(task)
            summary["skipped" if "skipped" in res else "tasks"][task] = res.get("skipped", res.get("summary"))
            self.metadata.setdefault("timings_seconds", {})[task] = round(time.time() - t0, 1)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()  # the run's peak: the headroom a change can use
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    def _digit_ids(self) -> np.ndarray:
        ids = []
        for k in range(10):
            enc = self.backend.tokenizer.encode(str(k), add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"the digit {k} is not a single token on this tokenizer: {enc}")
            ids.append(enc[0])
        return np.array(ids, dtype=np.int64)

    def _task(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        bases, sources, demos, info = sample_bases(cfg.seed, task, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata.setdefault("sampling", {})[task] = info
        kinds = list(cfg.sources)
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        pos0, _ids0, labels = query_tokens(backend, base_prompts[0], bases[0])
        P = len(pos0)
        members = set_members(labels)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        keep = np.ones(len(bases), dtype=bool)
        src_prompts: dict[str, list[Prompt | None]] = {k: [] for k in kinds}
        for i, base in enumerate(bases):
            pos, ids, _ = query_tokens(backend, base_prompts[i], base)
            if len(pos) != P:
                keep[i] = False
                self.rejections.add("align", task, "query token count differs", {"base": base.text(), "n": len(pos), "expected": P})
                for k in kinds:
                    src_prompts[k].append(None)
                continue
            positions[i] = pos
            for k in kinds:
                src = sources[i][k]
                if src is None:
                    keep[i] = False
                    self.rejections.add("source", task, f"no valid {k} source", {"base": base.text()})
                    src_prompts[k].append(None)
                    continue
                sp = render(demos[i], src)
                spos, sids, _ = query_tokens(backend, sp, src)
                if not (spos == pos and all(x == y for x, y, lab in zip(ids, sids, labels) if lab not in DIGITS)):
                    keep[i] = False
                    self.rejections.add("align", task, f"{k} source misaligned", {"base": base.text(), "source": src.text()})
                src_prompts[k].append(sp)
        log.info("%s: %d bases, %d query tokens %s, %d aligned with every source; sets %s", task, len(bases), P, labels,
                 int(keep.sum()), members)

        tid = self.digit_ids
        base_run = backend.run(base_prompts, token_ids=tid)
        base_ok = base_run.query_argmax == tid[[b.Y for b in bases]]
        accuracy = float(base_ok.mean())
        log.info("%s: few-shot accuracy %.3f on %d bases", task, accuracy, len(bases))
        if accuracy < cfg.min_accuracy:
            self.rejections.add("gate", task, "few-shot accuracy below the minimum", {"accuracy": accuracy, "min": cfg.min_accuracy})
            return {"skipped": {"reason": "few-shot accuracy below the minimum", "accuracy": accuracy}}
        src_ok = {}
        for k in kinds:
            idx = [i for i in range(len(bases)) if src_prompts[k][i] is not None]
            ok = np.zeros(len(bases), dtype=bool)
            ok[idx] = backend.run([src_prompts[k][i] for i in idx], token_ids=tid).query_argmax == tid[[sources[i][k].Y for i in idx]]
            src_ok[k] = ok
        base_mask = keep & (base_ok if cfg.gate else True)
        type_rows = {k: np.where(base_mask & (src_ok[k] if cfg.gate else keep))[0] for k in kinds}
        log.info("%s: bases kept per source %s (of %d)", task, {k: len(v) for k, v in type_rows.items()}, len(bases))

        L = backend.n_layers
        read_points = [m for m in cfg.read_points if 0 <= m <= L]
        if len(read_points) != len(cfg.read_points):
            raise ValueError(f"read points {cfg.read_points} outside 0..{L}")
        M = len(read_points)
        rp_t = torch.as_tensor(read_points)

        # check (D23): swapping a run's own states back at every query token reproduces it bit-exactly
        if cfg.identity_check:
            rows = np.where(base_mask)[0][: backend.cfg.batch_size]
            brs = [base_prompts[i] for i in rows]
            own = backend.capture_sites(brs, positions[rows])
            mid = read_points[M // 2]
            ref = backend.run(brs, token_ids=tid)
            got = backend.run(brs, token_ids=tid, site_patch=SitePatch(np.full(len(rows), mid), positions[rows], own[:, mid]))
            same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
            self.metadata.setdefault("identity_check", {})[f"{task}/read_point_{mid}/query"] = same
            if not same:
                raise RuntimeError(f"identity check failed: swapping the run's own query states at read point {mid} changed it")
            del own

        brng = derive_seed(cfg.seed, "posset", task, "bootstrap")
        level = cfg.alpha / (M * len(kinds))  # Bonferroni over the read points and sources (D52)
        rates: dict[str, Any] = {}
        gains: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        arrays: dict[str, np.ndarray] = {}
        for ki, k in enumerate(kinds):
            rows = type_rows[k]
            n = len(rows)
            if n < 2:
                log.info("%s/%s: fewer than two bases kept; skipped", task, k)
                continue
            brs = [base_prompts[i] for i in rows]
            srs = [src_prompts[k][i] for i in rows]
            pred = np.array([prediction(k, bases[i], sources[i][k]) for i in rows])
            base_am = base_run.query_argmax[rows]
            # the source states at the read points of the check: (n, M, P, d), captured in chunks of the batch size
            parts = []
            bs = backend.cfg.batch_size
            for s0 in range(0, n, bs):
                st = backend.capture_sites(srs[s0 : s0 + bs], positions[rows[s0 : s0 + bs]])
                parts.append(st[:, rp_t.to(st.device)])
                del st
            src_states = torch.cat(parts, dim=0)
            del parts
            ex, mm = np.meshgrid(np.arange(n), np.arange(M), indexing="ij")
            ex, mm = ex.ravel(), mm.ravel()
            rp = np.asarray(read_points)[mm]
            prompts = [brs[e] for e in ex]
            dev = src_states.device
            ex_t, mm_t = torch.as_tensor(ex, device=dev), torch.as_tensor(mm, device=dev)
            rates[k] = {"n": n}
            flips: dict[str, np.ndarray] = {}
            for si, name in enumerate(SETS):
                J = members[name]
                J_t = torch.as_tensor(J, device=dev)
                values = src_states[ex_t[:, None], mm_t[:, None], J_t[None, :]]  # (n*M, |S|, d)
                patch = SitePatch(rp, positions[rows][ex][:, J], values)
                res = backend.run(prompts, site_patch=patch, token_ids=tid)
                if cfg.determinism_check and "swap" not in self.metadata.get("determinism_check", {}):
                    sub = SitePatch(rp[:bs], positions[rows][ex[:bs]][:, J], values[:bs])
                    a = backend.run(prompts[:bs], site_patch=sub, token_ids=tid)
                    b = backend.run(prompts[:bs], site_patch=sub, token_ids=tid)
                    cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
                    self.metadata.setdefault("determinism_check", {})["swap"] = {"identical": cmp["identical"]}
                    if not cmp["identical"]:
                        raise RuntimeError("determinism check failed: a repeated set-swap batch differs")
                del values
                am = res.query_argmax.reshape(n, M)
                flip = am == tid[pred][:, None]
                change = am != base_am[:, None]
                flips[name] = flip
                seed = brng + 10 * ki + 2 * si
                rates[k][name] = {"flip": bootstrap_rate(flip, np.random.default_rng(seed), cfg.n_boot, cfg.alpha),
                                  "change": bootstrap_rate(change, np.random.default_rng(seed + 1), cfg.n_boot, cfg.alpha)}
                arrays[f"{k}/{name}/argmax"] = am.astype(np.int32)
            arrays[f"{k}/bases"] = rows
            del src_states
            gains[k] = {name: paired_gain(flips[name], flips["answer"],
                                          np.random.default_rng(derive_seed(cfg.seed, "posset", task, "gain", k, name)),
                                          cfg.n_boot_gain, level)
                        for name in SETS if name != "answer"}
            log.info("%s/%s: %d bases; flip rate per set at read points %s: %s", task, k, n, read_points,
                     {name: np.round(rates[k][name]["flip"]["mean"], 2).tolist() for name in SETS})
            log.info("%s/%s: non-digit gain %s (lower bounds %s)", task, k, np.round(gains[k]["non_digit"]["mean"], 3).tolist(),
                     np.round(gains[k]["non_digit"]["lo"], 3).tolist())

        decision = decide(gains, read_points, cfg.min_gain, cfg.jointly_enough, cfg.max_sites, cfg.control)
        log.info("%s: D52 rule: %s", task, decision)
        td = self.root / "core" / "tasks" / task
        td.mkdir(parents=True, exist_ok=True)
        result = {
            "task": task, "template": TEMPLATES[task], "labels": labels, "sets": members, "read_points": read_points,
            "marks": cfg.marks, "n_bases": len(bases), "n_kept": {k: int(len(v)) for k, v in type_rows.items()},
            "accuracy": accuracy, "source_accuracy": {k: float(src_ok[k][keep].mean()) for k in kinds},
            "gain_level": level, "rates": rates, "gains": gains, "decision": decision,
        }
        write_json(td / "posset.json", result)
        np.savez_compressed(td / "posset_arrays.npz", **{key.replace("/", "__"): v for key, v in arrays.items()})
        if cfg.figures:
            try:
                mc = cfg.model
                plot_task(result, self.root / "figures" / f"{task}.png",
                          mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}")
            except Exception as exc:  # a figure never fails a run
                log.warning("%s: figure failed: %s", task, exc)
        return {"summary": {"accuracy": accuracy, "n_kept": result["n_kept"], "decision": decision}}


def plot_task(result: dict[str, Any], path: Path, model_name: str) -> None:
    """One panel per source: the rate (source answer for the combined source, the bracket's prediction otherwise)
    against the read point, one line per set, with the D51 marks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"answer": "#0b0b0b", "answer_eq": "#2a78d6", "non_digit": "#eb6834", "answer_digits": "#1baf7a",
               "query": "#8a8985"}
    kinds = list(result["rates"])
    rps = result["read_points"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(2.9 * len(kinds) + 0.6, 2.9), sharey=True, squeeze=False)
    combined = COMBINED[result["task"]]
    for ax, k in zip(axes[0], kinds):
        for name in SETS:
            r = result["rates"][k][name]["flip"]
            ax.fill_between(rps, r["lo"], r["hi"], color=colours[name], alpha=0.12, lw=0)
            ax.plot(rps, r["mean"], color=colours[name], lw=1.5 if name != "query" else 1.0, marker="o", ms=2.5,
                    ls="-" if name != "query" else (0, (2, 1.5)), label=SET_LABELS[name])
        for lab, m in result.get("marks", {}).items():
            ax.axvline(m, color="#c3c2b7", lw=0.8, ls=(0, (1, 2)))
            ax.text(m, 1.01, lab, fontsize=6.5, ha="center", va="bottom", color="#52514e")
        ax.set_title(f"source changing {'every sum' if k == combined else k}: answer = "
                     f"{'source answer' if k == combined else 'its prediction'} (n {result['rates'][k]['n']})",
                     fontsize=7, loc="left", pad=12)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(rps)
        ax.tick_params(labelsize=6.5, length=2, colors="#52514e")
        ax.set_xlabel("read point", fontsize=7, color="#52514e")
        ax.grid(axis="y", color="#ecebe7", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("share of bases", fontsize=7, color="#52514e")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.1), ncol=5, fontsize=7, frameon=False)
    d = result["decision"]
    verdict = (f"D52 rule passes: joint DAS at {d['positions']} tokens, read points {d['sites']}" if d["passes"]
               else "D52 rule does not pass: no set of non-digit tokens adds to the answer token")
    fig.suptitle(f"Swapping sets of query tokens together: {result['template'].strip()} ({model_name}). {verdict}",
                 fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def run_posset(cfg: PossetConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return PositionSets(cfg, run_id, config_path).run()
