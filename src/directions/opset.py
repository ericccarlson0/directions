"""Where the operator sits between its token and the answer token: token-set swaps (docs/DECISIONS.md D59).

The query is split into six groups: O (the operator token), F (``a + b``), S (``c + d``), P (the other structural
tokens), E (`` =``) and A (the answer position). Each of 23 sets of groups is swapped as a whole at one read point
(D52's set swap), in two arms: the natural operator swap (D57: a base and its twin under the other operator) and the
added direction (D58: min expressions shown as max with v[6] at the operator token, the source the same prompt without
it). A set is sufficient where its flip rate's lower bound reaches 0.5; the gap is where neither O nor A alone is.

    uv run directions opset --config configs/opset_qwen3_8b.yaml --run-id opset_qwen3_8b_seed20260916
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import OpsetConfig
from .determinism import compare_arrays
from .model import Intervention, ModelBackend, SitePatch, environment_metadata, gpu_peak_memory, set_torch_determinism
from .opsteer import natural_direction, op_position, sample_opsteer, shown, steer_at
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed
from .varmap import bootstrap_rate, render, sample_bases

GROUPS = ("O", "F", "S", "P", "E", "A")
GROUP_LABELS = {"O": "operator", "F": "a + b", "S": "c + d", "P": "( , space )", "E": "=", "A": "answer"}
# D59's family of 23 sets, each a string of group letters in GROUPS order
SINGLES = list(GROUPS)
WITH_ANSWER = ["OA", "FA", "SA", "PA", "EA"]
ALL_BUT_ONE = ["".join(x for x in GROUPS if x != g) for g in GROUPS]
BLOCKS = ["FS", "PE", "FSPE", "OFSA", "OPEA"]
QUERY = "OFSPEA"
FAMILY = SINGLES + WITH_ANSWER + ALL_BUT_ONE + BLOCKS + [QUERY]
K_OPERANDS, K_STRUCTURE, ENDS = "OFSA", "OPEA", "OA"


def token_groups(labels: list[str]) -> dict[str, list[int]]:
    """The query-token indices of each group, from the labels of :func:`directions.varmap.query_tokens` with the
    operator labelled ``op``; the groups partition the query."""
    ans = len(labels) - 1
    if labels[ans] != "␣":
        raise ValueError(f"the last query token is not the answer position: {labels}")
    g = {"O": [j for j, lab in enumerate(labels) if lab == "op"], "A": [ans],
         "E": [j for j, lab in enumerate(labels) if lab.strip() == "="],
         "F": list(range(labels.index("a"), labels.index("b") + 1)),
         "S": list(range(labels.index("c"), labels.index("d") + 1))}
    taken = {j for v in g.values() for j in v}
    g["P"] = [j for j in range(len(labels)) if j not in taken]
    if any(not v for v in g.values()) or sum(len(v) for v in g.values()) != len(labels):
        raise ValueError(f"the groups do not partition the query {labels}: {g}")
    return {k: g[k] for k in GROUPS}


def members(name: str, groups: dict[str, list[int]]) -> list[int]:
    return sorted(j for letter in name for j in groups[letter])


def readings(rates: dict[str, dict[str, Any]], read_points: list[int], threshold: float) -> dict[str, Any]:
    """D59's readings of one arm: the gap, the class at each gap read point and the gap's reading (the majority
    class); beside them O+A's sufficiency, each group's necessity and the family's minimal sufficient sets."""
    lo = {k: np.asarray(r["flip"]["lo"]) for k, r in rates.items()}
    hi = {k: np.asarray(r["flip"]["hi"]) for k, r in rates.items()}

    def suff(k: str, i: int) -> bool:
        return bool(lo[k][i] >= threshold)

    def insuff(k: str, i: int) -> bool:
        return bool(hi[k][i] < threshold)

    gap = [i for i in range(len(read_points)) if not suff("O", i) and not suff("A", i)]
    classes = {}
    for i in gap:
        o, s = K_OPERANDS, K_STRUCTURE
        if suff(o, i) and suff(s, i):
            c = "either side"
        elif suff(o, i) and insuff(s, i):
            c = "operands"
        elif suff(s, i) and insuff(o, i):
            c = "structure"
        elif insuff(o, i) and insuff(s, i):
            c = "only together"
        else:
            c = "undecided"
        classes[int(read_points[i])] = c
    counts = Counter(classes.values())
    top = counts.most_common()
    if not top:
        reading = "no gap"
    elif len(top) > 1 and top[0][1] == top[1][1]:
        reading = "tie: " + ", ".join(sorted(c for c, n in top if n == top[0][1]))
    else:
        reading = top[0][0]
    necessity = {g: [int(read_points[i]) for i in range(len(read_points)) if insuff("".join(x for x in GROUPS if x != g), i)]
                 for g in GROUPS}
    minimal = {}
    for i, m in enumerate(read_points):
        ok = [k for k in rates if suff(k, i)]
        minimal[int(m)] = sorted((k for k in ok if not any(j != k and set(j) < set(k) for j in ok)), key=lambda x: (len(x), x))
    return {"gap": [int(read_points[i]) for i in gap], "classes": classes, "counts": dict(counts), "reading": reading,
            "ends_sufficient": [int(read_points[i]) for i in range(len(read_points)) if suff(ENDS, i)],
            "necessity": necessity, "minimal_sufficient": minimal}


class OperatorSets:
    def __init__(self, cfg: OpsetConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_opset_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core").mkdir(parents=True)
        (self.root / "figures").mkdir()
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        cfg = self.cfg
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        self.tid = self._digit_ids()
        L = self.backend.n_layers
        if max(cfg.read_points) > L:
            raise ValueError(f"read points {cfg.read_points} beyond the model's {L}")
        result: dict[str, Any] = {"read_points": list(cfg.read_points), "family": FAMILY, "arms": {}}
        arrays: dict[str, np.ndarray] = {}
        for arm in cfg.arms:
            t0 = time.time()
            setup = self._natural() if arm == "natural" else self._added()
            res, arr = self._sweep(arm, setup)
            result["arms"][arm] = res
            arrays.update({f"{arm}__{k}": v for k, v in arr.items()})
            self.metadata.setdefault("timings_seconds", {})[arm] = round(time.time() - t0, 1)
        write_json(self.root / "core" / "opset.json", result)
        np.savez_compressed(self.root / "core" / "opset_arrays.npz", **arrays)
        write_json(self.root / "core" / "summary.json",
                   {arm: {k: r[k] for k in ("n", "reading", "gap", "classes", "ends_sufficient")} for arm, r in result["arms"].items()})
        if cfg.figures:
            try:
                self._figure(result)
            except Exception as exc:  # a figure never fails a run
                self.log.warning("figure failed: %s", exc)
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()
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

    # ------------------------------------------------------------------ #

    def _natural(self) -> dict[str, Any]:
        """D57's operator swap: bases drawn as D57's, each with its twin under the other operator as the source."""
        cfg, backend, tid = self.cfg, self.backend, self.tid
        bases, sources, demos, info = sample_bases(cfg.seed, "maxmin", cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata.setdefault("sampling", {})["natural"] = info
        twins = [s["op"] for s in sources]
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        src_prompts = [render(demos[i], t) for i, t in enumerate(twins)]
        positions, keep, labels = self._align(base_prompts, bases, src_prompts, twins, differ="op")
        base_am = backend.run(base_prompts, token_ids=tid).query_argmax
        src_am = backend.run(src_prompts, token_ids=tid).query_argmax
        ok = keep & (base_am == tid[[b.Y for b in bases]]) & (src_am == tid[[t.Y for t in twins]])
        rows = np.flatnonzero(ok if self.cfg.gate else keep)
        self.log.info("natural: %d bases, %d aligned, %d with base and twin answered", len(bases), int(keep.sum()), int(ok.sum()))
        return {"rows": rows, "base_prompts": base_prompts, "src_prompts": src_prompts, "positions": positions,
                "labels": labels, "pred": np.array([t.Y for t in twins]), "base_am": base_am, "steer": None,
                "accuracy": {"base": float((base_am == tid[[b.Y for b in bases]])[keep].mean()),
                             "twin": float((src_am == tid[[t.Y for t in twins]])[keep].mean())},
                "examples": [b.text() for b in bases]}

    def _added(self) -> dict[str, Any]:
        """D58's steered runs: min expressions shown as max, v[m_inj] added at the operator token; the source is the
        same prompt without the direction (D58's draws and direction, recomputed)."""
        cfg, backend, tid = self.cfg, self.backend, self.tid
        draw = sample_opsteer(cfg.seed, cfg.n_dir, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata.setdefault("sampling", {})["added"] = draw["info"]
        direction = natural_direction(backend, draw["dir_set"], draw["dir_demos"], self.rejections, self.log)
        v = direction["v"]
        bases, demos = draw["bases"], draw["demos"]
        prompts = [render(demos[i], shown(b)) for i, b in enumerate(bases)]
        positions, keep, labels = self._align(prompts, [shown(b) for b in bases], prompts, [shown(b) for b in bases], differ=None)
        j_op = max(j for j, lab in enumerate(labels) if lab == "op")
        op_pos = positions[:, j_op]
        steered = backend.run(prompts, interventions=[steer_at(v, cfg.inject_read_point, op_pos)], token_ids=tid).query_argmax
        plain = backend.run(prompts, token_ids=tid).query_argmax
        ok = keep & (steered == tid[[b.Y for b in bases]]) & (plain == tid[[shown(b).Y for b in bases]])
        rows = np.flatnonzero(ok if cfg.gate else keep)
        self.log.info("added: %d bases, %d aligned, %d with the steered run min and the unsteered max", len(bases),
                      int(keep.sum()), int(ok.sum()))
        m0 = cfg.inject_read_point
        vec = v[m0].cpu().numpy().astype(np.float32)
        return {"rows": rows, "base_prompts": prompts, "src_prompts": prompts, "positions": positions, "labels": labels,
                "pred": np.array([shown(b).Y for b in bases]), "base_am": steered,
                "steer": lambda r: Intervention(m0, vec, 1.0, op_pos[r]),
                "accuracy": {"steered_min": float((steered == tid[[b.Y for b in bases]])[keep].mean()),
                             "unsteered_max": float((plain == tid[[shown(b).Y for b in bases]])[keep].mean())},
                "direction": {k: direction[k] for k in ("n_pairs", "norm", "pair_cosine")},
                "examples": [shown(b).text() for b in bases]}

    def _align(self, base_prompts: list[Prompt], bases: list, src_prompts: list[Prompt], srcs: list, differ: str | None
               ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Query positions per base; a base is kept when its source has the same positions and the same tokens but
        at the operator (``differ`` "op") or everywhere (``differ`` None: the same prompt)."""
        _, pos0, _, labels = op_position(self.backend, base_prompts[0], bases[0])
        P = len(pos0)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        keep = np.zeros(len(bases), dtype=bool)
        for i in range(len(bases)):
            _, pos, ids, lab = op_position(self.backend, base_prompts[i], bases[i])
            _, spos, sids, _ = op_position(self.backend, src_prompts[i], srcs[i])
            same = len(pos) == P and lab == labels and spos == pos and all(
                x == y for x, y, la in zip(ids, sids, labels) if not (differ == "op" and la == "op"))
            if same:
                positions[i], keep[i] = pos, True
            else:
                self.rejections.add("align", "maxmin", "base or source misaligned", {"base": bases[i].text()})
        return positions, keep, labels

    def _sweep(self, arm: str, s: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Every set of the family swapped from the source run at every read point of the sweep, on the kept bases."""
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        rows, labels = s["rows"], s["labels"]
        groups = token_groups(labels)
        n, read_points = len(rows), list(cfg.read_points)
        M = len(read_points)
        out: dict[str, Any] = {"n": int(n), "labels": labels, "groups": groups, "accuracy": s["accuracy"],
                               **({"direction": s["direction"]} if "direction" in s else {})}
        if n < 2:
            log.info("%s: fewer than two bases kept; no sweep", arm)
            return {**out, "rates": {}, **readings({}, [], cfg.sufficient)}, {}
        brs = [s["base_prompts"][i] for i in rows]
        srs = [s["src_prompts"][i] for i in rows]
        pos = s["positions"][rows]
        steer: Callable[[np.ndarray], Intervention] | None = s["steer"]

        def ivs(r: np.ndarray) -> list[Intervention]:
            return [] if steer is None else [steer(rows[r])]

        if cfg.identity_check:  # a run's own states swapped back at every query token change nothing
            r = np.arange(min(n, backend.cfg.batch_size))
            own = backend.capture_sites([brs[i] for i in r], pos[r], interventions=ivs(r))
            mid = read_points[M // 2]
            ref = backend.run([brs[i] for i in r], interventions=ivs(r), token_ids=tid)
            got = backend.run([brs[i] for i in r], interventions=ivs(r), token_ids=tid,
                              site_patch=SitePatch(np.full(len(r), mid), pos[r], own[:, mid]))
            same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
            self.metadata.setdefault("identity_check", {})[f"{arm}/read_point_{mid}/query"] = same
            if not same:
                raise RuntimeError(f"identity check failed ({arm}): a run's own query states at read point {mid} changed it")
            del own
        # the source states at the sweep's read points, (n, M, P, d), captured in chunks of the batch size
        bs = backend.cfg.batch_size
        rp_t = torch.as_tensor(read_points)
        parts = []
        for s0 in range(0, n, bs):
            st = backend.capture_sites(srs[s0 : s0 + bs], pos[s0 : s0 + bs])
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
        pred = tid[s["pred"][rows]]
        rates: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {"rows": rows}
        for name in FAMILY:
            J = members(name, groups)
            values = src_states[ex_t[:, None], mm_t[:, None], torch.as_tensor(J, device=dev)[None, :]]  # (n*M, |J|, d)
            patch = SitePatch(rp, pos[ex][:, J], values)
            res = backend.run(prompts, site_patch=patch, interventions=ivs(ex), token_ids=tid)
            if cfg.determinism_check and "swap" not in self.metadata.get("determinism_check", {}):
                sub = SitePatch(rp[:bs], pos[ex[:bs]][:, J], values[:bs])
                a = backend.run(prompts[:bs], site_patch=sub, interventions=ivs(ex[:bs]), token_ids=tid)
                b = backend.run(prompts[:bs], site_patch=sub, interventions=ivs(ex[:bs]), token_ids=tid)
                cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
                self.metadata.setdefault("determinism_check", {})["swap"] = {"identical": cmp["identical"]}
                if not cmp["identical"]:
                    raise RuntimeError("determinism check failed: a repeated set-swap batch differs")
            del values
            am = res.query_argmax.reshape(n, M)
            flip = am == pred[:, None]
            rates[name] = {"flip": bootstrap_rate(flip, np.random.default_rng(derive_seed(cfg.seed, "opset", arm, name)),
                                                  cfg.n_boot, cfg.alpha)}
            arrays[name] = am.astype(np.int32)
            log.info("%s/%s: flip rate at read points %s: %s", arm, name, read_points[::4],
                     [round(float(x), 2) for x in rates[name]["flip"]["mean"][::4]])
        del src_states
        # the check: the whole query swapped gives the source's own run (bit-exact up to the batch's padding in bf16)
        src_am = backend.run(srs, token_ids=tid).query_argmax
        query_check = float((arrays[QUERY] == src_am[:, None]).mean())
        self.metadata.setdefault("query_check", {})[arm] = query_check
        log.info("%s: the whole query swapped gives the source's answer on %.4f of the swaps", arm, query_check)
        read = readings(rates, read_points, cfg.sufficient)
        log.info("%s: gap %s; classes %s; reading %s; O+A sufficient at %s", arm, read["gap"], read["classes"], read["reading"],
                 read["ends_sufficient"])
        return {**out, "rates": rates, **read, "query_check": query_check, "examples": [s["examples"][i] for i in rows]}, arrays

    def _figure(self, result: dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mc = self.cfg.model
        name = mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}"
        arms = [a for a, r in result["arms"].items() if r.get("rates")]
        if not arms:
            return
        rps = np.asarray(result["read_points"])
        fig, axes = plt.subplots(2, len(arms), figsize=(6.2 * len(arms), 9.5), squeeze=False,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
        lines = {"O": ("#8a8985", "operator token alone"), "A": ("#0b0b0b", "answer token alone"),
                 K_OPERANDS: ("#2a78d6", "O + operands + A"), K_STRUCTURE: ("#eb6834", "O + structure + A"),
                 ENDS: ("#1baf7a", "O + A"), "FSPE": ("#8a5cd0", "everything but O and A")}
        for c, arm in enumerate(arms):
            r = result["arms"][arm]
            order = sorted(r["rates"], key=lambda k: (len(k), FAMILY.index(k)))
            mat = np.stack([np.asarray(r["rates"][k]["flip"]["mean"]) for k in order])
            ax = axes[0, c]
            im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(["+".join(k) for k in order], fontsize=6.5)
            step = max(1, len(rps) // 8)
            ax.set_xticks(range(0, len(rps), step))
            ax.set_xticklabels(rps[::step], fontsize=7)
            ax.set_title(f"{arm}: flip rate per set of token groups ({r['n']} bases)\n"
                         + ", ".join(f"{g} {GROUP_LABELS[g]}" for g in GROUPS), fontsize=8, loc="left")
            for i in r["gap"]:
                ax.axvline(list(rps).index(i), color="#d9a521", lw=0.4, alpha=0.6)
            fig.colorbar(im, ax=ax, fraction=0.03)
            ax = axes[1, c]
            for k, (col, lab) in lines.items():
                f = r["rates"][k]["flip"]
                ax.fill_between(rps, f["lo"], f["hi"], color=col, alpha=0.12, lw=0)
                ax.plot(rps, f["mean"], color=col, lw=1.4, label=lab)
            ax.axhline(0.5, color="#52514e", lw=0.6, ls=":")
            ax.set_xlabel("read point", fontsize=8)
            ax.set_ylabel("answer = the other operation's", fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, frameon=False, loc="center left")
            ax.set_title(f"reading of the gap: {r['reading']}", fontsize=8, loc="left")
        fig.suptitle(f"Token-set swaps of the operator (D59), {name}", fontsize=9, x=0.01, ha="left")
        fig.savefig(self.root / "figures" / "opset.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)


def run_opset(cfg: OpsetConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return OperatorSets(cfg, run_id=run_id, config_path=config_path).run()
