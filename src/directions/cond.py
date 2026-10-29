"""The full test of the conditional task (docs/DECISIONS.md D63): where a computed condition is formed, where it meets
the branch values, and whether the unchosen branch is computed.

``IF(g>h, a+b, c+d) = `` (the form D62's screen selected; the given condition ``IF(k, ...)`` is the matched arm). Four
parts, each its own run:

- ``map`` (D45/D57): whole-state swaps at every (read point, query token) from sources that flip the condition (both
  crossings flipped, so a site holding either digit flips it too), keep it with the ``h`` crossing flipped (a site
  holding the digit flips the answer, one holding the computed condition does not), change the chosen sum, or keep it
  with other digits;
- ``sets`` (D59): six token groups (the condition Cd, the chosen and the unchosen branch Bch and Bun, the other
  structure P, ``=`` E, the answer position A) and their unions swapped together, for the condition and the chosen sum;
- ``pairs`` (D61): two factors changed independently and together, the additive state h10 + h01 - h00 against the
  joint at every query token: the condition x both sums (CB), the condition x the unchosen sum (CU), the chosen
  branch's two digits (AB: where a sum is formed), condition digits that keep the condition x the chosen sum (CN: the
  null), and on ``if_mixed`` the sign x the condition's digits (SG: where the condition is computed);
- ``probe`` (D46): conditional linear probes of the condition and of the unchosen sum given the answer at the
  condition's last token, ``=`` and the answer position.

    uv run directions cond --config configs/cond_qwen3_8b.yaml --part map --run-id cond_map_qwen3_8b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import CondConfig
from .determinism import compare_arrays
from .interact import conversion_depth
from .model import (
    ModelBackend,
    SitePatch,
    environment_metadata,
    gpu_peak_memory,
    set_torch_determinism,
)
from .probes import conditional_probe
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .screen import SUM_PAIRS, CondExpr, demos_for, draw_balanced, query_tokens, render
from .seeds import derive_seed, rng_for
from .varmap import bootstrap_rate

GROUPS = ("Cd", "Bch", "Bun", "P", "E", "A")
GROUP_LABELS = {"Cd": "condition", "Bch": "chosen branch", "Bun": "unchosen branch", "P": "structure", "E": "=",
                "A": "answer"}
MAP_TYPES = {"if_gt": ("cond", "cond_same", "chosen", "chosen_same"),
             "ternary": ("cond", "cond_same", "chosen", "chosen_same"),
             "python": ("cond", "cond_same", "chosen", "chosen_same"),
             "if_mixed": ("cond", "cond_same", "sign", "chosen", "chosen_same"),
             "if_given": ("cond", "chosen", "chosen_same")}
PAIRS = {"if_gt": ("CB", "CU", "AB", "CN"), "ternary": ("CB", "CU", "AB", "CN"), "python": ("CB", "CU", "AB", "CN"),
         "if_mixed": ("SG", "CB", "CU", "AB", "CN"), "if_given": ("CB", "CU", "AB")}
# the token-set family: each a "+"-joined list of groups; the last is the whole query (the check)
SETS = ["Cd", "Bch", "Bun", "P", "E", "A", "E+A", "Cd+A", "Cd+E", "Cd+E+A", "P+E", "P+E+A", "Bch+A", "Bch+E+A",
        "Cd+Bch", "Cd+Bch+A", "Cd+Bch+Bun+P+E+A"]
SET_TYPES = ("cond", "chosen")
PATCHES = ("joint", "additive") + tuple(f"add_{g}" for g in GROUPS)
CATS = ("11", "10", "01", "00", "other")
PROBE_TARGETS = ("C", "C|Y", "Sun|Y")
FORMS = tuple(MAP_TYPES)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _C(e: CondExpr, g: int, h: int) -> int:
    return replace(e, g=g, h=h).C


def _pick(cands: list[CondExpr], rng: np.random.Generator, pool: set[str]) -> CondExpr | None:
    cands = [c for c in cands if c.text() not in pool]
    return cands[int(rng.integers(len(cands)))] if cands else None


def flip_cond(e: CondExpr, rng: np.random.Generator, pool: set[str]) -> CondExpr | None:
    """The condition flipped. With digits, both crossings flip too: (g', h), (g, h') and (g', h') all give the other
    outcome, so that a swap at a site holding either digit alone flips the answer; no comparison ties."""
    if e.form == "if_given":
        return _pick([replace(e, k=1 - e.k)], rng, pool)
    C = e.C
    cands = [replace(e, g=g2, h=h2) for g2 in range(10) for h2 in range(10)
             if g2 != h2 and g2 != e.h and h2 != e.g and _C(e, g2, h2) != C and _C(e, g2, e.h) != C and _C(e, e.g, h2) != C]
    return _pick(cands, rng, pool)


def same_cond(e: CondExpr, rng: np.random.Generator, pool: set[str], crossing: bool = True) -> CondExpr | None:
    """Other condition digits with the same outcome. With ``crossing``, the ``h`` crossing flips: (g, h') gives the
    other outcome, so a swap at a site holding the digit h' flips the answer and one holding the computed condition
    does not (D45's value test for the condition)."""
    if e.form == "if_given":
        return None
    C = e.C
    cands = [replace(e, g=g2, h=h2) for g2 in range(10) for h2 in range(10)
             if g2 != h2 and (g2, h2) != (e.g, e.h) and _C(e, g2, h2) == C
             and (not crossing or (h2 != e.g and g2 != e.h and _C(e, e.g, h2) != C))]
    return _pick(cands, rng, pool)


def flip_sign(e: CondExpr, pool: set[str]) -> CondExpr | None:
    s = replace(e, cmp="<" if e.cmp == ">" else ">")
    return None if s.text() in pool else s


def _branch(e: CondExpr, first: bool) -> tuple[int, int]:
    return (e.a, e.b) if first else (e.c, e.d)


def _with_branch(e: CondExpr, first: bool, x: int, y: int) -> CondExpr:
    return replace(e, a=x, b=y) if first else replace(e, c=x, d=y)


def change_branch(e: CondExpr, first: bool, rng: np.random.Generator, pool: set[str], avoid: set[int]) -> CondExpr | None:
    """One branch's sum changed to a value outside ``avoid`` (its own sum always avoided), the rest kept."""
    own = sum(_branch(e, first))
    cands = [_with_branch(e, first, x, y) for x, y in SUM_PAIRS if x + y != own and x + y not in avoid]
    return _pick([c for c in cands if c.S1 != c.S2], rng, pool)


def same_branch(e: CondExpr, first: bool, rng: np.random.Generator, pool: set[str]) -> CondExpr | None:
    """One branch's digits changed with its sum kept."""
    x0, y0 = _branch(e, first)
    return _pick([_with_branch(e, first, x, y) for x, y in SUM_PAIRS if x + y == x0 + y0 and (x, y) != (x0, y0)], rng, pool)


def map_sources(e: CondExpr, types: tuple[str, ...], rng: np.random.Generator, pool: set[str]) -> dict[str, tuple[CondExpr, int] | None]:
    """Each map type's source and the answer its swap predicts (None for a type without a valid source; a base enters
    each type's map on its own, as D45's)."""
    out: dict[str, tuple[CondExpr, int] | None] = {}
    first = bool(e.C)
    for k in types:
        if k == "cond":
            s, pred = flip_cond(e, rng, pool), e.other
        elif k == "cond_same":
            s, pred = same_cond(e, rng, pool), e.Y
        elif k == "sign":
            s, pred = flip_sign(e, pool), e.other
        elif k == "chosen":
            s = change_branch(e, first, rng, pool, {e.other})
            pred = s.Y if s is not None else -1
        elif k == "chosen_same":
            s, pred = same_branch(e, first, rng, pool), e.Y
        else:
            raise ValueError(f"unknown map type {k!r}")
        out[k] = None if s is None else (s, pred)
    return out


def pair_quad(pair: str, e: CondExpr, rng: np.random.Generator, pool: set[str]) -> list[CondExpr] | None:
    """The four prompts 00, 10, 01, 11 of a factor pair (None when the base has none)."""
    first = bool(e.C)
    if pair == "SG":  # the sign x the condition's digits (both crossings flipped): 11 flips twice, back to 00's answer
        x, z = flip_sign(e, pool), flip_cond(e, rng, pool)
        xz = None if z is None else flip_sign(z, pool)
        quad = None if None in (x, z, xz) else [e, x, z, xz]
    elif pair in ("CB", "CU", "CN"):
        x = same_cond(e, rng, pool, crossing=False) if pair == "CN" else flip_cond(e, rng, pool)
        if x is None:
            return None
        if pair == "CB":  # both sums changed, the four answers distinct
            cands = [replace(e, a=a, b=b, c=c, d=d) for (a, b) in SUM_PAIRS for (c, d) in SUM_PAIRS
                     if a + b != c + d and a + b != e.S1 and c + d != e.S2]
            cands = [z for z in cands if len({e.Y, e.other, z.Y, z.other}) == 4]
            z = _pick(cands, rng, pool)
        elif pair == "CU":  # the unchosen sum changed, to a value neither the chosen nor its own
            z = change_branch(e, not first, rng, pool, {e.Y})
        else:  # CN: the chosen sum changed; the condition keeps its outcome
            z = change_branch(e, first, rng, pool, {e.other})
        if z is None:
            return None
        xz = replace(x, a=z.a, b=z.b, c=z.c, d=z.d)
        quad = None if xz.text() in pool else [e, x, z, xz]
    elif pair == "AB":  # the chosen branch's first digit x its second: where a sum is formed
        x0, y0 = _branch(e, first)
        cands = [(x1, y1) for x1 in range(10) for y1 in range(10) if x1 != x0 and y1 != y0
                 and len({x0 + y0, x1 + y0, x0 + y1, x1 + y1, e.other}) == 5 and max(x1 + y0, x0 + y1, x1 + y1) <= 9]
        cands = [c for c in cands if not ({_with_branch(e, first, *xy).text() for xy in ((c[0], y0), (x0, c[1]), c)} & pool)]
        if not cands:
            return None
        x1, y1 = cands[int(rng.integers(len(cands)))]
        quad = [e, _with_branch(e, first, x1, y0), _with_branch(e, first, x0, y1), _with_branch(e, first, x1, y1)]
    else:
        raise ValueError(f"unknown pair {pair!r}")
    if quad is None or len({q.text() for q in quad}) != 4 or any(q.S1 == q.S2 for q in quad):
        return None
    return quad


def token_groups(labels: list[str], C: int) -> dict[str, list[int]]:
    """The query-token indices of each group for a base whose first branch is chosen iff ``C``; the groups partition
    the query."""
    ans = len(labels) - 1
    if labels[ans] != "␣":
        raise ValueError(f"the last query token is not the answer position: {labels}")
    cond = [j for j, lab in enumerate(labels) if lab in ("<g>", "<h>", "<k>", "<cmp>")]
    b1 = list(range(labels.index("<a>"), labels.index("<b>") + 1))
    b2 = list(range(labels.index("<c>"), labels.index("<d>") + 1))
    g = {"Cd": list(range(min(cond), max(cond) + 1)), "Bch": b1 if C else b2, "Bun": b2 if C else b1,
         "E": [j for j, lab in enumerate(labels) if lab.strip() == "="], "A": [ans]}
    taken = {j for v in g.values() for j in v}
    g["P"] = [j for j in range(len(labels)) if j not in taken]
    if any(not v for v in g.values()) or sum(len(v) for v in g.values()) != len(labels):
        raise ValueError(f"the groups do not partition the query {labels}: {g}")
    return {k: g[k] for k in GROUPS}


def _first(mask: np.ndarray, read_points: list[int]) -> int | None:
    idx = np.flatnonzero(np.asarray(mask))
    return int(read_points[idx[0]]) if len(idx) else None


def _last(mask: np.ndarray, read_points: list[int]) -> int | None:
    idx = np.flatnonzero(np.asarray(mask))
    return int(read_points[idx[-1]]) if len(idx) else None


def map_readouts(rates: dict[str, Any], labels: list[str], read_points: list[int], present: float, value_change: float
                 ) -> dict[str, Any]:
    """D63's map landmarks: the condition's arrival at the answer token (R_C), the last read point each condition
    token holds it (C_out), the chosen sum's arrival (G_ch), where the condition is held as a value (present for the
    flip, unchanged by the same-outcome source), the sign's landmarks on if_mixed, and the route."""
    ans = len(labels) - 1
    cond_tok = [j for j, lab in enumerate(labels) if lab in ("<g>", "<h>", "<k>", "<cmp>")]
    pres = {k: np.asarray(r["flip"]["lo"]) >= present for k, r in rates.items()}
    out: dict[str, Any] = {}
    if "cond" in pres:
        out["R_C"] = _first(pres["cond"][:, ans], read_points)
        out["C_out"] = {labels[j]: _last(pres["cond"][:, j], read_points) for j in cond_tok}
        out["route"] = {int(m): [labels[j] for j in np.flatnonzero(pres["cond"][i])] for i, m in enumerate(read_points)}
    if "cond" in pres and "cond_same" in rates:
        value = pres["cond"] & (np.asarray(rates["cond_same"]["change"]["hi"]) < value_change)
        digit = pres["cond"] & (np.asarray(rates["cond_same"]["change"]["lo"]) >= value_change)
        out["value_C"] = {labels[j]: _first(value[:, j], read_points) for j in range(len(labels)) if value[:, j].any()}
        out["digit_C"] = {labels[j]: [_first(digit[:, j], read_points), _last(digit[:, j], read_points)]
                          for j in range(len(labels)) if digit[:, j].any()}
    if "chosen" in pres:
        out["G_ch"] = _first(pres["chosen"][:, ans], read_points)
    if "sign" in pres:
        sign_tok = labels.index("<cmp>")
        out["R_sign"] = _first(pres["sign"][:, ans], read_points)
        out["sign_out"] = _last(pres["sign"][:, sign_tok], read_points)
    return out


def pair_reading(K: dict[str, int | None], localization: dict[str, Any]) -> dict[str, Any]:
    """D63's readings from the pairs' conversion depths."""
    k_cb, k_ab, k_cn, k_cu, k_sg = (K.get(p) for p in ("CB", "AB", "CN", "CU", "SG"))
    if k_cb is None:
        order = "never (the condition stays additively separable from the sums)"
    elif k_ab is None:
        order = "undecided (the chosen branch's digits never fail)"
    elif abs(k_cb - k_ab) <= 2:
        order = "together (the condition joins where the sum is formed)"
    elif k_cb >= k_ab + 3:
        order = "compute, then select (the sums are formed first)"
    else:
        order = "select, then compute (the condition joins the digits before the sum is formed)"
    out = {"K": K, "order": order, "null_clear": not (k_cn is not None and k_cb is not None and k_cn <= k_cb + 2),
           "CU_minus_CB": None if k_cu is None or k_cb is None else k_cu - k_cb, "localization": localization}
    if "SG" in K:
        loc = localization.get("SG", {})
        lows = [v["lowest"] for v in loc.values()]
        where = max(set(lows), key=lows.count) if lows else None
        out["computed"] = ("never (the sign and the digits stay separable)" if k_sg is None else
                           "at the condition's tokens" if where == "Cd" else
                           "deferred to = and the answer" if where in ("E", "A") else f"elsewhere ({where})")
    return out


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class Conditional:
    def __init__(self, cfg: CondConfig, part: str, run_id: str | None = None, config_path: str | None = None) -> None:
        if part not in ("map", "sets", "pairs", "probe"):
            raise ValueError(f"unknown part {part!r}")
        self.cfg, self.part = cfg, part
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_cond_{part}_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core").mkdir(parents=True)
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "part": part, "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        cfg = self.cfg
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        self.tid = self._digit_ids()
        L = self.backend.n_layers
        self.read_points = list(range(L + 1)) if cfg.read_points is None else [m for m in cfg.read_points if 0 <= m <= L]
        pool = draw_balanced(cfg.form, cfg.n_demo_pool, rng_for(cfg.seed, "cond", cfg.form, "pool"), set())
        self.pool, self.pool_texts = pool, {e.text() for e in pool}
        t0 = time.time()
        result = getattr(self, f"_{self.part}")()
        result.update({"form": cfg.form, "part": self.part, "read_points": self.read_points})
        write_json(self.root / "core" / f"{self.part}.json", result)
        self.metadata["timings_seconds"] = round(time.time() - t0, 1)
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

    def _bases(self, name: str, n: int) -> tuple[list[CondExpr], list[list[CondExpr]]]:
        cfg = self.cfg
        bases = draw_balanced(cfg.form, n, rng_for(cfg.seed, "cond", cfg.form, name, "bases"), self.pool_texts)
        demos = [demos_for(self.pool, cfg.n_shots, rng_for(cfg.seed, "cond", cfg.form, name, "demos", i)) for i in range(n)]
        return bases, demos

    def _aligned(self, prompts: list[Prompt], exprs: list[CondExpr]) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Each row's query positions, the reference labels, and whether the row is aligned with the reference: the
        same labels and the same template tokens (only the digit and sign slots may differ). The reference is the
        first tokenization this run sees, so every part's prompts are held to one layout."""
        toks = [query_tokens(self.backend, p, e) for p, e in zip(prompts, exprs)]
        if not hasattr(self, "_ref"):
            self._ref = (toks[0][2], toks[0][1])
        labels, ref_ids = self._ref
        ok = np.array([t[2] == labels and all(x == y for x, y, lab in zip(t[1], ref_ids, labels) if not lab.startswith("<"))
                       for t in toks])
        P = len(labels)
        pos = np.array([t[0] if len(t[0]) == P else [0] * P for t in toks], dtype=np.int64)
        return pos, labels, ok

    def _answered(self, prompts: list[Prompt], exprs: list[CondExpr]) -> tuple[np.ndarray, np.ndarray]:
        res = self.backend.run(prompts, token_ids=self.tid)
        return res.query_argmax == self.tid[[e.Y for e in exprs]], res.query_argmax

    def _identity(self, prompts: list[Prompt], positions: np.ndarray) -> None:
        if not self.cfg.identity_check or "identity_check" in self.metadata or not prompts:
            return
        backend, tid = self.backend, self.tid
        r = slice(0, min(len(prompts), backend.cfg.batch_size))
        own = backend.capture_sites(prompts[r], positions[r])
        mid = self.read_points[len(self.read_points) // 2]
        ref = backend.run(prompts[r], token_ids=tid)
        got = backend.run(prompts[r], token_ids=tid, site_patch=SitePatch(np.full(len(prompts[r]), mid), positions[r], own[:, mid]))
        same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
        self.metadata["identity_check"] = {f"{self.part}/read_point_{mid}/query": same}
        if not same:
            raise RuntimeError("identity check failed: a run's own query states swapped back changed it")

    def _determinism(self, prompts: list[Prompt], patch: SitePatch) -> None:
        if not self.cfg.determinism_check or "determinism_check" in self.metadata:
            return
        bs = min(self.backend.cfg.batch_size, len(prompts))
        sub = SitePatch(np.asarray(patch.layers)[:bs], np.asarray(patch.positions)[:bs], patch.values[:bs])
        a = self.backend.run(prompts[:bs], site_patch=sub, token_ids=self.tid)
        b = self.backend.run(prompts[:bs], site_patch=sub, token_ids=self.tid)
        cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
        self.metadata["determinism_check"] = {"patch": {"identical": cmp["identical"]}}
        if not cmp["identical"]:
            raise RuntimeError("determinism check failed: a repeated patch batch differs")

    # ------------------------------------------------------------------ #

    def _map(self) -> dict[str, Any]:
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        types = MAP_TYPES[cfg.form]
        bases, demos = self._bases("map", cfg.n_map)
        srcs = [map_sources(b, types, rng_for(cfg.seed, "cond", cfg.form, "map", "source", i), self.pool_texts)
                for i, b in enumerate(bases)]
        for i, sr in enumerate(srcs):
            if any(v is None for v in sr.values()):
                self.rejections.add("source", cfg.form, "no valid source of some type",
                                    {"base": bases[i].text(), "missing": [k for k, v in sr.items() if v is None]})
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        positions, labels, ok_b = self._aligned(base_prompts, bases)
        base_ok, base_am = self._answered(base_prompts, bases)
        log.info("map: %d bases, with a source of each type %s; query tokens %s; accuracy %.3f", len(bases),
                 {k: sum(sr[k] is not None for sr in srcs) for k in types}, labels, float(base_ok.mean()))
        ident = ok_b & (base_ok if cfg.gate else True)
        self._identity([base_prompts[i] for i in np.flatnonzero(ident)], positions[ident])
        rps, M, P = self.read_points, len(self.read_points), len(labels)
        rates, arrays, n_kept = {}, {}, {}
        for k in types:
            idx = np.flatnonzero([sr[k] is not None for sr in srcs])
            sp = [render(demos[i], srcs[i][k][0]) for i in idx]
            s_pos, _, s_ok = self._aligned(sp, [srcs[i][k][0] for i in idx])
            src_ok, _ = self._answered(sp, [srcs[i][k][0] for i in idx])
            aligned = ok_b[idx] & s_ok & np.all(s_pos == positions[idx], axis=1)
            for i in idx[~aligned]:
                self.rejections.add("align", cfg.form, f"{k} source misaligned", {"base": bases[i].text()})
            rows = idx[aligned & base_ok[idx] & src_ok] if cfg.gate else idx[aligned]
            n = len(rows)
            n_kept[k] = n
            if n < 2:
                log.info("map/%s: fewer than two bases kept", k)
                continue
            src_rows = [sp[int(np.flatnonzero(idx == i)[0])] for i in rows]
            states = backend.capture_sites(src_rows, positions[rows])
            ex, mm, jj = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(M), np.arange(P), indexing="ij"))
            rp = np.asarray(rps)[mm]
            dev = states.device
            values = states[torch.as_tensor(ex, device=dev), torch.as_tensor(rp, device=dev), torch.as_tensor(jj, device=dev)]
            prompts = [base_prompts[rows[e]] for e in ex]
            patch = SitePatch(rp, positions[rows][ex, jj], values)
            res = backend.run(prompts, site_patch=patch, token_ids=tid)
            self._determinism(prompts, patch)
            del values, states
            am = res.query_argmax.reshape(n, M, P)
            pred = np.array([srcs[i][k][1] for i in rows])
            flip = am == tid[pred][:, None, None]
            change = am != base_am[rows][:, None, None]
            seed = derive_seed(cfg.seed, "cond", "map", k, "bootstrap")
            rates[k] = {"n": n, "flip": bootstrap_rate(flip, np.random.default_rng(seed + 1), cfg.n_boot, cfg.alpha),
                        "change": bootstrap_rate(change, np.random.default_rng(seed + 2), cfg.n_boot, cfg.alpha)}
            arrays[f"{k}__argmax"], arrays[f"{k}__rows"] = am.astype(np.int32), rows
            j_a = P - 1
            log.info("map/%s: %d bases; flip at the answer token %s (read points %s)", k, n,
                     [round(float(x), 2) for x in rates[k]["flip"]["mean"][::4, j_a]], rps[::4])
        read = map_readouts(rates, labels, rps, cfg.present, cfg.value_change)
        log.info("map: landmarks %s", {k: v for k, v in read.items() if k != "route"})
        np.savez_compressed(self.root / "core" / "map_arrays.npz", **arrays)
        return {"labels": labels, "n_bases": len(bases), "n_kept": n_kept, "accuracy": float(base_ok.mean()), "rates": rates,
                "readouts": read, "thresholds": {"present": cfg.present, "value_change": cfg.value_change},
                "examples": [{"base": bases[i].text(), "Y": bases[i].Y,
                              **{k: srcs[i][k][0].text() for k in types if srcs[i][k] is not None}} for i in range(len(bases))]}

    # ------------------------------------------------------------------ #

    def _sets(self) -> dict[str, Any]:
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        bases, demos = self._bases("sets", cfg.n_sets)
        rps = [m for m in self.read_points if m >= cfg.sets_from]
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        positions, labels, ok_b = self._aligned(base_prompts, bases)
        base_ok, _ = self._answered(base_prompts, bases)
        groups = [token_groups(labels, b.C) for b in bases]
        log.info("sets: %d bases, accuracy %.3f; read points %d..%d", len(bases), float(base_ok.mean()), rps[0], rps[-1])
        out: dict[str, Any] = {"labels": labels, "sets": SETS, "set_read_points": rps, "arms": {}}
        for k in SET_TYPES:
            srcs = [map_sources(b, (k,), rng_for(cfg.seed, "cond", cfg.form, "sets", k, i), self.pool_texts) for i, b in enumerate(bases)]
            idx = np.flatnonzero(np.array([sr[k] is not None for sr in srcs]) & ok_b & (base_ok if cfg.gate else True))
            sp = [render(demos[i], srcs[i][k][0]) for i in idx]
            s_pos, _, s_ok = self._aligned(sp, [srcs[i][k][0] for i in idx])
            src_ok, _ = self._answered(sp, [srcs[i][k][0] for i in idx])
            aligned = s_ok & np.all(s_pos == positions[idx], axis=1)
            rows = idx[aligned & src_ok] if cfg.gate else idx[aligned]
            n = len(rows)
            if n < 2:
                log.info("sets/%s: fewer than two bases kept", k)
                continue
            src_rows = [sp[int(np.flatnonzero(idx == i)[0])] for i in rows]
            states = backend.capture_sites(src_rows, positions[rows])  # (n, L+1, P, d)
            pred = np.array([srcs[i][k][1] for i in rows])
            flips = np.zeros((len(SETS), n, len(rps)), dtype=bool)
            for si, name in enumerate(SETS):
                mem = [sorted(j for g in name.split("+") for j in groups[i][g]) for i in rows]
                width = max(len(m) for m in mem)
                mem = np.array([m + [m[-1]] * (width - len(m)) for m in mem], dtype=np.int64)  # padded with a repeat
                ex, mi = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(len(rps)), indexing="ij"))
                rp = np.asarray(rps)[mi]
                dev = states.device
                vals = states[torch.as_tensor(ex, device=dev)[:, None], torch.as_tensor(rp, device=dev)[:, None],
                              torch.as_tensor(mem[ex], device=dev)]
                prompts = [base_prompts[rows[e]] for e in ex]
                patch = SitePatch(rp, positions[rows][ex[:, None], mem[ex]], vals)
                res = backend.run(prompts, site_patch=patch, token_ids=tid)
                self._determinism(prompts, patch)
                flips[si] = (res.query_argmax == tid[pred][ex]).reshape(n, len(rps))
                del vals
            del states
            seed = derive_seed(cfg.seed, "cond", "sets", k, "bootstrap")
            rates = {name: bootstrap_rate(flips[si], np.random.default_rng(seed + si), cfg.n_boot, cfg.alpha) for si, name in enumerate(SETS)}
            suff = {name: (np.asarray(r["lo"]) >= cfg.present) for name, r in rates.items()}
            gap = [m for i, m in enumerate(rps) if not suff["Cd"][i] and not suff["A"][i]] if k == "cond" else []
            out["arms"][k] = {"n": n, "rates": rates,
                              "first_sufficient": {name: _first(s, rps) for name, s in suff.items()},
                              "last_sufficient": {name: _last(s, rps) for name, s in suff.items()},
                              "gap": gap, "sufficient_in_gap": {int(m): [name for name in SETS if suff[name][rps.index(m)]] for m in gap}}
            np.save(self.root / "core" / f"sets_{k}_flips.npy", flips)
            log.info("sets/%s: %d bases; first sufficient %s; gap %s", k, n, out["arms"][k]["first_sufficient"], gap)
        return out

    # ------------------------------------------------------------------ #

    def _pairs(self) -> dict[str, Any]:
        cfg, log = self.cfg, self.log
        bases, demos = self._bases("pairs", cfg.n_pairs)
        result: dict[str, Any] = {"patches": list(PATCHES), "categories": list(CATS), "pairs": {}}
        arrays: dict[str, np.ndarray] = {}
        for pair in cfg.pairs:
            t0 = time.time()
            res, arr = self._pair(pair, bases, demos)
            result["pairs"][pair] = res
            arrays.update({f"{pair}__{k}": v for k, v in arr.items()})
            self.metadata.setdefault("pair_timings_seconds", {})[pair] = round(time.time() - t0, 1)
        K = {p: r.get("K") for p, r in result["pairs"].items()}
        loc: dict[str, Any] = {}
        for p, r in result["pairs"].items():
            if r.get("K") is None or "rates" not in r:
                continue
            loc[p] = {}
            for m in range(r["K"] - 2, r["K"] + 3):
                if m in self.read_points:
                    i = self.read_points.index(m)
                    shares = {g: float(r["rates"][f"add_{g}"]["11"]["mean"][i]) for g in GROUPS}
                    loc[p][str(m)] = {"shares": shares, "lowest": min(shares, key=shares.get)}
        result["reading"] = pair_reading(K, loc)
        log.info("pairs: K %s; order %s; null clear %s; localization %s", K, result["reading"]["order"],
                 result["reading"]["null_clear"], {p: [v["lowest"] for v in x.values()] for p, x in loc.items()})
        np.savez_compressed(self.root / "core" / "pairs_arrays.npz", **arrays)
        return result

    def _pair(self, pair: str, bases: list[CondExpr], demos: list[list[CondExpr]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        quads = [pair_quad(pair, b, rng_for(cfg.seed, "cond", cfg.form, "pair", pair, i), self.pool_texts) for i, b in enumerate(bases)]
        idx = [i for i, q in enumerate(quads) if q is not None]
        if len(idx) < 2:
            return {"n": 0, "K": None}, {}
        prompts = [[render(demos[i], quads[i][k]) for i in idx] for k in range(4)]
        aligned = [self._aligned(prompts[k], [quads[i][k] for i in idx]) for k in range(4)]
        labels = aligned[0][1]
        positions = aligned[0][0]
        same = np.all([a[2] & np.all(a[0] == positions, axis=1) for a in aligned], axis=0)
        Y = np.array([[q.Y for q in quads[i]] for i in idx])
        answered = np.stack([self._answered(prompts[k], [quads[i][k] for i in idx])[0] for k in range(4)], axis=1)
        ok = same & answered.all(axis=1)
        rows = np.flatnonzero(ok if cfg.gate else same)[: cfg.pair_cap]
        n = len(rows)
        log.info("%s: %d bases with a quad, %d aligned; answered 00 %.3f, 10 %.3f, 01 %.3f, 11 %.3f; %d with all four, %d kept",
                 pair, len(idx), int(same.sum()), *[float(answered[same, k].mean()) for k in range(4)], int(ok.sum()), n)
        out: dict[str, Any] = {"n": int(n), "n_quads": len(idx), "n_all_four": int(ok.sum()), "labels": labels,
                               "accuracy": {k: float(answered[same, j].mean()) for j, k in enumerate(("00", "10", "01", "11"))}}
        if n < 2:
            return {**out, "K": None}, {}
        self._identity([prompts[0][i] for i in rows], positions[rows])
        rps, M, P = self.read_points, len(self.read_points), len(labels)
        T = len(PATCHES)
        am = np.full((T, n, M), -1, dtype=np.int64)
        ratio = np.zeros((n, M, P), dtype=np.float32)
        for c0 in range(0, n, cfg.chunk):
            rc = rows[c0 : c0 + cfg.chunk]
            nc = len(rc)
            h = [backend.capture_sites([prompts[k][i] for i in rc], positions[rc]) for k in range(4)]
            dev, dtype = h[0].device, h[0].dtype
            masks = {}
            for g in GROUPS:
                mk = torch.zeros(nc, P, 1, dtype=torch.bool, device=dev)
                for r, i in enumerate(rc):
                    mk[r, token_groups(labels, quads[idx[i]][0].C)[g]] = True
                masks[g] = mk
            base_rows = [prompts[0][i] for i in rc]
            for mi, m in enumerate(rps):
                h00, h10, h01, h11 = (x[:, m].float() for x in h)
                inter = (h11 - h10 - h01 + h00).norm(dim=-1)
                ratio[c0 : c0 + nc, mi] = (inter / ((h10 - h00).norm(dim=-1) + (h01 - h00).norm(dim=-1) + 1e-6)).cpu().numpy()
                add = (h10 + h01 - h00).to(dtype)
                joint = h[3][:, m]
                vals = [joint, add] + [torch.where(masks[g], add, joint) for g in GROUPS]
                values = torch.cat(vals, dim=0)
                patch = SitePatch(np.full(T * nc, m), np.tile(positions[rc], (T, 1)), values)
                res = backend.run(base_rows * T, site_patch=patch, token_ids=tid)
                self._determinism(base_rows * T, patch)
                am[:, c0 : c0 + nc, mi] = res.query_argmax.reshape(T, nc)
                del values, vals, add, joint
            del h
            log.info("%s: %d of %d bases patched", pair, min(c0 + nc, n), n)
        Yt = tid[Y[rows]]
        order = {"11": 3, "10": 1, "01": 2, "00": 0}
        cat = np.full(am.shape, len(CATS) - 1, dtype=np.int8)
        for ci in reversed(range(len(CATS) - 1)):  # where two prompts share an answer, the earlier category wins
            cat[am == Yt[None, :, order[CATS[ci]]][:, :, None]] = ci
        seed = derive_seed(cfg.seed, "cond", "pairs", pair, "bootstrap")
        rates = {name: {c: bootstrap_rate(cat[ti] == ci, np.random.default_rng(seed + 10 * ti + ci), cfg.n_boot, cfg.alpha)
                        for ci, c in enumerate(CATS)} for ti, name in enumerate(PATCHES)}
        K = conversion_depth(rates["additive"]["11"]["lo"], rates["additive"]["11"]["hi"], rps, cfg.sufficient)
        check0 = float((cat[1, :, 0] == cat[0, :, 0]).mean()) if rps[0] == 0 else None
        log.info("%s: the additive patch gives 11's answer at read points %s: %s; K %s (held at the first: %s); additive = "
                 "joint at read point 0 on %s of the bases", pair, rps[::4],
                 [round(float(x), 2) for x in rates["additive"]["11"]["mean"][::4]], K["K"], K["holds_at_first"], check0)
        out.update({"rates": rates, **K, "additive_equals_joint_at_0": check0, "ratio_mean": ratio.mean(axis=0),
                    "examples": [[q.text() for q in quads[idx[i]]] for i in rows]})
        return out, {"argmax": am.astype(np.int32), "rows": np.asarray(idx)[rows], "ratio": ratio}

    # ------------------------------------------------------------------ #

    def _probe(self) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        bases, demos = self._bases("probe", cfg.n_probe)
        prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        positions, labels, ok_b = self._aligned(prompts, bases)
        correct, _ = self._answered(prompts, bases)
        keep = np.flatnonzero(ok_b & correct) if cfg.gate else np.flatnonzero(ok_b)
        cond_last = max(j for j, lab in enumerate(labels) if lab in ("<g>", "<h>", "<k>", "<cmp>"))
        sites = {"Cd_last": cond_last, "E": next(j for j, lab in enumerate(labels) if lab.strip() == "="), "A": len(labels) - 1}
        log.info("probe: %d bases, %d kept (accuracy %.3f); sites %s", len(bases), len(keep), float(correct.mean()),
                 {k: labels[j] for k, j in sites.items()})
        feats = {s: [] for s in sites}
        for c0 in range(0, len(keep), cfg.chunk * 8):
            rc = keep[c0 : c0 + cfg.chunk * 8]
            st = backend.capture_sites([prompts[i] for i in rc], positions[rc][:, list(sites.values())])  # (n, L+1, 3, d)
            for si, s in enumerate(sites):
                feats[s].append(st[:, :, si].float().cpu().numpy())
            del st
        Yv = np.array([bases[i].Y for i in keep])
        Cv = np.array([bases[i].C for i in keep])
        Un = np.array([bases[i].other for i in keep])
        specs = {"C": (Cv, np.zeros(len(keep), dtype=np.int64), np.ones((len(keep), 2), dtype=bool)),
                 "C|Y": (Cv, Yv, np.ones((len(keep), 2), dtype=bool)),
                 "Sun|Y": (Un, Yv, np.arange(10)[None, :] != Yv[:, None])}
        out: dict[str, Any] = {"labels": labels, "sites": {k: labels[j] for k, j in sites.items()}, "n": len(keep),
                               "accuracy": float(correct.mean()), "probes": {}}
        for s in sites:
            x = np.concatenate(feats[s], axis=0).transpose(1, 0, 2)  # (L+1, n, d)
            feats[s] = None
            for name in PROBE_TARGETS:
                y, cond, allowed = specs[name]
                t0 = time.time()
                r = conditional_probe(x[self.read_points], y, cond, allowed, n_perm=cfg.n_perm, alpha=cfg.alpha,
                                      seed=derive_seed(cfg.seed, "cond", "probe", s, name), device=backend.device)
                out["probes"][f"{s}/{name}"] = {k: v for k, v in r.items() if k != "null"}
                log.info("probe %s/%s: %.0fs; carried at read points %s (max excess %.3f)", s, name, time.time() - t0,
                         [self.read_points[i] for i in np.flatnonzero(r["carried"])], float(np.max(r["excess"])))
            del x
        return out


def run_cond(cfg: CondConfig, part: str, run_id: str | None = None, config_path: str | None = None) -> Path:
    return Conditional(cfg, part, run_id=run_id, config_path=config_path).run()
