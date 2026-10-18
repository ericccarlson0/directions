"""Distributed alignment search at the answer token, scored by Cause and Isolate (docs/DECISIONS.md D48), or at a set
of query tokens jointly (D53: one subspace of their concatenated states; the prompt is cached up to the first
swapped token and the rest rerun on every pass).

A k-dimensional subspace (orthonormal basis B, k x d) of the answer token's residual stream at one read point is
swapped from a source run into a base run, h <- h + ((h_source - h) B^T) B, and the answer is read. The subspace is
fitted so that the answer follows a hypothesis about one variable (S1, S2, Y or the digit a): its Cause score is the
share of held-out pairs where it does; its Isolate score the share of pairs whose source differs in other things
only, where the answer must stay. Fits run on the last prompt token alone: a swap there changes nothing at earlier
positions, so their keys and values are computed once per batch and every pass reruns one token.

    uv run directions das --config configs/das_qwen3_1.7b.yaml --run-id das_qwen3_1.7b_seed20260916
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import DasConfig
from .determinism import compare_arrays
from .model import ModelBackend, environment_metadata, set_torch_determinism
from .probes import fold_ids, ridge_cv_accuracy
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .varmap import DIGITS, TEMPLATES, Expr, all_exprs, bootstrap_rate, query_tokens, render
from .varprobe import _jsonable

VARIABLES = ("S1", "S2", "Y", "a")
# the Isolate sources of each variable: the answer must stay at the base's when the variable's subspace is swapped
ISOLATE = {"S1": ("only_S2", "same_S1"), "S2": ("only_S1", "same_S2"), "Y": ("same_Y",), "a": ("only_S2", "only_b")}
SOURCE_KINDS = ("cause", "only_S1", "only_S2", "same_S1", "same_S2", "only_b", "same_Y")
N_CLASSES = {"S1": 19, "S2": 19, "S3": 19, "T": 19, "U": 19, "Y": 10, "a": 10, "C": 2}
# three brackets (D51): the sums, the merge intermediates T = S1 - S2 and U = S2 + S3, the answer and the digit a
TASK_VARIABLES = {"two_lane": VARIABLES, "three_lane": ("S1", "S2", "S3", "T", "U", "Y", "a")}
TASK_ISOLATE = {"two_lane": ISOLATE, "three_lane": {
    "S1": ("only_S2", "only_S3", "same_S1"), "S2": ("only_S1", "only_S3", "same_S2"), "S3": ("only_S1", "only_S2", "same_S3"),
    "T": ("only_S3", "same_T"), "U": ("only_S1", "same_U"), "Y": ("same_Y",), "a": ("only_S2", "only_b")}}
TASK_SOURCE_KINDS = {"two_lane": SOURCE_KINDS, "three_lane": ("cause", "only_S1", "only_S2", "only_S3", "same_S1", "same_S2",
                                                              "same_S3", "same_T", "same_U", "only_b", "same_Y")}
# max (D55): the sums, the comparison bit C (S1 > S2), the answer and the digit a. No one source tells all their
# hypotheses apart (max's predictions coincide), so each variable has a Cause source of its own.
TASK_VARIABLES["max"] = ("S1", "S2", "C", "Y", "a")
TASK_ISOLATE["max"] = {"S1": ("only_S2", "same_S1"), "S2": ("only_S1", "same_S2"), "C": ("same_C",), "Y": ("same_Y",),
                       "a": ("only_S2", "only_b")}
TASK_CAUSE = {"max": {v: f"cause_{v}" for v in TASK_VARIABLES["max"]}}
TASK_SOURCE_KINDS["max"] = tuple(TASK_CAUSE["max"].values()) + ("only_S1", "only_S2", "same_S1", "same_S2", "same_C",
                                                                  "same_Y", "only_b")


def cause_kind(task: str, var: str) -> str:
    """The source kind a variable's Cause is measured on: one shared source (D48, D51), or its own (max, D55)."""
    return TASK_CAUSE.get(task, {}).get(var, "cause")


# --------------------------------------------------------------------------- #
# The pairs
# --------------------------------------------------------------------------- #


def digits_of(exprs: Sequence[Expr]) -> np.ndarray:
    if exprs and exprs[0].e is not None:
        return np.array([[e.a, e.b, e.c, e.d, e.e, e.f] for e in exprs], dtype=np.int64)
    return np.array([[e.a, e.b, e.c, e.d] for e in exprs], dtype=np.int64)


def source_mask(kind: str, base: Expr, D: np.ndarray) -> np.ndarray:
    """Which candidate expressions (digits ``D``, (N, 4); three brackets (N, 6)) are valid sources of ``kind`` for
    ``base`` (D48, D51)."""
    if D.shape[1] == 6:
        return _source_mask3(kind, base, D)
    if base.op == "max":
        return _source_mask_max(kind, base, D)
    a, b, c, d = D.T
    S1, S2 = a + b, c + d
    Y = S1 - S2
    in_range = (Y >= 0) & (Y <= 9)
    if kind == "cause":
        t = np.stack([np.full(len(D), base.Y), S1 - base.S2, base.S1 - S2, Y, a + base.b - base.S2])
        ok = (a != base.a) & (b != base.b) & (S1 != base.S1) & (S2 != base.S2) & ((t >= 0) & (t <= 9)).all(axis=0)
        distinct = np.ones(len(D), dtype=bool)
        for i in range(5):
            for j in range(i + 1, 5):
                distinct &= t[i] != t[j]
        return ok & distinct
    same12 = (a == base.a) & (b == base.b)
    same34 = (c == base.c) & (d == base.d)
    if kind == "only_S1":
        return same34 & (S1 != base.S1) & in_range
    if kind == "only_S2":
        return same12 & (S2 != base.S2) & in_range
    if kind == "same_S1":
        return same34 & ~same12 & (S1 == base.S1)
    if kind == "same_S2":
        return same12 & ~same34 & (S2 == base.S2)
    if kind == "only_b":
        return (a == base.a) & same34 & (b != base.b) & in_range
    if kind == "same_Y":
        return (S1 != base.S1) & (S2 != base.S2) & (Y == base.Y)
    raise ValueError(f"unknown source kind {kind!r}")


def _source_mask3(kind: str, base: Expr, D: np.ndarray) -> np.ndarray:
    """Three brackets (D51): one Cause source per base (a and b changed, every sum changed, the eight answers that
    tell the hypotheses apart single digits and pairwise distinct) and the Isolate sources."""
    a, b, c, d, e, f = D.T
    S1, S2, S3 = a + b, c + d, e + f
    T, U, Y = S1 - S2, S2 + S3, S1 - S2 - S3
    in_range = (Y >= 0) & (Y <= 9)
    same1 = (a == base.a) & (b == base.b)
    same2 = (c == base.c) & (d == base.d)
    same3 = (e == base.e) & (f == base.f)
    if kind == "cause":
        t = np.stack([np.full(len(D), base.Y), S1 - base.S2 - base.S3, base.S1 - S2 - base.S3, base.S1 - base.S2 - S3,
                      T - base.S3, base.S1 - U, Y, a + base.b - base.S2 - base.S3])
        ok = ((a != base.a) & (b != base.b) & (S1 != base.S1) & (S2 != base.S2) & (S3 != base.S3)
              & ((t >= 0) & (t <= 9)).all(axis=0))
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                ok &= t[i] != t[j]
        return ok
    if kind == "only_S1":
        return same2 & same3 & (S1 != base.S1) & in_range
    if kind == "only_S2":
        return same1 & same3 & (S2 != base.S2) & in_range
    if kind == "only_S3":
        return same1 & same2 & (S3 != base.S3) & in_range
    if kind == "same_S1":
        return same2 & same3 & ~same1 & (S1 == base.S1)
    if kind == "same_S2":
        return same1 & same3 & ~same2 & (S2 == base.S2)
    if kind == "same_S3":
        return same1 & same2 & ~same3 & (S3 == base.S3)
    if kind == "same_T":
        return same3 & (S1 != base.S1) & (S2 != base.S2) & (T == base.T)
    if kind == "same_U":
        return same1 & (S2 != base.S2) & (S3 != base.S3) & (U == base.U)
    if kind == "only_b":
        return (a == base.a) & same2 & same3 & (b != base.b) & in_range
    if kind == "same_Y":
        return (S1 != base.S1) & (S2 != base.S2) & (S3 != base.S3) & (Y == base.Y)
    raise ValueError(f"unknown source kind {kind!r}")


def _max_targets(base: Expr, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
    """Max (D55): the answer a swap of each variable from sources with digits a, b, c, d predicts."""
    S1, S2 = a + b, c + d
    return {"S1": np.maximum(S1, base.S2), "S2": np.maximum(base.S1, S2),
            "C": np.where(S1 > S2, base.S1, base.S2), "Y": np.maximum(S1, S2), "a": np.maximum(a + base.b, base.S2)}


def _source_mask_max(kind: str, base: Expr, D: np.ndarray) -> np.ndarray:
    """Max (D55). A variable's Cause source: its swap's answer is a digit that differs from the base answer and from
    the answers of the S1, S2 and C swaps from the same source (the other sub-results; a subspace holding the whole
    answer is caught by Isolate). Isolate sources: the source's own answer differs from the base's (so that a
    subspace carrying more than the variable fails), except the same-value ones. Max leaves no room above a 9, so
    some bases have no source of some kind (sample_splits keeps them, without it)."""
    a, b, c, d = D.T
    S1, S2 = a + b, c + d
    Y, C = np.maximum(S1, S2), (S1 > S2).astype(np.int64)
    t = _max_targets(base, a, b, c, d)
    same12 = (a == base.a) & (b == base.b)
    same34 = (c == base.c) & (d == base.d)
    moves = Y != base.Y
    if kind.startswith("cause_"):
        v = kind[len("cause_"):]
        ok = (t[v] != base.Y) & (t[v] >= 0) & (t[v] <= 9)
        for u in ("S1", "S2", "C"):
            if u != v:
                ok &= t[v] != t[u]
        return ok
    if kind == "only_S1":
        return same34 & (S1 != base.S1) & moves
    if kind == "only_S2":
        return same12 & (S2 != base.S2) & moves
    if kind == "same_S1":
        return same34 & ~same12 & (S1 == base.S1)
    if kind == "same_S2":
        return same12 & ~same34 & (S2 == base.S2)
    if kind == "only_b":
        return (a == base.a) & same34 & (b != base.b) & moves
    if kind == "same_C":
        return (S1 != base.S1) & (S2 != base.S2) & (C == base.C) & moves
    if kind == "same_Y":
        return (S1 != base.S1) & (S2 != base.S2) & (Y == base.Y)
    raise ValueError(f"unknown source kind {kind!r}")


def targets(var: str, base: Expr, src: Expr) -> int:
    """The answer a Cause swap of ``var`` predicts (D48; three brackets D51, where S3 = 0 on two brackets; max D55)."""
    if base.op == "max":
        return int(_max_targets(base, *(np.array([x]) for x in (src.a, src.b, src.c, src.d)))[var][0])
    return {"S1": src.S1 - base.S2 - base.S3, "S2": base.S1 - src.S2 - base.S3, "S3": base.S1 - base.S2 - src.S3,
            "T": src.T - base.S3, "U": base.S1 - src.U, "Y": src.Y, "a": src.a + base.b - base.S2 - base.S3}[var]


def variable_value(var: str, e: Expr) -> int:
    if var == "C":
        return e.C
    return {"S1": e.S1, "S2": e.S2, "S3": e.S3, "T": e.T, "U": e.U, "Y": e.Y, "a": e.a}[var]


@dataclass
class Split:
    bases: list[Expr]
    demos: list[list[Expr]]
    sources: dict[str, list[Expr]] = field(default_factory=dict)  # kind -> one per base


def sample_splits(seed: int, sizes: dict[str, int], n_shots: int, n_demo_pool: int, kinds: dict[str, Sequence[str]],
                  rejections: RejectionLog | None = None, task: str = "two_lane") -> tuple[dict[str, Split], dict[str, Any]]:
    """Bases split by expression (train, val, test; disjoint), each with its demonstrations and one source per kind.
    On max (D55) a kind without a valid source is left None for that base, and a base is skipped only when it has
    no Cause source at all."""
    exprs = all_exprs(task)
    order = rng_for(seed, "das", "split").permutation(len(exprs))
    pool = [exprs[i] for i in order[:n_demo_pool]]
    rest = [exprs[i] for i in order[n_demo_pool:]]
    D = digits_of(rest)
    splits: dict[str, Split] = {}
    cursor, skipped = 0, 0
    for name, n in sizes.items():
        sp = Split([], [])
        sp.sources = {k: [] for k in kinds[name]}
        while len(sp.bases) < n and cursor < len(rest):
            base = rest[cursor]
            cursor += 1
            masks = {k: source_mask(k, base, D) for k in kinds[name]}
            optional = task in TASK_CAUSE
            if (not all(m.any() for m in masks.values()) if not optional
                    else not any(m.any() for k, m in masks.items() if k.startswith("cause"))):
                skipped += 1
                if rejections is not None:
                    rejections.add("source", name, "no valid source of some kind", {"base": base.text()})
                continue
            i = len(sp.bases)
            for k, m in masks.items():
                cand = np.flatnonzero(m)
                sp.sources[k].append(rest[int(cand[rng_for(seed, "das", name, i, k).integers(len(cand))])] if len(cand) else None)
            r = rng_for(seed, "das", "demos", name, i)
            sp.demos.append([pool[j] for j in r.choice(len(pool), size=n_shots, replace=False)])
            sp.bases.append(base)
        splits[name] = sp
    return splits, {"n_expressions": len(exprs), "skipped_without_source": skipped,
                    **{f"n_{k}": len(v.bases) for k, v in splits.items()}}


# --------------------------------------------------------------------------- #
# The engine: one token, many swaps
# --------------------------------------------------------------------------- #


def _math_attention():
    """The plain (deterministic) attention kernel for the single-token passes."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:  # older torch: nothing to select
        return contextlib.nullcontext()


class LastToken:
    """A batch of prompts with every token but the last ``n_tail`` run once (keys and values cached), and passes of
    those last tokens alone, optionally with a subspace swap at one read point. The swap acts on the tail tokens
    ``swap_idx`` jointly: their states are concatenated, ``(P d)``, and the subspace lives in that space (D53); or,
    with ``shared``, one subspace of the residual stream, ``(k, d)``, is swapped at each of them (D54). The default,
    one tail token swapped alone, is the answer token of D48."""

    def __init__(self, backend: ModelBackend, prompts: Sequence[Prompt], n_tail: int = 1,
                 swap_idx: Sequence[int] | None = None, shared: bool = False) -> None:
        self.backend = backend
        ids = [backend._encode_prompt(p)[0] for p in prompts]
        if not ids:
            raise ValueError("an empty batch")
        if len({len(x) for x in ids}) != 1:
            raise ValueError("the prompts of a batch must have the same length (no padding on this path)")
        self.ids = torch.as_tensor(np.array(ids, dtype=np.int64), device=backend.device)
        self.T = self.ids.shape[1]
        if not 1 <= n_tail < self.T:
            raise ValueError(f"n_tail {n_tail} outside 1..{self.T - 1}")
        self.n_tail = n_tail
        self.swap_idx = list(swap_idx) if swap_idx is not None else [n_tail - 1]
        if not self.swap_idx or min(self.swap_idx) < 0 or max(self.swap_idx) >= n_tail:
            raise ValueError("swap_idx must index the tail tokens")
        self.shared = shared
        with torch.no_grad(), _math_attention():
            out = backend.model.model(input_ids=self.ids[:, :-n_tail], use_cache=True)
        self.cache = out.past_key_values
        # every pass appends the last token to the cache; restoring each layer's state (the prefix tensors, with no
        # autograd history, and any counters) afterwards keeps one pass's graph out of the next
        self._saved = [dict(layer.__dict__) for layer in self.cache.layers]

    def logits(self, read_point: int | None = None, values: torch.Tensor | None = None,
               basis: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token logits (B, V) float32 at the last prompt token. With ``read_point``, the swapped tokens' states
        there, concatenated (``h``, ``(B, P d)``), are replaced by ``h + ((values - h) B^T) B`` (``basis``
        ``(k, P d)``), or by ``values`` when ``basis`` is None. With ``shared``, ``basis`` is ``(k, d)`` and the
        same swap acts on each token's state."""
        backend = self.backend
        L = backend.n_layers
        handle = None
        if read_point is not None:
            idx = self.swap_idx

            def swap(hidden: torch.Tensor) -> torch.Tensor:
                if idx == [hidden.shape[1] - 1]:  # the answer token alone (D48): unchanged arithmetic
                    h = hidden[:, -1].float()
                    v = values.to(h.device).float()
                    new = v if basis is None else h + ((v - h) @ basis.T) @ basis
                    return torch.cat([hidden[:, :-1], new[:, None].to(hidden.dtype)], dim=1)
                h = hidden[:, idx].float()  # (B, P, d)
                shape = h.shape
                v = values.to(h.device).float().reshape(shape)
                if basis is None:
                    new = v
                elif self.shared:  # one subspace of the residual stream at every swapped token (D54)
                    new = h + ((v - h) @ basis.T) @ basis
                else:  # one subspace of the concatenated states (D53)
                    hf, vf = h.reshape(shape[0], -1), v.reshape(shape[0], -1)
                    new = (hf + ((vf - hf) @ basis.T) @ basis).reshape(shape)
                out = hidden.clone()
                out[:, idx] = new.to(hidden.dtype)
                return out

            if read_point < L:
                def pre(module, args, kwargs):
                    if args:
                        return (swap(args[0]),) + tuple(args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = swap(kwargs["hidden_states"])
                    return args, kwargs

                handle = backend.blocks[read_point].register_forward_pre_hook(pre, with_kwargs=True)
            else:
                def post(module, args, output):
                    if isinstance(output, tuple):
                        return (swap(output[0]),) + tuple(output[1:])
                    return swap(output)

                handle = backend.blocks[L - 1].register_forward_hook(post)
        try:
            with _math_attention():
                out = backend.model.model(input_ids=self.ids[:, -self.n_tail :], past_key_values=self.cache, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()
            for layer, saved in zip(self.cache.layers, self._saved):
                layer.__dict__.clear()
                layer.__dict__.update(saved)
        return backend._lm_logits(out.last_hidden_state[:, -1]).float()


def _take(x: torch.Tensor, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """Rows of a state tensor, on ``device`` (the joint states live on the CPU in the model's dtype, D53)."""
    return x[torch.as_tensor(rows, device=x.device)].to(device)


class Fit:
    """One subspace: a free d x k matrix whose QR gives the basis, with its own optimizer."""

    def __init__(self, site: int, var: str, k: int, d: int, device: torch.device, seed: int, lr: float,
                 shuffled: bool = False) -> None:
        self.site, self.var, self.k, self.shuffled = site, var, k, shuffled
        g = torch.Generator(device="cpu").manual_seed(int(seed % (2**63 - 1)))
        self.W = torch.nn.Parameter(torch.randn(d, k, generator=g, dtype=torch.float32).to(device))
        self.opt = torch.optim.Adam([self.W], lr=lr)
        self.best: torch.Tensor = self.W.detach().clone()
        self.best_val = -1.0
        self.history: list[float] = []

    @property
    def key(self) -> str:
        return f"{self.site}/{self.var}/{self.k}" + ("/shuffled" if self.shuffled else "")

    def basis(self, W: torch.Tensor | None = None) -> torch.Tensor:
        return torch.linalg.qr(self.W if W is None else W).Q.T  # (k, d)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class DAS:
    def __init__(self, cfg: DasConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_das_{slug}_seed{cfg.seed}"
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

    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        cfg, log = self.cfg, self.log
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        backend = self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        for p in backend.model.parameters():
            p.requires_grad_(False)
        L = backend.n_layers
        if max(cfg.sites) > L or min(cfg.sites) < 0 or (cfg.control_site is not None and cfg.control_site not in cfg.sites):
            raise ValueError(f"sites must lie in 0..{L} and include the control site")
        self.metadata["model"] = backend.metadata()
        self.tid = np.array([self._single_token(str(k)) for k in range(10)], dtype=np.int64)
        t0 = time.time()
        self.isolate = TASK_ISOLATE[cfg.task]
        self.cause_of = {v: cause_kind(cfg.task, v) for v in cfg.variables}  # a variable's Cause source kind (D55)
        causes = tuple(dict.fromkeys(self.cause_of.values()))
        kinds = {"train": causes, "val": causes, "test": tuple(k for k in TASK_SOURCE_KINDS[cfg.task]
                                                                 if not k.startswith("cause") or k in causes)}
        splits, info = sample_splits(cfg.seed, {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}, cfg.n_shots,
                                     cfg.n_demo_pool, kinds, self.rejections, cfg.task)
        self.metadata["sampling"] = info
        self._geometry(splits)
        data = {name: self._prepare(name, sp) for name, sp in splits.items()}
        self.metadata.setdefault("timings_seconds", {})["prepare"] = round(time.time() - t0, 1)
        for name, dd in data.items():
            log.info("%s: %d bases, %d kept for the Cause pairs (base answered in both passes, source answered)", name,
                     len(dd["bases"]), int(dd["keep"].sum()))
            if len(set(self.cause_of.values())) > 1:  # a Cause source per variable (max, D55)
                log.info("%s: kept per variable %s", name, {v: int(x.sum()) for v, x in dd["keep_by_var"].items()})

        # the concatenated states of the swapped tokens (D53), or one token's state for a subspace shared by them (D54)
        D = backend.hidden_size * (1 if cfg.shared_basis else len(self.swap_idx))
        fits = [Fit(s, v, k, D, backend.device, derive_seed(cfg.seed, "das", "fit", s, v, k), cfg.lr)
                for s in cfg.sites for v in cfg.variables for k in cfg.ks]
        fits += [Fit(s, v, cfg.null_k, D, backend.device, derive_seed(cfg.seed, "das", "null", s, v), cfg.lr,
                     shuffled=True) for s in cfg.sites for v in cfg.variables]
        t0 = time.time()
        init_test = self._evaluate(data["test"], fits, cause_only=True, use_best=False)  # the untrained subspaces
        if cfg.determinism_check:
            self._determinism_check(data, fits[len(fits) // 2])
        self._train(data, fits)
        self.metadata["timings_seconds"]["train"] = round(time.time() - t0, 1)
        t0 = time.time()
        test = self._evaluate(data["test"], fits, cause_only=False, use_best=True)
        if cfg.determinism_check:
            again = self._evaluate(data["test"], [fits[0]], cause_only=False, use_best=True)
            cmp = compare_arrays({k: np.asarray(v) for k, v in test["pred"][fits[0].key].items()},
                                 {k: np.asarray(v) for k, v in again["pred"][fits[0].key].items()})
            self.metadata.setdefault("determinism_check", {})["evaluation"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("determinism check failed: a repeated evaluation differs")
        self.metadata["timings_seconds"]["evaluate"] = round(time.time() - t0, 1)
        result = self._read(data, fits, test, init_test)
        write_json(self.root / "core" / "das.json", _jsonable(result))
        np.savez_compressed(self.root / "core" / "das_predictions.npz",
                            **{f"{k}__{fk}": np.asarray(v) for fk, preds in test["pred"].items() for k, v in preds.items()})
        # the kept (best-validation) basis of every fit, (k, d) with orthonormal rows
        np.savez_compressed(self.root / "core" / "das_bases.npz",
                            **{f.key: f.basis(f.best).detach().cpu().numpy().astype(np.float32) for f in fits})
        if cfg.figures:
            try:
                plot(result, self.root / "figures" / "das.png",
                     cfg.model.name.split("/")[-1] if cfg.model.backend == "hf" else f"toy {cfg.model.toy.family}")
            except Exception as exc:  # a figure never fails a run
                log.warning("figure failed: %s", exc)
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        log.info("done -> %s", self.root)
        return self.root

    def _single_token(self, text: str) -> int:
        enc = self.backend.tokenizer.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{text!r} is not a single token on this tokenizer: {enc}")
        return enc[0]

    # ------------------------------------------------------------------ #

    def _geometry(self, splits: dict[str, Split]) -> None:
        """The query's tokens and the swapped ones (D53): the configured set, without the tokens before the first
        operand digit (identical in a base and its source, so a swap there is a no-op). Every prompt ends with the
        same query tokens, checked on every base."""
        cfg = self.cfg
        sp0 = next(iter(splits.values()))
        p0 = render(sp0.demos[0], sp0.bases[0])
        _, _, labels = query_tokens(self.backend, p0, sp0.bases[0])
        Q = len(labels)
        if cfg.positions == "answer":
            J = [Q - 1]
        else:
            from .posset import set_members

            first = min(j for j, lab in enumerate(labels) if lab in DIGITS)
            J = [j for j in set_members(labels)[cfg.positions] if j > first]
        for sp in splits.values():
            for i, b in enumerate(sp.bases):
                _, _, lab = query_tokens(self.backend, render(sp.demos[i], b), b)
                if lab != labels:
                    raise ValueError(f"query tokens differ between prompts: {lab} vs {labels}")
        self.query_labels, self.Q = labels, Q
        self.n_tail = Q - min(J)
        self.swap_idx = [j - min(J) for j in J]
        self.swap_q = J  # query-token indices of the swapped tokens
        self.metadata["swap"] = {"positions": cfg.positions, "labels": labels, "tokens": J,
                                 "token_labels": [labels[j] for j in J], "n_tail": self.n_tail,
                                 "shared_basis": cfg.shared_basis}
        self.log.info("swapped tokens %s of the query %s (rerun from query token %d: %d tokens per pass)",
                      [labels[j] for j in J], labels, min(J), self.n_tail)

    def _tail(self, prompts: Sequence[Prompt]) -> LastToken:
        return LastToken(self.backend, prompts, self.n_tail, self.swap_idx, shared=self.cfg.shared_basis)

    def _states(self, prompts: list[Prompt], sites: Sequence[int]) -> tuple[dict[int, torch.Tensor], np.ndarray]:
        """The swapped tokens' states at the sites and the model's answer, in chunks. The answer token alone (D48):
        float32 on the device. A set of tokens (D53): their states concatenated, ``(N, P d)``, kept in the model's
        dtype on the CPU (moved per batch)."""
        backend = self.backend
        if self.cfg.positions != "answer":
            parts_j: dict[int, list[torch.Tensor]] = {s: [] for s in sites}
            am_j = []
            for start in range(0, len(prompts), 512):
                chunk = prompts[start : start + 512]
                am_j.append(backend.run(chunk, token_ids=self.tid).query_argmax)
                T = np.array([len(backend._encode_prompt(p)[0]) for p in chunk])
                pos = T[:, None] - self.Q + np.asarray(self.swap_q)[None, :]
                st = backend.capture_sites(chunk, pos)  # (n, L+1, P, d)
                for s in sites:
                    parts_j[s].append(st[:, s].reshape(len(chunk), -1).cpu())
                del st
            return {s: torch.cat(v) for s, v in parts_j.items()}, np.concatenate(am_j)
        parts: dict[int, list[np.ndarray]] = {s: [] for s in sites}
        am = []
        for start in range(0, len(prompts), 512):
            r = backend.run(prompts[start : start + 512], capture=True, token_ids=self.tid)
            for s in sites:
                parts[s].append(r.residuals[s])
            am.append(r.query_argmax)
        return ({s: torch.as_tensor(np.concatenate(v), device=backend.device) for s, v in parts.items()},
                np.concatenate(am))

    def _prepare(self, name: str, sp: Split) -> dict[str, Any]:
        cfg = self.cfg
        prompts = [render(sp.demos[i], b) for i, b in enumerate(sp.bases)]
        base_states, base_am = self._states(prompts, cfg.sites)
        Y = np.array([b.Y for b in sp.bases])
        ok_full = base_am == self.tid[Y]
        # the base answered in the single-token pass too (the gate in the geometry the swaps run in)
        ok_last = np.zeros(len(prompts), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(prompts), self.cfg.batch_size):
                lt = self._tail(prompts[start : start + self.cfg.batch_size])
                ok_last[start : start + len(lt.ids)] = (lt.logits().argmax(dim=1).cpu().numpy() == self.tid[Y[start : start + len(lt.ids)]])
        if (ok_full != ok_last).any():
            self.rejections.add("noop", name, "the single-token pass and the full pass disagree on the base answer",
                                {"n": int((ok_full != ok_last).sum())})
        src_states, src_ok = {}, {}
        for kind, srcs in sp.sources.items():
            # a base without a source of this kind (max, D55) runs its own prompt in its place, never used
            present = np.array([x is not None for x in srcs])
            sp_prompts = [render(sp.demos[i], x if x is not None else sp.bases[i]) for i, x in enumerate(srcs)]
            st, am = self._states(sp_prompts, cfg.sites)
            ys = np.array([x.Y if x is not None else 0 for x in srcs])
            src_states[kind], src_ok[kind] = st, present & (am == self.tid[ys])
        if not cfg.gate:  # the toy smoke: every base and present source enters
            ok_full = ok_last = np.ones(len(prompts), dtype=bool)
            src_ok = {k: np.array([x is not None for x in sp.sources[k]]) for k in src_ok}
        keep_by_var = {v: ok_full & ok_last & src_ok[self.cause_of[v]] for v in cfg.variables}
        keep = np.logical_or.reduce(list(keep_by_var.values()))
        tgt = {}
        for v in cfg.variables:
            srcs = sp.sources[self.cause_of[v]]
            tgt[v] = np.array([targets(v, b, srcs[i]) if srcs[i] is not None else 0 for i, b in enumerate(sp.bases)])
        return {"bases": sp.bases, "prompts": prompts, "Y": Y, "keep": keep, "keep_by_var": keep_by_var,
                "base_ok": ok_full & ok_last, "base_states": base_states, "src_states": src_states, "src_ok": src_ok,
                "targets": tgt}

    # ------------------------------------------------------------------ #

    def _batches(self, idx: np.ndarray) -> list[np.ndarray]:
        bs = self.cfg.batch_size
        return [idx[i : i + bs] for i in range(0, len(idx), bs)]

    def _train(self, data: dict[str, Any], fits: list[Fit]) -> None:
        cfg, log = self.cfg, self.log
        tr = data["train"]
        idx = np.flatnonzero(tr["keep"])
        tid = torch.as_tensor(self.tid, device=self.backend.device)
        kv = {v: np.flatnonzero(tr["keep_by_var"][v]) for v in cfg.variables}
        shuffled = {v: rng_for(cfg.seed, "das", "shuffle", v).permutation(len(kv[v])) for v in cfg.variables}
        target_tok = {}  # per fit, the target token of every training base (shuffled among the kept bases for a null)
        mask = {v: torch.as_tensor(tr["keep_by_var"][v], device=self.backend.device) for v in cfg.variables}
        shared = all(np.array_equal(tr["keep_by_var"][v], tr["keep"]) for v in cfg.variables)  # one Cause source (D48-D54)
        for f in fits:
            t = tr["targets"][f.var].copy()
            if f.shuffled:
                t[kv[f.var]] = t[kv[f.var]][shuffled[f.var]]
            target_tok[f.key] = tid[torch.as_tensor(t, device=tid.device)]
        for epoch in range(cfg.epochs):
            order = idx[rng_for(cfg.seed, "das", "epoch", epoch).permutation(len(idx))]
            losses = {f.key: [] for f in fits}
            for b in self._batches(order):
                lt = self._tail([tr["prompts"][i] for i in b])
                bt = torch.as_tensor(b, device=self.backend.device)
                src = {(k, s): _take(tr["src_states"][k][s], b, self.backend.device)
                       for k in sorted(set(self.cause_of.values())) for s in cfg.sites}
                for f in fits:
                    m = None if shared else mask[f.var][bt]
                    if m is not None and not bool(m.any()):
                        continue
                    with torch.enable_grad():
                        logits = lt.logits(f.site, src[(self.cause_of[f.var], f.site)], f.basis())
                        y = target_tok[f.key][bt]
                        loss = (torch.nn.functional.cross_entropy(logits, y) if m is None
                                else torch.nn.functional.cross_entropy(logits[m], y[m]))
                        loss.backward()
                    f.opt.step()
                    f.opt.zero_grad(set_to_none=True)
                    losses[f.key].append(float(loss.detach()))
            val = self._evaluate(data["val"], fits, cause_only=True, use_best=False)
            for f in fits:
                c = float(np.mean(val["cause"][f.key])) if len(val["cause"][f.key]) else 0.0
                f.history.append(c)
                if c > f.best_val:
                    f.best_val, f.best = c, f.W.detach().clone()
            log.info("epoch %d: mean train loss %.3f; validation Cause, best fit per site: %s", epoch,
                     float(np.mean([np.mean(v) for v in losses.values()])),
                     {s: round(max(f.history[-1] for f in fits if f.site == s and not f.shuffled), 2) for s in cfg.sites})

    @torch.no_grad()
    def _evaluate(self, dd: dict[str, Any], fits: list[Fit], cause_only: bool, use_best: bool) -> dict[str, Any]:
        """Per fit: Cause (the answer is the hypothesis' answer) on the kept bases and, unless ``cause_only``, Isolate
        (the answer stays) on the bases whose Isolate source was answered too; predictions are kept per base."""
        tid = self.tid
        idx = np.flatnonzero(dd["keep"])
        bases_ok = dd["base_ok"]
        out: dict[str, Any] = {"cause": {f.key: [] for f in fits}, "isolate": {f.key: {} for f in fits},
                               "pred": {f.key: {} for f in fits}}
        bases_eval = np.flatnonzero(bases_ok) if not cause_only else idx
        preds: dict[str, dict[str, list]] = {f.key: {} for f in fits}
        for b in self._batches(bases_eval):
            lt = self._tail([dd["prompts"][i] for i in b])
            cache: dict[tuple[str, int], torch.Tensor] = {}
            for f in fits:
                basis = f.basis(f.best if use_best else None)
                # "cause" names the variable's own Cause source (one shared source before D55)
                kinds = [("cause", self.cause_of[f.var])] + ([] if cause_only else [(k, k) for k in self.isolate[f.var]])
                for label, kind in kinds:
                    if (kind, f.site) not in cache:
                        cache[(kind, f.site)] = _take(dd["src_states"][kind][f.site], b, self.backend.device)
                    am = lt.logits(f.site, cache[(kind, f.site)], basis).argmax(dim=1).cpu().numpy()
                    preds[f.key].setdefault(label, []).append(np.stack([b, am], axis=1))
        for f in fits:
            for kind, parts in preds[f.key].items():
                arr = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)
                rows, am = arr[:, 0], arr[:, 1]
                if kind == "cause":
                    sel = dd["keep_by_var"][f.var][rows]
                    hit = am[sel] == tid[dd["targets"][f.var][rows[sel]]]
                    out["cause"][f.key] = hit
                    # per-base hits, for paired comparisons between fits (D54's control rule)
                    out["pred"][f.key]["cause_hit"] = np.stack([rows[sel], hit.astype(np.int64)], axis=1)
                else:
                    sel = dd["src_ok"][kind][rows]
                    out["isolate"][f.key][kind] = am[sel] == tid[dd["Y"][rows[sel]]]
                out["pred"][f.key][kind] = arr
        return out

    def _determinism_check(self, data: dict[str, Any], fit: Fit) -> None:
        """One fit trained twice for two steps from the same start must come out bit-identical."""
        tr = data["train"]
        b = np.flatnonzero(tr["keep_by_var"][fit.var])[: self.cfg.batch_size]
        if len(b) == 0:
            raise RuntimeError("no training base passed the gate")
        lt = self._tail([tr["prompts"][i] for i in b])
        src = _take(tr["src_states"][self.cause_of[fit.var]][fit.site], b, self.backend.device)
        tid = torch.as_tensor(self.tid, device=self.backend.device)
        y = tid[torch.as_tensor(tr["targets"][fit.var][b], device=tid.device)]
        outs = []
        for _ in range(2):
            f = Fit(fit.site, fit.var, fit.k, fit.W.shape[0], self.backend.device, 12345, self.cfg.lr)
            for _step in range(2):
                with torch.enable_grad():
                    loss = torch.nn.functional.cross_entropy(lt.logits(f.site, src, f.basis()), y)
                    loss.backward()
                f.opt.step()
                f.opt.zero_grad(set_to_none=True)
            outs.append(f.W.detach().cpu().numpy())
        cmp = compare_arrays({"W": outs[0]}, {"W": outs[1]})
        self.metadata.setdefault("determinism_check", {})["training"] = {"identical": cmp["identical"]}
        if not cmp["identical"]:
            raise RuntimeError("determinism check failed: a repeated training run differs")

    # ------------------------------------------------------------------ #

    def _read(self, data: dict[str, Any], fits: list[Fit], test: dict[str, Any], init: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        rng = np.random.default_rng(derive_seed(cfg.seed, "das", "bootstrap"))

        def rate(x: np.ndarray) -> dict[str, float]:
            if len(x) == 0:
                return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
            r = bootstrap_rate(np.asarray(x, dtype=float), rng, cfg.n_boot, cfg.alpha)
            return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n": int(len(x))}

        cells: dict[str, Any] = {}
        by_key = {f.key: f for f in fits}
        te = data["test"]
        for s in cfg.sites:
            for v in cfg.variables:
                grid = {}
                for k in cfg.ks:
                    f = by_key[f"{s}/{v}/{k}"]
                    iso = {kind: rate(x) for kind, x in test["isolate"][f.key].items()}
                    grid[k] = {"val_cause": f.best_val, "val_history": f.history, "cause": rate(test["cause"][f.key]),
                               "isolate": iso, "isolate_mean": float(np.nanmean([x["mean"] for x in iso.values()])),
                               "isolate_lo": float(np.nanmin([x["lo"] for x in iso.values()])),
                               "untrained_cause": float(np.mean(init["cause"][f.key])) if len(init["cause"][f.key]) else None}
                best_val = max(grid[k]["val_cause"] for k in cfg.ks)
                kstar = min(k for k in cfg.ks if grid[k]["val_cause"] >= best_val - cfg.k_tolerance)
                g = grid[kstar]
                held = g["cause"]["lo"] >= cfg.cause_lower and g["isolate_lo"] >= cfg.isolate_lower
                null = by_key[f"{s}/{v}/{cfg.null_k}/shuffled"]
                cells[f"{s}/{v}"] = {
                    "site": s, "var": v, "k_star": kstar, "held": bool(held), "grid": grid,
                    "null_cause": rate(test["cause"][null.key]),
                    "projection_decodability": self._decodability(te, s, v, by_key[f"{s}/{v}/{kstar}"]),
                }
        sites = {}
        for s in cfg.sites:
            held = [v for v in cfg.variables if cells[f"{s}/{v}"]["held"]]
            sites[s] = {"held": held, "count": sum(v in held for v in ("S1", "S2", "S3", "T", "U", "Y")),
                        "sums_held": sum(v in held for v in ("S1", "S2", "S3")), "digits_in_use": "a" in held}
        self.log.info("held per site: %s", {s: x["held"] for s, x in sites.items()})
        clean = None if cfg.control_site is None else not sites[cfg.control_site]["held"]
        return {"task": cfg.task, "sites": cfg.sites, "control_site": cfg.control_site, "variables": list(cfg.variables), "ks": list(cfg.ks),
                "positions": cfg.positions, "shared_basis": cfg.shared_basis, "swap": self.metadata.get("swap"),
                "cells": cells, "per_site": sites, "control_clean": clean,
                "n": {name: {"bases": len(dd["bases"]), "kept": int(dd["keep"].sum()),
                             "kept_by_var": {v: int(x.sum()) for v, x in dd["keep_by_var"].items()}}
                      for name, dd in data.items()},
                "thresholds": {"cause_lower": cfg.cause_lower, "isolate_lower": cfg.isolate_lower, "k_tolerance": cfg.k_tolerance}}

    def _decodability(self, dd: dict[str, Any], site: int, var: str, fit: Fit) -> dict[str, float]:
        """Cross-validated linear decodability of the variable from the natural test states projected onto the fit."""
        idx = np.flatnonzero(dd["base_ok"])
        if len(idx) < 10:
            return {"accuracy": float("nan"), "chance": float("nan")}
        with torch.no_grad():
            B = fit.basis(fit.best)
            st = _take(dd["base_states"][site], idx, B.device).float()
            if self.cfg.shared_basis:  # the projection at each swapped token, side by side (D54)
                st = st.reshape(len(idx), len(self.swap_idx), -1)
            x = (st @ B.T).reshape(len(idx), -1).double()
        y = np.array([variable_value(var, dd["bases"][i]) for i in idx])
        C = N_CLASSES[var]
        allowed = torch.ones(len(idx), C, dtype=torch.bool, device=x.device)
        acc, _ = ridge_cv_accuracy(x, torch.as_tensor(y, device=x.device)[None], allowed,
                                   fold_ids(len(idx), 5, rng_for(self.cfg.seed, "das", "decode", site, var)), self.cfg.lambdas)
        return {"accuracy": float(acc[0]), "chance": float(np.bincount(y, minlength=C).max() / len(y))}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2 = "#0b0b0b", "#52514e"


def plot(result: dict[str, Any], path: Path, model_name: str) -> None:
    """Left: test Cause at k* per site (rows) and variable (columns), with Isolate and k* in each cell; held cells
    outlined. Right: test Cause against k, one panel per variable, one line per site."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    sites, variables, ks = result["sites"], result["variables"], result["ks"]
    cells = result["cells"]
    fig = plt.figure(figsize=(max(11.5, 4.0 + 1.6 * len(variables)), 0.5 * len(sites) + 2.6))
    gs = fig.add_gridspec(1, 1 + len(variables), width_ratios=[0.55 * len(variables)] + [1] * len(variables), wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    mat = np.array([[cells[f"{s}/{v}"]["grid"][cells[f"{s}/{v}"]["k_star"]]["cause"]["mean"] for v in variables] for s in sites])
    ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(sites):
        for j, v in enumerate(variables):
            c = cells[f"{s}/{v}"]
            g = c["grid"][c["k_star"]]
            ax.text(j, i, f"{g['cause']['mean']:.2f}\nIso {g['isolate_mean']:.2f}\nk={c['k_star']}", ha="center", va="center",
                    fontsize=6, color="#fcfcfb" if g["cause"]["mean"] > 0.55 else INK)
            if c["held"]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#eb6834", lw=1.6))
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables, fontsize=7.5)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([f"{s}" + (" (control)" if s == result["control_site"] else "") for s in sites], fontsize=7)
    joint = result.get("positions", "answer") != "answer"
    ax.set_ylabel("read point (" + ("swapped tokens, jointly" if joint else "answer token") + ")", fontsize=7, color=INK2)
    ax.set_title("test Cause at k* (outlined: held,\nCause lo >= 0.5 and Isolate lo >= 0.75)", fontsize=7.5, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    shades = np.linspace(0.35, 1.0, len(sites))
    for j, v in enumerate(variables):
        ax2 = fig.add_subplot(gs[0, 1 + j])
        for i, s in enumerate(sites):
            ys = [cells[f"{s}/{v}"]["grid"][k]["cause"]["mean"] for k in ks]
            ax2.plot(range(len(ks)), ys, color=plt.cm.Blues(shades[i]), lw=1.3, marker="o", ms=2.5,
                     ls=(0, (2, 1.5)) if s == result["control_site"] else "-", label=str(s))
            null = cells[f"{s}/{v}"]["null_cause"]["mean"]
            ax2.plot([len(ks) - 1], [null], marker="x", ms=3, color="#8a8985", ls="",
                     label=f"null, k={ks[-1]}" if i == len(sites) - 1 else None)
        ax2.set_xticks(range(len(ks)))
        ax2.set_xticklabels([str(k) for k in ks], fontsize=6.5)
        ax2.set_ylim(0, 1)
        ax2.set_title(f"Cause vs k: {v}", fontsize=7.5, loc="left")
        ax2.tick_params(labelsize=6.5, length=2, colors=INK2)
        ax2.grid(axis="y", color="#ecebe7", lw=0.6)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        if j == 0:
            ax2.legend(title="read point", fontsize=6, title_fontsize=6, frameon=False, loc="upper left")
        ax2.set_xlabel("k", fontsize=6.5, color=INK2)
    template = TEMPLATES[result.get("task", "two_lane")].strip()
    where = "the answer token"
    if joint:
        sw = result.get("swap") or {}
        where = f"{len(sw.get('tokens', []))} query tokens jointly ({result['positions'].replace('_', '-')} set"
        where += ", one subspace shared by the tokens)" if result.get("shared_basis") else ")"
    fig.suptitle(f"DAS at {where} of {template} ({model_name}): which sub-results are held as separate "
                 f"variables", fontsize=8.5, x=0.01, y=1.06, ha="left")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def run_das(cfg: DasConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return DAS(cfg, run_id, config_path).run()
