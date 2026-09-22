"""All-to-all comparison of downstream trajectories (docs/DECISIONS.md D32, D33).

Two finished runs of one model (the head-mean and the learned-vector run: same seed, splits and held-out
prompts) supply three constructions of a task direction, the pooled PC1, the head-mean function vector and
the learned vector. At common injection layers each is injected into the held-out zero-shot prompts, and
the residual difference it makes at every later read point,

    delta_m(v)(x) = h_m(x; h_l + v) - h_m(x),

is compared with the natural difference the demonstrations make,

    delta_m^ICL(x) = h_m(x; demos) - h_m(x)      (and its permuted-label contrast, h(demos) - h(deranged)),

and with the other constructions' differences, pair by pair, per example and per read point. Matched
isotropic directions at each construction's norm give the floor of every alignment; a second demonstration
sample gives its ceiling. Two further variants remove, from both vectors of a pair, the answer's unembedding
direction and the generic response (the mean trajectory of the isotropic controls). D33 repeats the whole
comparison at further multiples of the canonical strength, records the coherence of the random responses and
what the generic response is, and runs the patch test: whether the component a construction's trajectory
shares with the natural one is what carries the effect.

All geometry over the trajectories runs in torch on the model's device (D33; the NumPy functions below are
the reference the tests compare against), so a run is bounded by its forward passes.
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
from .stats import bootstrap_median_ci_rows, paired_bootstrap_test, paired_excess_test
from .tasks import Item

CONSTRUCTIONS = ("pca", "fv", "learned")  # the three implementations of a task direction
NATURAL = ("icl", "icl2", "task")  # the natural trajectories: two demonstration samples and the label contrast
REFERENCES = ("icl", "task")  # what a construction's alignment is summarised against
NAMED = NATURAL + CONSTRUCTIONS
ALIGN_MIN_EXCESS = 0.05  # exploratory labels only: a decline smaller than this is not a divergence
VARIANT_PREFIX = {"raw": "", "answer_removed": "noanswer_", "generic_removed": "nogeneric_"}
PATCH_EDITS = ("remove", "keep", "patch")


def _variant_prefix(tag: str) -> str:
    """The array-name prefix of a comparison variant (``cos_<prefix><a>~<b>``, ``floor_<prefix><c>_vs_<o>``)."""
    return VARIANT_PREFIX[tag]


def _factor_tag(f: float) -> str:
    return f"r{f:g}"


# --------------------------------------------------------------------------- #
# Geometry, NumPy reference (tested against naive loops; the torch path below is tested against these)
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


def block_writing(A: np.ndarray, B: np.ndarray, start: int = 0, U: np.ndarray | None = None) -> np.ndarray:
    """How much of the reference ``B`` each block writes into the trajectory ``A`` (D33 amended): at read
    point ``m + 1`` (the block just run), ``((a_{m+1} - a_m) . b_{m+1}) / |b_{m+1}|^2`` per example, for
    ``m >= start`` (L+1, n; nan at and below ``start``). With ``U`` (unit rows, (n, d) or (L+1, n, d)), that
    direction at ``m + 1`` is projected out of the block's increment and of the reference first. The
    reference moves with depth, so the increments do not telescope to the projection fraction."""
    L1, n, _ = A.shape
    out = np.full((L1, n), np.nan)
    for m in range(start, L1 - 1):
        u = _rows_at(U, m + 1)
        inc = remove_direction(np.asarray(A[m + 1], dtype=np.float64) - np.asarray(A[m], dtype=np.float64), u)
        b = remove_direction(np.asarray(B[m + 1], dtype=np.float64), u)
        nb2 = np.einsum("nd,nd->n", b, b)
        ok = nb2 > 0
        out[m + 1, ok] = np.einsum("nd,nd->n", inc[ok], b[ok]) / nb2[ok]
    return out


def mean_cosines(mean_a: np.ndarray, mean_b: np.ndarray, start: int = 0) -> np.ndarray:
    """Cosine between two population-mean trajectories (L+1, d) at every read point (L+1,)."""
    out = np.full(mean_a.shape[0], np.nan)
    for m in range(start, mean_a.shape[0]):
        na, nb = np.linalg.norm(mean_a[m]), np.linalg.norm(mean_b[m])
        if na > 0 and nb > 0:
            out[m] = float(mean_a[m] @ mean_b[m] / (na * nb))
    return out


def coherence_curve(deltas: list[np.ndarray], start: int) -> list[float | None]:
    """How much of a random push's downstream change is shared by all of them: at every read point from
    ``start`` on, the norm of the mean of the ``deltas`` (each (L+1, n, d)) over the mean of their norms, per
    example, summarised by the median over examples. 1 when every push produces the same change, near 0
    when the changes are unrelated; None below ``start``."""
    L1 = deltas[0].shape[0]
    out: list[float | None] = [None] * L1
    for m in range(start, L1):
        stack = np.stack([np.asarray(d[m], dtype=np.float64) for d in deltas])  # (K, n, d)
        mean_norm = np.linalg.norm(stack.mean(axis=0), axis=1)
        norm_mean = np.linalg.norm(stack, axis=2).mean(axis=0)
        ok = norm_mean > 0
        out[m] = float(np.median(mean_norm[ok] / norm_mean[ok])) if ok.any() else None
    return out


def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Geometry on the device (D33): the same quantities in torch, the trajectories never leaving the device
# --------------------------------------------------------------------------- #


class Geometry:
    """Trajectory geometry and the row-wise bootstraps in torch on ``device`` (the model's device; CPU in the
    tests). Inputs are tensors kept on the device as float32 (L+1, n, d); every computation is in float64;
    outputs come back as NumPy arrays. Each method mirrors the NumPy reference of the same name."""

    def __init__(self, device: Any, seed: int) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    # ---- tensors ------------------------------------------------------- #

    def tensor(self, x: Any, dtype: Any = None) -> Any:
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(self.device) if dtype is None else x.to(self.device, dtype)
        return t.as_tensor(np.asarray(x), dtype=dtype or t.float32, device=self.device)

    def unit_rows(self, X: Any) -> Any:
        t = self.torch
        X = X.to(t.float64)
        norm = X.norm(dim=-1, keepdim=True)
        return t.where(norm > 0, X / t.where(norm > 0, norm, t.ones_like(norm)), t.zeros_like(X))

    def median(self, x: Any, dim: int = 0) -> Any:
        """``np.median`` along ``dim`` (the mean of the two middle values when even)."""
        s, _ = x.sort(dim=dim)
        n = s.shape[dim]
        if n % 2:
            return s.narrow(dim, n // 2, 1).squeeze(dim)
        return 0.5 * (s.narrow(dim, n // 2 - 1, 1) + s.narrow(dim, n // 2, 1)).squeeze(dim)

    @staticmethod
    def _rows_at(U: Any, m: int) -> Any:
        if U is None:
            return None
        return U[m] if U.dim() == 3 else U

    # ---- per-example curves -------------------------------------------- #

    def pair_cosines(self, A: Any, B: Any, start: int = 0, U: Any = None) -> np.ndarray:
        t = self.torch
        L1, n, _ = A.shape
        out = t.full((L1, n), float("nan"), dtype=t.float64, device=self.device)
        for m in range(start, L1):
            u = self._rows_at(U, m)
            a, b = A[m].to(t.float64), B[m].to(t.float64)
            if u is not None:
                a = a - (a * u).sum(1, keepdim=True) * u
                b = b - (b * u).sum(1, keepdim=True) * u
            na, nb = a.norm(dim=1), b.norm(dim=1)
            ok = (na > 0) & (nb > 0)
            denom = t.where(ok, na * nb, t.ones_like(na))
            out[m] = t.where(ok, (a * b).sum(1) / denom, t.full_like(na, float("nan")))
        return out.cpu().numpy()

    def projection_fraction(self, A: Any, B: Any, start: int = 0) -> np.ndarray:
        t = self.torch
        L1, n, _ = A.shape
        out = t.full((L1, n), float("nan"), dtype=t.float64, device=self.device)
        for m in range(start, L1):
            a, b = A[m].to(t.float64), B[m].to(t.float64)
            nb2 = (b * b).sum(1)
            ok = nb2 > 0
            out[m] = t.where(ok, (a * b).sum(1) / t.where(ok, nb2, t.ones_like(nb2)), t.full_like(nb2, float("nan")))
        return out.cpu().numpy()

    def block_writing(self, A: Any, B: Any, start: int = 0, U: Any = None) -> np.ndarray:
        t = self.torch
        L1, n, _ = A.shape
        out = t.full((L1, n), float("nan"), dtype=t.float64, device=self.device)
        for m in range(start, L1 - 1):
            u = self._rows_at(U, m + 1)
            inc = A[m + 1].to(t.float64) - A[m].to(t.float64)
            b = B[m + 1].to(t.float64)
            if u is not None:
                inc = inc - (inc * u).sum(1, keepdim=True) * u
                b = b - (b * u).sum(1, keepdim=True) * u
            nb2 = (b * b).sum(1)
            ok = nb2 > 0
            out[m + 1] = t.where(ok, (inc * b).sum(1) / t.where(ok, nb2, t.ones_like(nb2)), t.full_like(nb2, float("nan")))
        return out.cpu().numpy()

    def coherence(self, deltas: list[Any], start: int) -> list[float | None]:
        t = self.torch
        L1 = deltas[0].shape[0]
        out: list[float | None] = [None] * L1
        for m in range(start, L1):
            stack = t.stack([d[m].to(t.float64) for d in deltas])  # (K, n, d)
            mean_norm = stack.mean(0).norm(dim=1)
            norm_mean = stack.norm(dim=2).mean(0)
            ok = norm_mean > 0
            if bool(ok.any()):
                out[m] = float(self.median(mean_norm[ok] / norm_mean[ok]).cpu())
        return out

    def mean_over_examples(self, X: Any) -> np.ndarray:
        """Population mean of a trajectory (L+1, n, d) -> (L+1, d) float64 on the host."""
        return X.to(self.torch.float64).mean(1).cpu().numpy()

    # ---- bootstraps ---------------------------------------------------- #

    def bootstrap_rows(self, X: np.ndarray, n_boot: int, alpha: float) -> dict[str, list[float]]:
        """Row-wise median with a percentile bootstrap CI (nan entries dropped), like
        :func:`directions.stats.bootstrap_median_ci_rows`, drawn on the device from this geometry's generator.
        Rows with the same number of finite entries are resampled in one batch."""
        t = self.torch
        Xt = t.as_tensor(np.asarray(X, dtype=np.float64), device=self.device)
        R = Xt.shape[0]
        finite = t.isfinite(Xt)
        counts = finite.sum(1)
        med = t.full((R,), float("nan"), dtype=t.float64, device=self.device)
        low = med.clone()
        high = med.clone()
        for k in t.unique(counts).tolist():
            rows = t.nonzero(counts == k).flatten()
            if k == 0:
                continue
            vals = t.stack([Xt[r][finite[r]] for r in rows.tolist()])  # (r, k)
            med[rows] = self.median(vals, dim=1)
            if k == 1:
                low[rows] = vals[:, 0]
                high[rows] = vals[:, 0]
                continue
            idx = t.randint(0, k, (rows.shape[0], n_boot, k), generator=self.gen, device=self.device)
            samp = t.gather(vals.unsqueeze(1).expand(-1, n_boot, -1), 2, idx)  # (r, n_boot, k)
            meds = self.median(samp, dim=2)  # (r, n_boot)
            q = t.quantile(meds, t.tensor([alpha / 2, 1 - alpha / 2], dtype=t.float64, device=self.device), dim=1)
            low[rows], high[rows] = q[0], q[1]
        return {"median": med.cpu().tolist(), "low": low.cpu().tolist(), "high": high.cpu().tolist(), "n": counts.cpu().tolist()}


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


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rx, ry = np.argsort(np.argsort(x[ok])).astype(np.float64), np.argsort(np.argsort(y[ok])).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarise_block_writing(W: np.ndarray, start: int, boot: Any, natural: np.ndarray | None = None) -> dict[str, Any]:
    """Per-block writing of the shared component (D33 amended), from the per-example curve ``W`` (L+1, n) of
    :func:`block_writing`: the median per block with its CI, the cumulative median, the block with the
    largest median share and that share of the positive total, the entropy ratio of the positive shares
    (1 = spread evenly over the downstream blocks, 0 = one block), and the rank correlation of the block
    profile with the natural trajectory's own profile when given. Blocks are named by the read point after
    them."""
    curve = boot(W.astype(np.float64))
    med = np.array([np.nan if v is None else v for v in curve["median"]], dtype=np.float64)
    blocks = np.arange(start + 1, W.shape[0])
    vals = med[start + 1:]
    finite = np.isfinite(vals)
    out: dict[str, Any] = {"start": start, "curve": curve,
                           "cumulative_median": np.nancumsum(np.where(finite, vals, 0.0)).tolist(),
                           "dominant_block": None, "dominant_share": None, "entropy_ratio": None,
                           "rank_correlation_with_natural": None}
    if finite.any():
        pos = np.where(finite & (vals > 0), vals, 0.0)
        total = pos.sum()
        i = int(np.nanargmax(np.where(finite, vals, -np.inf)))
        out["dominant_block"] = int(blocks[i])
        if total > 0:
            p = pos / total
            p = p[p > 0]
            out["dominant_share"] = float(pos[i] / total)
            out["entropy_ratio"] = float(-(p * np.log(p)).sum() / np.log(len(vals))) if len(vals) > 1 else 1.0
    if natural is not None:
        out["rank_correlation_with_natural"] = _spearman(vals, np.asarray(natural, dtype=np.float64)[start + 1:])
    return out


def summarise_alignment(
    real: np.ndarray, floor: np.ndarray, start: int, rng: np.random.Generator, n_boot: int, alpha: float,
    boot: Any = None,
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
    floor, and ``partial`` otherwise. ``boot(X)`` computes the row-wise bootstrap (the device path); the
    NumPy one with ``rng`` by default.
    """
    L = real.shape[0] - 1
    if boot is None:
        boot = lambda X: bootstrap_median_ci_rows(X, rng, n_boot=n_boot, alpha=alpha)  # noqa: E731
    med = boot(real)
    fl = boot(floor)
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
    extraction: list[Item]
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
        extraction=_items(splits.get("extraction", [])),
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
        # the geometry runs on the model's device (D33): its draws come from the config seed
        self.geo = Geometry(self.backend.device, derive_seed(self.cfg.seed, "trajectories_geometry"))
        self.metadata["linalg_device"] = str(self.backend.device)
        names = sorted(p.name for p in (self.learned_root / "core" / "tasks").iterdir() if p.is_dir())
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        loaded: list[TaskInputs] = []
        for name in names:
            inputs = _load_task(name, self.learned_root, self.fv_root, self.run_cfg)
            if inputs is None:
                summary["skipped"][name] = "no learned-vector selection in the learned run"
                self.log.info("[%s] skipped: %s", name, summary["skipped"][name])
                continue
            loaded.append(inputs)
        # D34: the pools' natural differences and unsteered residuals of every task, captured once up front so
        # that each task's subspace test can fit the leave-one-task-out subspace from the other tasks' pools
        self.pool_natural: dict[str, Any] = {}
        self.pool_base: dict[str, Any] = {}
        if self.cfg.subspace.enabled:
            for inputs in loaded:
                with self.prof.section("pool", inputs.name):
                    self._capture_pool(inputs)
        for inputs in loaded:
            with self.prof.section("task", inputs.name):
                summary["tasks"][inputs.name] = self._task(inputs)
        write_json(self.root / "core" / "summary.json", summary)
        self.log.info("summary:\n%s", format_summary(summary))

    # ------------------------------------------------------------------ #

    def _layers(self, inputs: TaskInputs) -> tuple[list[int], int, dict[int, list[str]]]:
        """The injection layers compared for a task, the primary one, and why each layer is in: ``primary``
        (the learned run's selected layer), ``fv_selected`` (the head-mean run's), ``neighbour_below`` /
        ``neighbour_above`` (the nearest candidate layers of the learned run on each side, D33 amended;
        a side with no candidate is noted), ``listed`` (an explicit list)."""
        primary = int(inputs.learned_qual["selection"]["layer"])
        roles: dict[int, list[str]] = {}

        def add(layer: int, role: str) -> None:
            roles.setdefault(layer, []).append(role)

        if isinstance(self.cfg.layers, list):
            for l in self.cfg.layers:
                add(int(l), "primary" if int(l) == primary else "listed")
            return sorted(roles), primary, {l: roles[l] for l in sorted(roles)}
        add(primary, "primary")
        if inputs.fv_qual and inputs.fv_qual.get("selection"):
            add(int(inputs.fv_qual["selection"]["layer"]), "fv_selected")
        k = self.cfg.neighbour_layers
        if k > 0:
            candidates = sorted(int(l) for l in inputs.learned_arrays["layers"])
            below = [l for l in candidates if l < primary][-k:]
            above = [l for l in candidates if l > primary][:k]
            for l in below:
                add(l, "neighbour_below")
            for l in above:
                add(l, "neighbour_above")
            for side, found in (("below", below), ("above", above)):
                if len(found) < k:
                    inputs.notes.append(f"{len(found)} of {k} candidate layers {side} the primary layer {primary} "
                                        f"(candidates {candidates})")
        return sorted(roles), primary, {l: roles[l] for l in sorted(roles)}

    def _constructions(self, inputs: TaskInputs, layer: int, calibration_base: ForwardResult,
                       calibration_prompts: list[Prompt]) -> dict[str, dict[str, Any]]:
        """Unit direction and canonical strength of every construction available at ``layer``."""
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
        layers, primary, roles = self._layers(inputs)
        log.info("[%s] %d held-out prompts; layers %s (primary %d; %s); strengths x%s; %s", name, len(ev), layers, primary,
                 ", ".join(f"{l}: {'/'.join(r)}" for l, r in roles.items()), cfg.strength_factors,
                 "; ".join(inputs.notes) or "both runs present")

        with self.prof.section("natural", name):
            base = self.backend.run(zs, capture=True)
            icl = self.backend.run(fs, capture=True)
            icl2 = self.backend.run(fs2, capture=True) if cfg.second_demo_sample else None
            deranged = self.backend.run(der, capture=True)
        assert base.residuals is not None and icl.residuals is not None and deranged.residuals is not None
        base_t = self.geo.tensor(base.residuals)
        natural: dict[str, Any] = {"icl": self.geo.tensor(icl.residuals) - base_t,
                                   "task": self.geo.tensor(icl.residuals) - self.geo.tensor(deranged.residuals)}
        if icl2 is not None:
            assert icl2.residuals is not None
            natural["icl2"] = self.geo.tensor(icl2.residuals) - base_t
        conditions: dict[str, Any] = {"base": base.metrics_dict(), "icl": icl.metrics_dict(), "deranged": deranged.metrics_dict()}
        if icl2 is not None:
            conditions["icl2"] = icl2.metrics_dict()
        gap = float(np.mean(icl.logprob_per_token) - np.mean(base.logprob_per_token))
        U = None
        if cfg.remove_answer_direction:
            U = self.geo.unit_rows(self.geo.tensor(self.backend.unembedding_directions(self.backend.first_target_token_ids(zs)),
                                                   self.geo.torch.float64))
        log.info("[%s] zero-shot lp/tok %.3f, few-shot %.3f (gap %.3f), deranged %.3f%s", name,
                 conditions["base"]["logprob_per_token_mean"], conditions["icl"]["logprob_per_token_mean"], gap,
                 conditions["deranged"]["logprob_per_token_mean"],
                 f", second sample {conditions['icl2']['logprob_per_token_mean']:.3f}" if icl2 is not None else "")

        cal_prompts = [zero_shot_prompt(pc, x) for x in inputs.calibration]
        with self.prof.section("calibration_base", name):
            cal_base = self.backend.run(cal_prompts)

        result: dict[str, Any] = {"task": name, "registry_task": inputs.learned_qual.get("registry_task"),
                                  "n_examples": len(ev), "n_read_points": self.backend.n_layers + 1,
                                  "layers": layers, "primary_layer": primary, "layer_roles": {str(l): r for l, r in roles.items()},
                                  "notes": inputs.notes,
                                  "conditions": conditions, "fewshot_gap_per_token": gap,
                                  "answer_direction_removed": cfg.remove_answer_direction,
                                  "strength_factors": list(cfg.strength_factors), "per_layer": {}}
        arrays: dict[str, np.ndarray] = {}
        for m, D in natural.items():
            arrays[f"mean_delta_{m}"] = self.geo.mean_over_examples(D).astype(np.float16)
        arrays["mean_residual"] = self.geo.mean_over_examples(base_t).astype(np.float16)
        for layer in layers:
            with self.prof.section("layer", name, layer):
                res, arr = self._layer(inputs, layer, layer == primary, zs, base, base_t, natural, U, cal_base, cal_prompts)
            result["per_layer"][str(layer)] = res
            arrays.update({f"L{layer}_{k}": v for k, v in arr.items()})
        del natural, base_t
        write_json(out_dir / "trajectories.json", result)
        np.savez_compressed(out_dir / "trajectories_arrays.npz", **arrays)
        if cfg.figures:
            from .figures import strength_figures, subspace_figures, trajectory_figures

            trajectory_figures(self.root / "figures", result)
            strength_figures(self.root / "figures", result)
            subspace_figures(self.root / "figures", result)
        return _task_summary(result)

    # ------------------------------------------------------------------ #

    def _layer(self, inputs: TaskInputs, layer: int, primary: bool, zs: list[Prompt], base: ForwardResult, base_t: Any,
               natural: dict[str, Any], U: Any, cal_base: ForwardResult, cal_prompts: list[Prompt],
               ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cfg, name = self.cfg, inputs.name
        with self.prof.section("strengths", name, layer):
            cons = self._constructions(inputs, layer, cal_base, cal_prompts)
        arrays: dict[str, np.ndarray] = {}
        per_factor: dict[float, dict[str, Any]] = {}
        deltas_by_factor: dict[float, dict[str, Any]] = {}
        generic_by_factor: dict[float, np.ndarray] = {}
        for f in cfg.strength_factors:
            with self.prof.section("factor", name, layer, _factor_tag(f)):
                info_f, arr_f, deltas_f, gmean = self._factor(inputs, layer, f, cons, zs, base, base_t, natural, U)
            per_factor[f] = info_f
            deltas_by_factor[f] = deltas_f
            if gmean is not None:
                generic_by_factor[f] = gmean
            prefix = "" if f == 1.0 else f"{_factor_tag(f)}_"
            arrays.update({f"{prefix}{k}": v for k, v in arr_f.items()})
        info = per_factor[1.0]
        info["other_strengths"] = {f"{f:g}": per_factor[f] for f in cfg.strength_factors if f != 1.0}
        # the same construction at two strengths: does the direction of its downstream change depend on the push?
        if len(cfg.strength_factors) > 1:
            with self.prof.section("cross_strength", name, layer):
                info["cross_strength"], arr = self._cross_strength(inputs, layer, cons, deltas_by_factor, generic_by_factor, natural)
            arrays.update(arr)
        if cfg.generic_diagnostics and generic_by_factor:
            with self.prof.section("diagnostics", name, layer):
                info["generic_diagnostics"] = self._generic_diagnostics(layer, generic_by_factor, base_t, zs, U)
        if cfg.patch.enabled and (primary or cfg.patch.layers == "all"):
            with self.prof.section("patch", name, layer):
                info["patch"] = self._patch_test(inputs, layer, cons, zs, base, base_t, natural, deltas_by_factor[1.0], info)
        if cfg.subspace.enabled and primary and name in self.pool_natural:
            with self.prof.section("subspace", name, layer):
                info["subspace"] = self._subspace_test(inputs, layer, cons, zs, base, base_t, natural, deltas_by_factor[1.0], info)
        for sub in (info, *info["other_strengths"].values()):
            sub.pop("_first_token_logprob", None)
        del deltas_by_factor
        return info, arrays

    # ------------------------------------------------------------------ #

    def _factor(self, inputs: TaskInputs, layer: int, factor: float, cons: dict[str, dict[str, Any]], zs: list[Prompt],
                base: ForwardResult, base_t: Any, natural: dict[str, Any], U: Any,
                ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], np.ndarray | None]:
        """The comparison at ``factor`` times each construction's canonical strength: steered passes, the
        matched isotropic floors, the generic response and its coherence, every pair's cosines in the three
        variants, the summaries and the ceilings."""
        cfg, name, log, seed, geo = self.cfg, inputs.name, self.log, self.cfg.seed, self.geo
        t = geo.torch
        deltas: dict[str, Any] = dict(natural)
        first_token: dict[str, np.ndarray] = {}
        info: dict[str, Any] = {"layer": layer, "strength_factor": factor, "constructions": {}, "isotropic": {"n": cfg.n_isotropic}}
        arrays: dict[str, np.ndarray] = {}
        base_lp = float(np.mean(base.logprob_per_token))
        for c in CONSTRUCTIONS:
            if c not in cons:
                continue
            spec = cons[c]
            alpha = factor * spec["alpha"]
            with self.prof.section("steered", name, layer, c):
                r = self.backend.run(zs, interventions=[Intervention(layer, spec["direction"], alpha)], capture=True)
            if cfg.determinism_check and "determinism_check" not in self.metadata and c == "learned":
                again = self.backend.run(zs, interventions=[Intervention(layer, spec["direction"], alpha)], capture=True)
                assert again.residuals is not None
                identical = bool(np.array_equal(again.residuals, r.residuals) and np.array_equal(again.logprob_sum, r.logprob_sum))
                self.metadata["determinism_check"] = {"task": name, "layer": layer, "construction": c, "identical": identical}
                if not identical:
                    raise RuntimeError("determinism check failed: a repeated steered pass differs")
            assert r.residuals is not None and r.first_token_logprob is not None
            deltas[c] = geo.tensor(r.residuals) - base_t
            first_token[c] = r.first_token_logprob
            m = r.metrics_dict()
            info["constructions"][c] = {k: v for k, v in spec.items() if k != "direction"}
            info["constructions"][c].update({"alpha": alpha, "canonical_alpha": spec["alpha"], "metrics": m,
                                             "improvement_per_token": m["logprob_per_token_mean"] - base_lp})
            arrays[f"mean_delta_{c}"] = geo.mean_over_examples(deltas[c]).astype(np.float16)
        present = [c for c in CONSTRUCTIONS if c in deltas]
        named = [n for n in NAMED if n in deltas]
        # matched isotropic directions at each construction's norm: the floor of every pair involving it
        iso_deltas: dict[str, list[Any]] = {c: [] for c in present}
        iso_metrics: dict[str, list[dict[str, float]]] = {c: [] for c in present}
        for c in present:
            alpha = info["constructions"][c]["alpha"]
            for k in range(cfg.n_isotropic):
                v = unit_vectors(rng_for(seed, "trajectories_isotropic", name, layer, c, k), 1, self.backend.hidden_size)[0]
                with self.prof.section("isotropic", name, layer):
                    r = self.backend.run(zs, interventions=[Intervention(layer, v, alpha)], capture=True)
                assert r.residuals is not None
                iso_deltas[c].append(geo.tensor(r.residuals) - base_t)
                iso_metrics[c].append(r.metrics_dict())
                del r
        for c in present:
            info["isotropic"][c] = {"alpha": info["constructions"][c]["alpha"],
                                    "improvement_per_token_mean": float(np.mean([m["logprob_per_token_mean"] for m in iso_metrics[c]]) - base_lp)}
        # The generic response (D32, amended): the mean of every isotropic control's trajectory, per example
        # and read point; removed from both vectors of a pair like the answer direction. A control's own floor
        # removes the mean of the other controls.
        all_iso = [d for c in present for d in iso_deltas[c]]
        G = None
        total = None
        gmean: np.ndarray | None = None
        if cfg.remove_generic_response and all_iso:
            total = t.zeros_like(all_iso[0], dtype=t.float64)
            for d in all_iso:
                total += d
            G = geo.unit_rows(total / len(all_iso))
            gmean = geo.mean_over_examples(total / len(all_iso))
            arrays["mean_generic"] = gmean.astype(np.float16)
            cos_ans = None
            if U is not None:
                cos_ans = [float(geo.median((G[m] * U).sum(1)).cpu()) if m >= layer else None for m in range(G.shape[0])]
            info["generic_response"] = {"n_controls": len(all_iso), "cos_with_answer_direction": cos_ans,
                                        # coherence of the random responses: the norm of their mean over the mean of their
                                        # norms, per example and read point (median over examples); 1 = every random push
                                        # produces the same downstream change, 0 = unrelated changes
                                        "coherence": geo.coherence(all_iso, layer),
                                        "coherence_by_construction": {c: geo.coherence(iso_deltas[c], layer) for c in present}}
        variants: dict[str, Any] = {"raw": None}
        if U is not None:
            variants["answer_removed"] = U
        if G is not None:
            variants["generic_removed"] = G
        # pairwise cosines among the named trajectories (from the injection layer on when a construction is involved)
        pairs: dict[str, Any] = {}
        for i, a in enumerate(named):
            for b in named[i + 1:]:
                start = layer if (a in CONSTRUCTIONS or b in CONSTRUCTIONS) else 0
                key = f"{a}~{b}"
                pairs[key] = {"a": a, "b": b, "start": start,
                              "mean_trajectory_cosine": mean_cosines(geo.mean_over_examples(deltas[a]), geo.mean_over_examples(deltas[b]), start).tolist()}
                for tag, R in variants.items():
                    arrays[f"cos_{_variant_prefix(tag)}{key}"] = geo.pair_cosines(deltas[a], deltas[b], start, R).astype(np.float32)
                if b in REFERENCES or a in REFERENCES:
                    ref, other = (b, a) if b in REFERENCES else (a, b)
                    arrays[f"projection_{other}_on_{ref}"] = geo.projection_fraction(deltas[other], deltas[ref], start).astype(np.float32)
        floors: dict[str, dict[str, dict[str, Any]]] = {tag: {c: {} for c in present} for tag in variants}
        for c in present:
            for d_iso in iso_deltas[c]:
                loo = None
                if G is not None:
                    loo = geo.unit_rows((total - d_iso) / (len(all_iso) - 1)) if len(all_iso) > 1 else G
                for other in named:
                    for tag, R in variants.items():
                        rem = loo if tag == "generic_removed" else R
                        floors[tag][c].setdefault(other, []).append(geo.pair_cosines(d_iso, deltas[other], layer, rem))
                del loo
        del all_iso, iso_deltas, total
        for tag in variants:
            for c in present:
                floors[tag][c] = {o: np.concatenate(v, axis=1) for o, v in floors[tag][c].items()}  # (L+1, k*n)
                for o in floors[tag][c]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan rows below the injection layer
                        arrays[f"floor_{_variant_prefix(tag)}{c}_vs_{o}"] = np.nanmedian(floors[tag][c][o], axis=1).astype(np.float32)
        # summaries: every construction against the natural trajectories and against the other constructions
        brng = rng_for(seed, "trajectories_bootstrap", name, layer, _factor_tag(factor))
        boot = lambda X: geo.bootstrap_rows(X, cfg.n_boot, cfg.ci_alpha)  # noqa: E731
        summaries: dict[str, Any] = {}
        for c in present:
            for o in named:
                if o == c:
                    continue
                key = f"{c}~{o}" if f"{c}~{o}" in pairs else f"{o}~{c}"
                for tag in variants:
                    real = arrays[f"cos_{_variant_prefix(tag)}{key}"].astype(np.float64)
                    s = summarise_alignment(real, floors[tag][c][o], layer, brng, cfg.n_boot, cfg.ci_alpha, boot=boot)
                    if tag == "raw":
                        summaries[f"{c}->{o}"] = s
                    else:
                        summaries[f"{c}->{o}"][tag] = s
        ceilings: dict[str, Any] = {}
        for tag in variants:
            suffix = "" if tag == "raw" else f"_{tag}"
            if "icl2" in deltas:
                ceilings[f"icl~icl2{suffix}"] = boot(arrays[f"cos_{_variant_prefix(tag)}icl~icl2"].astype(np.float64))
            ceilings[f"icl~task{suffix}"] = boot(arrays[f"cos_{_variant_prefix(tag)}icl~task"].astype(np.float64))
        # Per-block writing of the shared component (D33 amended): which blocks write the natural difference
        # into each trajectory, for the constructions and for the natural trajectories themselves.
        if "icl" in deltas:
            writing: dict[str, Any] = {}
            nat_med: dict[str, np.ndarray] = {}
            for tag in ("raw", "generic_removed"):
                if tag not in variants:
                    continue
                R = variants[tag]
                for a in named:
                    start = layer if a in CONSTRUCTIONS else 0
                    W = geo.block_writing(deltas[a], deltas["icl"], start, R)
                    arrays[f"written_{_variant_prefix(tag)}{a}_on_icl"] = W.astype(np.float32)
                    s = summarise_block_writing(W, start, boot, nat_med.get(tag))
                    if a == "icl":
                        nat_med[tag] = np.array([np.nan if v is None else v for v in s["curve"]["median"]])
                    writing.setdefault(a, {})[tag] = s
                for a in named:  # the constructions' rank correlation against the natural profile of the same variant
                    if a != "icl" and tag in nat_med:
                        writing[a][tag]["rank_correlation_with_natural"] = _spearman(
                            np.array([np.nan if v is None else v for v in writing[a][tag]["curve"]["median"]])[writing[a][tag]["start"] + 1:],
                            nat_med[tag][writing[a][tag]["start"] + 1:])
            info["block_writing"] = {"reference": "icl", "trajectories": writing}
        info.update({"pairs": pairs, "summaries": summaries, "ceilings": ceilings, "variants": list(variants)})
        for c in present:
            s = summaries.get(f"{c}->icl")
            if s and "peak" in s:
                log.info("[%s] L%d x%g %-7s alpha %.2f (%s) %+.2f nats/tok | cos with ICL: inj %.2f peak %.2f@%.2f final %.2f "
                         "(floor %.2f) %s%s", name, layer, factor, c, info["constructions"][c]["alpha"], cons[c]["source"],
                         info["constructions"][c]["improvement_per_token"], s["at_injection"], s["peak"],
                         s["peak_depth_fraction"], s["final"], s["floor_at_final"], s["label"], _variant_note(s))
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                s = summaries.get(f"{a}->{b}")
                if s and "peak" in s:
                    log.info("[%s] L%d x%g %s~%s: inj %.2f peak %.2f@%.2f final %.2f (floor %.2f) %s%s", name, layer, factor, a, b,
                             s["at_injection"], s["peak"], s["peak_depth_fraction"], s["final"], s["floor_at_final"], s["label"],
                             _variant_note(s))
        if "generic_response" in info:
            coh = [v for v in info["generic_response"]["coherence"] if v is not None]
            log.info("[%s] L%d x%g coherence of the random responses: %.2f right after injection, %.2f mid, %.2f at the end; "
                     "ceiling cos(icl, icl2): final %.2f (generic removed %s)", name, layer, factor,
                     coh[min(1, len(coh) - 1)], coh[len(coh) // 2], coh[-1],
                     ceilings.get("icl~icl2", {"median": [float("nan")]})["median"][-1],
                     f"{ceilings['icl~icl2_generic_removed']['median'][-1]:.2f}" if "icl~icl2_generic_removed" in ceilings else "n/a")
        kept = {c: deltas[c] for c in present}
        info["_first_token_logprob"] = first_token  # consumed by the patch test, not written
        return info, arrays, kept, gmean

    # ------------------------------------------------------------------ #

    def _cross_strength(self, inputs: TaskInputs, layer: int, cons: dict[str, dict[str, Any]],
                        deltas_by_factor: dict[float, dict[str, Any]], generic_by_factor: dict[float, np.ndarray],
                        natural: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The same construction's trajectories at the canonical strength and at each other factor: per-example
        cosine (raw and with the canonical generic response removed), summarised with the canonical floors'
        counterpart (the canonical isotropic controls against the weaker trajectory are not kept; the floor
        here is the weaker trajectory's own generic-removed floor against the construction at the canonical
        strength, taken from the factor's summaries), plus the cosine between the generic responses."""
        cfg, geo = self.cfg, self.geo
        out: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        base_f = 1.0
        for f in cfg.strength_factors:
            if f == base_f:
                continue
            tag = _factor_tag(f)
            entry: dict[str, Any] = {}
            for c in deltas_by_factor[base_f]:
                if c not in deltas_by_factor[f]:
                    continue
                A, B = deltas_by_factor[base_f][c], deltas_by_factor[f][c]
                cos = geo.pair_cosines(A, B, layer)
                arrays[f"{tag}_cross_cos_{c}"] = cos.astype(np.float32)
                boot = lambda X: geo.bootstrap_rows(X, cfg.n_boot, cfg.ci_alpha)  # noqa: E731
                entry[c] = {"curve": boot(cos.astype(np.float64)),
                            "mean_trajectory_cosine": mean_cosines(geo.mean_over_examples(A), geo.mean_over_examples(B), layer).tolist()}
                curve = entry[c]["curve"]["median"]
                vals = [v for v in curve[layer:] if v is not None and not np.isnan(v)]
                entry[c].update({"at_injection": vals[0] if vals else None, "final": vals[-1] if vals else None,
                                 "min_after_injection": min(vals[1:]) if len(vals) > 1 else None})
            if base_f in generic_by_factor and f in generic_by_factor:
                entry["generic_cosine"] = mean_cosines(generic_by_factor[base_f], generic_by_factor[f], layer).tolist()
            out[f"{f:g}"] = entry
            for c, e in entry.items():
                if c in CONSTRUCTIONS:
                    self.log.info("[%s] L%d %s at x1 vs x%g: cos at injection %.2f, min after %.2f, final %.2f", inputs.name, layer,
                                  c, f, e["at_injection"] or float("nan"), e["min_after_injection"] or float("nan"), e["final"] or float("nan"))
        return out, arrays

    # ------------------------------------------------------------------ #

    def _generic_diagnostics(self, layer: int, generic_by_factor: dict[float, np.ndarray], base_t: Any, zs: list[Prompt],
                             U: Any) -> dict[str, Any]:
        """What the generic response is (D33), from the population means: the fraction of its energy in its
        own top coordinates and in the unsteered residual's largest coordinates (the massive activations), its
        cosine with the residual mean, and its logit lens at the last read point (the tokens the mean random
        response promotes and demotes, through the final norm and the unembedding)."""
        cfg, geo, t = self.cfg, self.geo, self.geo.torch
        L1 = base_t.shape[0]
        L = L1 - 1
        mean_resid = geo.mean_over_examples(base_t)  # (L+1, d)
        mean_abs = base_t.abs().to(t.float64).mean(1).cpu().numpy()  # (L+1, d): which coordinates are large
        ks = [k for k in cfg.diagnostic_top_k if k <= mean_resid.shape[1]]
        out: dict[str, Any] = {"top_k": ks, "per_factor": {}}
        for f, g in generic_by_factor.items():
            energy_own = {str(k): [None] * L1 for k in ks}
            energy_massive = {str(k): [None] * L1 for k in ks}
            cos_resid = [None] * L1
            massive_coords = [None] * L1
            own_coords = [None] * L1
            for m in range(layer, L1):
                e = g[m] ** 2
                tot = float(e.sum())
                if tot <= 0:
                    continue
                order_own = np.argsort(-e)
                order_massive = np.argsort(-mean_abs[m])
                for k in ks:
                    energy_own[str(k)][m] = float(e[order_own[:k]].sum() / tot)
                    energy_massive[str(k)][m] = float(e[order_massive[:k]].sum() / tot)
                nr = np.linalg.norm(mean_resid[m])
                cos_resid[m] = float(g[m] @ mean_resid[m] / (np.sqrt(tot) * nr)) if nr > 0 else None
                massive_coords[m] = [int(i) for i in order_massive[:4]]
                own_coords[m] = [int(i) for i in order_own[:4]]
            entry: dict[str, Any] = {"energy_in_own_top_k": energy_own, "energy_in_residual_top_k": energy_massive,
                                     "cos_with_residual_mean": cos_resid, "residual_top_coordinates": massive_coords,
                                     "own_top_coordinates": own_coords,
                                     "norm": [float(np.linalg.norm(g[m])) if m >= layer else None for m in range(L1)],
                                     "residual_norm": [float(np.linalg.norm(mean_resid[m])) for m in range(L1)]}
            # logit lens at the last read point: the mean random response added to each prompt's final residual
            with t.inference_mode():
                h = base_t[L]
                gvec = geo.tensor(g[L], t.float32).unsqueeze(0).expand_as(h)
                delta_logits = (self.backend.logits_from_residual(h + gvec) - self.backend.logits_from_residual(h)).mean(0)
            top = t.topk(delta_logits, cfg.logit_lens_top)
            bottom = t.topk(-delta_logits, cfg.logit_lens_top)
            entry["logit_lens_last"] = {
                "promoted": [(self.backend.tokenizer.decode([int(i)]), float(v)) for v, i in zip(top.values.cpu(), top.indices.cpu())],
                "demoted": [(self.backend.tokenizer.decode([int(i)]), float(-v)) for v, i in zip(bottom.values.cpu(), bottom.indices.cpu())],
                "mean_abs_logit_change": float(delta_logits.abs().mean().cpu()),
            }
            out["per_factor"][f"{f:g}"] = entry
            self.log.info("[diag] L%d x%g generic: energy in own top 16 %.2f, in the residual's top 16 %.2f, cos with residual mean %.2f "
                          "at the end; promotes %s", layer, f, energy_own.get("16", [None] * L1)[L] or float("nan"),
                          energy_massive.get("16", [None] * L1)[L] or float("nan"), cos_resid[L] or float("nan"),
                          ", ".join(repr(s) for s, _ in entry["logit_lens_last"]["promoted"][:5]))
        return out

    # ------------------------------------------------------------------ #

    def _patch_test(self, inputs: TaskInputs, layer: int, cons: dict[str, dict[str, Any]], zs: list[Prompt], base: ForwardResult,
                    base_t: Any, natural: dict[str, Any], deltas: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Is the component a construction's trajectory shares with the natural one what carries the effect?
        At read points at fixed fractions of the downstream depth, the steered perturbation is edited on the
        query token before the remaining blocks run (its component along the prompt's natural difference
        removed, or kept alone), and, in the unsteered run, the natural difference itself is patched in. Each
        edit is matched against the same edit along random per-example directions (D29 mechanics)."""
        cfg, geo, t, name, seed = self.cfg, self.geo, self.geo.torch, inputs.name, self.cfg.seed
        L = base_t.shape[0] - 1
        ref = natural[cfg.patch.reference]
        steered_first: dict[str, np.ndarray] = info.get("_first_token_logprob", {})
        read_points = sorted({min(L, max(layer + 1, layer + int(round(fr * (L - layer))))) for fr in cfg.patch.depth_fractions})
        base_lp = base.logprob_per_token
        assert base.first_token_logprob is not None
        base_first = base.first_token_logprob
        brng = rng_for(seed, "trajectories_patch_bootstrap", name, layer)
        out: dict[str, Any] = {"reference": cfg.patch.reference, "read_points": read_points, "n_controls": cfg.patch.n_controls,
                               "constructions": {}}
        for c in cfg.patch.constructions:
            if c not in deltas:
                continue
            spec = info["constructions"][c]
            inject = Intervention(layer, cons[c]["direction"], spec["alpha"])
            full = spec["improvement_per_token"]
            full_first = float(np.mean(steered_first[c] - base_first)) if c in steered_first else 0.0
            rows = []
            for m in read_points:
                d_m = deltas[c][m].to(t.float64)
                u = geo.unit_rows(ref[m])  # the natural difference's direction, per prompt
                along = (d_m * u).sum(1, keepdim=True) * u
                edits = {"remove": (-along).float().cpu().numpy(), "keep": (along - d_m).float().cpu().numpy(),
                         "patch": ref[m].float().cpu().numpy()}
                real: dict[str, np.ndarray] = {}
                real_first: dict[str, np.ndarray] = {}
                ctl: dict[str, list[np.ndarray]] = {e: [] for e in PATCH_EDITS}
                ctl_first: dict[str, list[np.ndarray]] = {e: [] for e in PATCH_EDITS}
                for e, vec in edits.items():
                    ivs = [Intervention(m, vec, 1.0)] if e == "patch" else [inject, Intervention(m, vec, 1.0)]
                    r = self.backend.run(zs, interventions=ivs)
                    real[e] = r.logprob_per_token - base_lp
                    real_first[e] = r.first_token_logprob - base_first
                ref_norm = ref[m].to(t.float64).norm(dim=1, keepdim=True)
                for k in range(cfg.patch.n_controls):
                    R = geo.unit_rows(geo.tensor(unit_vectors(rng_for(seed, "trajectories_patch", name, layer, c, m, k), len(zs),
                                                              self.backend.hidden_size), t.float64))
                    along_r = (d_m * R).sum(1, keepdim=True) * R
                    cedits = {"remove": (-along_r).float().cpu().numpy(), "keep": (along_r - d_m).float().cpu().numpy(),
                              "patch": (R * ref_norm).float().cpu().numpy()}
                    for e, vec in cedits.items():
                        ivs = [Intervention(m, vec, 1.0)] if e == "patch" else [inject, Intervention(m, vec, 1.0)]
                        r = self.backend.run(zs, interventions=ivs)
                        ctl[e].append(r.logprob_per_token - base_lp)
                        ctl_first[e].append(r.first_token_logprob - base_first)
                align = info["summaries"].get(f"{c}->{cfg.patch.reference}", {}).get("generic_removed", {}).get("curve", {}).get("median")
                row: dict[str, Any] = {"read_point": m, "depth_fraction": (m - layer) / (L - layer) if L > layer else 1.0,
                                       "alignment_generic_removed": None if not align else align[m]}
                for e in PATCH_EDITS:
                    r_e, c_e = real[e], np.stack(ctl[e])
                    sign = -1.0 if e == "remove" else 1.0  # remove: the real edit should cost more; keep/patch: retain more
                    # An edit at the last read point reaches only the query position's own prediction (no block
                    # follows to carry it to later target positions), so the first-token effect is kept beside
                    # the per-token one; they coincide for one-token targets.
                    row[e] = {"effect": float(np.mean(r_e)), "retained": None if full == 0 else float(np.mean(r_e) / full),
                              "random_effect_mean": float(np.mean(c_e)),
                              "random_retained_mean": None if full == 0 else float(np.mean(c_e) / full),
                              "excess_vs_random": paired_excess_test(sign * r_e, sign * c_e, brng, n_boot=cfg.n_boot).__dict__,
                              "effect_test": paired_bootstrap_test(r_e + base_lp, base_lp, brng, n_boot=cfg.n_boot).__dict__,
                              "first_token_effect": float(np.mean(real_first[e])),
                              "first_token_retained": None if full_first == 0 else float(np.mean(real_first[e]) / full_first),
                              "first_token_random_retained_mean": None if full_first == 0 else float(np.mean(np.stack(ctl_first[e])) / full_first)}
                rows.append(row)
                self.log.info("[%s] L%d patch %s at m=%d (%.2f of the depth, alignment %s): remove keeps %.2f (random %.2f, p %.3f); "
                              "keep retains %.2f (random %.2f, p %.3f); natural patch alone gives %.2f of the effect (random %.2f, p %.3f)",
                              name, layer, c, m, row["depth_fraction"],
                              "n/a" if row["alignment_generic_removed"] is None else f"{row['alignment_generic_removed']:.2f}",
                              row["remove"]["retained"] or float("nan"), row["remove"]["random_retained_mean"] or float("nan"),
                              row["remove"]["excess_vs_random"]["p_value"],
                              row["keep"]["retained"] or float("nan"), row["keep"]["random_retained_mean"] or float("nan"),
                              row["keep"]["excess_vs_random"]["p_value"],
                              row["patch"]["retained"] or float("nan"), row["patch"]["random_retained_mean"] or float("nan"),
                              row["patch"]["excess_vs_random"]["p_value"])
            out["constructions"][c] = {"full_effect": full, "alpha": spec["alpha"], "rows": rows}
        return out


    # ------------------------------------------------------------------ #

    def _capture_pool(self, inputs: TaskInputs) -> None:
        """D34: the natural differences (demos minus none) and the unsteered residuals of the task's fitting
        pools at every read point, as float16 on the host: the subspaces are fitted on these, never on the
        held-out prompts."""
        cfg, pc, name = self.cfg, inputs.prompt_cfg, inputs.name
        seen: set[tuple[str, str]] = set()
        pool: list[Item] = []
        for split in cfg.subspace.pools:
            for x in getattr(inputs, split):
                if (x.input, x.output) not in seen:
                    seen.add((x.input, x.output))
                    pool.append(x)
        if len(pool) < 2:
            inputs.notes.append(f"subspace test skipped: fitting pool of {len(pool)} prompts")
            return
        zs = [zero_shot_prompt(pc, x) for x in pool]
        rng = rng_for(cfg.seed, "trajectories_subspace_fewshot", name)
        fs = [few_shot_prompt(pc, pool, x, rng) for x in pool]
        base = self.backend.run(zs, capture=True)
        icl = self.backend.run(fs, capture=True)
        assert base.residuals is not None and icl.residuals is not None
        t = self.geo.torch
        self.pool_natural[name] = t.as_tensor(icl.residuals - base.residuals).to(t.float16).cpu()
        self.pool_base[name] = t.as_tensor(base.residuals).to(t.float16).cpu()
        self.log.info("[%s] subspace fitting pool: %d prompts (%s); few-shot lp/tok %.3f, zero-shot %.3f", name, len(pool),
                      "+".join(cfg.subspace.pools), float(np.mean(icl.logprob_per_token)), float(np.mean(base.logprob_per_token)))

    def _subspace_test(self, inputs: TaskInputs, layer: int, cons: dict[str, dict[str, Any]], zs: list[Prompt],
                       base: ForwardResult, base_t: Any, natural: dict[str, Any], deltas: dict[str, Any],
                       info: dict[str, Any]) -> dict[str, Any]:
        """The causal dimensionality of the shared component (D34): at read points of the primary layer, the
        steered perturbation is kept only within (or stripped of) a k-dimensional subspace fitted on the pools'
        natural differences, or the prompt's natural difference projected on it is patched into the unsteered
        run, for k in the grid, against random per-example subspaces, the background's top-k subspace and the
        subspace of the other tasks' natural differences."""
        cfg, geo, t, name, seed = self.cfg, self.geo, self.geo.torch, inputs.name, self.cfg.seed
        sc = cfg.subspace
        L = base_t.shape[0] - 1
        kmax = max(sc.k_grid)
        ref = natural[cfg.patch.reference]
        steered_first: dict[str, np.ndarray] = info.get("_first_token_logprob", {})
        base_lp = base.logprob_per_token
        assert base.first_token_logprob is not None
        base_first = base.first_token_logprob
        brng = rng_for(seed, "trajectories_subspace_bootstrap", name, layer)
        others = [o for o in self.pool_natural if o != name]
        out: dict[str, Any] = {"reference": cfg.patch.reference, "k_grid": list(sc.k_grid), "n_controls": sc.n_controls,
                               "pool_size": int(self.pool_natural[name].shape[1]), "other_tasks": others, "constructions": {}}

        def fit(X: Any) -> Any:  # top-kmax right singular vectors of the rows of X (n, d) -> (d, kmax), float64
            _, _, Vh = t.linalg.svd(X.to(geo.device, t.float64), full_matrices=False)
            return Vh[:kmax].T.contiguous()

        def spectrum(X: Any) -> dict[str, Any]:
            s = t.linalg.svdvals(X.to(geo.device, t.float64))
            p = s ** 2 / (s ** 2).sum()
            return {"explained": p[:kmax].cpu().tolist(), "participation_ratio": float(((s ** 2).sum() ** 2 / (s ** 4).sum()).cpu()),
                    "rank": int(min(X.shape))}

        if sc.pool_spectrum_all_read_points:
            # D37: the pool's spectrum at every read point (the rank landmark), with the top component's cosine
            # with the held-out mean natural difference there
            prof = []
            for m in range(L + 1):
                own_X = self.pool_natural[name][m]
                sp = spectrum(own_X)
                top = fit(own_X)[:, 0]
                mean_nat = geo.unit_rows(ref[m].to(t.float64).mean(0, keepdim=True))[0]
                sp["top_component_cos_with_mean_natural"] = float(abs(top @ mean_nat).cpu())
                sp["read_point"] = m
                prof.append(sp)
            out["pool_spectrum"] = prof

        for c in sc.constructions:
            if c not in deltas:
                continue
            spec = info["constructions"][c]
            inject = Intervention(layer, cons[c]["direction"], spec["alpha"])
            full = spec["improvement_per_token"]
            full_first = float(np.mean(steered_first[c] - base_first)) if c in steered_first else 0.0
            points: list[tuple[int, str]] = []
            for fr in sc.depth_fractions:
                points.append((min(L, max(layer + 1, layer + int(round(fr * (L - layer))))), "fraction"))
            handover = None
            if sc.at_handover:
                rows = (((info.get("patch") or {}).get("constructions") or {}).get(c) or {}).get("rows", [])
                handover = next((r["read_point"] for r in rows if r["keep"]["retained"] is not None and r["keep"]["retained"] >= sc.handover_share), None)
                if handover is not None:
                    points.append((handover, "handover"))
            seen: dict[int, str] = {}
            for m, source in points:
                seen[m] = "handover" if source == "handover" or seen.get(m) == "handover" else source
            entries = []
            for m in sorted(seen):
                d_m = deltas[c][m].to(t.float64)
                ref_m = ref[m].to(t.float64)
                own_X = self.pool_natural[name][m]
                U: dict[str, Any] = {"own": fit(own_X)}
                spec_own = spectrum(own_X)
                if sc.other_tasks and others:
                    U["other_tasks"] = fit(t.cat([self.pool_natural[o][m] for o in others], 0))
                if sc.background:
                    U["background"] = fit(self.pool_base[name][m])
                # random per-example k-subspaces: orthonormal columns per example, (n, d, kmax)
                randoms = []
                for j in range(sc.n_controls):
                    R = rng_for(seed, "trajectories_subspace", name, layer, c, m, j).normal(size=(len(zs), self.backend.hidden_size, kmax))
                    Q, _ = t.linalg.qr(t.as_tensor(R, dtype=t.float64, device=geo.device))
                    randoms.append(Q)
                mean_nat = geo.unit_rows(ref_m.mean(0, keepdim=True))[0]
                entry: dict[str, Any] = {"read_point": m, "depth_fraction": (m - layer) / (L - layer) if L > layer else 1.0,
                                         "source": seen[m], "spectrum": spec_own, "overlap": {}, "rows": {},
                                         "top_component_cos_with_mean_natural": float(abs(U["own"][:, 0] @ mean_nat).cpu())}
                for src in ("other_tasks", "background"):
                    if src in U:
                        entry["overlap"][src] = [float(((U[src].T @ U["own"][:, :k]) ** 2).sum().cpu() / k) for k in sc.k_grid]
                patch_rows = (((info.get("patch") or {}).get("constructions") or {}).get(c) or {}).get("rows", [])
                pr = next((r for r in patch_rows if r["read_point"] == m), None)
                entry["per_prompt_keep"] = None if pr is None else pr["keep"]["retained"]
                entry["per_prompt_patch"] = None if pr is None else pr["patch"]["retained"]
                for k in sc.k_grid:
                    row: dict[str, Any] = {}
                    real: dict[str, dict[str, np.ndarray]] = {}
                    ctl: dict[str, list[np.ndarray]] = {e: [] for e in PATCH_EDITS}
                    ctl_first: dict[str, list[np.ndarray]] = {e: [] for e in PATCH_EDITS}

                    def edits_for(proj_d: Any, proj_ref: Any) -> dict[str, np.ndarray]:
                        return {"remove": (-proj_d).float().cpu().numpy(), "keep": (proj_d - d_m).float().cpu().numpy(),
                                "patch": proj_ref.float().cpu().numpy()}

                    def run_edits(edits: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
                        eff, eff_first = {}, {}
                        for e, vec in edits.items():
                            ivs = [Intervention(m, vec, 1.0)] if e == "patch" else [inject, Intervention(m, vec, 1.0)]
                            r = self.backend.run(zs, interventions=ivs)
                            eff[e] = r.logprob_per_token - base_lp
                            eff_first[e] = r.first_token_logprob - base_first
                        return eff, eff_first

                    real_first: dict[str, dict[str, np.ndarray]] = {}
                    for src, Uk in U.items():
                        Ukk = Uk[:, :k]
                        real[src], real_first[src] = run_edits(edits_for((d_m @ Ukk) @ Ukk.T, (ref_m @ Ukk) @ Ukk.T))
                    for Q in randoms:
                        Qk = Q[:, :, :k]  # (n, d, k)
                        proj_d = t.einsum("ndk,nk->nd", Qk, t.einsum("ndk,nd->nk", Qk, d_m))
                        proj_r = t.einsum("ndk,nk->nd", Qk, t.einsum("ndk,nd->nk", Qk, ref_m))
                        eff, eff_first = run_edits(edits_for(proj_d, proj_r))
                        for e in PATCH_EDITS:
                            ctl[e].append(eff[e])
                            ctl_first[e].append(eff_first[e])
                    for src in U:
                        row[src] = {}
                        for e in PATCH_EDITS:
                            r_e, c_e = real[src][e], np.stack(ctl[e])
                            sign = -1.0 if e == "remove" else 1.0
                            row[src][e] = {"effect": float(np.mean(r_e)), "retained": None if full == 0 else float(np.mean(r_e) / full),
                                           "excess_vs_random": paired_excess_test(sign * r_e, sign * c_e, brng, n_boot=cfg.n_boot).__dict__,
                                           "first_token_retained": None if full_first == 0 else float(np.mean(real_first[src][e]) / full_first)}
                    row["random"] = {e: {"effect": float(np.mean(np.stack(ctl[e]))),
                                         "retained": None if full == 0 else float(np.mean(np.stack(ctl[e])) / full),
                                         "first_token_retained": None if full_first == 0 else float(np.mean(np.stack(ctl_first[e])) / full_first)}
                                     for e in PATCH_EDITS}
                    entry["rows"][str(k)] = row
                    self.log.info("[%s] L%d subspace %s at m=%d k=%d: keep retains own %.2f / other tasks %s / background %s / random %.2f; "
                                  "patch alone own %.2f / other tasks %s / random %.2f; remove leaves own %.2f (random %.2f)",
                                  name, layer, c, m, k, row["own"]["keep"]["retained"] or float("nan"),
                                  _fmt((row.get("other_tasks") or {}).get("keep", {}).get("retained")),
                                  _fmt((row.get("background") or {}).get("keep", {}).get("retained")),
                                  row["random"]["keep"]["retained"] or float("nan"),
                                  row["own"]["patch"]["retained"] or float("nan"),
                                  _fmt((row.get("other_tasks") or {}).get("patch", {}).get("retained")),
                                  row["random"]["patch"]["retained"] or float("nan"),
                                  row["own"]["remove"]["retained"] or float("nan"), row["random"]["remove"]["retained"] or float("nan"))
                # k90: the smallest k at which the edit reaches the share (None if the grid does not reach it)
                entry["k90"] = {}
                for src in list(U) + ["random"]:
                    entry["k90"][src] = {}
                    for e in ("keep", "patch"):
                        entry["k90"][src][e] = next((k for k in sc.k_grid if (entry["rows"][str(k)][src][e]["retained"] or -1) >= sc.handover_share), None)
                entries.append(entry)
            out["constructions"][c] = {"full_effect": full, "alpha": spec["alpha"], "handover_read_point": handover, "read_points": entries}
        return out


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


def _construction_rows(info: dict[str, Any], gap: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for c, spec in info["constructions"].items():
        row: dict[str, Any] = {"alpha": spec["alpha"], "rho": spec["rho"], "source": spec["source"],
                               "qualified_here": spec.get("qualified_here"),
                               "improvement_per_token": spec["improvement_per_token"],
                               "gap_fraction": (spec["improvement_per_token"] / gap if gap else None),
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
        bw = ((info.get("block_writing") or {}).get("trajectories") or {}).get(c)
        if bw:
            row["block_writing"] = {tag: {k: v[k] for k in ("dominant_block", "dominant_share", "entropy_ratio",
                                                            "rank_correlation_with_natural")} for tag, v in bw.items()}
        rows[c] = row
    return rows


def _task_summary(result: dict[str, Any]) -> dict[str, Any]:
    """The per-task rows of ``core/summary.json``: per layer, strength factor and construction, the strength,
    the effect and the alignment summary against each natural trajectory and each other construction; the
    coherence, the cross-strength cosines and the patch test where present."""
    gap = result["fewshot_gap_per_token"]
    out: dict[str, Any] = {"layers": result["layers"], "primary_layer": result["primary_layer"],
                           "layer_roles": result.get("layer_roles", {}), "fewshot_gap_per_token": gap, "per_layer": {}}
    for layer, info in result["per_layer"].items():
        entry: dict[str, Any] = {"constructions": _construction_rows(info, gap),
                                 "natural_block_writing": {tag: {k: v[k] for k in ("dominant_block", "dominant_share", "entropy_ratio")}
                                                           for tag, v in (((info.get("block_writing") or {}).get("trajectories") or {}).get("icl") or {}).items()},
                                 "ceiling_final": {k: v["median"][-1] for k, v in info["ceilings"].items()},
                                 "coherence_final": (info.get("generic_response") or {}).get("coherence", [None])[-1],
                                 "other_strengths": {}}
        for f, sub in (info.get("other_strengths") or {}).items():
            entry["other_strengths"][f] = {"constructions": _construction_rows(sub, gap),
                                           "coherence_final": (sub.get("generic_response") or {}).get("coherence", [None])[-1]}
        if "cross_strength" in info:
            entry["cross_strength"] = {f: {c: {k: e.get(k) for k in ("at_injection", "min_after_injection", "final")}
                                           for c, e in sub.items() if c in CONSTRUCTIONS}
                                       for f, sub in info["cross_strength"].items()}
        if "patch" in info:
            entry["patch"] = {c: [{"read_point": r["read_point"], "depth_fraction": r["depth_fraction"],
                                   "alignment": r["alignment_generic_removed"],
                                   **{e: {"retained": r[e]["retained"], "random_retained": r[e]["random_retained_mean"],
                                          "p": r[e]["excess_vs_random"]["p_value"],
                                          "first_token_retained": r[e]["first_token_retained"]} for e in PATCH_EDITS}}
                                  for r in v["rows"]] for c, v in info["patch"]["constructions"].items()}
        if "subspace" in info:
            entry["subspace"] = {c: [{"read_point": p["read_point"], "depth_fraction": p["depth_fraction"], "source": p["source"],
                                      "k90": p["k90"], "per_prompt_keep": p["per_prompt_keep"],
                                      "explained": p["spectrum"]["explained"], "participation_ratio": p["spectrum"]["participation_ratio"],
                                      "overlap": p["overlap"],
                                      "keep_retained": {src: [p["rows"][str(k)][src]["keep"]["retained"] for k in info["subspace"]["k_grid"]]
                                                        for src in p["rows"][str(info["subspace"]["k_grid"][0])]},
                                      "patch_retained": {src: [p["rows"][str(k)][src]["patch"]["retained"] for k in info["subspace"]["k_grid"]]
                                                         for src in p["rows"][str(info["subspace"]["k_grid"][0])]}}
                                     for p in v["read_points"]] for c, v in info["subspace"]["constructions"].items()}
        out["per_layer"][layer] = entry
    return out


def format_summary(summary: dict[str, Any]) -> str:
    """One line per task, layer and construction at the canonical strength: effect, alignment with the natural
    trajectory (generic response removed: injection, peak, final, floor, label), the same at the other
    strengths, the cross-strength final cosine, the cross-construction finals, the coherence and the ceiling."""
    lines = ["| task | layer | construction | gap frac | inj | peak (depth) | final | floor | label | other strengths: gap, final (floor) | x1 vs other | vs pca | vs fv | vs learned | coherence | ceiling |",
             "|" + "---|" * 16]
    for task, t in summary.get("tasks", {}).items():
        for layer, info in t["per_layer"].items():
            for c, row in info["constructions"].items():
                a = row["alignment"].get("icl", {})
                g = a.get("generic_removed", a)
                cross = []
                for o in CONSTRUCTIONS:
                    x = row["alignment"].get(o)
                    x = x.get("generic_removed", x) if x else None
                    cross.append("" if o == c else ("-" if not x else f"{x['final']:.2f}"))
                others = []
                for f, sub in info.get("other_strengths", {}).items():
                    r2 = sub["constructions"].get(c)
                    if r2:
                        a2 = r2["alignment"].get("icl", {})
                        g2 = a2.get("generic_removed", a2)
                        others.append(f"x{f}: {_fmt(r2['gap_fraction'])}, {g2.get('final', float('nan')):.2f} ({g2.get('floor_at_final', float('nan')):.2f})")
                xs = [f"{v[c]['final']:.2f}" for f, v in info.get("cross_strength", {}).items() if c in v and v[c].get("final") is not None]
                gap = "-" if row["gap_fraction"] is None else f"{row['gap_fraction']:.2f}"
                coh = info.get("coherence_final")
                ceil = info["ceiling_final"]
                lines.append(
                    f"| {task} | {layer}{'*' if int(layer) == t['primary_layer'] else ''} | {c} | {gap} | "
                    f"{g.get('at_injection', a.get('at_injection', float('nan'))):.2f} | {g.get('peak', float('nan')):.2f} ({g.get('peak_depth_fraction', a.get('peak_depth_fraction', float('nan'))):.2f}) | "
                    f"{g.get('final', float('nan')):.2f} | {g.get('floor_at_final', float('nan')):.2f} | {g.get('label', '')} | {'; '.join(others) or '-'} | "
                    f"{', '.join(xs) or '-'} | {cross[0]} | {cross[1]} | {cross[2]} | {'-' if coh is None else f'{coh:.2f}'} | "
                    f"{ceil.get('icl~icl2_generic_removed', ceil.get('icl~icl2', float('nan'))):.2f} |")
            if "patch" in info:
                for c, rows in info["patch"].items():
                    for r in rows:
                        lines.append(f"| {task} | {layer} | patch {c} @{r['read_point']} ({r['depth_fraction']:.2f}, align {_fmt(r['alignment'])}) | "
                                     f"remove keeps {_fmt(r['remove']['retained'])} (rand {_fmt(r['remove']['random_retained'])}, p {r['remove']['p']:.3f}); "
                                     f"keep retains {_fmt(r['keep']['retained'])} (rand {_fmt(r['keep']['random_retained'])}, p {r['keep']['p']:.3f}); "
                                     f"natural patch {_fmt(r['patch']['retained'])} (rand {_fmt(r['patch']['random_retained'])}, p {r['patch']['p']:.3f}) |" + " |" * 12)
            for c, pts in (info.get("subspace") or {}).items():
                for p in pts:
                    keep = "; ".join(f"{src} {'/'.join(_fmt(v) for v in vals)}" for src, vals in p["keep_retained"].items())
                    lines.append(f"| {task} | {layer} | subspace {c} @{p['read_point']} ({p['source']}, per-prompt keep {_fmt(p['per_prompt_keep'])}) | "
                                 f"keep by k: {keep}; k90 own {p['k90'].get('own', {}).get('keep')} other {p['k90'].get('other_tasks', {}).get('keep')}; "
                                 f"explained {'/'.join(f'{v:.2f}' for v in p['explained'])} |" + " |" * 12)
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def run_trajectories(cfg: TrajectoriesConfig, fv_run: str | Path, learned_run: str | Path,
                     run_id: str | None = None, config_path: str | None = None) -> Path:
    return Trajectories(cfg, Path(fv_run), Path(learned_run), run_id, config_path).run()
