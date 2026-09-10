"""Automatic figures (matplotlib, headless).

Primary figures per validated task: control magnitude, layerwise amplification,
dimensional expansion, new-subspace creation, real vs random controls; cross-task
heatmaps; diagnostics (calibration grid, cross-seed stability).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .config import Config  # noqa: E402

if TYPE_CHECKING:
    from .pipeline import TaskState

# Reference palette (fixed order, never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REAL = SERIES[0]
RANDOM = "#9a9995"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#2a78d6", "#0d2f5c"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#eb6834", "#d9d8d3", "#2a78d6"])

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _ci_line(ax: plt.Axes, x: np.ndarray, summ: dict[str, list[float]], color: str, label: str) -> None:
    med = np.array(summ["median"], dtype=np.float64)
    lo = np.array(summ["low"], dtype=np.float64)
    hi = np.array(summ["high"], dtype=np.float64)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, med, color=color, label=label)


def _random_band(ax: plt.Axes, x: np.ndarray, curves: np.ndarray, label: str = "matched random controls") -> None:
    if curves.shape[0] == 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan columns below the intervention layer
        lo = np.nanpercentile(curves, 5, axis=0)
        hi = np.nanpercentile(curves, 95, axis=0)
        med = np.nanmedian(curves, axis=0)
    ax.fill_between(x, lo, hi, color=RANDOM, alpha=0.25, linewidth=0)
    ax.plot(x, med, color=RANDOM, linestyle="--", label=label)


def _mark_intervention(ax: plt.Axes, layer: int) -> None:
    ax.axvline(layer, color=TEXT2, linewidth=0.8, linestyle=":", label=None)


# --------------------------------------------------------------------------- #
# Per-task primary figures
# --------------------------------------------------------------------------- #


def task_figures(root: Path, st: "TaskState", cfg: Config) -> None:
    if st.profile is None or st.comparison is None:
        return
    figdir = root / "figures"
    dpi = cfg.figures.dpi
    p = st.profile
    ls, L = p.intervention_layer, p.n_layers
    layers = np.arange(L + 1)
    blocks = np.arange(L)
    comp = st.comparison["primary"]["metrics"]
    gate = st.comparison["gate_kinds"]
    rand = {m: _random_curves(root, st.name, m, gate) for m in comp}

    # 1. control magnitude
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _random_band(ax, layers, rand["magnitude"])
    _ci_line(ax, layers, p.summaries["magnitude"], REAL, "control direction")
    _mark_intervention(ax, ls)
    ax.set_xlabel("residual read point l")
    ax.set_ylabel("median S_l = ||δ_l|| / ||h_l||")
    ax.set_title(f"{st.name}: control magnitude through depth", color=TEXT)
    ax.legend()
    _save(fig, figdir / f"{st.name}_magnitude.png", dpi)

    # 2. amplification
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _random_band(ax, blocks, rand["log_gain"])
    _ci_line(ax, blocks, p.summaries["log_gain"], REAL, "control direction")
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    _mark_intervention(ax, ls)
    ax.set_xlabel("block l")
    ax.set_ylabel("median log G_l")
    ax.set_title(f"{st.name}: layerwise amplification", color=TEXT)
    ax.legend()
    _save(fig, figdir / f"{st.name}_log_gain.png", dpi)

    # 3. dimensional expansion (two panels, one axis each)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, name, ylabel in ((axes[0], "d_eff", "effective rank d_eff(D_l)"), (axes[1], "d90", "d90(D_l)")):
        _random_band(ax, layers, rand[name])
        ax.plot(layers, p.metric_curve(name), color=REAL, marker="o", markersize=3, label="control direction")
        if name == "d_eff":
            ax.plot(layers, p.d_eff_uncentered, color=SERIES[1], linestyle="-.", label="uncentered d_eff")
        _mark_intervention(ax, ls)
        ax.set_xlabel("residual read point l")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"{st.name}: dimensional expansion (undefined at the intervention layer)", color=TEXT)
    _save(fig, figdir / f"{st.name}_dimensionality.png", dpi)

    # 4. new-subspace creation
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _random_band(ax, blocks, rand["new_subspace_uncentered"], "random (uncentered)")
    ax.plot(blocks, p.new_subspace_uncentered, color=REAL, marker="o", markersize=3, label="N_l^unc (propagation vs creation)")
    ax.plot(blocks, p.new_subspace, color=SERIES[1], marker="s", markersize=3, label="N_l (centered)")
    _mark_intervention(ax, ls)
    ax.set_ylim(0, 1)
    ax.set_xlabel("block l")
    ax.set_ylabel("fraction of b_l outside P_l")
    ax.set_title(f"{st.name}: new-subspace creation", color=TEXT)
    ax.legend(fontsize=8)
    _save(fig, figdir / f"{st.name}_new_subspace.png", dpi)

    # 5. alignment + conversion
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    _random_band(axes[0], layers, rand["alignment"])
    _ci_line(axes[0], layers, p.summaries["alignment"], REAL, "control direction")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("median A_l = |cos(δ_l, v)|")
    axes[0].set_xlabel("residual read point l")
    _random_band(axes[1], blocks, rand["conversion"])
    _ci_line(axes[1], blocks, p.summaries["conversion"], REAL, "control direction")
    axes[1].set_ylabel("median C_l = ||b_l|| / ||δ_l||")
    axes[1].set_xlabel("block l")
    for ax in axes:
        _mark_intervention(ax, ls)
        ax.legend()
    fig.suptitle(f"{st.name}: control alignment and block conversion", color=TEXT)
    _save(fig, figdir / f"{st.name}_alignment_conversion.png", dpi)

    # 6. real vs random: per-layer z-scores for every primary metric
    metrics = ["magnitude", "log_gain", "conversion", "alignment", "d_eff", "d90", "new_subspace", "new_subspace_uncentered",
               "task_alignment", "gradient_alignment"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.5))
    for ax, m in zip(axes.ravel(), metrics):
        x = layers if len(comp[m]["real"]) == L + 1 else blocks
        z = np.array([r["z"] if r["z"] is not None else np.nan for r in comp[m]["per_layer"]], dtype=np.float64)
        ax.bar(x, z, color=REAL, width=0.8, linewidth=0)
        ax.axhline(0, color=TEXT2, linewidth=0.8)
        ax.axhline(2, color=RANDOM, linewidth=0.8, linestyle="--")
        ax.axhline(-2, color=RANDOM, linewidth=0.8, linestyle="--")
        _mark_intervention(ax, ls)
        ax.set_title(m, color=TEXT, fontsize=9)
        ax.set_ylabel("z vs random")
    fig.suptitle(f"{st.name}: real control vs {st.comparison['primary']['n_random']} matched random controls "
                 f"({'+'.join(gate)})", color=TEXT)
    _save(fig, figdir / f"{st.name}_vs_random.png", dpi)

    _structured_nulls_figure(root, figdir, st, dpi)
    _readouts_figure(root, figdir, st, dpi)
    # diagnostics
    _calibration_figure(figdir, st, dpi)
    _stability_figure(figdir, st, dpi)


KIND_COLORS = {"isotropic": RANDOM, "orthogonal": "#6d6c68", "covariance": SERIES[1], "other_task": SERIES[2],
               "demo_variation": SERIES[6]}


def _structured_nulls_figure(root: Path, figdir: Path, st: "TaskState", dpi: int) -> None:
    """Real control vs the median (and 5-95 % band) of every control kind."""
    p = st.profile
    assert p is not None and st.comparison is not None
    ls, L = p.intervention_layer, p.n_layers
    kinds = [k for k in KIND_COLORS if k in st.comparison["by_kind"]]
    if not kinds:
        return
    specs = [("log_gain", "median log G_l", np.arange(L)), ("alignment", "median A_l", np.arange(L + 1)),
             ("d_eff", "d_eff(D_l)", np.arange(L + 1)), ("new_subspace_uncentered", "N_l^unc", np.arange(L))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (m, ylabel, x) in zip(axes.ravel(), specs):
        for k in kinds:
            curves = _random_curves(root, st.name, m, [k])
            if curves.shape[0] == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(curves, axis=0)
                lo, hi = np.nanpercentile(curves, 5, axis=0), np.nanpercentile(curves, 95, axis=0)
            n = st.comparison["by_kind"][k]["n_random"]
            ax.fill_between(x, lo, hi, color=KIND_COLORS[k], alpha=0.12, linewidth=0)
            ax.plot(x, med, color=KIND_COLORS[k], linestyle="--", linewidth=1.5, label=f"{k} (n={n})")
        ax.plot(x, p.metric_curve(m), color=REAL, linewidth=2.2, label="control direction")
        _mark_intervention(ax, ls)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("block l" if len(x) == L else "residual read point l")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{st.name}: control direction vs structured nulls (median, 5-95 % band)", color=TEXT)
    _save(fig, figdir / f"{st.name}_structured_nulls.png", dpi)


def _readouts_figure(root: Path, figdir: Path, st: "TaskState", dpi: int) -> None:
    """Direction-specific readouts (D17): task-direction and gradient alignment vs every control kind."""
    p = st.profile
    assert p is not None and st.comparison is not None
    ls, L = p.intervention_layer, p.n_layers
    if p.readout_diagnostics is None or not (p.readout_diagnostics.get("task_directions_available")
                                             or p.readout_diagnostics.get("gradients_available")):
        return
    kinds = [k for k in KIND_COLORS if k in st.comparison["by_kind"]]
    x = np.arange(L + 1)
    specs = [("task_alignment", "median cos(δ_l, v_{t,l})  (task's own direction at l)"),
             ("gradient_alignment", "median cos(δ_l, ∇ log p(target))")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (m, ylabel) in zip(axes, specs):
        for k in kinds:
            curves = _random_curves(root, st.name, m, [k])
            if curves.shape[0] == 0 or np.all(np.isnan(curves)):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(curves, axis=0)
                lo, hi = np.nanpercentile(curves, 5, axis=0), np.nanpercentile(curves, 95, axis=0)
            n = st.comparison["by_kind"][k]["n_random"]
            ax.fill_between(x, lo, hi, color=KIND_COLORS[k], alpha=0.12, linewidth=0)
            ax.plot(x, med, color=KIND_COLORS[k], linestyle="--", linewidth=1.5, label=f"{k} (n={n})")
        _ci_line(ax, x, p.summaries[m], REAL, "control direction")
        ax.axhline(0, color=TEXT2, linewidth=0.8)
        _mark_intervention(ax, ls)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("residual read point l")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{st.name}: direction-specific readouts (median, 5-95 % band of each null)", color=TEXT)
    _save(fig, figdir / f"{st.name}_readouts.png", dpi)


_CURVE_CACHE: dict[tuple[Path, str], list[dict]] = {}


def _random_curves(root: Path, task: str, metric: str, kinds: list[str]) -> np.ndarray:
    """Per-control metric curves of the given kinds, from the task's layerwise.json."""
    from .runinfo import read_json

    key = (root, task)
    if key not in _CURVE_CACHE:
        _CURVE_CACHE[key] = read_json(root / "core" / "tasks" / task / "layerwise.json")["controls"]
    rows = [[np.nan if x is None else x for x in r["curves"][metric]] for r in _CURVE_CACHE[key] if r["kind"] in kinds]
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 0))


def _calibration_figure(figdir: Path, st: "TaskState", dpi: int) -> None:
    cal = st.calibration
    if cal is None or not cal.grid:
        return
    layers = sorted({g.layer for g in cal.grid})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, layer in enumerate(layers):
        pts = sorted([g for g in cal.grid if g.layer == layer], key=lambda g: g.rho)
        color = SERIES[i % len(SERIES)]
        ax.plot([g.rho for g in pts], [g.test.mean_diff for g in pts], color=color, marker="o", markersize=3,
                label=f"layer {layer}")
        rel = [g for g in pts if g.reliable]
        if rel:
            ax.scatter([g.rho for g in rel], [g.test.mean_diff for g in rel], s=60, facecolors="none",
                       edgecolors=color, linewidths=1.5)
        for g in pts:
            if g.screen is not None:
                nulls = [c["mean_diff"] for c in g.screen["controls"]]
                ax.scatter([g.rho] * len(nulls), nulls, s=8, color=RANDOM, alpha=0.6, linewidths=0)
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("relative strength ρ")
    ax.set_ylabel("mean Δ log p(target)/token")
    sel = "none" if cal.selected is None else f"layer {cal.selected.layer}, ρ={cal.selected.rho:g}"
    ax.set_title(f"{st.name}: calibration grid (selected: {sel}; circles = reliable, grey = random screen)", color=TEXT, fontsize=8)
    ax.legend(fontsize=7)
    _save(fig, figdir / f"{st.name}_calibration_grid.png", dpi)


def _stability_figure(figdir: Path, st: "TaskState", dpi: int) -> None:
    if not st.directions:
        return
    layers = sorted(st.directions)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(layers, [st.directions[l].stability for l in layers], color=REAL, width=0.8, linewidth=0)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("candidate layer")
    axes[0].set_ylabel("min pairwise |cos| across seeds")
    axes[1].bar(layers, [st.directions[l].pooled_explained_variance_ratio for l in layers], color=SERIES[1], width=0.8, linewidth=0)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("candidate layer")
    axes[1].set_ylabel("pooled PC1 explained-variance ratio")
    fig.suptitle(f"{st.name}: cross-seed direction stability", color=TEXT)
    _save(fig, figdir / f"{st.name}_stability.png", dpi)


# --------------------------------------------------------------------------- #
# Cross-task heatmaps
# --------------------------------------------------------------------------- #


def heatmaps(root: Path, states: dict[str, "TaskState"], cfg: Config) -> None:
    valid = {n: s for n, s in states.items() if s.profile is not None}
    if not valid:
        return
    figdir = root / "figures"
    L = next(iter(valid.values())).profile.n_layers  # type: ignore[union-attr]
    specs = [
        ("log_gain", "median log G_l", "diverging"),
        ("d_eff", "d_eff(D_l)", "sequential"),
        ("new_subspace_uncentered", "N_l^unc", "sequential"),
        ("new_subspace", "N_l (centered)", "sequential"),
    ]
    for metric, label, kind in specs:
        rows, names = [], []
        for n, s in valid.items():
            curve = s.profile.metric_curve(metric)  # type: ignore[union-attr]
            rows.append(curve)
            names.append(f"{n} (l*={s.profile.intervention_layer})")  # type: ignore[union-attr]
        M = np.array(rows, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(max(6, 0.28 * M.shape[1] + 2), 0.5 * len(names) + 1.6))
        if kind == "diverging":
            vmax = float(np.nanmax(np.abs(M))) if np.any(~np.isnan(M)) else 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.imshow(M, aspect="auto", cmap=DIV_CMAP, norm=norm)
        else:
            im = ax.imshow(M, aspect="auto", cmap=SEQ_CMAP)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("block l" if M.shape[1] == L else "residual read point l")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_title(f"task x layer: {label} (blank = undefined)", color=TEXT)
        _save(fig, figdir / f"heatmap_{metric}.png", cfg.figures.dpi)


def _function_vector_figure(root: Path, st: "TaskState", cfg: Config) -> None:
    """Average indirect effect of every attention head, with the selected heads marked (D21)."""
    assert st.head_effects is not None
    E = st.head_effects
    fig, ax = plt.subplots(figsize=(max(4.0, 0.22 * E.shape[1] + 1.5), max(3.2, 0.16 * E.shape[0] + 1.2)))
    lim = float(np.nanmax(np.abs(E))) or 1.0
    im = ax.imshow(E, aspect="auto", cmap=DIV_CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation="nearest")
    if st.fv is not None:
        ax.scatter([h["head"] for h in st.fv.heads], [h["layer"] for h in st.fv.heads], marker="s", s=28,
                   facecolors="none", edgecolors=TEXT, linewidths=1.0, label="selected heads")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("attention head")
    ax.set_ylabel("block")
    ax.set_title(f"{st.name}: head indirect effects on deranged-label prompts (Δ log p / token)", color=TEXT, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    _save(fig, root / "figures" / f"{st.name}_function_vector.png", cfg.figures.dpi)


def make_all_figures(root: Path, states: dict[str, "TaskState"], cfg: Config) -> None:
    for st in states.values():
        if st.head_effects is not None:
            _function_vector_figure(root, st, cfg)
        task_figures(root, st, cfg)
    heatmaps(root, states, cfg)
