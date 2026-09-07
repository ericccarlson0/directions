"""Automatic figure generation for a pilot run.

Figures follow ``docs/EXPERIMENT.md`` section "Primary Figures". All plotting
is non-interactive (Agg) so runs are headless and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .layerwise import LayerwiseMeasurement, quiet_nan_reductions  # noqa: E402
from .serialization import RunDirectory  # noqa: E402

REAL_COLOR = "#1f4e79"
CTRL_COLOR = "#b0b0b0"


def _ci_arrays(summaries: dict, key: str, n: int):
    med = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for k, v in summaries.get(key, {}).items():
        i = int(k)
        if i < n:
            med[i], lo[i], hi[i] = v["median"], v["ci_lo"], v["ci_hi"]
    return med, lo, hi


def _band(ax, x, med, lo, hi, label, color=REAL_COLOR):
    ax.plot(x, med, "-o", ms=3, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)


def _controls_band(ax, x, controls: dict, key: str, n: int):
    c = controls.get(key)
    if not c:
        return
    q05 = np.array([np.nan if v is None else v for v in c["q05"]][:n], dtype=float)
    q50 = np.array([np.nan if v is None else v for v in c["q50"]][:n], dtype=float)
    q95 = np.array([np.nan if v is None else v for v in c["q95"]][:n], dtype=float)
    ax.fill_between(x, q05, q95, color=CTRL_COLOR, alpha=0.45, linewidth=0,
                    label=f"random controls (5-95%, n={c['n_controls']})")
    ax.plot(x, q50, "--", color="#606060", lw=1, label="random control median")


def _finish(fig, ax, title, xlabel, ylabel, path: Path, dpi: int):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def per_task_figures(
    rundir: RunDirectory,
    task: str,
    m: LayerwiseMeasurement,
    controls: dict,
    dpi: int = 150,
    ext: str = "png",
) -> list[str]:
    """Figures 1-4 for a single task/control pair."""
    with quiet_nan_reductions():
        return _per_task_figures(rundir, task, m, controls, dpi, ext)


def _per_task_figures(rundir, task, m, controls, dpi, ext) -> list[str]:
    paths = []
    n_resid = m.n_layers + 1
    resid_x = np.arange(n_resid)
    block_x = np.arange(m.n_layers)

    # 1. control magnitude through depth
    fig, ax = plt.subplots(figsize=(6, 3.6))
    med, lo, hi = _ci_arrays(m.summaries, "S", n_resid)
    _controls_band(ax, resid_x, controls, "S", n_resid)
    _band(ax, resid_x, med, lo, hi, "control direction")
    ax.axvline(m.layer, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    p = rundir.figure_path(f"{task}__1_magnitude", ext)
    _finish(fig, ax, f"{task}: control magnitude through depth",
            "residual index $l$", r"median $S_l = \|\delta_l\|/\|h_l^{base}\|$", p, dpi)
    paths.append(str(p))

    # 2. layerwise amplification
    fig, ax = plt.subplots(figsize=(6, 3.6))
    med, lo, hi = _ci_arrays(m.summaries, "log_G", m.n_layers)
    _controls_band(ax, block_x, controls, "log_G", m.n_layers)
    _band(ax, block_x, med, lo, hi, "control direction")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(m.layer, color="k", ls=":", lw=1)
    p = rundir.figure_path(f"{task}__2_amplification", ext)
    _finish(fig, ax, f"{task}: layerwise amplification",
            "block $l$", r"median $\log G_l$", p, dpi)
    paths.append(str(p))

    # 3. dimensional expansion
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _controls_band(ax, resid_x, controls, "d_eff", n_resid)
    ax.plot(resid_x, m.d_eff, "-o", ms=3, color=REAL_COLOR, label=r"$d_{eff}(D_l)$")
    ax.plot(resid_x, m.d90, "-s", ms=3, color="#c0504d", label=r"$d_{90}(D_l)$")
    ax.axvline(m.layer, color="k", ls=":", lw=1)
    p = rundir.figure_path(f"{task}__3_dimensionality", ext)
    _finish(fig, ax, f"{task}: dimensional expansion", "residual index $l$",
            "effective dimensionality", p, dpi)
    paths.append(str(p))

    # 4. new-subspace creation
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _controls_band(ax, block_x, controls, "N", m.n_layers)
    ax.plot(block_x, m.N[: m.n_layers], "-o", ms=3, color=REAL_COLOR, label=r"$N_l$ (centered $P_l$)")
    ax.plot(block_x, m.N_uncentered[: m.n_layers], "-^", ms=3, color="#4f8f3f",
            label=r"$N_l$ (uncentered $P_l$)")
    ax.axvline(m.layer, color="k", ls=":", lw=1)
    ax.set_ylim(-0.02, 1.02)
    p = rundir.figure_path(f"{task}__4_new_subspace", ext)
    _finish(fig, ax, f"{task}: new-subspace creation", "block $l$",
            r"$N_l$ (fraction of $B_l$ outside $P_l$)", p, dpi)
    paths.append(str(p))
    return paths


def heatmaps(
    rundir: RunDirectory,
    measurements: dict[str, LayerwiseMeasurement],
    dpi: int = 150,
    ext: str = "png",
) -> list[str]:
    """Figure 5: task x layer heatmaps for log G, d_eff and N."""
    with quiet_nan_reductions():
        return _heatmaps(rundir, measurements, dpi, ext)


def _heatmaps(rundir, measurements, dpi, ext) -> list[str]:
    if not measurements:
        return []
    tasks = sorted(measurements)
    n_layers = max(m.n_layers for m in measurements.values())
    panels = {
        "log_G": (r"median $\log G_l$", "RdBu_r", lambda m: np.nanmedian(np.log(m.G), axis=1)[:n_layers]),
        "d_eff": (r"$d_{eff}(D_l)$", "viridis", lambda m: m.d_eff[: n_layers + 1]),
        "N": (r"$N_l$", "magma", lambda m: m.N[:n_layers]),
    }
    paths = []
    for key, (title, cmap, get) in panels.items():
        width = n_layers + 1 if key == "d_eff" else n_layers
        grid = np.full((len(tasks), width), np.nan)
        for i, t in enumerate(tasks):
            row = np.asarray(get(measurements[t]), dtype=float)
            grid[i, : row.size] = row
        fig, ax = plt.subplots(figsize=(max(6, width * 0.28), 1.1 + 0.45 * len(tasks)))
        vmax = np.nanmax(np.abs(grid)) if key == "log_G" else None
        im = ax.imshow(
            grid, aspect="auto", cmap=cmap, interpolation="nearest",
            **({"vmin": -vmax, "vmax": vmax} if key == "log_G" and np.isfinite(vmax or np.nan) else {}),
        )
        ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
        ax.set_xlabel("block $l$" if key != "d_eff" else "residual index $l$")
        ax.set_title(f"task x layer: {title}", fontsize=10)
        for i, t in enumerate(tasks):
            ax.plot(measurements[t].layer, i, "w|", ms=10, mew=1.5)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        p = rundir.figure_path(f"heatmap_{key}", ext)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        paths.append(str(p))
    return paths


def real_vs_random(
    rundir: RunDirectory,
    task: str,
    m: LayerwiseMeasurement,
    controls: dict,
    dpi: int = 150,
    ext: str = "png",
) -> str | None:
    """Figure 6: real control direction against the matched random-control null."""
    with quiet_nan_reductions():
        return _real_vs_random(rundir, task, m, controls, dpi, ext)


def _real_vs_random(rundir, task, m, controls, dpi, ext) -> str | None:
    if not controls:
        return None
    keys = [k for k in ("S", "log_G", "C", "d_eff", "N") if k in controls]
    if not keys:
        return None
    obs = {
        "S": lambda: np.nanmedian(m.S, axis=1),
        "log_G": lambda: np.nanmedian(np.log(m.G), axis=1),
        "C": lambda: np.nanmedian(m.C, axis=1),
        "d_eff": lambda: m.d_eff,
        "N": lambda: m.N,
    }
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.1), squeeze=False)
    for ax, key in zip(axes[0], keys):
        c = controls[key]
        samples = np.array([[np.nan if v is None else v for v in row] for row in c["samples"]], float)
        n = samples.shape[1]
        x = np.arange(n)
        for row in samples:
            ax.plot(x, row, color=CTRL_COLOR, lw=0.7, alpha=0.7)
        ax.plot(x, obs[key]()[:n], color=REAL_COLOR, lw=1.8, label="control direction")
        ax.axvline(m.layer, color="k", ls=":", lw=1)
        if key == "S":
            ax.set_yscale("log")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{task}: real vs matched random controls", fontsize=10)
    fig.tight_layout()
    p = rundir.figure_path(f"{task}__6_real_vs_random", ext)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return str(p)


def calibration_figure(
    rundir: RunDirectory, task: str, calib: dict, dpi: int = 150, ext: str = "png"
) -> str:
    """Diagnostic: the layer x strength calibration grid."""
    grid = calib["grid"]
    layers = sorted({p["layer"] for p in grid})
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for layer in layers:
        pts = sorted([p for p in grid if p["layer"] == layer], key=lambda p: p["rho"])
        ax.plot([p["rho"] for p in pts], [p["metric"] for p in pts], "-o", ms=3, label=f"layer {layer}")
    ax.axhline(calib["baseline_metric"], color="k", ls="--", lw=1, label="baseline (no steering)")
    sel = calib.get("selected")
    if sel:
        ax.plot(sel["rho"], sel["metric"], "*", ms=16, color="#c0504d", label="selected")
    ax.set_xscale("log")
    p = rundir.figure_path(f"{task}__0_calibration", ext)
    _finish(fig, ax, f"{task}: intervention calibration ({calib['metric']})",
            r"relative strength $\rho$", calib["metric"], p, dpi)
    return str(p)


def stability_figure(
    rundir: RunDirectory, extractions: dict[str, dict], dpi: int = 150, ext: str = "png"
) -> str | None:
    """Diagnostic: cross-seed direction stability by candidate layer, per task."""
    if not extractions:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for task, ext_data in sorted(extractions.items()):
        layers = sorted(int(k) for k in ext_data["layers"])
        ax.plot(layers, [ext_data["layers"][str(l)]["stability_mean"] for l in layers],
                "-o", ms=4, label=task)
    ax.set_ylim(0, 1.02)
    p = rundir.figure_path("extraction_stability", ext)
    _finish(fig, ax, "cross-seed control-direction stability", "candidate layer $l$",
            r"mean pairwise $|\cos|$", p, dpi)
    return str(p)
