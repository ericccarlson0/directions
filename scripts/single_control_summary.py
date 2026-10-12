"""The close-out statistics and figures of the single-control line (docs/SINGLE_CONTROL.md).

Reads the downloaded run directories named in ``RUNS`` under ``--runs-dir`` (one ``run-<workflow run id>/``
directory per run, as ``scripts/runpod_log.py <id> --download <runs-dir>/run-<id>`` writes them; only the
``.json`` files are read), writes tidy CSVs, ``stats.json`` and four figures to ``--out``. The 8B, post-trained
8B and Gemma 4 run directories of D34, D37 and D38 were removed from the US-CA-2 volume to free space
(docs/INFRA.md, item 6); their numbers come from STATUS.md (``STATUS_ONLY`` below, every row marked
``source=status`` in the CSVs) and enter the figures as ranges, never as per-task points.

usage: uv run --no-project --with numpy --with matplotlib python scripts/single_control_summary.py \
           --runs-dir <dir with run-<id>/> --out docs/single_control
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261012  # bootstrap seed (fixed; recorded in stats.json)
N_BOOT = 10000
SHARE = 0.9  # the hand-over rule of D33/D37: the first read point whose keep-only edit retains >= 0.9 of the effect
NON_LEXICAL = {"add_3", "arithmetic_words"}  # multi-token targets: the query-token edit rows are uninformative (STATUS, limitations)

# checkpoint -> (label, number of blocks, run directories relative to --runs-dir); seed 20260916 on Qwen3, 20260907 on OLMo 3
RUNS: dict[str, dict[str, Any]] = {
    "qwen3_0.6b": {"label": "Qwen3 0.6B", "L": 28, "params_b": 0.6,
                   "landmarks": "run-35685354053/landmarks_qwen3_0.6b_seed20260916",
                   "source": "run-35689345430/source_qwen3_0.6b_seed20260916",
                   "trajectories10": "run-35417248145/trajectories10_qwen3_0.6b_seed20260916",
                   "trajectories11": "run-35464684771/trajectories11_qwen3_0.6b_seed20260916",
                   "landmarks_wd": "run-37086330214/landmarks_wd_qwen3_0.6b_seed20260916",
                   "source_wd": "run-37086330214/source_wd_qwen3_0.6b_seed20260916"},
    "qwen3_1.7b": {"label": "Qwen3 1.7B", "L": 28, "params_b": 1.7,
                   "landmarks": "run-35682020094/landmarks_qwen3_1.7b_seed20260916",
                   "source": "run-35689544898/source_qwen3_1.7b_seed20260916",
                   "trajectories10": "run-35419811910/trajectories10_qwen3_1.7b_seed20260916",
                   "trajectories11": "run-35464728409/trajectories11_qwen3_1.7b_seed20260916",
                   "landmarks_wd": "run-37086359503/landmarks_wd_qwen3_1.7b_seed20260916",
                   "source_wd": "run-37086359503/source_wd_qwen3_1.7b_seed20260916"},
    "qwen3_4b": {"label": "Qwen3 4B", "L": 36, "params_b": 4.0,
                 "landmarks": "run-35682273805/landmarks_qwen3_4b_seed20260916",
                 "source": "run-35690463091/source_qwen3_4b_seed20260916",
                 "trajectories10": "run-35421215047/trajectories10_qwen3_4b_seed20260916",
                 "trajectories11": "run-35465035832/trajectories11_qwen3_4b_seed20260916",
                 "landmarks_wd": "run-37092043429/landmarks_wd_qwen3_4b_seed20260916",
                 "source_wd": "run-37092043429/source_wd_qwen3_4b_seed20260916"},
    "qwen3_8b": {"label": "Qwen3 8B", "L": 36, "params_b": 8.0,
                 "landmarks_wd": "run-37086384663/landmarks_wd_qwen3_8b_seed20260916",
                 "source_wd": "run-37086384663/source_wd_qwen3_8b_seed20260916"},
    "olmo3_7b": {"label": "OLMo 3 7B", "L": 32, "params_b": 7.0,
                 # the comparison (core/tasks) and the readout-only re-run (landmarks/landmarks.json) are two jobs (D37)
                 "landmarks": "run-35685649916/landmarks_olmo3_7b_seed20260907",
                 "source": "run-35690702732/source_olmo3_7b_seed20260907"},
}

# Numbers whose run directories no longer exist, as STATUS.md records them (section in the comment).
STATUS_ONLY: dict[str, Any] = {
    # D37 table: hand-over (learned, median read point), hand-over (head mean), universal heads' median write point
    "landmarks": {
        "qwen3_8b": {"label": "Qwen3 8B", "L": 36, "handover_learned": 24.0, "handover_fv": 24.5, "heads_median": 23.0},
        "qwen3_8b_post": {"label": "Qwen3 8B (post-trained)", "L": 36, "handover_learned": 24.0, "handover_fv": 25.0, "heads_median": 21.5},
        "gemma4_12b": {"label": "Gemma 4 12B", "L": 48, "handover_learned": 33.0, "handover_fv": 29.5, "heads_median": 32.0},
    },
    # Iteration 11 (D34), 8B: at the hand-over read point, the rank-1 own subspace patched alone (eight lexical tasks),
    # the other tasks' and the background's rank-1 patch, the random rank-1 patch; removal of the own rank-1 direction
    "rank_8b": {"own_patch_k1": (0.88, 0.98), "other_tasks_patch_k1": (-0.05, 0.34), "background_patch_k1": (0.0, 0.26),
                "random_patch_k1": (0.0, 0.06), "own_remove_k1": (0.51, 1.00), "own_remove_k1_median": 0.96},
    # D38 table: attention's share of the steered / natural aligning write (median; range), the random floor
    "source": {
        "qwen3_8b": {"label": "Qwen3 8B", "steered": (0.17, 0.11, 0.22), "natural": (0.34, 0.19, 0.37), "floor": 0.09,
                     "mlp_necessary": (1, 10)},
        "gemma4_12b": {"label": "Gemma 4 12B", "steered": (0.14, 0.10, 0.22), "natural": (0.21, 0.20, 0.26), "floor": 0.11,
                       "mlp_necessary": None},
    },
}

# Colour = checkpoint throughout: the Qwen3 sizes as one ordinal blue ramp (light = small), the other families as
# categorical slots (validated: dataviz validate_palette.js, ordinal and categorical, light surface).
COLORS = {"qwen3_0.6b": "#86b6ef", "qwen3_1.7b": "#3987e5", "qwen3_4b": "#1c5cab", "qwen3_8b": "#0d366b",
          "qwen3_8b_post": "#0d366b", "olmo3_7b": "#eb6834", "gemma4_12b": "#1baf7a"}
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def first_reaching(points: list[int], values: list[float | None], share: float = SHARE) -> int | None:
    for p, v in zip(points, values):
        if v is not None and v >= share:
            return int(p)
    return None


def first_crossing(points: list[int], values: list[float | None], share: float = SHARE) -> float | None:
    """The first read point at which ``values`` reaches ``share``, linearly interpolated from the grid point before.
    Iteration 10's grid is eighths of the *downstream* depth, so its points move with the injection layer; the
    snapped first point carries up to a grid step (2-4 read points) of quantisation that is correlated with the layer."""
    prev = None
    for p, v in zip(points, values):
        if v is None:
            prev = None
            continue
        if v >= share:
            if prev is not None and prev[1] < share:
                return prev[0] + (share - prev[1]) / (v - prev[1]) * (p - prev[0])
            return float(p)
        prev = (p, v)
    return None


def pooled_within_slope(groups: dict[Any, list[tuple[float, float]]]) -> float | None:
    """The slope of y on x with a free intercept per group (the within-group least-squares slope)."""
    num = den = 0.0
    for pts in groups.values():
        if len(pts) < 2:
            continue
        x = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        num += float(((x - x.mean()) * (y - y.mean())).sum())
        den += float(((x - x.mean()) ** 2).sum())
    return num / den if den > 0 else None


def cluster_bootstrap_slope(groups: dict[Any, list[tuple[float, float]]], rng: np.random.Generator) -> dict[str, Any]:
    keys = [k for k, v in groups.items() if len(v) >= 2]
    est = pooled_within_slope({k: groups[k] for k in keys})
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        s = pooled_within_slope({i: groups[keys[j]] for i, j in enumerate(pick)})
        if s is not None:
            boots.append(s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope": est, "ci95": [float(lo), float(hi)], "n_groups": len(keys), "n_points": sum(len(groups[k]) for k in keys)}


def spearman_exact(x: list[float], y: list[float]) -> dict[str, Any]:
    """Spearman's rho (average ranks for ties) with exact permutation p-values over all n! orderings."""
    def ranks(a: np.ndarray) -> np.ndarray:
        r = np.empty(len(a))
        r[np.argsort(a, kind="stable")] = np.arange(len(a), dtype=np.float64)
        for v in np.unique(a):
            r[a == v] = r[a == v].mean()
        return r

    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = [float(np.corrcoef(rx, ry[list(p)])[0, 1]) for p in itertools.permutations(range(len(ry)))]
    perms = np.array(perms)
    return {"rho": rho, "n": len(x), "p_one_sided": float((perms >= rho - 1e-12).mean()),
            "p_two_sided": float((np.abs(perms) >= abs(rho) - 1e-12).mean())}


def sign_test(n_pos: int, n: int) -> float:
    """Two-sided exact binomial p for n_pos of n under p = 1/2."""
    k = min(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------------------------------- extraction

def handover_curves(root: Path) -> tuple[list[dict], list[dict]]:
    """Per checkpoint, construction and task: the keep-only retention at every downstream read point of the primary
    layer (D37's patch test) and the summary read points (hand-over, heads)."""
    curves, per_task = [], []
    for ck, spec in RUNS.items():
        for key, construction in (("landmarks", "learned_d31"), ("landmarks_wd", "learned_d43")):
            if key not in spec:
                continue
            lm = load(root / spec[key] / "landmarks" / "landmarks.json")
            L = lm["n_layers"]
            for task, t in lm["tasks"].items():
                heads = (t.get("heads") or {}).get("median")
                for c_key, c_name in (("learned", construction), ("fv", "head_mean")):
                    h = (t.get("handover") or {}).get(c_key)
                    if not h:
                        continue
                    if construction == "learned_d43" and c_key == "fv":
                        continue  # the head-mean rows of the D44 runs repeat the D37 ones
                    for rp, k in zip(h["grid"], h["keep"]):
                        curves.append({"checkpoint": ck, "construction": c_name, "task": task, "lexical": task not in NON_LEXICAL,
                                       "primary_layer": t["primary_layer"], "L": L, "read_point": rp, "depth": rp / L, "keep_retained": k})
                    per_task.append({"checkpoint": ck, "construction": c_name, "task": task, "lexical": task not in NON_LEXICAL,
                                     "primary_layer": t["primary_layer"], "L": L, "handover": h["read_point"],
                                     "handover_depth": None if h["read_point"] is None else h["read_point"] / L,
                                     "heads_median": heads, "source": "run"})
    return curves, per_task


def handover_vs_layer(root: Path) -> list[dict]:
    """Iteration 10's patch grid (eighths of the downstream depth) at every compared injection layer."""
    rows = []
    for ck, spec in RUNS.items():
        if "trajectories10" not in spec:
            continue
        for f in sorted((root / spec["trajectories10"] / "core" / "tasks").glob("*/trajectories.json")):
            d = load(f)
            L = d["n_read_points"] - 1
            for layer, pl in d["per_layer"].items():
                pt = pl.get("patch")
                if not pt:
                    continue
                for c_key, c_name in (("learned", "learned_d31"), ("fv", "head_mean")):
                    cs = pt["constructions"].get(c_key)
                    if not cs:
                        continue
                    keeps = [(r.get("keep") or {}).get("retained") for r in cs["rows"]]
                    h = first_crossing(pt["read_points"], keeps)
                    rows.append({"checkpoint": ck, "construction": c_name, "task": d["task"], "lexical": d["task"] not in NON_LEXICAL,
                                 "layer": int(layer), "roles": "+".join(d["layer_roles"].get(layer, [])), "L": L,
                                 "effect_over_gap": cs["full_effect"] / d["fewshot_gap_per_token"],
                                 "grid": " ".join(map(str, pt["read_points"])), "keep": " ".join(f"{k:.3f}" for k in keeps),
                                 "handover_grid": first_reaching(pt["read_points"], keeps), "handover": h,
                                 "handover_half": first_crossing(pt["read_points"], keeps, 0.5),
                                 "handover_depth": None if h is None else h / L, "layer_depth": int(layer) / L})
    return rows


def rank_rows(root: Path) -> list[dict]:
    """Iteration 11 (D34) at the hand-over read point: keep / patch / remove within the top-k subspaces."""
    rows = []
    for ck, spec in RUNS.items():
        if "trajectories11" not in spec:
            continue
        for f in sorted((root / spec["trajectories11"] / "core" / "tasks").glob("*/trajectories.json")):
            d = load(f)
            sub = d["per_layer"][str(d["primary_layer"])].get("subspace")
            if not sub:
                continue
            c = sub["constructions"]["learned"]
            for rp in c["read_points"]:
                where = "handover" if rp["read_point"] == c["handover_read_point"] else ("last" if rp["read_point"] == d["n_read_points"] - 1 else "half_depth")
                for k, kinds in rp["rows"].items():
                    for kind, edits in kinds.items():
                        rows.append({"checkpoint": ck, "task": d["task"], "lexical": d["task"] not in NON_LEXICAL, "at": where,
                                     "read_point": rp["read_point"], "k": int(k), "subspace": kind,
                                     "keep": edits["keep"]["retained"], "patch": edits["patch"]["retained"],
                                     "remove": edits["remove"]["retained"], "per_prompt_keep": rp["per_prompt_keep"],
                                     "per_prompt_patch": rp["per_prompt_patch"], "source": "run"})
    return rows


def source_rows(root: Path) -> list[dict]:
    """D38 / D44: attention's share of the aligning write over the window to the hand-over, and the removals."""
    rows = []
    for ck, spec in RUNS.items():
        for key, construction in (("source", "learned_d31"), ("source_wd", "learned_d43")):
            if key not in spec:
                continue
            s = load(root / spec[key] / "core" / "summary.json")
            for task, t in s["tasks"].items():
                w = t["windows"]["to_handover"]
                rem = t["removal"]
                mlp, att = rem["mlp_along"], rem["att_along"]
                rows.append({"checkpoint": ck, "construction": construction, "task": task, "lexical": task not in NON_LEXICAL,
                             "att_share_steered": w["steered"]["att_share"]["median"], "att_share_natural": w["natural"]["att_share"]["median"],
                             "att_share_random": w["random"]["att_share"]["median"],
                             "mlp_along_removed_retained": mlp["retained"], "mlp_along_random_retained": mlp["random_retained"],
                             "att_along_removed_retained": att["retained"], "att_along_random_retained": att["random_retained"],
                             # D38's necessity rule: <= 0.5 retained with the matched random removal >= 0.9
                             "mlp_necessary": bool(mlp["retained"] <= 0.5 and mlp["random_retained"] >= 0.9),
                             "att_necessary": bool(att["retained"] <= 0.5 and att["random_retained"] >= 0.9), "source": "run"})
    return rows


# ---------------------------------------------------------------------------------------------------- statistics

def statistics(per_task: list[dict], vs_layer: list[dict], ranks: list[dict], source: list[dict]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {"seed": SEED, "n_boot": N_BOOT}

    # 1. where the hand-over is: per checkpoint and construction, over lexical tasks
    loc = {}
    for (ck, c), grp in itertools.groupby(sorted((r for r in per_task if r["lexical"] and r["handover"] is not None),
                                                 key=lambda r: (r["checkpoint"], r["construction"])),
                                          key=lambda r: (r["checkpoint"], r["construction"])):
        g = list(grp)
        hs = np.array([r["handover"] for r in g], float)
        L = g[0]["L"]
        loc[f"{ck}/{c}"] = {"n_tasks": len(g), "median": float(np.median(hs)), "min": int(hs.min()), "max": int(hs.max()),
                            "median_depth": float(np.median(hs) / L), "iqr_read_points": float(np.subtract(*np.percentile(hs, [75, 25])))}
    out["handover_location"] = loc

    # 2. does the hand-over move with the injection layer? within-task slope of hand-over on injection layer
    slopes: dict[str, Any] = {}
    variants = {"keep_0.9_interpolated": ("handover", 0.0), "keep_0.9_grid": ("handover_grid", 0.0),
                "keep_0.5_interpolated": ("handover_half", 0.0), "keep_0.9_interpolated_effect_ge_0.3_gap": ("handover", 0.3),
                "keep_0.5_interpolated_effect_ge_0.3_gap": ("handover_half", 0.3)}
    for vname, (field, min_effect) in variants.items():
        for c in ("learned_d31", "head_mean"):
            allg: dict[Any, list[tuple[float, float]]] = {}
            for ck in sorted({r["checkpoint"] for r in vs_layer}):
                g: dict[Any, list[tuple[float, float]]] = {}
                for r in vs_layer:
                    if (r["checkpoint"] == ck and r["construction"] == c and r["lexical"] and r[field] is not None
                            and r["effect_over_gap"] >= min_effect):
                        g.setdefault(r["task"], []).append((r["layer"], r[field]))
                if vname == "keep_0.9_interpolated":
                    slopes[f"{ck}/{c}"] = cluster_bootstrap_slope(g, rng)
                allg.update({(ck, k): v for k, v in g.items()})
            est = cluster_bootstrap_slope(allg, rng)
            # the spread of the injection layers the estimate rests on
            spans = [max(p[0] for p in v) - min(p[0] for p in v) for v in allg.values() if len(v) >= 2]
            est["injection_span_read_points"] = {"median": float(np.median(spans)), "max": int(max(spans))}
            slopes[f"pooled/{c}/{vname}"] = est
    out["handover_vs_injection_layer"] = {
        "note": "within-task slope of the hand-over read point on the injection layer; 0: a fixed read point, 1: a fixed "
                "distance after the injection. Primary: the keep-only edit's 0.9 crossing, interpolated on iteration 10's grid.",
        "estimates": slopes}

    # 3. the head mean hands over where the learned vector does (D37 runs, per task)
    diffs = []
    by = {(r["checkpoint"], r["task"]): r for r in per_task if r["construction"] == "learned_d31"}
    for r in per_task:
        if r["construction"] == "head_mean" and r["lexical"] and r["handover"] is not None:
            o = by.get((r["checkpoint"], r["task"]))
            if o and o["handover"] is not None:
                diffs.append(r["handover"] - o["handover"])
    d = np.array(diffs, float)
    out["head_mean_minus_learned_handover"] = {"n": len(d), "median": float(np.median(d)), "median_abs": float(np.median(np.abs(d))),
                                               "share_within_2_read_points": float((np.abs(d) <= 2).mean())}

    # 4. the hand-over against the universal heads' write depth across the seven checkpoints (D37)
    pts = []
    for ck, spec in RUNS.items():
        g = [r for r in per_task if r["checkpoint"] == ck and r["construction"] == "learned_d31"]
        if g:
            hs = [r["handover"] for r in g if r["handover"] is not None]
            hd = [r["heads_median"] for r in g if r["heads_median"] is not None]
            pts.append({"checkpoint": ck, "label": spec["label"], "L": spec["L"], "handover_learned": float(np.median(hs)),
                        "heads_median": float(np.median(hd)), "source": "run"})
    for ck, v in STATUS_ONLY["landmarks"].items():
        pts.append({"checkpoint": ck, "label": v["label"], "L": v["L"], "handover_learned": v["handover_learned"],
                    "heads_median": v["heads_median"], "source": "status"})
    for p in pts:
        p["handover_depth"] = p["handover_learned"] / p["L"]
        p["heads_depth"] = p["heads_median"] / p["L"]
        p["offset_read_points"] = p["heads_median"] - p["handover_learned"]
    sp = spearman_exact([p["handover_depth"] for p in pts], [p["heads_depth"] for p in pts])
    out["handover_vs_heads"] = {"points": pts, "spearman": sp,
                                "median_abs_offset_read_points": float(np.median([abs(p["offset_read_points"]) for p in pts])),
                                "depth_range_handover": [min(p["handover_depth"] for p in pts), max(p["handover_depth"] for p in pts)]}

    # 5. rank one at the hand-over (lexical tasks, the three checkpoints with their run directories)
    hand = [r for r in ranks if r["at"] == "handover" and r["lexical"]]
    rk = {}
    for kind in ("own", "other_tasks", "background", "random"):
        for k in (1, 8):
            v = np.array([r["patch"] for r in hand if r["subspace"] == kind and r["k"] == k])
            kv = np.array([r["keep"] for r in hand if r["subspace"] == kind and r["k"] == k])
            rk[f"{kind}/k{k}"] = {"n": len(v), "patch_median": float(np.median(v)), "patch_min": float(v.min()), "patch_max": float(v.max()),
                                  "keep_median": float(np.median(kv))}
    own1 = np.array([r["patch"] for r in hand if r["subspace"] == "own" and r["k"] == 1])
    ceil = np.array([r["per_prompt_patch"] for r in hand if r["subspace"] == "own" and r["k"] == 1])
    rk["own/k1_patch_over_per_prompt_ceiling"] = {"median": float(np.median(own1 / ceil)), "min": float((own1 / ceil).min())}
    rk["share_own_k1_patch_ge_0.8"] = float((own1 >= 0.8).mean())
    rem = {}
    for ck in [c for c in RUNS if "trajectories11" in RUNS[c]]:
        v = np.array([r["remove"] for r in hand if r["checkpoint"] == ck and r["subspace"] == "own" and r["k"] == 1])
        rem[ck] = {"n": len(v), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "source": "run"}
    rem["qwen3_8b"] = {"median": STATUS_ONLY["rank_8b"]["own_remove_k1_median"], "min": STATUS_ONLY["rank_8b"]["own_remove_k1"][0],
                       "max": STATUS_ONLY["rank_8b"]["own_remove_k1"][1], "source": "status"}
    # monotone trend of necessity with size over the per-task rows that exist (Kendall's tau-b against the size order)
    xs, ys = [], []
    for i, ck in enumerate(c for c in RUNS if "trajectories11" in RUNS[c]):
        for r in hand:
            if r["checkpoint"] == ck and r["subspace"] == "own" and r["k"] == 1:
                xs.append(i), ys.append(r["remove"])
    rk["remove_own_k1_by_size"] = rem
    rk["remove_own_k1_kendall_tau_vs_size"] = kendall_tau_perm(xs, ys, rng)
    out["rank_one"] = rk

    # 6. who writes the alignment: attention's share, steered against natural (paired per task)
    src = {}
    for c in ("learned_d31", "learned_d43"):
        g = [r for r in source if r["construction"] == c and r["att_share_steered"] is not None and r["att_share_natural"] is not None]
        n_lower = sum(r["att_share_steered"] < r["att_share_natural"] for r in g)
        ratio = np.array([r["att_share_steered"] / r["att_share_natural"] for r in g])
        src[c] = {"n_tasks": len(g), "steered_lower_than_natural": n_lower, "sign_test_p": sign_test(n_lower, len(g)),
                  "steered_median": float(np.median([r["att_share_steered"] for r in g])),
                  "natural_median": float(np.median([r["att_share_natural"] for r in g])),
                  "ratio_median": float(np.median(ratio)),
                  "att_along_never_necessary": not any(r["att_necessary"] for r in g),
                  "mlp_necessary_by_checkpoint": {ck: [sum(r["mlp_necessary"] for r in g if r["checkpoint"] == ck),
                                                       sum(1 for r in g if r["checkpoint"] == ck)]
                                                  for ck in RUNS if any(r["checkpoint"] == ck for r in g)}}
    src["learned_d31"]["mlp_necessary_by_checkpoint"]["qwen3_8b"] = list(STATUS_ONLY["source"]["qwen3_8b"]["mlp_necessary"]) + ["status"]
    out["source"] = src
    return out


def kendall_tau_perm(x: list[float], y: list[float], rng: np.random.Generator, n_perm: int = N_BOOT) -> dict[str, Any]:
    x, y = np.asarray(x, float), np.asarray(y, float)

    def tau(a: np.ndarray, b: np.ndarray) -> float:
        i, j = np.triu_indices(len(a), 1)
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        n0 = len(i)
        tx = (np.sign(a[i] - a[j]) == 0).sum()
        ty = (np.sign(b[i] - b[j]) == 0).sum()
        return float(s.sum() / math.sqrt((n0 - tx) * (n0 - ty)))

    t = tau(x, y)
    perm = np.array([tau(x, rng.permutation(y)) for _ in range(n_perm)])
    return {"tau_b": t, "n": len(x), "p_two_sided": float(((np.abs(perm) >= abs(t) - 1e-12).sum() + 1) / (n_perm + 1))}


# ---------------------------------------------------------------------------------------------------- figures

def style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)


def title(ax: Any, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")


def legend_block(ax: Any, x: float, y: float, dy: float, entries: list[tuple[str, str, str]]) -> None:
    """A direct-labelled legend in axes coordinates: (colour, marker kind 'line'|'dot'|'bar', label)."""
    for i, (color, kind, text) in enumerate(entries):
        yy = y - i * dy
        if kind == "line":
            ax.plot([x, x + 0.045], [yy, yy], transform=ax.transAxes, color=color, lw=1.8, solid_capstyle="round", clip_on=False)
        elif kind == "bar":
            ax.plot([x + 0.022] * 2, [yy - dy * 0.3, yy + dy * 0.3], transform=ax.transAxes, color=color, lw=2.2, clip_on=False)
        else:
            ax.scatter([x + 0.022], [yy], transform=ax.transAxes, s=18, color=color, edgecolor=SURFACE, lw=0.7, clip_on=False)
        ax.text(x + 0.06, yy, text, transform=ax.transAxes, fontsize=7.5, color=INK2, va="center")


def figures(out: Path, curves: list[dict], vs_layer: list[dict], ranks: list[dict], source: list[dict], stats: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    lab = {ck: RUNS[ck]["label"] if ck in RUNS else STATUS_ONLY["landmarks"].get(ck, {}).get("label", ck) for ck in COLORS}
    two_line = {ck: lab[ck].replace(" ", "\n", 1) if ck.startswith("qwen") else lab[ck].replace(" 3 ", " 3\n").replace(" 4 ", " 4\n")
                for ck in lab}
    est = stats["handover_vs_injection_layer"]["estimates"]

    # Figure 1: the hand-over is at a fixed depth
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    style(a)
    series = [("qwen3_0.6b", "learned_d43"), ("qwen3_1.7b", "learned_d43"), ("qwen3_4b", "learned_d43"), ("qwen3_8b", "learned_d43"),
              ("olmo3_7b", "learned_d31")]
    for ck, c in series:
        rows = [r for r in curves if r["checkpoint"] == ck and r["construction"] == c and r["lexical"]]
        L = rows[0]["L"]
        rps = sorted({r["read_point"] for r in rows})
        n_tasks = len({r["task"] for r in rows})
        xs, ys = [], []
        for rp in rps:
            v = [r["keep_retained"] for r in rows if r["read_point"] == rp and r["keep_retained"] is not None]
            if len(v) >= max(2, n_tasks // 2):  # read points most tasks' grids reach
                xs.append(rp / L), ys.append(float(np.median(v)))
        a.plot(xs, ys, color=COLORS[ck], lw=1.8, zorder=3, solid_capstyle="round")
    hv = {p["checkpoint"]: p for p in stats["handover_vs_heads"]["points"]}
    for i, (ck, _) in enumerate(series):
        if ck in hv:
            a.plot([hv[ck]["heads_depth"]], [-0.135 + 0.022 * i], marker="^", ms=6, color=COLORS[ck], mec=SURFACE, mew=1.0, zorder=4, clip_on=False)
    a.axhline(SHARE, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    a.text(0.262, SHARE + 0.03, "0.9 of the effect: the hand-over", fontsize=7.5, color=INK2)
    a.text(0.262, -0.035, "▲ each checkpoint's universal heads (median write depth)", fontsize=7.2, color=INK2)
    legend_block(a, 0.66, 0.50, 0.065, [(COLORS[ck], "line", lab[ck] + (" (full-norm vector)" if c == "learned_d31" else ""))
                                        for ck, c in series])
    a.set_xlim(0.25, 1.01)
    a.set_ylim(-0.16, 1.3)
    a.set_xlabel("read point (fraction of the stack)")
    a.set_ylabel("effect kept along the natural difference alone")
    title(a, "a  The steered state becomes the model's own at 0.6–0.7",
          "keep-only edit at every read point; learned vector at its primary layer; median over lexical tasks")
    style(b)
    rows = [r for r in vs_layer if r["construction"] == "learned_d31" and r["lexical"] and r["handover"] is not None]
    for ck in ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b"):
        g = [r for r in rows if r["checkpoint"] == ck]
        for task in sorted({r["task"] for r in g}):
            pts = sorted((r["layer_depth"], r["handover_depth"]) for r in g if r["task"] == task)
            if len(pts) >= 2:
                b.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[ck], lw=1.0, alpha=0.8, zorder=3)
            b.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, color=COLORS[ck], edgecolor=SURFACE, lw=0.8, zorder=4)
    b.plot([0.15, 0.6], [0.70, 1.15], color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=2)
    b.text(0.292, 0.86, "slope 1: a fixed distance\nafter the injection", fontsize=7, color=MUTED, ha="left", va="top")
    b.text(0.215, 0.905, "1.7B last_antonym: hands over only at the last read points", fontsize=6.8, color=MUTED, ha="left", va="center")
    s, h = est["pooled/learned_d31/keep_0.9_interpolated"], est["pooled/head_mean/keep_0.9_interpolated"]
    b.text(0.165, 0.435, f"learned vector: slope {s['slope']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]\n"
           f"head-mean vector: slope {h['slope']:.2f} [{h['ci95'][0]:.2f}, {h['ci95'][1]:.2f}]\n"
           "(within-task, 95% cluster bootstrap)", fontsize=7.3, color=INK2, va="bottom")
    legend_block(b, 0.72, 0.74, 0.06, [(COLORS[ck], "dot", lab[ck]) for ck in ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b")])
    b.set_xlim(0.15, 0.55)
    b.set_ylim(0.42, 1.0)
    b.set_xlabel("injection layer (fraction of the stack)")
    b.set_ylabel("hand-over read point (fraction of the stack)")
    title(b, "b  The hand-over does not follow the injection",
          "learned vector at its primary and neighbouring layers; 0.9 crossing")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "fig1_fixed_handover.png", dpi=200)
    plt.close(fig)

    # Figure 2: the hand-over against the universal heads' write depth
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    style(ax)
    pts = stats["handover_vs_heads"]["points"]
    ax.plot([0.45, 0.8], [0.45, 0.8], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.755, 0.772, "same depth", fontsize=7.5, color=MUTED, rotation=41, ha="center")
    offsets = {"qwen3_0.6b": (7, 4), "qwen3_1.7b": (7, -9), "qwen3_4b": (-8, 3), "qwen3_8b": (7, -11),
               "qwen3_8b_post": (7, -4), "olmo3_7b": (7, -3), "gemma4_12b": (-8, 5)}
    for p in pts:
        filled = p["source"] == "run"
        ax.scatter([p["handover_depth"]], [p["heads_depth"]], s=48, zorder=4, color=COLORS[p["checkpoint"]] if filled else SURFACE,
                   edgecolor=COLORS[p["checkpoint"]], lw=1.6)
        dx, dy = offsets[p["checkpoint"]]
        ax.annotate(p["label"], (p["handover_depth"], p["heads_depth"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="right" if dx < 0 else "left", va="center")
    sp = stats["handover_vs_heads"]["spearman"]
    ax.text(0.462, 0.79, f"Spearman ρ = {sp['rho']:.2f}, 7 checkpoints, 3 families\nexact permutation p = {sp['p_one_sided']:.3f} one-sided,"
            f" {sp['p_two_sided']:.3f} two-sided\nmedian |offset| {stats['handover_vs_heads']['median_abs_offset_read_points']:.1f} read point"
            "\nfilled: from the runs   open: from STATUS.md", fontsize=7.2, color=INK2, va="top")
    ax.set_xlim(0.45, 0.8)
    ax.set_ylim(0.45, 0.8)
    ax.set_xlabel("hand-over, learned vector (fraction of the stack)")
    ax.set_ylabel("universal heads' median write depth (fraction)")
    title(ax, "The hand-over is where the in-context heads write", "one point per checkpoint (D37)")
    fig.tight_layout()
    fig.savefig(out / "fig2_handover_vs_heads.png", dpi=200)
    plt.close(fig)

    # Figure 3: rank one, and sufficiency before necessity
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    style(a)
    kinds = [("own", "the task's own\nmean difference"), ("other_tasks", "other tasks'\ntop direction"),
             ("background", "the residual's\ntop direction"), ("random", "random\ndirection")]
    hand = [r for r in ranks if r["at"] == "handover" and r["lexical"] and r["k"] == 1]
    cks = ["qwen3_0.6b", "qwen3_1.7b", "qwen3_4b"]
    for i, (kind, _) in enumerate(kinds):
        for j, ck in enumerate(cks):
            v = [r["patch"] for r in hand if r["subspace"] == kind and r["checkpoint"] == ck]
            x = i + (j - 1.5) * 0.17 + np.linspace(-0.04, 0.04, len(v))
            a.scatter(x, v, s=15, color=COLORS[ck], edgecolor=SURFACE, lw=0.7, zorder=4)
        lo, hi = STATUS_ONLY["rank_8b"][f"{kind}_patch_k1"]
        a.plot([i + 1.5 * 0.17] * 2, [lo, hi], color=COLORS["qwen3_8b"], lw=2.2, solid_capstyle="round", zorder=4)
        allv = [r["patch"] for r in hand if r["subspace"] == kind]
        a.plot([i - 0.38, i + 0.38], [np.median(allv)] * 2, color=INK, lw=1.2, zorder=5)
    out_low = [r for r in hand if r["subspace"] == "own" and r["patch"] < 0.5]
    for r in out_low:
        a.text(-0.45, r["patch"] - 0.07, f"{lab[r['checkpoint']].split()[-1]} {r['task']}:\nno hand-over before the end",
               fontsize=6.8, color=MUTED, va="top")
    a.axhline(0, color=BASE, lw=0.8)
    a.set_xticks(range(len(kinds)))
    a.set_xticklabels([k[1] for k in kinds], fontsize=7.5, color=INK2)
    a.set_ylim(-0.25, 1.25)
    a.set_ylabel("effect reproduced by patching one direction")
    title(a, "a  One direction carries the effect at the hand-over", "patched alone into the unsteered run; lexical tasks; bar = median")
    legend_block(a, 0.66, 0.70, 0.065, [(COLORS[ck], "dot", lab[ck]) for ck in cks] + [(COLORS["qwen3_8b"], "bar", "Qwen3 8B (range, STATUS.md)")])
    style(b)
    rem = stats["rank_one"]["remove_own_k1_by_size"]
    order = ["qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b"]
    for i, ck in enumerate(order):
        if rem[ck]["source"] == "run":
            v = [r["remove"] for r in hand if r["subspace"] == "own" and r["checkpoint"] == ck]
            b.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=COLORS[ck], edgecolor=SURFACE, lw=0.7, zorder=4)
        else:
            b.plot([i, i], [rem[ck]["min"], rem[ck]["max"]], color=COLORS[ck], lw=2.2, solid_capstyle="round", zorder=4)
        b.plot([i - 0.25, i + 0.25], [rem[ck]["median"]] * 2, color=INK, lw=1.2, zorder=5)
    b.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    b.axhline(0.0, color=BASE, lw=0.8)
    b.text(-0.4, 1.05, "1: not needed, the rest of the perturbation suffices", fontsize=7.2, color=INK2)
    b.text(-0.4, -0.88, "≤ 0: needed, without it the rest is useless", fontsize=7.2, color=INK2)
    b.set_xticks(range(len(order)))
    b.set_xticklabels([two_line[c] for c in order], fontsize=7.5, color=INK2)
    b.set_xlim(-0.5, 3.5)
    b.set_ylim(-1.0, 1.15)
    b.set_ylabel("effect left after removing that direction")
    t = stats["rank_one"]["remove_own_k1_kendall_tau_vs_size"]
    title(b, "b  Necessary only in the small models",
          f"removal at the hand-over; Kendall τ = {t['tau_b']:.2f} vs size, p = {t['p_two_sided']:.3f} (0.6B–4B)")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "fig3_rank_one.png", dpi=200)
    plt.close(fig)

    # Figure 4: the MLPs convert; whether they are needed depends on the construction
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})
    style(a)
    cks = ["qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "olmo3_7b"]
    for i, ck in enumerate(cks):
        g = [r for r in source if r["checkpoint"] == ck and r["construction"] == "learned_d31" and r["att_share_steered"] is not None]
        for k, r in enumerate(g):
            dx = (k - len(g) / 2) * 0.025
            a.plot([i - 0.17 + dx, i + 0.17 + dx], [r["att_share_steered"], r["att_share_natural"]], color=GRID, lw=0.9, zorder=2)
            a.scatter([i - 0.17 + dx], [r["att_share_steered"]], s=15, color=COLORS[ck], edgecolor=SURFACE, lw=0.7, zorder=4)
            a.scatter([i + 0.17 + dx], [r["att_share_natural"]], s=15, color=SURFACE, edgecolor=COLORS[ck], lw=1.1, zorder=4)
        floor = np.median([r["att_share_random"] for r in g if r["att_share_random"] is not None])
        a.plot([i - 0.32, i + 0.32], [floor, floor], color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=3)
    for i, ck in enumerate(["qwen3_8b", "gemma4_12b"], start=len(cks)):
        v = STATUS_ONLY["source"][ck]
        for dx, (m, lo, hi), filled in ((-0.17, v["steered"], True), (0.17, v["natural"], False)):
            a.plot([i + dx] * 2, [lo, hi], color=COLORS[ck], lw=2.0, solid_capstyle="round", zorder=3)
            a.scatter([i + dx], [m], s=26, color=COLORS[ck] if filled else SURFACE, edgecolor=COLORS[ck], lw=1.3, zorder=4)
        a.plot([i - 0.32, i + 0.32], [v["floor"]] * 2, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=3)
    a.set_xticks(range(len(cks) + 2))
    a.set_xticklabels([two_line[c] for c in cks] + [two_line["qwen3_8b"] + "*", two_line["gemma4_12b"] + "*"], fontsize=7.5, color=INK2)
    a.set_ylim(0, 0.6)
    a.set_ylabel("attention's share of the aligning write")
    sd = stats["source"]["learned_d31"]
    title(a, "a  The MLPs write what turns the control into the task state",
          f"window injection → hand-over; ● steered  ○ natural; dashes: random floor; steered < natural on "
          f"{sd['steered_lower_than_natural']}/{sd['n_tasks']} tasks; * STATUS.md")
    style(b)
    order = ["qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b"]
    nec = stats["source"]
    for c, color, name in (("learned_d31", "#eda100", "full-norm\nvector (D31)"), ("learned_d43", "#6250d6", "weight-decayed\nvector (D43)")):
        ys = [nec[c]["mlp_necessary_by_checkpoint"][ck][0] / nec[c]["mlp_necessary_by_checkpoint"][ck][1] for ck in order]
        b.plot(range(len(order)), ys, color=color, lw=1.8, marker="o", ms=5.5, mec=SURFACE, mew=1.0, zorder=4)
        b.text(len(order) - 1 + 0.12, ys[-1], name, fontsize=7.5, color=INK2, va="center")
    b.set_xticks(range(len(order)))
    b.set_xticklabels([two_line[c] for c in order], fontsize=7.5, color=INK2)
    b.set_xlim(-0.3, 4.2)
    b.set_ylim(-0.05, 1.05)
    b.set_ylabel("share of tasks whose MLP aligning writes are needed")
    title(b, "b  Open: the size trend flips with the vector", "D38 necessity rule; the D31 8B point from STATUS.md")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "fig4_mlp_conversion.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/single_control"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data").mkdir(exist_ok=True)

    curves, per_task = handover_curves(args.runs_dir)
    vs_layer = handover_vs_layer(args.runs_dir)
    ranks = rank_rows(args.runs_dir)
    source = source_rows(args.runs_dir)
    stats = statistics(per_task, vs_layer, ranks, source)
    stats["runs"] = {ck: {k: v for k, v in spec.items() if "/" in str(v)} for ck, spec in RUNS.items()}
    stats["status_only"] = STATUS_ONLY

    write_csv(args.out / "data" / "handover_curves.csv", curves)
    write_csv(args.out / "data" / "handover_per_task.csv", per_task)
    write_csv(args.out / "data" / "handover_vs_injection_layer.csv", vs_layer)
    write_csv(args.out / "data" / "rank_subspace.csv", ranks)
    write_csv(args.out / "data" / "source_shares.csv", source)
    with open(args.out / "stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    if not args.no_figures:
        figures(args.out, curves, vs_layer, ranks, source, stats)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("runs", "status_only")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
