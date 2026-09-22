"""The within-family geometry of a parameterised task family (docs/DECISIONS.md D39, reading iii).

A family is one registry task under several labels whose parameter (the operand of add-k, the position of the
k-th word) is ordered. The learned run stores, per label, one learned vector per candidate layer plus the three
seed fits behind it; the head-mean run stores the function vector at the label's selected layer. This module
reads them from finished run directories and answers: are the family's vectors one direction or several, is the
spread across the parameter larger than the spread across seeds of one label, and does the cosine between two
labels' vectors fall with the distance between their parameters (the "ordered" reading)?
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def pairwise_cosines(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Cosine matrix of a list of vectors."""
    U = np.stack([_unit(v) for v in vectors])
    return U @ U.T


def seed_spread(per_seed: np.ndarray) -> dict[str, float | None]:
    """Pairwise |cosine| between the seed fits of one label at one layer (``per_seed``: seeds x dim)."""
    C = pairwise_cosines(list(per_seed))
    iu = np.triu_indices(len(per_seed), 1)
    vals = np.abs(C[iu])
    if vals.size == 0:
        return {"median": None, "min": None, "max": None}
    return {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=np.float64)
    for v in np.unique(a):
        idx = np.where(a == v)[0]
        r[idx] = r[idx].mean()
    return r


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _ranks(x), _ranks(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def order_test(cos: np.ndarray, params: Sequence[float], max_exact: int = 8, n_perm: int = 20000,
               seed: int = 0) -> dict[str, Any]:
    """Does the cosine between two labels' vectors fall with the distance between their parameters?

    Spearman's rho between the off-diagonal cosines and minus the parameter distance, over the label pairs; the
    null permutes the parameter labels over the vectors (exactly, over all label permutations, when there are at
    most ``max_exact`` labels; a Monte Carlo sample otherwise). One-sided: p is the share of permutations whose
    rho reaches the observed one. "ordered" is p <= 0.05.
    """
    cos = np.asarray(cos, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = len(params)
    iu = np.triu_indices(n, 1)
    if n < 3:
        return {"rho": None, "p": None, "n_labels": n, "ordered": False}
    c = cos[iu]

    def rho_for(perm: Sequence[int]) -> float:
        p = params[list(perm)]
        d = -np.abs(p[iu[0]] - p[iu[1]])
        return _spearman(c, d)

    rho = rho_for(range(n))
    if not np.isfinite(rho):
        return {"rho": None, "p": None, "n_labels": n, "ordered": False}
    if n <= max_exact:
        perms = list(itertools.permutations(range(n)))
        rhos = np.array([rho_for(p) for p in perms])
        exact = True
    else:
        rng = np.random.default_rng(seed)
        rhos = np.array([rho_for(rng.permutation(n)) for _ in range(n_perm)])
        exact = False
    p = float(np.mean(rhos >= rho - 1e-12))
    return {"rho": rho, "p": p, "n_labels": n, "n_perm": int(len(rhos)), "exact": exact, "ordered": bool(p <= 0.05)}


def _off_diagonal(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(C), 1)
    return C[iu]


def family_geometry(learned_run: str | Path, labels: Sequence[str], params: Sequence[float],
                    fv_run: str | Path | None = None) -> dict[str, Any]:
    """Read the family's vectors from finished runs and summarise their geometry per candidate layer.

    ``labels`` are the task keys (run directory names under ``core/tasks``) in the order of ``params``. Labels
    whose learned run holds no vectors (rejected tasks) are dropped, and reported.
    """
    learned_run = Path(learned_run)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for lab in labels:
        f = learned_run / "core" / "tasks" / lab / "directions.npz"
        if f.exists():
            a = dict(np.load(f))
            if "learned" in a:
                arrays[lab] = a
                continue
        missing.append(lab)
    kept = [lab for lab in labels if lab in arrays]
    kept_params = [p for lab, p in zip(labels, params) if lab in arrays]
    out: dict[str, Any] = {"labels": kept, "params": kept_params, "missing": missing, "per_layer": {}}
    if len(kept) < 2:
        return out
    layers = [int(l) for l in arrays[kept[0]]["layers"]]
    for li, layer in enumerate(layers):
        vecs = [arrays[lab]["learned"][li] for lab in kept]
        C = pairwise_cosines(vecs)
        spreads = {lab: seed_spread(arrays[lab]["learned_per_seed"][li]) for lab in kept
                   if "learned_per_seed" in arrays[lab]}
        seed_meds = [s["median"] for s in spreads.values() if s["median"] is not None]
        off = _off_diagonal(C)
        out["per_layer"][str(layer)] = {
            "cosines": C.tolist(),
            "cross_label": {"median": float(np.median(off)), "min": float(off.min()), "max": float(off.max())},
            "seed_spread": spreads,
            "seed_spread_median_over_labels": float(np.median(seed_meds)) if seed_meds else None,
            # a family's labels are "distinct directions" when the cross-label cosines sit below the seed spread
            # of a single label (the fit's own non-uniqueness), and "one direction" when they sit above it
            "cross_below_seed_spread": bool(seed_meds and float(np.median(off)) < float(np.median(seed_meds))),
            "order": order_test(C, kept_params),
        }
    # the head-mean vectors, each at its label's selected layer (one vector per label in the run)
    if fv_run is not None:
        fvs: dict[str, np.ndarray] = {}
        fv_layers: dict[str, int | None] = {}
        for lab in labels:
            f = Path(fv_run) / "core" / "tasks" / lab / "directions.npz"
            if f.exists():
                a = dict(np.load(f))
                if "fv_direction" in a:
                    fvs[lab] = a["fv_direction"]
                    q = Path(fv_run) / "core" / "tasks" / lab / "qualification.json"
                    fv_layers[lab] = None
                    if q.exists():
                        import json

                        sel = json.loads(q.read_text()).get("selection") or {}
                        fv_layers[lab] = sel.get("layer")
        if len(fvs) >= 2:
            fl = [lab for lab in labels if lab in fvs]
            C = pairwise_cosines([fvs[lab] for lab in fl])
            off = _off_diagonal(C)
            out["head_mean"] = {"labels": fl, "selected_layers": {lab: fv_layers[lab] for lab in fl},
                                "cosines": C.tolist(),
                                "cross_label": {"median": float(np.median(off)), "min": float(off.min()), "max": float(off.max())},
                                "order": order_test(C, [p for lab, p in zip(labels, params) if lab in fvs])}
        else:
            out["head_mean"] = {"labels": list(fvs), "note": "fewer than two labels carry a head-mean vector"}
    return out


def format_geometry(g: dict[str, Any], name: str = "") -> str:
    lines = [f"family geometry {name}: labels {g['labels']} (missing {g['missing']})"]

    def rho_str(o: dict[str, Any]) -> str:
        if o.get("rho") is None:
            return "n/a"
        return f"{o['rho']:+.2f} (p {o['p']:.3f}{'' if o.get('exact', True) else ' mc'})"

    for layer, r in g.get("per_layer", {}).items():
        cl = r["cross_label"]
        seed = r["seed_spread_median_over_labels"]
        seed_str = "n/a" if seed is None else f"{seed:.2f}"
        lines.append(f"  layer {layer}: cross-label cos median {cl['median']:.2f} [{cl['min']:.2f}, {cl['max']:.2f}]; "
                     f"seed spread median {seed_str}; distinct {r['cross_below_seed_spread']}; order rho {rho_str(r['order'])}")
    hm = g.get("head_mean")
    if hm and "cosines" in hm:
        cl = hm["cross_label"]
        lines.append(f"  head means ({hm['labels']}, layers {hm['selected_layers']}): cross-label cos median "
                     f"{cl['median']:.2f} [{cl['min']:.2f}, {cl['max']:.2f}]; order rho {rho_str(hm['order'])}")
    return "\n".join(lines)
