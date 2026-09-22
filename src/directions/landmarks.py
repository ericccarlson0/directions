"""The landmark test (docs/DECISIONS.md D37): does the hand-over read point coincide with a depth defined
without the control?

Four landmarks per model, each a depth profile with a knee, and the hand-over, all as read points ``m`` of a
stack of ``L`` blocks (read point ``m`` is the input of block ``m``; ``m/L`` is the fraction of the stack):

* the hand-over: the first read point at which keeping only the steered perturbation's component along the
  prompt's natural difference retains a share of the effect (the patch test at every downstream read point);
* the pool's rank: the first read point from which the top uncentred component of the pool's natural
  differences explains a fraction of their energy at every later read point;
* the verbal onset: the first read point at which a lens readout of the mean natural difference ranks a task
  word within the top ``rank_threshold``; the verbal exit is the last such read point;
* the universal heads: the read points the selected heads write to (block ``l`` writes to read point ``l+1``);
* the workspace band (the checkpoint with a published Jacobian lens only): the per-read-point rate at which
  the lens reads the paper's evaluation intermediates, and its onset.

Everything here is a pure function of stored results (numbers in, numbers out) so that the criteria can be
tested and applied identically to every model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Knees
# --------------------------------------------------------------------------- #


def first_reaching(values: Sequence[float | None], threshold: float) -> int | None:
    """The first index whose value is ``>= threshold`` (None values never qualify)."""
    for i, v in enumerate(values):
        if v is not None and v >= threshold:
            return i
    return None


def first_sustained(values: Sequence[float | None], threshold: float) -> int | None:
    """The first index from which every later value is ``>= threshold`` (None values break the run)."""
    onset: int | None = None
    for i, v in enumerate(values):
        if v is not None and v >= threshold:
            if onset is None:
                onset = i
        else:
            onset = None
    return onset


def last_reaching(values: Sequence[float | None], threshold: float) -> int | None:
    """The last index whose value is ``>= threshold``."""
    out = None
    for i, v in enumerate(values):
        if v is not None and v >= threshold:
            out = i
    return out


def rank_one_onset(explained_top1: Sequence[float | None], threshold: float = 0.7, fraction_of_max: float = 0.9) -> dict[str, Any]:
    """The rank landmark of a pool: ``sustained`` is the first read point from which the top component explains
    ``>= threshold`` of the energy at every later read point (the primary criterion); ``fraction_of_max`` the
    first read point reaching that fraction of the profile's maximum (secondary)."""
    vals = [None if v is None else float(v) for v in explained_top1]
    finite = [v for v in vals if v is not None]
    peak = max(finite) if finite else None
    return {"sustained": first_sustained(vals, threshold), "threshold": threshold,
            "fraction_of_max": None if peak is None else first_reaching(vals, fraction_of_max * peak),
            "max": peak, "argmax": None if peak is None else int(np.argmax([-np.inf if v is None else v for v in vals]))}


def verbal_window(best_ranks: Sequence[int | None], rank_threshold: int = 10) -> dict[str, Any]:
    """The verbal onset and exit of a lens profile: ``best_ranks[m]`` is the best task-word rank at read point
    ``m`` (1 is the top token). Onset: the first read point with rank ``<= rank_threshold``; exit: the last; peak:
    the read point of the best rank. None where the task is never named."""
    scores = [None if r is None else -float(r) for r in best_ranks]  # higher is better
    onset = first_reaching(scores, -float(rank_threshold))
    exit_ = last_reaching(scores, -float(rank_threshold))
    finite = [(s, i) for i, s in enumerate(scores) if s is not None]
    peak = max(finite)[1] if finite else None
    return {"onset": onset, "exit": exit_, "peak": peak, "best_rank": None if peak is None else int(-scores[peak]),
            "rank_threshold": rank_threshold}


def head_write_points(fv_heads: np.ndarray) -> dict[str, Any]:
    """The read points the selected heads write to: ``fv_heads`` is ``(n, 2)`` of (layer, head)."""
    heads = np.asarray(fv_heads)
    if heads.size == 0:
        return {"read_points": [], "median": None, "min": None, "max": None, "n": 0}
    pts = sorted(int(l) + 1 for l in heads[:, 0])
    return {"read_points": pts, "median": float(np.median(pts)), "min": pts[0], "max": pts[-1], "n": len(pts)}


def handover_read_point(rows: Sequence[dict[str, Any]], share: float = 0.9, key: str = "keep") -> int | None:
    """The first patch row whose ``key`` edit retains ``>= share`` of the effect (rows in read-point order)."""
    for r in sorted(rows, key=lambda r: r["read_point"]):
        v = (r.get(key) or {}).get("retained")
        if v is not None and v >= share:
            return int(r["read_point"])
    return None


def band_from_rates(rates: Sequence[float | None], fraction_of_max: float = 0.5) -> dict[str, Any]:
    """A band from a hit-rate profile over read points: onset is the first read point reaching
    ``fraction_of_max`` of the maximum, exit the last, peak the argmax."""
    vals = [None if v is None else float(v) for v in rates]
    finite = [v for v in vals if v is not None]
    if not finite or max(finite) <= 0:
        return {"onset": None, "exit": None, "peak": None, "max": max(finite) if finite else None}
    peak = max(finite)
    return {"onset": first_reaching(vals, fraction_of_max * peak), "exit": last_reaching(vals, fraction_of_max * peak),
            "peak": int(np.argmax([-np.inf if v is None else v for v in vals])), "max": peak, "fraction_of_max": fraction_of_max}


# --------------------------------------------------------------------------- #
# The lens-quality evaluation sets of the workspace paper (data/jlens_evaluations)
# --------------------------------------------------------------------------- #

NUMBER_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
                "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
                "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
                "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
                "100": "hundred"}
OPERATION_FORMS = {"addition": ["+", "plus", "add", "sum"], "subtraction": ["-", "minus", "subtract", "difference"],
                   "multiplication": ["*", "×", "times", "multiply", "product"], "division": ["/", "÷", "divided", "divide", "quotient"]}


def intermediate_forms(word: str, eval_set: str) -> list[str]:
    """The surface forms whose first token counts for an intermediate: the word with and without a leading
    space; for order-ops also the digit/number-word and symbol/word forms (the paper's synonym sets)."""
    forms = [word]
    if eval_set == "order-ops":
        w = word.strip().lower()
        if w in NUMBER_WORDS:
            forms.append(NUMBER_WORDS[w])
        for d, name in NUMBER_WORDS.items():
            if w == name:
                forms.append(d)
        forms.extend(OPERATION_FORMS.get(w, []))
    out: list[str] = []
    for f in forms:
        for s in (" " + f.strip(), f.strip()):
            if s not in out:
                out.append(s)
    return out


def readout_prompt(prompt: str, eval_set: str, has_target: bool = False) -> str:
    """The prompt whose last token is the paper's readout position. Poetry is read at the last newline (the end
    of line 1): the prompt is truncated after that newline. A prompt with a ``target`` is read at the token
    immediately preceding the target; under the paper's joint tokenisation of prompt and target a trailing
    space of the prompt merges into the target's first token, so that token is the last word of the prompt
    without the trailing space, which is what is returned. Otherwise the prompt itself (its final token)."""
    if eval_set == "poetry":
        i = prompt.rfind("\n")
        if i >= 0:
            return prompt[: i + 1]
    if has_target:
        return prompt.rstrip(" ")
    return prompt


def hit_rates(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, list[float]]:
    """Per read point, the fraction of intermediates whose rank is ``<= k``: ``ranks`` is ``(n, n_read_points)``
    (the minimum over the intermediate's forms)."""
    r = np.asarray(ranks, dtype=np.float64)
    return {f"hit@{k}": [float(np.mean(r[:, m] <= k)) for m in range(r.shape[1])] for k in ks}


# --------------------------------------------------------------------------- #
# Collecting a model's landmarks from stored results
# --------------------------------------------------------------------------- #


def _json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def collect_task_landmarks(task_json: dict[str, Any], readout: dict[str, Any] | None, fv_heads: np.ndarray | None,
                           share: float = 0.9, rank_threshold: int = 10, rank_one_threshold: float = 0.7,
                           lens_key: str = "logit_lens", min_task_mass: float = 1e-3) -> dict[str, Any]:
    """One task's landmarks from its ``trajectories.json`` (the landmark comparison run), its verbalise readout
    (``<task>.json`` of ``scripts/verbalise.py``) and the head-mean run's selected heads. A readout's rank counts
    as a verbal reading only if the task-word mass is at least ``min_task_mass``: a flat readout (every token
    at the same probability, as Gemma 4's final norm gives on some difference vectors) ranks a word first by
    tie without any mass on it."""
    layer = int(task_json["primary_layer"])
    L = int(task_json["n_read_points"]) - 1
    info = task_json["per_layer"][str(layer)]
    out: dict[str, Any] = {"primary_layer": layer, "n_layers": L, "handover": {}, "pool_rank": None, "verbal": {}, "heads": None}
    for c, rows in (((info.get("patch") or {}).get("constructions") or {}).items()):
        out["handover"][c] = {"read_point": handover_read_point(rows.get("rows", []), share),
                              "grid": [int(r["read_point"]) for r in rows.get("rows", [])],
                              "keep": [(r.get("keep") or {}).get("retained") for r in rows.get("rows", [])]}
    spec = ((info.get("subspace") or {}).get("pool_spectrum"))
    if spec:
        top1 = [s["explained"][0] if s.get("explained") else None for s in spec]
        pr = [s.get("participation_ratio") for s in spec]
        cos = [s.get("top_component_cos_with_mean_natural") for s in spec]
        out["pool_rank"] = {"explained_top1": top1, "participation_ratio": pr, "top_cos_with_mean": cos,
                            "knee": rank_one_onset(top1, rank_one_threshold),
                            "pr_le_1_5_sustained": first_sustained([None if p is None else -p for p in pr], -1.5)}
        if any("explained_centered" in s for s in spec):
            # exploratory: the same knees on the spectrum about the pool's mean (the prompt-to-prompt variation)
            top1c = [s["explained_centered"][0] if s.get("explained_centered") else None for s in spec]
            out["pool_rank"]["centered"] = {"explained_top1": top1c, "participation_ratio": [s.get("participation_ratio_centered") for s in spec],
                                            "mean_share": [s.get("mean_share") for s in spec], "knee": rank_one_onset(top1c, rank_one_threshold)}
    if readout:
        for lk in sorted({lens_key, "logit_lens", "jlens"}):
            ranks: list[int | None] = [None] * (L + 1)
            masses: list[float | None] = [None] * (L + 1)
            for row in readout.get("rows", []):
                r = (row.get("natural") or {}).get(lk)
                if r is not None:
                    masses[int(row["read_point"])] = r.get("task_mass")
                    valid = r.get("task_mass") is not None and r["task_mass"] >= min_task_mass
                    ranks[int(row["read_point"])] = r.get("task_best_rank") if valid else None
            if any(r is not None for r in ranks):
                out["verbal"][lk] = {"best_rank": ranks, "task_mass": masses, "window": verbal_window(ranks, rank_threshold)}
    if fv_heads is not None:
        out["heads"] = head_write_points(fv_heads)
    return out


def collect_model_landmarks(run_dir: Path, readout_dir: Path | None, fv_run: Path | None, lens_band: dict[str, Any] | None = None,
                            **kw: Any) -> dict[str, Any]:
    """Every task's landmarks of one landmark comparison run, plus the model's medians."""
    run_dir = Path(run_dir)
    meta = _json(run_dir / "metadata.json")
    tasks: dict[str, Any] = {}
    for tdir in sorted(p for p in (run_dir / "core" / "tasks").iterdir() if (p / "trajectories.json").exists()):
        tj = _json(tdir / "trajectories.json")
        ro = _json(readout_dir / f"{tdir.name}.json") if readout_dir and (readout_dir / f"{tdir.name}.json").exists() else None
        heads = None
        if fv_run is not None and (Path(fv_run) / "core" / "tasks" / tdir.name / "directions.npz").exists():
            arr = np.load(Path(fv_run) / "core" / "tasks" / tdir.name / "directions.npz")
            if "fv_heads" in arr.files:
                heads = np.asarray(arr["fv_heads"])
        tasks[tdir.name] = collect_task_landmarks(tj, ro, heads, **kw)
    L = next((t["n_layers"] for t in tasks.values()), None)
    return {"run": str(run_dir), "model": (meta.get("model") or {}).get("name"), "n_layers": L, "tasks": tasks,
            "lens_band": lens_band, "summary": model_summary(tasks, L)}


def _median(xs: list[float | int | None]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return float(np.median(v)) if v else None


def model_summary(tasks: dict[str, Any], L: int | None) -> dict[str, Any]:
    """Per model: the median read point of each landmark over tasks, with the task counts, and the per-task
    offsets of each landmark from the learned vector's hand-over."""
    def per_task(get: Any) -> dict[str, int | float | None]:
        return {name: get(t) for name, t in tasks.items()}

    hand_learned = per_task(lambda t: (t["handover"].get("learned") or {}).get("read_point"))
    hand_fv = per_task(lambda t: (t["handover"].get("fv") or {}).get("read_point"))
    rank_sus = per_task(lambda t: (t.get("pool_rank") or {}).get("knee", {}).get("sustained"))
    rank_fom = per_task(lambda t: (t.get("pool_rank") or {}).get("knee", {}).get("fraction_of_max"))
    rank_c = per_task(lambda t: ((t.get("pool_rank") or {}).get("centered") or {}).get("knee", {}).get("sustained"))
    rank_c_fom = per_task(lambda t: ((t.get("pool_rank") or {}).get("centered") or {}).get("knee", {}).get("fraction_of_max"))
    verbal_on = per_task(lambda t: ((t.get("verbal") or {}).get("logit_lens") or {}).get("window", {}).get("onset"))
    verbal_ex = per_task(lambda t: ((t.get("verbal") or {}).get("logit_lens") or {}).get("window", {}).get("exit"))
    verbal_on_j = per_task(lambda t: ((t.get("verbal") or {}).get("jlens") or {}).get("window", {}).get("onset"))
    verbal_ex_j = per_task(lambda t: ((t.get("verbal") or {}).get("jlens") or {}).get("window", {}).get("exit"))
    heads_med = per_task(lambda t: (t.get("heads") or {}).get("median"))
    landmarks = {"handover_learned": hand_learned, "handover_fv": hand_fv, "pool_rank_sustained": rank_sus,
                 "pool_rank_fraction_of_max": rank_fom, "pool_rank_centered_sustained": rank_c, "pool_rank_centered_fraction_of_max": rank_c_fom,
                 "verbal_onset": verbal_on, "verbal_exit": verbal_ex,
                 "verbal_onset_jlens": verbal_on_j, "verbal_exit_jlens": verbal_ex_j, "heads_median": heads_med}
    out: dict[str, Any] = {"n_layers": L, "per_task": landmarks, "median": {k: _median(list(v.values())) for k, v in landmarks.items()},
                           "n_tasks": {k: sum(1 for x in v.values() if x is not None) for k, v in landmarks.items()}}
    out["offset_from_handover"] = {}
    for k, v in landmarks.items():
        if k == "handover_learned":
            continue
        offs = [v[n] - hand_learned[n] for n in v if v[n] is not None and hand_learned.get(n) is not None]
        out["offset_from_handover"][k] = {"per_task": offs, "median": _median(offs), "median_abs": _median([abs(o) for o in offs])}
    return out


# --------------------------------------------------------------------------- #
# Across models
# --------------------------------------------------------------------------- #


def spearman_permutation(x: Sequence[float], y: Sequence[float], n_perm: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Spearman's rank correlation with a two-sided permutation p-value (exact enough for the handful of models)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p": None, "n": n}

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="stable")
        r = np.empty(n)
        r[order] = np.arange(n, dtype=np.float64)
        # average ranks for ties
        for v in np.unique(a):
            idx = np.where(a == v)[0]
            r[idx] = r[idx].mean()
        return r

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rp = float(np.corrcoef(rx, rng.permutation(ry))[0, 1])
        if abs(rp) >= abs(rho) - 1e-12:
            count += 1
    return {"rho": rho, "p": (count + 1) / (n_perm + 1), "n": n}


def across_models(models: dict[str, dict[str, Any]], landmarks: Sequence[str] = ("pool_rank_sustained", "pool_rank_fraction_of_max",
                                                                                   "verbal_onset", "verbal_exit", "heads_median", "handover_fv"),
                  n_perm: int = 20000) -> dict[str, Any]:
    """For each landmark: the models' (hand-over, landmark) pairs as fractions of the stack, the rank correlation
    across models, and the median absolute offset in read points."""
    out: dict[str, Any] = {}
    for lm in landmarks:
        pts = []
        for name, m in models.items():
            s = m["summary"]
            h, v, L = s["median"].get("handover_learned"), s["median"].get(lm), s["n_layers"]
            if h is None or v is None or not L:
                continue
            pts.append({"model": name, "handover": h, "landmark": v, "n_layers": L, "handover_frac": h / L, "landmark_frac": v / L,
                        "offset_read_points": v - h,
                        "median_abs_offset_per_task": (s["offset_from_handover"].get(lm) or {}).get("median_abs")})
        corr = spearman_permutation([p["handover_frac"] for p in pts], [p["landmark_frac"] for p in pts], n_perm=n_perm) if len(pts) >= 3 else {"rho": None, "p": None, "n": len(pts)}
        out[lm] = {"points": pts, "spearman": corr, "median_abs_offset_read_points": _median([abs(p["offset_read_points"]) for p in pts]),
                   "coincides": bool(corr.get("p") is not None and corr["p"] <= 0.05 and (_median([abs(p["offset_read_points"]) for p in pts]) or 99) <= 2)}
    return out
