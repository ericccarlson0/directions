"""scripts/varmap_summary.py: the gather, the digit read point and the fragile-row re-read on a synthetic map."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from directions.varmap import all_exprs, draw_source, prediction

_SPEC = importlib.util.spec_from_file_location("varmap_summary",
                                               Path(__file__).resolve().parents[1] / "scripts" / "varmap_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)

LABELS = ["(", "a", "+", "b", ")", " -", "c", " =", "␣"]
TID = {d: 15 + d for d in range(10)}


def _synthetic_map(tmp_path: Path) -> Path:
    """One-lane map, read points 0..3: the digits carry the operands up to read point 1, the answer token holds
    everything from read point 2; row 0 of S1 is fragile (its no-op swap misses the base answer at read point 3)."""
    rng = np.random.default_rng(0)
    pool = all_exprs("one_lane")
    bases = [pool[i] for i in (5, 40, 90)]
    types = ["S1", "S2", "S1_same", "both"]
    M, P, ans = 4, len(LABELS), len(LABELS) - 1
    sources = [{k: draw_source(k, b, pool, rng) for k in types} for b in bases]
    arrays, rates = {}, {}
    for k in types:
        n = len(bases)
        am = np.zeros((n, M, P), dtype=np.int32)
        pred = np.array([prediction(k, bases[i], sources[i][k]) for i in range(n)])
        for i, b in enumerate(bases):
            am[i] = TID[b.Y]
            am[i, 2:, ans] = TID[pred[i]]
            if k.endswith("_same"):
                am[i, :2, [1, 3]] = TID[(b.Y + 1) % 10]  # the digits matter up to read point 1
        if k == "S1":
            am[0, 3, 0] = TID[(bases[0].Y + 2) % 10]
        flip = am == np.array([TID[p] for p in pred])[:, None, None]
        arrays[f"{k}__flip"], arrays[f"{k}__argmax"], arrays[f"{k}__bases"] = flip, am, np.arange(n)
    result = {"task": "one_lane", "labels": LABELS, "read_points": list(range(M)), "accuracy": 1.0,
              "n_kept": {k: len(bases) for k in types}, "thresholds": {"present_flip": 0.5, "value_change": 0.25},
              "examples": [{"base": b.text(), "Y": b.Y, "sources": {k: sources[i][k].text() for k in types}}
                           for i, b in enumerate(bases)],
              "rates": {k: {} for k in types}}
    ids = {d: TID[d] for d in range(10)}
    for k, v in summary.rates_from_arrays(result, arrays, ids, {}).items():
        result["rates"][k] = {kk: {s: np.asarray(x).tolist() for s, x in v[kk].items()} for kk in ("flip", "change")}
    hyp = np.zeros((M, P))
    hyp[2:, ans] = 1.0
    result["rates"]["both"]["hypotheses"] = {"both": {"mean": hyp.tolist()}}
    (tmp_path / "varmap.json").write_text(json.dumps(result))
    np.savez_compressed(tmp_path / "varmap_arrays.npz", **arrays)
    return tmp_path


def test_parse_expr_inverts_text():
    maxmin = all_exprs("maxmin")
    for e in all_exprs("two_lane")[:50] + all_exprs("one_lane")[:50] + all_exprs("three_lane")[:50] + maxmin[:50] + maxmin[-50:]:
        assert summary.parse_expr(e.text()) == e


def test_summary_reads_the_gather_and_drops_the_fragile_row(tmp_path):
    s = summary.summarise_task(_synthetic_map(tmp_path))
    for key in ("as_run", "fragile_dropped"):
        d = s[key]
        assert d["gather_read_point"] == 2 and d["gather_holds_to_the_end"]
        assert d["last_digit_read_point"] == 1
        assert d["present_elsewhere"] == {"S1": False, "S2": False}
    assert s["as_run"]["both_swap_source_answer_at_gather"] == 1.0
    assert s["fragile_rows"] == {"S1": 1, "S2": 0, "S1_same": 0, "both": 0}
