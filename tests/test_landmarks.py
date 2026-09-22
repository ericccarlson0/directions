"""directions.landmarks: the knee criteria of the landmark test (D37) and the eval-set helpers."""

import numpy as np

from directions.landmarks import (across_models, band_from_rates, collect_task_landmarks, first_reaching, first_sustained,
                                  handover_read_point, head_write_points, hit_rates, intermediate_forms, last_reaching,
                                  model_summary, rank_one_onset, readout_prompt, spearman_permutation, verbal_window)


def test_first_sustained_needs_every_later_value_above_threshold():
    assert first_sustained([0.1, 0.8, 0.6, 0.9, 0.95], 0.7) == 3
    assert first_sustained([0.8, 0.9, 0.95], 0.7) == 0
    assert first_sustained([0.8, 0.9, None], 0.7) is None
    assert first_sustained([0.1, 0.2], 0.7) is None
    assert first_reaching([0.1, 0.8, 0.6], 0.7) == 1 and last_reaching([0.1, 0.8, 0.6, 0.75], 0.7) == 3


def test_rank_one_onset_reports_both_criteria():
    k = rank_one_onset([0.3, 0.5, 0.72, 0.68, 0.8, 0.9], threshold=0.7, fraction_of_max=0.9)
    assert k["sustained"] == 4 and k["fraction_of_max"] == 5 and k["argmax"] == 5 and k["max"] == 0.9
    assert rank_one_onset([None, None])["sustained"] is None


def test_verbal_window_from_best_ranks():
    w = verbal_window([None, 500, 40, 9, 1, 3, 30, 12000], rank_threshold=10)
    assert (w["onset"], w["exit"], w["peak"], w["best_rank"]) == (3, 5, 4, 1)
    w = verbal_window([None, 500, 40], rank_threshold=10)
    assert w["onset"] is None and w["exit"] is None and w["peak"] == 2


def test_head_write_points_and_handover():
    h = head_write_points(np.array([[17, 3], [17, 9], [19, 0], [24, 1]]))
    assert h["read_points"] == [18, 18, 20, 25] and h["median"] == 19.0 and h["min"] == 18 and h["max"] == 25
    assert head_write_points(np.zeros((0, 2)))["median"] is None
    rows = [{"read_point": 22, "keep": {"retained": 0.85}}, {"read_point": 19, "keep": {"retained": 0.3}},
            {"read_point": 25, "keep": {"retained": 0.95}}, {"read_point": 27, "keep": {"retained": 0.99}}]
    assert handover_read_point(rows, 0.9) == 25
    assert handover_read_point(rows, 0.999) is None


def test_band_from_rates():
    b = band_from_rates([0.0, 0.1, 0.3, 0.6, 0.8, 0.5, 0.2], fraction_of_max=0.5)
    assert (b["onset"], b["peak"], b["exit"]) == (3, 4, 5)
    assert band_from_rates([0.0, 0.0])["onset"] is None


def test_eval_set_helpers():
    assert readout_prompt("A rhyming couplet:\nline one,\nAnd the face of ", "poetry") == "A rhyming couplet:\nline one,\n"
    assert readout_prompt("Fact: the ocean is the ", "multihop") == "Fact: the ocean is the "
    assert intermediate_forms("Brazil", "multihop") == [" Brazil", "Brazil"]
    forms = intermediate_forms("5", "order-ops")
    assert " 5" in forms and " five" in forms and "five" in forms
    assert " *" in intermediate_forms("multiplication", "order-ops") and " times" in intermediate_forms("multiplication", "order-ops")
    assert " 20" in intermediate_forms("twenty", "order-ops")
    r = hit_rates(np.array([[1, 3, 50], [2, 12, 1]]), ks=(1, 10))
    assert r["hit@1"] == [0.5, 0.0, 0.5] and r["hit@10"] == [1.0, 0.5, 0.5]


def _task_json(L: int, layer: int, keep: list[float], explained: list[float]) -> dict:
    grid = list(range(layer + 1, L + 1))
    rows = [{"read_point": m, "keep": {"retained": k}} for m, k in zip(grid, keep)]
    spec = [{"explained": [e, 0.05], "participation_ratio": 1 / e, "top_component_cos_with_mean_natural": 0.99} for e in explained]
    return {"primary_layer": layer, "n_read_points": L + 1,
            "per_layer": {str(layer): {"patch": {"constructions": {"learned": {"rows": rows}, "fv": {"rows": rows[1:]}}},
                                       "subspace": {"pool_spectrum": spec}}}}


def test_collect_task_landmarks_and_model_summary():
    L, layer = 8, 2
    keep = [0.1, 0.4, 0.7, 0.92, 0.95, 1.0]           # read points 3..8: hand-over at 6
    explained = [0.2, 0.3, 0.5, 0.6, 0.72, 0.75, 0.8, 0.85, 0.9]  # read points 0..8: sustained ≥ 0.7 from 4
    tj = _task_json(L, layer, keep, explained)
    readout = {"rows": [{"read_point": m, "natural": {"logit_lens": {"task_best_rank": r, "task_mass": 0.01}}}
                        for m, r in zip(range(1, L + 1), [900, 300, 50, 8, 2, 30, 400, 5000])]}
    t = collect_task_landmarks(tj, readout, np.array([[3, 0], [4, 1], [4, 2]]))
    assert t["handover"]["learned"]["read_point"] == 6 and t["handover"]["fv"]["read_point"] == 6
    assert t["pool_rank"]["knee"]["sustained"] == 4
    assert t["verbal"]["logit_lens"]["window"] == {"onset": 4, "exit": 5, "peak": 5, "best_rank": 2, "rank_threshold": 10}
    assert t["heads"]["median"] == 5.0
    s = model_summary({"a": t, "b": t}, L)
    assert s["median"]["handover_learned"] == 6 and s["median"]["pool_rank_sustained"] == 4 and s["median"]["verbal_onset"] == 4
    assert s["offset_from_handover"]["pool_rank_sustained"]["median"] == -2 and s["offset_from_handover"]["heads_median"]["median_abs"] == 1
    cross = across_models({"m1": {"summary": s}, "m2": {"summary": model_summary({"a": t}, L)}, "m3": {"summary": s}}, landmarks=("pool_rank_sustained",), n_perm=200)
    r = cross["pool_rank_sustained"]
    assert len(r["points"]) == 3 and r["points"][0]["handover_frac"] == 6 / L and r["median_abs_offset_read_points"] == 2
    assert r["spearman"]["rho"] is None  # identical hand-overs: no rank correlation to compute
    assert r["coincides"] is False


def test_spearman_permutation():
    r = spearman_permutation([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], n_perm=2000, seed=1)
    assert abs(r["rho"] - 0.8286) < 0.01 and 0.0 < r["p"] < 0.2
    r = spearman_permutation([1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1], n_perm=2000, seed=1)
    assert r["rho"] == -1.0 and r["p"] < 0.01
    assert spearman_permutation([1, 2], [1, 2])["rho"] is None
