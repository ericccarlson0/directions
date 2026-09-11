"""The in-run determinism check (docs/DECISIONS.md D23)."""

import numpy as np

from directions.determinism import compare_arrays, determinism_report


def test_compare_arrays_bit_identity_and_nan():
    a = {"x": np.array([1.0, np.nan, 3.0]), "flag": np.array([True, False]), "none": None}
    b = {"x": np.array([1.0, np.nan, 3.0]), "flag": np.array([True, False]), "none": None}
    r = compare_arrays(a, b)
    assert r["identical"] and r["arrays"]["x"] == {"identical": True, "max_abs_diff": 0.0}
    assert r["arrays"]["none"] == {"identical": True, "max_abs_diff": None}
    b["x"] = np.array([1.0, np.nan, 3.0 + 1e-12])
    r = compare_arrays(a, b)
    assert not r["identical"] and not r["arrays"]["x"]["identical"] and r["arrays"]["x"]["max_abs_diff"] > 0
    assert r["arrays"]["flag"]["identical"]
    r = compare_arrays({"x": np.zeros(3)}, {"x": np.zeros(4)})
    assert not r["identical"] and r["arrays"]["x"]["shapes"] == [[3], [4]]
    r = compare_arrays({"x": np.zeros(3)}, {"x": None})
    assert not r["identical"]
    assert not compare_arrays({"flag": np.array([True])}, {"flag": np.array([False])})["identical"]


def test_determinism_report_verdict():
    ok = {"identical": True, "arrays": {}}
    bad = {"identical": False, "arrays": {"y": {"identical": False, "max_abs_diff": 1.0}}}
    assert determinism_report(ok, None, None, "t")["identical"]
    assert determinism_report(ok, ok, ok, "t")["identical"]
    rep = determinism_report(ok, bad, ok, "t")
    assert not rep["identical"] and rep["task"] == "t" and rep["gradients"] is bad
