"""The task screen (docs/DECISIONS.md D62): the forms' semantics and text, the balanced draws, the gaps and the
criteria, and a toy run."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from directions.config import load_screen_config
from directions.screen import (
    FORMATS,
    CondExpr,
    cells,
    demos_for,
    draw_form,
    gap,
    measures,
    passes,
    render,
    run_screen,
)


def test_forms_text_and_semantics():
    e = CondExpr("if_gt", 4, 2, 1, 3, g=7, h=3)
    assert e.text() == "IF(7>3, 4+2, 1+3)" and (e.C, e.Y, e.other, e.M) == (1, 6, 4, 1)
    e = CondExpr("if_mixed", 4, 2, 1, 3, g=7, h=3, cmp="<")
    assert e.text() == "IF(7<3, 4+2, 1+3)" and (e.C, e.Y, e.other, e.M) == (0, 4, 6, 0)
    e = CondExpr("if_given", 4, 2, 1, 3, k=0)
    assert e.text() == "IF(0, 4+2, 1+3)" and (e.C, e.Y) == (0, 4)
    e = CondExpr("ternary", 4, 2, 1, 3, g=2, h=3)
    assert e.text() == "2>3 ? 4+2 : 1+3" and (e.C, e.Y) == (0, 4)
    e = CondExpr("python", 4, 2, 1, 3, g=8, h=3)
    assert e.text() == "(4+2 if 8>3 else 1+3)" and (e.C, e.Y) == (1, 6)
    e = CondExpr("if_sum", 4, 2, 1, 3, g=8, h=3)
    assert e.text() == "IF(8+3>9, 4+2, 1+3)" and (e.C, e.Y) == (1, 6)
    assert CondExpr("if_sum", 4, 2, 1, 3, g=5, h=4).C == 0  # 9 is not above 9
    e = CondExpr("max", 4, 2, 1, 3)
    assert e.text() == "max(4+2, 1+3)" and (e.C, e.Y, e.M) == (1, 6, 1)
    slots = CondExpr("python", 4, 2, 1, 3, g=8, h=3).slots()
    assert [slots[i] for i in sorted(slots)] == ["a", "b", "g", "cmp", "h", "c", "d"]


@pytest.mark.parametrize("form", FORMATS)
def test_draws_balanced_and_disjoint(form):
    pool, queries = draw_form(20260916, form, 64, 32)
    assert len(queries) == 64 and len(pool) == 32
    assert len({e.text() for e in queries}) == 64 and not ({e.text() for e in pool} & {e.text() for e in queries})
    for xs, n in ((queries, 64), (pool, 32)):
        assert Counter(e.cell for e in xs) == {c: n // len(cells(form)) for c in cells(form)}
    for e in queries:
        assert e.S1 != e.S2 and 0 <= e.Y <= 9 and e.S1 <= 9 and e.S2 <= 9
        if form in ("if_gt", "if_mixed", "ternary", "python"):
            assert e.g != e.h
    again = draw_form(20260916, form, 64, 32)
    assert [e.text() for e in again[1]] == [e.text() for e in queries]
    demos = demos_for(pool, 8, np.random.default_rng(0))
    assert Counter(e.cell for e in demos) == {c: 8 // len(cells(form)) for c in cells(form)}
    p = render(demos, queries[0])
    assert p.prompt.endswith(f"\n{queries[0].text()} = ") and p.target == str(queries[0].Y)


def test_gap_and_criteria():
    rng = np.random.default_rng(1)
    g = gap(np.ones(200), np.r_[np.ones(100), np.zeros(100)], rng, 500, 0.05)
    assert g["mean"] == pytest.approx(0.5) and g["lo"] < 0.5 < g["hi"]
    ok = {"accuracy": {"lo": 0.75}, "gaps": {"branch": {"lo": -0.1, "hi": 0.1}, "shortcut": {"lo": -0.05, "hi": 0.14}}}
    assert passes(ok, True, 0.70, 0.15)["pass"]
    assert not passes(ok, False, 0.70, 0.15)["pass"]
    assert not passes({**ok, "accuracy": {"lo": 0.69}}, True, 0.70, 0.15)["pass"]
    bad = {**ok, "gaps": {"shortcut": {"lo": 0.05, "hi": 0.2}}}
    assert passes(bad, True, 0.70, 0.15) == {"accuracy": True, "gaps": False, "aligned": True, "pass": False}


def test_measures_errors():
    tid = np.arange(100, 110)
    q = [CondExpr("if_gt", 4, 2, 1, 3, g=7, h=3), CondExpr("if_gt", 4, 2, 1, 3, g=2, h=3),
         CondExpr("if_gt", 1, 2, 5, 3, g=7, h=3), CondExpr("if_gt", 1, 2, 5, 3, g=2, h=3)]
    # Y: 6, 4, 3, 8; answered: right, the other branch (6), another digit (0), not a digit
    m = measures(q, np.array([106, 106, 100, 7]), tid, np.random.default_rng(0), 50, 0.05)
    assert m["accuracy"]["mean"] == 0.25 and m["errors"] == {"n": 3, "other_branch": 1, "other_digit": 1, "not_a_digit": 1}
    assert set(m["gaps"]) == {"branch", "shortcut"} and m["cells"]["1/1"]["mean"] == 1.0


def test_toy_run(tmp_path):
    cfg = load_screen_config("configs/smoke_toy_screen.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_screen(cfg, run_id="smoke")
    s = json.loads((root / "core" / "screen.json").read_text())
    assert set(s["forms"]) == set(FORMATS)
    for form in FORMATS:
        m = s["forms"][form]["8"]
        assert m["aligned"] and m["n"] == 16 and m["labels"][-1] == "␣"
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["identical"]
    arr = np.load(root / "core" / "screen_arrays.npz")
    assert arr["if_gt__8__digit_logprobs"].shape == (16, 10)
