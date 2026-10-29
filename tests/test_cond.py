"""The conditional task in full (docs/DECISIONS.md D63): the sources and factorial quads do what their definitions say,
the token groups partition the query, the readings follow the rules, and every part runs on the toy model."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from directions.cond import (
    CATS,
    GROUPS,
    MAP_TYPES,
    PAIRS,
    map_readouts,
    map_sources,
    pair_quad,
    pair_reading,
    run_cond,
    token_groups,
)
from directions.config import load_cond_config
from directions.screen import CondExpr, draw_balanced
from directions.seeds import rng_for


@pytest.fixture(scope="module")
def draw():
    pool = draw_balanced("if_gt", 32, rng_for(0, "pool"), set())
    pt = {e.text() for e in pool}
    return draw_balanced("if_gt", 64, rng_for(0, "bases"), pt), pt


def test_map_sources(draw):
    bases, pt = draw
    n_cond = 0
    for i, b in enumerate(bases):
        s = map_sources(b, MAP_TYPES["if_gt"], rng_for(0, "s", i), pt)
        for v in s.values():
            if v is not None:
                assert v[0].text() not in pt and v[0].S1 != v[0].S2 and v[0].g != v[0].h
        if s["cond"] is not None:
            n_cond += 1
            c, pred = s["cond"]
            assert c.C != b.C and pred == b.other == c.Y and (c.a, c.b, c.c, c.d) == (b.a, b.b, b.c, b.d)
            assert replace(b, g=c.g).C != b.C and replace(b, h=c.h).C != b.C  # both crossings flip
        if s["cond_same"] is not None:
            cs, pred = s["cond_same"]
            assert cs.C == b.C and pred == b.Y and replace(b, h=cs.h).C != b.C  # the h crossing flips
        if s["chosen"] is not None:
            ch, pred = s["chosen"]
            assert ch.C == b.C and pred == ch.Y and ch.Y not in (b.Y, b.other)
        if s["chosen_same"] is not None:
            cs, pred = s["chosen_same"]
            assert cs.Y == b.Y == pred and cs.text() != b.text() and cs.other == b.other
    assert n_cond > 10
    g = CondExpr("if_given", 1, 2, 3, 4, k=1)
    s = map_sources(g, MAP_TYPES["if_given"], rng_for(0, "g"), set())
    assert s["cond"][0].k == 0 and s["cond"][1] == 7


@pytest.mark.parametrize("pair", PAIRS["if_gt"])
def test_pair_quads(draw, pair):
    bases, pt = draw
    quads = [pair_quad(pair, b, rng_for(0, "q", pair, i), pt) for i, b in enumerate(bases)]
    quads = [q for q in quads if q is not None]
    assert len(quads) > 10
    for q in quads:
        e, x, z, xz = q
        Y = [p.Y for p in q]
        assert len({p.text() for p in q}) == 4 and not ({p.text() for p in q} & pt)
        if pair == "CB":
            assert x.C != e.C and z.C == e.C and xz.C == x.C and len(set(Y)) == 4
            assert (z.S1 != e.S1 and z.S2 != e.S2) and (x.a, x.b, x.c, x.d) == (e.a, e.b, e.c, e.d)
        elif pair == "CU":
            assert Y[0] == Y[2] and len({Y[0], Y[1], Y[3]}) == 3 and x.C != e.C
        elif pair == "AB":
            assert len(set(Y)) == 4 and all(p.other == e.other for p in q) and all(p.C == e.C for p in q)
        elif pair == "CN":
            assert x.C == e.C and Y[0] == Y[1] and Y[2] == Y[3] and Y[0] != Y[2]
    mixed = CondExpr("if_mixed", 4, 2, 1, 3, g=5, h=3, cmp=">")
    q = pair_quad("SG", mixed, rng_for(0, "sg"), set())
    assert [p.C for p in q] == [1, 0, 0, 1] and q[1].cmp == "<" and q[3].cmp == "<"


def test_token_groups():
    labels = ["IF", "(", "<g>", "<cmp>", "<h>", ",", " ", "<a>", "+", "<b>", ",", " ", "<c>", "+", "<d>", ")", " =", "␣"]
    g1, g0 = token_groups(labels, 1), token_groups(labels, 0)
    assert g1["Cd"] == [2, 3, 4] and g1["Bch"] == [7, 8, 9] and g1["Bun"] == [12, 13, 14] and g0["Bch"] == [12, 13, 14]
    assert g1["E"] == [16] and g1["A"] == [17] and sorted(j for v in g1.values() for j in v) == list(range(18))
    assert tuple(g1) == GROUPS


def test_map_readouts():
    labels = ["IF", "(", "<g>", "<cmp>", "<h>", ",", " =", "␣"]
    rps = [0, 1, 2, 3]
    P = len(labels)

    def r(lo, hi=None):
        lo = np.asarray(lo, dtype=float)
        return {"lo": lo, "hi": lo if hi is None else np.asarray(hi, dtype=float), "mean": lo}

    cond_lo = np.zeros((4, P))
    cond_lo[:2, 2] = cond_lo[:3, 4] = 1  # g holds it to 1, h to 2
    cond_lo[1:3, 5] = 1  # "," from 1 to 2
    cond_lo[2:, 7] = 1  # the answer from 2
    same_change = np.zeros((4, P))
    same_change[:3, 4] = 1  # h holds the digit (cond_same changes the answer there)
    rates = {"cond": {"flip": r(cond_lo)}, "cond_same": {"flip": r(np.zeros((4, P))), "change": r(same_change)},
             "chosen": {"flip": r(np.eye(4, P, k=4))}}
    out = map_readouts(rates, labels, rps, 0.5, 0.5)
    assert out["R_C"] == 2 and out["C_out"] == {"<g>": 1, "<cmp>": None, "<h>": 2}
    assert out["value_C"] == {"<g>": 0, ",": 1, "␣": 2} and out["digit_C"] == {"<h>": [0, 2]}
    assert out["route"][1] == ["<g>", "<h>", ","] and out["G_ch"] == 3  # the eye puts the answer token at read point 3


def test_pair_reading():
    assert pair_reading({"CB": 28, "AB": 27, "CN": None, "CU": 28}, {})["order"].startswith("together")
    assert pair_reading({"CB": 30, "AB": 26, "CN": None}, {})["order"].startswith("compute, then select")
    assert pair_reading({"CB": 22, "AB": 26, "CN": None}, {})["order"].startswith("select, then compute")
    assert pair_reading({"CB": None, "AB": 26}, {})["order"].startswith("never")
    assert pair_reading({"CB": 28, "AB": 27, "CN": 29}, {})["null_clear"] is False
    assert pair_reading({"CB": 28, "AB": 27, "CN": 31, "CU": 25}, {})["CU_minus_CB"] == -3
    loc = {"SG": {"20": {"lowest": "Cd"}, "21": {"lowest": "Cd"}, "22": {"lowest": "A"}}}
    assert pair_reading({"SG": 21, "CB": 28, "AB": 27}, loc)["computed"] == "at the condition's tokens"
    assert CATS[0] == "11"


@pytest.mark.parametrize("part", ["map", "sets", "pairs", "probe", "steer"])
def test_toy_run(tmp_path, part):
    cfg = load_cond_config("configs/smoke_toy_cond.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_cond(cfg, part, run_id=part)
    out = json.loads((root / "core" / f"{part}.json").read_text())
    assert out["form"] == "if_gt" and out["read_points"] == [0, 1, 2]
    meta = json.loads((root / "metadata.json").read_text())
    if part == "pairs":
        assert all(r["additive_equals_joint_at_0"] == 1.0 for r in out["pairs"].values() if "rates" in r)
        assert "order" in out["reading"]
    if part in ("map", "pairs"):
        assert meta["identity_check"] and all(meta["identity_check"].values())
    if part == "probe":
        assert set(out["probes"]) == {f"{s}/{t}" for s in ("Cd_last", "E", "A") for t in ("C", "C|Y", "Sun|Y")}
    if part == "steer":
        assert set(out["arms"]) == {"h", "A"} and out["sites"] == {"h": "<h>", "A": "␣"}
        assert {"steer", "swap", "null0"} <= set(out["arms"]["h"]["rates"]) and out["n"] >= 2
