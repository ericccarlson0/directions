"""scripts/overview_data.py and scripts/overview_figures.py: the overview figures' data agrees with the recorded
summaries, the counts the figures draw are the entries' counts, and every figure renders."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # a dependency on fable; main's environment has none

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs/overview/data.json").read_text())


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rounding_keeps_the_file_small_and_json_clean():
    data = _load("overview_data")
    assert data.r3({"a": [0.12345, np.float32(0.5), float("nan")], "b": np.int64(3)}) == {"a": [0.123, 0.5, None],
                                                                                          "b": 3}


def test_landmarks_are_the_summaries():
    vm = {**json.loads((ROOT / "docs/varmap/summary.json").read_text()),
          **json.loads((ROOT / "docs/varmap/summary_three.json").read_text())}
    for r in DATA["arith"]["rows"]:
        key = f"varmap_{'three_' if r['task'] == 'three_lane' else ''}{r['model']}_seed20260916"
        as_run = vm[key][r["task"]]["as_run"]
        assert r["G"] == as_run["gather_read_point"]
        assert r["last_digit"] == as_run["last_digit_read_point"]
    d57 = json.loads((ROOT / "docs/varmap/d57_operator.json").read_text())
    op = DATA["maxmin"]["operator"]
    for m in ("qwen3_4b", "qwen3_8b"):
        for k in ("G", "O_out", "R_O"):
            assert op[m]["landmarks"][k] == d57[m]["map"]["as_run"][k]
    assert {m: (op[m]["two_site"]["T2"], op[m]["two_site"]["G2"]) for m in op} == {
        "qwen3_4b": (20, 24), "qwen3_8b": (20, 24), "qwen3_14b": (23, 29)}  # D60's outcome


def test_set_swap_rates_are_the_recorded_tables():
    # STATUS, D59 (4B natural arm at read points 16 / 20 / 24) and D60 (14B sums arm at 20 / 23 / 26 / 32)
    o = DATA["maxmin"]["operator"]
    rp = o["qwen3_4b"]["opset"]["read_points"]
    nat = o["qwen3_4b"]["opset"]["natural"]["rates"]
    assert [round(nat["OA"]["mean"][rp.index(m)], 2) for m in (16, 20, 24)] == [0.82, 0.77, 0.83]
    assert [round(nat["EA"]["mean"][rp.index(m)], 2) for m in (16, 20, 24)] == [0.06, 0.64, 0.88]
    rp = o["qwen3_14b"]["opset"]["read_points"]
    sums = o["qwen3_14b"]["opset"]["sums"]["rates"]
    assert [round(sums["FS"]["mean"][rp.index(m)], 2) for m in (20, 23, 26, 32)] == [0.95, 0.82, 0.61, 0.0]


def test_only_the_three_models_mapped_on_every_task():
    text = (ROOT / "docs/overview/data.json").read_text()
    assert "1.7b" not in text and "0.6b" not in text and "one_lane" not in text
    assert {r["model"] for r in DATA["arith"]["rows"]} == {"qwen3_4b", "qwen3_8b", "qwen3_14b"}
    assert set(DATA["maxmin"]["max"]) == set(DATA["maxmin"]["operator"]) == {"qwen3_4b", "qwen3_8b", "qwen3_14b"}


def test_the_count_of_held_sums_is_the_entries():
    """The figures read the count off the DAS summaries' `held` sets at G: two brackets 1 (exploratory), 2, 0 on
    4B-14B (D48-D50); three brackets 0 (D51), and 0 across the answer token and `=` (D54)."""
    rows = {(r["task"], r["model"]): r for r in DATA["arith"]["rows"]}
    das = DATA["arith"]["das"]

    def count(task, m):
        site = das[task][m]["sites"][str(rows[(task, m)]["G"])]
        return sum(1 for v in ("S1", "S2", "S3") if site.get(v, {}).get("held"))

    assert [count("two_lane", m) for m in ("qwen3_4b", "qwen3_8b", "qwen3_14b")] == [1, 2, 0]
    assert [count("three_lane", m) for m in ("qwen3_4b", "qwen3_8b", "qwen3_14b")] == [0, 0, 0]
    assert "extension" in das["two_lane"]["qwen3_4b"]["sites"]["24"]["S1"]["source"]
    for m, variants in das["three_lane_eq"].items():
        for v in variants.values():
            assert not any(c["held"] for site in v["sites"].values() for k, c in site.items() if k.startswith("S"))


def test_every_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    figs = _load("overview_figures")
    assert figs.main([str(ROOT / "docs/overview/data.json"), "--out", str(tmp_path)]) == 0
    for name in figs.FIGURES:
        assert (tmp_path / name).stat().st_size > 50_000


def _game(v):
    """Rates for D59's set family from a characteristic function v(set of group letters)."""
    names = ["O", "F", "S", "P", "E", "A", "FSPEA", "OSPEA", "OFPEA", "OFSEA", "OFSPA", "OFSPE", "OFSPEA"]
    return {n: {"mean": [float(v(set(n)))]} for n in names}


def test_group_shares_on_known_games():
    figs = _load("overview_figures")
    one = figs.group_shares(_game(lambda s: "O" in s))
    assert one[0]["O"][0] == 1 and sum(one[0][g][0] for g in "FSPEA") == 0 and one[1][0] == 0
    pair = figs.group_shares(_game(lambda s: {"O", "A"} <= s))  # held jointly: neither alone, both needed
    assert pair[0]["O"][0] == pair[0]["A"][0] == 0.5 and pair[1][0] == 0
    copies = figs.group_shares(_game(lambda s: bool({"O", "A"} & s)))  # redundant copies: either suffices
    assert copies[0]["O"][0] == copies[0]["A"][0] == 0.5 and copies[1][0] == 0
    vote = figs.group_shares(_game(lambda s: len({"O", "A", "E"} & s) >= 2))  # any two of three: no single share
    assert all(vote[0][g][0] == 0 for g in "OAE") and vote[1][0] == 1
