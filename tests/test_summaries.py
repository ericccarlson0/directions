"""scripts/composition_summary.py and scripts/serial_summary.py: the across-model tables from stored summaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import composition_summary  # noqa: E402
import serial_summary  # noqa: E402


def _verdict(v, peak=0.01, keep=0.3):
    return {"verdict": v, "first_positive": None, "first_negative_after": None, "excess_peak": peak, "excess_peak_read_point": 5,
            "natural_cos_peak": 0.9, "natural_cos_peak_read_point": 3, "keep_along_component_max": keep}


def _model(v_g, v_f="composed_only"):
    comp = {"references": ["g", "f"], "constructions": {"learned": {"handover": 9, "per_reference": {"g": _verdict(v_g), "f": _verdict(v_f)}, "peaks_in_step_order": True}}}
    return {"tasks": {"fg": {"handover": {"learned": {"read_point": 9}}, "composition": comp},
                      "g": {"handover": {"learned": {"read_point": 7}}}, "f": {"handover": {"learned": {"read_point": 8}}}}}


def test_composition_summary_rows_and_rule():
    models = {"qwen3_0.6b": _model("composed_only"), "qwen3_1.7b": _model("composed_only"), "qwen3_4b": _model("staircase"),
              "qwen3_8b": _model("composed_only"), "olmo3_7b": _model("staircase")}
    s = composition_summary.summarise(models)
    assert len(s["rows"]) == 5 and s["rows"][0]["components"]["g"]["component_handover"] == 7
    rule = s["rule"]["fg"]["learned"]
    assert rule["g"]["verdict"] == "composed_only" and rule["g"]["counts"] == {"composed_only": 3, "staircase": 1} and rule["g"]["n_sizes"] == 4
    assert rule["f"]["verdict"] == "composed_only"
    models["qwen3_8b"] = _model("staircase")
    assert composition_summary.summarise(models)["rule"]["fg"]["learned"]["g"]["verdict"] == "undecided"
    assert "| qwen3_4b | fg | learned | 9; g 7, f 8 | g: staircase" in composition_summary.table(s)


def test_serial_summary_best_layer_and_share():
    grid = [{"layer": 8, "serial": 1.0, "null_mean": 0.1, "second_alone": 0.5, "serial_over_null": {"p_value": 0.001}},
            {"layer": 11, "serial": 2.0, "null_mean": 0.2, "second_alone": 0.7, "serial_over_null": {"p_value": 0.001}}]
    comp = {"task": "fg", "steps": ["g", "f"], "construction": "fv", "first": {"label": "g", "layer": 8}, "first_alone": {"effect": 0.3},
            "composed_control": {"effect": 4.0, "layer": 8}, "superposition_at_first_layer": {"effect": 3.5, "angle_deg": 60.0},
            "grid": grid, "bright_layers": [8, 11], "reading": "everywhere"}
    rows = serial_summary.summarise({"m": {"compositions": [comp, {**comp, "construction": "learned", "composed_control": None, "grid": [], "bright_layers": [], "reading": "nowhere"}]}})
    assert rows[0]["best_layer"] == 11 and rows[0]["best_serial"] == 2.0 and rows[0]["best_share"] == 0.5 and rows[0]["serial_by_layer"] == {"8": 1.0, "11": 2.0}
    assert rows[1]["best_layer"] is None and rows[1]["best_share"] is None and rows[1]["composed_control"] is None
    assert "| m | fg | fv | +0.30 (8) | +4.00 (8) | +3.50 (60°) | 11, +2.00 (+0.20; +0.70) | 2/2 [8, 11] | everywhere |" in serial_summary.table(rows)
