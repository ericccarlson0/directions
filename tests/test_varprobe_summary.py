"""scripts/varprobe_summary.py: the D46 summary on a toy run's output."""

import importlib.util
from pathlib import Path

from directions.config import load_varprobe_config
from directions.varprobe import run_varprobe

_SPEC = importlib.util.spec_from_file_location("varprobe_summary",
                                               Path(__file__).resolve().parents[1] / "scripts" / "varprobe_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_summary_reads_a_toy_run(tmp_path):
    import json

    cfg = load_varprobe_config("configs/smoke_toy_varprobe.yaml")
    cfg.output_dir, cfg.n_examples, cfg.n_perm, cfg.figures = str(tmp_path), 30, 6, False
    root = run_varprobe(cfg, run_id="toy")
    r = json.loads((root / "core" / "tasks" / "two_lane" / "varprobe.json").read_text())
    s = summary.summarise(r, G=2)
    assert set(s["probes"]) == {"Y", "S1|Y", "a|S1", "c|S2"} and s["L"] == len(r["read_points"]) - 1
    assert 2 <= s["probes"]["S1|Y"]["peak_at"] <= s["L"]
    assert set(summary.rules(s)) == {"rule_1_answer_only", "rule_2_discarded_before_L", "carried_to_L"}
    assert summary.model_key("varprobe_qwen3_1.7b_seed20260916") == "qwen3_1.7b"
