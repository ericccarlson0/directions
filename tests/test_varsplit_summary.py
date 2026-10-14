"""scripts/varsplit_summary.py: the D47 summary on a toy run's output."""

import importlib.util
import json
from pathlib import Path

from directions.config import load_varsplit_config
from directions.varsplit import run_varsplit

_SPEC = importlib.util.spec_from_file_location("varsplit_summary",
                                               Path(__file__).resolve().parents[1] / "scripts" / "varsplit_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_summary_reads_a_toy_run(tmp_path):
    cfg = load_varsplit_config("configs/smoke_toy_varsplit.yaml")
    cfg.output_dir, cfg.figures = str(tmp_path), False
    root = run_varsplit(cfg, run_id="varsplit_toy")
    r = json.loads((root / "core" / "tasks" / "two_lane" / "varsplit.json").read_text())
    s = summary.summarise(r)
    assert set(s["probe"]) == {"S1|Y", "Y"} and set(s["probe"]["Y"]["blocks"]) == {"2", "3"}
    assert set(s["heads"]) == {"2", "3"} and s["heads"]["2"]["reading"] in ("lanes", "one channel", "mixed", "no carrier")
    assert summary.main([str(tmp_path), "--out", str(tmp_path / "s.json")]) == 0
