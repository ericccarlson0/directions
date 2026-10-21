"""The command line: a pilot whose task raised exits non-zero; compare treats git state as informational."""

import json
from pathlib import Path

import pytest

from directions.cli import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _run_dir(root: Path, metadata: dict, rejections: list[dict], summary: dict | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "rejections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejections))
    if summary is not None:
        (root / "summary.json").write_text(json.dumps(summary))
    return root


@pytest.mark.parametrize("errors, code", [(0, 0), (2, 1)])
def test_pilot_exits_non_zero_when_a_task_raised(tmp_path, monkeypatch, capsys, errors, code):
    """The pipeline records a task that raised as an "error" rejection and carries on; the command then fails."""
    import directions.pipeline as pipeline

    rejections = [{"stage": "fewshot", "task": "antonym", "reason": "accuracy below the gate", "details": {}}]
    rejections += [{"stage": "error", "task": f"t{i}", "reason": "RuntimeError: boom", "details": {}} for i in range(errors)]
    root = _run_dir(tmp_path / "run", {"seed": 1}, rejections)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **k: root)

    assert main(["pilot", "--config", str(CONFIGS / "smoke_toy.yaml")]) == code
    err = capsys.readouterr().err
    assert ("2 error(s) during the run" in err and "t1: RuntimeError: boom" in err) if errors else err == ""


def test_compare_reports_git_but_does_not_count_it(tmp_path, capsys):
    """Two runs of the same numerical path from different commits, or a clean and a dirty tree, compare identical."""
    a = _run_dir(tmp_path / "a", {"seed": 1, "git": {"commit": "abc", "dirty": False}}, [], {"x": 1.0})
    b = _run_dir(tmp_path / "b", {"seed": 1, "git": {"commit": "def", "dirty": True}}, [], {"x": 1.0})
    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "note (not counted): /metadata.json/git/commit" in out and "identical" in out


def test_compare_still_counts_a_real_difference(tmp_path, capsys):
    a = _run_dir(tmp_path / "a", {"seed": 1, "git": {"commit": "abc"}}, [], {"x": 1.0})
    b = _run_dir(tmp_path / "b", {"seed": 1, "git": {"commit": "abc"}}, [], {"x": 2.0})
    assert main(["compare", str(a), str(b)]) == 1
    assert "/summary.json/x: 1.0 vs 2.0" in capsys.readouterr().out
