"""scripts/runpod_log.py: read a run's live log or results from the network volume."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("runpod_log", Path(__file__).resolve().parents[1] / "scripts" / "runpod_log.py")
runpod_log = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_log)


def test_results_name_accepts_run_id_or_directory():
    assert runpod_log.results_name("34794257716") == "run-34794257716"
    assert runpod_log.results_name("run-34794257716") == "run-34794257716"
    assert runpod_log.results_name("results/run-1/") == "run-1"


def test_main_reports_missing_credentials(monkeypatch, capsys):
    for k in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert runpod_log.main(["1"]) == 2
    assert "RUNPOD_S3_SECRET_KEY" in capsys.readouterr().err
