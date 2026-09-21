"""scripts/runpod_balance.py: the account balance against a planned run."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("runpod_balance", Path(__file__).resolve().parents[1] / "scripts" / "runpod_balance.py")
runpod_balance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_balance)


def test_required_takes_the_estimate_with_a_margin():
    assert runpod_balance.required(None, None, None, 0.5) is None
    assert runpod_balance.required(6.0, None, None, 0.5) == 9.0
    assert runpod_balance.required(None, 2.0, 4.79, 0.0) == 9.58
    assert runpod_balance.required(6.0, 2.0, 4.79, 0.5) == 9.0  # --need wins over --hours/--rate


def test_verdict_compares_balance_with_the_requirement():
    assert runpod_balance.verdict(49.6, None) == (True, "balance 49.60 USD")
    ok, line = runpod_balance.verdict(49.6, 9.0)
    assert ok and line.endswith(": ok")
    ok, line = runpod_balance.verdict(5.0, 9.0)
    assert not ok and line.endswith(": INSUFFICIENT")


def test_main_gates_on_the_balance(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runpod_balance.main([]) == 2
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(runpod_balance, "fetch", lambda key: {"clientBalance": 5.0, "spendLimit": 80, "currentSpendPerHr": 0.0})
    assert runpod_balance.main([]) == 0
    assert runpod_balance.main(["--need", "6"]) == 1
    assert runpod_balance.main(["--need", "3", "--margin", "0.5"]) == 0
    assert "INSUFFICIENT" in capsys.readouterr().out
