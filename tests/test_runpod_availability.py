"""scripts/runpod_availability.py: data centers with GPUs of each tier."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "runpod_availability", Path(__file__).resolve().parents[1] / "scripts" / "runpod_availability.py")
runpod_availability = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_availability)


def test_availability_groups_available_gpus_by_tier_and_datacenter():
    dcs = [
        {"id": "EUR-NO-1", "listed": True, "gpuAvailability": [
            {"gpuTypeId": "NVIDIA GeForce RTX 4090", "available": True, "stockStatus": "Low"},
            {"gpuTypeId": "NVIDIA A100 80GB PCIe", "available": False, "stockStatus": None}]},
        {"id": "AP-IN-1", "listed": True, "gpuAvailability": [
            {"gpuTypeId": "NVIDIA H100 80GB HBM3", "available": True, "stockStatus": "Medium"}]},
    ]
    table = runpod_availability.availability(dcs)
    assert table["ADA_24"] == {"EUR-NO-1": [("GeForce RTX 4090", "Low")]}
    assert table["ADA_80_PRO"] == {"AP-IN-1": [("H100 80GB HBM3", "Medium")]}
    assert table["AMPERE_80"] == {}  # unavailable entries are dropped
    text = runpod_availability.format_table(table, None, None, flash_only=False)
    assert "ADA_80_PRO     AP-IN-1: H100 80GB HBM3 [Medium]" in text and "AMPERE_80      (none)" in text
    assert runpod_availability.format_table(table, "ADA_24", "AP-IN-1") == "ADA_24         (none)"
    # by default only the data centers Flash can place a volume in are shown (AP-IN-1 is not one)
    assert runpod_availability.format_table(table, "ADA_80_PRO", None) == "ADA_80_PRO     (none)"
    assert "EUR-NO-1" in runpod_availability.format_table(table, "ADA_24", None)


def test_main_needs_the_api_key(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runpod_availability.main([]) == 2
