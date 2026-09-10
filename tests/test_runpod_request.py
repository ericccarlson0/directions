import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "runpod_request", Path(__file__).resolve().parent.parent / "scripts" / "runpod_request.py"
)
runpod_request = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_request)

REPO = Path(__file__).resolve().parent.parent
REQUEST = "request: r1\ncommand: uv run directions pilot --config configs/smoke_toy.yaml\n"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "gpu-run.yaml"
    p.write_text(text)
    return p


def test_committed_request_file_is_valid() -> None:
    params = runpod_request.load_request(REPO / ".github" / "gpu-run.yaml")
    assert params["command"]
    assert params["gpu_tier"] in runpod_request.GPU_TIERS
    assert params["gpu_type"] == ""  # `gpu_type: ""  # comment` must not keep the quotes (run 34491675423)
    assert params["flash_env"] == "ci" and params["datacenter"] == "EUR-NO-1"


def test_push_reads_the_file_and_fills_defaults(tmp_path: Path) -> None:
    params = runpod_request.resolve("push", None, _write(tmp_path, REQUEST))
    assert params["request"] == "r1"
    assert params["command"] == "uv run directions pilot --config configs/smoke_toy.yaml"
    assert params["gpu_tier"] == "ADA_24" and params["gpu_type"] == "" and params["flash_env"] == "ci"
    assert params["timeout_minutes"] == "300" and params["max_output_mb"] == "8"


def test_numbers_and_nulls_in_the_file_become_strings(tmp_path: Path) -> None:
    text = REQUEST + "timeout_minutes: 45\nmax_output_mb: 2.5\ngpu_type:\n"
    params = runpod_request.load_request(_write(tmp_path, text))
    assert params["timeout_minutes"] == "45" and params["max_output_mb"] == "2.5" and params["gpu_type"] == ""


def test_dispatch_uses_inputs_and_ignores_the_file(tmp_path: Path) -> None:
    inputs = {"command": " python -c 'print(1)' ", "gpu_tier": "AMPERE_24", "gpu_type": "", "datacenter": "EUR-NO-1",
              "flash_env": "dev", "timeout_minutes": "10", "max_output_mb": "8"}
    params = runpod_request.resolve("workflow_dispatch", inputs, tmp_path / "missing.yaml")
    assert params["request"] == "manual"
    assert params["command"] == "python -c 'print(1)'"
    assert params["gpu_tier"] == "AMPERE_24" and params["flash_env"] == "dev"


def test_dispatch_requires_a_command() -> None:
    with pytest.raises(runpod_request.RequestError, match="command"):
        runpod_request.resolve("workflow_dispatch", {"command": ""}, Path("unused"))


@pytest.mark.parametrize(
    "text, message",
    [
        ("command: x\n", "`request` is required"),
        ("request: r\n", "`command` is required"),
        (REQUEST + "gpu_tier: H100\n", "gpu_tier"),
        (REQUEST + "flash_env: 'a b'\n", "flash_env"),
        (REQUEST + "timeout_minutes: soon\n", "timeout_minutes"),
        (REQUEST + "max_output_mb: 0\n", "max_output_mb"),
        (REQUEST + "datacenter: ''\n", "datacenter"),
        (REQUEST + "nonsense: 1\n", "unknown keys"),
        (REQUEST + "gpu_type: [a]\n", "scalar"),
        ("- a\n- b\n", "mapping"),
        ("request: r\ncommand: |\n  a\n  b\n", "one line"),
        ("request: r\nrequest: s\ncommand: x\n", "duplicate key"),
        ("request: r\ncommand x\n", "expected `key: value`"),
        ("request: r\n1bad: x\ncommand: x\n", "invalid key"),
        ("request: r\ncommand: 'unbalanced\n", "unbalanced quote"),
    ],
)
def test_invalid_requests_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(runpod_request.RequestError, match=message):
        runpod_request.load_request(_write(tmp_path, text))


def test_flat_mapping_parser_handles_comments_and_quotes() -> None:
    text = (
        "# leading comment\n"
        "request: r1   # trailing comment\n"
        "command: \"uv run x --flag '#notacomment'\"\n"
        "gpu_type: ''\n"
        'gpu_tier: ""   # quoted empty value with a trailing comment\n'
        "datacenter: 'EUR-NO-1'\n"
        "timeout_minutes: 45\n"
        "\n"
        "max_output_mb:2.5\n"
    )
    assert runpod_request.parse_flat_mapping(text) == {
        "request": "r1",
        "command": "uv run x --flag '#notacomment'",
        "gpu_type": "",
        "gpu_tier": "",
        "datacenter": "EUR-NO-1",
        "timeout_minutes": "45",
        "max_output_mb": "2.5",
    }


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(runpod_request.RequestError, match="not found"):
        runpod_request.load_request(tmp_path / "absent.yaml")


def test_main_prints_github_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, REQUEST + "gpu_tier: ADA_48_PRO\n")
    assert runpod_request.main(["--file", str(path), "--event", "push"]) == 0
    out = capsys.readouterr().out
    lines = dict(line.split("=", 1) for line in out.strip().splitlines())
    assert set(lines) == set(runpod_request.FIELDS)
    assert lines["gpu_tier"] == "ADA_48_PRO" and lines["command"].startswith("uv run")


def test_main_reports_errors_with_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, "request: r\n")
    assert runpod_request.main(["--file", str(path), "--event", "push"]) == 2
    assert "request error" in capsys.readouterr().err
    assert runpod_request.main(["--event", "workflow_dispatch", "--inputs-json", "[1]"]) == 2
