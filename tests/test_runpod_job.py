import importlib.util
import itertools
from pathlib import Path

import pytest

from directions.remote import pack_results

ENDPOINT = "https://x/v2/ep"
API_KEY = "k"
JOB_ID = "job1"
EXIT_CODE = 3

_SPEC = importlib.util.spec_from_file_location(
    "runpod_job", Path(__file__).resolve().parent.parent / "scripts" / "runpod_job.py"
)
runpod_job = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_job)


def _completed(**output) -> dict:
    return {"status": "COMPLETED", "output": output}


def _patch_requests(monkeypatch: pytest.MonkeyPatch, statuses) -> list[str]:
    calls: list[str] = []
    statuses = iter(statuses)

    def fake_request(url, api_key, payload=None):
        calls.append(url)
        return {"status": next(statuses), "output": {}}

    monkeypatch.setattr(runpod_job, "_request", fake_request)
    monkeypatch.setattr(runpod_job.time, "sleep", lambda s: None)
    return calls


def test_collect_unpacks_completed_job(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "metrics.json").write_text("{}")
    bundle = pack_results(src, max_output_mb=1)

    code = runpod_job.collect(_completed(returncode=0, log_tail="ok", **bundle), tmp_path / "out")

    assert code == 0
    assert (tmp_path / "out" / "metrics.json").exists()


def test_collect_propagates_remote_exit_code(tmp_path: Path) -> None:
    final = _completed(returncode=EXIT_CODE, log_tail="boom", results_tar_b64=None)
    assert runpod_job.collect(final, tmp_path / "out") == EXIT_CODE


def test_collect_fails_on_handler_exception(tmp_path: Path) -> None:
    final = _completed(success=False, error="TypeError", traceback="...")
    assert runpod_job.collect(final, tmp_path / "out") == 1


def test_collect_fails_on_platform_failure(tmp_path: Path) -> None:
    assert runpod_job.collect({"status": "FAILED", "error": "worker died"}, tmp_path / "out") == 1


def test_collect_fails_on_oversized_results_even_with_zero_exit(tmp_path: Path) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, results_error="too big")
    assert runpod_job.collect(final, tmp_path / "out") == 1


def test_wait_polls_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_requests(monkeypatch, ["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])

    final = runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=60, poll_s=0)

    assert final["status"] == "COMPLETED"
    assert all(url.endswith(f"/status/{JOB_ID}") for url in calls)
    assert len(calls) == 3


def test_wait_cancels_on_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_requests(monkeypatch, itertools.repeat("IN_PROGRESS"))
    clock = itertools.chain([0], itertools.repeat(100))
    monkeypatch.setattr(runpod_job.time, "time", lambda: next(clock))

    final = runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=10, poll_s=0)

    assert final["status"] == "TIMED_OUT"
    assert calls[-1].endswith(f"/cancel/{JOB_ID}")


def test_main_rejects_bad_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", API_KEY)
    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", "[]"]) == 1
    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", '"python"']) == 1
