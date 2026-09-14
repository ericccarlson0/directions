import json
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


def test_wait_calls_progress_after_every_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_requests(monkeypatch, ["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])
    ticks = []

    runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=60, poll_s=0, progress=lambda: ticks.append(1))

    assert len(ticks) == 3


class FakeLogObject:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.ranges: list[str] = []

    def get_object(self, Bucket: str, Key: str, Range: str):
        self.ranges.append(Range)
        start = int(Range.removeprefix("bytes=").rstrip("-"))
        if start >= len(self.content):
            raise RuntimeError("InvalidRange (416)")
        return {"Body": __import__("io").BytesIO(self.content[start:])}


def test_log_tail_prints_only_new_bytes(capsys: pytest.CaptureFixture) -> None:
    obj = FakeLogObject(b"line 1\n")
    tail = runpod_job.LogTail(obj, VOLUME_ID, "results/r1/runner.log")

    tail()
    obj.content += b"line 2\n"
    tail()
    tail()

    assert capsys.readouterr().out == "line 1\nline 2\n"
    assert obj.ranges == ["bytes=0-", "bytes=7-", "bytes=14-"]


def test_wait_cancels_on_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_requests(monkeypatch, itertools.repeat("IN_PROGRESS"))
    clock = itertools.chain([0], itertools.repeat(100))
    monkeypatch.setattr(runpod_job.time, "time", lambda: next(clock))

    final = runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=10, poll_s=0)

    assert final["status"] == "TIMED_OUT"
    assert calls[-1].endswith(f"/cancel/{JOB_ID}")


def test_wait_gives_up_when_no_worker_takes_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_requests(monkeypatch, itertools.repeat("IN_QUEUE"))
    clock = itertools.chain([0, 5, 15], itertools.repeat(100))
    monkeypatch.setattr(runpod_job.time, "time", lambda: next(clock))

    final = runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=1000, poll_s=0, queue_deadline_s=50)

    assert final["status"] == "NO_WORKER" and "IN_QUEUE" in final["error"]
    assert calls[-1].endswith(f"/cancel/{JOB_ID}")
    # a job that a worker has taken is not subject to the queue bound
    calls = _patch_requests(monkeypatch, itertools.chain(["IN_QUEUE", "IN_PROGRESS", "IN_PROGRESS"], itertools.repeat("COMPLETED")))
    clock = itertools.chain([0], itertools.repeat(100))
    monkeypatch.setattr(runpod_job.time, "time", lambda: next(clock))
    final = runpod_job.wait(ENDPOINT, API_KEY, JOB_ID, deadline_s=1000, poll_s=0, queue_deadline_s=50)
    assert final["status"] == "COMPLETED"


def test_main_rejects_bad_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", API_KEY)
    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", "[]"]) == 1
    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", '"python"']) == 1


VOLUME_ID = "vol1"
DATACENTER = "EUR-NO-1"
VOLUMES = [
    {"id": "other", "name": "directions", "dataCenter": "US-KS-2"},
    {"id": VOLUME_ID, "name": "directions", "dataCenter": DATACENTER},
]


class FakeS3:
    def __init__(self, keys: list[str], page_size: int = 2) -> None:
        self.keys = keys
        self.page_size = page_size
        self.downloaded: list[str] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):
        matching = [k for k in self.keys if k.startswith(Prefix)]
        for i in range(0, len(matching), self.page_size):
            yield {"Contents": [{"Key": k} for k in matching[i : i + self.page_size]]}

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_text(Key)
        self.downloaded.append(Key)


def test_download_results_fetches_every_object_under_the_prefix(tmp_path: Path) -> None:
    client = FakeS3(["results/r1/a.json", "results/r1/core/b.json", "results/r1/core/", "results/r2/c.json"])

    n = runpod_job.download_results(client, VOLUME_ID, "results/r1", tmp_path)

    assert n == 2
    assert (tmp_path / "a.json").read_text() == "results/r1/a.json"
    assert (tmp_path / "core" / "b.json").exists()
    assert not (tmp_path / "c.json").exists()


def test_resolve_volume_id_matches_name_and_datacenter() -> None:
    assert runpod_job.resolve_volume_id(VOLUMES, "directions", DATACENTER) == VOLUME_ID


def test_resolve_volume_id_accepts_the_data_center_id_key() -> None:
    volumes = [{"id": VOLUME_ID, "name": "directions", "dataCenterId": DATACENTER}]
    assert runpod_job.resolve_volume_id(volumes, "directions", DATACENTER) == VOLUME_ID


def test_resolve_volume_id_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="EU-CZ-1"):
        runpod_job.resolve_volume_id(VOLUMES, "directions", "EU-CZ-1")


def test_collect_downloads_volume_results(tmp_path: Path) -> None:
    seen = []

    def download(results_path, out):
        seen.append((results_path, out))
        return 4

    final = _completed(returncode=0, log_tail="", results_tar_b64=None, results_path="results/r1")

    assert runpod_job.collect(final, tmp_path / "out", download) == 0
    assert seen == [("results/r1", tmp_path / "out")]


def test_collect_fails_when_the_worker_reports_no_shipped_source(tmp_path: Path) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, code_fingerprint="old", expected_fingerprint="new")
    assert runpod_job.collect(final, tmp_path / "out", expected_source="a" * 64) == 1


def test_collect_without_an_expected_source_does_not_verify(tmp_path: Path) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, code_fingerprint="old", expected_fingerprint="new")
    assert runpod_job.collect(final, tmp_path / "out") == 0


def test_collect_fails_on_volume_results_without_a_downloader(tmp_path: Path) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, results_path="results/r1")
    assert runpod_job.collect(final, tmp_path / "out") == 1


def test_main_requires_s3_keys_before_submitting_for_volume_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", API_KEY)
    monkeypatch.delenv("RUNPOD_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_S3_SECRET_KEY", raising=False)
    monkeypatch.setattr(runpod_job, "submit", lambda *a, **k: pytest.fail("submitted without S3 keys"))

    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", '["python"]', "--results-name", "r1"]) == 1


SHIPPED = "a" * 64


def _stale() -> dict:
    """An old handler that ran (or refused) without reporting the shipped digest."""
    return _completed(returncode=0, code_fingerprint="old", expected_fingerprint="new")


def _patch_run(monkeypatch: pytest.MonkeyPatch, finals: list[dict]) -> tuple[list[str], list[float]]:
    ids = iter(f"job{i}" for i in itertools.count(1))
    submitted: list[str] = []
    slept: list[float] = []
    finals_iter = iter(finals)

    def fake_submit(endpoint_url, api_key, job_input):
        submitted.append(next(ids))
        return submitted[-1]

    monkeypatch.setattr(runpod_job, "submit", fake_submit)
    monkeypatch.setattr(runpod_job, "wait", lambda *a, **k: next(finals_iter))
    monkeypatch.setattr(runpod_job.time, "sleep", lambda s: slept.append(s))
    return submitted, slept


def test_run_resubmits_after_a_stale_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good = _completed(returncode=0, log_tail="", results_tar_b64=None, source_sha256=SHIPPED, handler_stale=True,
                      code_fingerprint="old", expected_fingerprint="new")
    submitted, slept = _patch_run(monkeypatch, [_stale(), _stale(), good])
    job_id_file = tmp_path / "job-id"
    job = {"source_sha256": SHIPPED}

    final = runpod_job.run(ENDPOINT, API_KEY, job, 100, 0, job_id_file=job_id_file, stale_retries=3, stale_wait_s=7)

    assert final is good
    assert submitted == ["job1", "job2", "job3"] and slept == [7, 7]
    assert job_id_file.read_text() == "job3"
    assert runpod_job.collect(final, tmp_path / "out", expected_source=SHIPPED) == 0  # stale handler, right source


def test_run_gives_up_after_the_retries_and_collect_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    submitted, slept = _patch_run(monkeypatch, [_stale()] * 3)

    final = runpod_job.run(ENDPOINT, API_KEY, {"source_sha256": SHIPPED}, 100, 0, stale_retries=2, stale_wait_s=1)

    assert runpod_job.is_stale(final, SHIPPED)
    assert submitted == ["job1", "job2", "job3"] and slept == [1, 1]
    assert runpod_job.collect(final, tmp_path / "out", expected_source=SHIPPED) == 1


def test_run_does_not_retry_a_failed_or_fresh_job(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = {"status": "FAILED", "output": {}, "error": "CUDA out of memory"}
    submitted, slept = _patch_run(monkeypatch, [failed])
    assert runpod_job.run(ENDPOINT, API_KEY, {"source_sha256": SHIPPED}, 100, 0) is failed
    assert submitted == ["job1"] and slept == []
    assert not runpod_job.is_stale(failed, SHIPPED)
    assert not runpod_job.is_stale(_completed(returncode=0, source_sha256=SHIPPED), SHIPPED)
    assert not runpod_job.is_stale(_stale(), None)  # no expected digest: nothing to verify


def test_is_stale_with_shipped_source_means_the_worker_did_not_run_it() -> None:
    ran = _completed(returncode=0, source_sha256=SHIPPED, code_fingerprint="old", expected_fingerprint="new")
    assert not runpod_job.is_stale(ran, SHIPPED)  # a stale handler that ran the shipped tree is fine
    old_handler = _completed(returncode=0, code_fingerprint="old", expected_fingerprint="new")
    assert runpod_job.is_stale(old_handler, SHIPPED)
    assert runpod_job.is_stale(_completed(returncode=0, source_sha256="b" * 64), SHIPPED)


def test_collect_with_shipped_source_warns_on_a_stale_handler_but_succeeds(tmp_path: Path, capsys) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, source_sha256=SHIPPED, source_dir="/app/jobs/a",
                       handler_stale=True, code_fingerprint="old", expected_fingerprint="new")
    assert runpod_job.collect(final, tmp_path / "out", expected_source=SHIPPED) == 0
    assert "harmless with shipped source" in capsys.readouterr().err


def test_collect_with_shipped_source_fails_when_the_worker_ran_something_else(tmp_path: Path, capsys) -> None:
    final = _completed(returncode=0, log_tail="", results_tar_b64=None, code_fingerprint="same", expected_fingerprint="same")
    assert runpod_job.collect(final, tmp_path / "out", expected_source=SHIPPED) == 1
    assert "did not run the shipped source" in capsys.readouterr().err


def test_run_retries_when_an_old_handler_ignores_the_shipped_source(monkeypatch: pytest.MonkeyPatch) -> None:
    old = _completed(returncode=0, code_fingerprint="old", expected_fingerprint="new")
    good = _completed(returncode=0, source_sha256=SHIPPED)
    submitted, slept = _patch_run(monkeypatch, [old, good])
    assert runpod_job.run(ENDPOINT, API_KEY, {"source_sha256": SHIPPED}, 100, 0, stale_wait_s=3) is good
    assert submitted == ["job1", "job2"] and slept == [3]


def test_main_ships_the_source_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", API_KEY)
    seen: dict = {}

    def fake_run(endpoint_url, api_key, job_input, *a, **k):
        seen.update(job_input)
        return _completed(returncode=0, log_tail="", results_tar_b64=None, source_sha256=job_input["source_sha256"])

    monkeypatch.setattr(runpod_job, "run", fake_run)
    monkeypatch.setattr(runpod_job, "pack_source", lambda root, commit: {"source_tar_b64": "x", "source_sha256": SHIPPED, "source_size_bytes": 3})

    assert runpod_job.main(["--endpoint-url", ENDPOINT, "--argv-json", '["python"]', "--out", str(tmp_path)]) == 0
    assert seen["source_sha256"] == SHIPPED and seen["source_tar_b64"] == "x"  # always shipped, no flag needed


def test_is_stale_recognises_an_old_handler_rejecting_shipped_inputs() -> None:
    failed = {"status": "FAILED", "error": "run_experiment() got an unexpected keyword argument 'source_sha256'", "workerId": "w1"}
    assert runpod_job.is_stale(failed, SHIPPED)
    assert not runpod_job.is_stale(failed, None)
    assert not runpod_job.is_stale({"status": "FAILED", "error": "CUDA out of memory"}, SHIPPED)


def test_run_terminates_the_stale_worker_before_resubmitting(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = {"status": "FAILED", "error": "run_experiment() got an unexpected keyword argument 'source_tar_b64'", "workerId": "w1"}
    good = _completed(returncode=0, source_sha256=SHIPPED)
    submitted, slept = _patch_run(monkeypatch, [failed, good])
    terminated: list[tuple[str, str]] = []
    monkeypatch.setattr(runpod_job, "terminate_worker", lambda api_key, worker_id: terminated.append((api_key, worker_id)) or True)

    assert runpod_job.run(ENDPOINT, API_KEY, {"source_sha256": SHIPPED}, 100, 0, stale_wait_s=2) is good
    assert terminated == [(API_KEY, "w1")] and submitted == ["job1", "job2"] and slept == [2]


def test_terminate_worker_posts_the_mutation_and_reports_rejections(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import io

    sent: list[dict] = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        sent.append(json.loads(req.data))
        assert req.get_header("User-agent") == runpod_job.USER_AGENT
        body = {"errors": [{"message": "no"}]} if len(sent) > 1 else {"data": {"podTerminate": None}}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(runpod_job.urllib.request, "urlopen", fake_urlopen)
    assert runpod_job.terminate_worker(API_KEY, "w1") is True
    assert sent[0]["variables"] == {"input": {"podId": "w1"}}
    assert runpod_job.terminate_worker(API_KEY, "w2") is False
    assert "rejected" in capsys.readouterr().err
