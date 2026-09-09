import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from directions.remote import ResultsTooLarge, execute, pack_results, unpack_results

MAX_OUTPUT_MB = 1
ONE_MIB = 1024 * 1024
METRICS_JSON = '{"a": 1}'
PNG_BYTES = b"\x89PNG fake"
GIT_COMMIT = "abc123"
EXIT_CODE = 3
MISSING_BINARY = "definitely-not-a-real-binary-xyz"


def _make_results(root: Path) -> Path:
    results = root / "results"
    (results / "core").mkdir(parents=True)
    (results / "core" / "metrics.json").write_text(METRICS_JSON)
    (results / "fig.png").write_bytes(PNG_BYTES)
    return results


def _tar_b64(member_name: str, data: bytes = b"x") -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def _execute(cwd: Path, argv: list[str], **kwargs) -> dict:
    return execute(argv=argv, cwd=cwd, results_dir="results", max_output_mb=MAX_OUTPUT_MB, **kwargs)


def _unpack(result: dict, dest: Path) -> Path:
    unpack_results(result["results_tar_b64"], dest)
    return dest


def test_pack_results_roundtrips_through_unpack_results(tmp_path: Path) -> None:
    results = _make_results(tmp_path / "worker")
    bundle = pack_results(results, max_output_mb=MAX_OUTPUT_MB)

    out = _unpack(bundle, tmp_path / "runner")

    assert (out / "core" / "metrics.json").read_text() == METRICS_JSON
    assert (out / "fig.png").read_bytes() == PNG_BYTES
    assert bundle["results_size_bytes"] > 0
    assert bundle["results_file_count"] == 2


def test_pack_results_rejects_oversized_results_and_names_the_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "big.bin").write_bytes(os.urandom(ONE_MIB))

    with pytest.raises(ResultsTooLarge, match="big.bin"):
        pack_results(results, max_output_mb=0.5)


def test_pack_results_missing_dir_gives_empty_bundle(tmp_path: Path) -> None:
    bundle = pack_results(tmp_path / "nope", max_output_mb=MAX_OUTPUT_MB)
    assert bundle["results_file_count"] == 0
    assert bundle["results_tar_b64"] is None


def test_unpack_results_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        unpack_results(_tar_b64("../escape.txt"), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_execute_runs_command_and_bundles_results(tmp_path: Path) -> None:
    script = (
        "import os, pathlib, sys;"
        "p = pathlib.Path('results/core'); p.mkdir(parents=True);"
        "(p / 'out.json').write_text(os.environ['DIRECTIONS_GIT_COMMIT']);"
        "print('hello from worker'); print('warn', file=sys.stderr)"
    )
    argv = [sys.executable, "-c", script]

    result = _execute(tmp_path, argv, extra_env={"DIRECTIONS_GIT_COMMIT": GIT_COMMIT})

    assert result["returncode"] == 0
    assert "hello from worker" in result["log_tail"]
    assert "warn" in result["log_tail"]

    out = _unpack(result, tmp_path / "unpacked")
    assert (out / "core" / "out.json").read_text() == GIT_COMMIT
    assert "hello from worker" in (out / "runner.log").read_text()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["argv"] == argv
    assert meta["returncode"] == 0
    assert meta["git_commit"] == GIT_COMMIT
    assert meta["duration_s"] >= 0


def test_execute_nonzero_exit_still_returns_bundle(tmp_path: Path) -> None:
    result = _execute(tmp_path, [sys.executable, "-c", f"import sys; print('boom'); sys.exit({EXIT_CODE})"])

    assert result["returncode"] == EXIT_CODE
    assert "boom" in result["log_tail"]
    out = _unpack(result, tmp_path / "unpacked")
    assert (out / "runner.log").exists()


def test_execute_missing_executable_is_reported(tmp_path: Path) -> None:
    result = _execute(tmp_path, [MISSING_BINARY])

    assert result["returncode"] != 0
    assert MISSING_BINARY in result["log_tail"]
