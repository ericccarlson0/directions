import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from directions.remote import (
    ResultsTooLarge,
    artifact_fingerprints,
    execute,
    pack_results,
    unpack_results,
)

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


WRITE_RESULT = "import pathlib; pathlib.Path('results/out.txt').write_text('ok')"


def _app_with_empty_results(root: Path) -> Path:
    cwd = root / "app"
    (cwd / "results").mkdir(parents=True)
    (cwd / "results" / ".gitkeep").write_text("")
    return cwd


def test_execute_leaves_results_on_the_volume_instead_of_a_tarball(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    cwd = _app_with_empty_results(tmp_path)

    result = _execute(cwd, [sys.executable, "-c", WRITE_RESULT], volume_root=volume, results_name="r1")

    assert result["returncode"] == 0
    assert result["results_tar_b64"] is None
    assert result["results_path"] == "results/r1"
    assert (volume / "results" / "r1" / "out.txt").read_text() == "ok"
    assert (volume / "results" / "r1" / "runner.log").exists()
    assert json.loads((volume / "results" / "r1" / "run_metadata.json").read_text())["results_on_volume"] is True
    assert (cwd / "results" / "out.txt").read_text() == "ok"
    assert result["results_file_count"] == 4


def test_execute_writes_the_log_to_the_volume_while_running(tmp_path: Path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    cwd = _app_with_empty_results(tmp_path)
    live = volume / "results" / "r1" / "runner.log"
    script = (
        "import pathlib, time, sys\n"
        "print('first', flush=True)\n"
        f"time.sleep(0.05); assert 'first' in pathlib.Path({str(live)!r}).read_text()\n"
        "print('second', flush=True)"
    )

    result = _execute(cwd, [sys.executable, "-c", script], volume_root=volume, results_name="r1")

    assert result["returncode"] == 0
    assert live.read_text().splitlines() == ["first", "second"]


def test_execute_falls_back_to_a_tarball_without_a_mounted_volume(tmp_path: Path) -> None:
    cwd = _app_with_empty_results(tmp_path)

    result = _execute(cwd, [sys.executable, "-c", WRITE_RESULT], volume_root=tmp_path / "missing", results_name="r1")

    assert result["results_tar_b64"] is not None
    assert "results_path" not in result


def test_artifact_fingerprints_reads_manifest_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "flash_manifest.json").write_text(json.dumps({"source_fingerprint": "abc"}))
    monkeypatch.setenv("_FLASH_SOURCE_FINGERPRINT", "def")

    assert artifact_fingerprints(tmp_path) == {"code_fingerprint": "abc", "expected_fingerprint": "def"}


def test_artifact_fingerprints_without_manifest_or_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("_FLASH_SOURCE_FINGERPRINT", raising=False)
    assert artifact_fingerprints(tmp_path) == {"code_fingerprint": None, "expected_fingerprint": None}


def test_stale_worker_requires_both_fingerprints_to_differ() -> None:
    from directions.remote import stale_worker

    assert stale_worker({"code_fingerprint": "a", "expected_fingerprint": "b"})
    assert not stale_worker({"code_fingerprint": "a", "expected_fingerprint": "a"})
    assert not stale_worker({"code_fingerprint": None, "expected_fingerprint": "b"})
    assert not stale_worker({"code_fingerprint": "a", "expected_fingerprint": None})


def _git_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("VALUE = 1\n")
    (repo / "README.md").write_text("hello\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"]):
        subprocess.run(cmd, cwd=repo, check=True, env=env)
    (repo / "untracked.txt").write_text("not committed\n")
    return repo


def test_pack_source_ships_the_committed_tree_and_unpack_verifies_it(tmp_path: Path) -> None:
    from directions.remote import pack_source, unpack_source

    repo = _git_repo(tmp_path)
    shipped = pack_source(repo)
    assert shipped["source_size_bytes"] > 0 and len(shipped["source_sha256"]) == 64
    assert pack_source(repo) == shipped  # deterministic (gzip mtime fixed)

    dest = unpack_source(shipped["source_tar_b64"], shipped["source_sha256"], tmp_path / "jobs")
    assert dest == tmp_path / "jobs" / shipped["source_sha256"][:16]
    assert (dest / "src" / "pkg" / "__init__.py").read_text() == "VALUE = 1\n"
    assert not (dest / "untracked.txt").exists()

    (dest / "src" / "pkg" / "__init__.py").write_text("VALUE = 2\n")
    assert unpack_source(shipped["source_tar_b64"], shipped["source_sha256"], tmp_path / "jobs") == dest
    assert (dest / "src" / "pkg" / "__init__.py").read_text() == "VALUE = 2\n"  # complete copy is reused as is

    (dest / ".source_sha256").write_text("stale")
    unpack_source(shipped["source_tar_b64"], shipped["source_sha256"], tmp_path / "jobs")
    assert (dest / "src" / "pkg" / "__init__.py").read_text() == "VALUE = 1\n"  # incomplete copy is replaced


def test_unpack_source_rejects_a_wrong_digest(tmp_path: Path) -> None:
    from directions.remote import pack_source, unpack_source

    shipped = pack_source(_git_repo(tmp_path))
    with pytest.raises(ValueError, match="digest mismatch"):
        unpack_source(shipped["source_tar_b64"], "0" * 64, tmp_path / "jobs")
    assert not (tmp_path / "jobs").exists()


def test_offline_argv_builds_the_environment_from_the_wheelhouse(tmp_path: Path) -> None:
    from directions.remote import offline_argv

    wrapped = offline_argv(["uv", "run", "directions", "pilot", "--config", "configs/a b.yaml"], tmp_path / "wheels")
    assert wrapped[:4] == ["bash", "-euo", "pipefail", "-c"]
    script = wrapped[4]
    assert "uv venv -q --python 3.11 .venv" in script
    assert "uv export --frozen --no-hashes --no-emit-project --no-dev" in script
    assert f"--no-index --find-links {tmp_path / 'wheels'}" in script and " -r .wheelhouse-requirements.txt ." in script
    assert script.endswith("exec uv run --no-sync directions pilot --config 'configs/a b.yaml'")
    # anything that is not `uv run` is left alone (probes run with the image's python)
    assert offline_argv(["python", "scripts/probe_bandwidth.py"], tmp_path) == ["python", "scripts/probe_bandwidth.py"]
