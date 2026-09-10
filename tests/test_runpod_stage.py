import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "runpod_stage", Path(__file__).resolve().parent.parent / "scripts" / "runpod_stage.py"
)
runpod_stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_stage)

REPO = Path(__file__).resolve().parent.parent


def _root(tmp_path: Path, with_optional: bool = True) -> Path:
    root = tmp_path / "root"
    (root / "experiments").mkdir(parents=True)
    (root / "src" / "directions").mkdir(parents=True)
    (root / "experiments" / "remote.py").write_text("handler\n")
    (root / "src" / "directions" / "remote.py").write_text("helpers\n")
    if with_optional:
        (root / "src" / "directions" / "__init__.py").write_text("")
    (root / "src" / "directions" / "pipeline.py").write_text("experiment code, never staged\n")
    return root


def test_stage_copies_only_the_handler_files(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dest = tmp_path / "app"
    (dest / "stale").mkdir(parents=True)  # a previous staging is replaced

    files = runpod_stage.stage(root, dest)

    assert sorted(map(str, files)) == ["experiments/remote.py", "src/directions/__init__.py", "src/directions/remote.py"]
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == sorted(map(str, files))
    assert not (dest / "stale").exists()


def test_digest_tracks_handler_content_and_params_but_not_experiment_code(tmp_path: Path) -> None:
    root = _root(tmp_path)
    base = runpod_stage.handler_digest(root, {"gpu_tier": "ADA_24"})
    assert base == runpod_stage.handler_digest(root, {"gpu_tier": "ADA_24"})

    (root / "src" / "directions" / "pipeline.py").write_text("changed experiment code\n")
    assert runpod_stage.handler_digest(root, {"gpu_tier": "ADA_24"}) == base

    assert runpod_stage.handler_digest(root, {"gpu_tier": "AMPERE_24"}) != base
    (root / "experiments" / "remote.py").write_text("handler v2\n")
    assert runpod_stage.handler_digest(root, {"gpu_tier": "ADA_24"}) != base


def test_optional_files_are_staged_only_when_present(tmp_path: Path) -> None:
    root = _root(tmp_path, with_optional=False)
    assert sorted(map(str, runpod_stage.staged_files(root))) == ["experiments/remote.py", "src/directions/remote.py"]


def test_missing_handler_is_an_error(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "experiments" / "remote.py").unlink()
    with pytest.raises(FileNotFoundError, match="experiments/remote.py"):
        runpod_stage.staged_files(root)
    assert runpod_stage.main(["--root", str(root), "--dest", str(tmp_path / "app")]) == 2


def test_main_stages_and_prints_the_digest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _root(tmp_path)
    dest = tmp_path / "app"
    assert runpod_stage.main(["--root", str(root), "--dest", str(dest), "--param", "gpu_tier=ADA_24", "--param", "x=1"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == runpod_stage.handler_digest(root, {"gpu_tier": "ADA_24", "x": "1"})
    assert (dest / "experiments" / "remote.py").exists()
    assert runpod_stage.main(["--root", str(root), "--dest", str(dest), "--param", "novalue"]) == 2


def test_repository_handler_stages_and_imports_without_the_pipeline(tmp_path: Path) -> None:
    """The staged app must import with only the standard library (plus runpod_flash on the runner)."""
    import ast

    files = runpod_stage.stage(REPO, tmp_path / "app")
    assert Path("experiments/remote.py") in files and Path("src/directions/remote.py") in files
    for rel in files:
        tree = ast.parse((REPO / rel).read_text())
        for node in tree.body:  # module-level imports only; guarded imports inside functions are optional
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                assert top in {"runpod_flash", "directions"} or top in _STDLIB, f"{rel} imports {name}"


_STDLIB = {
    "__future__", "argparse", "base64", "gzip", "hashlib", "io", "json", "os", "platform", "shutil",
    "subprocess", "sys", "tarfile", "time", "pathlib", "typing", "urllib", "re",
}
