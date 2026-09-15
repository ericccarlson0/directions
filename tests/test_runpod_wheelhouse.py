"""scripts/runpod_wheelhouse.py: the locked wheels staged on a network volume."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("runpod_wheelhouse", Path(__file__).resolve().parents[1] / "scripts" / "runpod_wheelhouse.py")
wh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wh)


def test_wheel_files_and_platform_tags(tmp_path: Path) -> None:
    for name in ("torch-2.14.0-cp311-cp311-manylinux_2_28_x86_64.whl", "numpy-2.4.6-cp311-cp311-manylinux_2_27_x86_64.whl",
                 "pyyaml-6.0-py3-none-any.whl", "requirements.txt"):
        (tmp_path / name).write_text("x")
    files = wh.wheel_files(tmp_path)
    assert [p.name[:5] for p in files] == ["numpy", "pyyam", "torch"]
    assert wh.platform_tags(files) == {"manylinux_2_28_x86_64": 1, "manylinux_2_27_x86_64": 1, "any": 1}


def test_upload_skips_wheels_already_present(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "a-1.0-py3-none-any.whl").write_bytes(b"aaaa")
    (tmp_path / "b-1.0-py3-none-any.whl").write_bytes(b"bb")
    uploaded: list[str] = []

    class FakeClient:
        def get_paginator(self, name):
            class P:
                def paginate(self, Bucket, Prefix):
                    return [{"Contents": [{"Key": Prefix + "a-1.0-py3-none-any.whl", "Size": 4}]}]
            return P()

        def upload_file(self, path, bucket, key):
            uploaded.append(key)

    monkeypatch.setattr(wh, "_client", lambda dc, name: (FakeClient(), "vol"))
    assert wh.upload(tmp_path, "EUR-NO-1", "directions") == 1
    assert uploaded == ["wheels/b-1.0-py3-none-any.whl"]
    assert "1 uploaded, 1 already present" in capsys.readouterr().out
