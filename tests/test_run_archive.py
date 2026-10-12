"""scripts/run_archive.py: the run archive's manifest selection and the verified fetch."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("run_archive", Path(__file__).resolve().parents[1] / "scripts" / "run_archive.py")
run_archive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_archive)


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("landmarks_x/core/summary.json", json.dumps({"ok": True}))
    return buf.getvalue()


class FakeClient:
    def __init__(self, payload: bytes):
        self.payload = payload

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.payload)


def test_archived_selects_by_run_and_status():
    entries = [{"run_id": "1", "status": "archived"}, {"run_id": "1", "status": "already_archived"},
               {"run_id": "2", "artifact": None}, {"run_id": "3", "status": "expired"}, {"run_id": "4", "status": "failed"}]
    assert len(run_archive.archived(entries)) == 2
    assert len(run_archive.archived(entries, "run-1")) == 2
    assert run_archive.archived(entries, "2") == []


def test_fetch_verifies_the_digest_and_unpacks(tmp_path):
    payload = _zip_bytes()
    entry = {"run_id": "35682027963", "name": "results-x", "volume": "v", "key": "archive/x.zip",
             "sha256": hashlib.sha256(payload).hexdigest(), "datacenter": "US-IL-1"}
    dest = run_archive.fetch(entry, tmp_path, client=FakeClient(payload))
    assert dest == tmp_path / "run-35682027963"
    assert json.loads((dest / "landmarks_x/core/summary.json").read_text()) == {"ok": True}
    assert not (tmp_path / "results-x.zip").exists()


def test_fetch_refuses_a_corrupted_copy(tmp_path):
    entry = {"run_id": "1", "name": "results-y", "volume": "v", "key": "archive/y.zip", "sha256": "0" * 64,
             "datacenter": "US-IL-1"}
    with pytest.raises(ValueError, match="does not match"):
        run_archive.fetch(entry, tmp_path, client=FakeClient(_zip_bytes()))
    assert not (tmp_path / "results-y.zip").exists()


class FakeUploadClient:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[key] = Path(path).read_bytes()

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}


def test_add_verifies_uploads_and_appends(tmp_path):
    payload = _zip_bytes()
    src = tmp_path / "artifact.zip"
    src.write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    client = FakeUploadClient()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    e = run_archive.add("42", "7", "results-ci-42-abc", digest, len(payload), "2027-01-01T00:00:00Z", src.as_uri(),
                        client, "vol", tmp_path / "tmp", manifest)
    assert client.objects["archive/github-artifacts/results-ci-42-abc.zip"] == payload
    assert run_archive.archived(run_archive.load_manifest(manifest), "42") == [e]
    with pytest.raises(ValueError, match="already archived"):
        run_archive.add("42", "7", "results-ci-42-abc", digest, len(payload), "x", src.as_uri(), client, "vol", tmp_path / "tmp", manifest)


def test_add_refuses_a_digest_mismatch(tmp_path):
    src = tmp_path / "artifact.zip"
    src.write_bytes(_zip_bytes())
    client = FakeUploadClient()
    with pytest.raises(ValueError, match="does not match GitHub"):
        run_archive.add("43", "8", "results-ci-43", "sha256:" + "0" * 64, 1, "x", src.as_uri(), client, "vol", tmp_path / "tmp",
                        tmp_path / "manifest.jsonl")
    assert client.objects == {} and not (tmp_path / "manifest.jsonl").exists()
