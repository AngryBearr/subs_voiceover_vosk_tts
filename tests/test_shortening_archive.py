from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from utils.shortening_archive import ArchiveError, snapshot, verify_archive


def _offline_run(root: Path, *, status: str = "complete") -> Path:
    run = root / "run"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (run / "evidence.txt").write_bytes(b"lossless\x00evidence")
    return run


def test_roundtrip_verification_restores_bytes(tmp_path: Path) -> None:
    source = _offline_run(tmp_path)
    destination, manifest, verification = snapshot(source, tmp_path / "artifacts")
    assert manifest["member_count"] == 2
    assert verification["restored_to_fresh_temporary_directory"] is True
    assert verify_archive(destination)["status"] == "verified"


def test_corrupt_archive_and_member_fail(tmp_path: Path) -> None:
    source = _offline_run(tmp_path)
    destination, _, _ = snapshot(source, tmp_path / "artifacts")
    archive = destination / "run.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ArchiveError):
        verify_archive(destination)

    original = destination / "run.tar.gz"
    rebuilt = tmp_path / "rebuilt.tar.gz"
    with tarfile.open(original, "r:gz") as source_archive, tarfile.open(rebuilt, "w:gz") as target_archive:
        for member in source_archive.getmembers():
            body = b"changed" if member.name == "evidence.txt" else source_archive.extractfile(member).read()  # type: ignore[union-attr]
            member.size = len(body)
            target_archive.addfile(member, io.BytesIO(body))
    original.write_bytes(rebuilt.read_bytes())
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["archive_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest["archive_size"] = original.stat().st_size
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArchiveError, match="member bytes"):
        verify_archive(destination)


def test_symlink_and_path_traversal_are_refused(tmp_path: Path) -> None:
    source = _offline_run(tmp_path)
    (source / "link").symlink_to(source / "evidence.txt")
    with pytest.raises(ArchiveError, match="symlink"):
        snapshot(source, tmp_path / "artifacts")

    safe_source = _offline_run(tmp_path / "second")
    destination, _, _ = snapshot(safe_source, tmp_path / "artifacts2")
    malicious = tmp_path / "malicious"
    malicious.mkdir()
    with tarfile.open(malicious / "run.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["members"] = [{"path": "../escape", "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}]
    manifest["archive_sha256"] = hashlib.sha256((malicious / "run.tar.gz").read_bytes()).hexdigest()
    manifest["archive_size"] = (malicious / "run.tar.gz").stat().st_size
    (malicious / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArchiveError):
        verify_archive(malicious)


def test_existing_destination_input_unchanged_and_incomplete_matrix_refused(tmp_path: Path) -> None:
    source = _offline_run(tmp_path)
    before = {path: path.read_bytes() for path in source.iterdir()}
    artifacts = tmp_path / "artifacts"
    snapshot(source, artifacts)
    with pytest.raises(ArchiveError, match="overwrite"):
        snapshot(source, artifacts)
    assert {path: path.read_bytes() for path in source.iterdir()} == before

    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "report.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (matrix / "ledger.json").write_text(json.dumps({"requests": {"x": {"status": "unknown"}}}), encoding="utf-8")
    with pytest.raises(ArchiveError, match="pending or unknown"):
        snapshot(matrix, tmp_path / "matrix-artifacts")

    (matrix / "ledger.json").write_text(json.dumps({"requests": {"x": {"status": "complete"}}}), encoding="utf-8")
    from filelock import FileLock
    from utils.sh08_matrix import _lock_path
    with FileLock(str(_lock_path(matrix))):
        with pytest.raises(ArchiveError, match="actively locked"):
            snapshot(matrix, tmp_path / "locked-artifacts")


def test_secret_and_existing_symlink_rejected(tmp_path: Path) -> None:
    source = _offline_run(tmp_path)
    (source / ".env").write_text("SECRET=x", encoding="utf-8")
    with pytest.raises(ArchiveError, match="secret"):
        snapshot(source, tmp_path / "artifacts")
