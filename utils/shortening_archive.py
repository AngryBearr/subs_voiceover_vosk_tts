"""Safe, lossless snapshots for selected offline shortening evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Iterable, Iterator


MAX_BYTES = 500_000_000
ARCHIVE_PURPOSE = "snapshot-only SH07 implementation evidence; no migration, cleanup, or retention"
SECRET_NAMES = {".env", ".envrc", ".env.local", ".env.production", ".env.development"}
SECRET_PARTS = ("auth", "token", "secret")


class ArchiveError(ValueError):
    """Raised when a snapshot cannot be trusted or safely created."""


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    size: int
    sha256: str
    data: bytes
    stat_signature: tuple[int, int, int, int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ArchiveError(f"cannot stat {path}: {exc}") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _safe_member_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ArchiveError(f"path escapes source directory: {path}") from exc
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or any("\\" in part or "\x00" in part for part in parts):
        raise ArchiveError(f"unsafe source member path: {relative}")
    return "/".join(parts)


def _secret_name(path: str) -> bool:
    name = Path(path).name.casefold()
    return name in SECRET_NAMES or any(part in name for part in SECRET_PARTS)


def _regular_files(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir() or root.is_symlink():
        raise ArchiveError(f"source directory is missing or not a real directory: {root}")
    result: list[tuple[Path, str]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ArchiveError(f"cannot inspect {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            member = _safe_member_path(root, path)
            if _secret_name(member):
                raise ArchiveError(f"refusing secret-looking source member: {member}")
            try:
                if entry.is_symlink():
                    raise ArchiveError(f"symlink source member is refused: {member}")
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    result.append((path, member))
                else:
                    raise ArchiveError(f"special source member is refused: {member}")
            except OSError as exc:
                raise ArchiveError(f"cannot inspect {path}: {exc}") from exc

    visit(root)
    return result


def _read_files(root: Path) -> list[FileSnapshot]:
    files = _regular_files(root)
    total = 0
    snapshots: list[FileSnapshot] = []
    for path, member in files:
        before = _stat_signature(path)
        if before[2] + total > MAX_BYTES:
            raise ArchiveError(f"source exceeds {MAX_BYTES} byte safety cap")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArchiveError(f"cannot read {path}: {exc}") from exc
        after = _stat_signature(path)
        if before != after or len(data) != before[2]:
            raise ArchiveError(f"source changed while reading: {path}")
        digest = _sha_bytes(data)
        snapshots.append(FileSnapshot(member, len(data), digest, data, after))
        total += len(data)
    return snapshots


def _assert_sources_unchanged(root: Path, snapshots: Iterable[FileSnapshot]) -> None:
    expected = list(snapshots)
    current = {member for _, member in _regular_files(root)}
    if current != {item.path for item in expected}:
        raise ArchiveError(f"source member set changed during snapshot: {root}")
    for snapshot in expected:
        path = root.joinpath(*snapshot.path.split("/"))
        if _stat_signature(path) != snapshot.stat_signature or _sha_file(path) != snapshot.sha256:
            raise ArchiveError(f"source changed during snapshot: {path}")


def _members(snapshots: Iterable[FileSnapshot]) -> list[dict[str, Any]]:
    return [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in sorted(snapshots, key=lambda item: item.path)
    ]


def _content_hash(member_list: list[dict[str, Any]]) -> str:
    identity = [{"path": item["path"], "sha256": item["sha256"]} for item in member_list]
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha_bytes(encoded)


def _write_archive(path: Path, snapshots: list[FileSnapshot]) -> None:
    try:
        with tarfile.open(path, mode="w:gz") as archive:
            for item in sorted(snapshots, key=lambda value: value.path):
                info = tarfile.TarInfo(item.path)
                info.size = item.size
                info.mode = 0o600
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(item.data))
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot write archive {path}: {exc}") from exc


def _archive_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ArchiveError(f"cannot read archive {path}: {exc}") from exc
    return digest.hexdigest(), size


def _json_dump(path: Path, value: Any) -> None:
    try:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ArchiveError(f"cannot write {path}: {exc}") from exc


def _report_complete(source: Path) -> bool:
    candidates = [source / "report.json", source / "summary.json", source / "preflight_audio" / "audio_report.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArchiveError(f"cannot read offline report {path}: {exc}") from exc
        if isinstance(value, dict) and value.get("status") == "complete":
            return True
        if isinstance(value, dict) and value.get("schema") == "sh08_duplicate_analysis.v1":
            integrity = value.get("source_integrity")
            if isinstance(integrity, dict) and integrity:
                return True
    return False


def _check_matrix_ready(source: Path) -> None:
    report_path = source / "report.json"
    ledger_path = source / "ledger.json"
    if not report_path.is_file() or not ledger_path.is_file():
        raise ArchiveError(f"matrix evidence requires report.json and ledger.json: {source}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError(f"cannot read matrix evidence: {source}: {exc}") from exc
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ArchiveError(f"matrix report is not complete: {source}")
    requests = ledger.get("requests") if isinstance(ledger, dict) else None
    if not isinstance(requests, dict):
        raise ArchiveError(f"matrix ledger requests are malformed: {source}")
    statuses = [row.get("status") for row in requests.values() if isinstance(row, dict)]
    if len(statuses) != len(requests) or any(status != "complete" for status in statuses):
        raise ArchiveError(f"matrix ledger has pending or unknown requests: {source}")


@contextmanager
def _with_matrix_lock(source: Path) -> Iterator[None]:
    try:
        from filelock import FileLock
        from utils.sh08_matrix import _lock_path
    except ImportError as exc:
        raise ArchiveError(f"matrix lock support is unavailable: {exc}") from exc
    try:
        lock = FileLock(str(_lock_path(source)))
        lock.acquire(timeout=0)
    except Exception as exc:
        raise ArchiveError(f"matrix directory is actively locked: {source}") from exc
    try:
        yield
    finally:
        lock.release()


def _verify_contents(archive_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {}
    for item in manifest["members"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ArchiveError("archive manifest member is malformed")
        if isinstance(item.get("size"), bool) or not isinstance(item.get("size"), int) or item["size"] < 0:
            raise ArchiveError("archive manifest member size is malformed")
        if len(item["sha256"]) != 64:
            raise ArchiveError("archive manifest member hash is malformed")
        try:
            int(item["sha256"], 16)
        except ValueError as exc:
            raise ArchiveError("archive manifest member hash is malformed") from exc
        member_path = Path(item["path"])
        if member_path.is_absolute() or ".." in member_path.parts or any(part in {"", "."} or "\\" in part or "\x00" in part for part in member_path.parts):
            raise ArchiveError(f"unsafe manifest member path: {item['path']}")
        if item["path"] in expected:
            raise ArchiveError(f"duplicate manifest member path: {item['path']}")
        expected[item["path"]] = item
    try:
        actual_hash, actual_size = _archive_hash(archive_path)
        if actual_hash != manifest["archive_sha256"] or actual_size != manifest["archive_size"]:
            raise ArchiveError("archive hash or size does not match manifest")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ArchiveError("archive member names differ from manifest")
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts or any(part in {"", "."} or "\\" in part or "\x00" in part for part in member_path.parts) or not member.isfile():
                    raise ArchiveError(f"unsafe or non-regular archive member: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ArchiveError(f"archive member has no readable body: {member.name}")
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                claim = expected[member.name]
                if size != claim["size"] or digest.hexdigest() != claim["sha256"]:
                    raise ArchiveError(f"archive member bytes differ from manifest: {member.name}")
        with tempfile.TemporaryDirectory(prefix="shortening-archive-restore-") as temporary:
            restore_root = Path(temporary)
            with tarfile.open(archive_path, mode="r:gz") as archive:
                for member in archive.getmembers():
                    target = restore_root.joinpath(*member.name.split("/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ArchiveError(f"cannot restore archive member: {member.name}")
                    with target.open("xb") as output:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            output.write(chunk)
                    claim = expected[member.name]
                    if target.stat().st_size != claim["size"] or _sha_file(target) != claim["sha256"]:
                        raise ArchiveError(f"restored member differs from manifest: {member.name}")
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot verify archive {archive_path}: {exc}") from exc
    return {
        "schema": "shortening_archive_verification.v1",
        "verified_at": _now(),
        "archive_sha256": actual_hash,
        "archive_size": actual_size,
        "checked_members": len(expected),
        "checked_bytes": sum(item["size"] for item in expected.values()),
        "restored_to_fresh_temporary_directory": True,
        "restore_proof": "validated tar members were restored to a fresh temporary directory and byte-compared; temporary directory was removed automatically",
        "status": "verified",
    }


def verify_archive(archive_dir: Path) -> dict[str, Any]:
    archive_dir = Path(archive_dir).resolve()
    manifest_path = archive_dir / "manifest.json"
    archive_path = archive_dir / ("corpus.tar.gz" if (archive_dir / "corpus.tar.gz").is_file() else "run.tar.gz")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError(f"cannot read archive manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("members"), list):
        raise ArchiveError("archive manifest is malformed")
    return _verify_contents(archive_path, manifest)


def _snapshot_unlocked(source: Path, destination: Path, kind: str, *, snapshots: list[FileSnapshot] | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    snapshots = snapshots if snapshots is not None else _read_files(source)
    member_list = _members(snapshots)
    content_hash = _content_hash(member_list)
    created = _now()
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArchiveError(f"cannot create artifact parent {parent}: {exc}") from exc
    if destination.exists():
        raise ArchiveError(f"refusing to overwrite existing destination: {destination}")
    with tempfile.TemporaryDirectory(prefix="shortening-archive-build-", dir=str(parent)) as temporary:
        archive_path = Path(temporary) / ("corpus.tar.gz" if kind == "corpus" else "run.tar.gz")
        _write_archive(archive_path, snapshots)
        _assert_sources_unchanged(source, snapshots)
        archive_sha, archive_size = _archive_hash(archive_path)
        manifest: dict[str, Any] = {
            "schema": "shortening_archive_manifest.v1",
            "archive_kind": kind,
            "source_resolved_dir": str(source.resolve()),
            "created_at": created,
            "snapshot_only_purpose": ARCHIVE_PURPOSE,
            "member_count": len(member_list),
            "member_bytes": sum(item["size"] for item in member_list),
            "content_hash": content_hash,
            "members": member_list,
            "archive_sha256": archive_sha,
            "archive_size": archive_size,
        }
        verification = _verify_contents(archive_path, manifest)
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise ArchiveError(f"refusing to overwrite existing destination: {destination}") from exc
        except OSError as exc:
            raise ArchiveError(f"cannot create snapshot destination {destination}: {exc}") from exc
        try:
            os.replace(archive_path, destination / archive_path.name)
            _json_dump(destination / "manifest.json", manifest)
            _json_dump(destination / "verification.json", verification)
        except OSError as exc:
            raise ArchiveError(f"cannot publish snapshot {destination}: {exc}") from exc
    return destination, manifest, verification


def snapshot(source: Path, artifacts_root: Path, *, corpus: bool = False) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = Path(source)
    if source.is_symlink():
        raise ArchiveError(f"source directory symlink is refused: {source}")
    source = source.resolve()
    if corpus:
        snapshots = _read_files(source)
        destination = Path(artifacts_root).resolve() / "corpus" / _content_hash(_members(snapshots))
        return _snapshot_unlocked(source, destination, "corpus", snapshots=snapshots)
    if (source / "ledger.json").is_file():
        _check_matrix_ready(source)
        destination = Path(artifacts_root).resolve() / "results" / source.name
        with _with_matrix_lock(source):
            _check_matrix_ready(source)
            return _snapshot_unlocked(source, destination, "run")
    if not _report_complete(source):
        raise ArchiveError(f"offline evidence has no complete report: {source}")
    destination = Path(artifacts_root).resolve() / "results" / source.name
    return _snapshot_unlocked(source, destination, "run")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("--run-dir", action="append", type=Path, default=[])
    snapshot_parser.add_argument("--corpus-dir", type=Path)
    snapshot_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        try:
            result = verify_archive(args.archive_dir)
            path = Path(args.archive_dir) / "verification.json"
            if not path.exists():
                _json_dump(path, result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        except ArchiveError as exc:
            print(f"ERROR: {exc}")
            return 2
    if not args.run_dir and args.corpus_dir is None:
        print("ERROR: snapshot needs --run-dir or --corpus-dir")
        return 2
    failures = 0
    for run_dir in args.run_dir:
        try:
            destination, manifest, _ = snapshot(run_dir, args.artifacts_root)
            print(json.dumps({"archive_dir": str(destination), "members": manifest["member_count"], "bytes": manifest["member_bytes"]}, sort_keys=True))
        except ArchiveError as exc:
            failures += 1
            print(f"ERROR {run_dir}: {exc}")
    if args.corpus_dir is not None:
        try:
            destination, manifest, _ = snapshot(args.corpus_dir, args.artifacts_root, corpus=True)
            print(json.dumps({"archive_dir": str(destination), "members": manifest["member_count"], "bytes": manifest["member_bytes"]}, sort_keys=True))
        except ArchiveError as exc:
            failures += 1
            print(f"ERROR {args.corpus_dir}: {exc}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
