"""Video files named on the stt.run command line."""

from __future__ import annotations

from pathlib import Path

from stt.run import expand_media


def test_file_with_glob_characters_in_its_name_is_taken_as_is(tmp_path: Path) -> None:
    video = tmp_path / "Survivor S51E07 The Girl at the Prom 1080p PMTP WEB-DL DDP5 1 H 264-RAWR[EZTVx.to].mkv"
    video.write_bytes(b"x")
    assert expand_media([str(video)]) == ([video], [])


def test_patterns_expand_and_unmatched_ones_are_reported(tmp_path: Path) -> None:
    a, b = tmp_path / "S51E01.mkv", tmp_path / "S51E02.mkv"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    files, missing = expand_media([str(tmp_path / "*.mkv"), str(tmp_path / "S99*.mkv"), str(tmp_path / "none.mkv")])
    assert files == [a, b]
    assert missing == [str(tmp_path / "S99*.mkv"), str(tmp_path / "none.mkv")]
