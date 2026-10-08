from __future__ import annotations

from utils.length_check import check_length as pure_check_length
from utils.length_mcp import check_length


def test_mcp_length_basis_matches_pure_checker() -> None:
    text = "Кавычки «и»…"
    expected = pure_check_length(text, 12, "normalized_tts_text")
    assert check_length(text, 12, "normalized_tts_text") == expected


def test_mcp_default_basis_matches_pure_checker() -> None:
    text = "a  b"
    assert check_length(text, 3) == pure_check_length(text, 3)
