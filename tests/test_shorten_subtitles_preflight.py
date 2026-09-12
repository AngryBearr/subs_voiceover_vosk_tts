from __future__ import annotations

import json
from pathlib import Path

from utils.shorten_subtitles import _parser, main


def test_cli_preflight_options_default_to_planner_speed_and_are_opt_in(tmp_path: Path) -> None:
    parser = _parser()
    args = parser.parse_args(["source.json", "--output-dir", "out"])
    assert args.max_speed == 1.5
    assert args.preflight_model_path is None
    opted = parser.parse_args([
        "source.json", "--output-dir", "out", "--max-speed", "1.3",
        "--preflight-model-path", "model", "--preflight-voice", "0", "--preflight-text-mode", "raw",
    ])
    assert opted.max_speed == 1.3
    assert opted.preflight_model_path == Path("model")
    assert opted.preflight_voice == 0
    assert opted.preflight_text_mode == "raw"


def test_cli_invalid_preflight_model_fails_before_backend(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([{"index": 1, "start": 0, "end": 100, "text": "long phrase"}]),
        encoding="utf-8",
    )

    class BackendMustNotStart:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("backend must not start for invalid preflight model")

    monkeypatch.setattr("utils.shorten_subtitles.CleanOpenCodeServer", BackendMustNotStart)
    result = main([
        str(source), "--output-dir", str(tmp_path / "run"), "--execute",
        "--generator", "opencode:none", "--repetitions", "1",
        "--short-count", "1",
        "--preflight-model-path", str(tmp_path / "missing-model"),
    ])
    assert result == 1
