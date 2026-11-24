"""Configuration overlays and diagnostics stay pure and keep the existing pipeline defaults."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from stt import config, run, subtitle_edit

ROOT = Path(__file__).resolve().parents[1]


def test_layers_and_leaf_provenance() -> None:
    layers = [
        ("base", {"a": {"b": 1, "c": 2}, "items": [1], "null": None}),
        ("local", {"a": {"b": 3}}),
        ("show:race", {"a": {"d": 4}}),
        ("season", {"a": {"c": 5}}),
        ("preset:fast", {"items": [2, 3]}),
        ("cli", {"a": {"b": 6}}),
    ]
    before = copy.deepcopy(layers)
    result, sources = config.resolve_config_layers(layers)
    assert result == {"a": {"b": 6, "c": 5, "d": 4}, "items": [2, 3], "null": None}
    assert sources == {"a.b": "cli", "a.c": "season", "a.d": "show:race",
                       "items": "preset:fast", "null": "base"}
    result["items"].append(4)
    result["a"]["b"] = 10
    assert layers == before


@pytest.mark.parametrize(
    "base,override,expected,paths",
    [
        ({"a": {"b": 1}}, {"a": 2}, {"a": 2}, {"a": "override"}),
        ({"a": 1}, {"a": {"b": 2}}, {"a": {"b": 2}}, {"a.b": "override"}),
        ({"a": {"b": {"c": 1}}}, {"a": {"b": None}}, {"a": {"b": None}},
         {"a.b": "override"}),
        ({"a": 1}, {"a": {}}, {"a": {}}, {"a": "override"}),
        ({"a": {"b": 1}}, {"a": {}}, {"a": {"b": 1}}, {"a.b": "base"}),
        ({"a": []}, {"a": {}}, {"a": {}}, {"a": "override"}),
        ({"a": {}}, {"a": []}, {"a": []}, {"a": "override"}),
    ],
)
def test_shape_changes_remove_stale_sources(
    base: dict[str, Any], override: dict[str, Any], expected: dict[str, Any], paths: dict[str, str],
) -> None:
    assert config.resolve_config_layers([("base", base), ("override", override)]) == (expected, paths)


def test_empty_and_equal_layers() -> None:
    assert config.resolve_config_layers([]) == ({}, {})
    assert config.resolve_config_layers([("base", {"a": 1}), ("local", {"a": 1}), ("cli", {})]) == (
        {"a": 1}, {"a": "local"})
    result, sources = config.resolve_config_layers([
        ("base", {"a": {"b": 1}}), ("local", {"a": None}), ("override", {"a": {"c": 2}}),
    ])
    assert result == {"a": {"c": 2}} and sources == {"a.c": "override"}


def test_deep_merge_has_no_mutable_aliases() -> None:
    base = {"a": {"b": [1]}, "keep": []}
    override = {"a": {"c": [2]}}
    result = config.deep_merge(base, override)
    result["a"]["b"].append(3)
    result["a"]["c"].append(4)
    result["keep"].append(5)
    assert base == {"a": {"b": [1]}, "keep": []}
    assert override == {"a": {"c": [2]}}


def test_load_config_and_legacy_reexports(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"a": {"b": 1, "c": 2}}', encoding="utf-8")
    assert config.load_config(path) == {"a": {"b": 1, "c": 2}}
    path.with_name("settings.local.json").write_text('{"a": {"b": 3}}', encoding="utf-8")
    assert config.load_config(path) == {"a": {"b": 3, "c": 2}}
    for name in ("load_config", "apply_preset", "apply_show", "show_of", "deep_merge"):
        assert getattr(run, name) is getattr(config, name)
    path.with_name("settings.local.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_config(path)


def test_survivor_defaults_are_unchanged() -> None:
    base = json.loads((ROOT / "stt_config.json").read_text(encoding="utf-8"))
    resolved, sources = config.resolve_config_layers([("base", base)])
    assert config.show_of(resolved, Path("/Survivor/Season_20/S20E01.mkv")) is None
    assert config.apply_show(resolved, None) is resolved
    assert config.apply_preset(resolved, "main") == base
    assert resolved["season_voices"] == {
        "enabled": True, "model": "resnet293-ft", "min_words": 20, "dissolve": "mixed", "guests": "home",
    }
    assert resolved["voice_bank"] == {"enabled": True, "identities": False, "stitch": False}
    assert resolved["voiceover"]["engine"] == "f5,teratts,vosk"
    assert resolved["diarizer_options"]["ahc_threshold"] == 0.5
    assert sources["voice_bank.identities"] == "base"


def test_voiceover_policy_precedence_and_purity() -> None:
    stt = {"voiceover": {"policy": {"base_syl_per_sec": 5.0}}}
    vo = {"policy": {"base_syl_per_sec": 4.0, "comfort_syl_per_sec": 6.3, "ceiling_syl_per_sec": 6.7}}
    policy = config.effective_voiceover_policy(stt, vo)
    assert policy == {"base_syl_per_sec": 5.0, "comfort_syl_per_sec": 6.3, "ceiling_syl_per_sec": 6.7}
    policy["base_syl_per_sec"] = 1.0
    assert stt["voiceover"]["policy"]["base_syl_per_sec"] == 5.0
    assert vo["policy"]["base_syl_per_sec"] == 4.0
    assert config.effective_voiceover_policy({}, {}) == {}
    assert config.effective_voiceover_policy({"voiceover": None}, {"policy": None}) == {}


def test_subtitle_editor_uses_the_shared_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "stt_config.json").write_text(
        '{"voiceover": {"policy": {"base_syl_per_sec": 5.2}}}', encoding="utf-8")
    (tmp_path / "voiceover_config.json").write_text(
        '{"policy": {"comfort_syl_per_sec": 6.1}}', encoding="utf-8")
    monkeypatch.setattr(subtitle_edit, "REPO_ROOT", tmp_path)
    assert subtitle_edit._policy() == {
        "base_syl_per_sec": 5.2, "comfort_syl_per_sec": 6.1, "ceiling_syl_per_sec": 6.7,
    }


def test_episode_voiceover_policy_matches_shared_helper(tmp_path: Path) -> None:
    stt = json.loads((ROOT / "stt_config.json").read_text(encoding="utf-8"))
    episode = run.Episode(Path("/Survivor/Season_20/S20E01.mkv"), tmp_path,
                          argparse.Namespace(season_glossary=None, force=None, engine="vosk"), stt)
    with patch.object(episode, "llm", return_value="model"):
        generated = json.loads(episode._voiceover_config().read_text(encoding="utf-8"))
    vo = json.loads((ROOT / "voiceover_config.json").read_text(encoding="utf-8"))
    assert generated["policy"] == config.effective_voiceover_policy(stt, vo)


def test_diagnostic_redacts_credentials_but_not_variable_names() -> None:
    raw = {"llm": {"api_key": "private", "api_key_env": "OPENROUTER_API_KEY", "max_tokens": 8192},
           "nested": [{"access_token": "private", "password": "private"}], "secret": "private"}
    text = config.format_config_sources(raw, {"llm.api_key": "local"})
    assert "private" not in text and "OPENROUTER_API_KEY" in text
    assert json.loads(text)["config"]["llm"]["max_tokens"] == 8192
    assert raw["secret"] == "private"


def test_dump_exits_before_models_media_validation_or_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "mode": "accurate", "api_key": "private",
        "presets": {"fast": {"settings": {"mode": "fast", "mix": {"duck_db": 10}}}},
        "shows": {"race": {"settings": {"mix": {"duck_db": 12}}}},
    }), encoding="utf-8")
    path.with_name("config.local.json").write_text('{"mix": {"duck_db": 9}}', encoding="utf-8")
    original_read = Path.read_text

    def read(own: Path, *args: Any, **kwargs: Any) -> str:
        assert own.name not in {".env", "auth.json"}
        return original_read(own, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(run, "run_all", lambda *a, **k: pytest.fail("no processing"))
    monkeypatch.setattr(run, "dotenv", lambda *a, **k: pytest.fail("no secrets"))
    monkeypatch.setattr(run, "export_llm_env", lambda *a, **k: pytest.fail("no environment changes"))
    run.main(["/missing/race/ep.mkv", "--config", str(path), "--dump-config", "--preset", "fast",
              "--mode", "accurate", "--engine", "vosk", "--out-root", "/work",
              "--no-subtitles", "--no-season-voices"])
    text = capsys.readouterr().out
    out = json.loads(text)
    assert "private" not in text
    assert out["config"]["mode"] == "accurate" and out["sources"]["mode"] == "cli"
    assert out["config"]["mix"]["duck_db"] == 12 and out["sources"]["mix.duck_db"] == "show:race"
    assert out["sources"]["voiceover.engine"] == "cli"
    assert out["config"]["paths"]["work"] == "/work"
    assert out["config"]["subtitles"]["enabled"] is False
    assert out["config"]["season_voices"]["enabled"] is False
    run.main(["--config", str(path), "--dump-config"])
    assert json.loads(capsys.readouterr().out)["sources"]["mix.duck_db"] == "local"
    run.main(["/missing/race/ep.mkv", "--config", str(path), "--dump-config",
              "--show", "none", "--preset", "fast"])
    assert json.loads(capsys.readouterr().out)["sources"]["mix.duck_db"] == "preset:fast"
    run.main(["--config", str(path), "--dump-config", "--show", "race"])
    assert json.loads(capsys.readouterr().out)["sources"]["mix.duck_db"] == "show:race"
    with pytest.raises(SystemExit):
        run.main(["--config", str(path)])
