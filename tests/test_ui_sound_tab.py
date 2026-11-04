"""The tab "Звук и размер видео": every button calls its function with the values of the right fields, nothing is encoded here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from stt import mix as stt_mix
from stt import mix_preview, run, video_compress
from ui import dub_app, settings


def _clicks(app: Any) -> dict[str, list[Any]]:
    """Button label -> what its click runs."""
    found: dict[str, list[Any]] = {}
    for fn in app.fns.values():
        for block_id, event in fn.targets:
            block = app.blocks.get(block_id)
            if event == "click" and block is not None and getattr(block, "value", None):
                found.setdefault(str(block.value), []).append(fn)
    return found


def _labels(fn: Any) -> list[str]:
    return [str(getattr(block, "label", "")) for block in fn.inputs]


@pytest.fixture(scope="module")
def clicks() -> dict[str, list[Any]]:
    return _clicks(dub_app.build())


@pytest.mark.parametrize("button,handler", [("Найти самое громкое место", "find_spot"), ("Послушать", "listen_mix"),
                                            ("Сохранить настройки", "save_mix_settings"),
                                            ("Пересвести эту серию", "remix")])
def test_button_runs_its_function(clicks: dict[str, list[Any]], button: str, handler: str) -> None:
    assert [fn.fn.__name__ for fn in clicks[button]] == [handler]


def test_sliders_reach_the_listen_function_in_order(clicks: dict[str, list[Any]]) -> None:
    labels = _labels(clicks["Послушать"][0])
    assert labels[1:4] == ["Серия (с готовой озвучкой)", "Озвучка", "С минуты"]
    assert labels[4:] == ["Голоса громче, дБ", "Музыка и шум под голосом тише, дБ", "Голоса оригинала под озвучкой тише, дБ",
                          "Общая громкость, LUFS"]


def test_saved_fields_go_to_their_settings(clicks: dict[str, list[Any]]) -> None:
    labels = _labels(clicks["Сохранить настройки"][0])
    paths = [s.path for s in dub_app.MIX_SETTINGS]
    assert list(zip(labels, paths)) == [
        ("Голоса громче, дБ", ("mix", "voice_gain_db")),
        ("Музыка и шум под голосом тише, дБ", ("mix", "rest_duck_db")),
        ("Голоса оригинала под озвучкой тише, дБ", ("mix", "duck_db")),
        ("Общая громкость, LUFS", ("mix", "loudness_lufs")),
        ("Сжимать видео при сборке серии", ("mix", "video", "compress")),
        ("Размер серии, ГБ", ("mix", "video", "target_gb")),
        ("Кодек", ("mix", "video", "codec")),
    ]


def test_saving_in_one_tab_refreshes_the_other() -> None:
    app = dub_app.build()
    by_id = dict(app.fns)
    followers: dict[str, list[list[str]]] = {}
    for fn in app.fns.values():
        after = getattr(fn, "trigger_after", None)
        if after is not None and after in by_id:
            followers.setdefault(by_id[after].fn.__name__, []).append([str(getattr(b, "label", "")) for b in fn.outputs])
    assert any("Голоса громче, дБ" in outputs for outputs in followers["save_settings"])
    assert any("Голоса громче, дБ" in outputs for outputs in followers["reset_settings"])
    assert any("Озвучка громче, дБ" in outputs for outputs in followers["save_mix_settings"])


def test_listen_passes_each_slider_to_its_parameter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "ep"
    folder.mkdir()
    (folder / "source.json").write_text(json.dumps({"media": "/video/ep.mkv"}), encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake(episode_dir: Path, media: Path, engine: str, start: float, seconds: float, gain: float, duck_db: float, rest: float,
             lufs: float, out: Path, **_: Any) -> Path:
        seen.update(episode_dir=episode_dir, media=media, engine=engine, start=start, gain=gain, duck=duck_db, rest=rest, lufs=lufs)
        return out

    monkeypatch.setattr(mix_preview, "render_preview", fake)
    dub_app.listen_mix(str(tmp_path), "ep", "ESpeech (голоса героев)", 2.5, 7, 9, 15, -18)
    assert seen == {"episode_dir": folder, "media": Path("/video/ep.mkv"), "engine": "f5", "start": 150.0, "gain": 7.0, "duck": 15.0,
                    "rest": 9.0, "lufs": -18.0}


def test_remix_asks_for_the_mix_stage_of_the_voiced_engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "ep"
    for engine in ("f5", "vosk"):
        (folder / f"voiceover.{engine}" / "export").mkdir(parents=True)
        (folder / f"voiceover.{engine}" / "export" / "track.wav").touch()
    (folder / "source.json").write_text(json.dumps({"media": "/video/season/ep.mkv"}), encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake(folder_: str, names: list[str], out_root: str, engines: list[str], *_: Any, stt_args: list[str] | None = None) -> str:
        seen.update(folder=folder_, names=names, out_root=out_root, engines=engines, stt_args=stt_args)
        return "ok"

    monkeypatch.setattr(dub_app, "start_job", fake)
    assert dub_app.remix(str(tmp_path), "ep") == "ok"
    assert seen == {"folder": "/video/season", "names": ["ep.mkv"], "out_root": str(tmp_path),
                    "engines": ["Vosk", "ESpeech (голоса героев)"], "stt_args": ["--force", "mix"]}


@pytest.mark.parametrize("sample,extra,suffix", [(False, [], "small"), (True, ["--sample-minute", "10", "--sample-seconds", "60"], "sample")])
def test_compress_buttons_start_the_right_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample: bool, extra: list[str],
                                                  suffix: str) -> None:
    video = tmp_path / "ep.ru.mkv"
    video.touch()
    seen: dict[str, Any] = {}

    class Process:
        def poll(self) -> int | None:
            return None

    def fake(cmd: list[str], **_: Any) -> Process:
        seen["cmd"] = cmd
        return Process()

    monkeypatch.setattr(dub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dub_app.subprocess, "Popen", fake)
    monkeypatch.setitem(dub_app.COMPRESS, "process", None)
    message = dub_app.start_compress(str(video), sample, 2.5, list(settings.VIDEO_CODECS)[1])
    assert seen["cmd"][1:] == ["-m", "stt.video_compress", str(video), "--target-gb", "2.5", "--codec", "h264_nvenc", *extra]
    assert f"ep.ru.{suffix}.mkv" in message


def _episode(tmp_path: Path, video: dict[str, Any]) -> run.Episode:
    media = tmp_path / "season" / "ep.mkv"
    media.parent.mkdir()
    media.touch()
    config = {"mode": "fast", "mix": {"tracks": "combined", "track_order": ["f5", "vosk"], "subtitles": False, "video": video},
              "voiceover": {"engines": {"f5": {}, "vosk": {}}, "engine": "f5"}, "paths": {}}
    args = argparse.Namespace(season_glossary=None, engine="f5", mode="fast", start=None, duration=None, force=None, until="mix",
                              planned_engines=["f5", "vosk"])
    episode = run.Episode(media, tmp_path / "work", args, config)
    episode.out.mkdir(parents=True)
    episode.track("f5").touch()
    return episode


def test_mux_uses_the_small_video_when_compression_is_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path, {"compress": True, "target_gb": 2.0, "codec": "hevc_nvenc", "original_audio_kbps": 160})
    encoded: dict[str, Any] = {}
    muxed: dict[str, Any] = {}

    def fake_encode(src: Path, out: Path, bitrate: int, codec: str, overlays: Any = None) -> Path:
        encoded.update(src=src, name=out.name, bitrate=bitrate, codec=codec)
        out.touch()
        return out

    def fake_mux(media: Path, tracks: list[Any], out: Path, srt: Any, start: float, duration: Any, **kwargs: Any) -> None:
        muxed.update(kwargs)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()

    monkeypatch.setattr(video_compress, "probe", lambda _: {"duration": 3600.0, "video_kbps": 9000.0})
    monkeypatch.setattr(video_compress, "encode_video", fake_encode)
    monkeypatch.setattr(stt_mix, "mux_tracks", fake_mux)
    episode.mux()
    # 2 GB over an hour, 2% kept for the container, two voice-over tracks at 192 kbps and the original at 160
    assert encoded == {"src": episode.media, "name": "video.hevc_nvenc.3811k.mkv", "bitrate": 3811, "codec": "hevc_nvenc"}
    assert muxed == {"video_source": episode.out / "video.hevc_nvenc.3811k.mkv", "original_audio_kbps": 160,
                     "onscreen_ass": None, "fonts": []}


def test_mux_copies_the_video_when_compression_is_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path, {"compress": False})
    muxed: dict[str, Any] = {}

    def fake_mux(media: Path, tracks: list[Any], out: Path, srt: Any, start: float, duration: Any, **kwargs: Any) -> None:
        muxed.update(kwargs)

    def never(*_: Any, **__: Any) -> None:
        raise AssertionError("no encoding when compression is off")

    monkeypatch.setattr(video_compress, "encode_video", never)
    monkeypatch.setattr(stt_mix, "mux_tracks", fake_mux)
    episode.mux()
    assert muxed == {"video_source": None, "original_audio_kbps": None, "onscreen_ass": None, "fonts": []}


ONSCREEN_ITEM = {"id": 0, "kind": "caption", "start": 1.0, "end": 2.0, "text": "There's a boat.", "descriptor": None,
                 "box": [0.38, 0.82, 0.62, 0.9], "voiced": True, "text_ru": "Там лодка."}


def _with_onscreen(episode: Any) -> None:
    (episode.out / "onscreen.json").write_text(json.dumps([{k: v for k, v in ONSCREEN_ITEM.items() if k != "text_ru"}]),
                                               encoding="utf-8")
    (episode.out / "onscreen.ru.json").write_text(json.dumps([ONSCREEN_ITEM], ensure_ascii=False), encoding="utf-8")


def test_mux_adds_the_onscreen_track_when_the_video_is_copied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path, {"compress": False})
    _with_onscreen(episode)
    muxed: dict[str, Any] = {}

    def fake_mux(media: Path, tracks: list[Any], out: Path, srt: Any, start: float, duration: Any, **kwargs: Any) -> None:
        muxed.update(kwargs)

    monkeypatch.setattr(video_compress, "probe", lambda _: {"duration": 60.0, "video_kbps": 9000.0, "width": 1920,
                                                            "height": 1080})
    monkeypatch.setattr(stt_mix, "mux_tracks", fake_mux)
    episode.mux()
    assert muxed["onscreen_ass"] == episode.out / "onscreen.ass"
    assert "Там лодка." in muxed["onscreen_ass"].read_text(encoding="utf-8")


def test_mux_burns_the_onscreen_text_into_the_compressed_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path, {"compress": True, "target_gb": 2.0, "codec": "hevc_nvenc", "original_audio_kbps": 160})
    _with_onscreen(episode)
    encoded: dict[str, Any] = {}
    muxed: dict[str, Any] = {}

    def fake_encode(src: Path, out: Path, bitrate: int, codec: str, overlays: Any = None) -> Path:
        encoded.update(name=out.name, overlays=overlays)
        out.touch()
        return out

    def fake_mux(media: Path, tracks: list[Any], out: Path, srt: Any, start: float, duration: Any, **kwargs: Any) -> None:
        muxed.update(kwargs)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()

    monkeypatch.setattr(video_compress, "probe", lambda _: {"duration": 3600.0, "video_kbps": 9000.0, "width": 1920,
                                                            "height": 1080})
    monkeypatch.setattr(video_compress, "encode_video", fake_encode)
    monkeypatch.setattr(stt_mix, "mux_tracks", fake_mux)
    episode.mux()
    assert encoded["name"] == "video.hevc_nvenc.3811k.onscreen.mkv" and len(encoded["overlays"]) == 1
    assert muxed["video_source"] == episode.out / "video.hevc_nvenc.3811k.onscreen.mkv"
    assert muxed["onscreen_ass"] is None  # burned in: no second copy as a track
