"""Faster episodes without other output: one voice separation, one combined mux, mixes beside the next engine's
voice-over, on-screen translation beside the dub. Fakes only: no audio models, no ffmpeg, no LLM."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt import mix as stt_mix
from stt import onscreen, run

RATE = 44100  # the separator's own rate in the fakes: no resampling


class FakeSeparator:
    """Stands in for Hybrid Demucs: the 'voices' are half of the signal."""

    created = 0
    rate = RATE

    def __init__(self, *_: Any, **__: Any) -> None:
        type(self).created += 1

    def separate_long(self, stereo: np.ndarray) -> np.ndarray:
        return (stereo * 0.5).astype(np.float32)


@pytest.fixture
def separator(monkeypatch: pytest.MonkeyPatch) -> type[FakeSeparator]:
    import stt.separate

    FakeSeparator.created = 0
    monkeypatch.setattr(stt.separate, "VocalSeparator", FakeSeparator)
    return FakeSeparator


# ---- 1. one separation per episode

def test_separation_is_computed_once_for_all_engines_even_at_the_same_time(
        separator: type[FakeSeparator], tmp_path: Path) -> None:
    original = np.random.default_rng(1).uniform(-0.5, 0.5, (RATE, 2)).astype(np.float32)
    cache = tmp_path / "separation" / "voices.test.npy"
    results: list[np.ndarray] = []

    def engine() -> None:
        results.append(np.array(stt_mix.separated_voices(original, RATE, cache)))

    threads = [threading.Thread(target=engine) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert separator.created == 1
    assert len(results) == 3 and all(np.array_equal(r, original * np.float32(0.5)) for r in results)
    assert [p.name for p in cache.parent.iterdir()] == [cache.name]  # no partial files left
    cache.write_bytes(b"broken")  # a damaged file is separated again, not trusted
    assert np.array_equal(stt_mix.separated_voices(original, RATE, cache), original * np.float32(0.5))
    assert separator.created == 2


def test_separation_key_follows_media_interval_rate_and_version(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"a")
    base = stt_mix.separation_path(tmp_path, media, 0.0, None, 48000)
    assert base == stt_mix.separation_path(tmp_path, media, 0.0, None, 48000)
    assert base != stt_mix.separation_path(tmp_path, media, 10.0, None, 48000)
    assert base != stt_mix.separation_path(tmp_path, media, 0.0, 60.0, 48000)
    assert base != stt_mix.separation_path(tmp_path, media, 0.0, None, 44100)
    monkeypatch.setattr(stt_mix, "SEPARATION_VERSION", stt_mix.SEPARATION_VERSION + 1)
    assert base != stt_mix.separation_path(tmp_path, media, 0.0, None, 48000)
    monkeypatch.undo()
    media.write_bytes(b"changed media")
    assert base != stt_mix.separation_path(tmp_path, media, 0.0, None, 48000)


def test_mix_main_with_cache_gives_the_same_audio_and_separates_once(
        separator: type[FakeSeparator], tmp_path: Path) -> None:
    media, voice = tmp_path / "ep.wav", tmp_path / "voice.wav"
    t = np.arange(RATE * 2) / RATE
    sf.write(str(media), np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32), RATE)
    sf.write(str(voice), (0.2 * np.sin(2 * np.pi * 440 * t[: RATE])).astype(np.float32), RATE)
    common = [str(media), str(voice), "--duck-voices-only", "--sample-rate", str(RATE), "--full-duration"]
    stt_mix.main([*common, "--out", str(tmp_path / "plain.wav")])  # no cache: as before
    for name in ("a", "b", "c"):  # three engines
        stt_mix.main([*common, "--out", str(tmp_path / f"{name}.wav"), "--separation-cache", str(tmp_path / "sep")])
    assert separator.created == 1 + 1
    plain, _ = sf.read(str(tmp_path / "plain.wav"))
    for name in ("a", "b", "c"):
        cached, _ = sf.read(str(tmp_path / f"{name}.wav"))
        assert np.array_equal(plain, cached)


# ---- 2. the combined video is muxed once

def _episode(tmp_path: Path, *, tracks: str = "combined", planned: list[str] | None = None,
             engine: str = "f5") -> run.Episode:
    media = tmp_path / "S" / "ep01.mkv"
    config = {"mix": {"tracks": tracks, "track_order": ["f5", "teratts", "vosk"], "mux": True, "duck_db": 12,
                      "subtitles": False},
              "voiceover": {"engines": {"f5": {}, "teratts": {}, "vosk": {}}}}
    args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine=engine, start=None, duration=None,
                              planned_engines=planned if planned is not None else ["f5", "teratts", "vosk"])
    episode = run.Episode(media, tmp_path / "out", args, config, lock=threading.Lock())
    episode.out.mkdir(parents=True, exist_ok=True)
    return episode


def _fake_mix_parts(monkeypatch: pytest.MonkeyPatch, muxed: list[list[str]]) -> None:
    monkeypatch.setattr(stt_mix, "main", lambda argv: Path(argv[argv.index("--out") + 1]).touch())
    monkeypatch.setattr(stt_mix, "encode_track", lambda wav, out, *a, **k: out.touch())

    def fake_mux(self: run.Episode) -> None:
        muxed.append([e for e in ("f5", "teratts", "vosk") if self.track(e).exists()])
        self.finished_video().parent.mkdir(parents=True, exist_ok=True)
        self.finished_video().touch()

    monkeypatch.setattr(run.Episode, "mux", fake_mux)


def test_combined_mux_waits_for_every_planned_engine_and_happens_once(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    muxed: list[list[str]] = []
    _fake_mix_parts(monkeypatch, muxed)
    caplog.set_level(logging.INFO, logger="stt.run")
    for engine in ("f5", "teratts"):
        _episode(tmp_path, engine=engine).mix()
        assert muxed == []
    assert "mux deferred until vosk finish" in caplog.text
    _episode(tmp_path, engine="vosk").mix()
    assert muxed == [["f5", "teratts", "vosk"]]
    _episode(tmp_path, engine="vosk").mux_when_ready()  # the same tracks again: nothing to redo
    assert len(muxed) == 1


def test_two_lanes_finishing_together_mux_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    muxed: list[list[str]] = []
    _fake_mix_parts(monkeypatch, muxed)
    lock = threading.Lock()
    episodes = [_episode(tmp_path, planned=["f5", "vosk"], engine=e) for e in ("f5", "vosk")]
    for ep in episodes:
        ep.lock = lock
        ep.track(ep.engine).touch()  # both tracks are there when each lane gets to the mux
    threads = [threading.Thread(target=ep.mux_when_ready) for ep in episodes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert muxed == [["f5", "vosk"]]


def test_mux_per_engine_in_separate_mode_single_engine_runs_and_reruns(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    muxed: list[list[str]] = []
    _fake_mix_parts(monkeypatch, muxed)
    _episode(tmp_path / "sep", tracks="separate", engine="f5").mix()
    _episode(tmp_path / "sep", tracks="separate", engine="vosk").mix()
    assert len(muxed) == 2  # a video per engine, as before
    _episode(tmp_path / "one", planned=["vosk"], engine="vosk").mix()
    assert muxed[2:] == [["vosk"]]  # a single-engine run muxes at once
    _episode(tmp_path / "rerun", planned=["f5", "teratts"], engine="f5").track("teratts").touch()
    _episode(tmp_path / "rerun", planned=["f5", "teratts"], engine="f5").mix()
    assert muxed[3:] == [["f5", "teratts"]]  # the other engine's track is from an earlier run: the full video now


# ---- 3. the next engine goes on while the previous mix runs

def test_next_engine_voices_while_the_previous_mix_runs_and_the_lane_joins_it(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    next_started = threading.Event()

    class FakeEpisode:
        def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict, lock: Any = None) -> None:
            self.args = args

        def run(self) -> None:
            engine, background = self.args.engine, self.args.mix_background
            events.append(f"{engine} voiceover")
            if engine == "teratts":
                next_started.set()

            def mix() -> None:
                if engine == "f5":
                    assert next_started.wait(5), "the next engine did not start while the mix ran"
                events.append(f"{engine} mix end")

            background.start(f"mix-{engine}", mix)

    monkeypatch.setattr(run, "Episode", FakeEpisode)
    args = argparse.Namespace(engine="f5,teratts", until="mix")
    run.run_lanes([tmp_path / "e1.mkv"], tmp_path, args, {}, ["f5", "teratts"])
    assert events.index("teratts voiceover") < events.index("f5 mix end")
    assert sorted(events[-2:]) == ["f5 mix end", "teratts mix end"]  # all mixes ended before run_lanes returned


def test_background_mix_error_reaches_the_caller_and_stops_the_lane(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started: list[str] = []

    class FakeEpisode:
        def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict, lock: Any = None) -> None:
            self.args = args

        def run(self) -> None:
            started.append(self.args.engine)

            def boom() -> None:
                raise RuntimeError("mix exploded")

            self.args.mix_background.start(f"mix-{self.args.engine}", boom)
            self.args.mix_background._threads[-1].join()

    monkeypatch.setattr(run, "Episode", FakeEpisode)
    with pytest.raises(RuntimeError, match="mix exploded"):
        run.run_lanes([tmp_path / "e1.mkv"], tmp_path, argparse.Namespace(engine="f5", until="mix"), {},
                      ["f5", "teratts"])
    assert started == ["f5"]  # the second engine was not started after the failure


def test_episode_run_in_background_keeps_stage_marks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path, engine="f5")
    background = run.Background()
    episode.args.mix_background = background
    release, mix_started = threading.Event(), threading.Event()
    order: list[str] = []

    def stage(name: str, *outputs: Path) -> Any:
        def work(self: run.Episode) -> None:
            order.append(name)
            if name == "mix":
                mix_started.set()
                assert release.wait(5)
            for output in outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
        return work

    out, vo = episode.out, episode.vo_dir
    for name, outputs in {"transcribe": [out / "subs.en.json"], "glossary": [out / "glossary.json"],
                          "speakers": [out / "speakers.llm.json"], "translate": [out / "subs.ru.json"],
                          "dub": [out / "dub.ru.f5.json"], "stress": [out / "stress.json"],
                          "voiceover": [vo / "export" / "track.wav"], "check": [vo / "speech_check.json"],
                          "mix": [episode.track("f5")]}.items():
        monkeypatch.setattr(run.Episode, name, stage(name, *outputs))
    episode.run()
    assert mix_started.wait(5) and order[-1] == "mix" and not episode.stamp("mix").exists()  # run() returned while the mix still works
    release.set()
    background.join()
    assert episode.stamp("mix").exists() and episode.stamp("check").exists()
    order.clear()
    again = _episode(tmp_path, engine="f5")  # a rerun: every stage is done
    again.args.mix_background = run.Background()
    again.run()
    assert order == []


# ---- 4. on-screen translation beside the dub

def test_onscreen_translation_runs_in_the_background_and_the_video_waits_for_it(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    release, calls = threading.Event(), []
    target = episode.out / "onscreen.ru.json"

    def slow(self: run.Episode) -> Path:
        calls.append(1)
        assert release.wait(5)
        target.touch()
        return target

    monkeypatch.setattr(run.Episode, "_translate_onscreen", slow)
    episode.start_onscreen()  # returns at once
    answers: list[Path | None] = []
    waiters = [threading.Thread(target=lambda: answers.append(episode.translate_onscreen())) for _ in range(2)]
    for waiter in waiters:
        waiter.start()
    assert answers == []  # the mux side is waiting
    release.set()
    for waiter in waiters:
        waiter.join(5)
    run.wait_onscreen()
    assert answers == [target, target] and calls == [1]  # one translation for everybody


def test_failed_onscreen_translation_gives_no_track_and_no_error(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path)

    def broken(self: run.Episode) -> Path:
        raise RuntimeError("LLM down")

    monkeypatch.setattr(run.Episode, "_translate_onscreen", broken)
    episode.start_onscreen()
    run.wait_onscreen()
    assert episode.translate_onscreen() is None


def test_without_a_background_job_the_translation_is_done_in_place(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    monkeypatch.setattr(run.Episode, "_translate_onscreen", lambda self: tmp_path / "x.json")
    assert episode.translate_onscreen() == tmp_path / "x.json"


# ---- batches of on-screen translation run side by side

class _Client:
    def __init__(self, meet: int = 0) -> None:
        self.meet, self.seen, self.lock = meet, 0, threading.Lock()
        self.together = threading.Event()
        self.prompts: list[str] = []

    def chat_json(self, system: str, user: str, **_: Any) -> dict[str, Any]:
        import json

        with self.lock:
            self.seen += 1
            self.prompts.append(user)
            if self.seen == self.meet:
                self.together.set()
        if self.meet and self.seen <= self.meet:
            assert self.together.wait(5), "the batches did not run at the same time"
        rows = [json.loads(line) for line in user.split("Lines:\n", 1)[1].splitlines()]
        return {"items": [{"i": r["i"], "ru": "RU " + r["en"]} for r in rows]}


def test_onscreen_batches_run_concurrently_with_the_same_result() -> None:
    items = [{"id": n, "kind": "caption", "text": f"TEXT {n}", "start": n, "end": n + 1, "voiced": False, "label": None}
             for n in range(8)]
    one = _Client()
    expected = onscreen.translate_items(items, [], one, batch_size=2, workers=1)
    many = _Client(meet=3)
    got = onscreen.translate_items(items, [], many, batch_size=2, workers=4)
    assert got == expected
    assert sorted(many.prompts) == sorted(one.prompts)  # the same batches and prompts
    assert [i["text_ru"] for i in got] == [f"RU TEXT {n}" for n in range(8)]


# ---- the separation cache goes away after the last mux

def test_separation_cache_is_removed_after_the_final_mux_only(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    muxed: list[list[str]] = []
    _fake_mix_parts(monkeypatch, muxed)
    caplog.set_level(logging.INFO, logger="stt.run")
    cache = _episode(tmp_path).out / "separation"
    cache.mkdir()
    (cache / "voices.x.npy").write_bytes(b"1")
    for engine in ("f5", "teratts"):  # tracks missing: the mux waits and a rerun can reuse the cache
        _episode(tmp_path, engine=engine).mix()
        assert cache.exists()
    _episode(tmp_path, engine="vosk").mix()
    assert muxed and not cache.exists()
    assert "voice separation cache removed" in caplog.text


def test_separation_cache_stays_when_the_mux_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_mix_parts(monkeypatch, [])

    def broken(self: run.Episode) -> None:
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(run.Episode, "mux", broken)
    episode = _episode(tmp_path, planned=["f5"], engine="f5")
    (episode.out / "separation").mkdir()
    with pytest.raises(RuntimeError):
        episode.mix()
    assert (episode.out / "separation").exists()


def test_separate_mode_removes_the_cache_after_the_last_planned_engine(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_mix_parts(monkeypatch, [])
    cache = _episode(tmp_path, tracks="separate", planned=["f5", "vosk"]).out / "separation"
    cache.mkdir()
    _episode(tmp_path, tracks="separate", planned=["f5", "vosk"], engine="f5").mix()
    assert cache.exists()
    _episode(tmp_path, tracks="separate", planned=["f5", "vosk"], engine="vosk").mix()
    assert not cache.exists()


# ---- the compressed video is made beside the voicing

def _video_episode(tmp_path: Path, name: str = "ep01", *, burn: bool = False, engine: str = "f5",
                   compress: bool = True) -> run.Episode:
    episode = _episode(tmp_path, engine=engine)
    episode.media = tmp_path / "S" / f"{name}.mkv"
    episode.out = tmp_path / "out" / name
    episode.out.mkdir(parents=True, exist_ok=True)
    episode.config["mix"]["video"] = {"compress": compress, "target_gb": 2.0, "codec": "hevc_nvenc",
                                      "original_audio_kbps": 160}
    episode.config["on_screen"] = {"burn_in_when_compressing": burn}
    return episode


@pytest.fixture
def encodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from stt import video_compress

    state: dict[str, Any] = {"calls": [], "running": 0, "max_running": 0, "release": threading.Event(),
                             "started": threading.Event(), "fail": False}
    lock = threading.Lock()
    state["release"].set()

    def encode(src: Path, out: Path, bitrate: int, codec: str, overlays: Any = None, **_: Any) -> Path:
        with lock:
            state["calls"].append((src.name, out.name))
            state["running"] += 1
            state["max_running"] = max(state["max_running"], state["running"])
        state["started"].set()
        assert state["release"].wait(5)
        with lock:
            state["running"] -= 1
        if state["fail"]:
            raise RuntimeError("nvenc failed")
        out.touch()
        return out

    monkeypatch.setattr(video_compress, "probe", lambda _: {"duration": 3600.0, "video_kbps": 9000.0,
                                                           "width": 1920, "height": 1080})
    monkeypatch.setattr(video_compress, "encode_video", encode)
    run._COMPRESS_JOBS.clear()
    return state


def test_compression_starts_with_the_voicing_and_the_mux_awaits_it_encoding_once(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encodes: dict[str, Any]) -> None:
    episode = _video_episode(tmp_path)
    encodes["release"].clear()
    muxed: list[Any] = []
    monkeypatch.setattr(stt_mix, "mux_tracks", lambda *a, **k: muxed.append(k["video_source"]))
    started_order: list[str] = []
    monkeypatch.setattr(run.Episode, "start_compression", lambda self, orig=run.Episode.start_compression:
                        (started_order.append("compress"), orig(self))[1])
    for name in ("transcribe", "glossary", "speakers", "translate", "stress", "voiceover", "check", "mix"):
        monkeypatch.setattr(run.Episode, name, lambda self, n=name: None)
    monkeypatch.setattr(run.Episode, "dub", lambda self: started_order.append("dub"))
    episode.args.until = "dub"
    episode.forced_from = 0  # every stage is needed
    episode.run()
    assert started_order == ["compress", "dub"]  # started at the first voicing stage, before it
    assert encodes["started"].wait(5)
    for engine in ("f5", "teratts", "vosk"):
        episode.track(engine).touch()
    result: list[Path | None] = []
    waiter = threading.Thread(target=lambda: result.append(episode.small_video(episode.config["mix"]["video"], 3)))
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive() and result == []  # the mux waits for the running encoding
    encodes["release"].set()
    waiter.join(5)
    assert result == [episode.out / "video.hevc_nvenc.3619k.mkv"]
    assert episode.small_video(episode.config["mix"]["video"], 3) == result[0]
    assert len(encodes["calls"]) == 1  # one encoding in all


def test_failed_background_compression_falls_back_to_copying(tmp_path: Path, encodes: dict[str, Any]) -> None:
    episode = _video_episode(tmp_path)
    encodes["fail"] = True
    episode.start_compression()
    assert episode.small_video(episode.config["mix"]["video"], 3) is None  # today's fallback: the video is copied
    assert len(encodes["calls"]) == 1  # no second try


def test_burn_in_off_switches_background_compression_off(tmp_path: Path, encodes: dict[str, Any]) -> None:
    episode = _video_episode(tmp_path, burn=True)
    episode.start_compression()
    assert run._COMPRESS_JOBS == {} and encodes["calls"] == []
    piece = _video_episode(tmp_path, "piece")
    piece.args.start, piece.args.duration = 60.0, 30.0
    piece.start_compression()
    off = _video_episode(tmp_path, "off", compress=False)
    off.start_compression()
    assert run._COMPRESS_JOBS == {} and encodes["calls"] == []


def test_burn_in_path_still_encodes_lazily_in_the_mux(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encodes: dict[str, Any]) -> None:
    from stt import onscreen_render

    episode = _video_episode(tmp_path, burn=True)
    (episode.out / "onscreen.ru.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(run.Episode, "translate_onscreen", lambda self: self.out / "onscreen.ru.json")
    monkeypatch.setattr(onscreen_render, "load_items", lambda path: [])
    monkeypatch.setattr(onscreen_render, "render_overlays", lambda items, w, h: ["overlay"])
    small = episode.small_video(episode.config["mix"]["video"], 3)
    assert small is not None and ".onscreen-" in small.name and len(encodes["calls"]) == 1


def test_a_season_encodes_one_video_at_a_time(tmp_path: Path, encodes: dict[str, Any]) -> None:
    encodes["release"].clear()
    first, second = _video_episode(tmp_path, "ep01"), _video_episode(tmp_path, "ep02")
    first.start_compression()
    assert encodes["started"].wait(5)
    second.start_compression()  # its job waits for the first
    threading.Event().wait(0.3)
    assert [c[0] for c in encodes["calls"]] == ["ep01.mkv"] and encodes["max_running"] == 1
    encodes["release"].set()
    assert second.small_video(second.config["mix"]["video"], 3) is not None
    assert [c[0] for c in encodes["calls"]] == ["ep01.mkv", "ep02.mkv"] and encodes["max_running"] == 1
