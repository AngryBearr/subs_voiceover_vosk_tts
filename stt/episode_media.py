"""Picture and sound of an ``stt.run`` episode: the Russian on-screen text (translated in the background), the
``mix`` stage (voice-over over the ducked original, the crowd layer, the widened voice band), the compressed video
and the finished video with its tracks (``mux``).

Split out of ``stt/run.py`` as a mixin of ``stt.run.Episode``; its log lines keep the ``stt.run`` logger.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stt import gpu_budget
from stt.gpu import VramPeak
from utils.llm_route import LLMStop

log = logging.getLogger("stt.run")


def finished_folder(config: dict[str, Any], media: Path) -> Path | None:
    """``paths.finished`` / ``paths.finished_folder`` with ``{season}`` = the video's folder name
    ("STT_{season}" -> "STT_Season 20"); None when finished videos stay in the episode folders."""
    paths = config.get("paths") or {}
    if not paths.get("finished"):
        return None
    return Path(paths["finished"]).expanduser() / str(paths.get("finished_folder") or "{season}").format(
        season=media.parent.name)



# On-screen translation of an episode running in the background (``Episode.start_onscreen``), by episode folder
_ONSCREEN_JOBS: dict[Path, Future[Path | None]] = {}
_ONSCREEN_GUARD = threading.Lock()
# Compressed videos being made in the background (``Episode.start_compression``), by the file they become
_COMPRESS_JOBS: dict[Path, Future[Path | None]] = {}
_COMPRESS_GUARD = threading.Lock()
_COMPRESS_GATE = threading.Lock()  # one video encoding at a time, whichever episode it is for
_MIX_GATE = threading.Lock()  # one mix at a time in the process: each holds the whole episode in RAM
# What the combined video of an episode was last muxed from (the tracks and their times), by episode folder
_MUXED: dict[Path, tuple[tuple[str, int], ...]] = {}



class MediaMixin:
    """``Episode`` methods of the on-screen text, the mix, the compressed video and the finished video."""

    media: Path
    out: Path
    args: Any
    config: dict[str, Any]
    engine: str
    vo_dir: Path
    lock: Any

    if TYPE_CHECKING:
        @property
        def vram_peaks(self) -> Path: ...

        def llm(self, stage: str) -> Any: ...

        def gpu_overlap(self, name: str, *, uses_gpu: bool = True) -> bool: ...

        def episode_minutes(self) -> float | None: ...

        def _engine(self) -> dict[str, Any]: ...

    def on_screen(self) -> dict[str, Any]:
        """``on_screen`` settings: the voice of captioned lines and the Russian on-screen text.
        Voice ``text`` (captions only shown as Russian text) needs the track: without it the voice becomes
        ``neutral`` so the captioned lines are not lost."""
        settings = {"voice": "text", "neutral_gender": "male", "track": True, "burn_in_when_compressing": False,
                    **(self.config.get("on_screen") or {})}
        if settings["voice"] == "text" and not settings["track"]:
            log.warning("[%s] on_screen.voice is 'text' but on_screen.track is off: captioned lines get the "
                        "neutral voice instead", self.media.stem)
            settings["voice"] = "neutral"
        return settings

    def start_onscreen(self) -> None:
        """Translate the on-screen text in a background thread; ``translate_onscreen`` (the video) waits for it."""
        job: Future[Path | None] = Future()
        with _ONSCREEN_GUARD:
            _ONSCREEN_JOBS[self.out] = job

        def work() -> None:
            try:
                job.set_result(self._translate_onscreen())
            except LLMStop as error:  # not swallowed: ``translate_onscreen`` raises it in the lane's thread
                job.set_exception(error)
            except BaseException as error:  # noqa: BLE001 - a failed translation only costs the track
                log.warning("[%s] on-screen text not translated (%s): no Russian on-screen track", self.media.stem,
                            error)
                job.set_result(None)

        threading.Thread(target=work, name=f"onscreen-{self.media.stem}").start()

    def translate_onscreen(self) -> Path | None:
        """``onscreen.ru.json``: on-screen captions, name cards and lower thirds in Russian (cached per text).
        None without OCR results or with the on-screen track off; a failed translation only costs the track.
        Waits for the background translation of this episode (``start_onscreen``) instead of asking again."""
        with _ONSCREEN_GUARD:
            job = _ONSCREEN_JOBS.get(self.out)
        if job is not None:
            return job.result()
        return self._translate_onscreen()

    def _translate_onscreen(self) -> Path | None:
        source, target = self.out / "onscreen.json", self.out / "onscreen.ru.json"
        if not source.exists() or not self.on_screen()["track"]:
            return None
        inputs = [p for p in (source, self.out / "subs.en.json", self.out / "speakers.json", self.out / "subs.ru.json")
                  if p.exists()]  # speakers: genders; subs.ru: the dub's wording the captions follow
        if target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):  # speakers: genders
            return target
        from stt import onscreen

        started = time.monotonic()
        try:
            onscreen.main(["translate", str(self.out), "--glossary", str(self.out / "glossary.txt"),
                           "--model", self.llm("translate")])
        except LLMStop:
            raise  # the subscription does not answer (or paying is not allowed): the run stops
        except Exception as error:  # noqa: BLE001 - the video is still made, without the on-screen track
            log.warning("[%s] on-screen text not translated (%s): no Russian on-screen track", self.media.stem, error)
            return None
        log.info("[%s] on-screen text translated in %d s", self.media.stem, round(time.monotonic() - started))
        return target if target.exists() else None


    def track(self, engine: str) -> Path:
        """The engine's finished voice-over mix (AAC), kept for the videos."""
        return self.out / f"{self.media.stem}.ru.{engine}.m4a"

    @property
    def finished(self) -> Path:
        """Folder of the finished video: ``paths.finished`` / ``paths.finished_folder`` (``{season}``:
        the video's folder name), else the episode folder (always for a test piece)."""
        folder = finished_folder(self.config, self.media)
        return self.out if folder is None or self.args.start or self.args.duration else folder

    def finished_video(self, engine: str | None = None) -> Path:
        """The video with all voice-overs (``mix.tracks: combined``) or with ``engine``'s (separate)."""
        combined = (self.config.get("mix") or {}).get("tracks", "combined") == "combined"
        return self.finished / f"{self.media.stem}.ru{'' if combined else f'.{engine or self.engine}'}.mkv"

    def mux(self) -> None:
        """The finished video: the voice-overs finished so far (combined) or this engine's (separate)."""
        from stt import mix

        cfg = self.config["mix"]
        if cfg.get("tracks", "combined") == "combined":
            order = [*cfg.get("track_order", []), *self.config["voiceover"].get("engines", {}), self.engine]
            engines = [e for e in dict.fromkeys(order) if self.track(e).exists()]
        else:
            engines = [self.engine]
        titles = cfg.get("track_titles") or {}
        tracks = [(self.track(e), titles.get(e, f"Русский ({e})")) for e in engines]
        srt = None
        if cfg.get("subtitles", True):  # the voiced lines (the same in every engine); the translation if none
            voiced = self.out / f"voiceover.{engines[0]}" / "export" / "display.srt"
            srt = voiced if voiced.exists() else self.out / "subs.ru.srt"
        out = self.finished_video()
        out.parent.mkdir(parents=True, exist_ok=True)
        video_cfg = cfg.get("video") or {}
        compress = bool(video_cfg.get("compress")) and not (self.args.start or self.args.duration)
        video_source = self.small_video(video_cfg, len(self.planned_tracks(engines))) if compress else None
        onscreen_ass, fonts = None, []
        if not (video_source and self.burn_onscreen()):  # burned into the compressed picture: no second copy
            onscreen_ass, fonts = self.onscreen_track()
        mix.mux_tracks(self.media, tracks, out, srt, self.args.start or 0.0, self.args.duration,
                       video_source=video_source,
                       original_audio_kbps=int(video_cfg.get("original_audio_kbps", 160)) if video_source else None,
                       onscreen_ass=onscreen_ass, fonts=fonts)
        log.info("[%s] %s: %s", self.media.stem, out, ", ".join(engines))
        if video_source:
            log.info("[%s] finished video %.2f GB", self.media.stem, out.stat().st_size / 1e9)

    def burn_onscreen(self) -> bool:
        return bool(self.on_screen()["burn_in_when_compressing"]) and self.translate_onscreen() is not None

    def onscreen_track(self) -> tuple[Path | None, list[Path]]:
        """The Russian on-screen text as an ASS track (default, forced) and the fonts to attach; (None, []) if none."""
        items = self.translate_onscreen()
        if items is None:
            return None, []
        from stt import onscreen_render, video_compress

        info = video_compress.probe(self.media)
        ass = onscreen_render.write_ass(onscreen_render.load_items(items), self.out / "onscreen.ass",
                                        int(info["width"]), int(info["height"]), offset_s=self.args.start or 0.0)
        return ass, list(onscreen_render.font_files() or []) if ass else []

    def planned_tracks(self, engines: list[str]) -> list[str]:
        """Voice-over tracks the finished video will have: those in it now and the engines of this run."""
        if (self.config["mix"].get("tracks", "combined") != "combined"):
            return engines
        return list(dict.fromkeys([*engines, *getattr(self.args, "planned_engines", [])]))

    def _small_target(self, video_cfg: dict[str, Any], tracks: int) -> tuple[dict[str, Any], int, Path] | None:
        """(probe, bitrate, file) of the compressed video; None when the picture is already small enough."""
        from stt import video_compress

        codec = str(video_cfg.get("codec", "hevc_nvenc"))
        original_kbps = int(video_cfg.get("original_audio_kbps", 160))
        info = video_compress.probe(self.media)
        bitrate = video_compress.video_bitrate(info["duration"], float(video_cfg.get("target_gb", 2.0)),
                                               tracks * 192 + original_kbps, info["video_kbps"])
        if bitrate is None:
            return None
        return info, bitrate, self.out / f"video.{codec}.{bitrate}k.mkv"

    def _encode_small(self, small: Path, bitrate: int, codec: str, overlays: Any = None) -> Path | None:
        """Make ``small`` (one encoding at a time); None when the encoding failed (the video is copied then)."""
        from stt import video_compress

        with _COMPRESS_GATE:
            if small.exists():
                return small
            started = time.monotonic()
            try:
                with VramPeak(interval_s=0.5) as peak:  # this process's total: includes a voice separation at the time
                    video_compress.encode_video(self.media, small, bitrate, codec, overlays=overlays)
                gpu_budget.record_peak(self.vram_peaks, "nvenc", peak.peak_mb)
            except Exception as error:  # noqa: BLE001 - a big video is better than no video
                log.warning("[%s] compressing the video failed (%s): the video is copied as it is", self.media.stem, error)
                return None
            log.info("[%s] compressed video made in %d s", self.media.stem, round(time.monotonic() - started))
        return small

    def start_compression(self) -> None:
        """Start the compressed video in the background, once per file, as soon as the episode's voicing starts, so
        the final mux finds it ready. Not for a test piece, nor with the on-screen text burned in (that picture needs
        the final translation: ``small_video`` makes it then). The bitrate counts the tracks of this run."""
        cfg = self.config.get("mix") or {}
        video_cfg = cfg.get("video") or {}
        if not (cfg.get("mux") and video_cfg.get("compress")) or self.args.start or self.args.duration \
                or self.on_screen()["burn_in_when_compressing"]:
            return
        existing = [e for e in self.config.get("voiceover", {}).get("engines", {}) if self.track(e).exists()]
        try:
            target = self._small_target(video_cfg, len(self.planned_tracks([*existing, self.engine])))
        except Exception as error:  # noqa: BLE001 - the mux tries again and falls back to copying
            log.warning("[%s] could not prepare the compressed video (%s)", self.media.stem, error)
            return
        if target is None:
            return
        _, bitrate, small = target
        gpu_encoder = str(video_cfg.get("codec", "hevc_nvenc")).endswith("_nvenc")
        if small.exists() or not self.gpu_overlap("compress", uses_gpu=gpu_encoder):
            return  # not enough video memory beside the voicing: the final mux compresses it (small_video)
        job: Future[Path | None] = Future()
        with _COMPRESS_GUARD:
            if small in _COMPRESS_JOBS or small.exists():
                return
            _COMPRESS_JOBS[small] = job

        def work() -> None:
            try:
                job.set_result(self._encode_small(small, bitrate, str(video_cfg.get("codec", "hevc_nvenc"))))
            except BaseException as error:  # noqa: BLE001
                log.warning("[%s] compressing the video failed (%s): the video is copied as it is", self.media.stem, error)
                job.set_result(None)

        log.info("[%s] compressing the video in the background (%d kbps)", self.media.stem, bitrate)
        threading.Thread(target=work, name=f"compress-{self.media.stem}", daemon=True).start()

    def small_video(self, video_cfg: dict[str, Any], tracks: int) -> Path | None:
        """The episode's video with a smaller picture for ``mix.video.target_gb`` (made once, shared by the engines'
        videos; the background one of ``start_compression`` is awaited, never made twice); None when the picture is
        already small enough or the encoding failed (the video is copied then)."""
        target = self._small_target(video_cfg, tracks)
        if target is None:
            log.info("[%s] the video is already close to the target size: copied as it is", self.media.stem)
            return None
        info, bitrate, small = target
        with _COMPRESS_GUARD:
            job = _COMPRESS_JOBS.get(small)
        if job is not None:
            if not job.done():
                log.info("[%s] waiting for the compressed video", self.media.stem)
            result = job.result()
            if result is None or result.exists():
                return result
        overlays = None
        if self.burn_onscreen():
            from stt import onscreen_render

            overlays = onscreen_render.render_overlays(onscreen_render.load_items(self.out / "onscreen.ru.json"),
                                                       int(info["width"]), int(info["height"]))
            # the burned-in text is part of the picture: a changed translation needs a new video
            digest = hashlib.sha256((self.out / "onscreen.ru.json").read_bytes()).hexdigest()[:8]
            small = small.with_name(small.stem + f".onscreen-{digest}" + small.suffix) if overlays else small
        if small.exists():
            return small
        return self._encode_small(small, bitrate, str(video_cfg.get("codec", "hevc_nvenc")), overlays)

    def mix(self) -> None:
        from stt import mix

        cfg = self.config["mix"]
        out_wav = self.out / f"{self.media.stem}.ru.{self.engine}.wav"
        voice = self.widened_track(self.vo_dir / "export" / "track.wav")
        argv = [str(self.media), str(voice), "--out", str(out_wav),
                "--duck-db", str(cfg["duck_db"])]
        if not self.args.start and not self.args.duration:
            argv.append("--full-duration")
        if cfg.get("duck") == "voices":  # lower only the original voices, keep music and effects
            argv += ["--duck-voices-only", "--rest-duck-db", str(cfg.get("rest_duck_db", 6.0)),
                     "--separation-cache", str(self.out / "separation"),  # one separation for all engines
                     "--vram-peaks", str(self.vram_peaks)]
        argv += ["--voice-gain-db", str(cfg.get("voice_gain_db", 3.0))]
        if cfg.get("loudness_lufs") is not None:  # the whole mix to this level, peaks limited (null: leave as mixed)
            argv += ["--loudness-lufs", str(cfg["loudness_lufs"])]
        if self.args.start:  # a test piece: the voice-over is on the episode timeline from 0
            argv += ["--start", str(self.args.start), "--voice-from-zero"]
        crowd = self.config.get("crowd", {})
        if crowd.get("enabled"):  # shouts the dub left out, on a layer of their own under the voice-over
            from stt import crowd as crowd_layer

            generic = ",".join(f"{k}={v}" for k, v in self._engine()["voices"].items())  # this engine's, not the last dub's
            crowd_layer.main([str(self.out), "--engine", self.engine, "--voices", str(crowd.get("voices", 1)),
                              "--neutral-gender", str(self.on_screen()["neutral_gender"]), "--generic-voices", generic]
                             + (["--solo"] if crowd.get("solo", True) else []))
            argv += ["--extra", f"{self.out / f'crowd.{self.engine}.wav'}:{crowd.get('gain_db', -3.0)}"]
        with _MIX_GATE:  # two lanes mixing at once, or a mix beside the F5 worker without room, got the process OOM-killed
            gpu_budget.wait_for_mix_ram(self.episode_minutes(), self.media.stem)
            mix.main(argv)
        started = time.monotonic()
        mix.encode_track(out_wav, self.track(self.engine))
        log.info("[%s] mix (%s): AAC encode %.0f s", self.media.stem, self.engine, time.monotonic() - started)
        if not cfg.get("keep_wav"):  # ~0.5 GB per episode and engine; the AAC track is what the video gets
            out_wav.unlink()
        if cfg.get("mux"):  # a test piece gets a short re-encoded video of the piece
            self.mux_when_ready()

    def widened_track(self, track: Path) -> Path:
        """The voice track to mix: for a cloned engine whose references are narrowband (old seasons), ``track.wide.wav``
        with the band drawn above their edge (``stt.widen``, ``mix.widen`` in the config); else ``track`` itself."""
        from stt import widen

        cfg = (self.config.get("mix") or {}).get("widen") or {}
        if not cfg.get("enabled", False) or self.engine not in cfg.get("engines", ["f5"]) or not track.exists():
            return track
        edge = widen.references_edge(self.out / f"speaker_voices.{self.engine}.json")
        if edge is None or edge >= float(cfg.get("max_band_hz", widen.MAX_BAND_HZ)):
            return track
        out = track.with_name("track.wide.wav")
        if out.exists() and out.stat().st_mtime >= track.stat().st_mtime:
            return out
        started = time.monotonic()
        if not widen.widen_track(track, out, edge):
            return track
        log.info("[%s] mix (%s): the voices' band widened above %.0f Hz (LavaSR) in %.0f s", self.media.stem, self.engine,
                 edge, time.monotonic() - started)
        return out

    def mux_when_ready(self) -> None:
        """Mux the finished video. ``combined``: once, by the engine whose track is the last of this run's engines
        (the others only log that the mux waits); ``separate``: this engine's own video."""
        combined = (self.config.get("mix") or {}).get("tracks", "combined") == "combined"
        with self.lock or nullcontext():  # one video for all engines: one writer at a time
            signature: tuple[tuple[str, int], ...] = ()
            if combined:
                planned = self.planned_tracks([self.engine])
                waiting = [e for e in planned if not self.track(e).exists()]
                if waiting:
                    log.info("[%s] mix (%s): mux deferred until %s finish", self.media.stem, self.engine,
                             ", ".join(waiting))
                    return
                signature = tuple(sorted((e, self.track(e).stat().st_mtime_ns) for e in planned))
                if _MUXED.get(self.out) == signature and self.finished_video().exists():
                    log.info("[%s] mix (%s): the video with these tracks is already made", self.media.stem, self.engine)
                    return
            started = time.monotonic()
            self.mux()
            log.info("[%s] mix (%s): mux %.0f s", self.media.stem, self.engine, time.monotonic() - started)
            if combined:
                _MUXED[self.out] = signature
            self._clear_separation()

    def _clear_separation(self) -> None:
        """Delete the cached voice separation (~1.5 GB per hour) once the video is made and every engine of this run
        has its track; before that (or after a failed mux, which never gets here) a rerun reuses it.
        Done whatever ``mix.keep_wav`` says: that setting is about the mixed WAVs."""
        cache = self.out / "separation"
        engines = dict.fromkeys([self.engine, *getattr(self.args, "planned_engines", [])])
        waiting = [e for e in engines if not self.track(e).exists()]
        if waiting or not cache.exists():
            return
        shutil.rmtree(cache, ignore_errors=True)
        log.info("[%s] mix (%s): voice separation cache removed (%s)", self.media.stem, self.engine, cache)
