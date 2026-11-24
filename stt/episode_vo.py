"""Voice-over side of an ``stt.run`` episode: the engine's settings and the voice of every speaker (own clip, season
stand-in, own-voice sample, the host's voice), the ``voiceover_config.json`` of the run, and the stages ``dub``,
``stress``, ``voiceover`` and ``check``.

Split out of ``stt/run.py`` as a mixin of ``stt.run.Episode``; its log lines keep the ``stt.run`` logger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stt.config import effective_voiceover_policy
from stt.runner import save_json
from stt.stage_runner import run_module

log = logging.getLogger("stt.run")


def _repo_root() -> Path:
    """``stt.run.REPO_ROOT`` when called (tests point it at a temporary folder)."""
    return Path(run_module().REPO_ROOT)


def _load_json(path: Path) -> Any:
    """``stt.run.load_json`` when called (tests replace it)."""
    return run_module().load_json(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



class VoiceoverMixin:
    """``Episode`` methods of the voice-over: voices, engine settings and the stages after the translation."""

    media: Path
    out: Path
    args: Any
    config: dict[str, Any]
    engine: str
    vo_dir: Path
    voice_bank: Path
    season_stress: Path
    lock: Any

    if TYPE_CHECKING:
        @property
        def cast(self) -> Path: ...

        def llm(self, stage: str) -> Any: ...

        def on_screen(self) -> dict[str, Any]: ...

        def _module(self, name: str, module: str, args: list[str], *, oom_retries: int = 2,
                    oom_wait_s: float = 60.0) -> None: ...

    def _engine(self) -> dict[str, Any]:
        cfg = self.config["voiceover"]
        engine = dict(cfg.get("engines", {}).get(self.engine, {}))
        if "voices" not in engine:
            raise SystemExit(f"voiceover.engines.{self.engine}.voices missing in the config")
        engine.setdefault("listen_stress", bool(cfg.get("listen_stress", True)))  # takes heard for checked stress
        host = cfg.get("host_voice")
        if host and self.engine == "vosk":  # the host's lines in a TeraTTS voice inside the Vosk track
            tera = {k: v for k, v in cfg.get("engines", {}).get("teratts", {}).items() if k not in ("voices", "sample_rate")}
            engine["teratts"] = {"listen_stress": engine["listen_stress"], **tera, "voice": host}
        if self.engine == "f5":  # speakers without a clip: TeraTTS voices, or a Vosk voice cloned by F5
            from voiceover.engine_f5 import TERATTS, fallback_references, voice_id

            vo = _load_json(_repo_root() / cfg.get("config", "voiceover_config.json"))
            vosk = cfg.get("engines", {}).get("vosk", {}).get("voices", {"female": 49, "male": 56})
            refs = fallback_references(self.references.parent / "fallback_voices",
                                       _repo_root() / vo["engine"]["model_path"], {g: int(v) for g, v in vosk.items()})
            auto = {g: voice_id(refs[g]) for g in refs}
            if engine.pop("fallback", "vosk") == "teratts":
                tera = {k: v for k, v in cfg.get("engines", {}).get("teratts", {}).items() if k != "sample_rate"}
                voices = tera.pop("voices", {"female": "ru_f1+ru_f2", "male": "ru_m5"})
                engine["teratts"] = {"listen_stress": engine["listen_stress"], **tera, "voice": voices["male"]}
                engine["voice"] = auto["male"]  # the default voice F5 clones (calibration probes it)
                auto = {g: TERATTS + str(voices.get(g, voices["male"])) for g in engine["voices"]}
            engine["voices"] = {g: auto[g] if v == "auto" else voice_id(v) for g, v in engine["voices"].items()}
        return engine

    @property
    def references(self) -> Path:
        """Voice-cloning clips of the season's people (stt.voice_bank --export)."""
        return self.voice_bank.with_name(self.voice_bank.name.replace(".voices.json", ".references"))

    def _reference_clip(self, ref: dict[str, Any], language: str, warned: set[str]) -> Path:
        """A person's clip for F5: with ``reference_language: "ru"`` the Russian sample (``<stem>.ru.wav`` + ``.ru.txt``,
        made by ``stt.ru_samples``) when both exist, else (logged once per person in ``warned``) the English clip."""
        english = (self.references / ref["wav"]).resolve()
        if language != "ru":
            return english
        russian = english.with_suffix(".ru.wav")
        if russian.is_file() and russian.with_suffix(".txt").is_file():
            return russian
        if english.name not in warned:
            warned.add(english.name)
            log.warning("[%s] no Russian sample for %s (stt.ru_samples): the English clip is used", self.media.stem,
                        ref.get("name") or english.stem)
        return english

    def _speaker_voices(self) -> Path:
        """{speaker id: voice} of this episode's people (F5 engine): their own clip, else a season stand-in."""
        from stt.voice_bank import person_key, same_key
        from utils.atomic_io import write_text_atomic
        from voiceover.engine_f5 import standin_voice, voice_id

        if self.voice_bank.exists():  # people new in this episode get a clip; others keep theirs
            # prints for everyone the final speakers name (subtitles, season voices), not only the LLM-sure first pass
            self._module("voices_refresh", "stt.voice_bank", [str(self.voice_bank), "--refresh-episode", str(self.out)])
            export = [str(self.voice_bank), "--export", str(self.references), "--missing"]
            if not (self.config.get("voice_bank") or {}).get("stitch", False):
                export.append("--no-stitch")
            self._module("voices", "stt.voice_bank", export)
        language = str(self.config["voiceover"].get("engines", {}).get("f5", {}).get("reference_language", "en"))
        if language == "ru" and not (_repo_root() / "xvoice_env" / "bin" / "python").exists():
            log.warning("[%s] xvoice_env is missing (bash scripts/setup_envs.sh xvoice): no Russian samples are made",
                        self.media.stem)
        elif language == "ru":  # makes only the missing / stale samples; a failure leaves English clips for those
            try:
                self._module("ru_samples", "stt.ru_samples", [str(self.references)])
            except (Exception, SystemExit) as error:  # noqa: BLE001 - the episode goes on with the English clips
                log.warning("[%s] Russian samples not made (%s), see logs/ru_samples.log: English clips are used "
                            "for the missing ones", self.media.stem, error)
        index = self.references / "references.json"
        refs = _load_json(index) if index.exists() else []
        cards = _load_json(self.out / "speakers.json")
        subs = self.out / "subs.en.json"
        spoken = {c.get("speaker") for c in _load_json(subs)} if subs.exists() else None  # None: do not filter
        pool: dict[str, list[str]] = self.config["voiceover"].get("engines", {}).get("f5", {}).get("standin_pool") or {}
        standins_path = self.references / "standins.json"
        standins: dict[str, dict[str, Any]] = _load_json(standins_path) if standins_path.exists() else {}
        before = json.dumps(standins, sort_keys=True)
        mapping: dict[str, str] = {}
        own = stand = generic = 0
        bare: dict[str, str] = {}  # speaker id -> gender of the people left on the generic voice
        warned: set[str] = set()
        for card in cards:
            key = person_key(card)
            if spoken is not None and card["id"] not in spoken:
                continue
            ref = next((r for r in refs if key and (rk := person_key(r)) and same_key(rk, key)), None)
            if ref:
                mapping[card["id"]] = voice_id(self._reference_clip(ref, language, warned))
                own += 1
                continue
            gender = card.get("gender") if card.get("gender") in pool else "male"
            if not key or not pool.get(gender):
                generic += 1
                bare[card["id"]] = str(card.get("gender"))
                continue
            known = next((k for k in standins if same_key(k, key)), None)
            if known is None:  # the pool voice of this gender that the fewest people have so far
                taken = [v["voice"] for v in standins.values() if v.get("gender") == gender]
                spec = min(pool[gender], key=lambda x: taken.count(x))
                known = key
                standins[key] = {"voice": spec, "gender": gender, "episode": self.out.name}
            spec = standins[known]["voice"]
            model = _repo_root() / _load_json(_repo_root() / self.config["voiceover"].get(
                "config", "voiceover_config.json"))["engine"]["model_path"] if spec.startswith("vosk:") else Path()
            mapping[card["id"]] = standin_voice(spec, self.references.parent / "fallback_voices", model)
            stand += 1
        if json.dumps(standins, sort_keys=True) != before:
            self.references.mkdir(parents=True, exist_ok=True)
            write_text_atomic(standins_path, json.dumps(standins, ensure_ascii=False, indent=1) + "\n")
        converted = self._own_voice_mapping(bare)
        mapping.update(converted)
        path = self.out / "speaker_voices.f5.json"
        save_json(path, mapping)
        log.info("[%s] F5 voices: %d speakers with their own clip, %d stand-ins from the pool, %d with an own-voice "
                 "sample (X-Voice), %d on the generic voice", self.media.stem, own, stand, len(converted),
                 generic - len(converted))
        return path

    def _own_voice_mapping(self, bare: dict[str, str]) -> dict[str, str]:
        """``{speaker id: voice id of a Russian sample}`` for the male / female speakers left on the generic voice
        when ``voiceover.engines.f5.own_voice.enabled``: X-Voice says their Russian lines in the voice of a clean
        prompt of their speech in this episode (``stt.own_voice``) and ESpeech clones that sample like any clip.
        Speakers without a prompt or sample stay as they are; the reasons are logged."""
        settings = self.config["voiceover"].get("engines", {}).get("f5", {}).get("own_voice") or {}
        todo = [sid for sid, gender in bare.items() if gender in ("male", "female")]
        if not settings.get("enabled") or not todo:
            return {}
        from stt.own_voice import build_voices
        from voiceover.engine_f5 import voice_id

        reasons: dict[str, str] = {}
        samples = build_voices(self.out, todo, settings, reasons=reasons)
        for sid, reason in reasons.items():
            log.info("[%s] no own-voice sample for %s (%s): the generic voice stays", self.media.stem, sid, reason)
        return {sid: voice_id(wav) for sid, wav in samples.items()}

    def _host_speakers(self) -> list[str]:
        """Speaker ids of the show's host: people of the season cast with a "host" role (else the
        speaker cards' own role)."""
        from stt.voice_bank import same_person

        cards = _load_json(self.out / "speakers.json")
        cast = _load_json(self.cast).get("cast", []) if self.cast.exists() else []
        hosts = [p["name"] for p in cast if "host" in str(p.get("role", "")).lower()]
        if hosts:
            return [c["id"] for c in cards if c.get("name") and any(same_person(c["name"], h) for h in hosts)]
        return [c["id"] for c in cards if re.search(r"\bhost\b", str(c.get("role", "")).lower())]

    def _host_voices(self) -> Path | None:
        """{speaker id: voice} giving the host ``voiceover.host_voice`` (TeraTTS) in the Vosk / TeraTTS track."""
        host = self.config["voiceover"].get("host_voice")
        if not host or self.engine not in ("vosk", "teratts"):
            return None
        voice = f"teratts:{host}" if self.engine == "vosk" else str(host)
        mapping = {sid: voice for sid in self._host_speakers()}
        path = self.out / f"speaker_voices.{self.engine}.json"
        save_json(path, mapping)
        log.info("[%s] host voice %s for %s", self.media.stem, host, ", ".join(mapping) or "nobody (no host found)")
        return path

    def dub(self) -> None:
        from stt import dub

        voices = ",".join(f"{k}={v}" for k, v in self._engine()["voices"].items())
        argv = [str(self.out), "--voices", voices, "--interjections", self.config["dub"].get("interjections", "overlap"),
                "--output", f"dub.ru.{self.engine}.json",
                "--fill-voice", str((self.config.get("fill") or {}).get("voice", "neutral"))]
        if self.engine == "f5":
            argv += ["--speaker-voices", str(self._speaker_voices())]
        elif (host := self._host_voices()) is not None:
            argv += ["--speaker-voices", str(host)]
        on_screen = self.on_screen()
        argv += ["--on-screen-voice", str(on_screen["voice"]), "--neutral-gender", str(on_screen["neutral_gender"])]
        dub.main(argv)

    def _voiceover_config(self) -> Path:
        """voiceover_config.json with this run's engine, voices, text preparation and timing; saved per engine."""
        cfg = self.config["voiceover"]
        vo_config = _load_json(_repo_root() / cfg.get("config", "voiceover_config.json"))
        engine = {k: v for k, v in self._engine().items() if k != "voices"}
        if "sample_rate" in engine:  # export rate of the track (TeraTTS speaks at 44.1 kHz)
            vo_config["export"] = {**vo_config.get("export", {}), "sample_rate": int(engine.pop("sample_rate"))}
        normalization = {**vo_config["engine"].get("normalization", {}), **cfg.get("normalization", {}),
                         **engine.pop("normalization", {})}
        if (self.out / "stress.json").exists():
            normalization["stress_dict"] = str((self.out / "stress.json").resolve())
        vo_config["engine"] = {**vo_config["engine"], **engine, "name": self.engine, "normalization": normalization,
                               "voice": engine.get("voice") or self._engine()["voices"]["male"]}
        vo_config["policy"] = effective_voiceover_policy(self.config, vo_config)
        vo_config["units"] = {**vo_config.get("units", {}), **cfg.get("units", {})}
        openrouter = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                      "api_key_env": "OPENROUTER_API_KEY", "reasoning": "default", "max_tokens": 8192}
        vo_config["llm"] = {**vo_config.get("llm", {}), **openrouter, "model": self.llm("voiceover")}  # shortening
        vo_config["verify"] = {**vo_config.get("verify", {}),
                               "llm": {**vo_config.get("verify", {}).get("llm", {}), **openrouter,
                                       "model": self.llm("voiceover")}}  # meaning check of shortened lines
        config_path = self.out / f"voiceover.{self.engine}.config.json"
        save_json(config_path, vo_config)
        return config_path

    def _stress_input_unchanged(self) -> bool:
        """The stress check was made on exactly the lines of this engine's dubbing script."""
        from voiceover.stress_check import lines_sha256

        dub, result = self.out / f"dub.ru.{self.engine}.json", self.out / "stress.json"
        try:
            recorded = _load_json(result).get("lines_sha256")
            return bool(recorded) and dub.exists() and recorded == lines_sha256(_load_json(dub))
        except (OSError, ValueError):
            return False

    def stress(self) -> None:
        cfg = self.config["voiceover"]
        if not cfg.get("stress_check", True):
            save_json(self.out / "stress.json", {"schema": "voiceover-stress-v3", "skipped": True})
            return
        models = self.llm("stress")
        screen = self.llm("stress_screen") if (self.config.get("llm") or {}).get("stages", {}).get("stress_screen") \
            else None
        args = [str((self.out / f"dub.ru.{self.engine}.json").resolve()), "--out", str((self.out / "stress.json").resolve()),
                "--config", str(self._voiceover_config().resolve()),  # the texts exactly as this engine prepares them
                "--models", ",".join(models if isinstance(models, list) else [models])]
        if (self.out / "glossary.json").exists():  # the show's names, checked in every case form
            args += ["--names", str((self.out / "glossary.json").resolve())]
        if cfg.get("season_stress", True):  # one stress per name for the whole season
            args += ["--season", str(self.season_stress.resolve())]
        if cfg.get("stress_proofread", False):  # sol reads every line; its proposals go to the vote
            proofreader = self.llm("stress_proofread")
            args += ["--proofread", proofreader[0] if isinstance(proofreader, list) else proofreader,
                     "--proofread-passes", str(int(cfg.get("stress_proofread_passes", 2)))]
        self._module("stress", "voiceover.stress_check", args + (["--screen", screen] if screen else ["--no-screen"]))

    def voiceover(self) -> None:
        config_path = self._voiceover_config()
        self._reset_voiceover_dir()
        self._module(f"voiceover.{self.engine}", "voiceover.pipeline", [
            "run", str((self.out / f"dub.ru.{self.engine}.json").resolve()), "--out", str(self.vo_dir.resolve()),
            "--config", str(config_path.resolve())])

    def _reset_voiceover_dir(self) -> None:
        """A new dubbing script needs a fresh voice-over folder; manual overrides are kept."""
        cues = self.vo_dir / "cues.json"
        if not cues.exists() or _load_json(cues).get("source_sha256") == _sha256(self.out / f"dub.ru.{self.engine}.json"):
            return
        overrides = self.vo_dir / "overrides.json"
        kept = overrides.read_bytes() if overrides.exists() else None
        shutil.rmtree(self.vo_dir)
        if kept is not None:
            self.vo_dir.mkdir(parents=True)
            overrides.write_bytes(kept)
        log.info("[%s] voiceover: dubbing script changed, started a fresh %s", self.media.stem, self.vo_dir.name)

    def check(self) -> None:
        if not self.config["voiceover"].get("speech_check", True):
            save_json(self.vo_dir / "speech_check.json", {"schema": "voiceover-speech-check-v1", "skipped": True})
            return
        self._module(f"check.{self.engine}", "voiceover.speech_check", [str(self.vo_dir.resolve()),
                                                                         "--root", str(_repo_root())])
        review = self.out / "review.json"
        with self.lock or nullcontext():  # the other lane may be writing it
            if self.config.get("review", True) and (
                    not review.exists() or review.stat().st_mtime < (self.out / "subs.ru.json").stat().st_mtime):
                try:  # lines to check by ear: a help for a person, never a reason to stop the episode
                    self._module("review", "stt.review", [str(self.out.resolve())])
                except SystemExit as error:
                    log.warning("[%s] %s", self.media.stem, error)
