"""Configuration values of the processing form, without application state."""

from __future__ import annotations

from typing import Any

from stt.run import apply_preset
from ui.app_paths import ENGINES, MODES, MODELS, TRANSPORTS

PREPARE_TITLE = "Подготовка сезона"
PREPARE_HELP = (
    "**Подготовить весь сезон** - распознаёт речь, текст на экране, кто говорит и имена во **всех** сериях папки "
    "(не только отмеченных), без перевода и озвучки. Нужна видеокарта, примерно 8-10 минут на серию. "
    "Зачем: голоса сезона собираются по всем сериям сразу и получаются точнее, а потом каждая серия только "
    "переводится и озвучивается (примерно на 15 минут быстрее). После подготовки можно открыть вкладку "
    "«Кто это?» и назвать голоса без имени. Уже переведённые серии не меняются.")


def job_config(mode: str, model: str, subtitles: bool, duck_voices: bool, stress: bool, refine: bool,
               f5_fast: bool = False, transport: str | None = None, preset: str | None = None,
               *, base: dict[str, Any]) -> dict[str, Any]:
    """Apply a preset and the form's options to the supplied settings."""
    config = apply_preset(base, preset)
    config["mode"] = MODES.get(mode, mode)
    config.setdefault("llm", {})["model"] = MODELS.get(model, model)
    if TRANSPORTS.get(transport or "") == "openrouter":  # pay per request instead of the subscription
        config["llm"].update(opencode=[], router="openrouter")
    config.setdefault("mix", {}).update(subtitles=subtitles, duck="voices" if duck_voices else "all")
    config.setdefault("voiceover", {})["stress_check"] = stress
    config["speaker_refine"] = {**(config.get("speaker_refine") or {}), "model": "resnet293" if refine else "none"}
    f5 = config["voiceover"].setdefault("engines", {}).setdefault("f5", {"voices": {"female": "auto", "male": "auto"}})
    nfe = int(f5.get("nfe", 32))
    f5["nfe"] = 7 if f5_fast else (32 if nfe == 7 else nfe)  # 7 steps: ~2x faster, by ear almost the same
    return config


def default_model(config: dict[str, Any]) -> str:
    model = config.get("llm", {}).get("model")
    return next((label for label, value in MODELS.items() if value == model), list(MODELS)[0])


def preset_choices(config: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, name) of the sets of settings in the config."""
    return [(str(p.get("label") or name), name) for name, p in (config.get("presets") or {}).items()]


def form_defaults(config: dict[str, Any]) -> list[Any]:
    """Starting values of the processing form in component order."""
    chosen = (config.get("ui") or {}).get("engines") or ["vosk"]
    f5 = ((config.get("voiceover") or {}).get("engines") or {}).get("f5") or {}
    return [[label for label, value in ENGINES.items() if value in chosen],
            next((label for label, value in MODES.items() if value == config.get("mode")), list(MODES)[0]),
            default_model(config),
            bool((config.get("mix") or {}).get("subtitles", True)), (config.get("mix") or {}).get("duck") == "voices",
            bool((config.get("voiceover") or {}).get("stress_check", True)),
            (config.get("speaker_refine") or {}).get("model", "resnet293") != "none", int(f5.get("nfe", 32)) == 7]
