"""Settings actions, voice fields and the settings tab."""

from __future__ import annotations

from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gradio as gr

from ui import folder_pick, season_settings, seasons, settings, voice_preview
from ui.app_jobs import Job
from ui.app_paths import NO_SEASON, _season
from ui.widgets import wait_state

TARGET_GLOBAL, TARGET_SEASON = "global", "season"
VOICE_ENGINE = {("voiceover", "engines", "vosk", "voices", "female"): "vosk",
                ("voiceover", "engines", "vosk", "voices", "male"): "vosk",
                ("voiceover", "engines", "teratts", "voices", "female"): "teratts",
                ("voiceover", "engines", "teratts", "voices", "male"): "teratts",
                ("voiceover", "host_voice"): "teratts"}
FIELDS_PER_ROW = 3


def target_choices(season: seasons.Season | None) -> list[tuple[str, str]]:
    """The radio "Сохранять изменения в"."""
    choices = [("общие настройки", TARGET_GLOBAL)]
    if season is not None:
        choices.append((f"настройки сезона {season.name}", TARGET_SEASON))
    return choices


def effect_text(season: seasons.Season | None) -> str:
    """Which settings the season runs with, in words."""
    if season is None:
        return "Сезон не выбран: действуют общие настройки."
    if season_settings.has_own(season.results):
        return (f"Сейчас для сезона {season.name} действуют: настройки сезона "
                f"(`{season_settings.season_file(season.results)}`).")
    return f"Сейчас для сезона {season.name} действуют: общие настройки."


def save_settings(folder: str, out_root: str, target: str, *values: Any,
                  base_config: Callable[[], dict[str, Any]]) -> str:
    """"Сохранить": the form goes to the global file or to the season's file, as the radio says."""
    season = _season(folder, out_root)
    try:
        if target == TARGET_SEASON and season is not None:
            path = season_settings.save_own(list(values), season.results, base_config())
            return (f"Сохранено в настройки сезона {season.name}: `{path}`. Действует со следующего запуска "
                    "и только для этого сезона.")
        path = season_settings.global_path()
        own, warnings = settings.save(list(values), local_path=path)
    except settings.SettingsError as exc:
        return "Не сохранено:\n\n" + "\n".join(f"- {p}" for p in exc.problems)
    text = f"Сохранено в общие настройки: `{path}` (своих значений: {own}). Действует со следующего запуска."
    if season is not None and season_settings.has_own(season.results):
        text += f" Сезон {season.name} использует свои настройки: на него это не влияет."
    return text + "".join(f"\n- {w}" for w in warnings)


def reset_settings(folder: str, out_root: str, target: str) -> str:
    """"Сбросить к стандартным": the chosen file gets the standard values."""
    season = _season(folder, out_root)
    if target == TARGET_SEASON and season is not None:
        path = season_settings.reset_own(season.results, settings.read_json(settings.DEFAULTS))
        return f"Настройки сезона {season.name} сброшены к стандартным: `{path}`."
    path = season_settings.global_path()
    settings.reset(path)
    return f"Общие настройки сброшены к стандартным (личные ключи вроде подписки остались): `{path}`."


def remove_season_settings(folder: str, out_root: str) -> str:
    season = _season(folder, out_root)
    if season is None:
        return NO_SEASON
    if not season_settings.remove_own(season.results):
        return f"У сезона {season.name} нет своих настроек: действуют общие."
    return (f"Файл настроек сезона {season.name} удалён (`{season_settings.season_file(season.results)}`): "
            "теперь для сезона действуют общие настройки.")


def apply_global(text: str) -> Any:
    """Use the file in the field as the global settings (made as a copy of the current ones when it is new)."""
    status, message = season_settings.choose_global(text)
    return gr.update() if status == "same" else message


def season_line_text(season: seasons.Season | None) -> str:
    if season is None:
        return seasons.summary(None)
    return f"{seasons.summary(season)} · действуют: {season_settings.in_effect(season.results)}"


def tts_cache(delete: bool, *, lock: LockType, current_job: Callable[[], Job | None]) -> str:
    """The TTS cache cleanup (``voiceover.cache_clean``) as a message: what goes, or what went."""
    from voiceover.cache_clean import apply, cache_dir, default_roots, plan

    cache = cache_dir()
    if not cache.is_dir():
        return "Кэша озвучки нет."
    with lock:
        job = current_job()
        if delete and job and job.state == "идёт":
            return "Идёт обработка: очистите кэш после неё."
    result = plan(cache, default_roots(), keep_days=3)
    gb = 2 ** 30
    text = (f"Кэш озвучки: {result.keep_bytes / gb:.1f} ГБ нужны сериям (или сделаны за последние 3 дня), "
            f"{result.drop_bytes / gb:.1f} ГБ ({len(result.drop)} файлов) не нужны.")
    if delete:
        apply(result)
        text = f"Удалено {result.drop_bytes / gb:.1f} ГБ из кэша озвучки; осталось {result.keep_bytes / gb:.1f} ГБ."
    return text


def _setting_field(setting: settings.Setting, value: Any) -> Any:
    if setting.kind == "bool":
        return gr.Checkbox(label=setting.label, value=value, info=setting.info or None)
    if setting.kind == "choice":
        return gr.Dropdown(label=setting.label, choices=list(setting.choices), value=value, info=setting.info or None,
                           min_width=280)
    if setting.kind == "engines":
        return gr.CheckboxGroup(label=setting.label, choices=list(setting.choices), value=value)
    if setting.kind == "vosk_voice":
        return gr.Dropdown(label=setting.label, choices=settings.vosk_voice_choices(setting.path[-1]), value=value,
                           info=setting.info or None, min_width=280)
    if setting.kind in ("number", "integer"):
        return gr.Number(label=setting.label, value=value, precision=0 if setting.kind == "integer" else None,
                         info=setting.info or None, min_width=200)
    if setting.kind == "voice":
        voices = settings.TERATTS_VOICES + sorted(settings.teratts_styles() - set(settings.TERATTS_VOICES))
        return gr.Dropdown(label=setting.label, choices=voices, value=value, allow_custom_value=True,
                           info=setting.info or None, min_width=280)
    return gr.Textbox(label=setting.label, value=value, info=setting.info or None, min_width=280)


def listen_voice(path: tuple[str, ...], voice: Any) -> tuple[Any, str]:
    """(sound, message) of the sample phrase in the voice now chosen in the form (not necessarily saved yet)."""
    engine = VOICE_ENGINE[tuple(path)]
    if voice is None or str(voice).strip() == "":
        return None, "Сначала выберите голос."
    wav, message = voice_preview.preview(engine, voice)
    return (str(wav) if wav else None), message


def _settings_group(group: str, values: dict[tuple[str, ...], Any], fields: dict[tuple[str, ...], Any]) -> None:
    """The fields of one group: the others in rows of three, voices with their "▶ Послушать", boxes one per line,
    so every label and hint is whole. Components are collected in ``fields`` by path (the order of the form
    is ``settings.SETTINGS``, whatever the order on the page)."""
    members = [s for s in settings.SETTINGS if s.group == group]
    plain = [s for s in members if s.kind != "bool" and s.path not in VOICE_ENGINE]
    voices = [s for s in members if s.path in VOICE_ENGINE]
    boxes = [s for s in members if s.kind == "bool"]
    for start in range(0, len(plain), FIELDS_PER_ROW):
        with gr.Row(equal_height=False):
            for setting in plain[start:start + FIELDS_PER_ROW]:
                fields[setting.path] = _setting_field(setting, values[setting.path])
    for setting in voices:
        with gr.Group():
            with gr.Row(equal_height=False):
                fields[setting.path] = _setting_field(setting, values[setting.path])
                listen = gr.Button("▶ Послушать", scale=0, min_width=160)
            note = gr.Markdown(visible=False)
            player = gr.Audio(label="Образец голоса", type="filepath", autoplay=True, interactive=False,
                              visible=False, show_label=False)
        listen.click(lambda voice, path=setting.path: _listen_started(path, voice), fields[setting.path],
                     [listen, note, player], queue=False).then(
            lambda voice, path=setting.path: _listen_outputs(path, voice), fields[setting.path],
            [listen, note, player])
    for setting in boxes:
        fields[setting.path] = _setting_field(setting, values[setting.path])


def _listen_started(path: tuple[str, ...], voice: Any) -> tuple[Any, Any, Any]:
    """Shown at once on the click: the button waits, the line says what happens and how long it may take."""
    engine = VOICE_ENGINE[tuple(path)]
    if voice is not None and str(voice).strip() and voice_preview.preview_path(engine, voice).exists():
        text = "⏳ Включаю образец…"
    elif engine == "vosk" and voice_preview.VOSK_PREPARING.is_set():
        text = "⏳ Образцы голосов Vosk ещё готовятся после запуска программы (до 1,5 мин), дождитесь…"
    else:
        wait = "до 45 с" if engine == "vosk" else "около 15 с"
        text = f"⏳ Готовлю образец этого голоса ({wait}, только в первый раз)…"
    return (gr.update(value="⏳ Готовлю…", interactive=False), gr.update(value=text, visible=True),
            gr.update(value=None, visible=False))


def _listen_outputs(path: tuple[str, ...], voice: Any) -> tuple[Any, Any, Any]:
    sound, message = listen_voice(path, voice)
    text = f"✅ {message} Звучит фраза-образец." if sound else f"❌ {message}"
    return (gr.update(value="▶ Послушать", interactive=True), gr.update(value=text, visible=True),
            gr.update(value=sound, visible=sound is not None))


@dataclass
class Tab:
    note: Any
    fields: list[Any]
    effect: Any
    target: Any
    remove: Any
    global_box: Any
    pick_global: Any
    use_global: Any
    browser: folder_pick.Browser
    pick_message: Any
    save: Any
    reset: Any
    message: Any


def build(config: dict[str, Any], cache_done: Callable[..., Any]) -> Tab:
    """Build the settings tab; application assembly wires settings-layer changes."""
    with gr.Tab("Настройки"):
        note = gr.Markdown(seasons.NO_SEASON_NOTE)
        with gr.Row(equal_height=False):
            global_box = gr.Textbox(label="Общие настройки (файл)", value=str(season_settings.global_path()),
                                    info="По умолчанию stt_config.local.json в папке программы; можно выбрать другой файл "
                                         "(если его нет, он создаётся как копия нынешних общих настроек)", scale=6)
            pick_global = gr.Button("📁 Выбрать…", scale=1, min_width=140)
            use_global = gr.Button("Использовать этот файл", scale=1, min_width=200)
        gr.Markdown("Каждый сезон сначала берёт общие настройки. Настройки сезона хранятся в папке его результатов "
                    "и действуют только для него.")
        global_pick_message = gr.Markdown()
        global_browser = folder_pick.build_browser("Выбор файла общих настроек")
        effect = gr.Markdown(effect_text(None))
        target = gr.Radio(label="Сохранять изменения в", choices=target_choices(None), value=TARGET_GLOBAL)
        gr.Markdown("Показаны значения, которые действуют для выбранного сезона. Сохраняются только отличия от "
                    "`stt_config.json` (прежняя версия файла: `.bak`); в настройки сезона записываются все значения.")
        by_path: dict[tuple[str, ...], Any] = {}
        values = dict(zip((s.path for s in settings.SETTINGS), settings.form_values(config)))
        for group in dict.fromkeys(s.group for s in settings.SETTINGS):
            gr.Markdown(f"### {group}")
            _settings_group(group, values, by_path)
        fields = [by_path[s.path] for s in settings.SETTINGS]
        with gr.Row():
            save_button = gr.Button("Сохранить", variant="primary")
            reset_button = gr.Button("Сбросить к стандартным")
            remove_button = gr.Button("Убрать настройки сезона (вернуть общие)", visible=False)
        settings_message = gr.Markdown()
        gr.Markdown("### Кэш озвучки\nКаждая озвученная реплика хранится в `output/tts_cache`: повторный "
                    "запуск берёт её оттуда. Реплики, которые не нужны ни одной серии, можно удалить.")
        with gr.Row():
            cache_check = gr.Button("Сколько освободится")
            cache_clean = gr.Button("Очистить кэш озвучки", variant="stop")
        cache_message = gr.Markdown()
        cache_check.click(lambda: wait_state("Считаю…", "Считаю размер кэша — до минуты…"), None,
                          [cache_check, cache_message], queue=False).then(
            lambda: cache_done(False, "Сколько освободится"), None, [cache_check, cache_message])
        cache_clean.click(lambda: wait_state("Очищаю…", "Удаляю ненужное из кэша — до минуты…"), None,
                          [cache_clean, cache_message], queue=False).then(
            lambda: cache_done(True, "Очистить кэш озвучки"), None, [cache_clean, cache_message])
    return Tab(note, fields, effect, target, remove_button, global_box, pick_global, use_global,
               global_browser, global_pick_message, save_button, reset_button, settings_message)
