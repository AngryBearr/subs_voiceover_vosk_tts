"""The episode-processing tab and its local list controls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from ui import seasons
from ui.app_forms import PREPARE_HELP
from ui.app_paths import ENGINES, MODES, MODELS, NO_SEASON, TRANSPORTS


@dataclass
class Tab:
    note: Any
    episodes: Any
    season_status: Any
    form: list[Any]
    voice_model: Any
    log: Any


def build(folder: Any, out_root: Any, strip: Any, strip_stop: Any, *, personal: dict[str, Any],
          start_values: list[Any], presets: list[tuple[str, str]], model_missing: bool,
          preset_form: Callable[..., Any], start_with_preset: Callable[..., str], prepare_season: Callable[..., str],
          stop_job: Callable[..., str], resume_job: Callable[..., str], refresh: Callable[..., Any],
          strip_view: Callable[..., Any], season_table: Callable[..., Any]) -> Tab:
    """Build processing controls and expose the form and season-list components."""
    with gr.Tab("Обработка"):
        note = gr.Markdown(seasons.NO_SEASON_NOTE)
        show = gr.Button("Обновить список серий")
        episodes = gr.CheckboxGroup(label="Серии сезона", choices=[])
        with gr.Row():
            select_all = gr.Button("Выбрать все", size="sm")
            select_none = gr.Button("Снять выбор", size="sm")
        season_status = gr.Dataframe(headers=["Серия", "Состояние", "Субтитры", "Готовое видео"], label="Серии сезона: состояние",
                                     interactive=False, wrap=True)
        preset = gr.Radio(label="Набор настроек (ниже можно поправить отдельные пункты)", choices=presets, value="main")
        with gr.Row():
            engines = gr.CheckboxGroup(label="Озвучка", choices=list(ENGINES), value=start_values[0])
            mode = gr.Radio(label="Распознавание", choices=list(MODES), value=start_values[1])
            with gr.Column():
                model = gr.Dropdown(label="Языковая модель", choices=list(MODELS), value=start_values[2])
                transport = gr.Radio(label="Запросы к модели", choices=list(TRANSPORTS),
                                     value=list(TRANSPORTS)[0], visible=bool(personal.get("llm_transport_choice")))
        with gr.Row():
            subtitles = gr.Checkbox(label="Вшить субтитры", value=start_values[3])
            duck_voices = gr.Checkbox(label="Приглушать только голоса (музыку оставить)", value=start_values[4])
            stress = gr.Checkbox(label="Проверять ударения", value=start_values[5])
            refine = gr.Checkbox(label="Уточнять спикеров по голосу", value=start_values[6])
            f5_fast = gr.Checkbox(label="ESpeech: быстрый режим (7 шагов)", value=start_values[7])
        voice_model = gr.Checkbox(label="Скачать дообученную модель голосов перед обработкой (115 МБ, один раз)",
                                  info="Точнее определяет, кто говорит. Без неё берётся обычная модель.",
                                  value=True, visible=model_missing)
        with gr.Accordion("Пробный отрывок (быстро проверить настройки)", open=False):
            preview = gr.Checkbox(label="Только отрывок (в папку _проба)", value=False)
            with gr.Row():
                preview_from = gr.Number(label="С минуты", value=10)
                preview_minutes = gr.Number(label="Минут", value=2)
        with gr.Row():
            start = gr.Button("Начать", variant="primary")
            prepare = gr.Button("Подготовить весь сезон")
            resume = gr.Button("▶ Продолжить", interactive=False)
            stop = gr.Button("Остановить", variant="stop")
        gr.Markdown(PREPARE_HELP)
        message = gr.Markdown()
        status = gr.Markdown("Нет запущенной обработки.")
        progress = gr.Dataframe(headers=["Серия"], label="Ход обработки по этапам", interactive=False, wrap=True)
        checkpoints = gr.Markdown(label="Контрольные точки")
        log = gr.Textbox(label="Ход работы", lines=12, max_lines=12, autoscroll=True, interactive=False)
        results = gr.Dataframe(headers=["Серия", "Движок", "Файл"], label="Готовые файлы", interactive=False,
                               wrap=True)
        with gr.Accordion("Подробный лог", open=False):
            raw_log = gr.Textbox(label="Лог", lines=12, max_lines=12, autoscroll=True, interactive=False)

        def rescan(f: str) -> tuple[Any, str]:
            if not f:
                return gr.update(), NO_SEASON
            names = seasons.video_names(Path(f).expanduser())
            return gr.update(choices=seasons.episode_choices(names), value=[]), (f"Найдено серий: {len(names)}." if names
                                                        else f"⚠ В папке {f} нет видео или она недоступна.")

        def pick_all(f: str) -> tuple[Any, str]:
            names = seasons.video_names(Path(f).expanduser()) if f else []
            return gr.update(value=names), (f"Выбрано серий: {len(names)}." if names else NO_SEASON)

        show.click(rescan, folder, [episodes, message])
        select_all.click(pick_all, folder, [episodes, message])
        select_none.click(lambda: (gr.update(value=[]), "Выбор снят."), None, [episodes, message])
        preset.change(preset_form, [preset, out_root], [engines, mode, subtitles, duck_voices, stress, refine, f5_fast])
        start.click(start_with_preset, [folder, episodes, out_root, engines, mode, model, subtitles, duck_voices, stress,
                                        refine, preview, preview_from, preview_minutes, f5_fast, transport, voice_model, preset],
                    message)
        prepare.click(prepare_season, [folder, episodes, out_root, engines, mode, model, subtitles, duck_voices, stress,
                                       refine, preview, preview_from, preview_minutes, f5_fast, transport, voice_model, preset],
                      message)
        stop.click(stop_job, None, message)
        resume.click(resume_job, None, message)
        gr.Timer(2.0).tick(refresh, None, [status, progress, log, results, checkpoints, raw_log, resume])
        gr.Timer(2.0).tick(strip_view, None, [strip, strip_stop], show_progress="hidden")
        strip_stop.click(stop_job, None, strip)
        gr.Timer(2.0).tick(season_table, [folder, out_root], season_status, show_progress="hidden")
    return Tab(note, episodes, season_status, [engines, mode, model, subtitles, duck_voices, stress, refine, f5_fast],
               voice_model, log)
