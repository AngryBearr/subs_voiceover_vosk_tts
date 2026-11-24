"""Season selection helpers and the shared season/status bar."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from ui import folder_pick, season_settings, seasons
from ui.app_paths import _root
from ui.tabs.settings import season_line_text


def season_picker(selected: seasons.Season | None = None, *,
                  base_config: Callable[[], dict[str, Any]]) -> Any:
    """The picker's choices and value: the remembered seasons (or the ones found next to the configured folders)."""
    found = seasons.available(base_config())
    current = selected or seasons.pick_current(found)
    if current is not None and current.source not in {s.source for s in found}:
        found.append(current)
    return gr.update(choices=seasons.choices(found), value=str(current.source) if current else None)


def select_season(source: str | None, previous: str = "", *,
                  base_config: Callable[[], dict[str, Any]]) -> tuple[Any, ...]:
    """Select a season and return its lists, summary and settings-copy question."""
    season = seasons.choose(source, base_config()) if source else None
    no_question = (gr.update(visible=False), "", gr.update(), None)
    if season is None:
        return "", "", seasons.summary(None), gr.update(choices=[], value=[]), "", [], *no_question
    names = seasons.video_names(season.source)
    if not season.source.is_dir():
        note = f"⚠ Папка с оригиналами {season.source} недоступна (диск не подключён?)."
    elif not names:
        note = f"⚠ В папке {season.source} нет видео."
    else:
        note = f"Сезон «{season.name}»: {seasons.episodes_word(len(names))}."
    question = no_question
    before = _root(previous) if previous else None
    if season_settings.should_ask(before, season.results, base_config()):
        assert before is not None
        previous_name = _season_name(before)
        question = (gr.update(visible=True), f"**У сезона {season.name} нет своих настроек. Какие взять?**",
                    gr.update(value=f"Настройки сезона {previous_name}"), [str(before), str(season.results)])
    return (str(season.source), str(season.results), season_line_text(season),
            gr.update(choices=seasons.episode_choices(names), value=[]), note, seasons.table(season), *question)


def _season_name(results: Path) -> str:
    """The name of the season whose results folder is ``results`` (from the remembered seasons, else the folder name)."""
    found = next((s for s in seasons.remembered() if s.results == results), None)
    if found is not None:
        return found.name
    return results.name[len(seasons.RESULTS_PREFIX):] if results.name.startswith(seasons.RESULTS_PREFIX) else results.name


def answer_settings_question(pending: list[str] | None, copy: bool) -> tuple[Any, str]:
    """The answer to "Какие взять?": copy the previous season's file into the new season's folder, or leave the
    global settings in effect. Returns (the panel, the message)."""
    hidden = gr.update(visible=False)
    if not pending or len(pending) != 2:
        return hidden, ""
    before, new = Path(pending[0]), Path(pending[1])
    name = _season_name(new)
    if not copy:
        return hidden, f"Для сезона {name} действуют общие настройки."
    try:
        path = season_settings.copy_own(before, new)
    except OSError as error:
        return hidden, f"❌ Не получилось скопировать настройки: {error}. Действуют общие настройки."
    return hidden, f"Для сезона {name} действуют настройки сезона {_season_name(before)}, скопированные в `{path}`."


def season_table(source: str, results: str) -> list[list[str]]:
    """The state of every episode of the season (the timer calls this: folder listings only)."""
    if not source or not results:
        return []
    return seasons.table(seasons.Season(Path(source), Path(results)))


def source_changed(text: str) -> tuple[str, Any]:
    """The folder of the new season was typed: what was found there, and the suggested results folder."""
    ok, note = seasons.describe_source(text)
    return note, (gr.update(value=str(seasons.suggest_results(text))) if ok else gr.update())


def results_changed(source: str, results: str) -> str:
    return seasons.describe_results(source, results)[1]


def save_season(source: str, results: str, *, base_config: Callable[[], dict[str, Any]],
                season_picker: Callable[..., Any]) -> tuple[str, Any, Any]:
    """"Сохранить сезон": check both folders, create the results folder, remember and select the season."""
    season, problem = seasons.validate(source, results)
    if season is None:
        return problem, gr.update(), gr.update()
    try:
        season.results.mkdir(parents=True, exist_ok=True)
        seasons.remember(season, base_config())
    except OSError as error:
        return f"❌ Не получилось сохранить: {error}", gr.update(), gr.update()
    return f"✅ Сезон «{season.name}» сохранён и выбран.", season_picker(season), gr.update(visible=False)


@dataclass
class Bar:
    folder: Any
    out_root: Any
    picker: Any
    line: Any
    ask_panel: Any
    ask_text: Any
    ask_copy: Any
    ask_global: Any
    pending: Any
    message: Any
    strip: Any
    stop: Any
    wire: Callable[[], None]


def build(status: str, running: bool, save_season: Callable[..., Any]) -> Bar:
    """Build the season bar and its local folder-picker controls."""
    folder = gr.State("")
    out_root = gr.State("")
    with gr.Row(equal_height=False):
        season_pick = gr.Dropdown(label="Сезон", choices=[], value=None, scale=5,
                                  info="Один сезон для всех вкладок: папка с оригиналами и папка результатов")
        new_season = gr.Button("＋ Новый сезон", scale=1, min_width=160)
    season_line = gr.Markdown(seasons.summary(None))
    with gr.Column(visible=False) as season_form:
        with gr.Row(equal_height=False):
            new_source = gr.Textbox(label="Папка с оригинальными сериями", placeholder="/media/.../Survivor/Season_48",
                                    scale=6)
            pick_source = gr.Button("📁 Выбрать…", scale=1, min_width=140)
        source_note = gr.Markdown()
        with gr.Row(equal_height=False):
            new_results = gr.Textbox(label="Папка результатов (можно изменить)",
                                     info="Здесь будет всё: папки серий, файлы сезона, готовые видео. "
                                          "Если папки нет, она создаётся при «Сохранить сезон»",
                                     scale=6)
            pick_results = gr.Button("📁 Выбрать…", scale=1, min_width=140)
        results_note = gr.Markdown()
        pick_message = gr.Markdown()
        season_browser = folder_pick.build_browser()
        with gr.Row():
            season_save = gr.Button("Сохранить сезон", variant="primary")
            season_cancel = gr.Button("Отмена")
    with gr.Column(visible=False) as ask_panel:
        ask_text = gr.Markdown()
        with gr.Row():
            ask_copy = gr.Button("Настройки сезона", variant="primary")
            ask_global = gr.Button("Общие настройки")
    pending = gr.State(None)
    season_message = gr.Markdown()
    with gr.Row(equal_height=True):
        strip = gr.Markdown(status, elem_id="task-strip")
        strip_stop = gr.Button("Остановить", variant="stop", size="sm", scale=0, min_width=140, visible=running)

    def wire() -> None:
        """Register local events at the original position in application assembly."""
        new_season.click(lambda: (gr.update(visible=True), "",
                                  "Укажите папку с оригинальными сериями, например /media/…/Survivor/Season_49."),
                         None, [season_form, season_message, source_note])
        season_cancel.click(lambda: (gr.update(visible=False), "Сезон не добавлен."), None, [season_form, season_message])
        new_source.change(source_changed, new_source, [source_note, new_results], show_progress="hidden")
        new_results.change(results_changed, [new_source, new_results], results_note, show_progress="hidden")
        season_save.click(save_season, [new_source, new_results], [season_message, season_pick, season_form])
        folder_pick.wire_browser(season_browser, {"source": new_source, "results": new_results},
                                 {"source": (pick_source, folder_pick.FOLDER, "Папка с оригинальными сериями"),
                                  "results": (pick_results, folder_pick.FOLDER, "Папка результатов")}, pick_message)

    return Bar(folder, out_root, season_pick, season_line, ask_panel, ask_text, ask_copy, ask_global, pending,
               season_message, strip, strip_stop, wire)
