"""Minimal local Gradio UI for the native pipeline editor.

Flow: create/open a project, import a native batch, review and edit one cue at a
time, explicitly approve text/timing, run the local audio queue only on demand,
then export the full draft SRT + JSON (the mixed voice-over is a separate file).

``utils.pipeline_editor`` is imported lazily; tests inject a fake backend via
``use_backend_for_tests``. No LLM runs here, TTS is consumed only by the explicit
button, analytics is off, and the server stays on 127.0.0.1.

Run (from repo root, with subs_env activated):
    uv run -m ui.native_editor
"""

from __future__ import annotations

import functools
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import gradio as gr

FIT_STATUSES = {"fits", "accelerated_fits"}
DEFAULT_CONFIG = "pipeline_config_vosk.json"
TABLE_HEADERS = ["Индекс", "Начало, мс", "Конец, мс", "Текст", "Статус", "Проверка", "Причина",
                 "Аудио, мс", "Окно, мс", "Аудио"]
BANNER = ("**Сокращения импортируются из native batch; здесь правки, проверка аудио и экспорт.** "
          "Модель перевода/сокращения автоматически не вызывается. "
          "Локальная озвучка (TTS) запускается только кнопкой «Озвучить сейчас».")
LEGEND = (
    "**Легенда статусов** (в таблице — исходные значения): `fits`/`accelerated_fits` — уместилась (после ускорения); "
    "`unresolved` — не уместилась; `pending_audio`/`needs_review`/`pending_recheck`/`pending_retry`/`needs_reconciliation` — "
    "ожидает аудио или проверки; `normalization_failed`/`tts_failed`/`audio_failed`/`audio_invalid` — сбои и невалидное аудио; "
    "`runtime_changed` — изменилось runtime-окружение. `review_status`: `pending` — ожидает проверки, "
    "`native_approved` — одобрено пайплайном, `human_approved` — одобрено человеком. Неизвестные значения показываются "
    "английским кодом. Подтверждение человека — это текст/тайминг, а не факт прослушивания; экспорт черновика не означает готовность."
)
RELOAD_HINT = ("Нажмите «Открыть», чтобы перезагрузить проект: правки не применялись и не будут "
               "объединены автоматически.")


class UiValidationError(ValueError):
    """Local form validation failure; raised before any backend call is made."""


_backend_override: Any | None = None


def use_backend_for_tests(backend: Any | None) -> None:
    """Inject a fake backend for unit tests; pass None to restore the lazy import."""
    global _backend_override
    _backend_override = backend


def get_backend() -> Any:
    """Return the editor backend module; imported lazily so the UI starts without it."""
    if _backend_override is not None:
        return _backend_override
    try:
        from utils import pipeline_editor
    except ImportError as exc:
        raise RuntimeError(f"Backend utils.pipeline_editor недоступен: {exc}") from exc
    return pipeline_editor


def project_cues(project: Any) -> list[dict[str, Any]]:
    return [cue for cue in (project or {}).get("cues", []) if isinstance(cue, dict)]


def cue_needs_attention(cue: dict[str, Any]) -> bool:
    """Prefer the backend flag; otherwise use the documented status/review contract."""
    if "needs_human" in cue:
        return bool(cue["needs_human"])
    return cue.get("status") not in FIT_STATUSES or cue.get("review_status") == "pending"


def filtered_cues(project: Any, only_attention: bool) -> list[dict[str, Any]]:
    cues = project_cues(project)
    return [cue for cue in cues if cue_needs_attention(cue)] if only_attention else cues


def find_cue(project: Any, cue_id: str | None) -> dict[str, Any] | None:
    if not cue_id:
        return None
    return next((cue for cue in project_cues(project) if cue.get("cue_id") == cue_id), None)


def cue_label(cue: dict[str, Any]) -> str:
    text = " ".join(str(cue.get("text", "")).split())
    prefix = text[:60] + ("…" if len(text) > 60 else "")
    return f"{cue.get('index')}. {prefix}" if prefix else f"{cue.get('index')}."


def cue_choices(cues: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(cue_label(cue), cue["cue_id"]) for cue in cues if cue.get("cue_id")]


def cue_rows(cues: list[dict[str, Any]]) -> list[list[Any]]:
    def shown(value: Any) -> Any:
        return "—" if value is None else value

    return [[cue.get("index"), cue.get("start"), cue.get("end"), cue.get("text", ""), cue.get("status"),
             cue.get("review_status"), cue.get("reason") or "", shown(cue.get("duration_ms")),
             shown(cue.get("available_ms")), "есть" if cue.get("audio_path") else "—"] for cue in cues]


def summary_text(project: Any) -> str:
    cues = project_cues(project)
    attention = sum(1 for cue in cues if cue_needs_attention(cue))
    with_audio = sum(1 for cue in cues if cue.get("audio_path"))
    failed = sum(1 for cue in cues if "failed" in str(cue.get("status")) or
                 cue.get("status") in {"unresolved", "audio_invalid"})
    ready = (project or {}).get("ready_for_mix")
    lines = [f"**Сводка:** всего {len(cues)}; требуют внимания {attention}; аудио есть {with_audio}; "
             f"аудио нет {len(cues) - with_audio}; сбоев {failed}; "
             f"готово к микс-сборке: {'да' if ready is True else 'нет' if ready is False else 'неизвестно'}."]
    if (project or {}).get("runtime_valid") is False:
        lines.append("⚠️ runtime/модель изменились или недоступны; нужен новый проект.")
    lines.append("Модель перевода/сокращения автоматически не вызывается; TTS расходуется только кнопкой «Озвучить сейчас».")
    return "\n\n".join(lines)


def cue_info(cue: dict[str, Any]) -> str:
    parts = [f"Кью #{cue.get('index')} · статус: {cue.get('status')} · проверка: {cue.get('review_status')}"]
    if cue.get("reason"):
        parts.append(f"причина: {cue['reason']}")
    if cue.get("available_ms") is not None:
        parts.append(f"окно: {cue['available_ms']} мс")
    if cue.get("duration_ms") is not None:
        parts.append(f"аудио: {cue['duration_ms']} мс")
    return " | ".join(parts)


def _form_values(cue: dict[str, Any] | None) -> tuple[Any, Any, Any, Any, Any]:
    if not cue:
        return "", None, None, "", gr.update(value=None)
    audio = cue.get("audio_path")
    audio_update = gr.update(value=str(audio)) if audio and Path(str(audio)).is_file() else gr.update(value=None)
    speaker = cue.get("speaker_id")
    return (str(cue.get("text", "")), int(cue.get("start") or 0), int(cue.get("end") or 0),
            str(speaker) if speaker else "", audio_update)


def _apply_project(state: dict[str, Any], project: dict[str, Any], message: str,
                   *, keep_cue: bool = True) -> tuple[Any, ...]:
    """Render the whole main editor output block for a fresh project dict."""
    visible = filtered_cues(project, bool(state.get("only_attention", True)))
    visible_ids = {item.get("cue_id") for item in visible}
    cue = find_cue(project, state.get("cue_id")) if keep_cue and state.get("cue_id") in visible_ids else None
    new_state = {**state, "project": project, "revision": project.get("revision"),
                 "cue_id": cue["cue_id"] if cue else None,
                 "visible_ids": [item.get("cue_id") for item in visible]}
    selector = gr.update(choices=cue_choices(visible), value=cue["cue_id"] if cue else None)
    return (new_state, message, cue_rows(visible), selector, summary_text(project), *_form_values(cue))


def _noop_state(state: dict[str, Any], message: str) -> tuple[Any, ...]:
    """Keep every widget exactly as the user left it (load errors must not wipe the form)."""
    return (state, message, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update())


def _fail_message(state: dict[str, Any], exc: Exception) -> str:
    message = f"Ошибка: {exc}"
    if isinstance(exc, UiValidationError):
        return message
    folder = state.get("folder")
    if folder:
        try:
            current = get_backend().load_project(Path(folder))
            if current.get("revision") != state.get("revision"):
                message += (f" Похоже на конфликт версий: в проекте «{current.get('revision')}», "
                            f"а у вас «{state.get('revision')}». {RELOAD_HINT}")
        except Exception:
            pass
    return message


def _revision_guard(backend: Any, folder: str, state: dict[str, Any]) -> str | None:
    """Gate contract calls that carry no expected_revision (adopt_batch, process_pending)."""
    if state.get("folder") != folder or not state.get("revision"):
        return None
    try:
        current = backend.load_project(Path(folder))
    except Exception as exc:
        return f"Ошибка: не удалось проверить версию проекта: {exc}"
    if current.get("revision") != state.get("revision"):
        return (f"Проект изменён (revision «{state.get('revision')}» → «{current.get('revision')}»). "
                f"{RELOAD_HINT}")
    return None


def _ui_endpoint(fn: Callable[..., tuple[Any, ...]]) -> Callable[..., tuple[Any, ...]]:
    """Wrap a main editor callback: errors keep the state and the user's unsaved form."""
    @functools.wraps(fn)
    def wrapper(*args: Any) -> tuple[Any, ...]:
        state = args[-1] if args and isinstance(args[-1], dict) else {}
        try:
            return fn(*args)
        except Exception as exc:
            return _noop_state(state, _fail_message(state, exc))
    return wrapper


def _edit_payload(state: dict[str, Any], text: Any, start: Any, end: Any,
                  speaker: Any) -> tuple[Path, dict[str, Any]]:
    folder, cue_id = state.get("folder"), state.get("cue_id")
    if not folder:
        raise UiValidationError("Сначала создайте или откройте проект.")
    if not cue_id:
        raise UiValidationError("Выберите кью в списке — правка не применялась.")
    text_value = text or ""
    if not text_value.strip():
        raise UiValidationError("Текст не может быть пустым — правка не применялась.")
    try:
        start_ms, end_ms = int(start), int(end)
    except (TypeError, ValueError) as exc:
        raise UiValidationError("Начало и конец должны быть числами в миллисекундах.") from exc
    if start_ms < 0 or end_ms <= start_ms:
        raise UiValidationError("Проверьте тайминг: конец должен быть больше начала (мс).")
    return Path(folder), {"cue_id": cue_id, "text": text_value, "start": start_ms, "end": end_ms,
                          "speaker_id": (speaker or "").strip() or None}


@_ui_endpoint
def on_create(source_file: Any, source_path: Any, folder: Any, config: Any,
              state: dict[str, Any]) -> tuple[Any, ...]:
    raw_source = str(source_path).strip() if source_path else ""
    source = Path(str(source_file)) if source_file else (Path(raw_source) if raw_source else None)
    if source is None:
        raise UiValidationError("Укажите исходный SRT/JSON: загрузите файл или впишите путь.")
    target = str(folder or "").strip()
    if not target:
        raise UiValidationError("Укажите папку нового проекта.")
    config_path = Path(str(config).strip() or DEFAULT_CONFIG)
    project = get_backend().create_project(source, Path(target), config_path)
    new_state = {**state, "folder": target, "cue_id": None}
    return _apply_project(new_state, project, f"Проект создан: {target} (revision {project.get('revision')}).",
                          keep_cue=False)


@_ui_endpoint
def on_open(folder: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    target = str(folder or "").strip()
    if not target:
        raise UiValidationError("Укажите путь к проекту.")
    project = get_backend().load_project(Path(target))
    message = f"Проект открыт: {target} (revision {project.get('revision')})."
    if state.get("cue_id"):
        message += " Несохранённые правки формы сброшены."
    return _apply_project({**state, "folder": target, "cue_id": None}, project, message, keep_cue=False)


@_ui_endpoint
def on_adopt(folder: Any, batch_path: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    target, batch = str(folder or "").strip(), str(batch_path or "").strip()
    if not target or not batch:
        raise UiValidationError("Укажите папку проекта и папку native batch.")
    backend = get_backend()
    guard = _revision_guard(backend, target, state)
    if guard:
        return _noop_state(state, guard)
    project = backend.adopt_batch(Path(target), Path(batch))
    return _apply_project({**state, "folder": target, "cue_id": None}, project,
                          "Batch импортирован (strict native validation).", keep_cue=False)


@_ui_endpoint
def on_import_srt(srt_file: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    folder = state.get("folder")
    if not folder:
        raise UiValidationError("Сначала создайте или откройте проект.")
    if not srt_file:
        raise UiValidationError("Загрузите отредактированный SRT.")
    project = get_backend().import_srt(Path(folder), Path(str(srt_file)),
                                       expected_revision=state.get("revision"))
    return _apply_project(state, project,
                          "SRT объединён с JSON-состоянием проекта на backend; текст/тайминги обновлены.",
                          keep_cue=True)


@_ui_endpoint
def on_filter(only_attention: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    project = state.get("project")
    new_state = {**state, "only_attention": bool(only_attention)}
    if not project:
        return _noop_state(new_state, "Фильтр переключён. Проект ещё не открыт.")
    visible = {cue.get("cue_id") for cue in filtered_cues(project, new_state["only_attention"])}
    hidden = bool(state.get("cue_id")) and state["cue_id"] not in visible
    message = ("Фильтр обновлён; выбранный кью скрыт, выбор сброшен, правки не применялись."
               if hidden else "Фильтр обновлён.")
    return _apply_project(new_state, project, message, keep_cue=not hidden)


@_ui_endpoint
def on_select(cue_id: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    cue = find_cue(state.get("project"), str(cue_id) if cue_id else None)
    visible = set(state.get("visible_ids") or [])
    if cue is None or cue["cue_id"] not in visible:
        return _noop_state(state, "Кью недоступен в текущем фильтре (скрыт или проект изменился): "
                                  "выбор не изменён. Обновите фильтр или переоткройте проект.")
    return ({**state, "cue_id": cue["cue_id"]}, cue_info(cue), gr.update(), gr.update(value=cue["cue_id"]),
            gr.update(), *_form_values(cue))


@_ui_endpoint
def on_save(text: Any, start: Any, end: Any, speaker: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    folder, edit = _edit_payload(state, text, start, end, speaker)
    project = get_backend().apply_edits(folder, [edit], expected_revision=state.get("revision"))
    return _apply_project(state, project, f"Правка сохранена (revision {project.get('revision')}).", keep_cue=True)


@_ui_endpoint
def on_save_approve(text: Any, start: Any, end: Any, speaker: Any,
                    state: dict[str, Any]) -> tuple[Any, ...]:
    folder, edit = _edit_payload(state, text, start, end, speaker)
    backend = get_backend()
    project = backend.apply_edits(folder, [edit], expected_revision=state.get("revision"))
    try:
        approved = backend.approve_cues(folder, [edit["cue_id"]], expected_revision=project.get("revision"))
    except Exception as exc:
        return _apply_project(state, project,
                              f"Правка сохранена (revision {project.get('revision')}), но подтверждение "
                              f"не выполнено: {exc}", keep_cue=True)
    return _apply_project(state, approved,
                          "Правка сохранена и кью подтверждён по уже сохранённой версии.", keep_cue=True)


@_ui_endpoint
def on_approve(state: dict[str, Any]) -> tuple[Any, ...]:
    folder, cue_id = state.get("folder"), state.get("cue_id")
    if not folder:
        raise UiValidationError("Сначала создайте или откройте проект.")
    if not cue_id:
        raise UiValidationError("Выберите кью для подтверждения.")
    project = get_backend().approve_cues(Path(folder), [cue_id], expected_revision=state.get("revision"))
    return _apply_project(state, project, "Выбранный кью подтверждён (текст/тайминг, проверка человеком).",
                          keep_cue=True)


@_ui_endpoint
def on_approve_all(confirmed: Any, state: dict[str, Any]) -> tuple[Any, ...]:
    if not confirmed:
        return _noop_state(state, "Массовое подтверждение не выполнено: поставьте галочку "
                                  "«Подтверждаю текст и тайминги всех требующих внимания».")
    project, folder = state.get("project"), state.get("folder")
    if not project or not folder:
        raise UiValidationError("Сначала создайте или откройте проект.")
    cue_ids = [cue["cue_id"] for cue in filtered_cues(project, True) if cue.get("cue_id")]
    if not cue_ids:
        return _apply_project(state, project, "Нет кью, требующих внимания: подтверждать нечего.", keep_cue=True)
    updated = get_backend().approve_cues(Path(folder), cue_ids, expected_revision=state.get("revision"))
    return _apply_project(state, updated,
                          f"Подтверждено человеком кью (текст/тайминг): {len(cue_ids)}.", keep_cue=True)


def on_clear_text(text: Any) -> tuple[str, str]:
    """Local-only reset: TTS is never consumed here, only by the explicit queue button."""
    return "", ("Поле текста очищено локально. Озвучка (TTS) не запускалась; она расходуется только "
                "кнопкой «Озвучить сейчас».")


def on_process(state: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
    """Explicit local TTS queue; the backend call is one long task, so we only report status."""
    yield _noop_state(state, "Запущена локальная озвучка (TTS) подтверждённых кью. Это может занять время.")
    try:
        folder = state.get("folder")
        if not folder:
            yield _noop_state(state, "Сначала создайте или откройте проект.")
            return
        backend = get_backend()
        guard = _revision_guard(backend, folder, state)
        if guard:
            yield _noop_state(state, guard)
            return
        project = backend.process_pending(Path(folder))
        yield _apply_project(state, project,
                             "Локальная озвучка обработана: попытки добавлены только для подтверждённых кью.",
                             keep_cue=True)
    except Exception as exc:
        yield _noop_state(state, _fail_message(state, exc))


def on_export(state: dict[str, Any]) -> tuple[str, Any]:
    try:
        folder = state.get("folder")
        if not folder:
            raise UiValidationError("Сначала создайте или откройте проект.")
        result = get_backend().export_project(Path(folder))
        files = [str(result["srt_path"]), str(result["json_path"])]
        return ("Экспорт не означает готовность: это полный черновик SRT и JSON со всеми статусами. "
                "Для финальной озвучки используйте отдельную кнопку «Собрать озвучку (mix)»."), gr.update(value=files)
    except Exception as exc:
        return f"Ошибка экспорта: {exc}", gr.update(value=None)


def on_mix(state: dict[str, Any]) -> tuple[str, Any]:
    try:
        folder = state.get("folder")
        if not folder:
            raise UiValidationError("Сначала создайте или откройте проект.")
        if (state.get("project") or {}).get("ready_for_mix") is False:
            raise UiValidationError("не все кью подтверждены и измерены; смотрите сводку и статусы.")
        backend = get_backend()
        guard = _revision_guard(backend, folder, state)
        if guard:
            return guard, gr.update(value=None)
        path = backend.mix_ready(Path(folder))
        return f"Озвучка собрана в отдельный файл: {path}", gr.update(value=str(path))
    except Exception as exc:
        return f"Ошибка: озвучка не собрана. {exc}", gr.update(value=None)


def build_demo() -> gr.Blocks:
    """Build the installed-Gradio Blocks app; no server, no backend import, no inference."""
    with gr.Blocks(title="Native Editor (локально)") as demo:
        gr.Markdown(BANNER)
        state = gr.State({})
        status = gr.Markdown("Откройте или создайте проект.")
        summary = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Проект")
                source_file = gr.File(label="Исходный файл (SRT или JSON)", file_types=[".srt", ".json"],
                                      type="filepath")
                source_path = gr.Textbox(label="…или путь к исходному файлу")
                folder_new = gr.Textbox(label="Папка нового проекта")
                config = gr.Textbox(label="Конфиг", value=DEFAULT_CONFIG)
                btn_create = gr.Button("Создать проект")
                project_folder = gr.Textbox(label="Путь к проекту (открытие, batch)")
                btn_open = gr.Button("Открыть проект")
                batch_path = gr.Textbox(label="Папка native batch")
                btn_adopt = gr.Button("Импортировать batch")
                srt_file = gr.File(label="Отредактированный SRT", file_types=[".srt"], type="filepath")
                btn_import_srt = gr.Button("Импортировать SRT в проект")
            with gr.Column():
                gr.Markdown("### Правка")
                only_attention = gr.Checkbox(label="Только требующие внимания", value=True)
                table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, wrap=True)
                selector = gr.Dropdown(label="Кью (индекс — текст)", choices=[], value=None)
                text = gr.Textbox(label="Текст", lines=3)
                with gr.Row():
                    start = gr.Number(label="Начало, мс", precision=0)
                    end = gr.Number(label="Конец, мс", precision=0)
                speaker = gr.Textbox(label="Говорящий (ручная метка, не голос TTS)")
                audio = gr.Audio(label="Аудио кью (если уже есть)", type="filepath", interactive=False)
                with gr.Row():
                    btn_save = gr.Button("Сохранить правку", variant="primary")
                    btn_save_approve = gr.Button("Сохранить и подтвердить")
                    btn_clear = gr.Button("Очистить поле текста")
                with gr.Row():
                    btn_approve = gr.Button("Подтвердить выбранное")
                    confirm_all = gr.Checkbox(label="Подтверждаю текст и тайминги всех требующих внимания",
                                              value=False)
                    btn_approve_all = gr.Button("Подтвердить все требующие внимания")
        with gr.Row():
            btn_process = gr.Button("Озвучить сейчас (локальная TTS, только по кнопке)", variant="stop")
            btn_export = gr.Button("Экспортировать черновик (SRT + JSON)")
            btn_mix = gr.Button("Собрать озвучку (mix)")
        with gr.Row():
            exports = gr.Files(label="Экспорт: полный черновик SRT + JSON")
            mix_file = gr.File(label="Файл озвучки (mix)")
        gr.Markdown(LEGEND)

        # One output block for all project-changing handlers; the tuple length must match it exactly.
        # ``.input`` (human-only) prevents programmatic selector refreshes from re-entering on_select.
        main_out = [state, status, table, selector, summary, text, start, end, speaker, audio]
        btn_create.click(on_create, [source_file, source_path, folder_new, config, state], main_out)
        btn_open.click(on_open, [project_folder, state], main_out)
        btn_adopt.click(on_adopt, [project_folder, batch_path, state], main_out)
        btn_import_srt.click(on_import_srt, [srt_file, state], main_out)
        selector.input(on_select, [selector, state], main_out)
        only_attention.input(on_filter, [only_attention, state], main_out)
        btn_save.click(on_save, [text, start, end, speaker, state], main_out)
        btn_save_approve.click(on_save_approve, [text, start, end, speaker, state], main_out)
        btn_approve.click(on_approve, [state], main_out)
        btn_approve_all.click(on_approve_all, [confirm_all, state], main_out)
        btn_clear.click(on_clear_text, [text], [text, status])
        btn_process.click(on_process, [state], main_out)
        btn_export.click(on_export, [state], [status, exports])
        btn_mix.click(on_mix, [state], [status, mix_file])
    return demo.queue()


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", share=False, inbrowser=False)


if __name__ == "__main__":
    main()
