"""Subtitle editing helpers and the check-and-edit tab."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from ui import seasons
from ui.app_paths import ENGINES, _episode, _root, _season_root

SHOW = ["Все реплики", "Для проверки", "С правками", "Не влезают по времени"]
GENDER_RU = {"Мужской": "male", "Женский": "female"}
FIT_MARK = {"ok": "", "fast": "быстро", "limit": "на пределе", "over": "не влезет"}


def list_episodes(out_root: str) -> list[str]:
    """Episode folders with a translation (lines to check and edit)."""
    return sorted(p.parent.name for p in _root(out_root).glob("*/subs.ru.json"))


def list_reviewed(out_root: str) -> list[str]:
    return sorted(p.parent.name for p in _root(out_root).glob("*/review.json"))


def _review(folder: Path) -> dict[int, dict[str, Any]]:
    path = folder / "review.json"
    return {int(i["line"]): i for i in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def _corrections(folder: Path) -> dict[str, Any]:
    path = folder / "corrections.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def review_note(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    what = item.get("heard") and f"слышно {item['heard']}" or ""
    if item.get("sounds_like"):
        what = (what + "; " if what else "") + f"похоже на {item['sounds_like']}"
    if item.get("note"):
        what = (what + "; " if what else "") + item["note"]
    return what


def edit_rows(out_root: str, episode: str | None, show: str = SHOW[0], query: str = "") -> list[list[Any]]:
    """Rows of the line table: №, time, speaker, gender, text, fit, the correction, what was heard."""
    from stt.subtitle_edit import _policy, current_lines, fit

    if not episode:
        return []
    folder = _episode(out_root, episode)
    review, fixed, policy = _review(folder), _corrections(folder), _policy()
    rows = []
    for line in current_lines(folder):
        verdict = fit(line.text, line.window_ms, policy).verdict
        fix = fixed.get(str(line.index))
        if (show == SHOW[1] and line.index not in review) or (show == SHOW[2] and not fix) \
                or (show == SHOW[3] and verdict != "over"):
            continue
        if query and query.lower() not in f"{line.text} {line.text_en} {line.speaker_name}".lower():
            continue
        gender = "не озвучивать" if line.skip else {"female": "жен", "male": "муж"}.get(line.gender, "?")
        rows.append([line.index, _clock(line.start), line.speaker_name, gender, line.text, FIT_MARK[verdict],
                     json.dumps(fix, ensure_ascii=False) if fix else "", review_note(review.get(line.index))])
    return rows


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def line_audio(out_root: str, episode: str, line: int, engines: list[str]) -> tuple[Any, Any, str]:
    """(original, voice-over) clips of a line as (rate, samples), and its text."""
    import soundfile as sf

    folder = _episode(out_root, episode)
    records = {r["index"]: r for r in json.loads((folder / "subs.ru.json").read_text(encoding="utf-8"))}
    record = records.get(int(line))
    if record is None:
        return None, None, "Нет такой реплики."
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    offset = float(source.get("start") or 0.0)
    start, end = record["start"] / 1000 - 0.5, record["end"] / 1000 + 1.0
    audio, rate = sf.read(str(folder / "audio.wav"), dtype="float32")
    original = (rate, audio[max(0, int((start - offset) * rate)):max(0, int((end - offset) * rate))])
    dub = None
    for engine in engines or ["f5", "teratts", "vosk"]:
        track = folder / f"voiceover.{engine}" / "export" / "track.wav"
        if track.exists():
            info = sf.info(str(track))
            samples, sr = sf.read(str(track), dtype="float32", start=int(max(0.0, start) * info.samplerate),
                                  stop=int((end + 0.5) * info.samplerate))
            dub = (sr, samples)
            break
    english = " ".join(record.get("text_en") or [])
    return original, dub, f"**{line}** · {record.get('speaker_name') or record.get('speaker')}\n\n" \
                          f"EN: {english}\n\nRU: {' '.join(record['text'])}"


def speaker_choices(out_root: str, episode: str) -> list[str]:
    if not episode:
        return []
    speakers = json.loads((_episode(out_root, episode) / "speakers.json").read_text(encoding="utf-8"))
    return [f"{s['id']}: {s.get('name') or s.get('label_ru') or '?'} ({s.get('gender')})"
            for s in speakers if s.get("cues")]


def line_form(out_root: str, episode: str, line: int) -> tuple[str, str, str | None, str, bool, str]:
    """(text, fit, speaker choice, gender, not voiced, hint) of a line as it will be voiced."""
    from stt.subtitle_edit import current_lines, fit

    folder = _episode(out_root, episode)
    current = next(ln for ln in current_lines(folder) if ln.index == int(line))
    choice = next((c for c in speaker_choices(out_root, episode) if c.split(":")[0] == current.speaker), None)
    note = review_note(_review(folder).get(current.index))
    hint = f"Подозрение: {note}." if note else ""
    fix = _corrections(folder).get(str(current.index))
    if fix:
        hint += f" Уже исправлено: `{json.dumps(fix, ensure_ascii=False)}`."
    return (current.text, fit(current.text, current.window_ms).text, choice,
            "Женский" if current.gender == "female" else "Мужской", current.skip, hint.strip())


def fit_text(out_root: str, episode: str, line: Any, text: str) -> str:
    from stt.subtitle_edit import current_lines, fit

    if not episode or line is None or not text:
        return ""
    current = next((ln for ln in current_lines(_episode(out_root, episode)) if ln.index == int(line)), None)
    return fit(text, current.window_ms).text if current else ""


def speaker_sample(out_root: str, episode: str, choice: str | None) -> Any:
    """A voice sample of a speaker: the season's voice clip, else their longest line in the episode."""
    import soundfile as sf

    from stt.voice_bank import same_person

    if not episode or not choice:
        return None
    folder = _episode(out_root, episode)
    sid = choice.split(":")[0]
    card = next((s for s in json.loads((folder / "speakers.json").read_text(encoding="utf-8")) if s["id"] == sid), None)
    if card is None:
        return None
    for index in [*folder.parent.glob("*.references/references.json"),
                  *_season_root(out_root).glob("*.references/references.json")]:
        ref = next((r for r in json.loads(index.read_text(encoding="utf-8"))
                    if card.get("name") and same_person(r["name"], card["name"])), None)
        if ref and (index.parent / ref["wav"]).exists():
            return str(index.parent / ref["wav"])
    records = [r for r in json.loads((folder / "subs.ru.json").read_text(encoding="utf-8")) if r.get("speaker") == sid]
    if not records:
        return None
    longest = max(records, key=lambda r: min(r["end"] - r["start"], 8000))
    offset = float(json.loads((folder / "source.json").read_text(encoding="utf-8")).get("start") or 0.0)
    info = sf.info(str(folder / "audio.wav"))
    begin = max(0.0, longest["start"] / 1000 - offset)
    samples, rate = sf.read(str(folder / "audio.wav"), dtype="float32", start=int(begin * info.samplerate),
                            stop=int((begin + min(longest["end"] - longest["start"], 8000) / 1000) * info.samplerate))
    return rate, samples


def save_line(out_root: str, episode: str, line: Any, text: str, choice: str | None, gender: str,
              skip: bool) -> str:
    """The line as edited in the form -> corrections.json (only what differs from the translation)."""
    from stt import subtitle_edit

    if not episode or line is None:
        return "Выберите серию и реплику."
    folder = _episode(out_root, episode)
    try:
        found = subtitle_edit.set_line(folder, int(line), text=text, speaker=choice.split(":")[0] if choice else None,
                                       gender=GENDER_RU.get(gender), skip=skip)
    except ValueError as error:
        return f"Не сохранено: {error}"
    if not found:
        return f"Реплика {int(line)}: ничего не изменилось."
    change = found[0]
    warn = f"\n\n⚠ {change.fit.text}" if change.fit and change.fit.verdict == "over" else ""
    return (f"Реплика {int(line)}: {'; '.join(change.what)}. Правок в серии: {len(_corrections(folder))}. "
            f"Когда закончите, нажмите «Переозвучить изменённое».{warn}")


def undo_line(out_root: str, episode: str, line: Any) -> str:
    from stt import subtitle_edit

    if not episode or line is None:
        return "Выберите серию и реплику."
    subtitle_edit.reset_line(_episode(out_root, episode), int(line))
    return f"Реплика {int(line)}: правка убрана, будет как в переводе."


def export_subtitles(out_root: str, episode: str, kind: str, scope: str) -> tuple[str | None, str]:
    """The episode's lines as a file for Subtitle Edit (ASS) or any editor (SRT)."""
    from stt import subtitle_edit

    if not episode:
        return None, "Выберите серию."
    folder = _episode(out_root, episode)
    review = scope != "Вся серия"
    if review and not (folder / "review.json").exists():
        return None, "У этой серии нет списка для проверки."
    srt = kind.startswith("SRT")
    out = folder / "edit" / f"{episode}.ru{'.review' if review else ''}{'.srt' if srt else '.ass'}"
    (subtitle_edit.export_srt if srt else subtitle_edit.export_ass)(folder, out, review=review)
    return str(out), f"Готово: `{out.name}`. Скачайте, поправьте и загрузите обратно ниже."


def import_rows(out_root: str, episode: str, path: str | None) -> tuple[list[list[Any]], str]:
    """What an edited file changes (nothing written yet)."""
    from stt import subtitle_edit

    if not episode or not path:
        return [], "Выберите серию и файл."
    found, warnings = subtitle_edit.changes(_episode(out_root, episode), subtitle_edit.read_edited(Path(path)))
    rows = [[c.index, c.time, "; ".join(c.what), FIT_MARK[c.fit.verdict] if c.fit else ""] for c in found]
    text = f"Изменённых реплик: {len(found)}." if found else "Отличий от текущих реплик нет."
    if warnings:
        text += "\n\nНе применится:\n" + "\n".join(f"- {w}" for w in warnings)
    return rows, text


def import_apply(out_root: str, episode: str, path: str | None) -> str:
    from stt import subtitle_edit

    if not episode or not path:
        return "Выберите серию и файл."
    folder = _episode(out_root, episode)
    found, warnings = subtitle_edit.changes(folder, subtitle_edit.read_edited(Path(path)))
    if not found:
        return "Отличий от текущих реплик нет: ничего не записано."
    subtitle_edit.apply(folder, found)
    over = sum(1 for c in found if c.fit and c.fit.verdict == "over")
    return (f"Записано правок: {len(found)} (`corrections.json`)." + (f" Не влезут по времени: {over}." if over else "")
            + (f" Пропущено: {len(warnings)}." if warnings else "") + " Нажмите «Переозвучить изменённое».")


@dataclass
class Tab:
    note: Any
    reload: Callable[..., Any]
    inputs: list[Any]
    outputs: list[Any]


def build(out_root: Any, revoice: Callable[..., str], log: Any) -> Tab:
    """Build the editing tab and expose its season-change binding."""
    with gr.Tab("Проверка и правка реплик"):
        note = gr.Markdown(seasons.NO_SEASON_NOTE)
        gr.Markdown("Здесь можно поправить русские реплики серии: текст, говорящего, пол голоса или "
                    "«не озвучивать». Правки копятся в `corrections.json` серии; кнопка «Переозвучить "
                    "изменённое» заново озвучит только изменённые реплики (остальные берутся готовыми), "
                    "серия пересоберётся за минуты. Ударение: `+` перед ударной гласной (`зам+ок`). "
                    "Подробно: вкладка «Справка».")
        with gr.Row():
            edit_episode = gr.Dropdown(label="Серия", choices=[], scale=4)
            edit_refresh = gr.Button("Обновить", scale=1)
        with gr.Row():
            edit_show = gr.Radio(SHOW, value=SHOW[1], label="Показать", scale=3)
            edit_query = gr.Textbox(label="Поиск по тексту или имени", scale=2)
        edit_table = gr.Dataframe(
            headers=["№", "Время", "Говорящий", "Пол", "Текст", "Длина", "Правка", "Подозрение"],
            label="Реплики (нажмите на строку, чтобы открыть её)", interactive=False, wrap=True, max_height=420)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                line_no = gr.Number(label="Реплика №", precision=0, interactive=False)
                line_text = gr.Markdown()
                with gr.Row():
                    original_audio = gr.Audio(label="Оригинал", interactive=False)
                    dub_audio = gr.Audio(label="Озвучка сейчас", interactive=False)
            with gr.Column(scale=3):
                line_hint = gr.Markdown()
                edit_text = gr.Textbox(label="Русский текст", lines=2)
                edit_fit = gr.Markdown()
                with gr.Row():
                    edit_speaker = gr.Dropdown(label="Говорящий", choices=[], scale=3)
                    edit_gender = gr.Radio(list(GENDER_RU), label="Пол голоса", scale=2)
                sample_audio = gr.Audio(label="Голос выбранного говорящего", interactive=False)
                edit_skip = gr.Checkbox(label="Не озвучивать эту реплику (останется оригинал)")
                with gr.Row():
                    save_button_line = gr.Button("Сохранить правку", variant="primary")
                    undo_button_line = gr.Button("Убрать правку")
        with gr.Accordion("Правка в Subtitle Edit (или другом редакторе субтитров)", open=False):
            gr.Markdown("1. Скачайте реплики. **ASS** - для Subtitle Edit: говорящий в колонке «Актёр», пол - "
                        "стиль «Мужской» / «Женский», стиль «Не озвучивать» убирает реплику из озвучки. "
                        "**SRT** - для любого редактора: только текст и имя в начале `[Имя] текст`.\n"
                        "2. Поправьте реплики. Время и колонку «Эффект» (номер реплики) не меняйте, новые "
                        "реплики не добавляйте: они сверяются с серией по номеру и времени.\n"
                        "3. Загрузите исправленный файл, нажмите «Показать изменения», проверьте список и "
                        "нажмите «Записать правки», затем «Переозвучить изменённое».")
            with gr.Row():
                export_kind = gr.Radio(["ASS (Subtitle Edit: говорящий и пол)", "SRT (текст и [Имя])"],
                                       value="ASS (Subtitle Edit: говорящий и пол)", label="Формат", scale=3)
                export_scope = gr.Radio(["Вся серия", "Только для проверки"], value="Вся серия", label="Что",
                                        scale=2)
                export_button = gr.Button("Скачать реплики", scale=1)
            export_out = gr.File(label="Файл для правки", interactive=False)
            import_file = gr.File(label="Исправленный файл (.ass или .srt)", file_types=[".ass", ".srt"],
                                  type="filepath")
            with gr.Row():
                import_preview = gr.Button("Показать изменения")
                import_button = gr.Button("Записать правки", variant="primary")
            import_table = gr.Dataframe(headers=["№", "Время", "Что изменится", "Длина"], interactive=False,
                                        wrap=True)
        with gr.Row():
            review_engines = gr.CheckboxGroup(label="Переозвучить движками (пусто: уже готовые)",
                                              choices=list(ENGINES), value=[], scale=3)
            revoice_button = gr.Button("Переозвучить изменённое", variant="primary", scale=1)
        edit_message = gr.Markdown()

        def load_episodes(root: str, episode: str | None, show: str, query: str) -> tuple[Any, Any, Any]:
            episodes = list_episodes(root)
            episode = episode if episode in episodes else (episodes[0] if episodes else None)
            return (gr.update(choices=episodes, value=episode), gr.update(value=edit_rows(root, episode, show, query)),
                    gr.update(choices=speaker_choices(root, episode), value=None))

        def refresh_episodes(root: str, episode: str | None, show: str, query: str) -> tuple[Any, ...]:
            """The "Обновить" button: the lists again, and a line saying what was found."""
            found = load_episodes(root, episode, show, query)
            count = len(found[0]["choices"])
            return (*found, f"Серий с переводом: {count}." if count else
                    "Переведённых серий в папке результатов пока нет: сначала запустите обработку.")

        def pick_line(root: str, episode: str, show: str, query: str, engines: list[str],
                      evt: gr.SelectData) -> tuple[Any, ...]:
            rows = edit_rows(root, episode, show, query)
            line = rows[evt.index[0]][0] if rows and evt.index[0] < len(rows) else None
            if line is None:
                return (None, "", None, None, "", "", "", None, None, False)
            try:
                original, dub, info = line_audio(root, episode, line, [ENGINES[e] for e in engines])
                text, fit_note, choice, gender, skip, hint = line_form(root, episode, line)
            except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
                log.exception("opening line %s", line)
                return line, f"❌ Не получилось открыть реплику {line}: {error}", None, None, "", "", "", None, None, False
            return line, info, original, dub, hint, text, fit_note, choice, gender, skip

        def saved(root: str, episode: str, show: str, query: str, message: str) -> tuple[str, Any]:
            return message, gr.update(value=edit_rows(root, episode, show, query))

        listing = [out_root, edit_episode, edit_show, edit_query]
        edit_refresh.click(refresh_episodes, listing, [edit_episode, edit_table, edit_speaker, edit_message])
        edit_episode.change(load_episodes, listing, [edit_episode, edit_table, edit_speaker])
        edit_show.change(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
        edit_query.submit(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
        edit_table.select(pick_line, [*listing, review_engines],
                          [line_no, line_text, original_audio, dub_audio, line_hint, edit_text, edit_fit,
                           edit_speaker, edit_gender, edit_skip])
        edit_text.change(fit_text, [out_root, edit_episode, line_no, edit_text], edit_fit)
        edit_speaker.change(speaker_sample, [out_root, edit_episode, edit_speaker], sample_audio)
        save_button_line.click(
            lambda r, e, sh, q, n, t, who, g, sk: saved(r, e, sh, q, save_line(r, e, n, t, who, g, sk)),
            [*listing, line_no, edit_text, edit_speaker, edit_gender, edit_skip], [edit_message, edit_table])
        undo_button_line.click(lambda r, e, sh, q, n: saved(r, e, sh, q, undo_line(r, e, n)),
                               [*listing, line_no], [edit_message, edit_table])
        export_button.click(export_subtitles, [out_root, edit_episode, export_kind, export_scope],
                            [export_out, edit_message])
        import_preview.click(import_rows, [out_root, edit_episode, import_file], [import_table, edit_message])
        import_button.click(lambda r, e, sh, q, f: saved(r, e, sh, q, import_apply(r, e, f)),
                            [*listing, import_file], [edit_message, edit_table])
        revoice_button.click(revoice, [out_root, edit_episode, review_engines], edit_message)
    return Tab(note, load_episodes, listing, [edit_episode, edit_table, edit_speaker])
