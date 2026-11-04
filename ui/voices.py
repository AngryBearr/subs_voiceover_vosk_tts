"""Tab "Голоса сезона": the voice clips of the season's people (what ESpeech clones), by hand or automatic.

Logic first (pure functions over the files of a season, tested without the browser), then ``build`` which lays the tab
out and wires it to the page (``ui.dub_app`` passes itself as ``host``: its task runner, settings and ESpeech label).

The files: the bank ``<results>/<Season>.voices.json`` (who speaks, prints of their voices), the clips
``<results>/<Season>.references/`` (``references.json``, ``manual.json``, ``missing.json``; ``stt.voice_bank``). The
heavy work never runs inside the page: automatic searches are tasks of the runner (``stt.voice_bank --export``), the
cutting of a hand-made clip is a subprocess (``stt.voice_bank --manual-clip --preview ... --json``), so the page never
holds GPU models. Light work (reading the files, saving or removing a hand-made clip, comparing names) is done in place.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

from stt import manual_clip, voice_bank
from ui import seasons

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = REPO_ROOT / "output" / "voice_clip_preview"  # cut clips waiting to be saved (inside the allowed paths)
PREVIEW_TIMEOUT_S = 900
STATE_LABEL = {"auto": "✅ найден сам", "manual": "✋ указан вручную", "missing": "❌ нет"}
GENDER_LABEL = {"male": "муж.", "female": "жен."}
GENDER_CHOICES = {"Мужской": "male", "Женский": "female"}
HEADERS = ["Имя", "По-русски", "Пол", "Речи, мин", "Образец", "Подробно"]
INTRO = ("Образцы голосов героев: по ним ESpeech озвучивает каждого героя его собственным голосом. Образец ищется "
         "сам при обработке серий; здесь можно послушать, заменить или указать отрывок вручную.")
NO_BANK = "Банк голосов появится после обработки первой серии сезона."
RANGES_HINT = ("По одному отрезку в строке: `12:30-12:41` (минуты:секунды) или `1:02:05-1:02:14`. Найдите в плеере место, "
               "где человек говорит один, например интервью; можно несколько отрезков, программа сама возьмёт 6–10 секунд "
               "речи, уберёт музыку и распознает текст.")
ESPEECH = "f5"


# ---------------------------------------------------------------- the people table

def bank_file(results: Path, season_name: str) -> Path:
    return results / f"{season_name}.voices.json"


def sort_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """People without a clip first, then the others; most speech first inside each group."""
    return sorted(states, key=lambda r: (r["state"] != "missing", -float(r.get("speech_s") or 0.0)))


def detail_text(row: dict[str, Any]) -> str:
    """"S48E01 · 12:30 · сходство 0.87" for a clip; the reason for a person without one."""
    if row["state"] == "missing":
        return row.get("reason_ru") or "причина не записана: нажмите «Найти другой автоматически»"
    parts = []
    if row.get("episode"):
        parts.append(voice_bank.short(str(row["episode"])))
    if row.get("start") is not None:
        parts.append(manual_clip.format_time(float(row["start"])))
    if row.get("score"):
        parts.append(f"сходство {float(row['score']):.2f}")
    return " · ".join(parts)


def table_rows(states: list[dict[str, Any]]) -> list[list[Any]]:
    return [[r["name"], r.get("label_ru") or "", GENDER_LABEL.get(str(r.get("gender")), "—"),
             round(float(r.get("speech_s") or 0.0) / 60, 1), STATE_LABEL[r["state"]], detail_text(r)]
            for r in sort_states(states)]


def read_states(bank: Path) -> list[dict[str, Any]]:
    """The rows of the table, or an empty list while the season has no bank."""
    if not bank.is_file():
        return []
    return sort_states(voice_bank.clip_states(bank))


def find_state(states: list[dict[str, Any]], name: str | None) -> dict[str, Any] | None:
    return next((r for r in states if r["name"] == name), None) if name else None


def card_text(row: dict[str, Any]) -> str:
    gender = {"male": "мужской голос", "female": "женский голос"}.get(str(row.get("gender")), "пол не известен")
    who = f"### {row['name']}" + (f" · {row['label_ru']}" if row.get("label_ru") else "")
    minutes = f"речи в сезоне: {float(row.get('speech_s') or 0.0) / 60:.1f} мин"
    return f"{who}\n\n{gender} · {minutes}\n\n**Образец:** {STATE_LABEL[row['state']]}" + (
        f" — {detail_text(row)}" if detail_text(row) else "")


# ---------------------------------------------------------------- re-voicing after a change of clips

def esp_folders(results: Path, source: Path) -> list[tuple[str, Path]]:
    """(video file name, work folder) of the episodes that already have an ESpeech voice-over."""
    found = []
    for name in seasons.video_names(source):
        folder = results / Path(name).stem
        if (folder / ".stages" / f"voiceover.{ESPEECH}").exists():
            found.append((name, folder))
    return found


def _voice_hash(voice: str) -> str:
    """The clip's fingerprint inside an F5 voice id (``<path>#<hash of clip and transcript>``): the path is left out, so
    a moved season folder does not look like new voices."""
    return voice.split("#", 1)[-1]


def episode_changes(refs: Path, folder: Path) -> list[str] | None:
    """Names of the people whose clip is not the one this ESpeech episode was scripted with (``speaker_voices.f5.json``
    of the dub stage against the clips now); None when the episode has no such record."""
    from voiceover.engine_f5 import voice_id

    record = folder / f"speaker_voices.{ESPEECH}.json"
    cards = folder / "speakers.json"
    if not (record.is_file() and cards.is_file()):
        return None
    try:
        recorded = json.loads(record.read_text(encoding="utf-8"))
        people = json.loads(cards.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    index = voice_bank.load_json_list(refs / "references.json")
    changed: set[str] = set()
    for card in people:
        if not card.get("name") or card.get("id") is None:
            continue
        entry = next((r for r in index if voice_bank.same_person(r["name"], card["name"])), None)
        wav = refs / entry["wav"] if entry else None
        now = _voice_hash(voice_id(wav.resolve())) if wav is not None and wav.is_file() else None
        before = _voice_hash(recorded[card["id"]]) if card["id"] in recorded else None
        if now != before:
            changed.add(str(entry["name"]) if entry else str(card["name"]))
    return sorted(changed)


def behind_episodes(refs: Path, folders: list[tuple[str, Path]]) -> tuple[list[tuple[str, Path]], list[str]]:
    """(the ESpeech episodes scripted with other clips than the ones now, the names of the people who changed).

    An episode without the dub stage's record of its voices counts as behind when the clip index is newer than its
    script (``dub.f5``): the same test ``stt.run`` makes."""
    index = refs / "references.json"
    if not index.is_file():
        return [], []
    behind: list[tuple[str, Path]] = []
    names: set[str] = set()
    for name, folder in folders:
        changes = episode_changes(refs, folder)
        mark = folder / ".stages" / f"dub.{ESPEECH}"
        if changes is None:
            if not mark.exists() or mark.stat().st_mtime < index.stat().st_mtime:
                behind.append((name, folder))
        elif changes:
            behind.append((name, folder))
            names.update(changes)
    return behind, sorted(names)


def banner(results: Path, source: Path, season_name: str) -> tuple[str, list[str]]:
    """(text, video names to voice again): empty text when no ESpeech episode is behind the clips."""
    refs = voice_bank.references_dir(bank_file(results, season_name))
    behind, names = behind_episodes(refs, esp_folders(results, source))
    if not behind:
        return "", []
    who = f"Образцы изменились: {', '.join(names)}." if names else "Образцы изменились после последней озвучки."
    return (f"{who} Чтобы новые голоса попали в озвучку ESpeech, переозвучьте серии. Заново озвучатся только реплики "
            "этих людей, остальные берутся готовыми из кэша."), [name for name, _ in behind]


def revoice_label(count: int) -> str:
    return f"Переозвучить ESpeech ({seasons.episodes_word(count)})"


# ---------------------------------------------------------------- cutting a clip by hand

def check_cut(source: Path, name: str | None, episode: str | None, ranges_text: str, row: dict[str, Any] | None,
              gender: str | None) -> str | None:
    """Why a clip cannot be cut yet (a short Russian message), else None. Ranges are checked first, on the spot."""
    if not name or row is None:
        return "❌ Сначала выберите человека в таблице."
    try:
        manual_clip.parse_ranges(ranges_text or "")
    except ValueError as error:
        return f"❌ {error}"
    if not episode:
        return "❌ Выберите серию, из которой берутся отрезки."
    if not (source / episode).is_file():
        return f"❌ Нет файла с видео: {source / episode}. Диск не подключён?"
    if row.get("gender") not in ("male", "female") and gender not in ("male", "female"):
        return "❌ Программа не знает пол этого человека: выберите «Мужской» или «Женский»."
    return None


def preview_command(bank: Path, name: str, media: Path, ranges_text: str, out: Path, device: str,
                    gender: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "stt.voice_bank", str(bank), "--manual-clip", "--person", name, "--media", str(media)]
    for start, end in manual_clip.parse_ranges(ranges_text):
        cmd += ["--range", f"{manual_clip.format_time(start)}-{manual_clip.format_time(end)}"]
    cmd += ["--preview", str(out), "--json", "--device", device]
    return cmd + (["--gender", gender] if gender else [])


def parse_cut_output(stdout: str) -> dict[str, Any]:
    """The JSON line the program printed (other lines are library noise)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return dict(json.loads(line))
    raise ValueError("нет ответа от программы")


def cut_clip(bank: Path, name: str, media: Path, ranges_text: str, device: str, gender: str | None = None,
             preview_dir: Path | None = None) -> dict[str, Any]:
    """Cut the clip in a subprocess; the result dict of ``stt.manual_clip.clip_json`` or ``{"error": <message>}``.
    Details of a failure go to the log, the message is short and Russian."""
    folder = preview_dir or PREVIEW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", name)
    for old in folder.glob(f"{safe}.*.wav"):
        old.unlink(missing_ok=True)
    out = folder / f"{safe}.{int(time.time())}.wav"
    cmd = preview_command(bank, name, media, ranges_text, out, device, gender)
    try:
        done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=PREVIEW_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": "Вырезание заняло слишком долго и было остановлено. Возьмите отрезки покороче."}
    except OSError as error:
        log.exception("cutting a clip: cannot start")
        return {"error": f"Не получилось запустить вырезание: {error}"}
    try:
        found = parse_cut_output(done.stdout)
    except ValueError:
        log.error("cutting a clip failed (code %s): %s", done.returncode, (done.stderr or done.stdout)[-2000:])
        last = next((x for x in reversed((done.stderr or "").splitlines()) if x.strip()), "")
        return {"error": f"Не получилось вырезать образец ({last[:150] or f'код {done.returncode}'}). Подробности в логе."}
    if "error" in found:
        return {"error": str(found["error"])}
    return found


def clip_info(found: dict[str, Any]) -> str:
    """The markdown under the player of a cut clip: length, similarity, the warnings."""
    similarity = found.get("similarity")
    text = f"Речи в клипе: **{float(found['duration']):.1f} с**" + (
        f" · сходство с голосом человека в других сериях: **{float(similarity):.2f}**" if similarity is not None else "")
    warnings = found.get("warnings") or []
    return text + ("".join(f"\n\n⚠ {w}" for w in warnings) if warnings else "\n\nПредупреждений нет.")


def save_clip(results: Path, season_name: str, source: Path, name: str, state: dict[str, Any] | None, text: str,
              gender: str | None) -> str:
    """Save the cut clip (``state`` is what ``cut_clip`` returned for this person) as the person's hand-made clip.
    Returns a message starting with ✅ or ❌."""
    if not state or state.get("person") != name or not Path(str(state.get("wav", ""))).is_file():
        return "❌ Сначала нажмите «Вырезать и послушать» для этого человека."
    bank_path = bank_file(results, season_name)
    try:
        bank = voice_bank.VoiceBank.load(bank_path)
        clip = manual_clip.ManualClip(Path(state["wav"]), float(state["duration"]), str(text).strip(),
                                      [{"start": a, "end": b} for a, b in state["pieces"]], list(state.get("warnings") or []),
                                      state.get("similarity"), state.get("music_db"))
        episode = manual_clip.episode_for_media(bank_path, source / str(state["episode"]))
        manual_clip.save_manual_clip(voice_bank.references_dir(bank_path), name, clip, text=str(text), gender=gender,
                                     episode=episode, ranges=[(a, b) for a, b in state["pieces"]], bank=bank)
    except (ValueError, OSError) as error:
        log.exception("saving a hand-made clip")
        return f"❌ Не сохранено: {error}"
    return f"✅ Образец сохранён: {name} — ✋ указан вручную. Автоматический поиск его больше не заменит."


def remove_clip(results: Path, season_name: str, name: str) -> str:
    """"Вернуть автоматический выбор": delete the hand-made clip; the automatic one comes back when it is still there."""
    bank_path = bank_file(results, season_name)
    refs = voice_bank.references_dir(bank_path)
    try:
        removed = manual_clip.remove_manual_clip(refs, name)
    except OSError as error:
        log.exception("removing a hand-made clip")
        return f"❌ Не получилось убрать ручной образец: {error}"
    if not removed:
        return f"У {name} нет ручного образца."
    row = find_state(read_states(bank_path), name)
    if row and row["state"] == "auto":
        return f"✅ Ручной образец убран: у {name} снова автоматический образец."
    return (f"✅ Ручной образец убран. Автоматического образца у {name} пока нет: нажмите «Найти другой автоматически» "
            "или «Пересобрать все образцы».")


def another_command(bank: Path, name: str) -> list[str]:
    return [sys.executable, "-m", "stt.voice_bank", str(bank), "--export", str(voice_bank.references_dir(bank)),
            "--person", name, "--reject"]


def rebuild_command(bank: Path) -> list[str]:
    return [sys.executable, "-m", "stt.voice_bank", str(bank), "--export", str(voice_bank.references_dir(bank))]


# ---------------------------------------------------------------- names check

RECHECK_HEADERS = ["Серия", "Сейчас записано", "Предлагается", "Сколько других серий так считают"]


def recheck_result(bank_path: Path) -> tuple[list[dict[str, Any]], str]:
    """(suggestions, message) of the names check (no models: it compares the voice prints already in the bank)."""
    if not bank_path.is_file():
        return [], NO_BANK
    try:
        suggestions = voice_bank.recheck(voice_bank.VoiceBank.load(bank_path))
    except (OSError, ValueError, KeyError) as error:
        log.exception("names check")
        return [], f"❌ Не получилось проверить имена: {error}"
    if not suggestions:
        return [], "✅ Расхождений нет: имена людей согласованы по всему сезону."
    return suggestions, (f"Найдено расхождений: {len(suggestions)}. Это голоса, которые в других сериях названы иначе. "
                         "Если согласны, нажмите «Записать исправления имён».")


def recheck_rows(suggestions: list[dict[str, Any]]) -> list[list[Any]]:
    return [[voice_bank.short(s["episode"]), f"{s['speaker']}: {s['name']}", s["suggested"],
             f"{s['support']} (за прежнее имя: {s['own_support']})"] for s in suggestions]


def write_fixes(bank_path: Path) -> tuple[list[dict[str, Any]], str]:
    """"Записать исправления имён": the renames go to ``speakers.override.json`` of the episodes."""
    suggestions, message = recheck_result(bank_path)
    if not suggestions:
        return suggestions, message
    try:
        written = voice_bank.write_overrides(voice_bank.VoiceBank.load(bank_path), suggestions)
    except OSError as error:
        log.exception("writing name fixes")
        return suggestions, f"❌ Не получилось записать исправления: {error}"
    episodes = ", ".join(voice_bank.short(p.parent.name) for p in written)
    return suggestions, (f"✅ Исправления записаны для серий: {episodes} (`speakers.override.json` в папке каждой). "
                         "Чтобы имена попали в реплики, обработайте эти серии заново: вкладка «Обработка», выберите их и "
                         "нажмите «Начать»; программа сама заметит исправления и пересчитает спикеров и всё после этого.")


# ---------------------------------------------------------------- the tab

class Tab:
    """What ``build`` hands back to the page: the line under the tab title, and the reload wired to the season picker."""

    def __init__(self) -> None:
        self.note: Any = None
        self.reload: Any = None
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []


def build(host: Any, folder: Any, out_root: Any, voice_model: Any) -> Tab:
    """Lay the tab out and wire it. ``host`` is ``ui.dub_app`` (the task runner and the settings); ``folder`` / ``out_root``
    are the page's states of the current season; ``voice_model`` is the "download the voice model" tick of the
    processing form (the "Уточнить, кто говорит" button takes it along)."""
    tab = Tab()
    tab.note = gr.Markdown(seasons.NO_SEASON_NOTE)
    gr.Markdown(INTRO)
    no_bank = gr.Markdown(NO_BANK, visible=False)
    with gr.Group(visible=False) as banner_box:
        banner_text = gr.Markdown()
        revoice = gr.Button("Переозвучить ESpeech", variant="primary")
    people = gr.Dataframe(headers=HEADERS, label="Люди сезона (нажмите на строку, чтобы открыть человека)",
                          interactive=False, wrap=True, max_height=380)
    person = gr.Dropdown(label="Человек", choices=[], value=None)
    card = gr.Markdown()
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            clip_audio = gr.Audio(label="Образец сейчас", type="filepath", interactive=False, visible=False)
        with gr.Column(scale=4):
            clip_text = gr.Markdown()
    with gr.Row():
        another = gr.Button("Найти другой автоматически", visible=False)
        revert = gr.Button("Вернуть автоматический выбор", visible=False)
    card_message = gr.Markdown()
    with gr.Accordion("✋ Указать отрывок вручную", open=False, visible=False) as manual:
        gr.Markdown(RANGES_HINT)
        with gr.Row(equal_height=False):
            episode = gr.Dropdown(label="Серия", choices=[], value=None, scale=3)
            ranges = gr.Textbox(label="Отрывки", lines=3, placeholder="1:58.4-2:03.0\n2:03.5-2:05.7", scale=3)
            gender = gr.Radio(label="Пол", choices=list(GENDER_CHOICES), value=None, visible=False, scale=1)
        cut = gr.Button("Вырезать и послушать", variant="primary")
        cut_message = gr.Markdown()
        cut_audio = gr.Audio(label="Вырезанный образец", type="filepath", interactive=False, visible=False)
        cut_info = gr.Markdown()
        cut_text = gr.Textbox(label="Текст отрывка (распознан автоматически, можно поправить)", lines=2,
                              interactive=True, visible=False)
        save = gr.Button("Сохранить как образец", variant="primary", visible=False)
        cut_state = gr.State(None)
    gr.Markdown("### Весь сезон")
    season_actions = {}
    for key, label, text in (
            ("rebuild", "Пересобрать все образцы",
             "Заново ищет лучший отрывок для каждого человека; ручные образцы не трогает; несколько минут."),
            ("recount", "Уточнить, кто говорит",
             "Заново распределяет слова между людьми сезона по их голосам; уже переведённые серии не меняются."),
            ("check", "Проверить имена",
             "Сравнивает голоса людей между сериями и показывает, где один человек записан под разными именами."),
            ("fixes", "Записать исправления имён",
             "Записывает найденные исправления в серии; чтобы они подействовали, серии нужно обработать заново.")):
        with gr.Row(equal_height=False):
            season_actions[key] = gr.Button(label, scale=2, min_width=240)
            gr.Markdown(text, elem_classes=["voices-action-note"])
    season_message = gr.Markdown()
    recheck_table = gr.Dataframe(headers=RECHECK_HEADERS, interactive=False, wrap=True, visible=False,
                                 label="Что предлагается исправить")

    # ---- helpers closed over the components
    def season_of(src: str, res: str) -> tuple[Path, Path, str]:
        return Path(src), Path(res), Path(src).name

    def states_of(src: str, res: str) -> list[dict[str, Any]]:
        if not src or not res:
            return []
        source, results, name = season_of(src, res)
        try:
            return read_states(bank_file(results, name))
        except (OSError, ValueError, KeyError):
            log.exception("reading the voice bank")
            return []

    view_outputs = [people, person, no_bank, banner_box, banner_text, revoice, episode]

    def view(src: str, res: str, selected: str | None) -> tuple[Any, ...]:
        """The table, the picker, the banner: everything that follows from the files of the season."""
        if not src or not res:
            return (gr.update(value=[]), gr.update(choices=[], value=None), gr.update(visible=False), gr.update(visible=False),
                    "", gr.update(), gr.update(choices=[], value=None))
        source, results, name = season_of(src, res)
        states = states_of(src, res)
        names = [r["name"] for r in states]
        text, todo = banner(results, source, name)
        return (gr.update(value=table_rows(states)),
                gr.update(choices=names, value=selected if selected in names else None),
                gr.update(visible=not states, value=NO_BANK), gr.update(visible=bool(todo)), text,
                gr.update(value=revoice_label(len(todo))), gr.update(choices=seasons.episode_choices(seasons.video_names(source))))

    core_outputs = [card, clip_audio, clip_text, another, revert, manual, gender]
    preview_outputs = [cut_audio, cut_info, cut_text, save, cut_state, cut_message, card_message]

    def card_core(src: str, res: str, who: str | None, fresh: bool) -> tuple[Any, ...]:
        """The person's card. ``fresh``: a person was picked or changed (the form opens and clears); else the files
        changed under an open card (nothing the person typed is touched)."""
        row = find_state(states_of(src, res), who)
        if row is None:
            return ("", gr.update(value=None, visible=False), "", gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False, open=False), gr.update(visible=False))
        wav = row.get("wav")
        have = bool(wav) and Path(str(wav)).is_file()
        manual_row = row["state"] == "manual"
        text = f"**Текст образца:** «{row['text']}»" if row.get("text") else ""
        unknown = row.get("gender") not in ("male", "female")
        return (card_text(row), gr.update(value=str(wav) if have else None, visible=have), text,
                gr.update(visible=not manual_row), gr.update(visible=manual_row),
                gr.update(visible=True, open=row["state"] == "missing") if fresh else gr.update(visible=True),
                gr.update(visible=unknown, value=None) if fresh else gr.update(visible=unknown))

    def preview_reset(card_msg: str = "", cut_msg: str = "") -> tuple[Any, ...]:
        return (gr.update(value=None, visible=False), "", gr.update(value="", visible=False), gr.update(visible=False),
                None, cut_msg, card_msg)

    def card_view(src: str, res: str, who: str | None) -> tuple[Any, ...]:
        return (*card_core(src, res, who, True), *preview_reset())

    all_outputs = [*view_outputs, *core_outputs, *preview_outputs]

    def refresh_all(src: str, res: str, who: str | None, card_msg: str = "", cut_msg: str = "") -> tuple[Any, ...]:
        return (*view(src, res, who), *card_core(src, res, who, True), *preview_reset(card_msg, cut_msg))

    def pick(src: str, res: str, evt: gr.SelectData) -> Any:
        rows = table_rows(states_of(src, res))
        row = getattr(evt, "row_value", None)
        who = row[0] if row else (rows[evt.index[0]][0] if rows and evt.index[0] < len(rows) else None)
        return gr.update(value=who)

    # ---- buttons: automatic search and the hand-made clip
    def run_another(src: str, res: str, who: str | None) -> str:
        row = find_state(states_of(src, res), who)
        if row is None:
            return "❌ Сначала выберите человека в таблице."
        if row["state"] == "manual":
            return "У этого человека ручной образец: автоматический поиск его не заменяет. Нажмите «Вернуть автоматический выбор»."
        source, results, name = season_of(src, res)
        return host.start_task(f"Другой образец: {who}", another_command(bank_file(results, name), str(who)), results)

    def run_revert(src: str, res: str, who: str | None) -> tuple[Any, ...]:
        if not who:
            return refresh_all(src, res, who, "❌ Сначала выберите человека в таблице.")
        source, results, name = season_of(src, res)
        return refresh_all(src, res, who, remove_clip(results, name, str(who)))

    def cut_started(src: str, res: str, who: str | None, ep: str | None, text: str, sex: str | None) -> tuple[Any, ...]:
        row = find_state(states_of(src, res), who)
        problem = check_cut(Path(src), who, ep, text, row, GENDER_CHOICES.get(sex or ""))
        if problem:
            return gr.update(value="Вырезать и послушать", interactive=True), problem
        busy = host.JOB if host._busy() else None
        wait = ("Вырезаю, очищаю от музыки и распознаю текст — около 15 секунд." if busy is None else
                f"Сейчас идёт «{busy.title}»: вырезаю на процессоре, это медленнее — около минуты.")
        return gr.update(value="⏳ Вырезаю…", interactive=False), f"⏳ {wait}"

    def cut_done(src: str, res: str, who: str | None, ep: str | None, text: str, sex: str | None) -> tuple[Any, ...]:
        ready = gr.update(value="Вырезать и послушать", interactive=True)
        row = find_state(states_of(src, res), who)
        picked = GENDER_CHOICES.get(sex or "")
        if check_cut(Path(src), who, ep, text, row, picked):
            return ready, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        source, results, name = season_of(src, res)
        device = "cpu" if host._busy() else "cuda"
        found = cut_clip(bank_file(results, name), str(who), source / str(ep), text, device, picked)
        if "error" in found:
            return (ready, f"❌ {found['error']}", gr.update(value=None, visible=False), "",
                    gr.update(value="", visible=False), gr.update(visible=False), None)
        state = {**found, "person": who, "episode": ep}
        return (ready, "✅ Готово: послушайте образец, при необходимости поправьте текст и сохраните.",
                gr.update(value=found["wav"], visible=True), clip_info(found),
                gr.update(value=found["text"], visible=True),
                gr.update(value=f"Сохранить как образец {who}", visible=True), state)

    def do_save(src: str, res: str, who: str | None, text: str, sex: str | None, state: dict[str, Any] | None
                ) -> tuple[Any, ...]:
        if not who:
            return refresh_all(src, res, who, cut_msg="❌ Сначала выберите человека в таблице.")
        source, results, name = season_of(src, res)
        message = save_clip(results, name, source, str(who), state, text, GENDER_CHOICES.get(sex or ""))
        if message.startswith("❌"):  # the cut clip stays: the person can fix the text and save again
            return (*[gr.update() for _ in all_outputs[:-2]], message, gr.update())
        return refresh_all(src, res, who, cut_msg=message)

    # ---- banner and season actions
    def do_revoice(src: str, res: str) -> str:
        source, results, name = season_of(src, res)
        _, todo = banner(results, source, name)
        if not todo:
            return "Все серии уже озвучены с нынешними образцами: переозвучивать нечего."
        config = host.season_base(res)
        values = host.form_defaults(config)
        label = next(k for k, v in host.ENGINES.items() if v == ESPEECH)
        return host.start_job(src, todo, res, [label], values[1], values[2], values[3], values[4], values[5], values[6],
                              False, 0, 0, f5_fast=values[7], stt_args=["--no-season-voices"], title="Переозвучка ESpeech")

    def run_rebuild(src: str, res: str) -> str:
        if not src or not res:
            return host.NO_SEASON
        source, results, name = season_of(src, res)
        if not bank_file(results, name).is_file():
            return NO_BANK
        return host.start_task("Сбор образцов голосов", rebuild_command(bank_file(results, name)), results)

    def run_recount(src: str, res: str, download: bool) -> str:
        return host.recount_season_voices(src, res, download)

    def check_started() -> tuple[Any, ...]:
        return (gr.update(value="⏳ Проверяю…", interactive=False),
                "⏳ Сравниваю голоса людей между сериями — несколько секунд…")

    def check_done(src: str, res: str) -> tuple[Any, ...]:
        ready = gr.update(value="Проверить имена", interactive=True)
        if not src or not res:
            return ready, host.NO_SEASON, gr.update(visible=False)
        source, results, name = season_of(src, res)
        suggestions, message = recheck_result(bank_file(results, name))
        return ready, message, gr.update(value=recheck_rows(suggestions), visible=bool(suggestions))

    def fixes_started() -> tuple[Any, ...]:
        return gr.update(value="⏳ Записываю…", interactive=False), "⏳ Сравниваю имена и записываю исправления…"

    def fixes_done(src: str, res: str) -> tuple[Any, ...]:
        ready = gr.update(value="Записать исправления имён", interactive=True)
        if not src or not res:
            return ready, host.NO_SEASON, gr.update(visible=False)
        source, results, name = season_of(src, res)
        suggestions, message = write_fixes(bank_file(results, name))
        return ready, message, gr.update(value=recheck_rows(suggestions), visible=bool(suggestions))

    inputs = [folder, out_root]
    person.change(card_view, [*inputs, person], [*core_outputs, *preview_outputs], show_progress="hidden")
    people.select(pick, inputs, person, show_progress="hidden")
    another.click(run_another, [*inputs, person], card_message)
    revert.click(run_revert, [*inputs, person], all_outputs)
    cut_inputs = [*inputs, person, episode, ranges, gender]
    cut.click(cut_started, cut_inputs, [cut, cut_message], queue=False).then(
        cut_done, cut_inputs, [cut, cut_message, cut_audio, cut_info, cut_text, save, cut_state])
    save.click(do_save, [*inputs, person, cut_text, gender, cut_state], all_outputs)
    revoice.click(do_revoice, inputs, card_message)
    season_actions["rebuild"].click(run_rebuild, inputs, season_message)
    season_actions["recount"].click(run_recount, [*inputs, voice_model], season_message)
    check_button, fixes_button = season_actions["check"], season_actions["fixes"]
    check_button.click(check_started, None, [check_button, season_message], queue=False).then(
        check_done, inputs, [check_button, season_message, recheck_table])
    fixes_button.click(fixes_started, None, [fixes_button, season_message], queue=False).then(
        fixes_done, inputs, [fixes_button, season_message, recheck_table])

    # ---- reloads: a season picked, and the files changing under the page (a task of the runner finished)
    stamp = gr.State("")

    def signature(src: str, res: str) -> str:
        if not src or not res:
            return ""
        source, results, name = season_of(src, res)
        refs = voice_bank.references_dir(bank_file(results, name))
        parts = [str(bank_file(results, name).stat().st_mtime) if bank_file(results, name).is_file() else "",
                 *[str((refs / f).stat().st_mtime) if (refs / f).exists() else "" for f in ("references.json", "missing.json")]]
        return "|".join(parts + [str(len(banner(results, source, name)[1]))])

    live_outputs = [*view_outputs, *core_outputs]

    def poll(src: str, res: str, who: str | None, last: str) -> tuple[Any, ...]:
        now = signature(src, res)
        if now == last:
            return (last, *[gr.update() for _ in live_outputs])
        return (now, *view(src, res, who), *card_core(src, res, who, False))

    gr.Timer(3.0).tick(poll, [*inputs, person, stamp], [stamp, *live_outputs], show_progress="hidden")
    def on_season(src: str, res: str) -> tuple[Any, ...]:
        """A season was picked: everything of the tab is read again, nobody is selected."""
        return refresh_all(src, res, None)

    tab.reload = on_season
    tab.inputs = inputs
    tab.outputs = all_outputs
    return tab
