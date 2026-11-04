"""The progress of a job in words: status lines, the table, the "Контрольные точки" text and a readable log.

Pure functions over what ``stt.run`` leaves on disk (stage marks ``<episode>/.stages/``, progress records
``<episode>/.progress/`` from ``stt.progress``, the log of the job); ``ui.dub_app`` only shows the result.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stt import progress
from stt.run import STAGES
from ui import settings

ENGINE_STAGES = ["dub", "voiceover", "check", "mix"]
SHARED = [stage for stage in STAGES if stage not in ENGINE_STAGES]
STAGE_RU = {"transcribe": "распознавание", "glossary": "словарь", "speakers": "спикеры", "translate": "перевод",
            "dub": "сценарий", "stress": "ударения", "voiceover": "озвучка", "check": "проверка речи",
            "mix": "сведение"}
STARTED = {"transcribe": "начато", "glossary": "начат", "speakers": "начаты", "translate": "начат", "dub": "начат",
           "stress": "начаты", "voiceover": "начата", "check": "начата", "mix": "начато"}
UNIT_RU = {"lines": "реплик", "parts": "частей", "requests": "запросов", "steps": "шагов", "clips": "клипов"}
Key = tuple[str, str | None]


def engine_label(engine: str | None) -> str:
    """Short name of an engine for tables and lines ("ESpeech", not "ESpeech (голоса героев)")."""
    label = next((name for name, value in settings.ENGINES.items() if value == engine), engine or "")
    return label.split(" (")[0]


def stage_title(stage: str) -> str:
    return STAGE_RU.get(stage, stage).capitalize()


def short_name(stem: str) -> str:
    """"S51E01" out of a long file name, else the name cut to a readable length."""
    found = re.search(r"S\d+E\d+", stem, re.IGNORECASE)
    return found.group(0).upper() if found else (stem if len(stem) <= 40 else stem[:37] + "...")


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 120:
        return f"{seconds} с"
    minutes = round(seconds / 60)
    return f"{minutes} мин" if minutes < 60 else f"{minutes // 60} ч {minutes % 60:02d} мин"


def eta_text(record: dict[str, Any], now: float) -> str | None:
    """"≈ 25 мин" from the speed since the start of the stage; only when something is done and the total is known."""
    done, total = record.get("done"), record.get("total")
    started = record.get("started")
    if not done or not total or not started or done >= total:
        return None
    elapsed = now - float(started)
    if elapsed <= 0:
        return None
    left = elapsed * (total - done) / done
    return "меньше минуты" if left < 60 else "≈ " + fmt_duration(left)


def count_text(record: dict[str, Any], *, short: bool = False) -> str:
    """"120 из 640 реплик (19 %)", or "3/12 частей" when ``short``; empty while the stage has no count."""
    done, total, unit = record.get("done"), record.get("total"), UNIT_RU.get(str(record.get("unit")), "")
    if done is None or not total:
        return ""
    if short:
        return f"{done}/{total} {unit}".strip()
    return f"{done} из {total} {unit} ({round(100 * done / total)} %)".replace("  ", " ")


def running_line(record: dict[str, Any], name: str, now: float) -> str:
    """"S51E01 · ESpeech · Озвучка: 120 из 640 реплик (19 %), осталось ≈ 25 мин"."""
    who = [name] + ([engine_label(record["engine"])] if record.get("engine") else [])
    what = count_text(record)
    eta = eta_text(record, now)
    if not what:
        what = f"идёт {fmt_duration(now - float(record.get('started') or now))}"
    return f"{' · '.join(who)} · {stage_title(record['stage'])}: {what}" + (f", осталось {eta}" if eta else "")


def stage_states(folder: Path, engines: Iterable[str], since: float | None = None) -> dict[Key, tuple[str, dict[str, Any] | None]]:
    """(state, progress record) of every stage of an episode: done | running | interrupted | failed | pending.

    A stage mark means done. Records older than ``since`` (the start of the job) belong to an earlier run and
    only count when they say ``done``."""
    records = {(r["stage"], r.get("engine") or None): r for r in progress.read_all(folder)}
    found: dict[Key, tuple[str, dict[str, Any] | None]] = {}
    keys: list[Key] = [(s, None) for s in SHARED] + [(s, e) for e in engines for s in ENGINE_STAGES]
    for stage, engine in keys:
        mark = folder / ".stages" / (f"{stage}.{engine}" if engine else stage)
        record = records.get((stage, engine))
        if mark.exists():
            found[(stage, engine)] = ("done", record)
        elif record is None or (since is not None and float(record.get("updated") or 0) < since - 5):
            found[(stage, engine)] = ("pending", None)
        elif record.get("state") == "running":
            found[(stage, engine)] = ("interrupted" if progress.interrupted(record) else "running", record)
        elif record.get("state") == "failed":
            found[(stage, engine)] = ("failed", record)
        else:
            found[(stage, engine)] = ("pending", record)
    return found


def group_cell(items: list[tuple[str, str, dict[str, Any] | None]], now: float) -> str:
    """One table cell from the (stage, state, record) of a group of stages."""
    running = [i for i in items if i[1] == "running"]
    if running:
        parts = []
        for stage, _, record in running:
            detail = [stage_title(stage)]
            if record and count_text(record, short=True):
                detail.append(count_text(record, short=True))
            eta = eta_text(record, now) if record else None
            parts.append(" · ".join(detail + ([eta] if eta else [])))
        return "⏳ " + "; ".join(parts)
    for state, mark, word in (("failed", "❌", "ошибка"), ("interrupted", "⏸", "прервано")):
        hit = [i for i in items if i[1] == state]
        if hit:
            stage, _, record = hit[0]
            what = stage_title(stage)
            if record and record.get("done") and record.get("total"):
                what += f" {record['done']}/{record['total']}"
            return f"{mark} {word}: {what}"
    done = sum(i[1] == "done" for i in items)
    if done == len(items):
        return "✅ готово"
    return f"✔ {done} из {len(items)}" if done else "—"


def progress_table(episodes: list[tuple[str, Path]], engines: list[str], since: float | None = None,
                   now: float | None = None) -> tuple[list[str], list[list[str]]]:
    """(headers, rows): Серия | Общие этапы | one column per engine."""
    now = time.time() if now is None else now
    headers = ["Серия", "Общие этапы", *(engine_label(e) for e in engines)]
    rows = []
    for name, folder in episodes:
        states = stage_states(folder, engines, since)
        row = [name, group_cell([(s, *states[(s, None)]) for s in SHARED], now)]
        row += [group_cell([(s, *states[(s, e)]) for s in ENGINE_STAGES], now) for e in engines]
        rows.append(row)
    return headers, rows


def status_lines(episodes: list[tuple[str, Path]], engines: list[str], since: float | None = None,
                 now: float | None = None) -> list[str]:
    """One line per running stage (the Vosk lane runs next to the others)."""
    now = time.time() if now is None else now
    lines = []
    for name, folder in episodes:
        for state, record in stage_states(folder, engines, since).values():
            if state == "running" and record:
                lines.append(running_line(record, name, now))
    return lines


def interrupted_lines(episodes: list[tuple[str, Path]], engines: list[str], since: float | None = None) -> list[str]:
    lines = []
    for name, folder in episodes:
        for state, record in stage_states(folder, engines, since).values():
            if state == "interrupted" and record:
                who = f" ({engine_label(record['engine'])})" if record.get("engine") else ""
                done = f", {count_text(record, short=True)}" if count_text(record, short=True) else ""
                lines.append(f"{name} · {stage_title(record['stage'])}{who}{done}")
    return lines


def checkpoints(episodes: list[tuple[str, Path]], engines: list[str], running: bool, since: float | None = None) -> str:
    """Markdown: what is saved per episode, what a stop keeps, and whether stopping is safe."""
    if not episodes:
        return ""
    out, notes = [], []
    for name, folder in episodes:
        states = stage_states(folder, engines, since)
        shared = [stage_title(s) for s in SHARED if states[(s, None)][0] == "done"]
        saved = [", ".join(shared)] if shared else []
        for engine in engines:
            mine = [stage_title(s) for s in ENGINE_STAGES if states[(s, engine)][0] == "done"]
            if mine:
                saved.append(f"{engine_label(engine)}: {', '.join(mine)}")
        out.append(f"- **{name}** · сохранено: {'; '.join(saved) if saved else 'пока ничего'}")
        for (stage, engine), (state, record) in states.items():
            if state in ("running", "interrupted") and record:
                who = f" ({engine_label(engine)})" if engine else ""
                note = progress.RESUME_NOTES.get(stage, "")
                out.append(f"    - {'идёт' if state == 'running' else 'прервано'}: {stage_title(stage)}{who}: {note}")
                if note and note not in notes:
                    notes.append(note)
    if running:
        tail = "; ".join(notes) if notes else "законченные этапы сохранены"
        out.append(f"\n**Остановить можно в любой момент:** {tail}. Продолжить потом можно кнопкой «▶ Продолжить».")
    elif notes:
        out.append(f"\n**Остановлено на этапе:** {'; '.join(notes)}. Кнопка «▶ Продолжить» запустит то же самое заново.")
    return "\n".join(out)


# ---------------------------------------------------------------- readable log

LINE = re.compile(r"^(?:\d{4}-\d\d-\d\d )(\d\d:\d\d):\d\d[,.]\d+ (\w+) ([\w.]+): (.*)$")
STAGE_LINE = re.compile(r"^\[(?P<stem>.+?)\] (?P<stage>\w+)(?: \((?P<engine>\w+)\))?(?P<rest>: .*| \.\.\.)$")
PROGRESS_REST = re.compile(r"^: (\d+)/(\d+) (\w+)$")
FINISHED_REST = re.compile(r"^: finished in (\d+) s$")
RUN_HEADER = re.compile(r"^=== (\d\d:\d\d):\d\d (\S+?): ")
KNOWN = (("The ChatGPT subscription login of OpenCode does not work",
          "вход в подписку ChatGPT (opencode) не работает: запросы идут через платный роутер"),
         ("via opencode failed", "запрос через подписку (opencode) не удался"))


def _cut(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain(level: str, text: str) -> str:
    """A warning or an error: the known causes in words, else the original text cut short."""
    known = next((words for pattern, words in KNOWN if pattern in text), None)
    return f"{'❌ Ошибка' if level in ('ERROR', 'CRITICAL') else '⚠ Предупреждение'}: {known or _cut(text)}"


def readable_log(text: str, limit: int = 60) -> list[str]:
    """One line per event, oldest first: stages started / finished / done earlier, progress, warnings, errors."""
    events: list[str] = []
    in_traceback = False
    for raw in text.splitlines():
        if in_traceback:
            if raw.startswith((" ", "\t")) or not raw.strip():
                continue
            in_traceback = False
            if not LINE.match(raw):  # the exception line closes the traceback
                events.append(f"❌ Ошибка: {_cut(raw)}")
                continue
        if raw.startswith("Traceback"):
            in_traceback = True
            continue
        header = RUN_HEADER.match(raw)
        if header:
            events.append(f"{header.group(1)} Запуск обработки (движки: {header.group(2)})")
            continue
        match = LINE.match(raw)
        if not match:
            if re.search(r"failed \(exit|Error\b", raw):
                events.append(f"❌ Ошибка: {_cut(raw)}")
            continue
        clock, level, _, message = match.groups()
        stage = STAGE_LINE.match(message)
        if stage and stage.group("stage") in STAGE_RU:
            events.append(_stage_event(clock, stage))
        elif level in ("WARNING", "ERROR", "CRITICAL"):
            events.append(f"{clock} {_plain(level, message)}")
    return [e for e in events if e][-limit:]


def _stage_event(clock: str, match: re.Match[str]) -> str:
    stage, engine, rest = match.group("stage"), match.group("engine"), match.group("rest")
    who = f"{short_name(match.group('stem'))} · " + (f"{engine_label(engine)} · " if engine else "")
    title = stage_title(stage)
    if rest == " ...":
        return f"{clock} {who}{title}: {STARTED[stage]}"
    if rest == ": done earlier":
        return f"{who}{title}: сделано раньше"
    if finished := FINISHED_REST.match(rest):
        return f"{clock} {who}{title}: готово за {fmt_duration(int(finished.group(1)))}"
    if counted := PROGRESS_REST.match(rest):
        done, total, unit = counted.groups()
        return f"{clock} {who}{title}: {done} из {total} {UNIT_RU.get(unit, unit)}"
    return ""



# ---------------------------------------------------------------- log of the other tasks (voice clips, video size)

PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[,.]\d+ (\w+) [\w.]+: ")
NOISE = ("warn", "tf32", ">>>", "issues/", "skipping the download", "re-enabled", "reproducib")
TASK_RULES = (
    (re.compile(r"^encoded (\d+)%$"), lambda m: f"Кодирование видео: {m.group(1)} %"),
    (re.compile(r"^(?P<name>.+?)\s+S\d+E\d+ [\d.]+-[\d.]+s score (?P<score>[\d.]+),"),
     lambda m: f"Образец готов: {m.group('name')} (сходство {m.group('score')})"),
    (re.compile(r"^(?P<name>.+?): no clean reference among (?P<n>\d+) candidates"),
     lambda m: f"{m.group('name')}: чистого образца не нашлось (кандидатов: {m.group('n')})"),
    (re.compile(r"^(?P<name>.+?): has a clip made by hand, kept"),
     lambda m: f"{m.group('name')}: ручной образец не трогаю"),
)


def task_log_lines(text: str, limit: int = 60) -> list[str]:
    """One line per event of the log of a task that is not ``stt.run``, oldest first: the known messages in words, the
    others as they are (cut short), errors as "❌ Ошибка: ...". Library noise is left out."""
    events: list[str] = []
    in_traceback = False
    for raw in text.splitlines():
        if in_traceback:
            if raw.startswith((" ", "\t")) or not raw.strip():
                continue
            in_traceback = False
            events.append(f"❌ Ошибка: {_cut(raw)}")
            continue
        if raw.startswith("Traceback"):
            in_traceback = True
            continue
        line = raw.strip()
        if not line or line.startswith("===") or any(word in line.lower() for word in NOISE):
            continue
        found = PREFIX.match(line)
        message = line[found.end():] if found else line
        level = found.group(1) if found else "INFO"
        if level in ("ERROR", "CRITICAL") or re.search(r"\bError\b", message):
            events.append(f"❌ Ошибка: {_cut(message)}")
            continue
        for pattern, words in TASK_RULES:
            if matched := pattern.match(message):
                events.append(words(matched))
                break
        else:
            events.append(_cut(message, 120))
    return events[-limit:]
