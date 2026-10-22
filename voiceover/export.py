"""Export of the final voiceover: track, subtitles, decision ledger, human queue and manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voiceover.placement import TEXT_DEAD_ZONE_SEC
from voiceover.render import RenderOptions, StretchFn, render, stretch_clip
from voiceover.timeline import TimelinePolicy

FORCED_MIN_REMOVED_SEC = 0.005  # the solver's own "needs_text" threshold
MANUAL_QUEUE_RESIDUAL_SEC = 0.05  # a manual text with a larger forced-tempo residual goes back to the queue
EXPORT_FILES = ("track.wav", "track_guide.md", "spoken.srt", "display.srt", "ledger.json", "human_queue.md",
                "manifest.json")


def srt_time(ms: int) -> str:
    hours, rest = divmod(max(0, int(ms)), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt_text(entries: list[tuple[int, int, str]]) -> str:
    """Standard SRT, entries (start_ms, end_ms, text) numbered from 1 in the given order."""
    blocks = [f"{number}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n"
              for number, (start, end, text) in enumerate(entries, start=1)]
    return "\n".join(blocks)


def forced_tempo(placement: dict[str, Any]) -> float | None:
    """Tempo that makes the clip end exactly at the placement's end, when text work left a deficit."""
    if placement["removed_sec"] <= FORCED_MIN_REMOVED_SEC:
        return None
    spoken = (placement["end_ms"] - placement["start_ms"]) / 1000.0
    if spoken <= 0:
        return None
    return round(float(placement["tempo_factor"]) * float(placement["natural_sec"]) / spoken, 6)


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _verify_notes(verdict: dict[str, Any] | None) -> list[str]:
    """Plain-Russian explanation of why the verifier refused a candidate (empty when it passed)."""
    if verdict is None or verdict.get("ok"):
        return []
    reasons = verdict.get("reasons") or []
    if "no_core_left" in reasons:
        return ["по разметке смысла в нём не осталось ключевых фрагментов, отклонён без запроса к модели"]
    if "bad_response" in reasons:
        return ["проверяющая модель не дала разборчивого ответа"]
    lost = set(verdict.get("core_lost") or [])
    notes = [str(a.get("why") or "").strip() for a in verdict.get("answers") or []
             if str(a.get("id")) in lost and not a.get("kept")]
    notes = [n for n in notes if n]
    return notes or ["потеряно ключевое содержание"]


def _tried_item(tried: dict[str, Any], row: dict[str, Any] | None, verdict: dict[str, Any] | None) -> dict[str, Any]:
    return {**tried, "text": row["text"] if row else None, "verify_notes": _verify_notes(verdict)}


# Queue reasons in plain Russian; the first is the primary one, the rest are added on top.
REASON_UNRESOLVED = "не влезло"
REASON_REJECTED = "отклонено проверкой смысла"
REASON_NO_CANDIDATES = "сократитель не дал вариантов"
REASON_LOSSY = "потеря смысла"
REASON_FORCED = "ускорено сверх нормы"
REASON_MANUAL = "ручной текст не влезает"


def _queue_reasons(item: dict[str, Any]) -> list[str]:
    if item.get("reviewed"):
        return []
    if item["status"] == "manual":  # the user's own text: queued only when it still needs forced tempo
        return [REASON_MANUAL] if item["final"]["removed_sec"] > MANUAL_QUEUE_RESIDUAL_SEC else []
    reasons = []
    if item["status"] == "unresolved":
        reasons.append(REASON_UNRESOLVED)
    if item["status"] == "rejected":
        reasons.append(REASON_NO_CANDIDATES if item["reject_reason"] == "no_candidates" else REASON_REJECTED)
    if item["final"]["removed_sec"] > TEXT_DEAD_ZONE_SEC:
        reasons.append(REASON_FORCED)
    if item["lossy"]:
        reasons.append(REASON_LOSSY)
    return reasons


def _why_here(item: dict[str, Any], reasons: list[str], base_rate: float, comfort_total: float) -> str:
    """One or two sentences a reviewer can act on: what went wrong and what the viewer hears now."""
    status = item["status"]
    parts: list[str] = []
    if status == "manual":
        parts.append("Ваш текст из overrides.json всё ещё не укладывается в отведённое время.")
    elif status == "unresolved":
        parts.append("Ни один вариант не укладывается в отведённое время: сокращения либо ещё длинные, "
                     "либо отклонены проверкой смысла. Взят самый короткий из допустимых.")
    elif status == "rejected" and item["reject_reason"] == "no_candidates":
        parts.append("Модель-сократитель не предложила ни одного варианта, поэтому звучит оригинал.")
    elif status == "rejected":
        parts.append("Все сокращения отклонены проверкой смысла, поэтому звучит оригинал.")
    elif status == "timing_limited":
        parts.append("Текст не сокращали: отведённое время слишком мало, сокращением это не решить.")
    elif status == "forced":
        parts.append("Текст не трогали, но по расписанию реплика не влезает в своё время.")
    if REASON_LOSSY in reasons:
        parts.append("Взято самое сильное сокращение: проверка смысла его пропустила, но часть содержания "
                     "потеряна, стоит убедиться, что главное осталось.")
    final = item["final"]
    speed = final["total_speed"] / base_rate
    comfort = comfort_total / base_rate
    if final["forced_tempo_factor"] is not None:
        parts.append(f"Чтобы не налезть на следующую реплику, она ускорена в {speed:.2f} раза "
                     f"(комфортный предел {comfort:.2f}).")
    elif speed > comfort + 1e-6:
        parts.append(f"Звучит с ускорением {speed:.2f}, выше комфортного предела {comfort:.2f}.")
    return " ".join(parts)


def _tried_lines(item: dict[str, Any]) -> list[str]:
    lines = []
    for tried in item["rungs_tried"]:
        text = tried.get("text") or f"ступень {tried['rung']}"
        seconds = f"{tried['speech_sec']:.1f} с" if tried.get("speech_sec") is not None else "не измерен"
        if tried["fits"] and tried["verify_ok"]:
            verdict = "влезает, смысл подтверждён"
        elif tried["verify_ok"]:
            verdict = "смысл подтверждён, но не влезает"
        else:
            fit = "влезает" if tried["fits"] else "не влезает"
            notes = "; ".join(tried.get("verify_notes") or []) or "отклонён проверкой смысла"
            verdict = f"{fit}, отклонён проверкой смысла: {notes}"
        chosen = " — **сейчас звучит**" if tried["rung"] == item["chosen_rung"] else ""
        lines.append(f"- «{_cell(text)}» ({seconds}): {verdict}{chosen}")
    return lines


QUEUE_INTRO = """\
# Очередь на ручную проверку

Здесь собраны реплики, за которые автоматика не ручается. Дорожка уже собрана и нигде не накладывается
сама на себя: в худшем случае реплика звучит быстрее обычного. Для каждой реплики ниже нужно решить одно
из двух: оставить как есть или дать свой текст.

Как работать:

1. Послушайте это место в `export/track.wav` (время указано в заголовке) и прочитайте, что звучит сейчас.
2. Если устраивает, скопируйте строку «оставить как есть» в `overrides.json`.
3. Если нет, напишите свой короткий вариант и скопируйте строку «свой текст», подставив его.
   Ориентир длины указан в строке «Время». Свой текст будет озвучен и измерен, но не будет
   проверяться на смысл: за него отвечаете вы.
4. Перезапустите прогон той же командой. Файл `overrides.json` из папки прогона (рядом с `report.md`)
   подхватывается сам; принятые реплики уходят из очереди, свои тексты попадают в дорожку.

`overrides.json` выглядит так (внутри `units` строки из этого файла, через запятую):

```json
{
  "schema": "voiceover-overrides-v1",
  "units": {
    "unit-7": {"accept": true, "note": "ускорение почти не слышно"},
    "unit-25": {"text": "Она думает иначе."}
  }
}
```

Поле `note` необязательное, это заметка для себя. Готовая заготовка со всеми репликами очереди есть
в конце файла.
"""


def render_queue(ledger: list[dict[str, Any]], *, base_rate: float = 1.0, comfort_total: float = 1.0) -> tuple[str, int]:
    """Russian human review file, one card per queued unit; returns (markdown, number of cards)."""
    rows = [(item, _queue_reasons(item)) for item in ledger]
    rows = sorted([(item, reasons) for item, reasons in rows if reasons], key=lambda r: r[0]["cue_start_ms"])
    counts: dict[str, int] = {}
    for _, reasons in rows:
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    accepted = sum(1 for item in ledger if item.get("reviewed"))
    manual = sum(1 for item in ledger if item["status"] == "manual")
    lines = [QUEUE_INTRO,
             f"Реплик в очереди: {len(rows)}. Причины: "
             f"{', '.join(f'{k} — {v}' for k, v in sorted(counts.items())) or '—'}.",
             f"Уже решено вручную: принято {accepted}, своих текстов {manual}.", ""]
    if not rows:
        lines.append("Пусто.")
        return "\n".join(lines) + "\n", 0
    for item, reasons in rows:
        uid = item["unit_id"]
        same = item["final_text"] == item["original_text"]
        lines += [f"## {_clock(item['cue_start_ms'])} · {uid} (субтитры {', '.join(item['cue_ids'])})", "",
                  f"**Причина:** {'; '.join(reasons)}.",
                  f"**Что случилось:** {_why_here(item, reasons, base_rate, comfort_total)}",
                  f"**Оригинал:** {_cell(item['original_text'])}",
                  f"**Сейчас звучит:** {'оригинал без изменений' if same else _cell(item['final_text'])}",
                  f"**Время:** без ускорения влезает примерно {item['target_base_sec']:.1f} с речи; "
                  f"оригинал занимает {item['original_speech_sec']:.1f} с, "
                  f"сейчас звучит {item['final_speech_sec']:.1f} с.",
                  *(["**Пробовали:**", *_tried_lines(item)] if item["rungs_tried"] else []),
                  f"**В overrides.json:** оставить как есть → `\"{uid}\": {{\"accept\": true}},` · "
                  f"свой текст → `\"{uid}\": {{\"text\": \"…\"}},`", ""]
    lines += ["## Заготовка overrides.json", "",
              "Все реплики очереди как «оставить как есть». Скопируйте в `overrides.json` в папке прогона и "
              "замените строки тех реплик, которым хотите дать свой текст.", "", "```json", "{",
              '  "schema": "voiceover-overrides-v1",', '  "units": {',
              *[f'    "{item["unit_id"]}": {{"accept": true}}{"," if i < len(rows) - 1 else ""}'
                for i, (item, _) in enumerate(rows)],
              "  }", "}", "```"]
    return "\n".join(lines) + "\n", len(rows)


def export_final(out_dir: Path, *, cues: list[dict[str, Any]], units: list[dict[str, Any]],
                 durations0: dict[str, Any], durations1: dict[str, Any], schedule1: dict[str, Any],
                 verified: dict[str, Any], policy: TimelinePolicy, sample_rate: int = 22050,
                 manifest_extra: dict[str, Any] | None = None,
                 stretch: StretchFn = stretch_clip, accepted: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Write all export files into ``out_dir`` and return a summary for the report.

    ``accepted`` lists units the user marked as reviewed; it also covers forced units the
    compressor never produced a choice for, which ``resolve`` cannot mark itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    placements = sorted(schedule1["placements"], key=lambda p: (p["start_ms"], p["unit_id"]))
    choices = {c["unit_id"]: c for c in schedule1["choices"]}
    units_by_id = {u["unit_id"]: u for u in units}
    cues_by_id = {c["cue_id"]: c for c in cues}
    rows0 = {r["unit_id"]: r for r in durations0["rows"]}
    rows1 = {(r["unit_id"], r["rung"]): r for r in durations1["rows"]}
    verdicts = {(v["unit_id"], v["rung"]): v for v in verified["verdicts"]}

    def final_text(uid: str) -> str:
        return choices[uid]["chosen_text"] if uid in choices and choices[uid]["chosen_rung"] is not None \
            else units_by_id[uid]["text"]

    # --- track: forced tempo guarantees that no clip overruns its end
    render_placements: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    forced: dict[str, float] = {}
    for placement in placements:
        uid = placement["unit_id"]
        tempo = forced_tempo(placement)
        if tempo is not None:
            forced[uid] = tempo
        render_placements.append({**placement, "tempo_factor": tempo if tempo is not None else placement["tempo_factor"],
                                  "total_speed": round(tempo * policy.base_rate, 4) if tempo is not None
                                  else placement["total_speed"]})
        if placement["text_changed"]:
            render_rows.append({**rows1[(uid, placement["chosen_rung"])], "unit_id": uid,
                                "display_text": choices[uid]["chosen_text"]})
        else:
            render_rows.append(rows0[uid])
    summary = render(render_placements, render_rows, cues, None,
                     RenderOptions(sample_rate=sample_rate, ticks=False, segment_marker=False),
                     out_dir / "track.wav", out_dir / "track_guide.md", stretch=stretch)

    # --- subtitles
    (out_dir / "spoken.srt").write_text(
        srt_text([(p["start_ms"], p["end_ms"], final_text(p["unit_id"])) for p in placements]), encoding="utf-8")
    display: list[tuple[int, int, str]] = []
    for unit in units:
        uid = unit["unit_id"]
        if uid in choices and choices[uid]["chosen_rung"] is not None:
            members = [cues_by_id[c] for c in unit["cue_ids"]]
            display.append((members[0]["start"], members[-1]["end"], choices[uid]["chosen_text"]))
        else:
            display += [(cues_by_id[c]["start"], cues_by_id[c]["end"], cues_by_id[c]["text"]) for c in unit["cue_ids"]]
    display.sort(key=lambda entry: entry[0])
    (out_dir / "display.srt").write_text(srt_text(display), encoding="utf-8")

    # --- ledger
    ledger: list[dict[str, Any]] = []
    for placement in placements:
        uid = placement["unit_id"]
        if uid not in choices and uid not in forced:
            continue
        choice = choices.get(uid)
        rung = choice["chosen_rung"] if choice else None
        verdict = verdicts.get((uid, rung)) if rung is not None else None
        factor = forced.get(uid)
        speed = round(factor * policy.base_rate, 4) if factor is not None else placement["total_speed"]
        ledger.append({
            "unit_id": uid, "cue_ids": list(placement["cue_ids"]), "scene_id": placement["scene_id"],
            "cue_start_ms": placement["cue_start_ms"],
            "original_text": units_by_id[uid]["text"], "final_text": final_text(uid),
            "text_changed": bool(placement["text_changed"]),
            "status": choice["status"] if choice else "forced", "chosen_rung": rung,
            "lossy": choice["lossy"] if choice else False, "pass": choice["pass"] if choice else None,
            "reject_reason": choice["reject_reason"] if choice else None,
            "reviewed": uid in accepted or bool(choice and choice.get("reviewed")),
            "original_speech_sec": choice["original_speech_sec"] if choice else rows0[uid]["speech_sec"],
            "final_speech_sec": choice["chosen_speech_sec"] if choice else rows0[uid]["speech_sec"],
            "target_base_sec": choice["target_base_sec_pass1"] if choice else placement["target_base_sec"],
            "final": {"start_ms": placement["start_ms"], "shift_ms": placement["shift_ms"],
                      "tempo_factor": placement["tempo_factor"], "forced_tempo_factor": factor,
                      "total_speed": speed, "end_ms": placement["end_ms"],
                      "removed_sec": placement["removed_sec"], "status": placement["status"]},
            "verify": ({"core_lost": verdict["core_lost"], "support_lost": verdict["support_lost"],
                        "reasons": verdict["reasons"]} if verdict else None),
            "rungs_tried": [_tried_item(t, rows1.get((uid, t["rung"])), verdicts.get((uid, t["rung"])))
                            for t in (choice["rungs_tried"] if choice else [])]})
    (out_dir / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    queue_text, queue_size = render_queue(ledger, base_rate=policy.base_rate, comfort_total=policy.tempo_comfort_total)
    (out_dir / "human_queue.md").write_text(queue_text, encoding="utf-8")

    # --- manifest
    by_status = {name: sum(1 for c in choices.values() if c["status"] == name)
                 for name in ("resolved", "unresolved", "rejected", "timing_limited", "manual")}
    counts = {"units": len(units), "changed": sum(1 for p in placements if p["text_changed"]), **by_status,
              "forced": sum(1 for p in placements if p["removed_sec"] > TEXT_DEAD_ZONE_SEC),
              "forced_tempo_applied": len(forced), "lossy": sum(1 for c in choices.values() if c["lossy"]),
              "collisions": sum(1 for p in placements if p["collision_ms"] > 0),
              "reviewed": sum(1 for item in ledger if item["reviewed"])}
    track = {"duration_sec": round(summary.duration_sec, 3), "sample_rate": sample_rate,
             "render_overlaps": summary.overlaps}
    manifest = {**(manifest_extra or {}), "counts": counts, "track": track}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return {"counts": counts, "track": track, "human_queue": queue_size, "files": list(EXPORT_FILES),
            "directory": str(out_dir)}
