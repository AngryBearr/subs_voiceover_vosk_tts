"""Decoding of a blind listening pack: read its key, join it with the ratings, aggregate, write the results.

This is the only module that reads ``key.json``; :mod:`ui.blind_listen` imports it only for an explicit decode
(``POST /api/decode`` with confirmation, or ``--analyze``). Supported key layouts:

* ``scenes`` - ``[{"file", "scene": {episode, era, scene_type, ...}, "mapping": {letter: {"variant_id", ...}}}]``
  (multitrack video packs);
* ``groups`` list - ``[{"folder", episode, kind, ..., "variants": {letter: {"variant", ...}}}]`` with an optional
  ``variants_legend`` (WAV group packs);
* ``clips`` - ``[{"file", "status", "reasons": [...]}]`` (single clips checked against a detector);
* ``blind_key/1`` - the universal key written by :func:`ui.blind_listen.write_pack_manifest`.

Results: ``results_decoded.json`` (machine readable) and ``results_decoded.md`` (Russian summary for the chat).
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ui.blind_listen import RESULTS_JSON, RESULTS_MD, Group, Pack, natural_key, pack_progress
from utils.atomic_io import write_text_atomic

FACET_KEYS = ("season", "episode", "era", "scene_type", "kind", "category", "type")
FACET_TITLES_RU: dict[str, str] = {
    "season": "Сезон", "episode": "Серия", "era": "Эпоха / качество исходника", "scene_type": "Тип сцены",
    "kind": "Тип реплики", "category": "Категория", "type": "Тип", "status": "Пометка в ключе",
    "reason": "Причина пометки", "speaker": "Спикер",
}
CLEAN_TAG = "Чисто"
VERDICT_TITLES_RU = {"defect": "Слышен брак", "clean": "Чисто на слух", "unsure": "Не уверен(а)",
                     "not_rated": "Не оценено"}


@dataclass
class KeyVariant:
    """Real identity of one blind variant."""

    name: str
    description: str = ""
    facets: dict[str, str | list[str]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyGroup:
    """Real identity of a group's variants plus the group's own facets (episode, scene type ...)."""

    facets: dict[str, str] = field(default_factory=dict)
    variants: dict[str, KeyVariant] = field(default_factory=dict)


@dataclass
class NormalizedKey:
    """Any supported key layout, reduced to groups → blind id → real variant."""

    schema: str
    groups: dict[str, KeyGroup]
    legend: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------- key loading


def key_path(pack: Pack) -> Path:
    """Where the pack's key lives."""
    path = Path(pack.key_file)
    return path if path.is_absolute() else pack.dir / path


def load_key(pack: Pack) -> NormalizedKey:
    """Read and normalise the pack's key (raises ``FileNotFoundError`` / ``ValueError``)."""
    path = key_path(pack)
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла ключа: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Непонятный формат ключа: {path}")
    return normalize_key(raw, pack)


def _scalar(value: Any) -> str | None:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _variant_name(entry: dict[str, Any], fallback: str) -> str:
    for name in ("name", "variant_id", "variant", "model", "label", "id"):
        text = _scalar(entry.get(name))
        if text:
            return text
    return fallback


def _variant_description(entry: dict[str, Any], name: str, legend: dict[str, str]) -> str:
    text = _scalar(entry.get("description")) or legend.get(name, "")
    if not text and _scalar(entry.get("nfe")):
        text = f"{entry['nfe']} шагов синтеза"
    return text


def _group_facets(entry: dict[str, Any]) -> dict[str, str]:
    facets: dict[str, str] = {}
    for name in FACET_KEYS:
        text = _scalar(entry.get(name))
        if text:
            facets[name] = text
    return facets


def _key_variants(mapping: Any, legend: dict[str, str]) -> dict[str, KeyVariant]:
    variants: dict[str, KeyVariant] = {}
    if not isinstance(mapping, dict):
        return variants
    for blind_id, entry in mapping.items():
        if isinstance(entry, dict):
            name = _variant_name(entry, str(blind_id))
            facets = {k: v for k, v in (entry.get("facets") or {}).items() if isinstance(v, (str, list))}
            info = {k: v for k, v in entry.items() if k not in ("facets",)}
            variants[str(blind_id)] = KeyVariant(name, _variant_description(entry, name, legend), facets, info)
        elif _scalar(entry):
            name = str(entry)
            variants[str(blind_id)] = KeyVariant(name, legend.get(name, ""))
    return variants


def normalize_key(raw: dict[str, Any], pack: Pack) -> NormalizedKey:
    """Reduce any supported key layout to :class:`NormalizedKey`."""
    legend = {str(k): str(v) for k, v in (raw.get("legend") or raw.get("variants_legend") or {}).items()}
    groups: dict[str, KeyGroup] = {}
    raw_groups = raw.get("groups")
    if raw.get("schema") == "blind_key/1" or isinstance(raw_groups, dict):
        for group_id, entry in (raw_groups or {}).items():
            entry = entry if isinstance(entry, dict) else {}
            facets = {k: str(v) for k, v in (entry.get("facets") or {}).items() if _scalar(v)}
            groups[str(group_id)] = KeyGroup(facets, _key_variants(entry.get("variants"), legend))
        return NormalizedKey("blind_key/1", groups, legend)
    if isinstance(raw.get("scenes"), list):
        for entry in raw["scenes"]:
            if not isinstance(entry, dict):
                continue
            scene: dict[str, Any] = entry["scene"] if isinstance(entry.get("scene"), dict) else {}
            group_id = str(entry.get("file") or scene.get("id") or "")
            groups[group_id] = KeyGroup(_group_facets(scene), _key_variants(entry.get("mapping"), legend))
        return NormalizedKey("scenes", groups, legend)
    if isinstance(raw.get("clips"), list):
        return NormalizedKey("clips", _clips_key(raw["clips"], pack), legend)
    if isinstance(raw_groups, list):
        for entry in raw_groups:
            if not isinstance(entry, dict):
                continue
            group_id = next((str(entry[k]) for k in ("folder", "id", "group", "file") if _scalar(entry.get(k))), "")
            groups[group_id] = KeyGroup(_group_facets(entry),
                                        _key_variants(entry.get("variants") or entry.get("mapping"), legend))
        return NormalizedKey("groups", groups, legend)
    raise ValueError("Непонятный формат ключа: нет scenes / groups / clips")


def _clips_key(clips: list[Any], pack: Pack) -> dict[str, KeyGroup]:
    """Clip keys name files; find the group (usually "all") holding each clip."""
    owner: dict[str, str] = {}
    for group in pack.groups:
        for item in group.variants:
            owner[item.id] = group.id
            if item.path is not None:
                owner[item.path.name] = group.id
    groups: dict[str, KeyGroup] = {}
    for entry in clips:
        if not isinstance(entry, dict) or not _scalar(entry.get("file")):
            continue
        file_name = str(entry["file"])
        stem = Path(file_name).stem
        group_id = owner.get(stem) or owner.get(file_name) or "all"
        reasons = [str(r) for r in entry.get("reasons") or [] if _scalar(r)]
        status = _scalar(entry.get("status")) or _variant_name(entry, "?")
        variant = KeyVariant(status, ", ".join(reasons),
                             {"status": status, "reason": reasons or ["нет"]},
                             {k: v for k, v in entry.items() if k in ("status", "reasons", "file")})
        groups.setdefault(group_id, KeyGroup()).variants[stem] = variant
    return groups


# --------------------------------------------------------------------------------------------- analysis


def _ranks(scores: dict[str, int]) -> dict[str, int]:
    """Competition ranks (1 = best; equal scores share a rank)."""
    return {item: 1 + sum(1 for other in scores.values() if other > value) for item, value in scores.items()}


def ear_verdict(score: int | None, tags: list[str]) -> str:
    """What the listener heard in a single clip: defect / clean / unsure / not_rated."""
    defect_tags = [t for t in tags if t != CLEAN_TAG]
    if score is None and not tags:
        return "not_rated"
    if defect_tags or (score is not None and score <= 2):
        return "defect"
    if (score is not None and score >= 4) or CLEAN_TAG in tags:
        return "clean"
    return "unsure"


def _group_rows(pack: Pack, group: Group, state: dict[str, Any], key_group: KeyGroup | None) -> dict[str, Any]:
    scores: dict[str, int] = {k: v for k, v in (state.get("scores") or {}).items() if isinstance(v, int)}
    notes: dict[str, str] = state.get("notes") or {}
    tags: dict[str, list[str]] = state.get("tags") or {}
    listened = set(state.get("listened") or [])
    span = group.scale_max - group.scale_min
    comparison = pack.mode == "comparison"
    ranks = _ranks(scores) if comparison else {}
    best = max(scores.values()) if scores else None
    worst = min(scores.values()) if scores else None
    rows: list[dict[str, Any]] = []
    for item in group.variants:
        variant = key_group.variants.get(item.id) if key_group else None
        score = scores.get(item.id)
        row: dict[str, Any] = {
            "blind": item.id, "label": item.label,
            "name": variant.name if variant else f"нет в ключе ({item.id})",
            "description": variant.description if variant else "",
            "facets": dict(variant.facets) if variant else {},
            "score": score,
            "norm": round((score - group.scale_min) / span, 4) if score is not None and span > 0 else None,
            "note": notes.get(item.id, ""), "tags": list(tags.get(item.id, [])), "listened": item.id in listened,
        }
        if comparison:
            row["rank"] = ranks.get(item.id)
            row["is_top"] = score is not None and score == best and len(scores) >= 2
            row["is_last"] = score is not None and score == worst and best != worst
        else:
            row["verdict"] = ear_verdict(score, row["tags"])
        rows.append(row)
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0), natural_key(r["blind"])))
    references = [{"label": ref.label, "listened": ref.id in listened, "note": notes.get(ref.id, "")}
                  for ref in group.references]
    return {
        "group": group.id, "title": group.title, "caption": group.caption,
        "facets": dict(key_group.facets) if key_group else {},
        "scale": {"min": group.scale_min, "max": group.scale_max},
        "n_variants": len(group.variants), "n_rated": len(scores),
        "complete": all(v.id in scores for v in group.variants), "comment": state.get("comment", ""),
        "variants": rows, "winners": [r["name"] for r in rows if r.get("is_top")], "references": references,
    }


def _stats(name: str, description: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r["score"] for r in rows if r["score"] is not None]
    norms = [r["norm"] for r in rows if r["norm"] is not None]
    ranks = [r["rank"] for r in rows if r.get("rank") is not None]
    return {
        "name": name, "description": description, "n": len(scores), "groups_seen": len(rows),
        "mean": round(statistics.fmean(scores), 3) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "mean_norm": round(statistics.fmean(norms), 4) if norms else None,
        "top_1_count": sum(1 for r in rows if r.get("is_top")),
        "last_count": sum(1 for r in rows if r.get("is_last")),
        "mean_rank": round(statistics.fmean(ranks), 3) if ranks else None,
    }


def _sorted_stats(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(stats, key=lambda s: (s["mean_norm"] is None, -(s["mean_norm"] or 0), -s["top_1_count"],
                                        -(s["mean"] or 0), natural_key(s["name"])))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    descriptions: dict[str, str] = {}
    for row in rows:
        by_name[row["name"]].append(row)
        descriptions.setdefault(row["name"], row["description"])
    return _sorted_stats([_stats(name, descriptions[name], items) for name, items in by_name.items()])


def _single_stats(value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stat = _stats(value, "", rows)
    for verdict in VERDICT_TITLES_RU:
        stat[verdict] = sum(1 for r in rows if r.get("verdict") == verdict)
    return stat


def _facet_values(group_row: dict[str, Any], row: dict[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for source in (group_row["facets"], row["facets"]):
        for facet, value in source.items():
            values[facet] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
    return values


def analyze_ratings(pack: Pack, ratings: dict[str, Any], key: NormalizedKey) -> dict[str, Any]:
    """Join ratings with the key and aggregate them (pure function, writes nothing)."""
    saved = ratings.get("groups") or {}
    groups = [_group_rows(pack, group, saved.get(group.id) or {}, key.groups.get(group.id)) for group in pack.groups]
    all_rows = [row for group in groups for row in group["variants"]]
    by_facet: dict[str, dict[str, list[dict[str, Any]]]] = {}
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for group in groups:
        for row in group["variants"]:
            for facet, facet_values in _facet_values(group, row).items():
                for value in facet_values:
                    buckets[facet][value].append(row)
    for facet, by_value in buckets.items():
        if facet == "status" or (pack.mode == "comparison" and len(by_value) < 2):
            continue  # status is already the variant name; one value = nothing to compare
        ordered = sorted(by_value.items(), key=lambda kv: natural_key(kv[0]))
        if pack.mode == "single":  # one line per value: how the clips with it were rated
            by_facet[facet] = {value: [_single_stats(value, rows)] for value, rows in ordered}
        else:
            by_facet[facet] = {value: _aggregate(rows) for value, rows in ordered}
    summary: dict[str, Any] = {"by_variant": _aggregate(all_rows), "by_facet": by_facet}
    if pack.mode == "single":
        crosstab: dict[str, dict[str, int]] = {}
        for row in all_rows:
            cell = crosstab.setdefault(row["name"], {v: 0 for v in VERDICT_TITLES_RU})
            cell[row["verdict"]] += 1
        summary["ear_vs_key"] = crosstab
    comments = [{"group": g["group"], "title": g["title"], "comment": g["comment"]} for g in groups if g["comment"]]
    notes = [{"group": g["group"], "blind": r["blind"], "name": r["name"], "note": r["note"]}
             for g in groups for r in g["variants"] if r["note"]]
    notes += [{"group": g["group"], "blind": "ориентир", "name": ref["label"], "note": ref["note"]}
              for g in groups for ref in g["references"] if ref["note"]]
    return {
        "schema": "blind_listen.results/1", "pack": pack.rel_id or pack.dir.name, "title": pack.title,
        "mode": pack.mode, "kind": pack.kind, "key_schema": key.schema, "legend": key.legend,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "decoded_at": ratings.get("decoded_at"),
        "unique_ranks": bool(ratings.get("unique_ranks")), "progress": pack_progress(pack, ratings),
        "scale_labels": pack.scale_labels, "groups": groups, "summary": summary, "comments": comments,
        "notes": notes,
    }


def decode_pack(pack: Pack, ratings: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Read the key, aggregate and write ``results_decoded.json`` / ``results_decoded.md`` into ``out_dir``."""
    results = analyze_ratings(pack, ratings, load_key(pack))
    write_text_atomic(out_dir / RESULTS_JSON, json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(out_dir / RESULTS_MD, render_markdown(results))
    return results


# --------------------------------------------------------------------------------------------- markdown


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{round(value * 100)}"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown(results: dict[str, Any], heading_level: int = 1) -> str:
    """Russian summary of decoded results, ready to paste into a chat."""
    h = "#" * heading_level
    progress = results.get("progress") or {}
    single = results.get("mode") == "single"
    lines = [f"{h} Итоги слепого прослушивания: {results.get('title', '')}", ""]
    unit = "клипов" if single else "групп полностью"
    lines.append(f"Расшифровано {results.get('generated_at', '')}. Оценено {unit}: "
                 f"{progress.get('done', 0)} из {progress.get('total', 0)}.")
    lines.append("")
    if single:
        lines += _single_markdown(results, h)
    else:
        lines += _comparison_markdown(results, h)
    comments = results.get("comments") or []
    if comments:
        lines += ["", f"{h}# Комментарии к группам", ""]
        lines += [f"- **{_cell(c['title'])}**: {_cell(c['comment'])}" for c in comments]
    notes = results.get("notes") or []
    if notes:
        lines += ["", f"{h}# Заметки к вариантам", ""]
        lines += [f"- **{_cell(n['group'])}**, {_cell(n['blind'])} = {_cell(n['name'])}: {_cell(n['note'])}"
                  for n in notes]
    return "\n".join(lines).rstrip() + "\n"


def _comparison_markdown(results: dict[str, Any], h: str) -> list[str]:
    summary = results.get("summary") or {}
    lines = ["Шкала в каждой группе 1..N, где N — число вариантов: 1 — хуже всех, N — лучше всех. "
             "«Балл, %» — оценка, приведённая к 0–100 (100 — лучший в группе), чтобы сравнивать группы "
             "с разным числом вариантов.", "", f"{h}# Общий рейтинг вариантов", ""]
    rows = []
    for place, stat in enumerate(summary.get("by_variant") or [], start=1):
        rows.append([place, f"**{stat['name']}**", stat["description"], _pct(stat["mean_norm"]),
                     _num(stat["mean"]), _num(stat["median"], 1), stat["top_1_count"], stat["last_count"],
                     f"{stat['n']} из {stat['groups_seen']}"])
    lines += _table(["Место", "Вариант", "Что это", "Балл, %", "Средняя", "Медиана", "1-х мест", "Последних",
                     "Оценок"], rows)
    for facet, values in (summary.get("by_facet") or {}).items():
        lines += ["", f"{h}# По категории «{FACET_TITLES_RU.get(facet, facet)}»", ""]
        facet_rows = []
        for value, stats in values.items():
            order = " · ".join(f"{s['name']} {_pct(s['mean_norm'])}" for s in stats if s["n"])
            leader = stats[0]["name"] if stats and stats[0]["n"] else "—"
            facet_rows.append([value, f"**{leader}**", order or "нет оценок"])
        lines += _table(["Значение", "Лидер", "Порядок (балл, %)"], facet_rows)
    lines += ["", f"{h}# По группам", ""]
    group_rows = []
    for group in results.get("groups") or []:
        rated = [r for r in group["variants"] if r["score"] is not None]
        order = " → ".join(f"{r['blind']}={r['name']} ({r['score']})" for r in rated) or "не оценено"
        missing = [r["blind"] for r in group["variants"] if r["score"] is None]
        if rated and missing:
            order += f" · без оценки: {', '.join(missing)}"
        group_rows.append([group["title"], order, group["comment"]])
    lines += _table(["Группа", "Лучший → худший (оценка)", "Комментарий"], group_rows)
    return lines


def _single_markdown(results: dict[str, Any], h: str) -> list[str]:
    summary = results.get("summary") or {}
    labels = results.get("scale_labels") or {}
    label_text = ", ".join(f"{k} — {v}" for k, v in sorted(labels.items(), key=lambda kv: natural_key(kv[0])))
    lines = [f"Каждый клип оценён отдельно{f' ({label_text})' if label_text else ''}. «Слышен брак» — оценка 1–2 "
             f"или тег дефекта; «Чисто на слух» — оценка 4–5 или тег «{CLEAN_TAG}».", "",
             f"{h}# Сводка по ключу", ""]
    crosstab = summary.get("ear_vs_key") or {}
    rows = []
    for stat in summary.get("by_variant") or []:
        cell = crosstab.get(stat["name"], {})
        rows.append([f"**{stat['name']}**", stat["groups_seen"], stat["n"], _num(stat["mean"]),
                     _num(stat["median"], 1), cell.get("defect", 0), cell.get("clean", 0), cell.get("unsure", 0)])
    lines += _table(["В ключе", "Клипов", "Оценено", "Средняя", "Медиана", "Слышен брак", "Чисто на слух",
                     "Не уверен(а)"], rows)
    for facet, values in (summary.get("by_facet") or {}).items():
        lines += ["", f"{h}# По категории «{FACET_TITLES_RU.get(facet, facet)}»", ""]
        facet_rows = [[value, stat["groups_seen"], stat["n"], _num(stat["mean"]), stat.get("defect", 0),
                       stat.get("clean", 0), stat.get("unsure", 0)]
                      for value, stats in values.items() for stat in stats[:1]]
        lines += _table(["Значение", "Клипов", "Оценено", "Средняя", "Слышен брак", "Чисто на слух",
                         "Не уверен(а)"], facet_rows)
    lines += ["", f"{h}# Клипы", ""]
    clip_rows = []
    for group in results.get("groups") or []:
        for row in sorted(group["variants"], key=lambda r: natural_key(r["blind"])):
            clip_rows.append([row["blind"], row["score"] if row["score"] is not None else "—",
                              ", ".join(row["tags"]), VERDICT_TITLES_RU.get(row.get("verdict", ""), ""),
                              row["name"], row["description"], row["note"]])
    lines += _table(["Клип", "Оценка", "Теги", "На слух", "По ключу", "Причины", "Заметка"], clip_rows)
    return lines
