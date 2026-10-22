"""Side-by-side comparison of translation variants of one transcription folder.

    subs_env/bin/python -m stt.compare_translations output/stt/ep01 --variants luna deepseek \
        [--ass existing.ru.ass] [--cps 14] [--from 0 --to 600]

Variant `X` is `subs.ru.X.json` (written by `stt.translate --suffix .X`). Writes
`translations.html` (all lines, speaker/gender, EN, existing subtitles, every variant;
over-budget lines highlighted) and prints a cost/time/length summary.
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
from pathlib import Path
from typing import Any

from stt.runner import load_json

log = logging.getLogger("stt.compare_translations")

_GENDER_MARK = {"female": "♀", "male": "♂"}


def _budget(record: dict[str, Any], cps: float) -> int:
    """Budget stored by stt.translate, or a flat-rate estimate for older outputs."""
    return record.get("budget_chars") or max(8, round(record["duration"] / 1000 * cps))


def _overlapping_text(cues: list[Any], start_s: float, end_s: float) -> str:
    texts = [c.text for c in cues if c.end > start_s and c.start < end_s
             and min(c.end, end_s) - max(c.start, start_s) >= 0.3 * min(c.end - c.start, end_s - start_s)]
    return " / ".join(texts)


def summary_rows(folder: Path, variants: list[str], records_by: dict[str, list[dict[str, Any]]],
                 cps: float) -> list[dict[str, Any]]:
    rows = []
    for v in variants:
        usage_path = folder / f"subs.ru.{v}.usage.json"
        usage = load_json(usage_path) if usage_path.exists() else {}
        recs = records_by[v]
        chars = sum(len(r["text"][0]) for r in recs)
        seconds = sum(r["duration"] for r in recs) / 1000
        rows.append(
            {
                "variant": v,
                "model": usage.get("model", "?"),
                "cost_usd": usage.get("usage", {}).get("cost_usd"),
                "elapsed_s": usage.get("elapsed_s"),
                "lines": len(recs),
                "over_budget": sum(len(r["text"][0]) > _budget(r, cps) * 1.2 for r in recs),
                "chars_per_s": round(chars / seconds, 1) if seconds else None,
            }
        )
    return rows


def render_html(
    title: str,
    base: list[dict[str, Any]],
    speakers: dict[str, dict[str, Any]],
    variants: list[str],
    records_by: dict[str, list[dict[str, Any]]],
    summary: list[dict[str, Any]],
    ass_cues: list[Any] | None,
    cps: float,
) -> str:
    esc = html.escape
    head = ["#", "Time", "Speaker", "EN"] + (["Existing RU (.ass)"] if ass_cues is not None else []) + variants
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(title)}</title><style>",
        "body{font-family:system-ui,sans-serif;font-size:14px;margin:16px;color:#1d1d1f}",
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:4px 6px;vertical-align:top}",
        "th{background:#f3f3f5;position:sticky;top:0}td.over{background:#fff1e6}",
        "td.t{white-space:nowrap;color:#666;font-size:12px}.f{color:#b0306a}.m{color:#2a5db0}.u{color:#888}",
        ".summary td,.summary th{position:static}",
        "</style></head><body>",
        f"<h2>{esc(title)}</h2>",
        f"<p>Highlighted: longer than 1.2 × budget ({cps:g} chars/s × cue duration).</p>",
        "<table class='summary'><tr><th>Variant</th><th>Model</th><th>Cost, $</th><th>Time, s</th>"
        "<th>Lines</th><th>Over budget</th><th>RU chars/s</th></tr>",
    ]
    for r in summary:
        out.append(
            f"<tr><td>{esc(r['variant'])}</td><td>{esc(r['model'])}</td><td>{r['cost_usd']}</td>"
            f"<td>{r['elapsed_s']}</td><td>{r['lines']}</td><td>{r['over_budget']}</td><td>{r['chars_per_s']}</td></tr>"
        )
    out.append("</table><br><table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>")
    by_index = {v: {r["index"]: r for r in records_by[v]} for v in variants}
    for rec in base:
        sid = rec.get("speaker") or "??"
        card = speakers.get(sid, {})
        gender = rec.get("gender", "unknown")
        cls = {"female": "f", "male": "m"}.get(gender, "u")
        label = card.get("label_ru") or card.get("name") or ""
        start = rec["start"] / 1000
        cells = [
            f"<td>{rec['index']}</td>",
            f"<td class='t'>{int(start // 60)}:{start % 60:04.1f}<br>{rec['duration'] / 1000:.1f}s</td>",
            f"<td class='{cls}'>{esc(sid)} {_GENDER_MARK.get(gender, '?')}<br>{esc(label)}</td>",
            f"<td>{esc(' '.join(rec.get('text_en') or rec['text']))}</td>",
        ]
        if ass_cues is not None:
            cells.append(f"<td>{esc(_overlapping_text(ass_cues, start, rec['end'] / 1000))}</td>")
        for v in variants:
            r = by_index[v].get(rec["index"])
            text = r["text"][0] if r else ""
            over = r is not None and len(text) > _budget(r, cps) * 1.2
            cells.append(f"<td class='{'over' if over else ''}'>{esc(text)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table></body></html>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare translation variants side by side")
    parser.add_argument("folder", help="Output folder of stt.transcribe")
    parser.add_argument("--variants", nargs="+", required=True, help="Suffixes X of subs.ru.X.json")
    parser.add_argument("--ass", default=None, help="Existing Russian subtitles (.ass) to show for reference")
    parser.add_argument("--cps", type=float, default=14.0)
    parser.add_argument("--from", dest="from_s", type=float, default=None, help="Only lines after this time, s")
    parser.add_argument("--to", dest="to_s", type=float, default=None, help="Only lines before this time, s")
    parser.add_argument("--out", default=None, help="HTML path (default: <folder>/translations.html)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    folder = Path(args.folder)
    records_by = {v: load_json(folder / f"subs.ru.{v}.json") for v in args.variants}
    base = records_by[args.variants[0]]
    if args.from_s is not None:
        base = [r for r in base if r["start"] >= args.from_s * 1000]
    if args.to_s is not None:
        base = [r for r in base if r["start"] < args.to_s * 1000]
    speakers = {s["id"]: s for s in load_json(folder / "speakers.json")}
    ass_cues = None
    if args.ass:
        from stt.reference import load_ass_cues

        ass_cues = load_ass_cues(args.ass, clean=True)
    summary = summary_rows(folder, args.variants, records_by, args.cps)
    page = render_html(f"Translations: {folder.name}", base, speakers, args.variants, records_by, summary,
                       ass_cues, args.cps)
    out = Path(args.out) if args.out else folder / "translations.html"
    out.write_text(page, encoding="utf-8")
    for r in summary:
        log.info("%-12s %-40s $%-10s %6ss over=%d chars/s=%s", r["variant"], r["model"], r["cost_usd"],
                 r["elapsed_s"], r["over_budget"], r["chars_per_s"])
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
