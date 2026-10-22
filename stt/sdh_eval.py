"""Partial diarization check on real episodes using speaker labels in SDH subtitles.

SDH captions name the speaker on some lines ("GEORGE: ..."), mostly off-screen or
ambiguous voices. For every named person we measure which predicted speaker covers their
labelled lines. Purity = share of a label's time on its main predicted speaker;
merges = named people whose main predicted speaker is also another person's.

    subs_env/bin/python -m stt.sdh_eval --srt ep.en.srt output/stt/ep/diarization.*.json
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stt.diarize import Diarization
from stt.runner import load_json
from utils.srt_to_json import parse_srt_file

_LABEL = re.compile(r"^\s*(?:-\s*)?([A-Z][A-Z .'-]{1,20}):")
# Generic labels that do not identify one person.
_GENERIC = {"MAN", "WOMAN", "MEN", "WOMEN", "ALL", "BOTH", "SURVIVORS", "HEROES", "VILLAINS", "TOGETHER",
            "CROWD", "GROUP", "NARRATOR", "VOICE", "PRODUCER"}


def labelled_spans(srt_path: str | Path) -> list[tuple[str, float, float]]:
    spans = []
    for item in parse_srt_file(str(srt_path)):
        match = _LABEL.match(item["text"][0])
        if match:
            spans.append((match.group(1).strip(), item["start"] / 1000, item["end"] / 1000))
    return spans


def evaluate(spans: list[tuple[str, float, float]], diarization: Diarization) -> dict[str, Any]:
    cover: dict[str, Counter[str]] = defaultdict(Counter)
    for label, start, end in spans:
        if label in _GENERIC:
            continue
        for t in diarization.turns:
            overlap = min(end, t.end) - max(start, t.start)
            if overlap > 0:
                cover[label][t.speaker] += overlap
    tops = {label: c.most_common(1)[0][0] for label, c in cover.items() if c}
    shared = Counter(tops.values())
    total = sum(sum(c.values()) for c in cover.values())
    top_time = sum(c.most_common(1)[0][1] for c in cover.values() if c)
    return {
        "speakers": len(diarization.speakers),
        "people": len(tops),
        "purity": round(top_time / total, 3) if total else None,
        "merged_people": sum(1 for label in tops if shared[tops[label]] > 1),
        "per_label": {label: dict(c.most_common(3)) for label, c in cover.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check diarization against SDH speaker labels")
    parser.add_argument("--srt", required=True, help="SDH subtitles with speaker labels")
    parser.add_argument("diarizations", nargs="+", help="diarization JSON files (stt.worker diarize output)")
    parser.add_argument("--offset", type=float, default=0.0, help="Audio start within the media, seconds")
    args = parser.parse_args(argv)
    spans = [(label, s - args.offset, e - args.offset) for label, s, e in labelled_spans(args.srt)]
    print("| Diarization | Speakers | Named people | Purity | People merged with another |")
    print("|---|---:|---:|---:|---:|")
    for path in args.diarizations:
        r = evaluate(spans, Diarization.from_dict(load_json(Path(path))))
        print(
            f"| {Path(path).name} | {r['speakers']} | {r['people']} | {r['purity'] * 100:.1f}% "
            f"| {r['merged_people']} |"
        )


if __name__ == "__main__":
    main()
