"""On-screen text (OCR): burned-in captions and name cards with their time ranges.

Reality TV captions hard-to-hear speech on screen and shows name cards when people first
speak. Captions are added to the transcript where ASR missed them; name cards become strong
evidence for speaker naming. Uses RapidOCR (PP-OCR ONNX models, Apache-2.0).

Frames are sampled at `fps` from reference frames only (B-frames are not decoded), scaled to
`width` and OCR'd only when the picture changed noticeably since the last OCR'd frame.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Detection:
    """One text line in one sampled frame; box is normalized (x0, y0, x1, y1)."""

    time: float
    text: str
    score: float
    box: tuple[float, float, float, float]


@dataclass
class TextEvent:
    """A text line visible from `start` to `end` (seconds on the media timeline)."""

    start: float
    end: float
    text: str
    kind: str  # caption | name_card | name_card_descriptor | title | overlay | scene
    box: tuple[float, float, float, float]
    score: float
    frames: int = 0
    variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(text: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(text))


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


class FrameOcr:
    """RapidOCR: text detection on the GPU, line recognition on the CPU.

    PP-OCR recognition on CUDA re-plans cuDNN for every new crop width (~300 ms per frame);
    on the CPU it takes a few ms per line, while detection is 5x faster on the GPU.
    Lines seen at the same place with the same look in the previous frame reuse their text.
    """

    def __init__(self, device: str = "cuda", min_score: float = 0.8) -> None:
        self.device = device
        self.min_score = min_score
        self._previous: list[tuple[tuple[float, float, float, float], np.ndarray, str, float]] = []

    def load(self) -> None:
        from rapidocr import LangRec, RapidOCR

        if self.device == "cuda":
            from stt.gpu import preload_cuda_libs

            preload_cuda_libs()
        common = {"Rec.lang_type": LangRec.EN, "Global.log_level": "error", "Det.box_thresh": 0.6}
        self._det = RapidOCR(params={**common, "EngineConfig.onnxruntime.use_cuda": self.device == "cuda"})
        self._rec = RapidOCR(params={**common, "EngineConfig.onnxruntime.use_cuda": False})

    @staticmethod
    def _thumb(crop: np.ndarray) -> np.ndarray:
        gray = crop.mean(axis=2)
        ys = np.linspace(0, gray.shape[0] - 1, 16).astype(int)
        xs = np.linspace(0, gray.shape[1] - 1, 64).astype(int)
        return gray[np.ix_(ys, xs)]

    def _reuse(self, box: tuple[float, float, float, float], thumb: np.ndarray) -> tuple[str, float] | None:
        for pbox, pthumb, text, score in self._previous:
            if max(abs(a - b) for a, b in zip(box, pbox)) < 0.01 and float(np.abs(thumb - pthumb).mean()) < 8:
                return text, score
        return None

    def __call__(self, image: np.ndarray, time: float) -> list[Detection]:
        h, w = image.shape[:2]
        det = self._det(image, use_cls=False, use_rec=False)
        boxes = det.boxes if det.boxes is not None else []
        current = []
        out: list[Detection] = []
        for box in boxes:
            xs, ys = box[:, 0], box[:, 1]
            x0, x1 = max(0, int(xs.min()) - 2), min(w, int(xs.max()) + 2)
            y0, y1 = max(0, int(ys.min()) - 2), min(h, int(ys.max()) + 2)
            if y1 - y0 < 8 or x1 - x0 < 8:
                continue
            nbox = (round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4))
            crop = image[y0:y1, x0:x1]
            thumb = self._thumb(crop)
            known = self._reuse(nbox, thumb)
            if known is None:
                rec = self._rec(crop, use_det=False, use_cls=False)
                text = str(rec.txts[0]).strip() if rec.txts else ""
                score = float(rec.scores[0]) if rec.scores else 0.0
            else:
                text, score = known
            current.append((nbox, thumb, text, score))
            if score >= self.min_score and _WORD.search(text):
                out.append(Detection(time, text, round(score, 3), nbox))
        self._previous = current
        return out


def scan_video(
    path: str | Path,
    ocr: FrameOcr,
    *,
    start: float = 0.0,
    duration: float | None = None,
    fps: float = 2.0,
    width: int = 960,
    change_threshold: float = 2.0,
) -> tuple[list[Detection], dict[str, Any]]:
    """Detections of sampled frames; unchanged frames reuse the previous OCR result."""
    import av

    detections: list[Detection] = []
    end = None if duration is None else start + duration
    step = 1.0 / fps
    next_time = start
    last_thumb: np.ndarray | None = None
    last_result: list[Detection] = []
    stats = {"sampled": 0, "ocr_runs": 0}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "NONREF"
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        height = int(round(width * stream.codec_context.height / stream.codec_context.width / 2) * 2)
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < next_time:
                continue
            if end is not None and t > end:
                break
            next_time = t + step
            stats["sampled"] += 1
            image = frame.to_ndarray(width=width, height=height, format="rgb24")
            thumb = image[::8, ::8].mean(axis=2)
            if last_thumb is not None and float(np.abs(thumb - last_thumb).mean()) < change_threshold:
                detections += [Detection(t, d.text, d.score, d.box) for d in last_result]
                continue
            last_thumb = thumb
            last_result = ocr(image, t)
            stats["ocr_runs"] += 1
            detections += last_result
    return detections, stats


def group_events(detections: list[Detection], step: float, *, max_gap: float = 1.6,
                 min_similarity: float = 0.8) -> list[TextEvent]:
    """Join detections of the same line in consecutive samples into timed events."""
    open_events: list[TextEvent] = []
    done: list[TextEvent] = []
    for det in sorted(detections, key=lambda d: d.time):
        for ev in list(open_events):
            if det.time - ev.end > max_gap:
                open_events.remove(ev)
                done.append(ev)
        match = None
        for ev in open_events:
            same_place = abs((ev.box[1] + ev.box[3]) / 2 - (det.box[1] + det.box[3]) / 2) < 0.06
            if same_place and similar(ev.text, det.text) >= min_similarity:
                match = ev
                break
        if match is None:
            open_events.append(TextEvent(det.time, det.time, det.text, "", det.box, det.score, 1, [det.text]))
            continue
        match.end = det.time
        match.frames += 1
        match.variants.append(det.text)
        if det.score > match.score:  # keep the most confident reading
            match.text, match.score, match.box = det.text, det.score, det.box
    done += open_events
    for ev in done:
        ev.end = round(ev.end + step, 3)
        ev.start = round(ev.start, 3)
        ev.variants = sorted(set(ev.variants))[:5]
    return sorted(done, key=lambda e: e.start)


_URL = re.compile(r"(www\.|https?://|\b\w+\.(com|au|net|org|tv|c)\b)", re.IGNORECASE)
_TITLE_WORD = re.compile(r"^([A-Z][A-Za-z'&.-]*|by|of|and|the|for|&)$")
_ENDS_SENTENCE = re.compile(r"[.!?,…\"']$")


def title_case(text: str) -> bool:
    """"Wardrobe Supervisor", "Art Department Buyer": a credit/title line, not dialogue."""
    words = text.split()
    return (len(words) >= 2 and not _ENDS_SENTENCE.search(text.strip())
            and sum(bool(_TITLE_WORD.match(w)) for w in words) >= 0.75 * len(words))
_SPEAKER_PREFIX = re.compile(r"^([A-Z][A-Za-z'-]{1,20}):\s+(.+)$")


def classify(events: list[TextEvent], total_s: float) -> list[TextEvent]:
    """Label events.

    caption     centred in the lower band, sentence-like: mixed case, not a Title Case line, no URLs;
    lower_third an upper-case line of 2+ words low on screen ("ARMY WIFE & COMBAT PILOT",
                "CURRENTLY IN 1ST PLACE", "THE LEGEND."): a graphic, not speech;
    name_card   lower-third name of the person on screen (left, large) - the descriptor line
                under it ("NEWS ANCHOR  VILLAINS") becomes `descriptor`;
    title       one huge upper-case word held for 2+ s (contestant intro titles; may be a place);
    credits     a roll of short caption-like lines (4+ within a few seconds, mostly titles);
    overlay     the same text recurring through the episode (hashtags, logos);
    scene       everything else (signs, props, clothes, written votes).
    """
    recurring: dict[str, float] = {}
    for ev in events:
        key = _norm(ev.text)
        recurring[key] = recurring.get(key, 0.0) + (ev.end - ev.start)
    for ev in events:
        x0, y0, x1, y1 = ev.box
        height = y1 - y0
        words = _WORD.findall(ev.text)
        centre = (x0 + x1) / 2
        letters = [c for c in ev.text if c.isalpha()]
        upper = bool(letters) and sum(c.isupper() for c in letters) >= 0.8 * len(letters)  # OCR may miss a few
        sentence_like = bool(re.search(r"[a-z]", ev.text)) and not upper
        if recurring[_norm(ev.text)] > max(60.0, 0.03 * total_s):
            ev.kind = "overlay"
        elif (y0 > 0.74 and abs(centre - 0.5) < 0.18 and sentence_like and not _URL.search(ev.text)
              and (len(words) >= 2 or re.search(r"[.!?]$", ev.text)) and not title_case(ev.text)):
            ev.kind = "caption"
        elif upper and x0 < 0.15 and 0.66 < y0 < 0.8 and 0.09 <= height <= 0.18 and len(words) <= 3:
            ev.kind = "name_card"
        elif upper and height >= 0.2 and len(words) == 1 and len(words[0]) >= 3 and ev.end - ev.start >= 2.0:
            ev.kind = "title"
        elif upper and len(words) >= 2 and y0 > 0.6:
            ev.kind = "lower_third"
        else:
            ev.kind = "scene"
    _mark_credits([ev for ev in events if ev.kind == "caption" or (ev.kind == "scene" and title_case(ev.text))])
    titles = [ev for ev in events if ev.kind == "title"]
    for ev in titles:  # logo variants ("AUSTRALIAN", "SURVYVOR") recur; real names appear once or twice
        if sum(1 for other in titles if similar(ev.text, other.text) >= 0.7) >= 3:
            ev.kind = "overlay"
    for ev in events:  # descriptor line right under a lower-third name, shown at the same time
        if ev.kind != "name_card":
            continue
        for other in events:
            if (other is not ev and abs(other.start - ev.start) < 0.3 and other.box[0] < 0.15
                    and 0.0 < other.box[1] - ev.box[1] < 0.2 and other.text.upper() == other.text):
                ev.variants = [*ev.variants, f"descriptor: {other.text}"]
                other.kind = "name_card_descriptor"
    return events


def _mark_credits(lines: list[TextEvent], max_gap: float = 3.0, min_run: int = 4) -> None:
    """Runs of short lines a few seconds apart, mostly without sentence punctuation, are credits."""
    lines = sorted(lines, key=lambda e: e.start)
    run: list[TextEvent] = []
    for ev in lines + [None]:  # type: ignore[list-item]
        if ev is not None and run and ev.start - run[-1].start <= max_gap:
            run.append(ev)
            continue
        if len(run) >= min_run and sum(title_case(e.text) or not _ENDS_SENTENCE.search(e.text.strip())
                                       for e in run) >= 0.6 * len(run):
            for e in run:
                e.kind = "credits"
        run = [ev] if ev is not None else []


@dataclass
class Caption:
    """One on-screen subtitle (both lines joined) with an optional speaker label ("Shonee: ...")."""

    start: float
    end: float
    text: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def captions(events: list[TextEvent]) -> list[Caption]:
    """Join the lines of two-line subtitles (same time span, stacked) and split off speaker labels."""
    lines = sorted((e for e in events if e.kind == "caption"), key=lambda e: (e.start, e.box[1]))
    out: list[Caption] = []
    last_box: tuple[float, float, float, float] | None = None
    for ev in lines:
        if out and last_box and abs(ev.start - out[-1].start) < 0.3 and abs(ev.end - out[-1].end) < 0.6 \
                and ev.box[1] > last_box[1]:
            out[-1].text = f"{out[-1].text} {ev.text}"
        else:
            out.append(Caption(ev.start, ev.end, ev.text))
        last_box = ev.box
    for cap in out:
        match = _SPEAKER_PREFIX.match(cap.text)
        if match and match.group(1).lower() not in {"note", "warning"}:
            cap.label, cap.text = match.group(1), match.group(2)
    return out


def summarize(events: list[TextEvent], total_s: float) -> dict[str, Any]:
    """Classify timed text events into captions (speech) and cards (names, titles, lower thirds)."""
    for ev in events:  # descriptors are re-derived by classify
        ev.variants = [v for v in ev.variants if not v.startswith("descriptor: ")]
    events = classify(events, max(total_s, 1.0))
    cards = [
        {"start": ev.start, "end": ev.end, "kind": ev.kind, "text": ev.text,
         "descriptor": next((v.split(": ", 1)[1] for v in ev.variants if v.startswith("descriptor: ")), None)}
        for ev in events if ev.kind in ("name_card", "title", "lower_third")
    ]
    return {"captions": [c.to_dict() for c in captions(events)], "cards": cards,
            "events": [ev.to_dict() for ev in events]}


def reclassify(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the current rules to the events of a saved scan (no new video pass needed)."""
    events = [TextEvent(**{**e, "box": tuple(e["box"])}) for e in data.get("events", [])]
    total = max((e.end for e in events), default=0.0)
    return {**data, **summarize(events, total)}


def run_ocr(path: str | Path, ocr: FrameOcr, *, start: float = 0.0, duration: float | None = None,
            fps: float = 2.0) -> dict[str, Any]:
    """Scan a media interval; times in the result are relative to `start` (the audio timeline)."""
    detections, stats = scan_video(path, ocr, start=start, duration=duration, fps=fps)
    total = duration if duration is not None else max((d.time for d in detections), default=0.0) - start
    events = group_events(detections, 1.0 / fps)
    for ev in events:
        ev.start, ev.end = round(ev.start - start, 3), round(ev.end - start, 3)
    return {"fps": fps, "stats": stats, **summarize(events, total)}
