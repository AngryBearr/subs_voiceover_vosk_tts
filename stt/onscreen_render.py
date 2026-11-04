"""Russian versions of the on-screen English text (captions, name cards, lower thirds) as a picture layer.

    items = load_items(Path("onscreen.ru.json"))
    ass = write_ass(items, Path("onscreen.ass"), 1920, 1080)       # soft track: stt.mix.mux_tracks(onscreen_ass=...)
    overlays = render_overlays(items, 1920, 1080)                   # burn-in: stt.video_compress (--onscreen)

Both outputs come from one layout, so they look alike. Captions sit on the English ones (an opaque dark box
covers the English text); name cards and lower thirds go to the opposite side of the screen (mirrored
horizontally, or above the English text when it is near the centre). Overlapping items are stacked.
All item times are on the episode timeline; ``offset_s`` is the start of a test piece (subtracted).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("stt.onscreen_render")

FONT_NAME = "DejaVu Sans"  # Cyrillic-capable; the files are attached to the mkv so players without it still render
FONT_DIRS = (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu"), Path("/usr/share/fonts/TTF"))
LINE_HEIGHT = 1.17  # line height / font size of DejaVu Sans
MARGIN = 0.02  # of the screen: the layer never goes closer to the edge
PAD = 0.012  # of the screen height: space around the text inside its box
CENTRE_ZONE = 0.15  # |centre x - 0.5| below this: an English card is "in the centre"
CAPTION_BG = (12, 12, 12, 255)
CARD_BG = (12, 12, 12, 190)
TEXT_COLOUR = (255, 255, 255, 255)

Rect = tuple[float, float, float, float]


@dataclass
class Overlay:
    """A picture layer for burn-in: ``rgba`` (h, w, 4 uint8) at (x, y) while start <= t < end (seconds)."""

    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray


@dataclass
class Line:
    text: str
    size: int
    bold: bool = False


@dataclass
class Cue:
    """One placed item: text lines centred at ``centre`` and a background rectangle ``rect`` (pixels)."""

    start: float
    end: float
    kind: str
    lines: list[Line]
    centre: tuple[float, float]
    rect: Rect
    bg: tuple[int, int, int, int] = CAPTION_BG
    item_id: int = 0
    # sort key for stacking: the cue keeps its place unless it collides with an earlier one
    order: int = field(default=0, repr=False)


def font_files() -> tuple[Path, Path] | None:
    """(regular, bold) DejaVu Sans files found on this machine, None if there are none."""
    for directory in FONT_DIRS:
        regular, bold = directory / "DejaVuSans.ttf", directory / "DejaVuSans-Bold.ttf"
        if regular.is_file() and bold.is_file():
            return regular, bold
    return None


def font_path() -> Path | None:
    """The regular font file (see ``font_files``)."""
    files = font_files()
    return files[0] if files else None


@lru_cache(maxsize=64)
def _pil_font(size: int, bold: bool) -> Any:
    from PIL import ImageFont

    em = max(1, round(size / LINE_HEIGHT))  # an ASS \fs is the line height, PIL's size is the em
    files = font_files()
    if files is None:
        return ImageFont.load_default(em)
    return ImageFont.truetype(str(files[1 if bold else 0]), em)


def _text_width(text: str, size: int, bold: bool) -> float:
    return float(_pil_font(size, bold).getlength(text))


def load_items(path: Path) -> list[dict[str, Any]]:
    """``onscreen.ru.json``: a list of items or ``{"items": [...]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["items"] if isinstance(data, dict) else data)


# ---------------------------------------------------------------- layout

def _wrap(text: str, size: int, bold: bool, max_width: float, max_lines: int = 2) -> list[str] | None:
    """Greedy word wrap into at most ``max_lines`` lines of at most ``max_width``; None if impossible."""
    words = text.split()
    if not words:
        return None
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if _text_width(f"{current} {word}", size, bold) <= max_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > max_lines or any(_text_width(x, size, bold) > max_width for x in lines):
        return None
    return lines


def _fit(text: str, size: int, bold: bool, max_width: float, min_size: int, max_lines: int = 2) -> tuple[list[str], int]:
    """Lines and font size: the largest size from ``size`` down to ``min_size`` where the text fits."""
    for candidate in range(size, min_size - 1, -2):
        lines = _wrap(text, candidate, bold, max_width, max_lines)
        if lines:
            return lines, candidate
    return _wrap(text, min_size, bold, 1e9, max_lines) or [text], min_size


def _block(lines: list[Line]) -> tuple[float, float]:
    width = max(_text_width(x.text, x.size, x.bold) for x in lines)
    height = sum(x.size * LINE_HEIGHT for x in lines)
    return width, height


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _place(kind: str, lines: list[Line], box: list[float] | None, width: int, height: int) -> Cue | None:
    """Centre and rectangle of one item before stacking (times are filled in by the caller)."""
    text_w, text_h = _block(lines)
    pad = PAD * height
    box_w, box_h = text_w + 2 * pad, text_h + 2 * pad
    margin_x, margin_y = MARGIN * width, MARGIN * height
    if kind == "caption":
        if box:
            ex0, ey0, ex1, ey1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
            cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
            # the English text must be covered: the rectangle is at least its box (plus a small margin)
            cover = (ex0 - pad / 2, ey0 - pad / 2, ex1 + pad / 2, ey1 + pad / 2)
        else:
            cx, cy = width / 2, height * 0.92 - box_h / 2
            cover = (cx, cy, cx, cy)
        rect = (min(cx - box_w / 2, cover[0]), min(cy - box_h / 2, cover[1]),
                max(cx + box_w / 2, cover[2]), max(cy + box_h / 2, cover[3]))
        bg = CAPTION_BG
    else:
        bg = CARD_BG
        if box:
            ex0, ey0, ex1, ey1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
            ecx, ecy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
            if abs(ecx / width - 0.5) < CENTRE_ZONE:  # near the centre: above the English text
                cx, cy = ecx, ey0 - pad - box_h / 2
            else:
                cx, cy = width - ecx, ecy
        else:
            cx, cy = width - margin_x - box_w / 2, margin_y + height * 0.04 + box_h / 2
        rect = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)
    # keep the layer on the screen (the text centre follows the rectangle's shift)
    dx = _clamp(rect[0], margin_x, width - margin_x - (rect[2] - rect[0])) - rect[0]
    dy = _clamp(rect[1], margin_y, height - margin_y - (rect[3] - rect[1])) - rect[1]
    rect = (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy)
    centre = (_clamp(cx + dx, 0, width), cy + dy)
    return Cue(0.0, 0.0, kind, lines, centre, rect, bg)


def _item_lines(item: dict[str, Any], width: int, height: int) -> list[Line]:
    """Font sizes from the English box height, long texts wrapped to two lines (a card: name + descriptor)."""
    kind = item.get("kind", "caption")
    text = str(item.get("text_ru") or "").strip()
    box = item.get("box")
    box_h = (box[3] - box[1]) * height if box else 0.06 * height
    if kind == "caption":
        english_lines = max(1, round(box_h / (0.075 * height)))
        size = int(_clamp(box_h / english_lines * 0.6, 0.032 * height, 0.06 * height))
        max_width = max(0.62 * width, (box[2] - box[0]) * width * 1.1) if box else 0.62 * width
        lines, size = _fit(text, size, False, min(max_width, 0.9 * width), int(0.028 * height))
        return [Line(x, size) for x in lines]
    size = int(_clamp(box_h * 0.5, 0.03 * height, 0.052 * height))
    descriptor = str(item.get("descriptor_ru") or "").strip()
    max_width = 0.36 * width
    if kind == "lower_third":
        lines, size = _fit(text, size, False, max_width, int(0.026 * height))
        return [Line(x, size) for x in lines]
    names, size = _fit(text, size, True, max_width, int(0.026 * height), max_lines=1)
    out = [Line(names[0], size, bold=True)]
    if descriptor:
        small = int(size * 0.75)
        out.append(Line(_fit(descriptor, small, False, max_width, int(0.022 * height), max_lines=1)[0][0], small))
    return out


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def layout(items: list[dict[str, Any]], width: int, height: int, offset_s: float = 0.0) -> list[Cue]:
    """Placed cues on the piece's timeline (times minus ``offset_s``); items without Russian text or outside
    the piece are skipped. Cues that overlap in time and in space are stacked (moved away from the screen edge)."""
    cues: list[Cue] = []
    for n, item in enumerate(items):
        if not str(item.get("text_ru") or "").strip():
            continue
        start, end = float(item["start"]) - offset_s, float(item["end"]) - offset_s
        if end <= 0 or end <= start:
            continue
        cue = _place(item.get("kind", "caption"), _item_lines(item, width, height), item.get("box"), width, height)
        if cue is None:
            continue
        cue.start, cue.end, cue.item_id, cue.order = max(start, 0.0), end, int(item.get("id", n)), n
        cues.append(cue)
    cues.sort(key=lambda c: (c.start, c.order))
    placed: list[Cue] = []
    gap = PAD * height / 2
    for cue in cues:
        for _ in range(20):
            hit = next((p for p in placed if p.start < cue.end and cue.start < p.end and _overlap(p.rect, cue.rect)),
                       None)
            if hit is None:
                break
            up = (cue.rect[1] + cue.rect[3]) / 2 > height / 2  # lower half: stack upwards, else downwards
            dy = (hit.rect[1] - gap - cue.rect[3]) if up else (hit.rect[3] + gap - cue.rect[1])
            cue.rect = (cue.rect[0], cue.rect[1] + dy, cue.rect[2], cue.rect[3] + dy)
            cue.centre = (cue.centre[0], cue.centre[1] + dy)
        placed.append(cue)
    return placed


# ---------------------------------------------------------------- ASS

def _ass_time(seconds: float) -> str:
    centi = int(round(seconds * 100))
    return f"{centi // 360000}:{centi // 6000 % 60:02d}:{centi // 100 % 60:02d}.{centi % 100:02d}"


def _ass_colour(rgba: tuple[int, int, int, int]) -> tuple[str, str]:
    """(``BBGGRR``, ``AA``) of an ASS colour/alpha override (alpha: 00 opaque, FF transparent)."""
    r, g, b, a = rgba
    return f"{b:02X}{g:02X}{r:02X}", f"{255 - a:02X}"


def _ass_text(text: str) -> str:
    return re.sub(r"[{}]", "", text).replace("\n", " ")


def write_ass(items: list[dict[str, Any]], out: Path, width: int, height: int, offset_s: float = 0.0) -> Path | None:
    """Write the items as an ASS file (PlayRes = video size); None (nothing written) when there is nothing to show.
    ``offset_s`` is subtracted from every time (the start of a test piece)."""
    cues = layout(items, width, height, offset_s)
    if not cues:
        return None
    styles = []
    for name in ("Caption", "Card", "LowerThird"):
        styles.append(f"Style: {name},{FONT_NAME},{int(0.045 * height)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
                      f"0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1")
    style_of = {"caption": "Caption", "name_card": "Card", "lower_third": "LowerThird"}
    events = []
    for cue in cues:
        t0, t1 = _ass_time(cue.start), _ass_time(cue.end)
        colour, alpha = _ass_colour(cue.bg)
        x0, y0, x1, y1 = (round(v) for v in cue.rect)
        events.append(f"Dialogue: 0,{t0},{t1},{style_of.get(cue.kind, 'Caption')},,0,0,0,,"
                      f"{{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H{colour}&\\1a&H{alpha}&\\p1}}"
                      f"m {x0} {y0} l {x1} {y0} {x1} {y1} {x0} {y1}{{\\p0}}")
        body = "\\N".join(f"{{\\fs{x.size}\\b{int(x.bold)}}}{_ass_text(x.text)}" for x in cue.lines)
        events.append(f"Dialogue: 1,{t0},{t1},{style_of.get(cue.kind, 'Caption')},,0,0,0,,"
                      f"{{\\an5\\pos({round(cue.centre[0])},{round(cue.centre[1])})}}{body}")
    header = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 2",
              "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
              "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
              "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
              "MarginV,Encoding", *styles, "", "[Events]",
              "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([*header, *events, ""]), encoding="utf-8")
    log.info("%s: %d on-screen cues", out, len(cues))
    return out


# ---------------------------------------------------------------- burn-in layers

def render_overlays(items: list[dict[str, Any]], width: int, height: int, offset_s: float = 0.0) -> list[Overlay]:
    """RGBA pictures (one per cue, only as big as its rectangle) for burning into the video."""
    from PIL import Image, ImageDraw

    overlays: list[Overlay] = []
    for cue in layout(items, width, height, offset_s):
        x0, y0 = int(round(cue.rect[0])), int(round(cue.rect[1]))
        x1, y1 = int(round(cue.rect[2])), int(round(cue.rect[3]))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        image = Image.new("RGBA", (x1 - x0, y1 - y0), cue.bg)
        draw = ImageDraw.Draw(image)
        top = cue.centre[1] - sum(x.size * LINE_HEIGHT for x in cue.lines) / 2 - y0
        for line in cue.lines:
            draw.text((cue.centre[0] - x0, top + line.size * LINE_HEIGHT / 2), line.text,
                      font=_pil_font(line.size, line.bold), fill=TEXT_COLOUR, anchor="mm")
            top += line.size * LINE_HEIGHT
        overlays.append(Overlay(cue.start, cue.end, x0, y0, np.asarray(image, dtype=np.uint8).copy()))
    return overlays


def blend(frame_rgb: np.ndarray, overlay: Overlay) -> None:
    """Alpha-blend ``overlay`` into ``frame_rgb`` (h, w, 3 uint8) in place."""
    h, w = frame_rgb.shape[:2]
    oh, ow = overlay.rgba.shape[:2]
    x1, y1 = min(overlay.x + ow, w), min(overlay.y + oh, h)
    if x1 <= overlay.x or y1 <= overlay.y:
        return
    part = overlay.rgba[: y1 - overlay.y, : x1 - overlay.x]
    alpha = part[..., 3:4].astype(np.float32) / 255.0
    region = frame_rgb[overlay.y:y1, overlay.x:x1].astype(np.float32)
    frame_rgb[overlay.y:y1, overlay.x:x1] = (region * (1 - alpha) + part[..., :3] * alpha + 0.5).astype(np.uint8)
