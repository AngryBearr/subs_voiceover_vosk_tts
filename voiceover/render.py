"""Draft voiceover renderer: mixes cached clips at scheduled times for listening checks."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from utils.ffmpeg import ffmpeg_exe

LOGGER = logging.getLogger(__name__)

StretchFn = Callable[[Path, float, int], np.ndarray]

TICK_MS = 25
TICK_HZ = 1000.0
TICK_DBFS = -20.0
MARKER_BEEP_MS = 60
MARKER_HZ = 600.0
MARKER_GAP_MS = 60
MARKER_DBFS = -12.0


@dataclass(frozen=True)
class Segment:
    """A window of the episode, in episode milliseconds."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class RenderOptions:
    sample_rate: int = 22050
    ticks: bool = False
    segment_gap_ms: int = 1500
    segment_marker: bool = True
    loudness_db: float | None = None  # speech level every clip is brought to (dBFS); None: as synthesised
    max_gain_db: float = 20.0


@dataclass(frozen=True)
class RenderSummary:
    duration_sec: float
    units_rendered: int
    units_skipped: int
    overlaps: int
    peak_before_normalise: float


def stretch_clip(path: Path, tempo_factor: float, sample_rate: int) -> np.ndarray:
    """Load a WAV as mono float32 at sample_rate, time-stretched with ffmpeg atempo (pitch-preserving)."""
    stretch = abs(tempo_factor - 1.0) > 1e-6
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.wav"
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path)]
        if stretch:
            cmd += ["-filter:a", f"atempo={tempo_factor:.6f}"]
        cmd += ["-ar", str(sample_rate), "-ac", "1", "-acodec", "pcm_s16le", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.strip()}")
        data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


def speech_level_db(clip: np.ndarray, sample_rate: int) -> float | None:
    """Level of the speech in a clip (dBFS): RMS of its 20 ms frames within 25 dB of the loudest one,
    so pauses and the clip's edges do not count."""
    hop = max(1, int(sample_rate * 0.02))
    frames = len(clip) // hop
    if frames == 0:
        return None
    rms = np.sqrt(np.mean(clip[: frames * hop].reshape(frames, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = rms[db > db.max() - 25]
    return float(20 * np.log10(np.sqrt(np.mean(loud ** 2))))


def level_clip(clip: np.ndarray, sample_rate: int, target_db: float, max_gain_db: float = 20.0,
               peak: float = 0.95) -> np.ndarray:
    """The clip at ``target_db`` of speech level: a cloned voice speaks as loud as its reference clip,
    so lines of different people (and some whispered short lines) came out 8-20 dB apart. The gain
    is at most ``max_gain_db`` and never lifts a peak over ``peak``."""
    level = speech_level_db(clip, sample_rate)
    if level is None or level < -70:  # silence: nothing to level
        return clip
    gain = 10 ** (min(target_db - level, max_gain_db) / 20)
    top = float(np.abs(clip).max()) * gain
    if top > peak:
        gain *= peak / top
    return (clip * gain).astype(np.float32)


def level_track(track: np.ndarray, sample_rate: int, spans: Sequence[tuple[int, int]], target_db: float,
                max_gain_db: float = 20.0, ramp_ms: int = 10) -> np.ndarray:
    """``level_clip`` applied afterwards to the lines of a rendered track (``spans``: start/end ms of
    each placed line), for tracks rendered before the levelling; short ramps between gains."""
    gain = np.ones(len(track), dtype=np.float32)
    ramp = max(1, int(sample_rate * ramp_ms / 1000))
    for start_ms, end_ms in spans:
        a, b = int(sample_rate * start_ms / 1000), min(len(track), int(sample_rate * end_ms / 1000))
        if b - a <= 2 * ramp:
            continue
        piece = track[a:b]
        leveled = level_clip(piece, sample_rate, target_db, max_gain_db)
        top = float(np.abs(piece).max())
        g = float(np.abs(leveled).max()) / top if top > 0 else 1.0
        gain[a:b] = g
        gain[a:a + ramp] = np.linspace(1.0, g, ramp, dtype=np.float32)
        gain[b - ramp:b] = np.linspace(g, 1.0, ramp, dtype=np.float32)
    return (track * gain).astype(np.float32)


def _tone(freq: float, ms: int, dbfs: float, sample_rate: int) -> np.ndarray:
    n = max(1, int(sample_rate * ms / 1000))
    t = np.arange(n) / sample_rate
    wave = np.sin(2 * np.pi * freq * t)
    ramp = min(n // 2, int(sample_rate * 0.002))
    if ramp > 0:
        env = np.ones(n)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        wave = wave * env
    return (wave * 10 ** (dbfs / 20)).astype(np.float32)


def _marker(sample_rate: int) -> np.ndarray:
    beep = _tone(MARKER_HZ, MARKER_BEEP_MS, MARKER_DBFS, sample_rate)
    gap = np.zeros(int(sample_rate * MARKER_GAP_MS / 1000), dtype=np.float32)
    return np.concatenate([beep, gap, beep])


def _add(buf: np.ndarray, offset: int, clip: np.ndarray) -> None:
    buf[offset : offset + len(clip)] += clip


def _fmt(ms: float) -> str:
    """Format milliseconds as mm:ss.s."""
    tenths = int(round(ms / 100))
    minutes, rest = divmod(tenths, 600)
    return f"{minutes:02d}:{rest / 10:04.1f}"


def _fmt_short(ms: float) -> str:
    total = int(round(ms / 1000))
    return f"{total // 60:02d}:{total % 60:02d}"


@dataclass
class _Row:
    placement: Mapping[str, Any]
    offset: int  # sample offset within the segment buffer
    clip: np.ndarray


def render(
    placements: Sequence[Mapping[str, Any]],
    duration_rows: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    segments: list[Segment] | None,
    options: RenderOptions,
    out_wav: Path,
    out_guide: Path,
    stretch: StretchFn = stretch_clip,
) -> RenderSummary:
    """Render a draft WAV and a Markdown listening guide from a schedule."""
    sr = options.sample_rate
    by_unit = {row["unit_id"]: row for row in duration_rows}
    ordered = sorted(placements, key=lambda p: (p["start_ms"], p["unit_id"]))
    if segments is None:
        last = max((int(p["end_ms"]) for p in ordered), default=0)
        segments = [Segment(0, last + 500)]

    gap = np.zeros(int(sr * options.segment_gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    guide: list[str] = ["# Руководство по прослушиванию", ""]
    rendered = skipped = overlaps = clamped = 0
    out_pos = 0  # samples written so far

    for index, seg in enumerate(segments, start=1):
        seg_samples = int(sr * (seg.end_ms - seg.start_ms) / 1000)
        rows: list[_Row] = []
        for p in ordered:
            if not seg.start_ms <= int(p["cue_start_ms"]) < seg.end_ms:
                continue
            drow = by_unit.get(p["unit_id"])
            if drow is None or drow.get("error") or not drow.get("path"):
                skipped += 1
                continue
            rel = int(p["start_ms"]) - seg.start_ms
            if rel < 0:
                clamped += 1
                rel = 0
            clip = stretch(Path(drow["path"]), float(p["tempo_factor"]), sr)
            if options.loudness_db is not None:
                clip = level_clip(clip, sr, options.loudness_db, options.max_gain_db)
            rows.append(_Row(p, int(sr * rel / 1000), clip))
        length = max([seg_samples] + [r.offset + len(r.clip) for r in rows])
        buf = np.zeros(length, dtype=np.float32)
        for r in rows:
            _add(buf, r.offset, r.clip)
        rendered += len(rows)
        for a, b in zip(rows, rows[1:]):
            if a.offset + len(a.clip) > b.offset:
                overlaps += 1
        if options.ticks:
            tick = _tone(TICK_HZ, TICK_MS, TICK_DBFS, sr)
            for r in rows:
                p = r.placement
                if p.get("members"):
                    starts = [int(m["start"]) for m in p["members"]]
                else:
                    starts = [int(p["cue_start_ms"])]
                for s in starts:
                    if seg.start_ms <= s < seg.end_ms:
                        _add(buf, int(sr * (s - seg.start_ms) / 1000), tick[: max(0, length - int(sr * (s - seg.start_ms) / 1000))])
        if options.segment_marker:
            m = _marker(sr)
            buf[: len(m)] += m[: len(buf)]

        file_start_ms = out_pos * 1000 / sr
        guide += [
            f"## Фрагмент {index}: эпизод {_fmt_short(seg.start_ms)}–{_fmt_short(seg.end_ms)} "
            f"(файл {_fmt_short(file_start_ms)}–{_fmt_short(file_start_ms + len(buf) * 1000 / sr)})",
            "",
            "| время в файле | время в эпизоде | статус | скорость | сдвиг мс | вырезать с | слито | дрейф мс | текст |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            p = r.placement
            removed = float(p.get("removed_sec", 0.0))
            members = p.get("members") or []
            merged = bool(p.get("merged"))
            drift = max((abs(int(m["drift_ms"])) for m in members[1:]), default=0) if merged else None
            text = str(by_unit[p["unit_id"]].get("display_text", "")).replace("|", "\\|").replace("\n", " ")
            if len(text) > 80:
                text = text[:79] + "…"
            guide.append(
                "| " + " | ".join([
                    _fmt(file_start_ms + r.offset * 1000 / sr),
                    _fmt(int(p["start_ms"])),
                    str(p["status"]),
                    f"{float(p['total_speed']):.2f}",
                    str(int(p["shift_ms"])),
                    f"{removed:.1f}" if removed >= 0.05 else "",
                    str(len(p["cue_ids"])) if merged else "",
                    str(drift) if drift is not None else "",
                    text,
                ]) + " |"
            )
        guide.append("")
        parts.append(buf)
        out_pos += len(buf)
        if index < len(segments):
            parts.append(gap)
            out_pos += len(gap)

    mix = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    peak = float(np.abs(mix).max()) if len(mix) else 0.0
    if peak > 1.0:
        mix = mix * (10 ** (-1.0 / 20) / peak)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), mix, sr, subtype="PCM_16")
    if clamped:
        guide += [f"Примечание: {clamped} реплик(и) начинались раньше окна и сдвинуты к началу фрагмента.", ""]
        LOGGER.info("%d placements clamped to segment start", clamped)
    out_guide.parent.mkdir(parents=True, exist_ok=True)
    out_guide.write_text("\n".join(guide), encoding="utf-8")
    return RenderSummary(len(mix) / sr, rendered, skipped, overlaps, peak)


def parse_segments(spec: str) -> list[Segment]:
    """Parse '45-120,450-540' (episode seconds) into segments."""
    result: list[Segment] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, sep, b = chunk.partition("-")
        if not sep:
            raise ValueError(f"bad segment {chunk!r}, expected start-end")
        start, end = float(a), float(b)
        if end <= start:
            raise ValueError(f"segment end must exceed start: {chunk!r}")
        result.append(Segment(int(round(start * 1000)), int(round(end * 1000))))
    if not result:
        raise ValueError("no segments given")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a draft voiceover WAV from a schedule.")
    parser.add_argument("--run", type=Path, required=True, help="run dir with schedule0/durations0/cues json")
    parser.add_argument("--out", type=Path, required=True, help="output WAV path")
    parser.add_argument("--segments", default=None, help="comma list of start-end in episode seconds")
    parser.add_argument("--ticks", action="store_true", help="add a tick at every subtitle cue start")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    schedule = json.loads((args.run / "schedule0.json").read_text(encoding="utf-8"))
    durations = json.loads((args.run / "durations0.json").read_text(encoding="utf-8"))
    cues = json.loads((args.run / "cues.json").read_text(encoding="utf-8"))["cues"]
    segments = parse_segments(args.segments) if args.segments else None
    options = RenderOptions(sample_rate=args.sample_rate, ticks=args.ticks)
    guide = args.out.with_name(f"{args.out.stem}_guide.md")
    s = render(schedule["placements"], durations["rows"], cues, segments, options, args.out, guide)
    LOGGER.info("summary: %s", s)
    print(
        f"{args.out}: {s.duration_sec:.1f}s, rendered={s.units_rendered}, skipped={s.units_skipped}, "
        f"overlaps={s.overlaps}, peak={s.peak_before_normalise:.3f}, guide={guide}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
