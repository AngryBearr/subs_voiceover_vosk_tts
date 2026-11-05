"""On-screen translations: ASS layout, burn-in layers, muxing and burning into frames (synthetic data only)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from stt import mix
from stt.onscreen_render import Overlay, blend, font_files, layout, load_items, render_overlays, write_ass
from stt.video_compress import encode_video
from utils.ffmpeg import ffmpeg_exe

W, H = 1920, 1080
CAPTION_BOX = [0.3823, 0.8167, 0.6198, 0.9019]
CARD_BOX = [0.1354, 0.7352, 0.3156, 0.8463]


def item(kind: str, start: float, end: float, text_ru: str, box: list[float] | None, descriptor_ru: str | None = None,
         n: int = 1) -> dict[str, Any]:
    return {"id": n, "kind": kind, "start": start, "end": end, "text": "x", "descriptor": None, "box": box,
            "voiced": False, "text_ru": text_ru, "descriptor_ru": descriptor_ru}


def cue_by_kind(cues: list, kind: str):  # noqa: ANN001
    return next(c for c in cues if c.kind == kind)


def test_caption_sits_on_the_english_box_and_covers_it() -> None:
    cue = layout([item("caption", 1, 2, "Там лодка.", CAPTION_BOX)], W, H)[0]
    ex = (CAPTION_BOX[0] + CAPTION_BOX[2]) / 2 * W
    ey = (CAPTION_BOX[1] + CAPTION_BOX[3]) / 2 * H
    assert cue.centre == pytest.approx((ex, ey), abs=1)
    assert cue.rect[0] <= CAPTION_BOX[0] * W and cue.rect[2] >= CAPTION_BOX[2] * W
    assert cue.rect[1] <= CAPTION_BOX[1] * H and cue.rect[3] >= CAPTION_BOX[3] * H


def covers(rect: tuple[float, float, float, float], box: list[float]) -> bool:
    return (rect[0] <= box[0] * W and rect[1] <= box[1] * H and rect[2] >= box[2] * W and rect[3] >= box[3] * H)


def test_name_card_and_lower_third_cover_the_english_box() -> None:
    third_box = [0.1, 0.2, 0.3, 0.26]
    cards = layout([item("name_card", 1, 3, "ДЕВИН", CARD_BOX, "АКТЁР"),
                    item("lower_third", 1, 3, "ДУХ ТИХОГО ОКЕАНА", third_box, n=2)], W, H)
    card, third = cue_by_kind(cards, "name_card"), cue_by_kind(cards, "lower_third")
    assert [x.text for x in card.lines] == ["ДЕВИН", "АКТЁР"] and card.lines[0].bold and not card.lines[1].bold
    assert covers(card.rect, CARD_BOX) and covers(third.rect, third_box)
    # not mirrored: the text stays on the English side of the screen
    assert card.centre[0] == pytest.approx((CARD_BOX[0] + CARD_BOX[2]) / 2 * W, abs=0.15 * W)
    assert card.centre[0] < 0.5 * W and third.centre[0] < 0.5 * W
    assert card.bg[3] == 217  # about 85% opaque
    # a small padding only: the box does not wander far from the English one
    assert card.rect[0] >= CARD_BOX[0] * W - 0.1 * W and card.rect[3] <= CARD_BOX[3] * H + 0.05 * H


def test_multi_line_card_and_lower_third_use_their_lines() -> None:
    card_box, third_box = [0.05, 0.70, 0.25, 0.80], [0.70, 0.20, 0.95, 0.30]
    cues = layout([item("name_card", 1, 3, "ХОЛЛИ", card_box, "ТРЕНЕР ПО ПЛАВАНИЮ\nПЛЕМЯ ЭСПАДА"),
                   item("lower_third", 1, 3, "ЭСПАДА - старшее племя\nДЕНЬ 1", third_box, n=2)], W, H)
    card, third = cue_by_kind(cues, "name_card"), cue_by_kind(cues, "lower_third")
    assert [x.text for x in card.lines] == ["ХОЛЛИ", "ТРЕНЕР ПО ПЛАВАНИЮ", "ПЛЕМЯ ЭСПАДА"]
    assert card.lines[0].bold and card.lines[1].size < card.lines[0].size
    assert [x.text for x in third.lines] == ["ЭСПАДА - старшее племя", "ДЕНЬ 1"]
    for cue, box in ((card, card_box), (third, third_box)):
        assert covers(cue.rect, box)
        assert cue.rect[2] - cue.rect[0] <= (box[2] - box[0]) * W * 1.3 + 0.05 * W  # at most ~30% wider
        assert 0 <= cue.rect[0] and cue.rect[2] <= W and 0 <= cue.rect[1] and cue.rect[3] <= H
        block = sum(x.size * 1.17 for x in cue.lines)
        assert block <= cue.rect[3] - cue.rect[1]  # the text fits its box


def test_long_russian_card_shrinks_or_wraps_within_the_width() -> None:
    box = [0.05, 0.70, 0.20, 0.78]
    cue = layout([item("name_card", 1, 3, "АЛЕКСАНДРА", box, "ПРОФЕССИОНАЛЬНЫЙ ГОНЩИК\nПЛЕМЯ")], W, H)[0]
    assert covers(cue.rect, box) and cue.rect[2] - cue.rect[0] <= (box[2] - box[0]) * W * 1.3 + 0.05 * W
    assert len(cue.lines) >= 3


def test_old_items_without_lines_still_render() -> None:
    cue = layout([item("lower_third", 1, 2, "ДУХ", [0.7, 0.2, 0.9, 0.25])], W, H)[0]
    assert [x.text for x in cue.lines] == ["ДУХ"]


def test_items_without_box_use_defaults() -> None:
    caption = layout([item("caption", 1, 2, "Привет", None)], W, H)[0]
    assert caption.centre[0] == pytest.approx(W / 2, abs=1) and caption.centre[1] > 0.8 * H


def test_stacked_cards_never_overlap_and_stay_on_screen() -> None:
    cues = layout([item("name_card", 1, 3, "ДЕВИН", CARD_BOX, "АКТЁР"),
                   item("lower_third", 2, 4, "ДУХ\nДЕНЬ 1", CARD_BOX, n=2)], W, H)
    a, b = cues
    assert a.rect[1] >= b.rect[3] or b.rect[1] >= a.rect[3]
    assert all(0 <= c.rect[1] and c.rect[3] <= H for c in cues)


def test_overlapping_items_are_stacked_not_overlapped() -> None:
    cues = layout([item("caption", 1, 3, "Первая", CAPTION_BOX), item("caption", 2, 4, "Вторая", CAPTION_BOX, n=2),
                   item("caption", 5, 6, "Позже", CAPTION_BOX, n=3)], W, H)
    first, second, later = cues
    assert first.rect[1] >= second.rect[3] or second.rect[1] >= first.rect[3]
    assert later.rect == first.rect  # no time overlap: stays on the English box


def test_long_caption_wraps_to_two_lines_and_fits() -> None:
    text = "Я ждал племена, а получил совсем не то, чего хотел услышать в первый день игры"
    cue = layout([item("caption", 1, 2, text, CAPTION_BOX)], W, H)[0]
    assert len(cue.lines) <= 2
    assert cue.rect[0] >= 0 and cue.rect[2] <= W and cue.rect[3] <= H


def test_offset_and_skipped_items() -> None:
    items = [item("caption", 100, 102, "В куске", CAPTION_BOX), item("caption", 10, 12, "До начала", CAPTION_BOX, n=2),
             item("caption", 100, 102, "", CAPTION_BOX, n=3)]
    cues = layout(items, W, H, offset_s=90)
    assert [(c.start, c.end) for c in cues] == [(10.0, 12.0)]


def test_write_ass(tmp_path: Path) -> None:
    items = [item("caption", 1.5, 3.25, "Там лодка.", CAPTION_BOX),
             item("name_card", 4, 6, "ДЕВИН", CARD_BOX, "АКТЁР", n=2),
             item("lower_third", 4, 6, "ДУХ", [0.7, 0.2, 0.9, 0.25], n=3)]
    path = write_ass(items, tmp_path / "o.ass", W, H, offset_s=1.0)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in text and "PlayResY: 1080" in text
    for style in ("Caption", "Card", "LowerThird"):
        assert re.search(rf"^Style: {style},DejaVu Sans,", text, re.M)
    assert "Dialogue: 1,0:00:00.50,0:00:02.25,Caption" in text  # offset subtracted
    ex, ey = round((CAPTION_BOX[0] + CAPTION_BOX[2]) / 2 * W), round((CAPTION_BOX[1] + CAPTION_BOX[3]) / 2 * H)
    assert f"\\pos({ex},{ey})" in text
    card = cue_by_kind(layout(items, W, H, offset_s=1.0), "name_card")
    px, py = round(card.centre[0]), round(card.centre[1])
    assert re.search(rf"\\pos\({px},{py}\)\}}\{{\\fs\d+\\b1\}}ДЕВИН\\N\{{\\fs\d+\\b0\}}АКТЁР", text)
    x0, y0, x1, y1 = (round(v) for v in card.rect)  # the box drawing covers the English card
    assert f"m {x0} {y0} l {x1} {y0} {x1} {y1} {x0} {y1}" in text
    assert x0 <= CARD_BOX[0] * W and x1 >= CARD_BOX[2] * W and y0 <= CARD_BOX[1] * H and y1 >= CARD_BOX[3] * H
    assert text.count("\\p1") == 3  # one background drawing per item


def test_write_ass_nothing_to_show(tmp_path: Path) -> None:
    assert write_ass([], tmp_path / "o.ass", W, H) is None
    assert write_ass([item("caption", 1, 2, "", None)], tmp_path / "o.ass", W, H) is None
    assert not (tmp_path / "o.ass").exists()


def test_load_items_accepts_both_shapes(tmp_path: Path) -> None:
    one = item("caption", 1, 2, "Привет", None)
    (tmp_path / "a.json").write_text(json.dumps([one]), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"items": [one]}), encoding="utf-8")
    assert load_items(tmp_path / "a.json") == load_items(tmp_path / "b.json") == [one]


def test_render_overlays_shape_alpha_and_position() -> None:
    items = [item("caption", 1, 2, "Там лодка.", CAPTION_BOX), item("name_card", 1, 2, "ДЕВИН", CARD_BOX, "АКТЁР", n=2)]
    overlays = render_overlays(items, W, H)
    assert len(overlays) == 2
    caption, card = overlays
    cues = layout(items, W, H)
    for overlay, cue in zip(sorted(overlays, key=lambda o: o.x), sorted(cues, key=lambda c: c.rect[0])):
        assert overlay.x == round(cue.rect[0]) and overlay.y == round(cue.rect[1])
        assert overlay.rgba.dtype == np.uint8 and overlay.rgba.shape[2] == 4
    assert caption.rgba[0, 0, 3] == 255 and card.rgba[0, 0, 3] == 217  # opaque caption box, translucent card
    card_cue = cue_by_kind(cues, "name_card")
    assert card.x <= CARD_BOX[0] * W and card.x + card.rgba.shape[1] >= CARD_BOX[2] * W  # over the English card
    assert card.y <= CARD_BOX[1] * H and card.y + card.rgba.shape[0] >= CARD_BOX[3] * H
    assert card.x == round(card_cue.rect[0])
    assert (caption.rgba[..., :3] > 200).any()  # white text drawn
    assert (card.start, card.end) == (1.0, 2.0)


def test_blend_alpha() -> None:
    frame = np.full((10, 10, 3), 100, np.uint8)
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = 200, 128
    blend(frame, Overlay(0, 1, 8, 8, rgba))  # partly off the frame: clipped
    assert frame[9, 9, 0] == 150 and frame[7, 7, 0] == 100


def test_font_files_exist() -> None:
    files = font_files()
    assert files is not None and all(f.is_file() for f in files)


def _run_mux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> list[str]:
    seen: list[list[str]] = []
    monkeypatch.setattr(mix.subprocess, "run", lambda cmd, **kw: seen.append(list(cmd)))
    monkeypatch.setattr(Path, "replace", lambda self, target: None)
    mix.mux_tracks(tmp_path / "v.mkv", [(tmp_path / "a.m4a", "Русский")], tmp_path / "o.mkv", **kwargs)
    return seen[0]


def test_mux_tracks_adds_forced_ass_and_unsets_srt_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cmd = _run_mux(monkeypatch, tmp_path, srt=tmp_path / "s.srt", onscreen_ass=tmp_path / "o.ass",
                   fonts=[tmp_path / "f.ttf", tmp_path / "fb.ttf"])
    pairs = list(zip(cmd, cmd[1:]))
    inputs = [b for a, b in pairs if a == "-i"]
    assert inputs == [str(tmp_path / n) for n in ("v.mkv", "a.m4a", "s.srt", "o.ass")]
    assert ("-map", "2:s:0") in pairs and ("-map", "3:s:0") in pairs
    assert ("-c:s:0", "srt") in pairs and ("-c:s:1", "ass") in pairs
    assert ("-disposition:s:0", "0") in pairs and ("-disposition:s:1", "default+forced") in pairs
    assert "title=Надписи на экране" in cmd and cmd.count("-attach") == 2
    assert ("-metadata:s:t:1", "mimetype=application/x-truetype-font") in pairs


def test_mux_tracks_ass_only_and_unchanged_without_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cmd = _run_mux(monkeypatch, tmp_path, srt=None, onscreen_ass=tmp_path / "o.ass")
    assert ("-c:s:0", "ass") in zip(cmd, cmd[1:]) and ("-disposition:s:0", "default+forced") in zip(cmd, cmd[1:])
    plain = _run_mux(monkeypatch, tmp_path, srt=tmp_path / "s.srt")
    assert not any("disposition:s" in x for x in plain) and "-attach" not in plain
    assert ("-c:s:0", "srt") in zip(plain, plain[1:])


def test_mux_tracks_piece_keeps_ass_but_not_srt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cmd = _run_mux(monkeypatch, tmp_path, srt=tmp_path / "s.srt", start=360.0, duration=60.0,
                   onscreen_ass=tmp_path / "o.ass")
    inputs = [b for a, b in zip(cmd, cmd[1:]) if a == "-i"]
    assert inputs == [str(tmp_path / n) for n in ("v.mkv", "a.m4a", "o.ass")]
    assert cmd.index("-ss") < cmd.index("-i")  # only the video is cut: the ASS is already shifted by write_ass
    assert ("-c:s:0", "ass") in zip(cmd, cmd[1:])


def test_mux_tracks_video_source_input_follows_ass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cmd = _run_mux(monkeypatch, tmp_path, srt=tmp_path / "s.srt", onscreen_ass=tmp_path / "o.ass",
                   video_source=tmp_path / "small.mkv")
    assert ("-map", "4:v:0") in zip(cmd, cmd[1:])


def _make_video(path: Path, seconds: int = 3) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=black:size=320x240:rate=25:duration={seconds}", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(path)], check=True)


def _brightness_by_time(path: Path) -> dict[int, float]:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        return {int(round(float(f.pts * video.time_base) * 25)): float(f.to_ndarray(format="gray").mean())
                for f in container.decode(video)}


def test_burn_in_changes_only_frames_in_range_and_keeps_timestamps(tmp_path: Path) -> None:
    src = tmp_path / "src.mkv"
    _make_video(src)
    rgba = np.zeros((60, 100, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = 255, 255
    out = encode_video(src, tmp_path / "burn.mkv", 300, codec="libx264", overlays=[Overlay(1.0, 2.0, 20, 20, rgba)])
    plain = encode_video(src, tmp_path / "plain.mkv", 300, codec="libx264")
    brightness, reference = _brightness_by_time(out), _brightness_by_time(plain)
    assert sorted(brightness) == sorted(reference)
    for frame, value in brightness.items():
        if 25 <= frame < 50:
            assert value > reference[frame] + 10
        else:
            assert value == pytest.approx(reference[frame], abs=0.5)


def test_burn_in_piece_uses_piece_time(tmp_path: Path) -> None:
    src = tmp_path / "src.mkv"
    _make_video(src)
    rgba = np.full((60, 100, 4), 255, np.uint8)
    out = encode_video(src, tmp_path / "piece.mkv", 300, codec="libx264", start_s=1.0, seconds=1.0,
                       overlays=[Overlay(0.0, 0.4, 20, 20, rgba)])
    brightness = _brightness_by_time(out)
    assert brightness[0] > brightness[20] + 10 and brightness[9] > brightness[20] + 10
    assert min(brightness) == 0
