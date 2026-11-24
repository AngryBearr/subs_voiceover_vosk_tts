"""Sound preview helpers and the sound/video-size tab."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from ui import seasons, settings
from ui.app_paths import ENGINES, REPO_ROOT, _episode, _root
from ui.widgets import wait_state

log = logging.getLogger("ui.dub_app")
MIX_SECONDS = 45.0
MIX_SETTINGS = [next(s for s in settings.SETTINGS if s.path == path) for path in (
    ("mix", "voice_gain_db"), ("mix", "rest_duck_db"), ("mix", "duck_db"), ("mix", "loudness_lufs"),
    ("mix", "video", "compress"), ("mix", "video", "target_gb"), ("mix", "video", "codec"))]


def mix_form_defaults(config: dict[str, Any]) -> list[Any]:
    """Values of the sound form in the order of ``MIX_SETTINGS``."""
    return [settings.to_form(s, settings.get(config, s.path)) for s in MIX_SETTINGS]


def episode_media(out_root: str, episode: str) -> Path:
    """The original video of a work folder (``source.json``)."""
    return Path(json.loads((_episode(out_root, episode) / "source.json").read_text(encoding="utf-8"))["media"])


def voiced_engines(out_root: str, episode: str) -> list[str]:
    """Labels of the engines that have a voice-over track in the episode."""
    folder = _episode(out_root, episode)
    return [label for label, value in ENGINES.items() if (folder / f"voiceover.{value}" / "export" / "track.wav").exists()]


def mix_episodes(out_root: str) -> Any:
    found = [p.parents[2].name for p in _root(out_root).glob("*/voiceover.*/export/track.wav")]
    return gr.update(choices=sorted(set(found)))


def mix_episodes_listed(out_root: str) -> tuple[Any, str]:
    """The list of the "Обновить список" button: (the dropdown, what was found)."""
    update = mix_episodes(out_root) if out_root else gr.update(choices=[])
    count = len(update["choices"])
    return update, (f"Серий с готовой озвучкой: {count}." if count else
                    "Серий с готовой озвучкой в папке результатов сезона пока нет (или сезон не выбран).")


def find_spot(out_root: str, episode: str | None, engine: str) -> tuple[Any, str]:
    """(minute, message) of the place where the voice-over has the hardest time against the music."""
    if not episode:
        return gr.update(), "Выберите серию."
    from stt import mix_preview

    try:
        spot = mix_preview.loudest_spot(_episode(out_root, episode), episode_media(out_root, episode),
                                        ENGINES[engine], MIX_SECONDS)
    except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
        log.exception("loudest spot")
        return gr.update(), f"❌ Не получилось найти место: {error}"
    return round(spot / 60, 2), f"✅ Самое громкое место: с {int(spot // 60)}:{int(spot % 60):02d}. Минута подставлена выше."


def listen_mix(out_root: str, episode: str | None, engine: str, minute: float, gain: float, rest: float,
               duck_db: float, lufs: float, *, season_base: Callable[[str], dict[str, Any]]) -> tuple[Any, Any, str]:
    """(new mix, the mix now in the episode, message) of a 45-second piece from ``minute``."""
    if not episode:
        return None, None, "Выберите серию."
    from stt import mix_preview

    folder, label = _episode(out_root, episode), engine
    start = max(0.0, float(minute or 0.0) * 60)
    out_dir = REPO_ROOT / "output" / "mix_preview"
    try:
        new = mix_preview.render_preview(folder, episode_media(out_root, episode), ENGINES[label], start, MIX_SECONDS,
                                        float(gain), float(duck_db), float(rest), float(lufs), out_dir / "new.wav",
                                        duck=season_base(out_root).get("mix", {}).get("duck"))
    except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
        log.exception("mix preview")
        return None, None, f"❌ Не получилось собрать отрывок: {error}"
    track = folder / f"{episode}.ru.{ENGINES[label]}.m4a"
    now = None
    if track.exists():
        now = str(mix_preview.existing_piece(track, start, MIX_SECONDS, out_dir / "now.wav"))
    return str(new), now, ("✅ Слева: с новыми настройками; справа: как сейчас в серии." if now else
                           "✅ Готово. Серия ещё не собрана: сравнивать не с чем.")


def videos_listed(out_root: str) -> tuple[Any, str]:
    """The list of the "Обновить список" button of the compression: (the dropdown, what was found)."""
    update = finished_videos(out_root)
    count = len(update["choices"])
    return update, (f"Готовых видео: {count}." if count else
                    "Готовых видео (`*.ru.mkv`) в папке результатов пока нет: впишите путь к файлу вручную.")


def finished_videos(out_root: str) -> Any:
    """Finished videos (``*.ru.mkv``) at the top of the results folder (and in the work folders of older runs)."""
    found: list[Path] = []
    if out_root:
        found += _root(out_root).glob("*.ru.mkv")
        found += _root(out_root).glob("*/*.ru.mkv")
    return gr.update(choices=sorted(str(p) for p in dict.fromkeys(found)))


@dataclass
class Tab:
    note: Any
    inputs: list[Any]
    save: Any
    saved: Any
    episode: Any
    video: Any


def build(out_root: Any, mix_start: list[Any], *, find_spot_done: Callable[..., Any],
          listen_mix_done: Callable[..., Any], remix: Callable[..., str],
          start_compress: Callable[..., str]) -> Tab:
    """Build the sound tab and expose its settings and season-change components."""
    with gr.Tab("Звук и размер видео"):
        note = gr.Markdown(seasons.NO_SEASON_NOTE)
        gr.Markdown("### Громкость голосов\nСоберите короткий отрывок (45 секунд) с вашими настройками и "
                    "сравните с тем, что сейчас в серии. Остальную серию программа не трогает. Серии берутся из папки результатов выбранного сезона.")
        with gr.Row():
            mix_episode = gr.Dropdown(label="Серия (с готовой озвучкой)", choices=[], scale=3)
            mix_refresh = gr.Button("Обновить список", scale=1)
            mix_engine = gr.Radio(label="Озвучка", choices=list(ENGINES), value=list(ENGINES)[-1], scale=3)
        with gr.Row():
            mix_minute = gr.Number(label="С минуты", value=10)
            mix_spot = gr.Button("Найти самое громкое место")
        with gr.Row():
            mix_gain = gr.Slider(-3, 12, step=1, value=mix_start[0], label="Голоса громче, дБ")
            mix_rest = gr.Slider(0, 15, step=1, value=mix_start[1], label="Музыка и шум под голосом тише, дБ")
        with gr.Row():
            mix_duck = gr.Slider(6, 24, step=1, value=mix_start[2], label="Голоса оригинала под озвучкой тише, дБ")
            mix_lufs = gr.Slider(-23, -12, step=1, value=mix_start[3], label="Общая громкость, LUFS")
        mix_listen = gr.Button("Послушать", variant="primary")
        with gr.Row():
            mix_new = gr.Audio(label="С этими настройками", type="filepath")
            mix_now = gr.Audio(label="Как сейчас в серии", type="filepath")
        mix_message = gr.Markdown()
        gr.Markdown("### Сжатие видео\nРазрешение остаётся исходным; видео кодирует видеокарта.")
        with gr.Row():
            vid_compress = gr.Checkbox(label="Сжимать видео при сборке серии", value=mix_start[4])
            vid_gb = gr.Slider(0.8, 6, step=0.1, value=mix_start[5], label="Размер серии, ГБ")
            vid_codec = gr.Radio(label="Кодек", choices=list(settings.VIDEO_CODECS), value=mix_start[6])
        with gr.Row():
            mix_save = gr.Button("Сохранить настройки")
            mix_redo = gr.Button("Пересвести эту серию")
        mix_saved = gr.Markdown()
        gr.Markdown("«Пересвести» собирает серию заново с сохранёнными настройками; озвучка не повторяется. "
                    "Ход виден во вкладке «Обработка».")
        gr.Markdown("### Сжать готовое видео\nИсходный файл не меняется: рядом появится файл `.small.mkv` "
                    "(или `.sample.mkv` для пробного куска: 60 секунд с 10-й минуты).")
        with gr.Row():
            vid_path = gr.Dropdown(label="Готовое видео (можно вписать путь)", choices=[], allow_custom_value=True,
                                   scale=4)
            vid_refresh = gr.Button("Обновить список", scale=1)
        with gr.Row():
            vid_go = gr.Button("Сжать", variant="primary")
            vid_sample = gr.Button("Пробный кусок 60 с")
        vid_message = gr.Markdown()
        mix_inputs = [mix_gain, mix_rest, mix_duck, mix_lufs, vid_compress, vid_gb, vid_codec]
        mix_refresh.click(mix_episodes_listed, out_root, [mix_episode, mix_message])
        mix_spot.click(lambda: wait_state("Ищу…", "Ищу самое громкое место серии — до 30 секунд…"), None,
                       [mix_spot, mix_message], queue=False).then(
            find_spot_done, [out_root, mix_episode, mix_engine], [mix_spot, mix_minute, mix_message])
        mix_listen.click(lambda: wait_state("Собираю…", "Собираю отрывок в 45 секунд — около 20 секунд…"), None,
                         [mix_listen, mix_message], queue=False).then(
            listen_mix_done, [out_root, mix_episode, mix_engine, mix_minute, mix_gain, mix_rest, mix_duck, mix_lufs],
            [mix_listen, mix_new, mix_now, mix_message])
        mix_redo.click(remix, [out_root, mix_episode], mix_saved)
        vid_refresh.click(videos_listed, out_root, [vid_path, vid_message])
        vid_go.click(lambda p, g, c, r: start_compress(p, False, g, c, r), [vid_path, vid_gb, vid_codec, out_root],
                     vid_message)
        vid_sample.click(lambda p, g, c, r: start_compress(p, True, g, c, r), [vid_path, vid_gb, vid_codec, out_root],
                         vid_message)
    return Tab(note, mix_inputs, mix_save, mix_saved, mix_episode, vid_path)
