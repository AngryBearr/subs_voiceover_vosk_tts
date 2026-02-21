"""Gradio UI for subtitle voice-over pipeline (Vosk TTS / Edge TTS).

Run (from repo root, with vosk_env activated):
    uv run -m ui.gradio_app

MVP UX: two tabs:
- Configuration
- Subtitles / Run

The UI calls existing in-repo modules directly (no shelling out).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    subs_json: Path
    analyzed_json: Optional[Path]
    normalized_json: Path
    segments_dir: Path
    mix_wav: Path
    mix_mp3: Optional[Path]


def _now_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _list_model_dirs() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.as_posix() for p in MODELS_DIR.iterdir() if p.is_dir()])


def _load_vosk_speakers(model_dir: str) -> List[Tuple[int, str]]:
    """Return list of (speaker_id, speaker_name)."""
    try:
        cfg_path = Path(model_dir) / "config.json"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        speaker_id_map = data.get("speaker_id_map") or {}
        if isinstance(speaker_id_map, dict) and speaker_id_map:
            out: List[Tuple[int, str]] = []
            for name, sid in speaker_id_map.items():
                try:
                    out.append((int(sid), str(name)))
                except Exception:
                    continue
            out.sort(key=lambda x: x[0])
            return out

        # Fallback: use num_speakers if present
        n = int(data.get("num_speakers", 0) or 0)
        if n > 0:
            return [(i, f"speaker_{i}") for i in range(n)]
    except Exception:
        pass
    return [(0, "speaker_0")]


def _speaker_choices(model_dir: str) -> List[str]:
    speakers = _load_vosk_speakers(model_dir)
    return [f"{sid}: {name}" for sid, name in speakers]


def _parse_speaker_choice(choice: str) -> int:
    # "8: some_name" -> 8
    return int(str(choice).split(":", 1)[0].strip())


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _seconds_to_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # allow "mm:ss" or "ss"
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            mm = float(parts[0])
            ss = float(parts[1])
            return int(round((mm * 60.0 + ss) * 1000.0))
    return int(round(float(s) * 1000.0))


def _normalize_gender_mode_items(items: List[Dict[str, Any]], mode: str, gender_key: str = "gender") -> List[Dict[str, Any]]:
    if mode == "use_gender":
        return items
    forced = "male" if mode == "force_male" else "female"
    out: List[Dict[str, Any]] = []
    for it in items:
        new_it = dict(it)
        new_it[gender_key] = forced
        out.append(new_it)
    return out


def run_pipeline_from_ui(
    *,
    engine: str,
    srt_file_path: str,
    output_root: str,
    do_analysis: bool,
    avg_chars_per_sec: float,
    max_mismatch_ratio: float,
    do_runorm: bool,
    do_ruaccent: bool,
    norm_device: str,
    accent_device: str,
    batch_size: int,
    vosk_model_dir: str,
    vosk_device: str,
    gender_mode: str,
    vosk_male_voice_choice: str,
    vosk_female_voice_choice: str,
    use_analysis_speech_rate: bool,
    base_speech_rate: float,
    max_extra_pct: float,
    edge_male_voice: str,
    edge_female_voice: str,
    edge_rate: str,
    edge_volume: str,
    edge_pitch: str,
    reduce_silence_enabled: bool,
    audio_analysis_enabled: bool,
    resample_enabled: bool,
    resample_target_sr: int,
    mix_sample_rate: int,
    mix_channels: int,
    export_wav: bool,
    export_mp3: bool,
    video_duration_sec: str,
) -> Any:
    """Generator for Gradio: yields (log_text, files)."""

    log_lines: List[str] = []

    def log(msg: str) -> Tuple[str, Optional[List[str]]]:
        log_lines.append(msg)
        return "\n".join(log_lines), None

    yield log("[ui] Starting...")

    if not srt_file_path:
        raise ValueError("Please upload an .srt file")
    srt_path = Path(srt_file_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    run_dir = Path(output_root).expanduser().resolve() / f"run_{_now_slug()}"
    json_dir = run_dir / "json"
    segments_dir = run_dir / "segments"
    mix_dir = run_dir / "mix"
    json_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    yield log(f"[ui] Output: {run_dir}")

    # 1) SRT -> JSON items
    from utils.srt_to_json import parse_srt_file

    subs: List[Dict[str, Any]] = parse_srt_file(str(srt_path))
    subs_json = json_dir / "subs.json"
    _save_json(subs_json, subs)
    yield log(f"[ui] Parsed SRT: {len(subs)} entries")

    # 2) Analyze (optional)
    analyzed_items = subs
    analyzed_json: Optional[Path] = None
    if bool(do_analysis):
        from utils.text_analysis import analyze_subtitles_items

        analysis_result = analyze_subtitles_items(
            subs,
            avg_chars_per_sec=float(avg_chars_per_sec),
            max_mismatch_ratio=float(max_mismatch_ratio),
        )
        analyzed_items = analysis_result["items"]
        analyzed_json = json_dir / "subs_analyzed.json"
        _save_json(analyzed_json, analyzed_items)

        critical_items = analysis_result.get("critical_items") or []
        yield log(f"[ui] Analysis: critical={len(critical_items)}")

        if not critical_items:
            pass
    else:
        yield log("[ui] Analysis: skipped")

    # 3) Normalize (RUNorm / RUAccent)
    normalized_items = analyzed_items
    if do_runorm and do_ruaccent:
        from utils.text_normalization import normalize_subtitles_items

        normalized_items = normalize_subtitles_items(
            analyzed_items,
            norm_device=str(norm_device),
            accent_device=str(accent_device),
            batch_size=int(batch_size),
        )
        yield log("[ui] Normalization: RUNorm + RUAccent")
    elif do_runorm:
        from utils.text_normalizer import _load_runorm, normalize_subtitles_runorm_only

        normalizer = _load_runorm(device=str(norm_device))
        normalized_items = normalize_subtitles_runorm_only(
            analyzed_items,
            normalizer,
            batch_size=int(batch_size),
        )
        yield log("[ui] Normalization: RUNorm only")
    else:
        yield log("[ui] Normalization: skipped")

    # Apply gender override (if requested)
    gender_key = "gender"
    normalized_items = _normalize_gender_mode_items(normalized_items, gender_mode, gender_key=gender_key)

    normalized_json = json_dir / "subs_normalized.json"
    _save_json(normalized_json, normalized_items)

    # 4) Synthesize segments
    if engine == "vosk":
        from synthesize.synthesize_batch import synthesize_json_lines

        model_path = str(vosk_model_dir).strip()
        if not model_path:
            raise ValueError("Vosk model directory must be set")

        male_voice_id = _parse_speaker_choice(vosk_male_voice_choice)
        female_voice_id = _parse_speaker_choice(vosk_female_voice_choice)

        # gender_mode already applied to items. For safety, keep per-gender mapping.
        voice_male = int(male_voice_id)
        voice_female = int(female_voice_id)

        use_analysis_sr = bool(use_analysis_speech_rate) and bool(do_analysis)

        yield log(
            f"[ui] Vosk TTS: model={model_path} device={vosk_device} male={voice_male} female={voice_female}"
        )

        synthesize_json_lines(
            json_path=str(normalized_json),
            model_path=model_path,
            device=str(vosk_device),
            output_folder=str(segments_dir),
            file_prefix="",
            voice=int(voice_male),
            voice_male=int(voice_male),
            voice_female=int(voice_female),
            gender_key=gender_key,
            default_speech_rate=float(base_speech_rate),
            use_analysis_speech_rate=bool(use_analysis_sr),
            base_speech_rate=float(base_speech_rate),
            max_extra_pct=float(max_extra_pct),
            output_naming="index",
            text_key="text",
        )
        yield log("[ui] Vosk TTS: done")

    elif engine == "edge":
        from synthesize.edge_tts_batch import synthesize_json_to_segments_dir_by_gender

        male_voice = str(edge_male_voice).strip()
        female_voice = str(edge_female_voice).strip()
        if not male_voice or not female_voice:
            raise ValueError("Edge male/female voice names must be set")

        yield log(f"[ui] Edge TTS: male={male_voice} female={female_voice}")

        synthesize_json_to_segments_dir_by_gender(
            json_path=Path(normalized_json),
            out_dir=segments_dir,
            male_voice=male_voice,
            female_voice=female_voice,
            default_gender="male",
            gender_key=gender_key,
            rate=str(edge_rate),
            volume=str(edge_volume),
            pitch=str(edge_pitch),
            keep_mp3=False,
        )
        yield log("[ui] Edge TTS: done")

    else:
        raise ValueError(f"Unknown engine: {engine}")

    # 5) Post-processing (optional)
    if reduce_silence_enabled:
        from utils.reduce_silence import reduce_silence_in_dir

        reduce_silence_in_dir(
            segments_dir,
            in_place=True,
            pattern="*.wav",
        )
        yield log("[ui] Reduce silence: done")

    if audio_analysis_enabled:
        from utils.audio_analysis import check_and_speedup_segments

        check_and_speedup_segments(
            normalized_items,
            segments_dir,
            file_prefix="",
            ext=".wav",
            in_place=True,
            prefer_ffmpeg=True,
        )
        yield log("[ui] Audio analysis/speedup: done")

    if resample_enabled:
        from utils.audio_resample import resample_wav_dir

        resample_wav_dir(
            segments_dir,
            target_sr=int(resample_target_sr),
            pattern="*.wav",
            in_place=True,
        )
        yield log(f"[ui] Resample segments: {resample_target_sr} Hz")

    # 6) Mix and export
    from utils.audio_mixer import mix_segments

    total_duration_ms = _seconds_to_ms(video_duration_sec)

    output_basename = mix_dir / "voiceover"
    transcode_to = "mp3" if export_mp3 else None

    final_path = mix_segments(
        subtitles_json=subs_json,
        segments_folder=segments_dir,
        output_audio=output_basename,
        sample_rate=int(mix_sample_rate),
        channels=int(mix_channels),
        transcode_to=transcode_to,
        total_duration_ms=total_duration_ms,
        expected_index_ext=".wav",
    )

    mix_wav = output_basename.with_suffix(".wav")
    mix_mp3 = output_basename.with_suffix(".mp3") if export_mp3 else None

    files: List[str] = []
    if export_wav and mix_wav.exists():
        files.append(str(mix_wav))
    if export_mp3 and mix_mp3 is not None and mix_mp3.exists():
        files.append(str(mix_mp3))

    yield "\n".join(log_lines + [f"[ui] Done. Final: {final_path}"]), files


def build_demo() -> gr.Blocks:
    model_dirs = _list_model_dirs()
    default_model = model_dirs[0] if model_dirs else ""
    default_speakers = _speaker_choices(default_model) if default_model else ["0: speaker_0"]

    with gr.Blocks(title="Subs Voiceover (Vosk/Edge)") as demo:
        gr.Markdown("# Subtitle voice-over UI (MVP)")

        with gr.Tabs():
            with gr.Tab("Configuration"):
                engine = gr.Dropdown(
                    choices=["vosk", "edge"],
                    value="vosk",
                    label="Engine (Vosk or Edge)",
                )

                with gr.Row():
                    export_wav = gr.Checkbox(value=True, label="Export WAV")
                    export_mp3 = gr.Checkbox(value=True, label="Export MP3")

                gender_mode = gr.Dropdown(
                    choices=[
                        ("Use gender from subtitles", "use_gender"),
                        ("Force male", "force_male"),
                        ("Force female", "force_female"),
                    ],
                    value="use_gender",
                    label="Gender mode",
                )

                gr.Markdown("## Analysis")
                do_analysis = gr.Checkbox(value=True, label="Analyze subtitles")
                with gr.Row():
                    avg_chars_per_sec = gr.Number(value=13.0, label="Avg chars/sec")
                    max_mismatch_ratio = gr.Number(value=1.3, label="Max mismatch ratio")

                gr.Markdown("## Subtitle preprocessing")
                with gr.Row():
                    do_runorm = gr.Checkbox(value=True, label="RUNorm normalization")
                    do_ruaccent = gr.Checkbox(value=True, label="RUAccent accentuation")

                with gr.Row():
                    norm_device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="RUNorm device")
                    accent_device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="RUAccent device")
                    batch_size = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Batch size")

                gr.Markdown("## Vosk TTS")
                vosk_model_dir = gr.Dropdown(
                    choices=model_dirs,
                    value=default_model,
                    label="Vosk model directory",
                    interactive=True,
                )
                vosk_device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="Vosk device")

                with gr.Row():
                    vosk_male_voice_choice = gr.Dropdown(
                        choices=default_speakers,
                        value=default_speakers[0],
                        label="Male voice (speaker id)",
                    )
                    vosk_female_voice_choice = gr.Dropdown(
                        choices=default_speakers,
                        value=default_speakers[0],
                        label="Female voice (speaker id)",
                    )

                use_analysis_speech_rate = gr.Checkbox(value=True, label="Use analysis-driven speech rate")
                base_speech_rate = gr.Dropdown(
                    choices=[1.0, 1.1, 1.25, 1.5],
                    value=1.25,
                    label="Base speech rate",
                )
                max_extra_pct = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.2,
                    step=0.01,
                    label="Max extra pct over base",
                )

                gr.Markdown("## Edge TTS")
                with gr.Row():
                    edge_male_voice = gr.Textbox(value="ru-RU-DmitryNeural", label="Edge male voice")
                    edge_female_voice = gr.Textbox(value="ru-RU-SvetlanaNeural", label="Edge female voice")
                with gr.Row():
                    edge_rate = gr.Textbox(value="+0%", label="Edge rate")
                    edge_volume = gr.Textbox(value="+0%", label="Edge volume")
                    edge_pitch = gr.Textbox(value="+0Hz", label="Edge pitch")

                gr.Markdown("## Post-processing")
                with gr.Row():
                    reduce_silence_enabled = gr.Checkbox(value=True, label="Reduce silence")
                    audio_analysis_enabled = gr.Checkbox(value=True, label="Speedup if too long")
                    resample_enabled = gr.Checkbox(value=True, label="Resample segments")
                resample_target_sr = gr.Dropdown(choices=[44100, 48000], value=44100, label="Resample target SR")

                gr.Markdown("## Mix")
                with gr.Row():
                    mix_sample_rate = gr.Dropdown(choices=[44100, 48000], value=48000, label="Output sample rate")
                    mix_channels = gr.Dropdown(choices=[1, 2], value=1, label="Channels")

            with gr.Tab("Subtitles / Run"):
                srt_file = gr.File(label="Upload .srt", file_types=[".srt"], type="filepath")
                output_root = gr.Textbox(value=str((REPO_ROOT / "out" / "gradio").as_posix()), label="Output root")
                video_duration_sec = gr.Textbox(
                    value="",
                    label="Video duration (seconds or mm:ss) — optional",
                )

                run_btn = gr.Button("Run")
                log_box = gr.Textbox(label="Log", lines=18)
                outputs = gr.Files(label="Outputs")

                def _on_engine_change(e: str):
                    # Recommendations per engine
                    if e == "vosk":
                        return True, True
                    return True, False

                engine.change(_on_engine_change, inputs=[engine], outputs=[do_runorm, do_ruaccent])

                def _on_model_change(model_dir: str):
                    choices = _speaker_choices(model_dir)
                    default = choices[0] if choices else "0: speaker_0"
                    return gr.update(choices=choices, value=default), gr.update(choices=choices, value=default)

                vosk_model_dir.change(
                    _on_model_change,
                    inputs=[vosk_model_dir],
                    outputs=[vosk_male_voice_choice, vosk_female_voice_choice],
                )

                run_btn.click(
                    run_pipeline_from_ui,
                    inputs=[
                        engine,
                        srt_file,
                        output_root,
                        do_analysis,
                        avg_chars_per_sec,
                        max_mismatch_ratio,
                        do_runorm,
                        do_ruaccent,
                        norm_device,
                        accent_device,
                        batch_size,
                        vosk_model_dir,
                        vosk_device,
                        gender_mode,
                        vosk_male_voice_choice,
                        vosk_female_voice_choice,
                        use_analysis_speech_rate,
                        base_speech_rate,
                        max_extra_pct,
                        edge_male_voice,
                        edge_female_voice,
                        edge_rate,
                        edge_volume,
                        edge_pitch,
                        reduce_silence_enabled,
                        audio_analysis_enabled,
                        resample_enabled,
                        resample_target_sr,
                        mix_sample_rate,
                        mix_channels,
                        export_wav,
                        export_mp3,
                        video_duration_sec,
                    ],
                    outputs=[log_box, outputs],
                )

        demo.queue()

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
