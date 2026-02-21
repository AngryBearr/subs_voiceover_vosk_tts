from __future__ import annotations

import asyncio
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from edge_tts import Communicate  # type: ignore

from scripts.compute_sps_from_manifest import compute_sps_from_manifest
from utils.text_sanitize import count_symbols_for_sps


@dataclass(frozen=True)
class VoiceRunResult:
    voice: str
    out_dir: Path
    manifest_path: Path
    metrics_path: Path
    mean_sps: float
    min_sps: float
    max_sps: float
    mean_spm: float
    min_spm: float
    max_spm: float


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list")
    return data  # type: ignore[return-value]


def _join_text_field(item: Dict[str, Any]) -> str:
    text_val = item.get("text", []) or []
    if isinstance(text_val, list):
        return " ".join(str(t) for t in text_val if t is not None).strip()
    return str(text_val).strip()


def _ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is required to convert edge-tts mp3 to wav. Ensure ffmpeg is in PATH."
        ) from e


def _normalize_gender(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s.startswith(("f", "ж")):
        return "female"
    if s.startswith(("m", "м")):
        return "male"
    return None


def _select_voice_and_speaker_id(
    item: Dict[str, Any],
    *,
    male_voice: str,
    female_voice: str,
    default_gender: str = "male",
    gender_key: str = "gender",
) -> tuple[str, int]:
    gender = _normalize_gender(item.get(gender_key))
    if gender is None:
        gender = _normalize_gender(default_gender) or "male"
    if gender == "female":
        return female_voice, 1
    return male_voice, 0


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


async def synthesize_one_to_wav(
    *,
    text: str,
    voice: str,
    out_wav: Path,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    connect_timeout: int = 30,
    receive_timeout: int = 300,
    retries: int = 3,
    keep_mp3: bool = False,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if out_wav.exists() and out_wav.stat().st_size > 0:
        return

    # edge-tts currently hardcodes mp3 output format in its websocket config.
    # So we save to mp3 and convert to wav via ffmpeg.
    mp3_path = out_wav.with_suffix(".mp3")

    # Resume: if mp3 exists but wav doesn't, only convert.
    if mp3_path.exists() and (not out_wav.exists() or out_wav.stat().st_size == 0):
        _mp3_to_wav(mp3_path, out_wav)
        if not keep_mp3:
            try:
                mp3_path.unlink(missing_ok=True)
            except Exception:
                pass
        return

    last_err: Optional[BaseException] = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            communicate = Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                connect_timeout=int(connect_timeout),
                receive_timeout=int(receive_timeout),
            )
            await communicate.save(str(mp3_path))
            _mp3_to_wav(mp3_path, out_wav)
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt >= int(retries):
                break
            # Simple backoff; edge-tts is network-bound and can transiently fail.
            await asyncio.sleep(1.5 * attempt)

    if last_err is not None:
        raise RuntimeError(
            f"edge-tts failed after {retries} attempt(s) for voice={voice} -> {out_wav}"
        ) from last_err

    if not keep_mp3:
        try:
            mp3_path.unlink(missing_ok=True)
        except Exception:
            pass


def synthesize_json_to_voice_folder(
    *,
    json_path: Path,
    voice: str,
    out_dir: Path,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    keep_mp3: bool = False,
    connect_timeout: int = 30,
    receive_timeout: int = 300,
    retries: int = 3,
    limit_items: Optional[int] = None,
) -> VoiceRunResult:
    _ensure_ffmpeg_available()

    items = _load_json_list(json_path)
    if limit_items is not None:
        items = items[: int(limit_items)]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write WAVs and manifest
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "speaker_id", "filename", "text_length"])

        for item in items:
            idx = item.get("index")
            if idx is None:
                raise ValueError("Each JSON item must have 'index'")
            n = int(idx)
            text = _join_text_field(item)
            text_len = count_symbols_for_sps(text)

            wav_path = out_dir / f"{n}.wav"
            asyncio.run(
                synthesize_one_to_wav(
                    text=text,
                    voice=voice,
                    out_wav=wav_path,
                    rate=rate,
                    volume=volume,
                    pitch=pitch,
                    connect_timeout=int(connect_timeout),
                    receive_timeout=int(receive_timeout),
                    retries=int(retries),
                    keep_mp3=keep_mp3,
                )
            )

            w.writerow([n, 0, wav_path.name, text_len])

    metrics = compute_sps_from_manifest(manifest_path, audio_root=out_dir)

    mean_sps = float(metrics["mean_cps"])
    min_sps = float(metrics["min_cps"])
    max_sps = float(metrics["max_cps"])

    return VoiceRunResult(
        voice=voice,
        out_dir=out_dir,
        manifest_path=manifest_path,
        metrics_path=Path(metrics["metrics_path"]),
        mean_sps=mean_sps,
        min_sps=min_sps,
        max_sps=max_sps,
        mean_spm=mean_sps * 60.0,
        min_spm=min_sps * 60.0,
        max_spm=max_sps * 60.0,
    )


def synthesize_json_to_segments_dir_by_gender(
    *,
    json_path: Path,
    out_dir: Path,
    male_voice: str,
    female_voice: str,
    default_gender: str = "male",
    gender_key: str = "gender",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    keep_mp3: bool = False,
    connect_timeout: int = 30,
    receive_timeout: int = 300,
    retries: int = 3,
    limit_items: Optional[int] = None,
) -> VoiceRunResult:
    _ensure_ffmpeg_available()

    items = _load_json_list(json_path)
    if limit_items is not None:
        items = items[: int(limit_items)]

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "speaker_id", "filename", "text_length"])

        for item in items:
            idx = item.get("index")
            if idx is None:
                raise ValueError("Each JSON item must have 'index'")
            n = int(idx)
            text = _join_text_field(item)
            text_len = count_symbols_for_sps(text)

            voice, speaker_id = _select_voice_and_speaker_id(
                item,
                male_voice=str(male_voice),
                female_voice=str(female_voice),
                default_gender=str(default_gender),
                gender_key=str(gender_key),
            )

            wav_path = out_dir / f"{n}.wav"
            asyncio.run(
                synthesize_one_to_wav(
                    text=text,
                    voice=voice,
                    out_wav=wav_path,
                    rate=rate,
                    volume=volume,
                    pitch=pitch,
                    connect_timeout=int(connect_timeout),
                    receive_timeout=int(receive_timeout),
                    retries=int(retries),
                    keep_mp3=keep_mp3,
                )
            )

            w.writerow([n, speaker_id, wav_path.name, text_len])

    metrics = compute_sps_from_manifest(manifest_path, audio_root=out_dir)
    mean_sps = float(metrics["mean_cps"])
    min_sps = float(metrics["min_cps"])
    max_sps = float(metrics["max_cps"])

    voice_label = f"gendered(male={male_voice},female={female_voice})"
    return VoiceRunResult(
        voice=voice_label,
        out_dir=out_dir,
        manifest_path=manifest_path,
        metrics_path=Path(metrics["metrics_path"]),
        mean_sps=mean_sps,
        min_sps=min_sps,
        max_sps=max_sps,
        mean_spm=mean_sps * 60.0,
        min_spm=min_sps * 60.0,
        max_spm=max_sps * 60.0,
    )


def synthesize_two_voices(
    *,
    json_path: Path,
    voices: Iterable[str],
    output_root: Path,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    keep_mp3: bool = False,
    connect_timeout: int = 30,
    receive_timeout: int = 300,
    retries: int = 3,
) -> List[VoiceRunResult]:
    results: List[VoiceRunResult] = []
    for v in voices:
        v = str(v).strip()
        if not v:
            continue
        out_dir = output_root / v
        res = synthesize_json_to_voice_folder(
            json_path=json_path,
            voice=v,
            out_dir=out_dir,
            rate=rate,
            volume=volume,
            pitch=pitch,
            keep_mp3=keep_mp3,
            connect_timeout=int(connect_timeout),
            receive_timeout=int(receive_timeout),
            retries=int(retries),
        )
        results.append(res)
    return results
