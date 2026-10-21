"""English STT: local PixIT separation/ASR or explicit OpenRouter audio-upload mode."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from numbers import Integral
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.json_to_srt import json_to_srt

LOG = logging.getLogger(__name__)
UNKNOWN = "SPEAKER_UNKNOWN"
PIXIT_REPO = "pyannote/speech-separation-ami-1.0"
PIXIT_REV = "9486b106945ae0cc0784041a08bfcdba5edadfb9"
SEPARATION_REPO = "pyannote/separation-ami-1.0"
SEPARATION_REV = "4d38e95cfd067c894b8b60b00761831fb01e4a8c"
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
PARAKEET_REV = "0bbb45a3365852604aef28b538a8f066f4ccaa85"
GENDER_URL = (
    "https://huggingface.co/Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2/resolve/"
    "8aedeb1a399945e52e6f3f61714abe07c1c48742/onnx/model_quantized.onnx"
)
GENDER_SHA = "fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252"
SR = 16000


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    speaker: str = UNKNOWN


def load_overrides(path: Path | None) -> dict[str, str]:
    """Load strict speaker-to-acoustic-label overrides, never an identity map."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--speaker-genders must contain a JSON object")
    for speaker, label in data.items():
        if not re.fullmatch(r"SPEAKER_\d{2,}", speaker):
            raise ValueError(f"Invalid override speaker ID: {speaker!r}; use SPEAKER_00, etc.")
        if label not in ("male", "female", "unknown"):
            raise ValueError(f"Invalid label for {speaker}: {label!r}; use male/female/unknown")
    return data


def validate_options(
    input_path: Path, output_prefix: Path, *, start: float, duration: float | None,
    threads: int, num_speakers: int | None, gender_threshold: float,
    speaker_genders: Path | None = None,
) -> dict[str, str]:
    """Validate cheap parameters and source/output collisions before downloads."""
    if not input_path.is_file():
        raise ValueError(f"Input media not found: {input_path}")
    if not math.isfinite(start) or start < 0:
        raise ValueError("--start must be finite and nonnegative")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("--duration must be finite and positive")
    if not 1 <= threads <= 32:
        raise ValueError("--threads must be between 1 and 32")
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("--num-speakers must be positive")
    if not math.isfinite(gender_threshold) or not 0.5 < gender_threshold <= 1:
        raise ValueError("--gender-threshold must be greater than .5 and at most 1")
    sources = {input_path.resolve()}
    if speaker_genders is not None:
        sources.add(speaker_genders.resolve())
    for suffix in (".json", ".srt", ".speakers.json"):
        output = Path(str(output_prefix) + suffix)
        if output.resolve() in sources or any(
            output.exists() and source.exists() and output.samefile(source) for source in sources
        ):
            raise ValueError(f"Output would overwrite input or override file: {output}")
        if output.exists() and not output.is_file():
            raise ValueError(f"Output is not a regular file: {output}")
    return load_overrides(speaker_genders)


def download(url: str, path: Path, *, sha256: str | None = None) -> Path:
    """Download atomically; an interrupted transfer never becomes a cache hit."""
    if path.is_file():
        if sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return path
        raise ValueError(f"Cached model checksum mismatch: {path}; remove it and retry")
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s -> %s", url, path)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "local-stt/1.0"})
        with os.fdopen(fd, "wb") as target, urllib.request.urlopen(request, timeout=60) as source:
            size = source.headers.get("Content-Length")
            received = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
                received += len(block)
            if size is not None and received != int(size):
                raise ValueError(f"Incomplete download: {received}/{size} bytes")
            if received == 0:
                raise ValueError("Empty model download")
        if sha256 is not None and digest.hexdigest() != sha256:
            raise ValueError(f"Model checksum mismatch for {url}")
        os.replace(temporary, path)
        LOG.info("Downloaded %.1f MiB", received / 1024**2)
    except Exception as exc:
        raise RuntimeError(f"Cannot download {url}: {exc}. Check network/cache and retry.") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def configure_hub(cache: Path) -> None:
    """Keep model caches local and disable telemetry; never log credentials."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Standard HTTPS avoids the optional Xet transport and keeps cache paths explicit.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    os.environ["PYANNOTE_CACHE"] = str((cache / "pyannote").resolve())
    # Imports may already have happened while reading local credentials.
    import sys
    if "huggingface_hub.constants" in sys.modules:
        sys.modules["huggingface_hub.constants"].HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
        sys.modules["huggingface_hub.constants"].HF_HUB_DISABLE_XET = True


def require_hf_token() -> str:
    """Read only local HF auth. Never request or display a token in chat/logs."""
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise ValueError(
            "PixIT needs local Hugging Face authentication. Accept the conditions at "
            f"https://huggingface.co/{PIXIT_REPO} and https://huggingface.co/{SEPARATION_REPO}, "
            "then run huggingface-cli login locally (or set HF_TOKEN). Never share your token in chat."
        )
    return token


def decode_media(input_path: Path, start: float, duration: float | None) -> Any:
    """Decode exactly one selected clip into raw mono float32 at 16 kHz."""
    import numpy as np

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ValueError("ffmpeg and ffprobe must be installed and available in PATH")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(input_path.resolve())], capture_output=True, text=True, check=False,
    )
    if probe.returncode:
        raise ValueError(f"Cannot inspect input media: {probe.stderr.strip()}")
    info = json.loads(probe.stdout)
    if not any(stream.get("codec_type") == "audio" for stream in info.get("streams", [])):
        raise ValueError("Input media has no audio stream")
    length = info.get("format", {}).get("duration")
    if length is not None and start >= float(length):
        raise ValueError(f"--start {start} is outside input duration {length}s")
    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-ss", str(start),
            "-i", str(input_path.resolve())]
    if duration is not None:
        args += ["-t", str(duration)]
    args += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    process = subprocess.run(args, capture_output=True, check=False)
    if process.returncode:
        raise ValueError(f"Cannot decode input media: {process.stderr.decode(errors='replace').strip()}")
    audio = np.frombuffer(process.stdout, dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ValueError("Decoded media contains nonfinite samples")
    return audio


def map_sources(annotation: Any, sources: Any, clip_duration: float) -> tuple[list[Turn], dict[str, dict[str, Any]]]:
    """Bind raw numeric cluster IDs to their source columns, before upstream renaming."""
    labels = list(annotation.labels())
    if not labels:
        return [], {}
    if sources is None or len(sources.data.shape) != 2:
        raise ValueError(f"PixIT needs review: {len(labels)} raw labels but no valid source matrix; cannot bind speakers")
    channels = sources.data.shape[1]
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label < channels:
            raise ValueError(
                f"PixIT needs review: raw cluster {label!r} has no source channel "
                f"({len(labels)} active labels, {channels} source channels). "
                "No voice will be fabricated. Retry with --num-speakers if the global count is known; "
                "this is not guaranteed to resolve upstream overcounting."
            )
    window = sources.sliding_window
    if (not math.isfinite(window.start) or abs(window.start) > 2 / SR
            or not math.isclose(window.step, 1 / SR, rel_tol=0, abs_tol=1e-9)
            or not 0 < window.duration <= 3 / SR):
        raise ValueError("Unsupported separated-source sample timing; expected an aligned 16kHz waveform")
    if sources.data.shape[0] / SR + window.start < clip_duration - 2 / SR:
        raise ValueError("Separated sources are shorter than the selected clip")
    raw = sorted(annotation.itertracks(yield_label=True), key=lambda item: (item[0].start, item[2]))
    mapping: dict[str, dict[str, Any]] = {}
    names: dict[Any, str] = {}
    turns = []
    for segment, _, label in raw:
        if not all(math.isfinite(t) for t in (segment.start, segment.end)) or segment.end <= segment.start:
            raise ValueError("Invalid diarization interval")
        start, end = max(0, segment.start), min(clip_duration, segment.end)
        if end <= start:
            continue
        if label not in names:
            speaker = f"SPEAKER_{len(names):02d}"
            names[label] = speaker
            mapping[speaker] = {"pipeline_label": f"SPEAKER_{labels.index(label):02d}",
                                "raw_cluster": int(label), "source_channel": int(label),
                                "source_start": window.start, "separation_status": "estimated"}
        turns.append(Turn(start, end, names[label]))
    if len(mapping) != len(labels):
        raise ValueError("Some source labels have no active intervals in the selected clip")
    return turns, mapping


def separate(audio: Any, cache: Path, threads: int, num_speakers: int | None) -> tuple[list[Turn], Any, dict[str, dict[str, Any]]]:
    """One integrated PixIT invocation retains simultaneous global speaker tracks."""
    # Auth is never saved with output assets.
    configure_hub(cache)
    token = require_hf_token()
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url, try_to_load_from_cache
    from huggingface_hub.utils import HfHubHTTPError
    import numpy as np
    import torch
    import yaml
    from pyannote.audio import Pipeline

    torch.set_num_threads(threads)
    LOG.info("PixIT AMI: gated MIT checkpoint ~1.19 GiB, WavLM-large ~1.2 GB plus embeddings; CPU")
    try:
        # Fetch both small gated files first; do not start multi-GB downloads on missing approval.
        config_path = hf_hub_download(PIXIT_REPO, "config.yaml", revision=PIXIT_REV,
                                      token=token, cache_dir=str(cache / "hub"))
        cached = try_to_load_from_cache(SEPARATION_REPO, "pytorch_model.bin", revision=SEPARATION_REV,
                                        cache_dir=str(cache / "pyannote"))
        if not isinstance(cached, str):
            get_hf_file_metadata(hf_hub_url(SEPARATION_REPO, "pytorch_model.bin", revision=SEPARATION_REV), token=token)
    except HfHubHTTPError:
        raise ValueError(
            f"PixIT access failed. Accept BOTH gates: https://huggingface.co/{PIXIT_REPO} and "
            f"https://huggingface.co/{SEPARATION_REPO}; check local HF login/read permissions."
        ) from None
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("pipeline", {}).get("name") != "pyannote.audio.pipelines.SpeechSeparation":
        raise ValueError("Pinned PixIT config has an unexpected pipeline class")
    if config.get("preprocessors"):
        raise ValueError("Remote pipeline preprocessors are not allowed")
    params = config["pipeline"]["params"]
    params.pop("use_auth_token", None)
    params["segmentation"] = {"checkpoint": f"{SEPARATION_REPO}@{SEPARATION_REV}",
                              "cache_dir": str(cache / "pyannote"), "strict": False}
    config["device"] = "cpu"
    cache.mkdir(parents=True, exist_ok=True)
    local_config = cache / "pixit-pinned.yaml"
    local_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    pipeline = Pipeline.from_pretrained(local_config, use_auth_token=token, cache_dir=str(cache / "pyannote"))
    if pipeline is None:
        raise ValueError("PixIT could not load; check BOTH AMI gates and local HF authentication")
    model = pipeline._segmentation.model
    if model.n_sources != 3 or model.hparams.sample_rate != SR:
        raise ValueError("Pinned PixIT must expose three local sources at 16kHz")
    pipeline.to(torch.device("cpu"))
    discrete = None

    def progress(stage: str, artefact: Any, **kwargs: Any) -> None:
        nonlocal discrete
        if stage == "discrete_diarization":
            # Hold the original reference: leakage removal mutates it before the
            # pipeline's final to_annotation, so the same activity is reconstructed.
            discrete = artefact
        if kwargs.get("completed") in (None, 0, kwargs.get("total")):
            LOG.info("PixIT stage: %s (%s/%s)", stage, kwargs.get("completed"), kwargs.get("total"))

    with torch.inference_mode():
        annotation, sources, _ = pipeline(
            {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": SR},
            num_speakers=num_speakers, return_embeddings=True, hook=progress,
        )
    if discrete is None:
        if annotation.labels() or sources is not None:
            raise ValueError("PixIT needs review: missing raw discrete diarization; cannot bind source channels")
        return [], None, {}
    raw_annotation = pipeline.to_annotation(
        discrete, min_duration_on=0.0, min_duration_off=pipeline.segmentation.min_duration_off,
    )
    turns, mapping = map_sources(raw_annotation, sources, len(audio) / SR)
    if sources is not None and not np.isfinite(sources.data).all():
        raise ValueError("PixIT returned nonfinite separated audio")
    from utils.stt_refiner import refine_triples

    refine_triples(audio, turns, sources, mapping, pipeline._embedding, cache)
    LOG.info("PixIT: %d turns, %d global sources; preserve overlapping intervals", len(turns), len(mapping))
    return turns, sources, mapping


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def speech_chunks(turns: list[Turn], clip_end: float, cap: float = 28.0) -> list[tuple[float, float]]:
    """Union padded speech intervals then cut disjoint bounded ASR chunks."""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("ASR chunk cap must be finite and positive")
    intervals = merge_intervals([(max(0.0, t.start - .2), min(clip_end, t.end + .2)) for t in turns])
    chunks = []
    for start, end in intervals:
        while end - start > cap:
            chunks.append((start, start + cap))
            start += cap
        if end > start:
            chunks.append((start, end))
    return chunks


def clip_time(value: float, chunk_duration: float) -> float:
    """Clamp at most one 80ms ASR grid frame past a clip; reject malformed times."""
    if (not math.isfinite(chunk_duration) or chunk_duration < 0 or not math.isfinite(value)
            or value < 0 or value > chunk_duration + .08 + 1e-9):
        raise ValueError("Invalid ASR timing: outside clip by more than one 80ms frame")
    return min(value, chunk_duration)


def whisper_words(raw_words: list[Any], chunk_duration: float, offset: float,
                  speaker: str) -> list[Word]:
    """Validate Whisper starts/ends together; retain bounded boundary emissions."""
    words = []
    previous_time = -1.0
    for word in raw_words:
        start, end = clip_time(word.start, chunk_duration), clip_time(word.end, chunk_duration)
        if word.start < previous_time or word.end < word.start:
            raise ValueError("Nonmonotonic Whisper word timing")
        previous_time = word.start
        if word.word.strip():
            words.append(Word(offset + start, offset + end, word.word.strip(), speaker))
    return words


def token_words(tokens: list[str], timestamps: list[float], chunk_duration: float,
                offset: float = 0, speaker: str = UNKNOWN) -> list[Word]:
    """Join Parakeet subwords; approximate ends from the next word or chunk boundary."""
    if len(tokens) != len(timestamps):
        raise ValueError("Parakeet tokens/timestamps lengths differ")
    groups: list[tuple[float, float, str]] = []
    previous_time = -1.0
    for token, timestamp in zip(tokens, timestamps):
        clipped = clip_time(timestamp, chunk_duration)
        if timestamp < previous_time:
            raise ValueError("Nonmonotonic Parakeet token timing")
        previous_time = timestamp
        timestamp = clipped
        boundary = token.startswith("▁") or (bool(token) and token[0].isspace())
        piece = token.replace("▁", " ").strip()
        if not piece:
            continue
        punctuation = not any(character.isalnum() for character in piece)
        opening = boundary and piece in ("(", "[", "{", '"', "“", "‘")
        pending_opening = bool(groups) and all(c in "([{\"“‘" for c in groups[-1][2])
        if groups and (not boundary or (punctuation and not opening) or pending_opening):
            start, _, text = groups[-1]
            groups[-1] = (start, timestamp, text + piece)
        else:
            groups.append((timestamp, timestamp, piece))
    words = []
    for i, (start, last, text) in enumerate(groups):
        next_start = groups[i + 1][0] if i + 1 < len(groups) else chunk_duration
        end = min(chunk_duration, next_start, last + .8)
        words.append(Word(offset + start, offset + max(start, end), text, speaker))
    return words


def transcribe(
    tracks: dict[str, Any], turns: list[Turn], cache: Path, threads: int,
    preset: str, whisper_model: str, refinement_windows: dict[str, list[dict[str, Any]]] | None = None,
) -> list[Word]:
    words: list[Word] = []
    configure_hub(cache)
    if preset == "fast":
        from huggingface_hub import snapshot_download
        import onnx_asr
        import onnxruntime as ort

        LOG.info("Fast ASR: Parakeet TDT v2 int8, CC-BY-4.0 (~630 MiB), pinned ONNX weights")
        directory = snapshot_download(
            PARAKEET_REPO, revision=PARAKEET_REV, token=False, cache_dir=str(cache / "hub"),
            allow_patterns=["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "config.json", "vocab.txt"],
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        recognizer = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", directory, quantization="int8", sess_options=options,
            providers=["CPUExecutionProvider"],
        ).with_timestamps()
    else:
        # Disable implicit credentials/telemetry before importing the Hugging Face client.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        from faster_whisper import WhisperModel

        LOG.info("Accurate preset: Whisper %s, CPU int8 (large-v3-turbo ~1.62 GB, MIT)", whisper_model)
        recognizer = WhisperModel(
            whisper_model, device="cpu", compute_type="int8", cpu_threads=threads,
            download_root=str(cache / "whisper"),
        )
    for speaker, audio in tracks.items():
        chunks = asr_chunks([turn for turn in turns if turn.speaker == speaker], len(audio) / SR,
                            (refinement_windows or {}).get(speaker, []))
        for number, (start, end) in enumerate(chunks, 1):
            LOG.info("ASR %s chunk %d/%d: %.2f–%.2fs", speaker, number, len(chunks), start, end)
            first, last = round(start * SR), round(end * SR)
            offset = first / SR
            samples = audio[first:last]
            if not len(samples) or abs(samples).max() <= 1e-5:
                continue
            if preset == "fast":
                result = recognizer.recognize(samples, sample_rate=SR)
                if result.text.strip() and not result.tokens:
                    raise ValueError("Parakeet returned text without timed tokens")
                chunk_words = token_words(list(result.tokens), list(result.timestamps), len(samples) / SR, offset, speaker)
                if result.text.strip() and not chunk_words:
                    raise ValueError("Parakeet returned speech without usable timed words")
                words.extend(chunk_words)
            else:
                segments, _ = recognizer.transcribe(
                    samples, language="en", task="transcribe", word_timestamps=True, beam_size=5, vad_filter=True,
                )
                raw_words = []
                for segment in segments:
                    if segment.text.strip() and not segment.words:
                        raise ValueError("Whisper returned speech without word timestamps")
                    raw_words.extend(segment.words or [])
                words.extend(whisper_words(raw_words, len(samples) / SR, offset, speaker))
    return sorted(words, key=lambda word: word.start)


def asr_chunks(turns: list[Turn], duration: float, refined: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Union real source activity/context, then disjoint repaired-context ASR routes."""
    speaker = turns[0].speaker if turns else UNKNOWN
    intervals = turns + [Turn(w["start"], w["end"], speaker) for w in refined]
    chunks = speech_chunks(intervals, duration)
    if refined:
        routes = merge_intervals([(max(0., w["start"] - .2), min(duration, w["end"] + .2)) for w in refined])
        cuts = {edge for route in routes for edge in route}
        split = []
        for start, end in chunks:
            points = sorted({start, end} | {p for p in cuts if start + .32 < p < end - .32})
            split.extend(zip(points, points[1:]))
        chunks = split
    return chunks


def clean_samples(turns: list[Turn], speaker: str, cap: float = 15.0) -> list[tuple[float, float]]:
    """Remove all competing speech, trim boundaries, and select up to 15s evidence."""
    own = merge_intervals([(t.start, t.end) for t in turns if t.speaker == speaker])
    other = merge_intervals([(t.start, t.end) for t in turns if t.speaker != speaker])
    clean = []
    remaining = cap
    for start, end in own:
        pieces = [(start, end)]
        for left, right in other:
            next_pieces = []
            for a, b in pieces:
                if right <= a or left >= b:
                    next_pieces.append((a, b))
                else:
                    next_pieces.extend((x, y) for x, y in ((a, min(b, left)), (max(a, right), b)) if y > x)
            pieces = next_pieces
        for a, b in pieces:
            a, b = a + .1, b - .1
            while b - a >= 1.0 and remaining >= 1.0:
                length = min(b - a, 5.0, remaining)
                clean.append((a, a + length))
                a += length
                remaining -= length
    return clean


def aggregate_gender(
    scores: list[tuple[float, float]], durations: list[float], threshold: float,
) -> tuple[str, float | None]:
    """Weighted probabilities with abstention on short, weak or conflicting evidence."""
    if len(scores) != len(durations):
        raise ValueError("Gender scores/evidence lengths differ")
    if not scores:
        return "unknown", None
    for score, duration in zip(scores, durations):
        if duration <= 0 or not math.isfinite(duration) or len(score) != 2 or any(
            not math.isfinite(s) or not 0 <= s <= 1 for s in score
        ) or abs(sum(score) - 1) > 1e-5:
            raise ValueError("Invalid gender probabilities or evidence duration")
    total = sum(durations)
    if total < 3.0:
        return "unknown", None
    mean = [sum(score[i] * length for score, length in zip(scores, durations)) / total for i in range(2)]
    winner = 0 if mean[0] >= mean[1] else 1
    # Any opposite or low-confidence sample invalidates a confident binary estimate.
    consistent = all(score[winner] >= threshold for score in scores)
    confidence = mean[winner]
    return (("male", "female")[winner] if consistent and confidence >= threshold else "unknown"), confidence


def estimate_genders(
    audio: Any, turns: list[Turn], cache: Path, threads: int, threshold: float,
    overrides: dict[str, str],
) -> dict[str, dict[str, Any]]:
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    unexpected = set(overrides) - set(speakers)
    if unexpected:
        raise ValueError(f"Override IDs absent from this clip: {sorted(unexpected)}; available: {speakers}")
    session = None
    metadata: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        samples = clean_samples(turns, speaker)
        durations = [b - a for a, b in samples]
        scores: list[tuple[float, float]] = []
        manual = speaker in overrides
        if not manual and sum(durations) >= 3:
            import numpy as np

            if session is None:
                import onnxruntime as ort

                LOG.info("Gender ECAPA q8-v2: MIT, 16 MB, pinned SHA256; acoustic estimates only")
                model = download(GENDER_URL, cache / "gender-ecapa-q8-v2.onnx", sha256=GENDER_SHA)
                options = ort.SessionOptions()
                options.intra_op_num_threads = threads
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
                inputs, outputs = session.get_inputs(), session.get_outputs()
                if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 2:
                    raise ValueError(f"Unexpected gender ONNX input interface: {[(i.name, i.type, i.shape) for i in inputs]}")
                if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 2:
                    raise ValueError(f"Unexpected gender ONNX output interface: {[(o.name, o.type, o.shape) for o in outputs]}")
                LOG.info("Gender ONNX interface: %s %s -> %s %s", inputs[0].name, inputs[0].shape,
                         outputs[0].name, outputs[0].shape)
            for start, end in samples:
                waveform = audio[round(start * SR):round(end * SR)][None, :].astype(np.float32)
                logits = np.asarray(session.run(None, {session.get_inputs()[0].name: waveform})[0])
                if logits.shape != (1, 2) or not np.isfinite(logits).all():
                    raise ValueError(f"Unexpected gender logits: shape={logits.shape}")
                probabilities = np.exp(logits[0] - logits[0].max())
                probabilities /= probabilities.sum()
                scores.append((float(probabilities[0]), float(probabilities[1])))
        if manual:
            label, confidence, source = overrides[speaker], None, "manual"
        else:
            label, confidence = aggregate_gender(scores, durations if scores else [], threshold)
            source = "model" if label != "unknown" else "unknown"
        metadata[speaker] = {
            "gender": label, "gender_confidence": confidence, "gender_source": source,
            "manual": manual, "evidence_duration": round(sum(durations) if scores else 0, 3),
            "scores": [{"male": s[0], "female": s[1], "duration": round(d, 3)} for s, d in zip(scores, durations)],
        }
        LOG.info("%s: %s, score=%s, evidence=%.1fs, source=%s", speaker, label, confidence,
                 metadata[speaker]["evidence_duration"], source)
    return metadata


def subtitle_records(
    words: list[Word], turns: list[Turn], speakers: dict[str, dict[str, Any]], offset: float = 0,
) -> list[dict[str, Any]]:
    """Split timed words on speaker changes, sentence ends, gaps and length limits."""
    records: list[dict[str, Any]] = []
    group: list[Word] = []
    current = UNKNOWN
    overlapping = False

    def flush() -> None:
        if not group:
            return
        metadata = speakers.get(current, {})
        start = max(0, round((group[0].start + offset) * 1000))
        end = max(start, round((max(w.end for w in group) + offset) * 1000))
        text = " ".join(w.text for w in group)
        # At most two approximately 42-character lines; a single long word is preserved.
        lines: list[str] = []
        line = ""
        for word in text.split():
            if line and len(line) + 1 + len(word) > 42 and not lines:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        records.append({
            "index": len(records) + 1, "start": start, "end": end, "duration": end - start,
            "text": lines, "speaker_id": current, "overlap": overlapping,
            "gender": metadata.get("gender", "unknown"),
            "gender_confidence": metadata.get("gender_confidence"),
            "gender_source": metadata.get("gender_source", "unknown"),
        })

    # Group each source independently: simultaneous identical phrases are not duplicates.
    for word in sorted(words, key=lambda w: (w.speaker, w.start)):
        if not word.text.strip():
            continue
        if not all(math.isfinite(t) for t in (word.start, word.end)) or word.start < 0 or word.end < word.start:
            raise ValueError(f"Invalid ASR word timing: {word}")
        speaker = word.speaker
        overlap = any(t.speaker != speaker and min(t.end, word.end) > max(t.start, word.start) for t in turns)
        if group and (speaker != current or word.end - group[0].start > 6
                      or sum(len(w.text) + 1 for w in group) + len(word.text) > 84
                      or word.start - group[-1].end > .8):
            flush()
            group = []
            overlapping = False
        current = speaker
        group.append(word)
        overlapping = overlapping or overlap
        if re.search(r"[.!?][\"')]*$", word.text):
            flush()
            group = []
            overlapping = False
    flush()
    records.sort(key=lambda record: (record["start"], record["end"], record["speaker_id"]))
    for index, record in enumerate(records, 1):
        record["index"] = index
        record["separation_status"] = speakers.get(record["speaker_id"], {}).get("separation_status", "not_run")
        left, right = record["start"] / 1000 - offset, record["end"] / 1000 - offset
        boundaries = sorted({left, right} | {max(left, min(right, t.start)) for t in turns}
                            | {max(left, min(right, t.end)) for t in turns})
        concurrent = max((len({t.speaker for t in turns if t.start < (a + b) / 2 < t.end})
                          for a, b in zip(boundaries, boundaries[1:])), default=0)
        record["needs_review"] = concurrent >= 3 or record["speaker_id"] == UNKNOWN
    return records


def write_outputs(prefix: Path, records: list[dict[str, Any]], speakers: dict[str, dict[str, Any]]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(prefix) + ".json")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(prefix) + ".speakers.json").write_text(
        json.dumps(speakers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    json_to_srt(str(json_path), str(prefix) + ".srt")


def run_pipeline(
    input_path: Path, output_prefix: Path, *, preset: str = "fast", start: float = 0,
    duration: float | None = None, threads: int = 4, model_dir: Path = Path("models/stt"),
    num_speakers: int | None = None, speaker_genders: Path | None = None,
    gender_threshold: float = .8, whisper_model: str = "large-v3-turbo",
    api_model: str = "microsoft/mai-transcribe-2", api_mode: str = "direct", api_budget: float = .10,
) -> list[dict[str, Any]]:
    """Run each model once over a clip; export times on the original media timeline."""
    if preset == "api":
        from utils.stt_openrouter import run_api

        return run_api(input_path, output_prefix, start=start, duration=duration, threads=threads,
            model_dir=model_dir, num_speakers=num_speakers, speaker_genders=speaker_genders,
            gender_threshold=gender_threshold, model=api_model, mode=api_mode, budget=api_budget)
    if preset not in ("fast", "accurate"):
        raise ValueError("--preset must be fast, accurate or api")
    overrides = validate_options(
        input_path, output_prefix, start=start, duration=duration, threads=threads,
        num_speakers=num_speakers, gender_threshold=gender_threshold, speaker_genders=speaker_genders,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    LOG.info("Decoding %s: start=%.3fs duration=%s", input_path, start, duration)
    audio = decode_media(input_path, start, duration)
    clip_duration = len(audio) / SR
    LOG.info("Decoded %.3fs mono 16kHz", clip_duration)
    # Digital silence needs neither credentials, downloads nor model inference.
    if len(audio) and abs(audio).max() > 1e-5:
        turns, sources, mapping = separate(audio, model_dir, threads, num_speakers)
    else:
        turns, sources, mapping = [], None, {}
    available = {t.speaker for t in turns}
    if set(overrides) - available:
        raise ValueError(f"Override IDs absent from this clip: {sorted(set(overrides) - available)}; available: {sorted(available)}")
    if turns:
        import numpy as np

        tracks = {}
        for speaker, info in mapping.items():
            waveform = sources.data[:, info["source_channel"]]
            origin = round(info["source_start"] * SR)
            track = np.zeros(len(audio), dtype=np.float32)
            source_first, destination_first = max(0, -origin), max(0, origin)
            length = min(len(waveform) - source_first, len(track) - destination_first)
            track[destination_first:destination_first + length] = waveform[source_first:source_first + length]
            tracks[speaker] = track
        refinement_windows = {speaker: info["refinement_windows"] for speaker, info in mapping.items()
                              if info.get("refinement_windows")}
        words = transcribe(tracks, turns, model_dir, threads, preset, whisper_model,
                           refinement_windows=refinement_windows or None)
        speakers = estimate_genders(audio, turns, model_dir, threads, gender_threshold, overrides)
        for speaker, info in speakers.items():
            info.update(mapping[speaker], gender_evidence_source="original_nonoverlapping_audio")
        records = subtitle_records(words, turns, speakers, offset=start)
    else:
        LOG.info("No speech detected; writing empty subtitles and speaker metadata")
        records, speakers = [], {}
    write_outputs(output_prefix, records, speakers)
    elapsed = time.monotonic() - started
    LOG.info("Done: %d subtitles, %d speakers, %.2fs elapsed, RTF %.3f (downloads included if any)",
             len(records), len(speakers), elapsed, elapsed / clip_duration if clip_duration else 0)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local English video or audio")
    parser.add_argument("-o", "--output-prefix", type=Path, required=True)
    parser.add_argument("--preset", choices=("fast", "accurate", "api"), default="fast")
    parser.add_argument("--api-model", choices=("microsoft/mai-transcribe-2", "meta/muse-voice-transcribe-1.0"),
                        default="microsoft/mai-transcribe-2")
    parser.add_argument("--api-mode", choices=("direct", "hybrid"), default="direct",
                        help="API mode only: explicit audio upload, native STT or local separation + remote STT")
    parser.add_argument("--api-budget", type=float, default=.10,
                        help="Client USD guard; shared output-directory ledger, not a server-side billing cap")
    parser.add_argument("--start", type=float, default=0, help="Clip start, seconds on original timeline")
    parser.add_argument("--duration", type=float, help="Selected clip duration in seconds")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1), help="CPU threads, 1–32")
    parser.add_argument("--model-dir", type=Path, default=Path("models/stt"))
    parser.add_argument("--num-speakers", type=int, help="Known positive speaker count (optional)")
    parser.add_argument("--speaker-genders", type=Path, help="JSON speaker label overrides")
    parser.add_argument("--gender-threshold", type=float, default=.8)
    parser.add_argument("--whisper-model", default="large-v3-turbo", help="Accurate preset model name or local directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        run_pipeline(
            args.input, args.output_prefix, preset=args.preset, start=args.start, duration=args.duration,
            threads=args.threads, model_dir=args.model_dir, num_speakers=args.num_speakers,
            speaker_genders=args.speaker_genders, gender_threshold=args.gender_threshold,
            whisper_model=args.whisper_model,
            api_model=args.api_model, api_mode=args.api_mode, api_budget=args.api_budget,
        )
    except Exception as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
