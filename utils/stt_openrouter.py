"""Explicit OpenRouter STT uploads; real timestamps only, client-side billing ledger."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import textwrap
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import wave

MAI = "microsoft/mai-transcribe-2"
MUSE = "meta/muse-voice-transcribe-1.0"
RATES_PER_HOUR = {MAI: .10, MUSE: .18}
# Candidate OpenRouter contracts for a bounded short-clip experiment, not verified live support.
EXPERIMENT_OPTIONS = {
    "elevenlabs/scribe-v2": {"diarize": True},
    "deepgram/nova-3": {"provider": {"options": {"deepgram": {"diarize": True}}}},
    # Observed 2026-10-13: diarize=True rejected on this route; retained as historical attempt only.
    "x-ai/grok-stt-1.0": {"diarize": True},
    "google/gemini-3.5-transcribe": {"diarize": True},
}
EXPERIMENT_MAX_SECONDS = 30
EXPERIMENT_RESERVATION = .05  # Fixed conservative reserve, not a provider audio-hour rate.
ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
MAX_SECONDS = 180
MAX_BYTES = 25 * 1024 * 1024
API_ENDPOINT_TOLERANCE = .25
UNKNOWN = "SPEAKER_UNKNOWN"


def load_key(env_file: Path = Path(".env")) -> str:
    """Read only the needed local dotenv assignment as data; environment wins, never eval."""
    value = os.environ.get("OPENROUTER_API_KEY")
    if value is None and env_file.is_file():
        found = False
        with env_file.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                match = re.match(r"^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*$", line)
                if not match:
                    continue
                if found:
                    raise ValueError("Duplicate OPENROUTER_API_KEY assignments; review local configuration")
                found = True
                raw = match[1]
                if raw.startswith('"'):
                    try:
                        value, end = json.JSONDecoder().raw_decode(raw)
                        if raw[end:].strip() and not raw[end:].lstrip().startswith("#"):
                            raise ValueError()
                    except (ValueError, json.JSONDecodeError):
                        raise ValueError("Invalid quoted OPENROUTER_API_KEY assignment") from None
                elif raw.startswith("'"):
                    closing = raw.find("'", 1)
                    if closing < 0 or (raw[closing + 1:].strip() and not raw[closing + 1:].lstrip().startswith("#")):
                        raise ValueError("Invalid quoted OPENROUTER_API_KEY assignment")
                    value = raw[1:closing]
                else:
                    value = raw.split(" #", 1)[0].strip()
    if not value:
        raise ValueError("Set OPENROUTER_API_KEY in the environment or local ignored .env before API mode")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Invalid OPENROUTER_API_KEY format; no shell expansion or header controls allowed")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sanitized(value: Any, key: str, audio_data: str | None = None) -> Any:
    """Do not persist authentication, inline audio, or secret echoes in vendor replies."""
    if isinstance(value, dict):
        return {str(k): ("[redacted]" if re.sub(r"[^a-z0-9]", "", str(k).casefold()) in
                {"authorization", "apikey", "openrouterapikey", "token", "secret", "accesstoken",
                 "refreshtoken", "clientsecret", "secretkey", "apisecret", "inputaudio", "base64", "audiodata"}
                else sanitized(v, key, audio_data)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(item, key, audio_data) for item in value]
    if isinstance(value, str):
        text = re.sub(re.escape(key), "[redacted]", value, flags=re.IGNORECASE)
        return (text.replace(audio_data, "[redacted inline audio]") if audio_data else text)[:100000]
    if isinstance(value, float) and not math.isfinite(value):
        return "[invalid nonfinite number]"
    return value


def pcm_wav(audio: Any) -> bytes:
    """Deterministic mono PCM16/16kHz: round after clip; complete WAV container, not float WAV."""
    import numpy as np

    samples = np.asarray(audio)
    if samples.ndim != 1 or not 0 < len(samples) <= MAX_SECONDS * 16000 or not np.isfinite(samples).all():
        raise ValueError("API audio must be finite mono samples, 0–180 seconds at 16kHz")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return stream.getvalue()


def wav_seconds(data: bytes) -> float:
    if len(data) > MAX_BYTES:
        raise ValueError("Inline WAV exceeds the 25MiB client guard")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("API requires mono PCM16 WAV at 16kHz")
            frames = audio.getnframes()
            if len(audio.readframes(frames)) != frames * 2:
                raise ValueError("Truncated WAV payload")
    except (wave.Error, EOFError):
        raise ValueError("Invalid WAV payload") from None
    duration = frames / 16000
    if not 0 < duration <= MAX_SECONDS:
        raise ValueError("API request duration must be within 0–180 seconds")
    return duration


def request_body(model: str, data: bytes) -> dict[str, Any]:
    seconds = wav_seconds(data)
    if model not in RATES_PER_HOUR and model not in EXPERIMENT_OPTIONS:
        raise ValueError("Unsupported API model")
    if model in EXPERIMENT_OPTIONS and seconds > EXPERIMENT_MAX_SECONDS:
        raise ValueError("Experimental API models require clips <=30 seconds")
    body: dict[str, Any] = {"model": model, "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": "wav"}}
    if model == MAI:
        body.update(response_format="verbose_json", timestamp_granularities=["segment", "word"],
                    provider={"options": {"azure": {"diarization": {"enabled": True}}}})
    elif model == MUSE:
        # Candidate passthrough only: live OpenRouter normalization has not been verified.
        body["provider"] = {"options": {"meta": {"mode": "DIARIZATION", "languageBias": ["English"]}}}
    else:
        body.update(response_format="verbose_json", timestamp_granularities=["segment", "word"])
        body.update(json.loads(json.dumps(EXPERIMENT_OPTIONS[model])))
    return body


class BudgetLedger:
    """Shared across models/chunks; reserve before POST, unknown billing blocks subsequent POSTs."""

    def __init__(self, budget: float, path: Path | None = None, *, max_requests: int = 18):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("--api-budget must be finite and positive")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        self.budget, self.path = budget, path
        self.max_requests = max_requests
        self.entries: list[dict[str, Any]] = []
        if path is not None and any(p.is_symlink() for p in
                                   (path, path.with_suffix(".lock"), path.with_suffix(".pending.json"))):
            raise ValueError("API ledger paths must not be symlinks")
        self.reload()

    def reload(self) -> None:
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") != 1 or not isinstance(saved.get("requests"), list):
                raise ValueError("Invalid API budget ledger; review before any upload")
            self.entries = saved["requests"]
            for entry in self.entries:
                if not isinstance(entry, dict) or "actual_cost_usd" not in entry:
                    raise ValueError("Invalid API budget ledger request")
                cost = entry["actual_cost_usd"]
                if cost is not None:
                    number(cost, "ledger cost")

    def save(self) -> None:
        if self.path is not None:
            # Atomic replace: a crash cannot leave partially-written billing state.
            temporary = self.path.with_suffix(".pending.json")
            if temporary.exists() or temporary.is_symlink():
                raise ValueError("Pending ledger file already exists; review billing state before proceeding")
            if self.path.exists() and not self.path.stat().st_mode & 0o222:
                raise ValueError("Read-only ledger snapshot must not be overwritten")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(json.dumps({"version": 1, "budget_usd": self.budget, "requests": self.entries,
                    "actual_total_usd": self.total, "billing_unknown": any(e["actual_cost_usd"] is None for e in self.entries)},
                    ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.path)

    @contextmanager
    def locked(self):
        if self.path is None:
            yield
            return
        import fcntl  # Linux offline/live runtime; fail before upload on unsupported platforms.

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another API caller is updating this ledger; no upload attempted") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def total(self) -> float | None:
        return None if any(e["actual_cost_usd"] is None for e in self.entries) else sum(e["actual_cost_usd"] for e in self.entries)

    def reserve(self, model: str, seconds: float) -> dict[str, Any]:
        with self.locked():
            self.reload()
            estimate = self.check(model, seconds)
            entry = {"request_scope": f"request_{len(self.entries) + 1:04d}", "model": model, "seconds": seconds,
                      "preflight_cost_usd": estimate, "actual_cost_usd": None, "status": "pending_billing"}
            if model in EXPERIMENT_OPTIONS:
                entry["preflight_cost_kind"] = "fixed_experiment_reservation_not_provider_rate"
            self.entries.append(entry)
            self.save()
            return entry

    def check(self, model: str, seconds: float) -> float:
        if model not in RATES_PER_HOUR and model not in EXPERIMENT_OPTIONS:
            raise ValueError("Unsupported API model")
        number(seconds, "request duration")
        if not 0 < seconds <= MAX_SECONDS:
            raise ValueError("Request duration exceeds client limit")
        if model in EXPERIMENT_OPTIONS and (seconds > EXPERIMENT_MAX_SECONDS or self.budget > 1):
            raise ValueError("Experimental models require <=30s clips and a shared budget <=1 USD")
        if self.total is None:
            raise ValueError("Previous API request has unknown billing; reconcile ledger before another paid request")
        # MAI/Muse use audio-hour rates, not pricing.prompt; experimental models use no guessed tariff.
        estimate = (EXPERIMENT_RESERVATION if model in EXPERIMENT_OPTIONS else
                    max(.001, seconds * RATES_PER_HOUR[model] / 3600 * 10))
        if len(self.entries) >= self.max_requests or self.total + estimate > self.budget:
            raise ValueError("API client request/budget limit exceeded before upload")
        return estimate


def number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Invalid nonnegative numeric {field}")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"Invalid nonnegative numeric {field}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative numeric {field}")
    return result


class Client:
    def __init__(self, key: str, ledger: BudgetLedger, opener: Any = None):
        self.key, self.ledger = key, ledger
        self.opener = opener or urllib.request.urlopen

    def cost_lookup(self, generation: str) -> tuple[float | None, dict[str, Any] | None]:
        url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode({"id": generation})
        for _ in range(2):
            try:
                request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.key})
                with self.opener(request, timeout=70) as response:
                    data = json.loads(response.read(MAX_BYTES + 1))
                cost = number(data["data"]["total_cost"], "generation cost")
                return cost, sanitized(data, self.key)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
                continue  # Read-only GET retry, never a charged POST retry.
        return None, None

    def call(self, model: str, data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
        seconds = wav_seconds(data)
        payload = request_body(model, data)
        body = json.dumps(payload).encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError("Encoded JSON request exceeds the 25MiB client guard")
        request = urllib.request.Request(ENDPOINT, data=body, method="POST",
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        entry = self.ledger.reserve(model, seconds)
        entry["wav_sha256"] = hashlib.sha256(data).hexdigest()
        begun = time.monotonic()
        try:
            with self.opener(request, timeout=70) as response:
                generation = response.headers.get("X-Generation-Id")
                entry["generation_id"] = generation
                content = response.read(MAX_BYTES + 1)
                if len(content) > MAX_BYTES:
                    raise ValueError("API response exceeds client size limit")
                raw = json.loads(content)
                entry["post_wall_seconds"] = time.monotonic() - begun
                if not isinstance(raw, dict):
                    raise ValueError("API response must be an object")
                raw = sanitized(raw, self.key, payload["input_audio"]["data"])
                entry["response"] = raw
                usage = raw.get("usage", {})
                cost = usage.get("cost", usage.get("costUSD")) if isinstance(usage, dict) else None
                if cost is not None:
                    entry["actual_cost_usd"] = number(cost, "usage cost")
                    entry["cost_origin"] = "response_usage"
                elif generation:
                    cost, metadata = self.cost_lookup(generation)
                    entry["actual_cost_usd"], entry["generation_metadata"] = cost, metadata
                    entry["cost_origin"] = "generation_lookup" if cost is not None else "unknown"
                entry["status"] = "received" if entry["actual_cost_usd"] is not None else "billing_unknown"
                return raw, entry
        except urllib.error.HTTPError as exc:
            entry["status"], entry["http_status"] = "http_error_billing_unknown", exc.code
            # Persist sanitized structured error evidence, never log reflected audio/auth.
            try:
                content = exc.read(MAX_BYTES + 1)
                if len(content) <= MAX_BYTES:
                    entry["response"] = sanitized(json.loads(content), self.key, payload["input_audio"]["data"])
            except (ValueError, OSError):
                pass
            generation = exc.headers.get("X-Generation-Id") if exc.headers else None
            entry["generation_id"] = generation
            if generation:
                cost, metadata = self.cost_lookup(generation)
                entry["actual_cost_usd"], entry["generation_metadata"] = cost, metadata
                entry["cost_origin"] = "generation_lookup" if cost is not None else "unknown"
                if cost is not None:
                    entry["status"] = "http_error_cost_known"
            billing = "unknown" if entry["actual_cost_usd"] is None else "known; inspect ledger"
            raise ValueError(f"OpenRouter HTTP {exc.code}; no POST retry; billing {billing}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            entry["status"] = "transport_error_billing_unknown"
            raise ValueError("OpenRouter transport/timeout failure; no POST retry; billing remains unknown") from None
        except (ValueError, TypeError):
            entry["status"] = "invalid_reply_billing_unknown"
            raise ValueError("Invalid OpenRouter reply; no POST retry; review saved response/billing") from None
        finally:
            entry["wall_seconds"] = time.monotonic() - begun
            self.ledger.save()


@dataclass
class Parsed:
    text: str
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    timing_origin: str
    diagnostics: dict[str, Any]


def parse_reply(raw: dict[str, Any], duration: float) -> Parsed:
    """Normalize typed native timing fields only; Meta turns are candidate format, not live proof."""
    duration = number(duration, "clip duration")
    if duration <= 0:
        raise ValueError("API clip duration must be positive")
    text = raw.get("text", "")
    if not isinstance(text, str):
        raise ValueError("API text must be a string")

    # Export one representation only; authoritative native Meta turns take precedence.
    selected = next((name for name in ("turns", "segments", "words") if raw.get(name)), None)
    diagnostics: dict[str, Any] = {"selected_collection": selected,
        "ignored_collections": [name for name in ("turns", "segments", "words")
                                if name in raw and name != selected],
        "skipped_blank_items": 0, "skipped_non_speech_items": 0, "clipped_endpoints": []}

    def normalize(item: Any, index: int) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            raise ValueError("API timed item must be an object")
        native, word = selected == "turns", selected == "words"
        if word and item.get("type") in ("spacing", "audio_event"):
            diagnostics["skipped_non_speech_items"] += 1
            return None
        if word and "type" in item and item["type"] != "word":
            raise ValueError("Unknown API word item type")
        value = item.get("transcript" if native else ("word" if word else "text"))
        if word and "word" not in item:
            value = item.get("text")
        if not isinstance(value, str):
            raise ValueError("API timed item text must be a string")
        if not value.strip():
            diagnostics["skipped_blank_items"] += 1
            return None
        start = number(item.get("startMs" if native else "start"), "start") / (1000 if native else 1)
        end = number(item.get("endMs" if native else "end"), "end") / (1000 if native else 1)
        if end < start:
            raise ValueError("API timestamp end precedes start")
        if start >= duration or end <= 0:
            raise ValueError("API cue is wholly outside the clip")
        if end > duration + API_ENDPOINT_TOLERANCE + 1e-9:
            raise ValueError("API endpoint overshoot exceeds 250ms")
        clipped = end > duration
        if clipped:
            diagnostics["clipped_endpoints"].append({"item_index": index, "raw_start_seconds": start,
                "raw_end_seconds": end, "end_seconds": duration, "overshoot_seconds": end - duration})
            end = duration
        label = item.get("speaker")
        if label is None:
            label = item.get("speaker_label")
        if label is None and word:
            label = item.get("speaker_id")
        if isinstance(label, float):
            if not math.isfinite(label) or label < 0 or not label.is_integer():
                raise ValueError("Invalid API speaker label")
            label = int(label)
        if isinstance(label, str):
            label = label.strip()
        if label is not None and (isinstance(label, bool) or not isinstance(label, (str, int))
                                  or (isinstance(label, int) and label < 0)
                                  or (isinstance(label, str) and not 0 < len(label.strip()) <= 128)):
            raise ValueError("Invalid API speaker label")
        return {"start": start, "end": end, "text": value.strip(), "label": label,
                "timestamp_clipped": clipped}

    normalized = []
    if selected is not None:
        if not isinstance(raw[selected], list):
            raise ValueError(f"API {selected} must be a list")
        for index, item in enumerate(raw[selected]):
            result = normalize(item, index)
            if result is not None:
                normalized.append(result)
    segments = normalized if selected in ("turns", "segments") else []
    words = normalized if selected == "words" else []
    if not text.strip():
        text = " ".join(item["text"] for item in normalized)
    origins = {"turns": "native_turns_ms", "segments": "native_segments_seconds", "words": "native_words_seconds"}
    return Parsed(text, segments, words, origins[selected] if normalized else "unavailable", diagnostics)


def records_for(parsed: Parsed, scope: str, *, offset: float = 0, speaker: str | None = None,
                speakers: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Keep segment boundaries/overlap verbatim; no synthetic per-word alignment or gender inference."""
    speakers = dict(speakers or {})
    labels: dict[Any, str] = {}
    for item in sorted(parsed.segments or parsed.words, key=lambda x: x["start"]):
        label = item["label"]
        if speaker is None and label is not None and label not in labels:
            labels[label] = f"SPEAKER_{len(labels):02d}"
    records = []
    for item in parsed.segments or parsed.words:
        identity = speaker or labels.get(item["label"], UNKNOWN)
        metadata = speakers.setdefault(identity, {"gender": "unknown", "gender_confidence": None,
            "gender_source": "unknown", "separation_status": "api_diarization_not_audio_separation"})
        if speaker is None:
            metadata.update(api_request_scope=scope, api_speaker_label=item["label"],
                            identity_origin="native_api_label" if item["label"] is not None else "unknown")
        start, end = round((item["start"] + offset) * 1000), round((item["end"] + offset) * 1000)
        records.append({"index": 0, "start": start, "end": end, "duration": end - start,
            "text": textwrap.wrap(item["text"], width=42, break_long_words=False, break_on_hyphens=False),
            "speaker_id": identity, "gender": metadata.get("gender", "unknown"),
            "gender_confidence": metadata.get("gender_confidence"), "gender_source": metadata.get("gender_source", "unknown"),
            "separation_status": metadata["separation_status"], "api_request_scope": scope,
            "timestamp_origin": parsed.timing_origin, "timestamp_clipped": item["timestamp_clipped"],
            "needs_review": True})
    records.sort(key=lambda r: (r["start"], r["end"], r["speaker_id"]))
    for i, record in enumerate(records, 1):
        record["index"] = i
        record["overlap"] = any(other["speaker_id"] != record["speaker_id"] and other["start"] < record["end"]
                                and other["end"] > record["start"] for other in records)
    return records, speakers


def validate_api(input_path: Path, prefix: Path, duration: float | None, model: str, mode: str,
                 budget: float, overrides: Path | None) -> Path:
    if model not in RATES_PER_HOUR or mode not in ("direct", "hybrid"):
        raise ValueError("Invalid API model or mode")
    if duration is None or not math.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
        raise ValueError("API mode requires explicit --duration within 0–180s; comparison clips should be <=30s")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("--api-budget must be finite and positive")
    if input_path.name.startswith(".env") or (overrides and overrides.name.startswith(".env")):
        raise ValueError("Secret configuration is not an API audio input")
    ledger = prefix.parent / ".stt_api_budget.json"
    if ledger.with_suffix(".pending.json").exists():
        raise ValueError("Pending ledger file exists; review before API mode")
    protected = [input_path, Path(".env"), ledger, ledger.with_suffix(".lock"),
                 ledger.with_suffix(".pending.json")] + ([overrides] if overrides else [])
    for suffix in (".json", ".srt", ".speakers.json", ".api.json", ".transcript.txt"):
        output = Path(str(prefix) + suffix)
        if output.name.startswith(".env") or output.exists() or output.is_symlink() or any(
            output.resolve() == path.resolve() or (output.exists() and path.exists() and output.samefile(path))
            for path in protected
        ):
            raise ValueError(f"API output collision or existing file: {output}")
    for target in (ledger, ledger.with_suffix(".lock"), ledger.with_suffix(".pending.json")):
        if target.is_symlink():
            raise ValueError("API ledger paths must not be symlinks")
        if any(target.resolve() == path.resolve() or (target.exists() and path.exists() and target.samefile(path))
               for path in [input_path, Path(".env")] + ([overrides] if overrides else [])):
            raise ValueError("API ledger would overwrite input, credentials or overrides")
    return ledger


def run_api(input_path: Path, prefix: Path, *, start: float, duration: float | None, threads: int,
            model_dir: Path, num_speakers: int | None, speaker_genders: Path | None,
            gender_threshold: float, model: str, mode: str, budget: float) -> list[dict[str, Any]]:
    from utils import stt_pipeline as stt

    ledger_path = validate_api(input_path, prefix, duration, model, mode, budget, speaker_genders)
    overrides = stt.validate_options(input_path, prefix, start=start, duration=duration, threads=threads,
        num_speakers=num_speakers, gender_threshold=gender_threshold, speaker_genders=speaker_genders)
    key = load_key()  # Before decoding, uploading, local model loading, or output creation.
    ledger = BudgetLedger(budget, ledger_path)
    ledger.check(model, duration)
    if mode == "direct" and overrides:
        raise ValueError("Direct API manual speaker overrides require reviewed native IDs; use hybrid or edit afterward")
    client = Client(key, ledger)
    begun = time.monotonic()
    audio = stt.decode_media(input_path, start, duration)
    records, speakers, transcripts, evidence = [], {}, [], []
    timed = True
    exported = False

    def capture_reply(raw: dict[str, Any], entry: dict[str, Any], seconds: float, tag: str = "") -> Parsed:
        # Preserve successful paid text/evidence even when its selected timings are invalid.
        evidence.append(entry)
        text = raw.get("text")
        position = len(transcripts) if isinstance(text, str) and text.strip() else None
        if position is not None:
            transcripts.append(tag + text)
        try:
            parsed = parse_reply(raw, seconds)
        except ValueError as exc:
            entry["timing_error"] = str(exc)
            raise
        entry["timing_normalization"] = parsed.diagnostics
        if position is not None:
            transcripts[position] = tag + parsed.text
        elif parsed.text.strip():
            transcripts.append(tag + parsed.text)
        return parsed

    try:
        if mode == "direct":
            raw, entry = client.call(model, pcm_wav(audio))
            parsed = capture_reply(raw, entry, len(audio) / stt.SR)
            records, speakers = records_for(parsed, entry["request_scope"], offset=start)
            timed = bool(parsed.segments or parsed.words)
        else:
            import numpy as np

            turns, sources, mapping = stt.separate(audio, model_dir, threads, num_speakers)
            if set(overrides) - set(mapping):
                raise ValueError("Manual override IDs are absent from local separation")
            speakers = stt.estimate_genders(audio, turns, model_dir, threads, gender_threshold, overrides)
            for speaker, info in mapping.items():
                speakers[speaker].update(info, identity_origin="local_pixit_source",
                    api_separation_origin="local_pixit_sepformer_not_remote", gender_evidence_source="original_nonoverlapping_audio")
                track = np.zeros(len(audio), dtype=np.float32)
                origin = round(info["source_start"] * stt.SR)
                source_first, destination_first = max(0, -origin), max(0, origin)
                count = min(len(sources.data) - source_first, len(track) - destination_first)
                track[destination_first:destination_first + count] = sources.data[source_first:source_first + count, info["source_channel"]]
                chunks = stt.asr_chunks([t for t in turns if t.speaker == speaker], len(track) / stt.SR,
                                        info.get("refinement_windows", []))
                for left, right in chunks:
                    first, last = round(left * stt.SR), round(right * stt.SR)
                    samples = track[first:last]
                    if not len(samples) or np.abs(samples).max() <= 1e-5:
                        continue
                    # PixIT's SI-SDR gain is arbitrary. Attenuate only for PCM range,
                    # not a separation-quality fix; never hard-clip huge local peaks.
                    peak = float(np.abs(samples).max())
                    transport_gain = min(1., 1. / peak)
                    raw, entry = client.call(model, pcm_wav(samples * transport_gain))
                    entry.update(local_source_speaker=speaker, local_source_peak=peak,
                                 pcm_transport_gain=transport_gain)
                    parsed = capture_reply(raw, entry, len(samples) / stt.SR, f"[{speaker} {start + left:.3f}s] ")
                    rows, _ = records_for(parsed, entry["request_scope"], offset=start + first / stt.SR,
                                           speaker=speaker, speakers=speakers)
                    records.extend(rows)
                    timed = timed and bool(parsed.segments or parsed.words)
            records.sort(key=lambda r: (r["start"], r["end"], r["speaker_id"]))
            for i, record in enumerate(records, 1):
                record["index"] = i
                record["overlap"] = any(other["speaker_id"] != record["speaker_id"] and
                    other["start"] < record["end"] and other["end"] > record["start"] for other in records)
        if timed:
            stt.write_outputs(prefix, records, speakers)
            exported = True
        else:
            records = []
        if not timed:
            raise ValueError("API returned text without usable native timing; transcript/raw API saved, JSON/SRT unavailable")
        return records
    finally:
        if transcripts:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            Path(str(prefix) + ".transcript.txt").write_text("\n".join(transcripts) + "\n", encoding="utf-8")
        write_json(Path(str(prefix) + ".api.json"), {"model": model, "mode": mode,
            "live_capability_claim": "Only saved response fields are evidence; no remote audio separation or gender",
            "original_start_seconds": start, "clip_seconds": len(audio) / stt.SR,
            "pipeline_wall_seconds": time.monotonic() - begun, "ledger": str(ledger_path),
            "actual_total_usd": ledger.total, "requests": evidence,
            "ledger_requests": ledger.entries, "timing_export_available": exported})
