"""Plan or explicitly upload whole media as sequential MAI chunks; never retry a POST."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import wave

from utils import stt_openrouter as api
from utils.json_to_srt import json_to_srt

RATE, CHUNK_FRAMES = 16000, 60 * 16000
SUFFIXES = (".txt", ".json", ".srt", ".speakers.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_json(left: Any, right: Any) -> bool:
    """Keep JSON booleans/floats distinct from integer timestamps/labels when checking caches."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def history_hash(entries: list[dict[str, Any]], count: int) -> str:
    return digest(json.dumps(entries[:count], sort_keys=True, ensure_ascii=False).encode())


def safe_path(path: Path) -> None:
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink paths are not allowed")
    require(not any(p.name.startswith(".env") for p in (path, *path.parents)), "Credential path rejected")


def validate_paths(source: Path, out: Path, ledger: Path | None) -> None:
    for path in (source, out, *([ledger, ledger.with_suffix(".lock"), ledger.with_suffix(".pending.json")]
                               if ledger is not None else [])):
        safe_path(path)
    require(source.is_file(), "Input media must be an existing file")
    require(not out.exists() or out.is_dir(), "Output must be a directory")
    protected = [source, Path(".env")] + ([ledger, ledger.with_suffix(".lock"),
                 ledger.with_suffix(".pending.json")] if ledger is not None else [])
    if ledger is not None:
        require(ledger.is_file(), "Use an existing shared budget ledger; never create/reset one here")
        require(not ledger.resolve().is_relative_to(out.resolve()), "Ledger must be outside episode output")
        require(ledger.name != "ledger_snapshot.json" and "stt_api_results" not in ledger.parts,
                "Historical snapshot is not an active ledger")
        require(not ledger.with_suffix(".pending.json").exists(), "Pending ledger file; reconcile first")
        require(not any(ledger.samefile(p) for p in protected[:2] if p.exists()), "Ledger/input/credential collision")
    require(not (Path(".env").exists() and source.samefile(Path(".env"))), "Credential input rejected")
    for path in out.rglob("*") if out.exists() else []:
        safe_path(path)
        require(not any(path.samefile(p) for p in protected if p.exists()), "Output/protected-file collision")
        require(not path.name.endswith(".pending"), "Pending episode write; review before resuming")


def save(path: Path, value: Any) -> None:
    safe_path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_stats(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {"path": str(source.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_matches(source: Path, plan: dict[str, Any], *, check_hash: bool = False) -> None:
    actual, expected = source_stats(source), plan["source"]
    require(Path(expected["path"]).resolve() == source.resolve(), "Different input from saved plan")
    require(all(actual[k] == expected[k] for k in ("bytes", "mtime_ns")), "Source changed since preparation")
    if check_hash and expected.get("sha256"):
        require(file_hash(source) == expected["sha256"], "Source hash changed since preparation")


def probe(source: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,sample_rate,channels,start_time:format=duration", "-of", "json", str(source.resolve())],
        check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    require(stream is not None, "No audio stream")
    require(stream.get("start_time") in (None, "N/A") or float(stream["start_time"]) == 0,
            "Nonzero audio stream start requires reviewed timeline handling; no upload")
    seconds = float(data["format"]["duration"])
    require(math.isfinite(seconds) and seconds > 0, "Invalid media duration")
    return {**source_stats(source), "selected_audio_stream": stream, "duration_seconds": seconds}


def wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    return output.getvalue()


def chunk_path(out: Path, chunk: dict[str, Any], name: str) -> Path:
    return out / "chunks" / f"{chunk['number']:04d}" / name


def validate_plan(source: Path, out: Path, plan: dict[str, Any], ledger: api.BudgetLedger) -> None:
    source_matches(source, plan, check_hash=True)
    stream = plan["source"].get("selected_audio_stream", {})
    require(stream.get("start_time") in (None, "N/A") or float(stream["start_time"]) == 0,
            "Saved nonzero audio start requires reviewed timeline handling")
    require(plan["model"] == api.MAI and plan["mode"] == "direct", "Saved model/mode mismatch")
    require(plan.get("rate", RATE) == RATE and plan.get("chunk_seconds", 60) == 60, "Saved format mismatch")
    require(Path(plan["active_ledger"]).resolve() == ledger.path.resolve(), "Different shared ledger")
    count = plan["historical_reservations"]
    require(type(count) is int and 0 <= count <= len(ledger.entries), "Invalid historical boundary")
    require(history_hash(ledger.entries, count) == plan["historical_entries_sha256"], "Ledger history changed")
    cursor = 0
    require(bool(plan["chunks"]), "Empty chunk plan")
    for number, chunk in enumerate(plan["chunks"], 1):
        frames = chunk["end_sample"] - cursor
        require(type(chunk["number"]) is int and chunk["number"] == number and type(chunk["start_sample"]) is int
                and type(chunk["end_sample"]) is int and chunk["start_sample"] == cursor
                and 0 < frames <= CHUNK_FRAMES and chunk["seconds"] == frames / RATE, "Invalid sample-grid plan")
        require(number == len(plan["chunks"]) or frames == CHUNK_FRAMES, "Nonfinal short chunk")
        require(chunk["expected_request_scope"] == f"request_{count + number:04d}", "Request scope mismatch")
        require(Path(chunk["wav_path"]).resolve() == chunk_path(out, chunk, "audio.wav").resolve(), "WAV path mismatch")
        data = chunk_path(out, chunk, "audio.wav").read_bytes()
        require(digest(data) == chunk["wav_sha256"] and api.wav_seconds(data) == chunk["seconds"], "WAV hash/format mismatch")
        require(same_json(read(chunk_path(out, chunk, "input.json")), chunk), "Chunk input metadata mismatch")
        cursor = chunk["end_sample"]
    require(cursor == plan["decoded_frames"] and cursor / RATE == plan["decoded_duration_seconds"], "Coverage mismatch")


def native(entry: dict[str, Any], chunk: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = api.parse_reply(entry["response"], chunk["seconds"])
    require(not parsed.text.strip() or bool(parsed.segments or parsed.words), "Nonempty text without native timing; STOP")
    rows, speakers = api.records_for(parsed, entry["request_scope"], offset=chunk["start_sample"] / RATE)
    namespace = f"CHUNK_{chunk['number']:04d}_"
    for row in rows:
        row.update(speaker_id=namespace + row["speaker_id"], chunk_number=chunk["number"], boundary_review_required=True)
    return rows, {namespace + identity: {**info, "identity_policy": "request-scoped, not episode-global",
        "chunk_number": chunk["number"], "chunk_start_sample": chunk["start_sample"],
        "chunk_end_sample": chunk["end_sample"], "needs_review": True} for identity, info in speakers.items()}


def bound_entry(ledger: api.BudgetLedger, plan: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any] | None:
    position = plan["historical_reservations"] + chunk["number"] - 1
    if len(ledger.entries) <= position:
        return None
    entry = ledger.entries[position]
    require(entry.get("request_scope") == chunk["expected_request_scope"] and entry.get("model") == api.MAI
            and entry.get("wav_sha256") == chunk["wav_sha256"] and entry.get("seconds") == chunk["seconds"],
            "Ledger request/input binding mismatch; no paid retry")
    require(entry.get("status") == "received" and entry.get("actual_cost_usd") is not None
            and isinstance(entry.get("response"), dict), "Lost reply/unknown billing/failed request; no paid retry")
    api.number(entry["actual_cost_usd"], "actual cost")
    usage = entry["response"].get("usage", {})
    if isinstance(usage, dict) and usage.get("seconds") is not None:
        seconds = api.number(usage["seconds"], "billed seconds")
        require(0 < seconds <= math.ceil(chunk["seconds"]) + .25
                and math.isclose(entry["actual_cost_usd"], seconds * .1 / 3600, rel_tol=1e-5, abs_tol=1e-9),
                "Billed duration/cost mismatch; STOP")
    return entry


def collect(out: Path, plan: dict[str, Any], ledger: api.BudgetLedger) -> tuple[list, dict, int]:
    rows, speakers, completed = [], {}, 0
    gap = False
    for chunk in plan["chunks"]:
        cache = chunk_path(out, chunk, "response.json")
        entry = bound_entry(ledger, plan, chunk)
        attempt = chunk_path(out, chunk, "attempt.json")
        if entry is not None and attempt.exists():
            require(read(attempt).get("chunk") == chunk, "Attempt/input metadata mismatch")
        if cache.exists():
            evidence = read(cache)
            require(same_json(evidence["chunk"], chunk) and same_json(evidence["entry"], entry), "Cached response/ledger mismatch")
            require(evidence["status"] in ("received", "received_cached_ledger", "native_validated"), "Failed cache; STOP")
            require(not gap, "Noncontiguous cached replies")
            current, metadata = native(entry, chunk)
            require(evidence.get("native_cue_count", len(current)) == len(current), "Cached native cue count mismatch")
            rows.extend(current)
            speakers.update(metadata)
            completed += 1
        else:
            gap = True
            require(entry is not None or not chunk_path(out, chunk, "attempt.json").exists(),
                    "Prior attempt without returned response; no paid retry")
    rows.sort(key=lambda row: (row["start"], row["end"], row["speaker_id"]))
    for number, row in enumerate(rows, 1):
        row["index"] = number
    return rows, speakers, completed


def transcript(rows: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{r['start'] / 1000:.3f}–{r['end'] / 1000:.3f}s] {r['speaker_id']}\n" +
                        " ".join(r["text"]) for r in rows) + "\n"


def exports_match(out: Path, rows: list, speakers: dict) -> None:
    require(same_json(read(out / "episode.json"), rows) and same_json(read(out / "episode.speakers.json"), speakers),
            "Completed JSON export differs from native replies")
    require((out / "episode.txt").read_text(encoding="utf-8") == transcript(rows), "Completed text mismatch")
    require((out / "episode.srt").read_text(encoding="utf-8") == json_to_srt(str(out / "episode.json")), "Completed SRT mismatch")


def publish(out: Path, rows: list, speakers: dict, *, partial: bool = False) -> None:
    from utils.stt_pipeline import write_outputs

    name = "episode.partial" if partial else "episode"
    with tempfile.TemporaryDirectory(prefix="export_", dir=out) as directory:
        prefix = Path(directory) / name
        write_outputs(prefix, rows, speakers)
        prefix.with_suffix(prefix.suffix + ".txt").write_text(transcript(rows), encoding="utf-8")
        for suffix in SUFFIXES:
            prepared, target = Path(str(prefix) + suffix), out / (name + suffix)
            safe_path(target)
            if partial:
                prepared.replace(target)
            elif target.exists():
                require(target.read_bytes() == prepared.read_bytes(), "Existing final export collision; no overwrite")
            else:
                os.link(prepared, target)  # Exclusive atomic publication; cannot overwrite existing files.


def prepare(source: Path, out: Path, info: dict, ledger: api.BudgetLedger) -> dict[str, Any]:
    original_hash = file_hash(source)
    audio_path = out / "episode_audio.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source.resolve()),
        "-map", f"0:{info['selected_audio_stream']['index']}", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(audio_path)], check=True)
    chunks = []
    history = len(ledger.entries)
    with wave.open(str(audio_path), "rb") as audio:
        require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) ==
                (1, 2, RATE, "NONE"), "Decoded WAV format mismatch")
        frames = audio.getnframes()
        require(frames > 0, "Empty decoded audio")
        for number, start in enumerate(range(0, frames, CHUNK_FRAMES), 1):
            end = min(start + CHUNK_FRAMES, frames)
            pcm = audio.readframes(end - start)
            require(len(pcm) == (end - start) * 2, "Truncated decoded PCM")
            data = wav(pcm)
            directory = out / "chunks" / f"{number:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            path = directory / "audio.wav"
            with path.open("xb") as output:
                output.write(data)
            chunk = {"number": number, "start_sample": start, "end_sample": end, "seconds": (end - start) / RATE,
                "wav_path": str(path.resolve()), "wav_sha256": digest(data), "PCM_samples_sha256": digest(pcm),
                "WAV_bytes": len(data), "expected_request_scope": f"request_{history + number:04d}"}
            save(directory / "input.json", chunk)
            chunks.append(chunk)
    require(source_stats(source) == {k: info[k] for k in ("path", "bytes", "mtime_ns")}, "Source changed during decode")
    plan = {"model": api.MAI, "mode": "direct", "source": {**info, "sha256": original_hash},
        "active_ledger": str(ledger.path.resolve()), "max_requests": ledger.max_requests, "budget_usd": ledger.budget,
        "historical_reservations": history, "historical_entries_sha256": history_hash(ledger.entries, history),
        "decoded_frames": frames, "decoded_duration_seconds": frames / RATE, "chunks": chunks,
        "audio_path": str(audio_path.resolve()), "no_POST_retries": True, "rate": RATE, "chunk_seconds": 60}
    save(out / "run_plan.json", plan)
    return plan


@contextmanager
def output_lock(out: Path):
    import fcntl

    with (out / "episode_run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another episode launcher owns this output") from None
        yield


def run_episode(source: Path, out: Path, *, allow_upload: bool = False, budget_ledger: Path | None = None,
                budget: float = 1, max_requests: int = 72) -> dict[str, Any]:
    """Plan without writes/credentials; completed native caches are always a read-only no-op."""
    require(not isinstance(budget, bool) and math.isfinite(budget) and 0 < budget <= 1, "Budget must be within (0, 1] USD")
    explicit_ledger = budget_ledger is not None
    api.BudgetLedger(budget, max_requests=max_requests)  # Validate positive non-bool integer cap.
    validate_paths(source, out, budget_ledger)
    plan_path = out / "run_plan.json"
    plan = read(plan_path) if plan_path.exists() else None
    if plan is not None and budget_ledger is None:
        budget_ledger = Path(plan["active_ledger"])
        validate_paths(source, out, budget_ledger)
    ledger = api.BudgetLedger(budget, budget_ledger, max_requests=max_requests) if budget_ledger is not None else None
    if ledger is not None:
        require(ledger.total is not None and ledger.total <= budget, "Unknown/exceeded shared billing; STOP")
    rows, speakers, completed = [], {}, 0
    if plan is not None:
        validate_plan(source, out, plan, ledger)
        rows, speakers, completed = collect(out, plan, ledger)
        manifest = read(out / "manifest.json") if (out / "manifest.json").exists() else {}
        if manifest.get("status") == "complete":
            require(completed == len(plan["chunks"]), "False complete flag; cached replies missing")
            exports_match(out, rows, speakers)
            return {"status": "complete", "completed_chunks": completed, "total_chunks": completed,
                "completed_input_seconds": plan["decoded_duration_seconds"], "total_input_seconds": plan["decoded_duration_seconds"],
                "actual_shared_USD": ledger.total, "native_cues": len(rows), "new_POSTs": 0,
                "empty_native_chunks": sorted({c["number"] for c in plan["chunks"]} - {r["chunk_number"] for r in rows}),
                "outputs": [str(out / ("episode" + suffix)) for suffix in SUFFIXES]}
        if any((out / ("episode" + suffix)).exists() for suffix in SUFFIXES):
            require(completed == len(plan["chunks"]), "Final output collision before all cached replies; STOP")
            for suffix, expected in ((".json", rows), (".speakers.json", speakers)):
                path = out / ("episode" + suffix)
                require(not path.exists() or same_json(read(path), expected), "Existing final JSON collision")
            path = out / "episode.txt"
            require(not path.exists() or path.read_text(encoding="utf-8") == transcript(rows), "Existing final text collision")
            path = out / "episode.srt"
            require(not path.exists() or ((out / "episode.json").exists()
                    and path.read_text(encoding="utf-8") == json_to_srt(str(out / "episode.json"))), "Existing final SRT collision")
        info = plan["source"]
        seconds = plan["decoded_duration_seconds"]
        count = len(plan["chunks"])
    else:
        require(not out.exists() or not any(out.iterdir()), "Nonempty output without recognized run plan; preserve it")
        info = probe(source)
        seconds, count = info["duration_seconds"], math.ceil(info["duration_seconds"] / 60)
    required_cap = ((plan["historical_reservations"] if plan else len(ledger.entries)) + count) if ledger else None
    if not allow_upload:
        remaining_seconds = ([c["seconds"] for c in plan["chunks"] if bound_entry(ledger, plan, c) is None]
                             if plan else [min(60, seconds - 60 * i) for i in range(count)])
        return {"status": "plan_only", "model": api.MAI, "source": info, "total_chunks": count,
            "completed_chunks": completed, "total_input_seconds": seconds, "required_total_request_cap": required_cap,
            "max_requests": max_requests, "budget_usd": budget, "actual_shared_USD": ledger.total if ledger else None,
            "remaining_POSTs": len(remaining_seconds), "forecast_new_USD": sum(math.ceil(s) * .1 / 3600 for s in remaining_seconds),
            "transport": "mono PCM16 WAV / 16000Hz / <=60s; no overlap", "no_decode_or_upload": True}
    require(ledger is not None, "--budget-ledger is required for new uploads")
    require(explicit_ledger or (plan is not None and all(bound_entry(ledger, plan, c) is not None for c in plan["chunks"])),
            "Explicit --budget-ledger is required before any new paid request")
    require(required_cap <= max_requests, "Explicit --max-requests insufficient for entire plan; never raised automatically")
    require(ledger.path.stat().st_mode & 0o222 != 0, "Read-only ledger is evidence, not an active upload budget")
    key = None
    posted = 0
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        # Repeat controls after acquiring lock; never use stale planning state for a POST.
        validate_paths(source, out, budget_ledger)
        if plan is None:
            require(set(p.name for p in out.iterdir()) == {"episode_run.lock"}, "Output changed since planning")
            ledger.check(api.MAI, min(60, seconds))
            key = api.load_key()
            plan = prepare(source, out, info, ledger)
        validate_plan(source, out, plan, ledger)
        require(plan["historical_reservations"] + len(plan["chunks"]) <= max_requests, "Decoded plan exceeds explicit request cap")
        try:
            for chunk in plan["chunks"]:
                source_matches(source, plan)
                ledger.reload()
                require(ledger.total is not None and ledger.total <= budget, "Unknown/exceeded shared billing; STOP")
                require(history_hash(ledger.entries, plan["historical_reservations"]) == plan["historical_entries_sha256"],
                        "Ledger history changed")
                cache = chunk_path(out, chunk, "response.json")
                entry = bound_entry(ledger, plan, chunk)
                if cache.exists():
                    require(same_json(read(cache)["entry"], entry) and same_json(read(cache)["chunk"], chunk), "Cached binding mismatch")
                elif entry is None:
                    require(not chunk_path(out, chunk, "attempt.json").exists(), "Prior attempt; no paid retry")
                    require(len(ledger.entries) + 1 == int(chunk["expected_request_scope"].split("_")[1]), "Shared ledger changed")
                    ledger.check(api.MAI, chunk["seconds"])
                    key = key or api.load_key()
                    save(chunk_path(out, chunk, "attempt.json"), {"chunk": chunk, "status": "POST_started"})
                    print(json.dumps({"status": "waiting", "chunk": chunk["number"], "total_chunks": len(plan["chunks"])}), flush=True)
                    data = chunk_path(out, chunk, "audio.wav").read_bytes()
                    require(digest(data) == chunk["wav_sha256"], "WAV changed before POST")
                    try:
                        posted += 1
                        _, entry = api.Client(key, ledger).call(api.MAI, data)
                    except Exception:
                        ledger.reload()
                        position = plan["historical_reservations"] + chunk["number"] - 1
                        if len(ledger.entries) > position:
                            save(cache, {"status": "request_failed", "chunk": chunk, "entry": ledger.entries[position]})
                        raise
                    # Save actual native reply and cost before any timing interpretation.
                    save(cache, {"status": "received", "chunk": chunk, "entry": entry})
                if not cache.exists():
                    save(cache, {"status": "received", "chunk": chunk, "entry": entry})  # Recover returned ledger reply.
                raw_text = entry["response"].get("text", "")
                if isinstance(raw_text, str):
                    text_path = chunk_path(out, chunk, "transcript.txt")
                    if text_path.exists():
                        require(text_path.read_text(encoding="utf-8") == raw_text + "\n", "Cached transcript collision")
                    else:
                        with text_path.open("x", encoding="utf-8") as output:
                            output.write(raw_text + "\n")
                bound_entry(ledger, plan, chunk)
                require(ledger.total is not None and ledger.total <= budget, "Unknown/exceeded shared billing after reply; STOP")
                current, _ = native(entry, chunk)
                evidence = read(cache)
                if evidence["status"] != "native_validated":
                    evidence.update(status="native_validated", native_cue_count=len(current))
                    save(cache, evidence)
                print(json.dumps({"status": "progress", "completed_chunks": chunk["number"], "total_chunks": len(plan["chunks"]),
                    "completed_input_seconds": chunk["end_sample"] / RATE, "total_input_seconds": plan["decoded_duration_seconds"],
                    "actual_shared_USD": ledger.total}), flush=True)
            rows, speakers, completed = collect(out, plan, ledger)
            source_matches(source, plan, check_hash=True)
            publish(out, rows, speakers)
            result = {"status": "complete", "completed_chunks": completed, "total_chunks": len(plan["chunks"]),
                "completed_input_seconds": plan["decoded_duration_seconds"], "total_input_seconds": plan["decoded_duration_seconds"],
                "actual_shared_USD": ledger.total, "native_cues": len(rows), "new_POSTs": posted,
                "active_ledger": str(ledger.path.resolve()), "max_requests": max_requests, "budget_usd": budget,
                "empty_native_chunks": [c["number"] for c in plan["chunks"]
                    if read(chunk_path(out, c, "response.json"))["native_cue_count"] == 0]}
            save(out / "manifest.json", result)
            return result
        except Exception as exc:
            # Previously validated native chunks remain available; never fabricate failed chunk timings.
            failed_chunk = chunk["number"]
            partial_rows, partial_speakers = [], {}
            partial_count = 0
            for chunk in plan["chunks"]:
                cache = chunk_path(out, chunk, "response.json")
                if not cache.exists() or read(cache).get("status") != "native_validated":
                    break
                current, metadata = native(read(cache)["entry"], chunk)
                partial_rows.extend(current)
                partial_speakers.update(metadata)
                partial_count += 1
            partial_rows.sort(key=lambda row: (row["start"], row["end"], row["speaker_id"]))
            for index, row in enumerate(partial_rows, 1):
                row["index"] = index
            publish(out, partial_rows, partial_speakers, partial=True)
            save(out / "manifest.json", {"status": "blocked", "error": api.sanitized(str(exc), key) if key else str(exc),
                "no_POST_retry": True, "actual_shared_USD": ledger.total, "blocked_chunk": failed_chunk,
                "completed_chunks": partial_count, "new_POSTs": posted, "max_requests": max_requests, "budget_usd": budget})
            raise ValueError("Episode stopped; inspect saved response/ledger and partial outputs; no paid retry") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-upload", action="store_true", help="Explicitly authorize chargeable audio-only uploads")
    parser.add_argument("--budget-ledger", type=Path, help="Existing active shared ledger; never a snapshot/new output budget")
    parser.add_argument("--budget", type=float, default=1)
    parser.add_argument("--max-requests", type=int, default=72, help="Explicit total shared cap; not automatically extended")
    args = parser.parse_args(argv)
    try:
        result = run_episode(args.input, args.output_dir, allow_upload=args.allow_upload,
                             budget_ledger=args.budget_ledger, budget=args.budget, max_requests=args.max_requests)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError, wave.Error) as exc:
        parser.exit(1, f"Episode launcher: {exc}\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
