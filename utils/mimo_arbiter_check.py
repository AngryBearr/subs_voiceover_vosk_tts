"""Offline preparation and bounded, opt-in OmniRoute arbiter comparison."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from utils.shortening_prompts import ARBITER_SYSTEM_PROMPT, build_review_prompt, parse_review


SOURCE = Path("output/openrouter_episode_20260930_clean")
DEST = Path("output/mimo_arbiter_check_20261001")
MODEL = "opencode-go/mimo-v2.6-pro"
BASE_URL = "http://127.0.0.1:20128/v1"
IDS = ("cue-295", "cue-11", "cue-400", "cue-1516", "cue-531", "cue-1519", "cue-432", "cue-818")
Transport = Callable[[str, dict[str, Any] | None, str, float], dict[str, Any]]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _ordered_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore prompt field order after canonical manifest JSON sorts keys."""
    return [{key: entry[key] for key in ("cue_id", "text", "is_target", "start", "end") if key in entry}
            for entry in context]


def prepare(source: Path = SOURCE, dest: Path = DEST) -> dict[str, Any]:
    """Freeze exact historical candidate/verdict pairs before any model request."""
    path = dest / "manifest.json"
    if path.exists() or (dest / "calls.jsonl").exists():
        raise FileExistsError("experiment already prepared or attempted; never overwrite it")
    outcomes = {row["cue_id"]: row for row in json.loads((source / "outcomes.json").read_text())["cues"]}
    records = {row["cue_id"]: row for row in json.loads((source / "report.json").read_text())["records"]}
    cases: list[dict[str, Any]] = []
    for cue_id in IDS:
        outcome, record = outcomes[cue_id], records[cue_id]
        expected = "accept" if outcome["outcome"] == "accepted_correction" else "revise"
        reviews = [entry for entry in record["history"] if entry.get("stage") in ("review", "recheck")
                   and entry.get("judgment", {}).get("verdict") == expected
                   and entry.get("candidate") != record["original"]]
        if not reviews:
            raise ValueError(f"missing matching historical review: {cue_id}")
        selected = next((entry for entry in reversed(reviews) if entry["candidate"] == record["candidate"]), reviews[-1])
        candidate = selected["candidate"]
        local = next(entry["local"] for entry in reversed(record["history"])
                     if entry.get("candidate") == candidate and isinstance(entry.get("local"), dict))
        if local["actual_chars"] > outcome["max_chars"] or not local["valid"]:
            raise ValueError(f"candidate invalid or over limit: {cue_id}")
        context = [
            {key: neighbor[key] for key in ("cue_id", "text", "start", "end") if key in neighbor}
            for neighbor in outcome["original_context_before"][-3:]
        ] + [{"cue_id": cue_id, "text": record["original"], "is_target": True}] + [
            {key: neighbor[key] for key in ("cue_id", "text", "start", "end") if key in neighbor}
            for neighbor in outcome["original_context_after"][:3]
        ]
        target = {"cue_id": cue_id, "original": record["original"], "candidate": candidate,
                  "max_chars": outcome["max_chars"], "actual_chars": local["actual_chars"],
                  "length_basis": outcome["length_basis"]}
        prompt = build_review_prompt(context, [target])
        cases.append({"cue_id": cue_id, "target": target, "context": context,
                      "input_sha256": _digest({"system": ARBITER_SYSTEM_PROMPT, "prompt": prompt}),
                      "sol": selected["judgment"], "historical_final_text": record["final_text"],
                      "historical_stage": selected["stage"]})
    manifest = {"source": str(source), "source_report_sha256": hashlib.sha256((source / "report.json").read_bytes()).hexdigest(),
                "provider": "omniroute", "remote_model_configured_not_live_verified": MODEL,
                "cases": cases, "comparison_note": "Sol judged these exact candidates; contexts and single-candidate mode can differ from its original batched review. Sol is not ground truth."}
    dest.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(manifest) + b"\n")
    return manifest


def http_transport(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=_json_bytes(body) if body is not None else None,
                                     headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("non-object provider response")
    return result


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


BARE_MODEL = "mimo-v2.6-pro"
IDENTITY_SOURCES = (
    "https://github.com/diegosouzapw/OmniRoute/blob/3c6f7177/open-sse/services/model.ts#L461-L469",
    "https://github.com/diegosouzapw/OmniRoute/blob/3c6f7177/open-sse/services/model.ts#L757-L771",
    "https://github.com/diegosouzapw/OmniRoute/blob/3c6f7177/open-sse/handlers/chatCore.ts#L4578-L4586",
)


def _identity(catalog: dict[str, Any], dest: Path, model: str) -> bool:
    """Verify this one pinned route against live catalog and documented routing."""
    entries = catalog.get("data")
    if not isinstance(entries, list):
        raise ValueError("model catalog is malformed")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("id") == model]
    if len(matches) != 1:
        raise ValueError("exact model ID is not uniquely present in catalog")
    entry = matches[0]
    verified = (model == MODEL and entry.get("owned_by") == "opencode-go"
                and entry.get("root") == BARE_MODEL and entry.get("name") == BARE_MODEL
                and not any(isinstance(other, dict) and other.get("id") == BARE_MODEL for other in entries))
    if verified:
        proof = {"requested": MODEL, "returned": BARE_MODEL, "catalog_entry":
                 {key: entry[key] for key in ("id", "owned_by", "root", "name")},
                 "source_urls": list(IDENTITY_SOURCES),
                 "scope": "This route's model-name mapping only; not proof of upstream account, fallback policy or price."}
        path = dest / "identity_verification.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != proof:
                raise ValueError("saved route identity proof differs from current catalog")
        else:
            with path.open("x", encoding="utf-8") as output:
                json.dump(proof, output, ensure_ascii=False, indent=2)
                output.write("\n")
    return verified


def _content(reply: dict[str, Any], model: str, *, alias_verified: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if reply.get("model") != model and not (alias_verified and model == MODEL and reply.get("model") == BARE_MODEL):
        raise ValueError("response model does not match requested remote ID")
    choices = reply.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected one response choice")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError("response did not finish normally")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("missing text response content")
    return message, reply.get("usage") if isinstance(reply.get("usage"), dict) else {}


def report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Escape every saved field; a missing response is never a model verdict."""
    esc = lambda value: html.escape(str(value), quote=True)
    by_id = {row["cue_id"]: row for row in rows if row.get("event") in {"start", "complete", "validation"}}
    result = ["<!doctype html><meta charset='utf-8'><title>MiMo arbiter check</title>",
              "<h1>Experimental single-candidate arbiter comparison</h1>",
              f"<p>Provider: {esc(manifest['provider'])}; configured remote ID: {esc(manifest['remote_model_configured_not_live_verified'])}. "
              "Mode: one candidate per request; max_tokens=4096; reasoning variant not set. "
              "Saved Sol decisions are a reference, not ground truth; candidate context/mode may differ. "
              "Historical final text is not a newly accepted subtitle. Unknown cost is not zero.</p>"]
    for case in manifest["cases"]:
        row = by_id.get(case["cue_id"], {})
        judgment = row.get("judgment")
        result += [f"<section><h2>{esc(case['cue_id'])}: {esc(row.get('status', 'not attempted'))}</h2>",
                   f"<p>Original: {esc(case['target']['original'])}</p><p>Candidate: {esc(case['target']['candidate'])}</p>",
                   f"<p>Limit: {esc(case['target']['max_chars'])}; measured: {esc(case['target']['actual_chars'])}; "
                   f"basis: {esc(case['target']['length_basis'])}</p>",
                   f"<p>Context: {esc(' | '.join(x['text'] for x in case['context']))}</p>",
                   f"<p>Sol ({esc(case['historical_stage'])}): {esc(case['sol']['verdict'])}; "
                   f"issues: {esc(case['sol']['issues'])}; advice: {esc(case['sol']['advice'])}</p>",
                   f"<p>MiMo: {esc(judgment['verdict'] if judgment else 'no valid verdict')}; "
                   f"issues: {esc(judgment['issues'] if judgment else row.get('error', ''))}; "
                   f"advice: {esc(judgment['advice'] if judgment else '')}</p>",
                   f"<p>Receipt note: {esc(row.get('note', 'none'))}</p>",
                   f"<p>Endpoint: {esc(row.get('base_url', 'not requested'))}; started: {esc(row.get('started_at', 'not started'))}; "
                   f"elapsed seconds: {esc(row.get('elapsed_sec', 'unknown'))}; returned model: {esc(row.get('returned_model', 'none'))}; "
                   f"tokens: {esc(row.get('usage', 'unknown'))}; "
                   f"cost USD: {esc(row.get('cost_usd') if row.get('cost_usd') is not None else 'unknown')}</p>",
                   f"<p>Historical accepted final text (old run): {esc(case['historical_final_text'])}</p></section>"]
    return "\n".join(result) + "\n"


def _saved_events(ledger: Path, manifest: dict[str, Any], base_url: str, model: str,
                  ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    starts: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cue_id, event = row.get("cue_id"), row.get("event")
        if event == "start":
            if len(starts) >= 8 or cue_id != IDS[len(starts)]:
                raise ValueError("unexpected or duplicate request start")
            case = manifest["cases"][len(starts)]
            if (row.get("input_sha256") != case["input_sha256"] or row.get("model") != model
                    or row.get("base_url") != base_url or row.get("max_tokens") != 4096):
                raise ValueError("saved request differs from frozen run")
            starts[cue_id] = row
        elif event == "complete":
            if cue_id not in starts or cue_id in completed or cue_id != list(starts)[-1]:
                raise ValueError("orphan or duplicate completion")
            completed[cue_id] = row
        elif event == "validation":
            if cue_id != IDS[0] or cue_id not in completed or cue_id in validations:
                raise ValueError("unexpected duplicate offline validation")
            validations[cue_id] = row
        else:
            raise ValueError("unknown ledger event")
        events.append(row)
    if not starts or any(cue_id not in completed for cue_id in starts):
        raise ValueError("unresolved request start; do not retry it")
    if any(cue_id != IDS[0] and completed[cue_id].get("status") != "ok" for cue_id in completed):
        raise ValueError("later failed request cannot be retried")
    return events, completed, validations


def run(manifest: dict[str, Any], dest: Path, base_url: str, model: str, key: str,
        transport: Transport = http_transport, *, resume: bool = False) -> list[dict[str, Any]]:
    if model != MODEL or model != manifest["remote_model_configured_not_live_verified"] or manifest.get("provider") != "omniroute":
        raise ValueError("model must match frozen configured remote ID")
    if not key:
        raise ValueError("OMNIROUTE_API_KEY is missing")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "https" and not (parsed_url.scheme == "http" and parsed_url.hostname in {"127.0.0.1", "localhost", "::1"}):
        raise ValueError("credential-bearing endpoint must use HTTPS or local loopback")
    ledger = dest / "calls.jsonl"
    if not resume and ledger.exists():
        raise FileExistsError("prior attempts recorded: no retry or duplicate run")
    if resume and not ledger.exists():
        raise FileNotFoundError("no previous experiment to resume")
    cases = manifest["cases"]
    if len(cases) != 8 or [case["cue_id"] for case in cases] != list(IDS):
        raise ValueError("frozen case IDs must contain exactly the eight selected cues")
    prompts: list[str] = []
    for case in cases:
        if case["target"]["cue_id"] != case["cue_id"]:
            raise ValueError("frozen target cue ID mismatch")
        prompt = build_review_prompt(_ordered_context(case["context"]), [case["target"]])
        if _digest({"system": ARBITER_SYSTEM_PROMPT, "prompt": prompt}) != case["input_sha256"]:
            raise ValueError(f"frozen input changed: {case['cue_id']}")
        prompts.append(prompt)
    alias_verified = False
    if not resume:
        catalog = transport(base_url.rstrip("/") + "/models", None, key, 15.0)
        alias_verified = _identity(catalog, dest, model)
    # Existing ledger uses an OS lock; new runs atomically create the same file.
    with ledger.open("r+" if resume else "x+", encoding="utf-8") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events, completed, validations = _saved_events(ledger, manifest, base_url, model) if resume else ([], {}, {})
        if len(completed) == 8:
            raise FileExistsError("all eight physical requests have already completed")
        # The live catalog plus pinned source mapping is needed for bare response IDs.
        if resume:
            discovered = transport(base_url.rstrip("/") + "/models", None, key, 15.0)
            alias_verified = _identity(discovered, dest, model)
        rows: list[dict[str, Any]] = []
        if resume:
            first = completed[IDS[0]]
            if first.get("status") == "error":
                if not alias_verified or first.get("returned_model") != BARE_MODEL or first.get("error") != "ValueError":
                    raise ValueError("first saved error is not the verified bare-model mismatch")
                reply = first.get("response")
                if not isinstance(reply, dict):
                    raise ValueError("first response is unavailable")
                text, _ = _content(reply, model, alias_verified=True)
                judgment = parse_review(text["content"], [IDS[0]])[IDS[0]]
                expected_validation = {**{k: first[k] for k in ("cue_id", "returned_model", "usage", "cost_usd", "elapsed_sec", "started_at", "base_url")},
                                       "event": "validation", "status": "ok", "judgment": judgment,
                                       "response_sha256": _digest(reply),
                                       "note": "Offline validation of existing response after pinned route identity verification; no extra POST."}
                if IDS[0] in validations:
                    if validations[IDS[0]] != expected_validation:
                        raise ValueError("prior validation differs from the saved raw response")
                else:
                    _append(ledger, expected_validation)
                    events.append(expected_validation)
                    validations[IDS[0]] = expected_validation
                rows.append(validations[IDS[0]])
            else:
                rows.append(first)
            rows.extend(completed[cue_id] for cue_id in IDS[1:len(completed)])
            if any(row.get("status") != "ok" for row in rows):
                raise ValueError("saved failed attempt blocks continuation")
        if resume:
            (dest / "report.html").write_text(report(manifest, events), encoding="utf-8")
        started = time.monotonic()
        for case, prompt in zip(cases[len(completed):], prompts[len(completed):]):
            if len(rows) >= 8 or time.monotonic() - started >= 1200:
                break
            _request_one(manifest, ledger, dest, events, rows, case, prompt, base_url, model, key,
                         alias_verified, transport)
            if rows[-1]["status"] != "ok":
                break
        return rows


def _request_one(manifest: dict[str, Any], ledger: Path, dest: Path,
                 events: list[dict[str, Any]], rows: list[dict[str, Any]], case: dict[str, Any],
                 prompt: str, base_url: str, model: str, key: str, alias_verified: bool,
                 transport: Transport) -> None:
        cue_id = case["cue_id"]
        started_at = datetime.now(timezone.utc).isoformat()
        start_clock = time.monotonic()
        start_row = {"event": "start", "cue_id": cue_id, "status": "outcome unknown (started)",
                     "input_sha256": case["input_sha256"], "provider": "omniroute", "model": model,
                     "base_url": base_url, "mode": "single_candidate", "max_tokens": 4096,
                     "reasoning_variant": None, "started_at": started_at}
        _append(ledger, start_row)
        events.append(start_row)
        (dest / "report.html").write_text(report(manifest, events), encoding="utf-8")
        request = {"model": model, "messages": [{"role": "system", "content": ARBITER_SYSTEM_PROMPT},
                                                  {"role": "user", "content": prompt}], "max_tokens": 4096}
        row: dict[str, Any] = {**start_row, "event": "complete", "status": "error", "cost_usd": None}
        try:
            reply = transport(base_url.rstrip("/") + "/chat/completions", request, key, 120.0)
            row["response"] = reply
            row["returned_model"] = reply.get("model")
            row["usage"] = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
            cost = row["usage"].get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                row["cost_usd"] = cost
            message, _ = _content(reply, model, alias_verified=alias_verified)
            row["judgment"] = parse_review(message["content"], [cue_id])[cue_id]
            row["status"] = "ok"
        except urllib.error.HTTPError as exc:
            row["error"] = f"HTTP {exc.code}"
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            row["error"] = type(exc).__name__
        except (ValueError, KeyError, TypeError) as exc:
            row["error"] = type(exc).__name__
        row["elapsed_sec"] = round(time.monotonic() - start_clock, 3)
        _append(ledger, row)
        rows.append(row)
        events.append(row)
        (dest / "report.html").write_text(report(manifest, events), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="freeze eight cases offline; no model call")
    mode.add_argument("--run", action="store_true", help="opt-in: authenticated model calls, maximum eight")
    mode.add_argument("--resume", action="store_true", help="validate saved first response offline; send only untouched cases")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-key-env", default="OMNIROUTE_API_KEY")
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare()
            print("Prepared eight historical cases offline; no model requests.")
            return 0
        manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
        rows = run(manifest, DEST, args.base_url, args.model, os.environ.get(args.api_key_env, ""), resume=args.resume)
    except urllib.error.HTTPError as exc:
        print(f"Experiment stopped: HTTP {exc.code}")
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Experiment stopped: {type(exc).__name__}")
        return 1
    valid = sum(row["status"] == "ok" for row in rows)
    accepted = sum(row.get("judgment", {}).get("verdict") == "accept" for row in rows)
    print(f"Completed {len(rows)} request attempts; valid responses: {valid}; accept verdicts: {accepted}")
    return 0 if len(rows) == 8 and valid == 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
