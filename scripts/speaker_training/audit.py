"""Read-only audit of saved speaker datasets; write conservative, new training manifests.

Run from the repository root: uv run -m scripts.speaker_training.audit --out output/<new-run>
Subtitle agreement verifies text/time, not the biological identity of the voice. Unsupported
diarizer-only labels and unrecognised shouts are deliberately not admitted to the clean set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from stt.subtitles import WordIndex, names_person, pin_pieces, read_pieces, tokens

RATE = 16000
POLICY = {
    "episode_named_match_min": 0.30,
    "row_match_min": 0.80,
    "row_precision_min": 0.70,
    "row_min_tokens": 3,
    "anchor_min_tokens": 5,
    "local_radius_s": 60.0,
    "local_match_min": 0.40,
    "local_min_pieces": 5,
    "max_anchor_distance_s": 60.0,
    "max_anchor_offset_difference_s": 2.0,
    "row_boundary_tolerance_s": 0.50,
    "stored_clip_rounding_tolerance_samples": 9,
    "pseudo": "excluded: no independent per-row person label",
    "noise": "excluded: original index has no time provenance",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path, hashed: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "resolved": str(path.resolve()), "exists": True,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            **({"sha256": sha256(path)} if hashed else {})}


def valid_span(start: float, end: float, duration: float) -> bool:
    return all(math.isfinite(x) for x in (start, end, duration)) and 0 <= start < end <= duration


def clip_matches(audio: np.ndarray, stored: np.ndarray, t0: float) -> bool:
    """Allow only the documented 1 ms rounding of t0, then compare every saved sample."""
    first = round(t0 * RATE)
    for offset in range(-9, 10):
        a = first + offset
        if 0 <= a and a + len(stored) <= len(audio):
            if np.array_equal(audio[a:a + min(64, len(stored))], stored[:64]):
                if np.array_equal(audio[a:a + len(stored)], stored):
                    return True
    return False


def overlap_conflicts(rows: list[dict[str, Any]], label_key: str) -> set[int]:
    """Flag both sides of incompatible labels on the same audible interval."""
    bad: set[int] = set()
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["start"])
    for pos, i in enumerate(ordered):
        a = rows[i]
        for j in ordered[pos + 1:]:
            b = rows[j]
            if b["start"] >= a["end"]:
                break
            if a[label_key] != b[label_key] and min(a["end"], b["end"]) - b["start"] > 0.1:
                bad.update((i, j))
    return bad


def discover() -> dict[str, dict[str, Any]]:
    """All prepared subtitle sources, plus the episode-specific experimental datasets."""
    root = Path("output/subs_train")
    episodes: dict[str, dict[str, Any]] = {}
    for key, info in load(root / "counts.json").items():
        season, name = key.split("/")
        code = name.split(".")[0]
        episodes[f"{season}/{code}"] = {"season": season, "ep": code, "folder": root / season / code,
                                        "srt": root / key, "media": info["media"]}
    for folder in sorted(Path("output/exp_boundary/s51/mf_x1").glob("*")):
        match = re.search(r"S51E\d+", folder.name)
        if folder.is_dir() and match:
            code = match[0]
            episodes[f"S51/{code}"].update(folder=folder,
                                            srt=Path("output/s51_subs") / f"{folder.name}.sdh.srt")
    for folder in sorted(Path("output/exp_spk/au10").glob("E*")):
        if not (folder / "source.json").is_file():
            continue
        media = Path(load(folder / "source.json")["media"])
        code = f"S10{folder.name}"
        srts = sorted((p for p in media.parent.glob(f"*{code}*.srt")
                       if "_synced" not in p.name and "ru" not in p.stem.lower().split(".")),
                      key=lambda p: -p.stat().st_size)
        episodes[f"AU10/{folder.name}"] = {"season": "AU10", "ep": folder.name, "folder": folder,
                                            "srt": srts[0] if srts else folder / "MISSING.srt",
                                            "media": str(media)}
    return episodes


def episode_audit(info: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder, srt = info["folder"], info["srt"]
    asr_path = next((folder / n for n in ("asr.whisper-large-v3.json", "asr.whisper-large-v3-turbo.json", "asr.json")
                     if (folder / n).is_file()), folder / "MISSING.asr.json")
    paths = [folder / "audio.wav", asr_path, srt, folder / "source.json", folder / "lines.json"]
    report: dict[str, Any] = {"season": info["season"], "ep": info["ep"], "folder": str(folder),
                              "files": [metadata(p, p.name != "audio.wav") for p in paths],
                              "media": metadata(Path(info["media"]), False), "reasons": []}
    if not all(p.is_file() for p in paths[:3]):
        report.update(status="unverified", reasons=["missing_audio_asr_or_subtitles"])
        return report, []
    with wave.open(str(folder / "audio.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (RATE, 1, 2)
        duration = wav.getnframes() / RATE
    report["duration_s"] = duration
    source = load(folder / "source.json") if (folder / "source.json").is_file() else {}
    if not source or not report["media"]["exists"]:
        report["reasons"].append("missing_source_media_or_metadata")
    elif str(Path(source.get("media", "")).resolve()) != str(Path(info["media"]).resolve()):
        report["reasons"].append("source_path_disagreement")
    elif source.get("size") != report["media"]["size"] or source.get("start") != 0.0:
        report["reasons"].append("source_metadata_disagreement")
    asr = load(asr_path)
    report["asr"] = {k: asr.get(k) for k in ("model", "model_id", "hotwords", "transcribe_s")}
    raw_words = [w for seg in asr["segments"] for w in seg.get("words") or []]
    bad_words = sum(not valid_span(float(w["start"]), float(w["end"]), duration) for w in raw_words)
    report["invalid_asr_words"] = bad_words
    words = sorted((SimpleNamespace(start=float(w["start"]), end=float(w["end"]), text=w["text"],
                                    source=w.get("source")) for w in raw_words
                    if valid_span(float(w["start"]), float(w["end"]), duration) and w.get("source") != "ocr"),
                   key=lambda w: w.start)
    pieces = read_pieces(srt)
    pinned, offset = pin_pieces(pieces, words)
    hits = {id(p.piece): p for p in pinned}
    index = WordIndex(words)
    rows: list[dict[str, Any]] = []
    for i, piece in enumerate(pieces):
        hit = hits.get(id(piece))
        row: dict[str, Any] = {"piece_id": i, "person": piece.person, "named": names_person(piece.person),
                               "text": piece.text, "tokens": len(piece.tokens),
                               "cue_start": piece.cue_start, "cue_end": piece.cue_end,
                               "start": hit.start if hit else piece.cue_start,
                               "end": hit.end if hit else piece.cue_end,
                               "match": hit.match if hit else 0.0, "precision": 0.0,
                               "reasons": []}
        if hit:
            a, b = index.between(hit.start - .001, hit.end + .001)
            matches = sum(x.size for x in SequenceMatcher(None, piece.tokens, index.toks[a:b], autojunk=False).get_matching_blocks())
            row.update(precision=matches / max(b - a, 1), audio_text=" ".join(w.text for w in words[hit.first:hit.last + 1]),
                       offset_s=round(hit.start - piece.cue_start, 3),
                       words=[[w.start, w.end] for w in words[hit.first:hit.last + 1]])
        if not hit or row["match"] < POLICY["row_match_min"] or row["precision"] < POLICY["row_precision_min"]:
            row["reasons"].append("low_lexical_match")
        if len(piece.tokens) < POLICY["row_min_tokens"]:
            row["reasons"].append("short_ambiguous_text")
        if not valid_span(row["start"], row["end"], duration):
            row["reasons"].append("invalid_or_out_of_range_time")
        rows.append(row)
    rows.sort(key=lambda r: r["cue_start"])
    anchors = [r for r in rows if not r["reasons"] and r["tokens"] >= POLICY["anchor_min_tokens"]]
    anchor_times = [r["cue_start"] for r in anchors]
    cue_times = [r["cue_start"] for r in rows]
    matched = [r["match"] >= POLICY["row_match_min"] and r["precision"] >= POLICY["row_precision_min"] for r in rows]
    prefix = np.r_[0, np.cumsum(matched)]
    named = [r for r in rows if r["named"]]
    ratio = sum(r["match"] >= .6 for r in named) / max(len(named), 1)
    if len(named) >= 20 and ratio < POLICY["episode_named_match_min"]:
        report["reasons"].append("episode_low_named_match")
    conflicts = overlap_conflicts([r for r in rows if r["named"]], "person")
    for i, row in enumerate(named):
        if i in conflicts:
            row["reasons"].append("conflicting_subtitle_labels")
    for row in rows:
        t = row["cue_start"]
        a, b = bisect_left(cue_times, t - 60), bisect_right(cue_times, t + 60)
        local = float(prefix[b] - prefix[a]) / max(b - a, 1)
        row.update(local_match=round(local, 4), local_pieces=b - a)
        if b - a < POLICY["local_min_pieces"] or local < POLICY["local_match_min"]:
            row["reasons"].append("low_local_coverage")
        # Do not count a row as its own corroborating anchor. Require another on each side.
        left, right = bisect_left(anchor_times, t), bisect_right(anchor_times, t)
        before = anchors[left - 1] if left else None
        after = anchors[right] if right < len(anchors) else None
        if before is None or after is None or max(t - before["cue_start"], after["cue_start"] - t) > 60:
            row["reasons"].append("unbracketed_or_extrapolated")
        elif abs(before["offset_s"] - after["offset_s"]) > 2:
            row["reasons"].append("local_cut_or_changing_offset")
        else:
            row["anchor_offsets"] = [before["offset_s"], after["offset_s"]]
            if "offset_s" in row and abs(row["offset_s"] - (before["offset_s"] + after["offset_s"]) / 2) > 2:
                row["reasons"].append("row_offset_disagrees_with_neighbors")
        row["reasons"] += report["reasons"]
        row["status"] = "excluded" if row["reasons"] else "text_time_verified"
    offsets = [r["offset_s"] for r in anchors]
    report.update(status="excluded" if report["reasons"] else "audited", pieces=len(rows), named=len(named),
                  named_pinned=sum(r["match"] >= .6 for r in named), named_match=round(ratio, 4),
                  offset_s=offset, anchors=len(anchors),
                  anchor_offset_quantiles=np.quantile(offsets, [0, .1, .5, .9, 1]).tolist() if offsets else [],
                  offset_type=("unanchored" if not offsets else "changing_or_cut" if np.ptp(offsets) > 4
                               else "constant_offset" if abs(float(np.median(offsets))) > 1 else "near_sync"),
                  piece_statuses=dict(Counter(r["status"] for r in rows)),
                  piece_exclusions=dict(Counter(reason for r in rows for reason in r["reasons"])))
    if (folder / "lines.json").is_file():
        old = load(folder / "lines.json")
        report["saved_alignment"] = {"stats": old.get("stats"), "offset_s": old.get("offset_s"),
                                      "older_than_inputs": any((folder / "lines.json").stat().st_mtime_ns < p.stat().st_mtime_ns
                                                               for p in (srt, asr_path)),
                                      "predates_current_guard": not old.get("stats", {}).get("out_of_sync", False)
                                      and "episode_low_named_match" in report["reasons"]}
    return report, rows


def episode_key(seg: dict[str, Any]) -> str:
    season, ep = seg["season"], seg["ep"]
    return f"{season}/{season + ep if season == 'S51' and ep.startswith('E') else ep}"


def row_match(seg: dict[str, Any], person: str, pieces: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [p for p in pieces if p["named"] and p["person"] == person
                  and abs(p["start"] - seg["start"]) <= .5 and abs(p["end"] - seg["end"]) <= .5]
    return min(candidates, key=lambda p: abs(p["start"] - seg["start"]) + abs(p["end"] - seg["end"])) if candidates else None


def saved_row_reasons(saved: dict[str, Any], match: dict[str, Any] | None) -> list[str]:
    """A newly matching phrase must also be the phrase in the old, actually trained row."""
    reasons = list(match["reasons"]) if match else ["no_matching_subtitle_row_at_saved_time"]
    if match and tokens(saved.get("text", "")) != tokens(match["text"]):
        reasons.append("saved_text_differs_from_subtitle")
    if saved.get("other_voice_s", 0) > .1:
        reasons.append("competing_voice_in_saved_row")
    return sorted(set(reasons))


def identity_aliases(genders: dict[str, Any]) -> dict[str, list[str]]:
    """Find class collisions in the supplied person metadata; do not guess an alias resolution."""
    groups: dict[str, list[str]] = defaultdict(list)
    for label, info in genders.items():
        if ":" in label and isinstance(info, dict) and info.get("who"):
            groups[label.split(":", 1)[0] + ":" + " ".join(tokens(info["who"]))].append(label)
    return {person: labels for person, labels in groups.items() if len(labels) > 1}


def unidentified_classes(people: dict[str, Any]) -> set[str]:
    """An occupational subtitle label such as DRIVER need not denote one recurring person."""
    return {label for label, info in people.items() if isinstance(info, dict)
            and (not info.get("who") or "unknown" in tokens(info["who"]))}


def finalize(out: Path) -> None:
    """Intersect saved-row provenance, full audit, fresh audio strata and training eligibility."""
    summary = load(out / "summary.json")
    if summary.get("finalized"):
        raise FileExistsError("This audit has already been finalized")
    checks = load(out / "audio_checks.json")
    source_pcm = load(out / "source_pcm_checks.json")
    if not source_pcm or any(r["status"] != "verified" for r in source_pcm):
        raise ValueError("Unresolved stratified source-media PCM correspondence")
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in checks["clips"]:
        by_ep[clip["episode"]].append(clip)
    saved_by_ep: dict[str, list[dict[str, Any]]] = {}
    saved_reports: dict[str, Any] = {}
    for key, info in sorted(discover().items()):
        pieces = load(out / "pieces" / f"{key.replace('/', '_')}.json")
        old_path = info["folder"] / "lines.json"
        old = load(old_path) if old_path.is_file() else {}
        records = []
        for kind, lines in (("gold", old.get("lines", [])), ("cue", old.get("cue_lines", []))):
            for i, saved in enumerate(lines):
                match = row_match(saved, saved["person"], pieces)
                records.append({"kind": kind, "row_id": i, "person": saved["person"],
                                "start": saved["start"], "end": saved["end"], "text": saved["text"],
                                "reasons": saved_row_reasons(saved, match),
                                "piece_id": match["piece_id"] if match else None})
        saved_by_ep[key] = records
        write(out / "saved_alignment_rows" / f"{key.replace('/', '_')}.json", records)
        if info["season"] == "S46":
            approved = {r["row_id"] for r in records if r["kind"] == "gold" and not r["reasons"]}
            if len(by_ep[key]) != 3 or any(r["status"] != "verified" for r in by_ep[key]):
                approved = set()
            write(out / "dev_lines" / f"{info['ep']}.json",
                  {"lines": [r for i, r in enumerate(old.get("lines", [])) if i in approved],
                   "source": metadata(old_path), "role": "development only; not an untouched final test"})
        # Per-region evidence makes partial cut mismatches visible even with a good episode average.
        bins = []
        for bin_id in sorted({int(p["cue_start"] // 120) for p in pieces}):
            local = [p for p in pieces if int(p["cue_start"] // 120) == bin_id]
            offsets = [p["offset_s"] for p in local if p["match"] >= .8 and p["precision"] >= .7 and p["tokens"] >= 5]
            bins.append({"cue_start": bin_id * 120, "cue_end": (bin_id + 1) * 120,
                         "pieces": len(local), "matched": sum(p["match"] >= .8 and p["precision"] >= .7 for p in local),
                         "anchors": len(offsets), "median_offset_s": float(np.median(offsets)) if offsets else None})
        write(out / "regions" / f"{key.replace('/', '_')}.json", bins)
        saved_reports[key] = {"rows": len(records), "kinds": dict(Counter(r["kind"] for r in records)),
                              "exclusions": dict(Counter(reason for r in records for reason in r["reasons"])),
                              "all_fresh_strata_verified": len(by_ep[key]) == 3 and all(r["status"] == "verified" for r in by_ep[key])}
    original = load(Path("output/exp_spk/data_p2/index.json"))
    assert sha256(Path("output/exp_spk/data_p2/index.json")) == summary["datasets"]["data_p2"]["manifest"]["sha256"]
    assert sha256(Path("output/exp_spk/data_p2/audio.i16")) == summary["datasets"]["data_p2"]["audio"]["sha256"]
    genders = load(Path("output/exp_gender/people_gender.json"))
    aliases = identity_aliases(genders)
    ambiguous_classes = {label for labels in aliases.values() for label in labels}
    unidentified = unidentified_classes(genders)
    write(out / "identity_mapping.json", {"classes": original["classes"], "supplied_person_metadata": genders,
                                          "same_season_alias_collisions": aliases,
                                          "unidentified_classes": sorted(unidentified),
                                          "policy": "Exclude colliding classes; metadata is inferred, so do not silently merge identities.",
                                          "dev_cast_overlap": {"S50:CHARLIE": "S46 Charlie Davis", "S50:TIFFANY": "S46 Tiffany Ervin"},
                                          "dev_note": "S46 has no training episodes, but is not an entirely unseen cast; Jeff Probst is shared too."})
    statuses = load(out / "rows/data_p2.json")
    selected: list[dict[str, Any]] = []
    for status in statuses:
        seg = original["segments"][status["row_id"]]
        key = status["episode"]
        saved = [r for r in saved_by_ep.get(key, []) if r["kind"] == seg["kind"]
                 and r["person"] == status["label"].split(":", 1)[-1]
                 and abs(r["start"] - seg["start"]) < .001 and abs(r["end"] - seg["end"]) < .001]
        if seg["kind"] in {"gold", "cue"}:
            if len(saved) != 1:
                status["reasons"].append("missing_or_ambiguous_saved_row_provenance")
            else:
                status["reasons"] += saved[0]["reasons"]
        if not saved_reports.get(key, {}).get("all_fresh_strata_verified"):
            status["reasons"].append("fresh_audio_stratum_unverified")
        if status["label"] in ambiguous_classes:
            status["reasons"].append("unresolved_same_person_multiple_classes")
        if status["label"] in unidentified:
            status["reasons"].append("unidentified_person_not_one_verified_identity")
        if seg.get("words") and any(not valid_span(a, b, seg["end"]) or a < seg["start"] for a, b in seg["words"]):
            status["reasons"].append("invalid_saved_word_time")
        status["reasons"] = sorted(set(status["reasons"]))
        status["status"] = "excluded" if status["reasons"] else "included"
        if not status["reasons"]:
            selected.append({**seg, "audit_row_id": status["row_id"]})
    gold_counts = Counter(s["cls"] for s in selected if s["kind"] == "gold")
    insufficient = {s["audit_row_id"] for s in selected if gold_counts[s["cls"]] < 2}
    selected = [s for s in selected if s["audit_row_id"] not in insufficient]
    for status in statuses:
        if status["row_id"] in insufficient:
            status.update(status="excluded", reasons=status["reasons"] + ["class_has_fewer_than_two_verified_gold_rows"])
    write(out / "rows/data_p2.final.json", statuses)
    names = {v: k for k, v in original["classes"].items()}
    gender_rows = []
    for seg in selected:
        gender = genders.get(names[seg["cls"]])
        gender = gender if isinstance(gender, str) else (gender or {}).get("gender")
        if gender in {"male", "female"} and seg["end"] - seg["start"] >= .3:
            gender_rows.append(seg)
    for name, rows in (("clean_speaker", selected), ("clean_gender", gender_rows)):
        path = out / name / "index.json"
        initial = load(path)
        write(path.with_name("index.provisional.json"), initial)
        write(path, {**initial, "segments": rows,
                     "audit": {**initial["audit"], "finalized": True, "fresh_audio_checks": metadata(out / "audio_checks.json")}})
    summary.update(finalized=True, saved_alignment_audit=saved_reports,
                   clean_speaker_rows=len(selected), clean_gender_rows=len(gender_rows),
                   clean_kinds=dict(Counter(s["kind"] for s in selected)),
                   clean_hours=sum(s["end"] - s["start"] for s in selected) / 3600,
                   clean_episodes=len({s["ep"] for s in selected}),
                   clean_seasons=dict(Counter(s["season"] for s in selected)),
                   clean_classes=len({s["cls"] for s in selected}),
                   class_gold_counts=dict(Counter(names[s["cls"]] for s in selected if s["kind"] == "gold")),
                   final_exclusions=dict(Counter(reason for r in statuses for reason in r["reasons"])),
                   final_exclusive_primary_reasons=dict(Counter(r["reasons"][0] if r["reasons"] else "included" for r in statuses)),
                   fresh_audio={"clips": len(checks["clips"]), "statuses": dict(Counter(r["status"] for r in checks["clips"]))})
    write(out / "summary.json", summary)
    for manifest in sorted(Path("output/exp_spk").glob("*/index.json")):
        write(out / "noise_rows" / f"{manifest.parent.name}.json",
              [{"row_id": i, **row, "status": "excluded", "reasons": ["noise_has_no_episode_time_provenance"]}
               for i, row in enumerate(load(manifest)["noise"])])
    write(out / "verification.json", {"finalized": True, "training_started": False,
                                       "included_rows_have_no_exclusion_reasons": all(not r["reasons"] for r in statuses if r["status"] == "included"),
                                       "speaker_manifest": metadata(out / "clean_speaker/index.json"),
                                       "gender_manifest": metadata(out / "clean_gender/index.json"),
                                       "all_saved_alignment_rows": sum(r["rows"] for r in saved_reports.values()),
                                       "script": metadata(Path(__file__))})
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("clean_")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true", help="intersect completed audit with fresh audio checks and saved-row provenance")
    args = parser.parse_args()
    out: Path = args.out
    if args.finalize:
        finalize(out)
        return
    if (out / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite finished audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "policy.json", POLICY)
    episodes = discover()
    reports: dict[str, dict[str, Any]] = {}
    pieces_by_ep: dict[str, list[dict[str, Any]]] = {}
    for key, info in sorted(episodes.items()):
        report, pieces = episode_audit(info)
        reports[key], pieces_by_ep[key] = report, pieces
        write(out / "episodes" / f"{key.replace('/', '_')}.json", report)
        write(out / "pieces" / f"{key.replace('/', '_')}.json", pieces)
        print(key, report["status"], report.get("named_match"), flush=True)
    manifests = sorted(Path("output/exp_spk").glob("*/index.json"))
    datasets: dict[str, Any] = {}
    cleaned: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    overlap: dict[str, set[str]] = defaultdict(set)
    genders = load(Path("output/exp_gender/people_gender.json"))
    for manifest in manifests:
        data = load(manifest)
        source_audio = manifest.parent / "audio.i16"
        audio = np.memmap(source_audio, dtype=np.int16, mode="r")
        names = {v: k for k, v in data["classes"].items()}
        rows: list[dict[str, Any]] = []
        by_ep: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for i, seg in enumerate(data["segments"]):
            by_ep[episode_key(seg)].append((i, seg))
        for key, items in sorted(by_ep.items()):
            report = reports.get(key)
            folder = episodes[key]["folder"] if key in episodes else None
            wav_audio: np.ndarray | None = None
            if folder and (folder / "audio.wav").is_file():
                with wave.open(str(folder / "audio.wav"), "rb") as wav:
                    wav_audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            conflicts = overlap_conflicts([s for _, s in items], "cls")
            seen: set[tuple[Any, ...]] = set()
            for j, (i, seg) in enumerate(items):
                label = names[seg["cls"]]
                person = label.split(":", 1)[-1]
                reasons: list[str] = []
                match = row_match(seg, person, pieces_by_ep.get(key, []))
                if seg["kind"] == "pseudo" or "names_f" in manifest.parent.name:
                    reasons.append("unsupported_pseudo_identity")
                if not report or report["status"] == "unverified":
                    reasons.append("missing_source_evidence")
                else:
                    reasons += report["reasons"]
                if not match:
                    reasons.append("no_matching_subtitle_row_at_saved_time")
                else:
                    reasons += match["reasons"]
                if not valid_span(seg["start"], seg["end"], report.get("duration_s", 0) if report else 0):
                    reasons.append("invalid_or_out_of_range_time")
                bounds_ok = (isinstance(seg["off"], int) and isinstance(seg["n"], int) and seg["off"] >= 0
                             and seg["n"] > 0 and seg["off"] + seg["n"] <= len(audio))
                if not bounds_ok:
                    reasons.append("invalid_packed_audio_bounds")
                elif wav_audio is None or not clip_matches(wav_audio, audio[seg["off"]:seg["off"] + seg["n"]], seg["t0"]):
                    reasons.append("packed_audio_differs_from_episode")
                if j in conflicts:
                    reasons.append("conflicting_training_labels")
                identity = (key, seg["start"], seg["end"], label)
                if identity in seen:
                    reasons.append("duplicate_training_row")
                seen.add(identity)
                if label == "_HOST" and not re.fullmatch(r"S\d+", seg["season"]):
                    reasons.append("host_class_merges_different_people")
                if seg["season"] in {"S46", "S51", "AU10"}:
                    reasons.append("reserved_dev_or_previously_evaluated_season")
                reasons = sorted(set(reasons))
                row = {"manifest": str(manifest), "row_id": i, "episode": key, "label": label,
                       "kind": seg["kind"], "start": seg["start"], "end": seg["end"],
                       "status": "excluded" if reasons else "included", "reasons": reasons,
                       "text_evidence": {k: match[k] for k in ("piece_id", "text", "audio_text", "match", "precision", "local_match")
                                         if match and k in match}}
                rows.append(row)
                if manifest.parent.name == "data_p2" and not reasons:
                    cleaned.append({**seg, "audit_row_id": i})
            overlap[key].add(manifest.parent.name)
        write(out / "rows" / f"{manifest.parent.name}.json", sorted(rows, key=lambda r: r["row_id"]))
        all_rows += rows
        datasets[manifest.parent.name] = {
            "manifest": metadata(manifest), "audio": metadata(source_audio), "args": data.get("args"),
            "rows": len(rows), "hours": sum(s["end"] - s["start"] for s in data["segments"]) / 3600,
            "episodes": len(by_ep), "classes": len(names), "seasons": dict(Counter(s["season"] for s in data["segments"])),
            "kinds": dict(Counter(s["kind"] for s in data["segments"])),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "exclusions": dict(Counter(reason for r in rows for reason in r["reasons"])),
            "noise_rows_excluded": len(data["noise"]),
        }
        print(manifest, datasets[manifest.parent.name]["statuses"], flush=True)
    original = load(Path("output/exp_spk/data_p2/index.json"))
    names = {v: k for k, v in original["classes"].items()}
    gender_cleaned: list[dict[str, Any]] = []
    for seg in cleaned:
        gender = genders.get(names[seg["cls"]])
        gender = gender if isinstance(gender, str) else (gender or {}).get("gender")
        if seg["kind"] in {"gold", "cue"} and gender in {"male", "female"}:
            gender_cleaned.append(seg)
    # Retain the original numeric class IDs and packed offsets; make an immutable audio link.
    # Empty noise disables augmentations whose old rows have no episode time coordinates.
    for name, selected in (("clean_speaker", cleaned), ("clean_gender", gender_cleaned)):
        write(out / name / "index.json", {**original, "segments": selected, "noise": [], "stats": {},
                                          "audit": {"policy": str(out / "policy.json"), "source": datasets["data_p2"]["manifest"],
                                                    "audio_sha256": datasets["data_p2"]["audio"]["sha256"]}})
        (out / name / "audio.i16").symlink_to(Path("output/exp_spk/data_p2/audio.i16").resolve())
    lineage = []
    for recipe in sorted(Path("output/exp_spk").glob("*/args.json")):
        recipe_data = load(recipe)
        lineage.append({"recipe": metadata(recipe), "args": recipe_data,
                        "data_exists": (Path(recipe_data["data"]) / "index.json").is_file(),
                        "checkpoints": [metadata(p) for p in sorted(recipe.parent.glob("*/avg_model.pt"))]})
    write(out / "lineage.json", lineage)
    write(out / "datasets.json", datasets)
    write(out / "shared_episodes.json", {k: sorted(v) for k, v in overlap.items() if len(v) > 1})
    names = {v: k for k, v in original["classes"].items()}
    summary = {"policy": POLICY, "episodes": len(episodes),
               "episode_statuses": dict(Counter(r["status"] for r in reports.values())),
               "episode_exclusions": {k: r["reasons"] for k, r in reports.items() if r["reasons"]},
               "all_saved_rows": len(all_rows), "datasets": datasets,
               "clean_speaker_rows": len(cleaned), "clean_gender_rows": len(gender_cleaned),
               "clean_kinds": dict(Counter(s["kind"] for s in cleaned)),
               "clean_hours": sum(s["end"] - s["start"] for s in cleaned) / 3600,
               "clean_episodes": len({s["ep"] for s in cleaned}),
               "clean_seasons": dict(Counter(s["season"] for s in cleaned)),
               "clean_classes": len({s["cls"] for s in cleaned}),
               "class_gold_counts": dict(Counter(names[s["cls"]] for s in cleaned if s["kind"] == "gold")),
               "limitations": ["ASR cache provenance is metadata-based, additionally checked by new stratified audio recognition separately.",
                               "Text/time agreement does not independently prove the named person's identity.",
                               "No manual hearing of the entire corpus; uncertain rows excluded.",
                               "data_p1 is missing: r1/r2 training membership cannot be fully reconstructed.",
                               "Gender person labels include language-model-inferred metadata, not independently verified gender."]}
    write(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("clean_")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
