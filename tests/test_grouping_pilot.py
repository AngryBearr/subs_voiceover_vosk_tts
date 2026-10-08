"""Textual boundaries are review heuristics, never automatic utterance approval."""

from __future__ import annotations

import wave
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from utils.alignment_pilot import Cue, allocate, assemble
from utils.grouping_pilot import candidate, classify_timing, ending, joined_timing, measure_joined, scan
from utils.vosk_calibration import max_chars_for_window


def cue(position: int, text: str, *, status: str = "manual", current: str | None = None) -> Cue:
    return Cue(f"cue-{position}", position, position + 1, position * 2500,
               position * 2500 + 2300, current or text, text, (current or text,), status, 2300)


def calibration() -> tuple[dict, dict]:
    identity = {"engine": "vosk", "voice": 8, "rate": 1.5, "model_config_sha256": "config",
                "model_fingerprint": "model", "normalizer": "vosk_punctuation_v1_no_ruaccent",
                "normalizer_sha256": "normalizer"}
    profile = {"identity": identity, "status": "ok", "rate": 1.5,
               "normalizer": identity["normalizer"], "input_character_basis": "normalized_tts_text",
               "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.5, 0.06]}},
               "char_linear_guard_sec": 0.6}
    return profile, identity


def test_continuation_is_closed_heuristic_not_automatic_speaker_approval() -> None:
    cues = [cue(0, "Я ухожу, потому что жду,"), cue(1, "когда их союз развалится.")]
    row = candidate(cues, 0, 2)
    assert row["source_state"] == "source_closed_heuristic"
    assert row["original_joined_text"] == "Я ухожу, потому что жду, когда их союз развалится."
    assert row["review_required"] and "speaker_unknown" in row["review_reasons"]
    assert row["timing_evidence"] == "predicted_not_measured"
    assert row["measurement_texts"] == [c.original_text for c in cues]


def test_actual_cue_1519_style_ellipsis_and_lowercase_start_are_ambiguous() -> None:
    cues = [cue(0, "я ухожу, потому что действительно жду..."),
            cue(1, "с нетерпением когда их союз развалится.", status="accepted", current="жду краха их союза")]
    row = candidate(cues, 0, 2)
    assert row["source_state"] == "ambiguous_or_incomplete"
    assert {"ellipsis_ambiguous", "open_start_boundary"} <= set(row["reason_codes"])
    assert row["eligibility"] == "not_selected_for_audio"
    assert row["original_joined_text"] != row["current_joined_text"]
    assert row["measurement_text_policy"] == "all_originals_due_to_boundary_conflict"
    assert classify_timing(row, {"proposal": None}, [1, 1]) == "ambiguous_no_auto_audio_or_export"


@pytest.mark.parametrize("text", ['Сказал «хватит.»', 'Сказал "хватит!"', 'Сказал (хватит.)', "Сказал «жду…»"])
def test_quote_and_parenthesis_punctuation(text: str) -> None:
    assert ending(text) == ("ellipsis" if "…" in text else "terminal")
    row = candidate([cue(0, text), cue(1, "следующее предложение.")], 0, 2)
    assert row["eligibility"] == "not_selected_for_audio"


@pytest.mark.parametrize("text", ["Да.", "Нет.", "Ага.", "Ок.", "Да, конечно.", "Почему?"])
def test_separate_replies_and_questions_not_joined(text: str) -> None:
    row = candidate([cue(0, "Я думаю,"), cue(1, text)], 0, 2)
    assert row["eligibility"] == "not_selected_for_audio"
    assert "short_reply_boundary" in row["reason_codes"] or "question_requires_turn_review" in row["reason_codes"]


def test_complete_sentence_is_not_a_continuation() -> None:
    row = candidate([cue(0, "Мы ушли."), cue(1, "они остались.")], 0, 2)
    assert row["source_state"] == "not_safe" and "terminal_crossing" in row["reason_codes"]


def test_sentence_boundary_inside_cue_does_not_make_one_complete_utterance() -> None:
    row = candidate([cue(0, "Мы ушли. А если вернёмся,"), cue(1, "поможем всем.")], 0, 2)
    assert "internal_terminal_crossing" in row["reason_codes"]
    assert row["eligibility"] == "not_selected_for_audio"
    assert ending('Сказал «хватит.» )') == "terminal"


def test_internal_ellipsis_is_not_a_confident_continuation() -> None:
    row = candidate([cue(0, "Мы... решили,"), cue(1, "что вернёмся.")], 0, 2)
    assert "ellipsis_ambiguous" in row["reason_codes"]
    assert row["source_state"] == "ambiguous_or_incomplete"


def test_four_fragments_cannot_be_presented_as_complete_three_cue_phrase() -> None:
    cues = [cue(0, "Мы решили,"), cue(1, "что после игры,"),
            cue(2, "когда вернёмся,"), cue(3, "поможем всем.")]
    prefix, suffix = candidate(cues, 0, 3), candidate(cues, 1, 3)
    assert "utterance_exceeds_three_cues" in prefix["reason_codes"]
    assert "open_end_boundary" in prefix["reason_codes"]
    assert "utterance_exceeds_three_cues" in suffix["reason_codes"]
    assert "open_start_boundary" in suffix["reason_codes"]
    assert prefix["eligibility"] == suffix["eligibility"] == "not_selected_for_audio"


def test_full_triple_preferred_over_incomplete_pairs_and_selection_disjoint() -> None:
    profile, identity = calibration()
    cues = [cue(0, "Мы решили,"), cue(1, "что после игры,"), cue(2, "поможем всем."),
            cue(3, "Я знаю,"), cue(4, "что нужно сделать.")]
    result = scan(cues, profile, identity)
    assert result["disjoint_group_ids"] == ["utterance-0-2", "utterance-3-4"]
    assert result["pilot_group_ids"] == result["disjoint_group_ids"]
    selected = [r for r in result["candidates"] if r["group_id"] in result["disjoint_group_ids"]]
    positions = [p for row in selected for p in row["positions"]]
    assert len(positions) == len(set(positions))
    for row in selected:
        prediction = row["prediction"]
        assert prediction["max_chars_normalized_single_unit"] == max_chars_for_window(
            profile, prediction["single_utterance_speech_window_ms"], expected_identity=identity)
    assert all(r["timing_evidence"] == "predicted_not_measured" for r in result["candidates"])


@pytest.mark.parametrize("kind,reason", [("speaker", "speaker_boundary"),
    ("dialogue", "explicit_dialogue_marker"), ("gap", "large_gap_boundary"),
    ("overlap", "source_overlap"), ("external", "overlap_with_previous_external_cue")])
def test_actor_and_temporal_blockers(kind: str, reason: str) -> None:
    cues = [cue(0, "Закончено."), cue(1, "Мы знаем,"), cue(2, "что нужно сделать.")]
    if kind == "speaker":
        cues[1], cues[2] = replace(cues[1], speaker="A"), replace(cues[2], speaker="B")
    elif kind == "dialogue":
        cues[1] = replace(cues[1], dialogue_marker=True)
    elif kind == "gap":
        cues[2] = replace(cues[2], start=cues[1].end + 501)
    elif kind == "overlap":
        cues[2] = replace(cues[2], start=cues[1].end - 1)
    else:
        cues[0] = replace(cues[0], end=cues[1].start + 1)
    row = candidate(cues, 1, 2)
    assert reason in row["reason_codes"] and row["source_state"] == "not_safe"


def test_changed_neighbor_boundaries_use_originals_and_keep_both_versions() -> None:
    cues = [cue(0, "Я хочу знать,", status="accepted", current="Я хочу знать."),
            cue(1, "что произошло.")]
    row = candidate(cues, 0, 2)
    assert row["source_state"] == "source_closed_heuristic"
    assert row["changed_neighbor_ids"] == row["current_text_conflict_ids"] == ["cue-0"]
    assert row["measurement_texts"] == [c.original_text for c in cues]
    assert row["current_joined_text"] == "Я хочу знать. что произошло."
    assert "changed_neighbors_joint_context_unreviewed" in row["review_reasons"]


def test_real_pcm_padding_preserves_original_text_and_silence(tmp_path: Path) -> None:
    cues = [replace(cue(0, "Я думаю,"), start=0, end=300),
            replace(cue(1, "что мы поможем."), start=500, end=1200)]
    row = candidate(cues, 0, 2)
    before = list(cues)
    allocation = allocate(cues, [400, 500])
    assert classify_timing(row, allocation, [400, 500]) == "timing_only_feasible_inspect"
    paths = [tmp_path / f"{i}.wav" for i in range(2)]
    for path, frames in zip(paths, [400, 500]):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(1000)
            wav.writeframes(b"\x01\x00" * frames)
    evidence = assemble(paths, allocation["proposal"]["placements"], tmp_path / "group.wav", 0, 1200)
    assert evidence["speech_bytes_unchanged"] and evidence["inserted_silence_verified_zero"]
    assert evidence["frames"] == 1200 and evidence["speech_frames"] == 900
    assert evidence["inter_cue_zero_frames"] == [200]
    assert cues == before
    assert [p["text"] for p in allocation["proposal"]["placements"]] == [c.text for c in cues]


def test_deficit_vs_shift_conflict_does_not_claim_rewrite_measured() -> None:
    cues = [replace(cue(0, "Я думаю,"), start=0, end=200),
            replace(cue(1, "что мы поможем."), start=300, end=2500)]
    row = candidate(cues, 0, 2)
    assert classify_timing(row, allocate(cues, [700, 500]), [700, 500]) == "timing_shift_conflict"
    assert classify_timing(row, allocate(cues, [1500, 1200]), [1500, 1200]) == "needs_group_shortening"
    assert row["timing_evidence"] == "predicted_not_measured"


def test_joined_fit_is_not_a_pause_or_internal_alignment_guarantee() -> None:
    raw_only = joined_timing(1400, 1000, 1500, 200)
    assert raw_only["fits_raw_span"] and not raw_only["fits_reserved_budget_diagnostic"]
    assert raw_only["timing_status"] == "fits_span_needs_pause_and_speaker_review"
    both = joined_timing(1300, 1000, 1500, 200)
    assert both["fits_raw_span"] and both["fits_reserved_budget_diagnostic"]
    assert not both["internal_pauses_verified"] and not both["internal_cue_alignment_known"]
    assert both["per_cue_start_shift_ms"] is None and not both["production_approved"]
    assert joined_timing(1501, 1000, 1500, 200)["timing_status"] == "joined_duration_exceeds_span"
    with pytest.raises(ValueError):
        joined_timing(0, 1000, 1500, 200)


def test_joined_uses_only_six_frozen_texts_and_never_regenerates_individuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import utils.grouping_pilot as module

    pilot, output = tmp_path / "prior", tmp_path / "joined"
    (pilot / "individual" / "wav").mkdir(parents=True)
    _, identity = calibration()
    settings = {"voice": 8, "rate": 1.5, "preprocessing": "vosk_punctuation_v1_no_ruaccent",
                "model_path": "unused-test-model", "identity": identity}
    rows: list[dict[str, Any]] = []
    for i in range(6):
        texts = [f"Мы думаем {i},", "что поможем."] + (["Точно."] if i == 3 else [])
        for j in range(len(texts)):
            path = pilot / "individual" / "wav" / f"{i}_{j}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(1000)
                wav.writeframes(b"\x01\x00")
        rows.append({"group_id": f"group-{i}", "cue_ids": [f"cue-{i * 3 + j}" for j in range(len(texts))],
            "positions": [i * 3 + j for j in range(len(texts))], "measurement_texts": texts,
            "original_joined_text": " ".join(texts), "current_joined_text": "Different saved current text.",
            "measurement_text_policy": "all_originals_due_to_boundary_conflict", "cues": [{"original_text": t} for t in texts],
            "source_start_ms": i * 3000, "source_end_ms": i * 3000 + 1500, "source_span_ms": 1500,
            "duration_ms_ceil": [1000] * len(texts), "review_reasons": ["speaker_unknown"]})
    (pilot / "candidates.json").write_text(json.dumps({"settings": settings,
        "pilot_group_ids": [r["group_id"] for r in rows], "candidates": rows}), encoding="utf-8")
    hashes = module.frozen_hashes([pilot])
    verification_calls: list[Path] = []
    synthesis_calls: list[dict[str, Any]] = []
    fail_index: int | None = None
    monkeypatch.setattr(module, "verify_saved", lambda path: verification_calls.append(path))
    monkeypatch.setattr(module, "model_metadata", lambda path: {})
    monkeypatch.setattr(module, "profile_identity", lambda metadata, **kwargs: identity)

    def fake_audio(report: dict[str, Any], destination: Path, **kwargs: Any) -> dict[str, Any]:
        from utils.vosk_text import normalize_vosk_text

        synthesis_calls.append(report)
        assert destination.name == "continuous" and destination.parent in {output, tmp_path / "failed"}
        assert kwargs["voice"] == 8 and kwargs["speech_rate"] == 1.5
        assert kwargs["text_mode"] == "punctuation" and kwargs["limit"] == 6
        destination.mkdir()
        measured: list[dict[str, Any]] = []
        for i, row in enumerate(report["records"]):
            if i == fail_index:
                measured.append({"cue_id": row["cue_id"], "error": {"type": "SyntheticFailure"}})
                continue
            path = destination / f"group-{i}.wav"
            frames = 1600 if i == 5 else 1000 + 100 * i
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(1000)
                wav.writeframes(b"\x01\x00" * frames)
            measured.append({"cue_id": row["cue_id"], "original": {"path": path.name,
                "frames": frames, "samplerate": 1000, "display_text": row["original"],
                "tts_text": normalize_vosk_text(row["original"])}})
        return {"records": measured, "summary": {"physical_file_count": 6 if fail_index is None else 5}}

    monkeypatch.setattr(module, "verify_audio", fake_audio)
    result = measure_joined(pilot, output)
    assert len(synthesis_calls) == 1 and len(synthesis_calls[0]["records"]) == 6
    assert [r["original"] for r in synthesis_calls[0]["records"]] == [" ".join(r["measurement_texts"]) for r in rows]
    assert result["summary"] == {"groups": 6, "physical_joined_wavs": 6, "fits_raw_span": 5,
                                  "fits_reserved_budget_diagnostic": 3, "synthesis_errors": 0}
    assert result["individual_clips_regenerated"] == 0 and module.frozen_hashes([pilot]) == hashes
    assert verification_calls == [pilot, pilot]
    assert not list(output.rglob("*.srt"))
    with pytest.raises(FileExistsError):
        measure_joined(pilot, output)
    assert len(synthesis_calls) == 1 and len(verification_calls) == 2
    fail_index = 2
    failed = measure_joined(pilot, tmp_path / "failed")
    assert failed["summary"]["synthesis_errors"] == 1
    assert failed["groups"][2]["timing_status"] == "synthesis_error"
    assert failed["groups"][2]["fits_raw_span"] is None
    assert "audio" not in failed["groups"][2]
    assert module.frozen_hashes([pilot]) == hashes and len(synthesis_calls) == 2
