"""Translation proofreading: accepted and rejected corrections, untouched lines, the report, the run.py flag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest import mock

from stt import translate
from stt.run import Episode


def _records() -> list[dict[str, Any]]:
    texts = ["Ana, nice to meet you.", "I was so tired.", "We go home now.", "I did it.", "Fine."]
    genders = ["female", "female", "male", "female", "male"]
    return [{"index": n, "start": n * 3000, "end": n * 3000 + 2000, "text": [t], "speaker": f"S0{n}", "gender": g}
            for n, (t, g) in enumerate(zip(texts, genders), start=1)]


TRANSLATED = {1: "Ана, приятно познаться.", 2: "Я так устала.", 3: "Мы идём домой сейчас.", 4: "Я сделала.", 5: "Хорошо."}
BUDGETS = {1: 30, 2: 20, 3: 40, 4: 20, 5: 20}


class Reader:
    """Answers with fixed corrections; remembers the requests."""

    def __init__(self, answers: list[dict[str, Any]] | Exception) -> None:
        self.answers, self.prompts = answers, []

    def chat_json(self, system: str, user: str, **_: Any) -> dict[str, Any]:
        self.prompts.append(user)
        if isinstance(self.answers, Exception):
            raise self.answers
        return {"items": self.answers}


def _run(answers: list[dict[str, Any]] | Exception, **kwargs: Any) -> tuple[dict[int, str], list[dict[str, Any]], Reader]:
    translated, reader = dict(TRANSLATED), Reader(answers)
    report = translate.proofread_pass(_records(), translated, reader, budgets=BUDGETS, tolerance=1.15,  # type: ignore[arg-type]
                                      **kwargs)
    return translated, report, reader


def test_accepted_fix_changes_only_that_line_and_is_reported() -> None:
    translated, report, _ = _run([{"i": 1, "ru": "Ана, приятно познакомиться.", "why": "non-word"}])
    assert translated[1] == "Ана, приятно познакомиться."
    assert {k: v for k, v in translated.items() if k != 1} == {k: v for k, v in TRANSLATED.items() if k != 1}
    assert report == [{"index": 1, "before": "Ана, приятно познаться.", "after": "Ана, приятно познакомиться.",
                       "why": "non-word", "status": "accepted", "reason": None}]


def test_over_budget_fix_is_rejected_but_a_line_that_was_over_may_not_grow() -> None:
    long_fix = "Ана, очень приятно с вами познакомиться."  # 40 chars, budget 30 x 1.15
    translated, report, _ = _run([{"i": 1, "ru": long_fix, "why": "style"}])
    assert translated[1] == TRANSLATED[1] and report[0]["reason"] == "over budget"
    over = {**TRANSLATED, 3: "Мы идём домой прямо сейчас, и это совершенно точно так и есть."}  # already over budget 40 x 1.15
    reader = Reader([{"i": 3, "ru": "Мы идём домой прямо сейчас, и это совершенно точно так и есть, да.", "why": "x"}])
    report = translate.proofread_pass(_records(), over, reader, budgets=BUDGETS, tolerance=1.15)  # type: ignore[arg-type]
    assert report[0]["reason"] == "over budget"
    shorter = Reader([{"i": 3, "ru": "Мы идём домой прямо сейчас, и это совершенно точно так.", "why": "x"}])
    report = translate.proofread_pass(_records(), over, shorter, budgets=BUDGETS, tolerance=1.15)  # type: ignore[arg-type]
    assert report[0]["status"] == "accepted"


def test_gender_mismatch_fix_is_rejected() -> None:
    translated, report, _ = _run([{"i": 4, "ru": "Я сделал.", "why": "x"}])  # speaker is female
    assert translated[4] == "Я сделала." and report[0]["reason"] == "gender mismatch"
    translated, report, _ = _run([{"i": 3, "ru": "Мы идём домой.", "why": "x"}])  # male, no first-person form
    assert translated[3] == "Мы идём домой." and report[0]["status"] == "accepted"


def test_unchanged_unknown_and_empty_answers_are_ignored() -> None:
    translated, report, _ = _run([{"i": 2, "ru": TRANSLATED[2]}, {"i": 99, "ru": "Лишнее."}, {"i": 5, "ru": " "},
                                  "junk", {"i": 1}])
    assert translated == TRANSLATED
    assert [(r["index"], r["reason"]) for r in report] == [(2, "unchanged")]


def test_batches_have_context_and_a_failing_batch_leaves_the_lines() -> None:
    translated, report, reader = _run([{"i": 4, "ru": "Я полностью сделала.", "why": "x"}], batch_size=2, context=1,
                                      speakers=[{"id": "S01", "label_ru": "Ана", "gender": "female", "cues": 1}],
                                      glossary="Ana -> Ана", parallel=1)
    assert len(reader.prompts) == 3  # lines 1-2, 3-4, 5
    second = [json.loads(x) for x in reader.prompts[1].split("Lines:\n", 1)[1].splitlines()]
    assert [bool(x.get("ctx")) for x in second] == [True, False, False, True]
    assert [x.get("i") for x in second] == [None, 3, 4, None] and second[1]["g"] == "male" and "Ана" in reader.prompts[0]
    assert "Ana -> Ана" in reader.prompts[0]
    broken, report, _ = _run(RuntimeError("down"))
    assert broken == TRANSLATED and report == []


def test_main_writes_the_proofread_report_and_usage(tmp_path: Path) -> None:
    records = [{**r, "text": r["text"]} for r in _records()]
    (tmp_path / "subs.en.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "speakers.json").write_text("[]", encoding="utf-8")

    class Client:
        def __init__(self, model: str, **_: Any) -> None:
            self.model, self.usage = model, mock.Mock(cost=0.0, to_dict=lambda: {"calls": 1})

        def chat_json(self, system: str, user: str, **_: Any) -> dict[str, Any]:
            if system == translate.PROOFREAD_PROMPT:
                return {"items": [{"i": 1, "ru": "Ана, приятно познакомиться.", "why": "non-word"}]}
            rows = [json.loads(x) for x in user.split("Lines:\n", 1)[1].splitlines()]
            return {"items": [{"i": r["i"], "ru": TRANSLATED[r["i"]]} for r in rows if not r.get("ctx")]}

    with mock.patch.object(translate, "OpenRouterClient", Client):
        translate.main([str(tmp_path), "--parallel", "1", "--proofread-model", "x/reader"])
    out = json.loads((tmp_path / "subs.ru.json").read_text(encoding="utf-8"))
    assert out[0]["text"] == ["Ана, приятно познакомиться."] and out[1]["text"] == [TRANSLATED[2]]
    report = json.loads((tmp_path / "subs.ru.proofread.json").read_text(encoding="utf-8"))
    assert report[0]["status"] == "accepted" and report[0]["index"] == 1
    usage = json.loads((tmp_path / "subs.ru.usage.json").read_text(encoding="utf-8"))
    assert usage["proofread"]["changed"] == 1 and usage["proofread"]["model"] == "x/reader"
    with mock.patch.object(translate, "OpenRouterClient", Client):  # off by default: no report
        translate.main([str(tmp_path), "--parallel", "1", "--suffix", ".plain"])
    assert not (tmp_path / "subs.ru.plain.proofread.json").exists()


def test_run_passes_the_proofread_model_when_enabled(tmp_path: Path) -> None:
    args = argparse.Namespace(season_glossary=None, force=None, until="translate")
    config = {"llm": {"model": "openai/gpt-6-luna", "stages": {"translate_proofread": "openai/gpt-6.1-sol"}},
              "translate": {"cps": 14.0, "extend": 0.6, "parallel": 2, "proofread": True}, "on_screen": {"track": False}}
    episode = Episode(Path("ep.mkv"), tmp_path, args, config)
    with mock.patch.object(translate, "main") as main:
        episode.translate()
    argv = main.call_args[0][0]
    assert argv[argv.index("--proofread-model") + 1] == "openrouter:openai/gpt-6.1-sol"
    assert argv[argv.index("--model") + 1] == "openrouter:openai/gpt-6-luna"
    config["translate"]["proofread"] = False
    with mock.patch.object(translate, "main") as main:
        Episode(Path("ep.mkv"), tmp_path, args, config).translate()
    assert "--proofread-model" not in main.call_args[0][0]
