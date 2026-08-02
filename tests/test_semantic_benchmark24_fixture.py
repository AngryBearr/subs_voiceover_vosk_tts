import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "skill_test" / "semantic_benchmark24"


def _load(name: str):
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def test_benchmark24_fixture_is_complete_and_reproducible() -> None:
    manifest = _load("manifest.json")
    assert len(list(FIXTURE.glob("*.json"))) == 7
    cases = manifest["cases"]
    assert len(cases) == 24
    assert sum(case["clear"] for case in cases) == 21
    assert sum(not case["clear"] for case in cases) == 3
    assert {expected: sum(case["expected"] == expected for case in cases) for expected in ("pass", "fail")} == {
        "pass": 10,
        "fail": 14,
    }
    assert {episode: sum(case["episode"] == episode for case in cases) for episode in ("E01", "E03", "E04")} == {
        "E01": 10,
        "E03": 5,
        "E04": 9,
    }

    seen = set()
    for case in cases:
        key = (case["episode"], case["index"])
        assert key not in seen
        seen.add(key)
        assert case["expected"] in {"pass", "fail"}
        if case["episode"] == "E01":
            assert case["source_context_path"] == "skill_test/S01_E01_ru_analyzed.json"
        else:
            assert case["source_context_file"] in {"S01E03_ru_analyzed.json", "S01E04_ru_analyzed.json"}

    for episode in ("e01", "e03", "e04"):
        original = _load(f"{episode}_original.json")
        candidate = _load(f"{episode}_candidate.json")
        assert len(original) == len(candidate)
        assert len(original) == sum(case["episode"].lower() == episode for case in cases)
        for original_item, candidate_item in zip(original, candidate):
            assert original_item["index"] == candidate_item["index"]
            assert {**original_item, "text": None} == {**candidate_item, "text": None}
            case = next(case for case in cases if case["episode"].lower() == episode and case["index"] == original_item["index"])
            assert " ".join(original_item["text"]) == case["original"]
            assert " ".join(candidate_item["text"]) == case["candidate"]

    assert any(
        entry.get("episode") == "E01" and entry.get("index") == 128
        for entry in manifest["revision_history"]
    )
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert not re.search(r"(?:/home/|/tmp/|[A-Za-z]:[\\/])", serialized)
    assert (ROOT / "skill_test" / "S01_E01_ru_analyzed.json").is_file()
