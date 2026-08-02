import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "skill_test" / "semantic_benchmark60"
def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark60_fixture_is_complete_and_reproducible() -> None:
    manifest = _load(FIXTURE / "manifest.json")
    assert isinstance(manifest, dict)
    cases = manifest["cases"]
    assert len(list(FIXTURE.glob("*.json"))) == 7
    assert len(cases) == 60
    assert manifest["benchmark"] == "semantic-benchmark60"
    assert sum(case["clear"] for case in cases) == 57
    assert sum(not case["clear"] for case in cases) == 3
    assert {value: sum(case["expected"] == value for case in cases) for value in ("pass", "fail")} == {"pass": 28, "fail": 32}
    assert {episode: sum(case["episode"] == episode for case in cases) for episode in ("E01", "E03", "E04")} == {"E01": 22, "E03": 17, "E04": 21}

    seen = set()
    for case in cases:
        pair = (case["episode"], case["index"])
        assert pair not in seen
        seen.add(pair)
        assert case["expected"] in {"pass", "fail"}
        assert (case.get("source_context_path") == "skill_test/S01_E01_ru_analyzed.json") if case["episode"] == "E01" else case.get("source_context_file") in {"S01E03_ru_analyzed.json", "S01E04_ru_analyzed.json"}

    historical = _load(ROOT / "skill_test/semantic_benchmark24/manifest.json")["cases"]
    assert cases[:24] == historical
    historical_pairs = {(case["episode"], case["index"]) for case in historical}
    expansion_pairs = [(case["episode"], case["index"]) for case in cases[24:]]
    assert len(expansion_pairs) == len(set(expansion_pairs)) == 36
    assert not historical_pairs & set(expansion_pairs)
    for episode in ("e01", "e03", "e04"):
        original = _load(FIXTURE / f"{episode}_original.json")
        candidate = _load(FIXTURE / f"{episode}_candidate.json")
        episode_cases = [case for case in cases if case["episode"].lower() == episode]
        assert len(original) == len(candidate) == len(episode_cases)
        assert [item["index"] for item in original] == [case["index"] for case in episode_cases]
        for original_item, candidate_item, case in zip(original, candidate, episode_cases):
            assert {**original_item, "text": None} == {**candidate_item, "text": None}
            assert " ".join(original_item["text"]) == case["original"]
            assert " ".join(candidate_item["text"]) == case["candidate"]

    assert manifest["revision_history"] == _load(ROOT / "skill_test/semantic_benchmark24/manifest.json")["revision_history"]
    expansion = manifest["expansion_history"][0]
    assert expansion == {"date": "2026-08-02", "added_cases": 36, "reviewed": "manually reviewed clear cases", "reviewed_count": 36, "expected_distribution": {"pass": 18, "fail": 18}, "reviewer": "reviewer audit", "neighbor_checks": ["E01/242", "E03/122", "E03/994"]}
    assert next(case for case in cases if (case["episode"], case["index"]) == ("E01", 128))["clear"] is False
    assert next(case for case in cases if (case["episode"], case["index"]) == ("E03", 122))["rationale"] == 'Loses the evaluation "Это было так глупо" and its link to not sending the observer.'
    assert manifest["source_context_provenance"] == _load(ROOT / "skill_test/semantic_benchmark24/manifest.json")["source_context_provenance"]
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert not re.search(r"(?:/home/|/tmp/|[A-Za-z]:[\\/])", serialized)
