"""TTS cache cleanup: takes named by an episode or made recently stay, the rest goes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from voiceover.cache_clean import apply, plan


def _take(cache: Path, engine_set: str, key: str, age_days: float) -> None:
    folder = cache / engine_set
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.time() - age_days * 86400
    for name in (f"{key}.wav", f"{key}.json"):
        (folder / name).write_bytes(b"x" * 10)
        os.utime(folder / name, (stamp, stamp))


def test_used_and_fresh_takes_stay(tmp_path: Path) -> None:
    cache, root = tmp_path / "output" / "tts_cache", tmp_path / "output" / "season"
    used, stale, fresh = "a" * 16, "b" * 16, "c" * 16
    _take(cache, used, "1111", 30)
    _take(cache, used, "2222", 30)  # same set, no episode names it
    _take(cache, stale, "3333", 30)
    _take(cache, fresh, "4444", 0.1)
    episode = root / "ep01" / "voiceover.vosk"
    episode.mkdir(parents=True)
    (episode / "durations0.json").write_text(json.dumps({"rows": [{"path": f"output/tts_cache/{used}/1111.wav"}]}))
    result = plan(cache, [root], keep_days=3)
    assert sorted(p.name for p in result.drop) == ["2222.json", "2222.wav", "3333.json", "3333.wav"]
    assert result.sets_emptied == [cache / stale]
    apply(result)
    assert sorted(p.name for p in cache.iterdir()) == [used, fresh]
    assert sorted(p.name for p in (cache / used).iterdir()) == ["1111.json", "1111.wav"]
