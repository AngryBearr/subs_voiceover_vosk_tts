"""stt.preflight: a run stops at the start when an environment its settings need is missing."""

from __future__ import annotations

import re
from pathlib import Path

from stt import preflight
from stt.run import REPO_ROOT, STAGES

BASE = {"diarizer": "diarizen-v2", "mix": {"widen": {"enabled": True, "engines": ["f5"]}}}


def _make_env(root: Path, name: str) -> None:
    (root / name / "bin").mkdir(parents=True)
    (root / name / "bin" / "python").write_text("")


def test_all_needed_envs_listed() -> None:
    need = preflight.needed_envs(BASE, ["f5", "teratts", "vosk"], "mix", STAGES)
    assert set(need) == {"diarizen_env", "f5_env", "lava_env"}


def test_widen_needs_lava_only_with_its_engine_and_the_mix() -> None:
    assert "lava_env" not in preflight.needed_envs(BASE, ["teratts", "vosk"], "mix", STAGES)
    assert "lava_env" not in preflight.needed_envs(BASE, ["f5"], "speakers", STAGES)  # --prepare
    off = {**BASE, "mix": {"widen": {"enabled": False}}}
    assert "lava_env" not in preflight.needed_envs(off, ["f5"], "mix", STAGES)


def test_prepare_does_not_need_f5_env() -> None:
    assert "f5_env" not in preflight.needed_envs(BASE, ["f5"], "speakers", STAGES)


def test_pyannote_needs_no_extra_env() -> None:
    need = preflight.needed_envs({"diarizer": "pyannote-community-1"}, ["vosk"], "mix", STAGES)
    assert need == {}


def test_missing_envs_reported(tmp_path: Path) -> None:
    _make_env(tmp_path, "diarizen_env")
    _make_env(tmp_path, "f5_env")
    missing = preflight.missing_envs(tmp_path, BASE, ["f5"], "mix", STAGES)
    assert len(missing) == 1 and missing[0].startswith("lava_env")
    assert "link_shared.sh" in preflight.message(missing)
    _make_env(tmp_path, "lava_env")
    assert preflight.missing_envs(tmp_path, BASE, ["f5"], "mix", STAGES) == []


def test_link_script_lists_the_same_items() -> None:
    script = (REPO_ROOT / "scripts" / "link_shared.sh").read_text(encoding="utf-8")
    items = re.search(r"^ITEMS=\((.*)\)$", script, re.M)
    assert items is not None
    assert tuple(items.group(1).split()) == preflight.SHARED_ITEMS
