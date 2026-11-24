"""stt.preflight: a run stops at the start when an environment its settings need is missing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from stt import preflight
from stt.run import REPO_ROOT, STAGES

BASE = {"diarizer": "diarizen-v2", "mix": {"widen": {"enabled": True, "engines": ["f5"]}}}


def _make_env(root: Path, name: str) -> None:
    (root / name / "bin").mkdir(parents=True)
    (root / name / "bin" / "python").write_text("")
    if name == "f5_env":
        site = root / name / "lib" / "python3.12" / "site-packages"
        site.mkdir(parents=True)
        shared_site = root / "subs_env" / "lib" / "python3.12" / "site-packages"
        shared_site.mkdir(parents=True)
        (site / "subs_env.pth").write_text(str(shared_site.resolve()) + "\n", encoding="utf-8")


def test_all_needed_envs_listed() -> None:
    need = preflight.needed_envs(BASE, ["f5", "teratts", "vosk"], "mix", STAGES)
    assert set(need) == {"diarizen_env", "f5_env", "lava_env"}


def test_widen_needs_lava_only_with_its_engine_and_the_mix() -> None:
    assert "lava_env" not in preflight.needed_envs(BASE, ["teratts", "vosk"], "mix", STAGES)
    assert "lava_env" not in preflight.needed_envs(BASE, ["f5"], "speakers", STAGES)  # --prepare
    off = {**BASE, "mix": {"widen": {"enabled": False}}}
    assert "lava_env" not in preflight.needed_envs(off, ["f5"], "mix", STAGES)


def test_russian_samples_need_xvoice_env_in_the_dub_stage_only() -> None:
    ru = {**BASE, "voiceover": {"engines": {"f5": {"reference_language": "ru"}}}}
    assert "xvoice_env" in preflight.needed_envs(ru, ["f5"], "mix", STAGES)
    assert "xvoice_env" not in preflight.needed_envs(ru, ["f5"], "speakers", STAGES)  # --prepare
    assert "xvoice_env" not in preflight.needed_envs(ru, ["teratts"], "mix", STAGES)
    en = {**BASE, "voiceover": {"engines": {"f5": {"reference_language": "en"}}}}
    assert "xvoice_env" not in preflight.needed_envs(en, ["f5"], "mix", STAGES)
    assert "xvoice_env" not in preflight.needed_envs(BASE, ["f5"], "mix", STAGES)


def test_own_voice_needs_xvoice_env_from_the_dub_stage_with_f5_only_when_enabled() -> None:
    on = {**BASE, "voiceover": {"engines": {"f5": {"reference_language": "en", "own_voice": {"enabled": True}}}}}
    assert "xvoice_env" in preflight.needed_envs(on, ["f5"], "mix", STAGES)
    assert "xvoice_env" in preflight.needed_envs(on, ["f5"], "dub", STAGES)
    assert "xvoice_env" not in preflight.needed_envs(on, ["f5"], "speakers", STAGES)  # --prepare
    assert "xvoice_env" not in preflight.needed_envs(on, ["teratts"], "mix", STAGES)
    off = {**BASE, "voiceover": {"engines": {"f5": {"reference_language": "en", "own_voice": {"enabled": False}}}}}
    assert "xvoice_env" not in preflight.needed_envs(off, ["f5"], "mix", STAGES)
    assert "xvoice_env" not in preflight.needed_envs(BASE, ["f5"], "mix", STAGES)


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


@pytest.mark.parametrize("contents", [None, "", "# only a comment\n", "/missing/shared/site-packages\n",
                                     "import nonexistent\n"])
def test_f5_pth_missing_or_invalid_is_reported(tmp_path: Path, contents: str | None) -> None:
    _make_env(tmp_path, "f5_env")
    pth = tmp_path / "f5_env/lib/python3.12/site-packages/subs_env.pth"
    if contents is None:
        pth.unlink()
    else:
        pth.write_text(contents, encoding="utf-8")
    missing = preflight.missing_envs(tmp_path, {"diarizer": "pyannote-community-1"}, ["f5"], "voiceover", STAGES)
    assert len(missing) == 1
    assert "f5_env" in missing[0] and "subs_env.pth" in missing[0] and "setup_envs.sh f5" in missing[0]
    assert preflight.missing_envs(tmp_path, BASE, ["f5"], "speakers", STAGES) == [
        "diarizen_env (who speaks when: «diarizen-v2» (setting diarizer))"]


def test_f5_pth_checks_symlink_targets_without_running_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def no_process(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight must not launch an interpreter")

    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    _make_env(tmp_path, "f5_env")
    pth = tmp_path / "f5_env/lib/python3.12/site-packages/subs_env.pth"
    shared_site = tmp_path / "subs_env/lib/python3.12/site-packages"
    link = tmp_path / "shared-packages"
    link.symlink_to(shared_site, target_is_directory=True)
    pth.write_text(f"# Shared packages\n\n{link}\n", encoding="utf-8")
    assert preflight.f5_shared_packages_error(tmp_path) is None
    shared_site.rmdir()
    assert "shared-packages" in (preflight.f5_shared_packages_error(tmp_path) or "")
    pth.unlink()
    pth.symlink_to(tmp_path / "missing.pth")
    assert "cannot read" in (preflight.f5_shared_packages_error(tmp_path) or "")


def test_f5_pth_relative_paths_follow_site_packages(tmp_path: Path) -> None:
    _make_env(tmp_path, "f5_env")
    pth = tmp_path / "f5_env/lib/python3.12/site-packages/subs_env.pth"
    pth.write_text("../../../../subs_env/lib/python3.12/site-packages\n", encoding="utf-8")
    assert preflight.f5_shared_packages_error(tmp_path) is None


def test_opencode_report_warns_without_a_pinned_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import opencode_pin

    monkeypatch.setattr(opencode_pin, "describe", lambda: ("/usr/bin/opencode (v2.4.0, PATH)", False))
    text, pinned = preflight.opencode_report()
    assert not pinned and "python -m utils.opencode_pin" in text
    monkeypatch.setattr(opencode_pin, "describe", lambda: ("/s/current/opencode (v2.4.0, pinned)", True))
    assert preflight.opencode_report() == ("/s/current/opencode (v2.4.0, pinned)", True)
