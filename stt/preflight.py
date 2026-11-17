"""Check before a run that every environment the settings need is present in this checkout.

A missing optional environment used to be noticed only deep inside the run, as a warning in the log, and the run went
on with a worse result (17.11: ``lava_env`` was not linked into a worktree, the narrowband S01E01 was mixed without
the voice widening). ``stt.run`` now stops at the start instead, naming what is missing and how to fix it: link the
shared folder (``bash scripts/link_shared.sh``), install the environment, or switch the setting off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# what scripts/link_shared.sh links from the shared folder (keep in sync)
SHARED_ITEMS = ("subs_env", "f5_env", "diarizen_env", "lava_env", "stt_qwen_env", "models", "output", ".env",
                "stt_config.local.json", "ui.local.json")


def needed_envs(config: dict[str, Any], engines: list[str], until: str, stages: list[str]) -> dict[str, str]:
    """``{environment: why}`` for the extra virtual environments this run will use."""
    from stt.diarize import DIARIZERS

    reach = stages.index(until) if until in stages else len(stages) - 1
    need: dict[str, str] = {}
    spec = DIARIZERS.get(str(config.get("diarizer", "")))
    if spec is not None and spec.env != "subs_env":
        need[spec.env] = f"who speaks when: «{config.get('diarizer')}» (setting diarizer)"
    if "f5" in engines and reach >= stages.index("voiceover"):
        need["f5_env"] = "ESpeech voice-over (engine f5)"
    widen = (config.get("mix") or {}).get("widen") or {}
    if (widen.get("enabled") and reach >= stages.index("mix")
            and any(e in engines for e in widen.get("engines", ["f5"]))):
        need["lava_env"] = "widening of the cloned voices of old seasons (setting mix.widen.enabled)"
    return need


def missing_envs(root: Path, config: dict[str, Any], engines: list[str], until: str,
                 stages: list[str]) -> list[str]:
    """Lines naming each needed environment whose interpreter is not in ``root``."""
    return [f"{env} ({why})" for env, why in needed_envs(config, engines, until, stages).items()
            if not (root / env / "bin" / "python").exists()]


def message(missing: list[str]) -> str:
    return ("Missing environments for this run:\n  " + "\n  ".join(missing) + "\n"
            "Fix: in this checkout run `bash scripts/link_shared.sh` (links the shared environments), or install the "
            "environment (`bash scripts/setup_envs.sh <name>`), or switch the setting off.")
