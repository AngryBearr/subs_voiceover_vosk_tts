"""Shared paths and labels for the dubbing application."""

from pathlib import Path

from ui import seasons, settings

REPO_ROOT = Path(__file__).resolve().parents[1]
HELP = Path(__file__).with_name("help.md")
VIDEO_EXT = seasons.VIDEO_EXT
ENGINES, MODES, MODELS = settings.ENGINES, settings.MODES, settings.MODELS
TRANSPORTS = {"Подписка (opencode)": "opencode", "API OpenRouter": "openrouter"}
NO_SEASON = "Сначала выберите сезон вверху страницы (или добавьте его кнопкой «＋ Новый сезон»)."


def _root(out_root: str) -> Path:
    root = Path(out_root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def _season_root(out_root: str) -> Path:
    """Folder of the season files (voice banks, voice clips): the results folder of the season."""
    return _root(out_root)


def _episode(out_root: str, episode: str) -> Path:
    return _root(out_root) / episode


def _season(folder: str, out_root: str) -> seasons.Season | None:
    return seasons.Season(Path(folder).expanduser(), _root(out_root)) if folder and out_root else None
