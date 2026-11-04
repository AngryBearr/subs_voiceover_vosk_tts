"""Choosing a folder (or a file) for a text field of the UI.

The UI runs on the same computer as the person's desktop in the usual case, so the native dialog of the desktop
(``zenity``) is used when there is a desktop (``DISPLAY`` / ``WAYLAND_DISPLAY``) and ``zenity`` is installed. The
handler blocks until the person picks or cancels. Otherwise (no desktop, no ``zenity``, the page is opened from another
computer, or the dialog could not open) the page shows a small folder browser built from the pure helpers below.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gradio as gr  # type: ignore

MEDIA_ROOT = Path("/media")
LIST_LIMIT = 500  # folders shown in the browser
FOLDER, FILE = "folder", "file"  # what is chosen


@dataclass(frozen=True)
class Pick:
    """The answer of the desktop dialog: ``path`` when picked, else ``status`` says why not."""

    path: str | None
    status: str  # picked | cancelled | unavailable
    error: str = ""


# ---------------------------------------------------------------- the desktop dialog

def has_desktop(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def dialog_available(env: Mapping[str, str] | None = None) -> bool:
    """A desktop and ``zenity`` are there."""
    return has_desktop(env) and shutil.which("zenity") is not None


def user_name() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return os.environ.get("USER", "")


def media_dir(media_root: Path = MEDIA_ROOT, user: str | None = None) -> Path:
    return media_root / (user_name() if user is None else user)


def existing_parent(path: Path) -> Path | None:
    """``path`` or its nearest existing folder above (never the root of the disk)."""
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return None if candidate == Path(candidate.anchor) else candidate
    return None


def start_dir(current: str, media_root: Path = MEDIA_ROOT, home: Path | None = None,
              user: str | None = None) -> Path:
    """Where a dialog or the browser opens: the folder of the current field value, else ``/media/<user>`` when it
    exists, else the home folder."""
    text = current.strip().strip("'\"")
    if text:
        path = Path(text).expanduser()
        if path.is_absolute():
            found = existing_parent(path)
            if found is not None:
                return found
    media = media_dir(media_root, user)
    return media if media.is_dir() else (home or Path.home())


def zenity_command(kind: str, start: Path, title: str, file_name: str = "") -> list[str]:
    """The ``zenity`` command line: a folder dialog, or (``FILE``) a save dialog for a ``*.json`` file."""
    if kind == FILE:
        target = start / file_name if file_name else start / ""
        return ["zenity", "--file-selection", "--save", "--confirm-overwrite", f"--title={title}",
                f"--filename={target}", "--file-filter=JSON | *.json"]
    return ["zenity", "--file-selection", "--directory", f"--title={title}", f"--filename={start}/"]


def run_dialog(kind: str, current: str, title: str, env: Mapping[str, str] | None = None,
               media_root: Path = MEDIA_ROOT) -> Pick:
    """Open the desktop dialog and wait for the person. ``unavailable``: no desktop / no ``zenity`` / it failed."""
    if not dialog_available(env):
        return Pick(None, "unavailable")
    start = start_dir(current, media_root)
    file_name = Path(current.strip()).name if kind == FILE and current.strip() else ""
    command = zenity_command(kind, start, title, file_name)
    try:
        done = subprocess.run(command, capture_output=True, text=True, env=dict(env) if env is not None else None)
    except OSError as error:
        return Pick(None, "unavailable", str(error))
    if done.returncode == 0:
        chosen = done.stdout.strip().splitlines()[0] if done.stdout.strip() else ""
        return Pick(chosen, "picked") if chosen else Pick(None, "cancelled")
    if done.returncode == 1:  # zenity: cancel or closed
        return Pick(None, "cancelled")
    return Pick(None, "unavailable", done.stderr.strip())


# ---------------------------------------------------------------- the browser in the page

def list_subfolders(path: Path) -> tuple[list[str], str]:
    """(names of the visible subfolders sorted, a message when the folder cannot be read)."""
    try:
        names = sorted((p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")),
                       key=str.casefold)
    except OSError as error:
        return [], f"Папку нельзя прочитать: {error.strerror or error}"
    return names[:LIST_LIMIT], (f"Показаны первые {LIST_LIMIT} папок." if len(names) > LIST_LIMIT else "")


def parent_of(path: Path) -> Path:
    return path.parent


def shortcuts(media_root: Path = MEDIA_ROOT, home: Path | None = None, user: str | None = None) -> list[tuple[str, Path]]:
    """(label, folder) for the quick buttons; ``/media/<user>`` only when it exists."""
    found = []
    media = media_dir(media_root, user)
    if media.is_dir():
        found.append((str(media), media))
    found.append(("Домашняя папка", home or Path.home()))
    return found


def enter(path: Path, name: str) -> Path:
    """The subfolder ``name`` of ``path`` (``path`` itself when there is no such folder)."""
    target = path / name
    return target if target.is_dir() else path


def video_count(path: Path) -> int:
    from ui import seasons  # a late import: ``seasons`` is not needed for the dialog

    return len(seasons.video_names(path))


def browser_view(path: Path, kind: str = FOLDER) -> tuple[str, list[list[str]]]:
    """(the line with the current folder, the rows of its subfolders)."""
    names, problem = list_subfolders(path)
    line = f"Сейчас: `{path}`"
    if kind == FOLDER:
        videos = video_count(path)
        line += f" · видео здесь: {videos}" if videos else ""
    if problem:
        line += f"\n\n{problem}"
    return line, [[name] for name in names]


def chosen_value(path: Path, kind: str, file_name: str) -> str:
    """What goes into the field: the folder, or (``FILE``) the file in it."""
    return str(path / (file_name.strip() or "stt_config.local.json")) if kind == FILE else str(path)


# ---------------------------------------------------------------- the UI wiring

@dataclass
class Browser:
    """The components of one in-page browser (``build_browser``)."""

    panel: Any
    path: Any  # gr.State: the folder shown
    target: Any  # gr.State: the key of the field being filled
    kind: Any  # gr.State: FOLDER | FILE
    line: Any
    listing: Any
    file_name: Any
    up: Any
    shortcut_buttons: list[Any]
    choose: Any
    cancel: Any


def build_browser(title: str = "Выбор папки") -> Browser:
    """The hidden panel with the folder list; wire it with ``wire_browser``."""
    with gr.Column(visible=False) as panel:
        gr.Markdown(f"**{title}** (на этом компьютере нет окна выбора, поэтому выбор прямо на странице)")
        path_state, target, kind = gr.State(""), gr.State(""), gr.State(FOLDER)
        line = gr.Markdown()
        with gr.Row():
            up = gr.Button("⬆ Вверх", scale=1, min_width=120)
            quick = [gr.Button(label, scale=2) for label, _ in shortcuts()]
        listing = gr.Dataframe(headers=["Папки (нажмите, чтобы войти)"], interactive=False, wrap=True, max_height=300)
        file_name = gr.Textbox(label="Имя файла", visible=False)
        with gr.Row():
            choose = gr.Button("Выбрать эту папку", variant="primary")
            cancel = gr.Button("Отмена")
    return Browser(panel, path_state, target, kind, line, listing, file_name, up, quick, choose, cancel)


def open_picker(kind: str, key: str, current: str, title: str) -> tuple[Any, ...]:
    """The "📁 Выбрать…" click. Outputs: (field, message, panel, path, target, kind, line, listing, file_name, choose).

    With a desktop dialog the field gets the answer at once; otherwise the browser opens at the start folder."""
    pick = run_dialog(kind, current, title)
    nothing = gr.update()
    if pick.status == "picked" and pick.path:
        return (gr.update(value=pick.path), "", gr.update(visible=False), nothing, key, kind, nothing, nothing, nothing,
                nothing)
    if pick.status == "cancelled":
        return nothing, "Выбор отменён, ничего не изменилось.", gr.update(visible=False), nothing, key, kind, nothing, \
            nothing, nothing, nothing
    start = start_dir(current)
    line, rows = browser_view(start, kind)
    name = Path(current.strip()).name if kind == FILE and current.strip() else "stt_config.local.json"
    return (nothing, "", gr.update(visible=True), str(start), key, kind, line, rows,
            gr.update(visible=kind == FILE, value=name), gr.update(value="Выбрать эту папку" if kind == FOLDER
                                                                   else "Выбрать этот файл в папке"))


def navigate(path: str, kind: str, how: str) -> tuple[str, str, list[list[str]]]:
    """Move the browser: ``how`` is ``up``, a subfolder name, or ``@<folder>`` for a shortcut. Returns the new
    (path, line, rows)."""
    here = Path(path or Path.home())
    if how == "up":
        there = parent_of(here)
    elif how.startswith("@"):
        there = Path(how[1:])
        there = there if there.is_dir() else (Path.home())
    else:
        there = enter(here, how)
    line, rows = browser_view(there, kind)
    return str(there), line, rows


def wire_browser(browser: Browser, fields: dict[str, Any], buttons: dict[str, tuple[Any, str, str]],
                 messages: Any) -> list[Any]:
    """Connect the buttons "📁 Выбрать…" and the browser panel.

    ``fields`` maps a key to its Textbox, ``buttons`` maps a key to (button, kind, dialog title), ``messages`` is
    a Markdown for short answers. Returns the events that fill a field (to chain further handlers, e.g. applying
    the chosen settings file)."""
    outputs = [messages, browser.panel, browser.path, browser.target, browser.kind, browser.line, browser.listing,
               browser.file_name, browser.choose]
    events = []
    for key, (button, kind, title) in buttons.items():
        def click(current: str, key: str = key, kind: str = kind, title: str = title) -> tuple[Any, ...]:
            return open_picker(kind, key, current, title)

        event = button.click(click, fields[key], [fields[key], *outputs], show_progress="hidden")
        events.append(event)

    state = [browser.path, browser.kind]

    def go(path: str, kind: str, evt: gr.SelectData) -> tuple[str, str, list[list[str]]]:
        rows, _ = list_subfolders(Path(path))
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return navigate(path, kind, rows[index] if 0 <= index < len(rows) else "up") if rows else (path, "", [])

    browser.listing.select(go, state, [browser.path, browser.line, browser.listing], show_progress="hidden")
    browser.up.click(lambda p, k: navigate(p, k, "up"), state, [browser.path, browser.line, browser.listing],
                     show_progress="hidden")
    for button, (_, folder) in zip(browser.shortcut_buttons, shortcuts()):
        button.click(lambda p, k, f=str(folder): navigate(p, k, "@" + f), state,
                     [browser.path, browser.line, browser.listing], show_progress="hidden")

    def choose(path: str, target: str, kind: str, name: str) -> tuple[Any, ...]:
        value = chosen_value(Path(path), kind, name)
        return (*[gr.update(value=value) if key == target else gr.update() for key in fields], gr.update(visible=False))

    chosen = browser.choose.click(choose, [browser.path, browser.target, browser.kind, browser.file_name],
                                  [*fields.values(), browser.panel], show_progress="hidden")
    browser.cancel.click(lambda: gr.update(visible=False), None, browser.panel, show_progress="hidden")
    return [*events, chosen]
