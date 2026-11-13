"""A JSON command line over the settings layers and the job launcher of the UI, for the Claude Code skill that asks the
person about launch settings and starts a season. The skill, the UI and ``stt.run`` agree because this module reuses
``ui.settings``, ``ui.season_settings`` and ``ui.seasons`` (it does not import ``ui.dub_app``: that pulls in gradio).

    subs_env/bin/python -m ui.run_setup seasons
    subs_env/bin/python -m ui.run_setup show [TARGET]
    subs_env/bin/python -m ui.run_setup save --scope global|season [TARGET] KEY=VALUE ...
    subs_env/bin/python -m ui.run_setup forget-season [TARGET]
    subs_env/bin/python -m ui.run_setup launch [TARGET] [--episodes E ...] [--all] [--engines vosk,f5] [--dry-run] ...
    subs_env/bin/python -m ui.run_setup status [TARGET]
    subs_env/bin/python -m ui.run_setup ui [--port 7861] [--open]

Every command prints one JSON object. Errors are ``{"error": code, "message": <Russian text>}`` with exit code 2 (bad
input) or 3 (busy). TARGET: nothing (the current season of the UI), a season name / source / results folder, a
folder with videos, or one video file (then that episode is the default selection).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from stt import progress as stt_progress
from stt.run import apply_preset, apply_show, dotenv, show_of
from ui import season_settings, seasons, settings
from utils.atomic_io import write_text_atomic

log = logging.getLogger("ui.run_setup")

REPO_ROOT = settings.REPO_ROOT
LAST_JOB = REPO_ROOT / "output" / "ui_last_job.json"  # the file ``ui.dub_app`` keeps its last job in
UI_LOG = REPO_ROOT / "output" / "ui_app.log"
UI_PORT = 7861
JOB_TITLE = "Обработка серий (запущено из Claude)"
LAUNCH_ENGINES = ("f5", "teratts", "vosk")  # the order they run in: the heroes' voices first
KEYS = ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "HF_TOKEN")
BY_KEY: dict[str, settings.Setting] = {".".join(s.path): s for s in settings.SETTINGS}
_MISSING = object()


class CliError(Exception):
    """An error printed as JSON; ``exit_code`` 2 is bad input, 3 is busy."""

    def __init__(self, code: str, message: str, exit_code: int = 2, **extra: Any) -> None:
        super().__init__(message)
        self.code, self.message, self.exit_code, self.extra = code, message, exit_code, extra


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError("bad_arguments", f"Неверные аргументы команды: {message}")


# ---------------------------------------------------------------- small helpers

def _dig(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    """The value at ``path``; ``_MISSING`` when a key is absent (``settings.get`` would give the page's fallback)."""
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dicts as ``{"a.b.c": value}``."""
    if not isinstance(node, dict):
        return {prefix: node} if prefix else {}
    out: dict[str, Any] = {}
    for key, value in node.items():
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)) if isinstance(value, dict)
                   else {f"{prefix}.{key}" if prefix else str(key): value})
    return out


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _split_pairs(items: list[str]) -> list[tuple[str, Any]]:
    pairs = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise CliError("bad_pair", f"Ожидалось КЛЮЧ=ЗНАЧЕНИЕ, получено: {item!r}")
        pairs.append((key.strip(), _parse_value(value)))
    return pairs


def _tail_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- target

@dataclass(frozen=True)
class Target:
    season: seasons.Season
    episode: str | None = None  # a video file name when the target was a file


def _known() -> list[seasons.Season]:
    return seasons.available(season_settings.global_config())


def _same(a: Path, b: Path) -> bool:
    return a == b or str(a).rstrip("/") == str(b).rstrip("/")


def resolve_target(text: str | None) -> Target:
    """The season a command works on (see the module docstring for what TARGET may be)."""
    known = _known()
    if not text or not text.strip():
        current = seasons.pick_current(known)
        if current is None:
            raise CliError("no_season", "Сезон не выбран: нет ни одного сезона в интерфейсе. Укажите папку с сериями.")
        return Target(current)
    text = text.strip()
    path = seasons.absolute(text)
    for season in known:
        if (text.lower() == season.name.lower() or _same(path, season.source) or _same(path, season.results)):
            return Target(season)
    if path.is_file() and path.suffix.lower() in seasons.VIDEO_EXT:
        folder = path.parent
        season = next((s for s in known if _same(s.source, folder)), None)
        return Target(season or seasons.Season(folder, seasons.suggest_results(folder)), path.name)
    if path.is_dir():
        if seasons.video_names(path):
            return Target(seasons.Season(path, seasons.suggest_results(path)))
        sibling = path.parent / path.name[len(seasons.RESULTS_PREFIX):]
        if path.name.startswith(seasons.RESULTS_PREFIX) and seasons.video_names(sibling):
            return Target(seasons.Season(sibling, path))
    raise CliError("bad_target", f"Не нашёл сезон «{text}»: это не известный сезон, не папка с видео и не видеофайл.",
                   known=[s.name for s in known])


# ---------------------------------------------------------------- episodes and processes

def _code(name: str) -> str | None:
    match = seasons._CODE.search(Path(name).stem)
    return f"S{int(match.group(1)):02d}E{int(match.group(2)):02d}" if match else None


def video_minutes(path: Path) -> float | None:
    """Length of a video in minutes (1 decimal), read by PyAV like the rest of the pipeline; None on any failure."""
    try:
        import av  # noqa: PLC0415 - heavy, only needed here

        with av.open(str(path)) as container:
            if container.duration:
                return round(container.duration / av.time_base / 60, 1)
    except Exception:  # noqa: BLE001 - a missing or broken file only means "length unknown"
        return None
    return None


def episode_dicts(season: seasons.Season) -> list[dict[str, Any]]:
    return [{"video": r.video, "code": _code(r.video), "label": seasons.episode_label(r.video), "state": r.state,
             "finished": r.finished is not None, "duration_min": video_minutes(season.source / r.video)}
            for r in seasons.episode_rows(season)]


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def stt_run_processes() -> list[dict[str, Any]]:
    """Live processes with an ``stt.run`` argument (mirrors ``dub_app._is_task_process``, which checks one pid)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        argv = _cmdline(int(entry.name))
        if "stt.run" in argv:
            found.append({"pid": int(entry.name), "cmd": " ".join(argv)[:300]})
    return found


def _log_tail(path: Path, lines: int = 20) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 65536))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [line.rstrip() for line in text.splitlines() if line.strip()][-lines:]


def last_job_info(path: Path | None = None) -> dict[str, Any] | None:
    """The job of ``output/ui_last_job.json`` with ``alive`` and the end of its log."""
    data = _tail_json(path or LAST_JOB)
    if not data:
        return None
    pid = data.get("pid")
    alive = bool(pid) and stt_progress.process_alive({"pid": pid, "boot": data.get("boot") or ""}) \
        and "stt.run" in _cmdline(int(pid))
    return {"state": data.get("state"), "title": data.get("title"), "alive": alive, "started": data.get("started"),
            "files": data.get("files") or [], "engines": data.get("engines") or [], "log": data.get("log_path"),
            "log_tail": _log_tail(Path(data["log_path"])) if data.get("log_path") else []}


def running() -> dict[str, Any]:
    return {"processes": stt_run_processes(), "last_job": last_job_info()}


# ---------------------------------------------------------------- commands: seasons, show, status

def cmd_seasons(_: argparse.Namespace) -> dict[str, Any]:
    known = _known()
    current = seasons.pick_current(known)
    rows = []
    for season in known:
        eps = seasons.episode_rows(season)
        rows.append({"name": season.name, "source": str(season.source), "results": str(season.results),
                     "source_ok": bool(seasons.video_names(season.source)), "episodes": len(eps),
                     "finished": sum(1 for e in eps if e.finished), "has_own_settings": season_settings.has_own(season.results)})
    return {"current": str(current.source) if current else None, "seasons": rows}


def _display(setting: settings.Setting, value: Any) -> Any:
    if setting.kind in ("choice", "engines"):
        return settings.to_form(setting, value)
    if setting.kind == "vosk_voice" and isinstance(value, (int, float)):
        label = next((lab for lab, number in settings.vosk_voice_choices(setting.path[-1]) if number == int(value)), None)
        return label or str(value)
    return value


def _setting_rows(config_std: dict[str, Any], config_global: dict[str, Any], results: Path,
                  show_settings: dict[str, Any]) -> list[dict[str, Any]]:
    own = season_settings.read_own(results)
    effective = season_settings.effective(config_global, results)
    rows = []
    for key, setting in BY_KEY.items():
        std, glob, eff = (settings.get(c, setting.path) for c in (config_std, config_global, effective))
        in_season = _dig(own, setting.path) is not _MISSING
        row: dict[str, Any] = {
            "key": key, "label": setting.label, "group": setting.group, "kind": setting.kind, "info": setting.info,
            "choices": dict(setting.choices) if setting.choices else None,
            "standard": std, "global": glob, "season": _dig(own, setting.path) if in_season else None,
            "effective": eff, "display": _display(setting, eff),
            "source": "season" if in_season else ("global" if glob != std else "standard")}
        if key in show_settings:
            row["show_override"] = show_settings[key]
        rows.append(row)
    return rows


def _env_info(results: Path) -> dict[str, Any]:
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        name, total, free = [part.strip() for part in out.stdout.splitlines()[0].split(",")]
        gpu = {"name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    file_env = dotenv()
    keys = {k: bool(os.environ.get(k) or file_env.get(k)) for k in KEYS}
    keys["HF_TOKEN"] = keys["HF_TOKEN"] or (Path.home() / ".cache" / "huggingface" / "token").is_file()
    opencode = shutil.which("opencode") or (str(p) if (p := Path.home() / ".opencode" / "bin" / "opencode").is_file() else None)
    folder = results
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    try:
        free_gb = round(shutil.disk_usage(folder).free / 1e9, 1)
    except OSError:
        free_gb = None
    return {"f5_env": (REPO_ROOT / "f5_env" / "bin" / "python").is_file(),
            "diarizen_env": (REPO_ROOT / "diarizen_env" / "bin" / "python").is_file(),
            "lava_env": (REPO_ROOT / "lava_env" / "bin" / "python").is_file(), "gpu": gpu, "keys": keys,
            "opencode": opencode, "disk_free_gb": free_gb}


def ui_url(port: int = UI_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def ui_running(port: int = UI_PORT) -> bool:
    try:
        with urllib.request.urlopen(ui_url(port), timeout=1):
            return True
    except (OSError, ValueError):
        return False


def _launch_defaults(effective: dict[str, Any]) -> dict[str, Any]:
    engines = _dig(effective, ("ui", "engines"))
    if engines is _MISSING or not engines:
        engines = [e.strip() for e in str(_dig(effective, ("voiceover", "engine")) or "").split(",") if e.strip()]
    llm = effective.get("llm") or {}
    return {"engines": [e for e in LAUNCH_ENGINES if e in engines] or ["vosk"], "mode": effective.get("mode"),
            "llm": {"model": llm.get("model"), "opencode": llm.get("opencode"), "router": llm.get("router"),
                    "stages": llm.get("stages")}}


def _show_of(effective: dict[str, Any], season: seasons.Season) -> str | None:
    names = seasons.video_names(season.source)
    return show_of(effective, season.source / names[0]) if names else None


def cmd_show(args: argparse.Namespace) -> dict[str, Any]:
    target = resolve_target(args.target)
    season = target.season
    config_std = settings.read_json(settings.DEFAULTS)
    config_global = season_settings.global_config()
    effective = season_settings.effective(config_global, season.results)
    show = _show_of(effective, season)
    profile = (effective.get("shows") or {}).get(show) or {} if show else {}
    show_settings = _flatten(profile.get("settings") or {})
    episodes = episode_dicts(season)
    unfinished = [e["duration_min"] for e in episodes if not e["finished"] and e["duration_min"] is not None]
    return {
        "season": {"name": season.name, "source": str(season.source), "results": str(season.results),
                   "results_exists": season.results.is_dir(), "settings_file": str(season_settings.season_file(season.results)),
                   "has_own_settings": season_settings.has_own(season.results),
                   "in_effect": season_settings.in_effect(season.results), "global_file": str(season_settings.global_path()),
                   "show": show, "show_label": profile.get("label") if show else None, "show_settings": show_settings,
                   "total_unfinished_min": round(sum(unfinished), 1)},
        "target_episode": target.episode,
        "episodes": episodes,
        "settings": _setting_rows(config_std, config_global, season.results, show_settings),
        "presets": [{"name": name, "label": (p or {}).get("label", name), "settings": _flatten((p or {}).get("settings") or {})}
                    for name, p in (effective.get("presets") or {}).items()],
        "launch_defaults": _launch_defaults(effective),
        "env": _env_info(season.results),
        "running": running(),
        "ui": {"url": ui_url(), "running": ui_running()},
    }


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    season = resolve_target(args.target).season
    return {"season": {"name": season.name, "results": str(season.results)}, "episodes": episode_dicts(season),
            "running": running()}


# ---------------------------------------------------------------- save, forget

def _check_value(setting: settings.Setting, key: str, value: Any) -> None:
    """Early, readable validation by kind; ``from_form`` (inside the save functions) checks ranges and voices."""
    def bad(why: str) -> NoReturn:
        raise CliError("bad_value", f"{key}: {why}", key=key, value=value)

    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if setting.kind == "choice":
        if value not in setting.choices.values():
            bad(f"допустимо одно из значений: {', '.join(json.dumps(v, ensure_ascii=False) for v in setting.choices.values())}")
    elif setting.kind == "engines":
        allowed = list(setting.choices.values())
        if not isinstance(value, list) or not all(v in allowed for v in value):
            bad(f"нужен список из значений: {', '.join(allowed)}")
    elif setting.kind == "bool":
        if not isinstance(value, bool):
            bad("нужно true или false")
    elif setting.kind in ("number", "integer", "vosk_voice"):
        if not numeric:
            bad("нужно число")
    elif setting.kind in ("text", "folder", "voice"):
        if not (isinstance(value, str) or (value is None and setting.nullable)):
            bad("нужна строка")


def cmd_save(args: argparse.Namespace) -> dict[str, Any]:
    items = list(args.items)
    target_text = items.pop(0) if items and "=" not in items[0] else None
    pairs = _split_pairs(items)
    if not pairs:
        raise CliError("no_values", "Не заданы значения: укажите KEY=VALUE.", allowed=list(BY_KEY))
    for key, value in pairs:
        if key not in BY_KEY:
            raise CliError("bad_key", f"Нет такой настройки: {key}", key=key, allowed=list(BY_KEY))
        _check_value(BY_KEY[key], key, value)
    only = [BY_KEY[key] for key, _ in pairs]
    form = [settings.to_form(BY_KEY[key], value) if BY_KEY[key].kind in ("choice", "engines") else
            ("" if value is None else value) if BY_KEY[key].kind in ("text", "folder", "voice") else value
            for key, value in pairs]
    saved = {key: value for key, value in pairs}
    warnings: list[str] = []
    result: dict[str, Any] = {}
    try:
        if args.scope == "global":
            path = season_settings.global_path()
            result["created"] = not path.is_file()
            _, warnings = settings.save(form, local_path=path, only=only)
        else:
            season = resolve_target(target_text).season
            path = season_settings.season_file(season.results)
            result["created"] = not path.is_file()
            season_settings.save_own(form, season.results, base=season_settings.global_config(), only=only)
    except settings.SettingsError as exc:
        raise CliError("bad_value", "; ".join(exc.problems), problems=exc.problems) from exc
    result = {"scope": args.scope, "file": str(path), "saved": saved, "warnings": warnings, **result}
    if args.scope == "season" and result["created"]:
        result["note"] = ("У сезона теперь свой файл настроек: в нём копия всех настроек, и сезон больше не следует "
                          "за последующими изменениями общих настроек.")
    return result


def cmd_forget(args: argparse.Namespace) -> dict[str, Any]:
    season = resolve_target(args.target).season
    return {"removed": season_settings.remove_own(season.results),
            "file": str(season_settings.season_file(season.results))}


# ---------------------------------------------------------------- launch

def select_episodes(target: Target, tokens: list[str], everything: bool) -> list[str]:
    """Video file names to run: the named ones (file names or codes), all, the target file, else the unfinished."""
    rows = seasons.episode_rows(target.season)
    if tokens:
        chosen: list[str] = []
        for token in tokens:
            code = seasons._CODE.fullmatch(token.strip())
            hits = [r.video for r in rows if token in (r.video, r.stem)]
            if code:
                hits = hits or [r.video for r in rows if (m := seasons._CODE.search(r.stem))
                                and (int(m.group(1)), int(m.group(2))) == (int(code.group(1)), int(code.group(2)))]
            if not hits:
                raise CliError("bad_episode", f"В сезоне нет серии «{token}».", available=[r.video for r in rows])
            chosen += [h for h in hits if h not in chosen]
        return chosen
    if everything:
        return [r.video for r in rows]
    if target.episode:
        return [target.episode]
    return [r.video for r in rows if r.finished is None]


def build_config(season: seasons.Season, preset: str | None, transport: str | None,
                 overrides: list[tuple[str, Any]]) -> dict[str, Any]:
    """The config of a job; mirrors ``dub_app.start_job`` + ``dub_app.job_config`` (without the form's options)."""
    base = season_settings.effective(season_settings.global_config(), season.results)
    try:
        config = apply_preset(copy.deepcopy(base), preset)
    except SystemExit as exc:
        raise CliError("bad_preset", str(exc), presets=list((base.get("presets") or {}))) from exc
    config = copy.deepcopy(config)
    if transport == "openrouter":  # pay per request instead of the subscription (TRANSPORTS in dub_app)
        config.setdefault("llm", {}).update(opencode=[], router="openrouter")
    for key, value in overrides:
        settings.put(config, tuple(key.split(".")), value)
    return seasons.season_config(config, season)


def systemd_env() -> dict[str, str]:
    """The environment a detached job needs: PATH (with the opencode folder), HOME and the GPU / cache variables."""
    path = os.environ.get("PATH", "")
    opencode = Path.home() / ".opencode" / "bin"
    if opencode.is_dir() and str(opencode) not in path.split(":"):
        path = f"{path}:{opencode}" if path else str(opencode)
    env = {"PATH": path, "HOME": str(Path.home())}
    env.update({k: os.environ[k] for k in ("HF_HOME", "CUDA_VISIBLE_DEVICES") if os.environ.get(k)})
    return env


def _wait_pid(unit: str, seconds: float = 5.0) -> int | None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit], capture_output=True,
                             text=True, check=False)
        pid = int(out.stdout.strip() or 0) if out.stdout.strip().isdigit() else 0
        if pid and "stt.run" in _cmdline(pid):
            return pid
        time.sleep(0.25)
    return None


def start_process(cmd: list[str], log_path: Path, unit: str, wait_for: str = "stt.run") -> dict[str, Any]:
    """Start ``cmd`` detached from this session: a systemd user unit, else a new session (``detached``: "systemd" |
    "setsid"). Returns ``{"pid", "detached", "warnings"}``; tests replace this function."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = systemd_env()
    if shutil.which("systemd-run"):
        wrapper = ["bash", "-c", 'log=$1; shift; exec "$@" >>"$log" 2>&1', "bash", str(log_path), *cmd]
        run = ["systemd-run", "--user", f"--unit={unit}", "--collect", f"--working-directory={REPO_ROOT}",
               *[f"--setenv={k}={v}" for k, v in env.items()], *wrapper]
        done = subprocess.run(run, capture_output=True, text=True, check=False)
        if done.returncode == 0:
            pid = _wait_pid(unit) if wait_for == "stt.run" else None
            if pid is None and wait_for != "stt.run":  # no marker to look for: take the unit's main process
                out = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                                     capture_output=True, text=True, check=False)
                pid = int(out.stdout.strip()) if out.stdout.strip().isdigit() and int(out.stdout.strip()) else None
            warnings = [] if pid else ["Процесс запущен, но его pid пока не найден: проверьте статус командой status."]
            return {"pid": pid, "detached": "systemd", "warnings": warnings}
        log.warning("systemd-run failed: %s", done.stderr.strip())
    with log_path.open("ab") as handle:
        process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, **env})
    return {"pid": process.pid, "detached": "setsid",
            "warnings": ["Запущено без systemd: задача может остановиться, если сессия агента перезапустится."]}


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower() or "season"


def register_job(path: Path, data: dict[str, Any]) -> None:
    """Write the job in the format of ``dub_app.save_job`` so an open UI finds it (``dub_app.adopt_external_job``)."""
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def cmd_launch(args: argparse.Namespace) -> dict[str, Any]:
    target = resolve_target(args.target)
    season = target.season
    names = select_episodes(target, args.episodes or [], args.all)
    if not names:
        raise CliError("no_episodes", "Нечего запускать: все серии уже готовы. Укажите --episodes или --all.")
    overrides = _split_pairs([item for group in args.set or [] for item in group])
    config = build_config(season, args.preset, args.transport, overrides)
    # without --engines: the engines ticked by default after the preset (``fast`` keeps only Vosk)
    engines = [e.strip() for e in (args.engines or "").split(",") if e.strip()] or _launch_defaults(config)["engines"]
    bad = [e for e in engines if e not in LAUNCH_ENGINES]
    if bad:
        raise CliError("bad_engine", f"Неизвестный движок: {', '.join(bad)}. Допустимо: {', '.join(LAUNCH_ENGINES)}.")
    engines = [e for e in LAUNCH_ENGINES if e in engines]
    root = season.results
    extra: list[str] = []
    preview = None
    if args.preview_minutes:
        root = root / "_проба"  # as start_job: a short piece in its own folder
        start = args.preview_from or 0.0
        extra += ["--start", str(start * 60), "--duration", str(args.preview_minutes * 60)]
        config["paths"] = {**(config.get("paths") or {}), "season": None, "finished": None}
        preview = {"from_minutes": start, "minutes": args.preview_minutes, "folder": str(root)}
    if args.no_season_voices:
        extra.append("--no-season-voices")
    if args.prepare:
        extra.append("--prepare")
    warnings: list[str] = []
    busy = stt_run_processes()
    if busy:
        pids = [p["pid"] for p in busy]
        if not args.dry_run:
            raise CliError("busy", f"Уже идёт обработка (stt.run, pid {', '.join(map(str, pids))}): дождитесь или остановите её.",
                           exit_code=3, pids=pids)
        warnings.append(f"Сейчас идёт обработка (stt.run, pid {', '.join(map(str, pids))}): настоящий запуск будет отклонён.")
    files = [season.source / n for n in names]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ui_dir = root / ".ui"
    config_path, log_path = ui_dir / f"config_{stamp}.json", ui_dir / f"job_{stamp}.log"
    unit = f"{args.unit_prefix}-{_slug(season.name)}-{stamp}"
    cmd = [sys.executable, "-m", "stt.run", *map(str, files), "--out-root", str(root), "--config", str(config_path),
           "--engine", ",".join(engines), *extra]
    standard = settings.read_json(settings.DEFAULTS)
    changed: dict[str, Any] = {}
    for key, setting in BY_KEY.items():
        value = settings.get(config, setting.path)
        if value != settings.get(standard, setting.path):
            changed[key] = value
    changed.update({key: value for key, value in overrides})
    show = _show_of(config, season)
    result: dict[str, Any] = {
        "started": False, "dry_run": bool(args.dry_run), "unit": unit, "pid": None, "detached": None,
        "episodes": [seasons.episode_label(n) for n in names], "files": [str(f) for f in files], "engines": engines,
        "out_root": str(root), "config": str(config_path), "log": str(log_path), "command": cmd, "preview": preview,
        "preset": args.preset, "show": show, "changed_from_standard": changed,
        "stop": f"systemctl --user stop {unit}", "warnings": warnings}
    if args.dry_run:
        return result
    ui_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    started = start_process(cmd, log_path, unit)
    result.update(started=True, pid=started["pid"], detached=started["detached"])
    result["warnings"] = warnings + list(started.get("warnings") or [])
    if started["detached"] != "systemd":
        result["stop"] = f"kill {started['pid']}" if started["pid"] else "kill <pid stt.run>"
    register_job(LAST_JOB, {
        "files": [str(f) for f in files], "out_root": str(root), "engines": engines, "config_path": str(config_path),
        "log_path": str(log_path), "extra": extra, "pid": started["pid"], "boot": stt_progress.boot_id(),
        "started": time.time(), "finished": None, "state": "идёт", "title": JOB_TITLE, "cmd": None, "result": ""})
    return result


# ---------------------------------------------------------------- ui

def cmd_ui(args: argparse.Namespace) -> dict[str, Any]:
    url = ui_url(args.port)
    if ui_running(args.port):
        out: dict[str, Any] = {"running": True, "started": False, "url": url}
    else:
        started = start_process([sys.executable, "-m", "ui.dub_app", "--port", str(args.port)], UI_LOG, "dub-ui",
                                wait_for="")
        deadline = time.time() + 90
        while time.time() < deadline and not ui_running(args.port):
            time.sleep(1)
        out = {"running": ui_running(args.port), "started": True, "url": url, "unit": "dub-ui",
               "detached": started["detached"], "warnings": started.get("warnings") or []}
    if args.open and out["running"] and shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return out


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="ui.run_setup", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    sub.add_parser("seasons").set_defaults(func=cmd_seasons)
    for name, func in (("show", cmd_show), ("status", cmd_status), ("forget-season", cmd_forget)):
        p = sub.add_parser(name)
        p.add_argument("target", nargs="?")
        p.set_defaults(func=func)
    p = sub.add_parser("save")
    p.add_argument("--scope", choices=["global", "season"], required=True)
    p.add_argument("items", nargs="*", help="[TARGET] KEY=VALUE ...")
    p.set_defaults(func=cmd_save)
    p = sub.add_parser("launch")
    p.add_argument("target", nargs="?")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--all", action="store_true")
    p.add_argument("--engines")
    p.add_argument("--preset")
    p.add_argument("--transport", choices=["subscription", "openrouter"])
    p.add_argument("--set", nargs="+", action="append", metavar="KEY=VALUE")
    p.add_argument("--preview-from", type=float, default=None, help="minutes")
    p.add_argument("--preview-minutes", type=float, default=None)
    p.add_argument("--no-season-voices", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--unit-prefix", default="dub")
    p.set_defaults(func=cmd_launch)
    p = sub.add_parser("ui")
    p.add_argument("--port", type=int, default=UI_PORT)
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        out = args.func(args)
    except CliError as error:
        out = {"error": error.code, "message": error.message, **error.extra}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return error.exit_code
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
