"""Pure configuration overlays shared by the pipeline and the UI."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries; replace lists and scalars without sharing mutable values."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = (deep_merge(out[key], value)
                    if isinstance(value, dict) and isinstance(out.get(key), dict)
                    else copy.deepcopy(value))
    return out


def _leaves(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted leaf paths; lists and empty dictionaries are atomic values."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            out.update(_leaves(value, path))
        else:
            out[path] = value
    return out


def resolve_config_layers(
    layers: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply layers in caller order and attribute each final leaf to its last writer."""
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, layer in layers:
        config = deep_merge(config, layer)
        leaves = _leaves(config)
        sources = {path: source for path, source in sources.items() if path in leaves}
        sources.update({path: name for path in _leaves(layer) if path in leaves})
    return config, sources


def read_config_layers(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read the base JSON and its optional personal sibling, never environment or auth files."""
    layers = [("base", json.loads(path.read_text(encoding="utf-8")))]
    local = path.with_name(path.stem + ".local.json")
    if local.exists():
        layers.append(("local", json.loads(local.read_text(encoding="utf-8"))))
    return layers


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config with ``<name>.local.json`` on top."""
    return resolve_config_layers(read_config_layers(path))[0]


def apply_preset(config: dict[str, Any], name: str | None) -> dict[str, Any]:
    """Overlay a named preset; no name returns the input unchanged."""
    if not name:
        return config
    presets = config.get("presets") or {}
    if name not in presets:
        raise SystemExit(f"Unknown preset {name!r}; the config has: {', '.join(presets) or 'none'}")
    return deep_merge(config, presets[name].get("settings") or {})


def show_of(config: dict[str, Any], media: Path, forced: str | None = None) -> str | None:
    """Find the first show whose words occur in the media path; forced ``none`` disables it."""
    shows = config.get("shows") or {}
    if forced:
        if forced.lower() == "none":
            return None
        if forced not in shows:
            raise SystemExit(f"Unknown show {forced!r}; the config has: {', '.join(shows) or 'none'}")
        return forced
    from stt.subtitles import path_words

    path = path_words(media)
    return next((key for key in shows if path_words(key) in path), None)


def apply_show(config: dict[str, Any], name: str | None) -> dict[str, Any]:
    """Overlay the show's settings on an independent copy; no name returns the input."""
    if not name:
        return config
    return deep_merge(config, (config.get("shows") or {})[name].get("settings") or {})


def effective_voiceover_policy(stt_cfg: dict[str, Any], vo_cfg: dict[str, Any]) -> dict[str, Any]:
    """Voice-over timing policy with the STT run's policy taking precedence."""
    return deep_merge(vo_cfg.get("policy") or {}, (stt_cfg.get("voiceover") or {}).get("policy") or {})


def format_config_sources(config: dict[str, Any], sources: dict[str, str]) -> str:
    """Format diagnostic JSON, redacting credential values while keeping environment variable names."""
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ("<redacted>" if sensitive(key) else redact(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    def sensitive(key: str) -> bool:
        name = key.lower().replace("-", "_")
        return not name.endswith("_env") and (
            name in {"auth", "authorization", "credentials", "keys"}
            or any(part in name for part in ("api_key", "apikey", "token", "password", "secret"))
            and name not in {"max_tokens", "max_output_tokens"}
        )

    return json.dumps({"config": redact(config), "sources": dict(sorted(sources.items()))},
                      ensure_ascii=False, indent=2)
