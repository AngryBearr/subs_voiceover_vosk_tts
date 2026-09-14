"""Pure SH08 model and accounting specifications.

This module is intentionally limited to immutable-in-practice data and scalar
constants.  It does not import or perform transport, filesystem, or network
work.
"""

from __future__ import annotations

from typing import Any


OPENCODE_REQUIRED_VERSION = "1.18.29"
MAX_TOKENS = 16384


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "luna": {
        "provider": "opencode", "model": "openai/gpt-5.6-luna", "variant": "medium",
        "input_per_million": 0.0, "output_per_million": 0.0,
        "context_limit": 1048576, "max_tokens": MAX_TOKENS, "quota_units": 6, "subscription": True,
    },
    "deepseek": {
        "provider": "deepseek", "model": "deepseek-flash", "variant": "off",
        "input_per_million": 0.30, "output_per_million": 1.20,
        "context_limit": 1048576, "max_tokens": MAX_TOKENS, "quota_units": 1, "subscription": False,
    },
    "sol": {
        "provider": "opencode", "model": "openai/gpt-5.6-sol", "variant": "medium",
        "input_per_million": 0.0, "output_per_million": 0.0,
        "context_limit": 1048576, "max_tokens": MAX_TOKENS, "quota_units": 6, "subscription": True,
    },
    "gemini": {
        "provider": "openrouter", "model": "google/gemini-3.8-flash", "variant": "low",
        "endpoint_tag": "google-ai-studio/flex", "input_per_million": 0.375, "output_per_million": 1.875,
        "context_limit": 1048576, "max_tokens": MAX_TOKENS, "quota_units": 1, "subscription": False,
    },
    "muse": {
        "provider": "openrouter", "model": "meta/muse-spark-1.3-contributor", "variant": "low",
        "endpoint_tag": "meta", "reasoning_mandatory": True,
        "input_per_million": 0.10, "output_per_million": 0.20,
        "context_limit": 1048576, "max_tokens": MAX_TOKENS, "quota_units": 1, "subscription": False,
    },
}
