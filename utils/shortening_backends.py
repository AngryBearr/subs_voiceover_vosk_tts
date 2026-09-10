"""Live transports for the shortening runner.

The runner remains transport-agnostic; this module is the only live adapter.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from utils.deepseek_limits import resolve_deepseek_token_limit
from utils.llm_types import DetailedResponse
from utils.opencode_clean import CleanOpenCodeServer
from utils.shortening_runner import ModelSpec


OPENAI_MODELS: dict[str, set[str]] = {
    "openai/gpt-5.6-luna": {"none", "low", "medium", "high"},
    "openai/gpt-5.6-sol": {"medium"},
}
DEEPSEEK_VARIANTS = {"off", "low", "high", "max"}


def transport_runtime_metadata() -> dict[str, str]:
    """Return safe runtime provenance for a live transport invocation."""
    try:
        import aiohttp
    except ModuleNotFoundError:
        raise RuntimeError("missing_transport_dependency:aiohttp") from None
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "active_virtual_env": virtual_env or sys.prefix,
        "aiohttp_version": str(aiohttp.__version__),
        "virtual_env": virtual_env,
    }


class LiveBackend:
    """Dispatch DeepSeek directly and OpenCode through isolated sessions."""

    def __init__(self, server: CleanOpenCodeServer, *, deepseek_api_key: str | None = None,
                 timeout: float = 300.0, max_tokens: int | None = None,
                 deepseek_base_url: str = "https://api.deepseek.com") -> None:
        self.runtime_metadata = transport_runtime_metadata()
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.server = server
        self.deepseek_api_key = deepseek_api_key
        self.timeout = float(timeout)
        self.max_tokens = max_tokens
        self.deepseek_base_url = deepseek_base_url

    async def __call__(self, spec: ModelSpec, system_prompt: str, user_prompt: str,
                       output_schema: dict[str, Any], allow_length_tool: bool) -> DetailedResponse:
        if spec.provider == "deepseek":
            return await self._deepseek(spec, system_prompt, user_prompt, output_schema, allow_length_tool)
        if spec.provider == "opencode":
            if spec.model not in OPENAI_MODELS or spec.variant not in OPENAI_MODELS[spec.model]:
                return DetailedResponse(None, [], "unsupported_openai_model_variant")
            return await self._opencode(spec, system_prompt, user_prompt, output_schema, allow_length_tool)
        return DetailedResponse(None, [], "unsupported_provider")

    async def _deepseek(self, spec: ModelSpec, system_prompt: str, user_prompt: str,
                        output_schema: dict[str, Any], allow_length_tool: bool) -> DetailedResponse:
        if spec.model not in {"deepseek-flash", "deepseek-v4-flash"}:
            return DetailedResponse(None, [], "deepseek_model_required")
        if spec.variant not in DEEPSEEK_VARIANTS:
            return DetailedResponse(None, [], "deepseek_variant_required")
        if not self.deepseek_api_key:
            return DetailedResponse(None, [], "deepseek_api_key_required")
        from utils.deepseek_transport import request_json_detailed
        limit = resolve_deepseek_token_limit(spec.variant, self.max_tokens)
        return await request_json_detailed(
            api_key=self.deepseek_api_key, model=spec.model, system_prompt=system_prompt,
            user_prompt=user_prompt, base_url=self.deepseek_base_url, timeout=self.timeout,
            reasoning_effort=spec.variant, max_tokens=limit, allow_length_tool=allow_length_tool,
        )

    async def _opencode(self, spec: ModelSpec, system_prompt: str, user_prompt: str,
                        output_schema: dict[str, Any], allow_length_tool: bool) -> DetailedResponse:
        session_id: str | None = None
        try:
            from utils import opencode_transport
            # A neutral title prevents model identity and experiment metadata from
            # entering the session conversation or its server-side title.
            session_id = await opencode_transport.create_session(
                self.server.base_url, title="subtitle-review", auth_header=self.server.auth_header
            )
            return await opencode_transport.send_prompt_detailed(
                self.server.base_url, session_id, system_prompt, user_prompt,
                spec.model, spec.variant, auth_header=self.server.auth_header,
                output_schema=output_schema, allow_length_tool=allow_length_tool,
                timeout=self.timeout,
            )
        except ModuleNotFoundError as exc:
            module_name = exc.name
            if isinstance(module_name, str) and module_name and all(
                part.isidentifier() for part in module_name.split(".")
            ):
                return DetailedResponse(None, [], f"ModuleNotFoundError:{module_name}")
            return DetailedResponse(None, [], type(exc).__name__)
        except Exception as exc:
            return DetailedResponse(None, [], type(exc).__name__)
        finally:
            if session_id is not None:
                try:
                    await opencode_transport.delete_session(
                        self.server.base_url, session_id, auth_header=self.server.auth_header
                    )
                except Exception:
                    pass
