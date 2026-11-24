"""Small component and feedback helpers shared by the page's tabs."""

from __future__ import annotations

from typing import Any

import gradio as gr


def wait_state(label: str, note: str) -> tuple[Any, str]:
    """First step of a medium action (a few seconds to a minute): the button waits, a line says what happens."""
    return gr.update(value=f"⏳ {label}", interactive=False), f"⏳ {note}"


def ready_button(label: str) -> Any:
    return gr.update(value=label, interactive=True)
