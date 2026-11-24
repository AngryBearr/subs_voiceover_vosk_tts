"""The application's help tab."""

from pathlib import Path

import gradio as gr


def build(path: Path) -> None:
    """Show the existing help text."""
    with gr.Tab("Справка"):
        gr.Markdown(path.read_text(encoding="utf-8") if path.exists() else "Справка не найдена.")
