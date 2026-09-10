"""FastMCP stdio server exposing only the pure length-check function."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from utils.length_check import check_length as pure_check_length

mcp = FastMCP("length")


@mcp.tool()
def check_length(text: str, max_chars: int) -> dict[str, int | bool]:
    """Check a string's Python code-point length against a character budget."""
    return pure_check_length(text, max_chars)


if __name__ == "__main__":
    mcp.run(transport="stdio")
