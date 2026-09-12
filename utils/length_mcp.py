"""FastMCP stdio server exposing only the pure length-check function."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from utils.length_check import LengthBasis, check_length as pure_check_length

mcp = FastMCP("length")


@mcp.tool()
def check_length(
    text: str, max_chars: int, length_basis: LengthBasis = "display_text"
) -> dict[str, int | bool]:
    """Check a string's code-point length using the selected character basis."""
    return pure_check_length(text, max_chars, length_basis)


if __name__ == "__main__":
    mcp.run(transport="stdio")
