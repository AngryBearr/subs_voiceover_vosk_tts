from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SRTIssue:
    block_no: int
    message: str
    snippet: str


class SRTParseError(ValueError):
    """Raised when the SRT file cannot be parsed without losing entries."""

    def __init__(self, *, path: str, issues: List[SRTIssue]):
        self.path = path
        self.issues = issues
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines: List[str] = [f"SRT parse failed for: {self.path}", "Problems:"]
        for issue in self.issues:
            snippet = issue.snippet.replace("\n", "\\n")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            lines.append(f"- Block #{issue.block_no}: {issue.message}. Snippet: '{snippet}'")
        return "\n".join(lines)


_TIME_RE = re.compile(r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[\.,](?P<ms>\d{1,3}))?")


def parse_time_to_ms(time_str: str) -> int:
    """Convert SRT time to milliseconds.

    Supports common variants:
    - HH:MM:SS,mmm
    - HH:MM:SS.mmm
    - MM:SS,mmm (hours omitted)
    - HH:MM:SS (milliseconds omitted)
    """

    t = time_str.strip().lstrip("\ufeff")
    m = _TIME_RE.fullmatch(t)
    if not m:
        raise ValueError(f"Invalid time format: {time_str!r}")

    hours = int(m.group("h") or 0)
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    ms_raw = m.group("ms") or "0"
    ms = int(ms_raw.ljust(3, "0")[:3])
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + ms


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_blocks(content: str) -> List[str]:
    # Split on 2+ newlines, preserving content inside blocks.
    blocks = re.split(r"\n{2,}", content.strip())
    return [b for b in blocks if b.strip()]


def _extract_times(timestamp_line: str) -> Tuple[str, str]:
    # Be tolerant to arrow variations and extra spaces.
    # Pick the first two time-like patterns.
    matches = list(_TIME_RE.finditer(timestamp_line))
    if len(matches) < 2:
        raise ValueError(f"Invalid timestamp line: {timestamp_line!r}")
    return matches[0].group(0), matches[1].group(0)


def _safe_snippet(block: str) -> str:
    lines = _normalize_newlines(block).split("\n")
    return "\n".join(lines[:6]).strip()


def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """Parse SRT subtitle file and convert to JSON items.

    Important behavior:
    - Never silently drops blocks.
    - Attempts best-effort recovery for minor issues (e.g., BOM before index).
    - If a block cannot be parsed (e.g., missing/invalid timestamps), raises SRTParseError
      with detailed diagnostics so the pipeline can stop and show the user what went wrong.
    """

    path = Path(srt_path)
    if not path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    # utf-8-sig strips BOM (common cause of the first index failing to parse)
    content = path.read_text(encoding="utf-8-sig")
    content = _normalize_newlines(content)

    subtitle_blocks = _split_blocks(content)
    result: List[Dict[str, Any]] = []
    fatal_issues: List[SRTIssue] = []
    recoverable_issues: List[SRTIssue] = []

    for block_no, block in enumerate(subtitle_blocks, start=1):
        lines = [ln.strip().lstrip("\ufeff") for ln in block.split("\n")]
        lines = [ln for ln in lines if ln != ""]
        snippet = _safe_snippet(block)

        if len(lines) < 2:
            fatal_issues.append(
                SRTIssue(
                    block_no=block_no,
                    message="Block is too short (expected at least index + timestamps)",
                    snippet=snippet,
                )
            )
            continue

        raw_index = lines[0]
        index: Optional[int] = None
        try:
            index = int(raw_index)
        except ValueError:
            m = re.search(r"\d+", raw_index)
            if m:
                index = int(m.group(0))
                recoverable_issues.append(
                    SRTIssue(
                        block_no=block_no,
                        message=f"Index was not a чистым числом ({raw_index!r}); extracted {index}",
                        snippet=snippet,
                    )
                )
            else:
                # Fall back to sequential index (still not silent).
                index = len(result) + 1
                recoverable_issues.append(
                    SRTIssue(
                        block_no=block_no,
                        message=f"Index missing/unparseable ({raw_index!r}); using sequential index {index}",
                        snippet=snippet,
                    )
                )

        timestamp_line = lines[1]
        try:
            start_time, end_time = _extract_times(timestamp_line)
            start_ms = parse_time_to_ms(start_time)
            end_ms = parse_time_to_ms(end_time)
        except ValueError as e:
            fatal_issues.append(
                SRTIssue(
                    block_no=block_no,
                    message=str(e),
                    snippet=snippet,
                )
            )
            continue

        if end_ms < start_ms:
            # Keep as fatal: otherwise duration becomes negative and breaks downstream.
            fatal_issues.append(
                SRTIssue(
                    block_no=block_no,
                    message=f"End time is before start time ({start_ms} > {end_ms})",
                    snippet=snippet,
                )
            )
            continue

        duration = end_ms - start_ms

        text_lines = lines[2:]
        text_array = [line.strip() for line in text_lines if line.strip()]
        if not text_array:
            # Downstream expects list[str]; keep a placeholder to avoid schema break.
            text_array = [""]
            recoverable_issues.append(
                SRTIssue(
                    block_no=block_no,
                    message="Empty subtitle text; replaced with empty string",
                    snippet=snippet,
                )
            )

        entry: Dict[str, Any] = {
            "text": text_array,
            "index": int(index),
            "start": int(start_ms),
            "end": int(end_ms),
            "duration": int(duration),
            "gender": "male",
        }
        result.append(entry)

    if recoverable_issues:
        logging.warning("SRT parsed with %d recoverable issues (no blocks dropped).", len(recoverable_issues))
        for issue in recoverable_issues[:50]:
            logging.warning("%s", issue.message)
        if len(recoverable_issues) > 50:
            logging.warning("... and %d more recoverable issues", len(recoverable_issues) - 50)

    if fatal_issues:
        raise SRTParseError(path=srt_path, issues=fatal_issues)

    return result


def srt_to_json(srt_path: str, output_path: str = None) -> Optional[str]:
    """
    Convert SRT file to JSON format and optionally save to file.
    
    Args:
        srt_path (str): Path to the SRT file
        output_path (str, optional): Path to save the JSON output. If None, returns JSON string
        
    Returns:
        JSON string if output_path is None, otherwise None
    """
    # Parse the SRT file
    subtitles = parse_srt_file(srt_path)
    
    # Convert to JSON
    json_output = json.dumps(subtitles, ensure_ascii=False, indent=4)
    
    # Save to file if output path is provided
    if output_path:
        Path(output_path).write_text(json_output, encoding="utf-8")
        logging.info("JSON output saved to: %s", output_path)
        return None

    return json_output


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SRT subtitle file to JSON format')
    parser.add_argument('srt_file', help='Path to the SRT file')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="[%(levelname)s] %(message)s")

    try:
        result = srt_to_json(args.srt_file, args.output)
    except SRTParseError as e:
        logging.error("%s", str(e))
        raise SystemExit(1)

    if result:
        print(result)


if __name__ == "__main__":
    main()
