---
name: subtitle-shortener_v2
description: Iteratively shorten subtitles or subtitles JSON with semantic shortening, chunking, cross-chunk context, length thresholds, soft joint rewrites, subagents for many chunks, and temporary workdir cleanup. Use when reducing subtitle text until no segment remains above the target threshold.
---

# Subtitle Shortener

Iteratively reduce subtitle text while preserving meaning, timing structure, and local coherence.

This skill is self-contained. Use only files inside this skill directory.

## Files

- `scripts/analyze_text.py` - analyze subtitle timing and text fit.
- `scripts/prepare_shortening_chunks.py` - build iteration manifests and chunk files.
- `scripts/merge_shortening_results.py` - merge strict edit JSON back into subtitle JSON.
- `scripts/reanalyze_shortened_json.py` - refresh analysis after edits.
- `scripts/iterative_subtitle_shortening.py` - run the full iterative workflow.
- `scripts/report_shortening_progress.py` - inspect analyzed JSON or workflow progress.
- `reference.md` - chunk and result JSON formats.

## Instructions

1. Work only with the self-contained scripts in this skill. Do not import or call project subtitle helpers outside this directory.
2. Start from analyzed subtitle JSON, or first run `scripts/analyze_text.py` if analysis is missing or stale.
3. Create a temporary work directory named with the current date and time for manifests, chunks, result files, reports, and partial outputs.
4. Build chunks with `scripts/prepare_shortening_chunks.py`. Use the active threshold, context halo, and soft joint rewrite hints from the chunk payload.
5. Process chunks iteratively. Shorten semantically, not mechanically: preserve meaning, speaker intent, references, and flow; remove redundancy, filler, and wording that does not change meaning.
6. When several adjacent target segments form one thought, use a soft joint rewrite inside that cluster before splitting back into per-segment output.
7. When there are many chunks, use subagents in parallel. Each subagent should read one chunk and return only strict result JSON.
8. Merge edits with `scripts/merge_shortening_results.py`, re-run analysis with `scripts/reanalyze_shortened_json.py`, and continue until no segment remains above the threshold. Do not stop early.
9. Save the final cleaned JSON and final report outside the temp workdir, then delete the temp workdir.

## Output Rules

- Keep the original segment order and structure unless the task explicitly allows schema changes.
- Preserve JSON validity and required fields.
- Prefer the smallest edit that keeps the meaning intact.
- If a segment is already at or below threshold, leave it unchanged unless needed for a soft joint rewrite with neighbors.
- Result files must stay strict and minimal: `{"chunk_id": <int>, "edits": [{"index": <int>, "text": ["..."]}]}`.
- Task completion means every segment is at or below the threshold.

## Quick Start

```bash
python scripts/iterative_subtitle_shortening.py input.json --output final.json --final-report final_report.json --max-mismatch-ratio 1.5 --cleanup-temp
```
