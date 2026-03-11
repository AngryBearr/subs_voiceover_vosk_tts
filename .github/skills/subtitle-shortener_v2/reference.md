# Subtitle Shortener Reference

Scripts in `.github/skills/subtitle-shortener_v2/scripts/` are self-contained and import only sibling files.

## Chunk Format

`prepare_shortening_chunks.py` writes `chunk_*.json` with:

```json
{
  "schema": "subtitle-shortening-chunk/v1",
  "iteration": 1,
  "chunk_id": 1,
  "target_count": 2,
  "context_count": 3,
  "indices": [101, 102, 103],
  "target_indices": [102, 103],
  "soft_joint_groups": [
    {
      "target_positions": [5, 6],
      "item_positions": [3, 4, 5, 6, 7],
      "hint": "These target items likely form one thought. Prefer a soft joint rewrite before splitting back."
    }
  ],
  "items": [
    {
      "index": 102,
      "position": 5,
      "role": "target",
      "text": ["Original subtitle line"],
      "effective_ratio": 1.82,
      "target_ratio": 1.47,
      "edit_rule": "Rewrite this item only if needed to reach target_ratio."
    }
  ]
}
```

## Result Format

Result files must be strict JSON objects:

```json
{
  "chunk_id": 1,
  "edits": [
    {
      "index": 102,
      "text": ["Shorter subtitle line"]
    }
  ]
}
```

Rules:

- Edit only `target` items.
- Omit unchanged targets.
- Keep `index` stable.
- Keep `text` as `list[str]` in results.

## Basic Flow

1. `analyze_text.py` builds or refreshes `analysis` fields.
2. `prepare_shortening_chunks.py` selects targets using the active threshold and writes per-iteration chunks.
3. `iterative_subtitle_shortening.py` prepares chunks, waits for `*.result.json`, merges edits, reanalyzes, and repeats.
4. `report_shortening_progress.py` reads analyzed JSON, manifests, iteration reports, or final reports.
