from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
# Ensure repo root in path so utils can be imported
from .compress_critical_segments import load_items, collect_critical_segments, build_agent_payload

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: run_collect.py path')
        sys.exit(1)
    path = Path(sys.argv[1])
    items = load_items(path)
    segments = collect_critical_segments(items, window=3, min_combined_ratio=1.5)
    payload = build_agent_payload(segments)
    print(payload)
