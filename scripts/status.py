"""Status line of the long jobs of this project: stage, percent done, elapsed and remaining time.

    subs_env/bin/python -I scripts/status.py            # print once
    subs_env/bin/python -I scripts/status.py --loop 60  # rewrite output/STATUS.md every 60 s (run with nohup)

Jobs it knows: DiariZen fine-tuning runs (output/exp_finetune/run*/train.log), the training-data build
(output/subs_train/build_audio.log), the word times (output/subs_train/build_words.log), the subtitle extraction
(output/subs_train/extract.log), the speaker model runs (output/exp_spk/*/train.log, output/exp_spk/run_*.log), the
experiment harness runs (output/exp_boundary/run_*.log, output/exp_tse/run_*.log). A job is "running" while its log changed in the last 3 minutes.
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "output"  # the logs and STATUS.md live under output/
STATUS = HERE / "STATUS.md"
SEEN = HERE / ".status_seen.json"  # log path -> the time this script first saw it (Linux has no creation time)
STALE_S = 180
_STAMP = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")


def started_at(log: Path, text: str | None = None) -> float:
    """Start time of a job: the first timestamp written in its log, else when this script first saw the log."""
    import json

    seen = json.loads(SEEN.read_text(encoding="utf-8")) if SEEN.exists() else {}
    key = str(log)
    if key not in seen:
        seen[key] = log.stat().st_mtime if not text else min(log.stat().st_mtime, time.time())
        SEEN.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    if text:
        m = _STAMP.search(text)
        if m:
            year = datetime.now().year
            stamp = datetime(year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
            return min(stamp.timestamp(), seen[key])
    return seen[key]


def _fmt(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def _alive(path: Path) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < STALE_S


def training(log: Path) -> dict | None:
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    steps = re.search(r"`steps_per_epoch`: (\d+)", text)
    epochs = re.search(r"`max_epochs`: (\d+)", text)
    if not steps or not epochs:
        return None
    per_epoch, max_epochs = int(steps.group(1)), int(epochs.group(1))
    vals = re.findall(r"Validation Loss/DER on epoch (\d+): ([\d.]+) / ([\d.]+)", text)
    done_epochs = len(vals)
    bars = re.findall(r"(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+), +([\d.]+)it/s\]", text)
    step, rate = (int(bars[-1][0]), float(bars[-1][4])) if bars else (0, None)
    total = per_epoch * max_epochs
    done = done_epochs * per_epoch + (step if step < per_epoch else 0)
    frac = min(done / total, 1.0) if total else 0.0
    t0 = started_at(log, text)
    started = datetime.fromtimestamp(t0)
    elapsed = (log.stat().st_mtime if finished_now(vals, max_epochs) else time.time()) - t0
    remaining = (total - done) / rate if rate else (elapsed / frac - elapsed if frac > 0.02 else None)
    finished = done_epochs >= max_epochs
    last_val = f"loss {vals[-1][1]} / DER {vals[-1][2]} after epoch {vals[-1][0]}" if vals else "no validation yet"
    return {"job": f"fine-tuning {log.parent.name}", "stage": ("finished" if finished else
            f"epoch {min(done_epochs + 1, max_epochs)}/{max_epochs}, step {step}/{per_epoch}"),
            "percent": round(100 * frac, 1), "elapsed": _fmt(elapsed), "remaining": "0" if finished else _fmt(remaining),
            "note": last_val, "running": _alive(log) and not finished, "started": started.strftime("%H:%M")}


def finished_now(vals: list, max_epochs: int) -> bool:
    return len(vals) >= max_epochs


def line_counter(log: Path, job: str, done_pattern: str, total: int | None, stage: str) -> dict | None:
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    done = len(re.findall(done_pattern, text, re.M))
    finished = bool(re.search(r"BUILD DONE|ALL DONE|LOC DONE|EMBED DONE|WORDS DONE|TRAIN DONE|LISTEN DONE", text)) or (
        job == "subtitle extraction" and (HERE / "subs_train/counts.json").exists()
        and (HERE / "subs_train/counts.json").stat().st_mtime >= log.stat().st_mtime)
    t0 = started_at(log, text)
    elapsed = (log.stat().st_mtime if finished else time.time()) - t0
    frac = (done / total) if total else None
    remaining = None if finished or not frac or frac == 0 else elapsed / frac - elapsed
    return {"job": job, "stage": "finished" if finished else stage, "percent": round(100 * frac, 1) if frac is not None else None,
            "elapsed": _fmt(elapsed), "remaining": "0" if finished else _fmt(remaining),
            "note": f"{done}" + (f" of {total}" if total else "") + " done", "running": _alive(log) and not finished,
            "started": datetime.fromtimestamp(t0).strftime("%H:%M")}


def spk_training(log: Path) -> dict | None:
    """Speaker model run: lines "epoch E/N step S/M loss L ..." and "val epoch E ..." (output/exp_spk/train.py)."""
    text = log.read_text(encoding="utf-8", errors="replace")
    bars = re.findall(r"epoch (\d+)/(\d+) step (\d+)/(\d+)", text)
    if not bars:
        return None
    epoch, epochs, step, steps = (int(x) for x in bars[-1])
    frac = ((epoch - 1) + step / max(steps, 1)) / max(epochs, 1)
    finished = "TRAIN DONE" in text
    t0 = started_at(log, text)
    elapsed = (log.stat().st_mtime if finished else time.time()) - t0
    vals = re.findall(r"val (epoch .*)$", text, re.M)
    return {"job": f"speaker model {log.parent.name}", "stage": "finished" if finished else f"epoch {epoch}/{epochs}, step {step}/{steps}",
            "percent": round(100 * frac, 1), "elapsed": _fmt(elapsed),
            "remaining": "0" if finished else _fmt(elapsed / frac - elapsed if frac > 0.01 else None),
            "note": vals[-1][:90] if vals else "no validation yet", "running": _alive(log) and not finished,
            "started": datetime.fromtimestamp(t0).strftime("%H:%M")}


def build_total() -> int | None:
    counts = HERE / "subs_train/counts.json"
    if not counts.exists():
        return None
    import json
    c = json.loads(counts.read_text(encoding="utf-8"))
    return sum(1 for k, v in c.items() if k.endswith(".sdh.srt") and v["labels"] >= 100 and not k.split("/")[1].startswith(
        ("S51E01", "S51E02", "S51E03", "S51E04", "S51E05")))


def collect() -> list[dict]:
    rows: list[dict] = []
    for log in sorted(HERE.glob("exp_finetune/run*/train.log")):
        if (row := training(log)):
            rows.append(row)
    if (row := line_counter(HERE / "subs_train/build_audio.log", "training data: audio + diarization",
                            r"^\{'code'", build_total(), "decoding + DiariZen + refinement per episode")):
        rows.append(row)
    if (row := line_counter(HERE / "subs_train/extract.log", "subtitle extraction", r"^S\d\d/", None, "ffmpeg per file")):
        rows.append(row)
    if (row := line_counter(HERE / "subs_train/build_words.log", "training data: word times (Whisper)",
                            r"^\[\d+/\d+\] ", 35, "Whisper large-v3 per episode")):
        rows.append(row)
    for log in sorted(HERE.glob("subs_train/build_words_*.log")):
        if (row := line_counter(log, f"training data: {log.stem}", r"^\[\d+/\d+\] ", None, "decode + Whisper per episode")):
            rows.append(row)
    for log in sorted(HERE.glob("exp_spk/*/train.log")):
        if (row := spk_training(log)):
            rows.append(row)
    for log in sorted(HERE.glob("exp_spk/run_*.log")):
        if (row := line_counter(log, f"speaker model {log.stem}", r"^=== |^\{'code'", None, "per episode or step")):
            rows.append(row)
    for log in sorted(HERE.glob("stt_listen/run_*.log")):
        # 19 stages per episode with 3 engines; a season run (several episodes in one log) counts them all
        text = log.read_text(encoding="utf-8", errors="replace")
        episodes = max(1, len(set(re.findall(r"\[([^\]]+)\] transcribe \.\.\.$", text, re.M))))
        if (row := line_counter(log, f"voice-over {log.stem}", r"\] [a-z]+(?: \(\w+\))? \.\.\.$", 19 * episodes,
                                f"stages of stt.run (3 engines, {episodes} episode{'s' if episodes > 1 else ''})")):
            if log.with_suffix(".done").exists():
                row.update(stage="finished", remaining="0", running=False)
            rows.append(row)
    for log in sorted(HERE.glob("exp_boundary/run_*.log")):
        if (row := line_counter(log, f"harness {log.stem}", r"^=== ", None, "stt.transcribe per episode / variant")):
            rows.append(row)
    for log in sorted(HERE.glob("exp_tse/run_*.log")):
        if (row := line_counter(log, f"voice sample {log.stem}", r"^=== ", None, "per line or episode")):
            rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# Status ({now})", "", "| Job | State | Stage | Done | Elapsed | Remaining | Note | Started |",
             "|---|---|---|---:|---:|---:|---|---|"]
    running = [r for r in rows if r["running"]]
    for r in sorted(rows, key=lambda r: (not r["running"], r["job"])):
        state = "RUNNING" if r["running"] else ("done" if r["stage"] == "finished" else "stopped")
        pct = f"{r['percent']}%" if r["percent"] is not None else "?"
        lines.append(f"| {r['job']} | {state} | {r['stage']} | {pct} | {r['elapsed']} | {r['remaining']} | {r['note']} | {r['started']} |")
    if not running:
        lines += ["", "Nothing is running right now."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loop", type=int, default=0, help="seconds between rewrites of STATUS.md; 0 = print once")
    args = parser.parse_args()
    while True:
        text = render(collect())
        STATUS.write_text(text, encoding="utf-8")
        if not args.loop:
            print(text)
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
