"""Russian speech check of a finished voice-over: does the TTS say what it was given?

Every clip of the final track is transcribed by GigaAM-v3 (Russian CTC ASR, MIT) and compared
with the text the engine received (stress marks and punctuation removed, ё = е):

- pronunciation: character error rate (CER) and the words the ASR heard differently —
  abbreviations read letter by letter, swallowed endings, mumbled or skipped words (a clip is
  flagged above 15% CER or with a word not heard / heard less than 60% similar);
- tempo: speech rate in syllables per second and the most stretched word (its duration per
  letter against the clip's median), which catches drawn-out phrases;
- stress is not judged here: from audio it could not be told reliably (see
  ``voiceover.stress_check``, which checks it on the text before synthesis).

Writes ``speech_check.json`` and a readable ``speech_check.md`` into the voice-over folder.

    uv run -m voiceover.speech_check output/stt/ep01/voiceover [--device cuda]
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voiceover.normalize import SPELLED
from voiceover.text import syllables

log = logging.getLogger("voiceover.speech_check")

ASR_RATE = 16_000
_LETTERS = re.compile(r"[а-яё]+")
_SOUND = str.maketrans({"ё": "е", "э": "е", "ь": None, "ъ": None})
CER_FLAG = 0.15  # a clip above this character error rate is listed for review
WORD_SIMILARITY_FLAG = 0.6  # ...and so is one with a word heard this differently or not heard at all
STRETCH_FLAG = 2.5  # a word this many times longer per letter than the clip median is "drawn out"
SLOW_FLAG = 3.5  # syllables per second
FAST_FLAG = 8.0


def plain_words(text: str) -> list[str]:
    """Lower-case words without stress marks and punctuation, spelled by sound: ё/э = е, no ь/ъ,
    no doubled letters (the ASR spells names as it hears them: Вэл - велл, Дейл - дэйл)."""
    text = text.lower().replace("+", "").translate(_SOUND)
    return [re.sub(r"(.)\1+", r"\1", word) for word in _LETTERS.findall(text)]


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance between the two strings over the reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(reference)


def bad_word(diff: dict[str, str]) -> bool:
    """A word of the text that was not heard, or heard as something quite different."""
    said, heard = diff["said"], diff["heard"]
    if len(said) < 3:
        return False
    return not heard or difflib.SequenceMatcher(a=said, b=heard).ratio() < WORD_SIMILARITY_FLAG


def word_diffs(reference: list[str], hypothesis: list[str]) -> list[dict[str, str]]:
    """Words of the text the ASR heard differently: [{said, heard}] ("" = missing/extra)."""
    out = []
    matcher = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag != "equal":
            out.append({"said": " ".join(reference[a0:a1]), "heard": " ".join(hypothesis[b0:b1])})
    return out


def final_clips(run_dir: Path) -> list[dict[str, Any]]:
    """The clip of every placed unit as exported (original text or the chosen shorter one)."""
    load = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    schedule, durations0 = load("schedule1.json"), load("durations0.json")
    durations1 = load("durations1.json") if (run_dir / "durations1.json").exists() else {"rows": []}
    rows0 = {r["unit_id"]: r for r in durations0["rows"]}
    rows1 = {(r["unit_id"], r["rung"]): r for r in durations1["rows"]}
    clips = []
    for p in sorted(schedule["placements"], key=lambda p: p["start_ms"]):
        row = rows1[(p["unit_id"], p["chosen_rung"])] if p.get("text_changed") else rows0[p["unit_id"]]
        clips.append({"unit_id": p["unit_id"], "start_ms": p["start_ms"], "tempo": p.get("tempo_factor", 1.0),
                      "tts_text": row["tts_text"], "path": row["path"], "voice": row.get("voice")})
    return clips


class Recognizer:
    """GigaAM-v3 CTC with per-word times."""

    def __init__(self, device: str = "cuda") -> None:
        import gigaam

        self.model = gigaam.load_model("v3_ctc", device=device)
        self.device = device

    def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[str, list[tuple[str, float, float]]]:
        import soxr
        import torch

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio16 = soxr.resample(audio.astype(np.float32), sample_rate, ASR_RATE)
        wav = torch.from_numpy(audio16)[None].to(self.device)
        length = torch.tensor([wav.shape[1]], device=self.device)
        with torch.inference_mode():
            encoded, encoded_len = self.model.forward(wav, length)
            text, words = self.model._decode(encoded, encoded_len, length, True)[0]
        return text, [(w.text, float(w.start), float(w.end)) for w in words or []]


def check_clip(clip: dict[str, Any], heard: str, words: list[tuple[str, float, float]], speech_sec: float
               ) -> dict[str, Any]:
    given = clip["tts_text"].replace("+", "")
    for abbreviation, spelled in SPELLED.items():  # the ASR writes a spelled-out США as "сша"
        given = given.replace(spelled, abbreviation)
    reference = plain_words(given)
    hypothesis = plain_words(heard)
    cer = char_error_rate(" ".join(reference), " ".join(hypothesis))
    per_letter = [(end - start) / max(1, len(text)) for text, start, end in words if len(text) >= 2]
    median = statistics.median(per_letter) if per_letter else 0.0
    stretch = max(((end - start) / max(1, len(text)) / median, text) for text, start, end in words
                  if len(text) >= 2) if median > 0 else (0.0, "")
    rate = syllables(clip["tts_text"]) / speech_sec if speech_sec > 0 else 0.0
    diffs = word_diffs(reference, hypothesis)
    flags = []
    if cer > CER_FLAG or any(bad_word(d) for d in diffs):
        flags.append("pronunciation")
    if stretch[0] > STRETCH_FLAG:
        flags.append("drawn_out")
    if syllables(clip["tts_text"]) >= 4 and not SLOW_FLAG <= rate <= FAST_FLAG:
        flags.append("slow" if rate < SLOW_FLAG else "fast")
    return {**clip, "heard": heard, "cer": round(cer, 3), "diffs": diffs,
            "syl_per_sec": round(rate, 2), "stretch": round(stretch[0], 2), "stretched_word": stretch[1],
            "flags": flags}


def clip_score(result: dict[str, Any]) -> float:
    """Lower is better: character errors, words not heard / garbled, a drawn-out word."""
    return (result["cer"] + 0.3 * sum(bad_word(d) for d in result["diffs"])
            + 0.1 * max(0.0, result["stretch"] - 2.0))


def check_audio(recognizer: "Recognizer", tts_text: str, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Speech check of one synthesized clip (used by engines to retry bad takes)."""
    heard, words = recognizer(audio, sample_rate)
    return check_clip({"tts_text": tts_text}, heard, words, len(audio) / sample_rate)


def check_run(run_dir: Path, root: Path, device: str = "cuda") -> dict[str, Any]:
    clips = final_clips(run_dir)
    recognizer = Recognizer(device)
    results = []
    for clip in clips:
        path = Path(clip["path"]) if Path(clip["path"]).is_absolute() else root / clip["path"]
        audio, sr = sf.read(str(path), dtype="float32")
        heard, words = recognizer(audio, sr)
        results.append(check_clip(clip, heard, words, len(audio) / sr))
    cers = [r["cer"] for r in results]
    summary = {
        "clips": len(results),
        "mean_cer": round(statistics.fmean(cers), 4) if cers else None,
        "clips_cer_over_flag": sum(1 for c in cers if c > CER_FLAG),
        "median_syl_per_sec": round(statistics.median(r["syl_per_sec"] for r in results), 2) if results else None,
        "flags": {flag: sum(1 for r in results if flag in r["flags"]) for flag in ("pronunciation", "drawn_out",
                                                                                     "slow", "fast")},
    }
    return {"schema": "voiceover-speech-check-v1", "asr": "GigaAM-v3 ctc", "summary": summary, "clips": results}


def _ms(value: int) -> str:
    seconds = value // 1000
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def render_markdown(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "# Speech check",
        "",
        f"ASR: {result['asr']}. Clips: {s['clips']}; mean CER {s['mean_cer']}; "
        f"over {CER_FLAG:.0%} CER: {s['clips_cer_over_flag']}; median tempo {s['median_syl_per_sec']} syl/s.",
        f"Flags: {', '.join(f'{k} {v}' for k, v in s['flags'].items())}.",
        "",
        "| time | flags | CER | text given to TTS | heard by ASR | differences |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["clips"]:
        if not r["flags"]:
            continue
        diffs = "; ".join(f"{d['said'] or '∅'} → {d['heard'] or '∅'}" for d in r["diffs"][:4])
        extra = f" (stretched: {r['stretched_word']} ×{r['stretch']})" if "drawn_out" in r["flags"] else ""
        lines.append(f"| {_ms(r['start_ms'])} | {', '.join(r['flags'])}{extra} | {r['cer']:.2f} | "
                     f"{r['tts_text'].replace('|', '/')} | {r['heard']} | {diffs} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the voice-over clips with a Russian ASR (GigaAM-v3)")
    parser.add_argument("run_dir", type=Path, help="voiceover.pipeline output folder")
    parser.add_argument("--root", type=Path, default=Path("."), help="Folder the clip paths are relative to")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    result = check_run(args.run_dir, args.root, args.device)
    (args.run_dir / "speech_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                    encoding="utf-8")
    (args.run_dir / "speech_check.md").write_text(render_markdown(result), encoding="utf-8")
    log.info("speech check: %s -> %s", result["summary"], args.run_dir / "speech_check.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
