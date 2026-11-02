"""Training-data build, step 3: Whisper large-v3 word times of the built episodes, to pin each SDH-labelled line to
the words really spoken (the cue timing alone is off by about 0.3 s). Output: <season>/<code>/asr.whisper-large-v3.json.

    subs_env/bin/python scripts/speaker_training/build_words.py [--batch-size 48] [--only S48E01]
Resumable: finished episodes are skipped. Prints "[i/n] code ..." per episode for scripts/status.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path.cwd() / "output" / "subs_train"  # the build folder (a symlink to the data drive here)
REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from stt.transcribe import prepare_audio  # noqa: E402
from stt.worker import call  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--only")
    parser.add_argument("--seasons", help="comma-separated, in this order; decodes the audio first when it is missing (hunted seasons)")
    parser.add_argument("--model", default="whisper-large-v3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    media: dict[str, str] = {}
    if args.seasons:
        counts = json.loads((HERE / "counts.json").read_text(encoding="utf-8"))
        todo = []
        for season in args.seasons.split(","):
            for key, row in sorted(counts.items()):
                if key.startswith(season + "/") and key.endswith(".sdh.srt"):
                    code = key.split("/")[1].split(".")[0]
                    todo.append(HERE / season / code)
                    media[code] = row["media"]
    else:
        todo = [p.parent for p in sorted(HERE.glob("S*/S*E*/audio.wav")) if not args.only or p.parent.name == args.only]
    t0 = time.perf_counter()
    for i, folder in enumerate(todo, 1):
        out = folder / f"asr.{args.model}.json"
        if out.exists() or (folder / "asr.whisper-large-v3.json").exists():
            print(f"[{i}/{len(todo)}] {folder.name} cached", flush=True)
            continue
        t1 = time.perf_counter()
        try:
            if not (folder / "chunks.json").exists():
                if folder.name not in media:
                    raise RuntimeError("no audio and no media known")
                folder.mkdir(parents=True, exist_ok=True)
                prepare_audio(Path(media[folder.name]), folder, 0.0, None, False)
            (folder / "logs").mkdir(exist_ok=True)
            call("subs_env", ["asr", "--audio", str(folder / "audio.wav"), "--chunks", str(folder / "chunks.json"),
                              "--model", args.model, "--out", str(out), "--device", args.device,
                              "--batch-size", str(args.batch_size)], out, folder / "logs" / "asr.whisper.log")
            print(f"[{i}/{len(todo)}] {folder.name} {time.perf_counter() - t1:.0f} s, "
                  f"total {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        except Exception as exc:  # noqa: BLE001 - go on with the next episode, the log names the failure
            print(f"[{i}/{len(todo)}] {folder.name} FAILED {exc}", flush=True)
    print("WORDS DONE", flush=True)


if __name__ == "__main__":
    main()
