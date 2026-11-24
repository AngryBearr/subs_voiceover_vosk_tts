# Research scripts

One-off benchmark and comparison scripts kept for reference. The pipeline does not import them.
Run from the repository root with the main environment:

| Script | What it does | Run |
|---|---|---|
| `bench.py` | ASR benchmark on a clip against reference subtitles; per-model summary | `subs_env/bin/python -m research.bench asr --media ep.mkv --srt ep.en.srt --out output/stt_bench/ep` / `... summary <dirs>` |
| `bench_diar.py` | Diarizer benchmark on VoxConverse parquet shards | `subs_env/bin/python -m research.bench_diar --parquet <file> --out output/stt_bench/diar` |
| `compare_translations.py` | Side-by-side HTML of translation variants of one episode | `subs_env/bin/python -m research.compare_translations output/stt/ep --variants a b` |
| `synthesize_test.py` | Quick Vosk TTS smoke test with a sample text | `subs_env/bin/python -m research.synthesize_test -t "Привет, мир" --device cpu` |

Results are summarised in `docs/stt_benchmarks.md` and `docs/experiments.md`.
