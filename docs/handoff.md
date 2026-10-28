# Handoff (27.10.2026, updated 28.10 night)

For the next agent working on this repository. Read `AGENTS.md` first, then `docs/stt_local.md`
(how the pipeline works) and `docs/roadmap.md` (what comes next).

## The project

A local GPU pipeline (RTX 4070 Ti SUPER 16 GB, Linux) that turns English reality-TV episodes
(Survivor US, Australian Survivor) into a Russian voice-over: ASR (Cohere + Whisper fused by an
LLM, CTC word times), diarization (pyannote community-1 + WeSpeaker ResNet293 refinement),
speaker names and gender (LLM + season voice bank + cast list from the web), OCR of on-screen
text, LLM translation fitted to the timing, three TTS engines (Vosk on the CPU, TeraTTS, ESpeech
F5 with each person's cloned voice), speech check (GigaAM), mixing with ducking of the original
voices, one mkv per episode with every voice-over as a track and the voiced Russian subtitles.

One command per episode or season: `subs_env/bin/python -m stt.run "<video or glob>" --engine
f5|vosk|teratts`; browser UI: `subs_env/bin/python -m ui.dub_app`.

## Where things are

- Code: `stt/` (recognition, translation, run, combine, voice bank, cast), `voiceover/` (TTS
  engines, timeline solver, render, speech check), `ui/dub_app.py`, `utils/`.
- Branch `feat/stt-voiceover-pipeline` (was `merge/stt-voiceover`) in the worktree `~/.t3/worktrees/subs_voiceover_vosk_tts/t3-e7bde976`,
  pushed (fast-forward) to `origin/develop`. The user's own checkout
  `~/Projects/subs_voiceover_vosk_tts` is on `feat/shortening-subtitles-v3`: do not touch it.
- Environments: `subs_env` (everything), `f5_env` (f5-tts for the ESpeech worker), `diarizen_env`
  (DiariZen, uses `models/diarizen/repo`); `stt_qwen_env` (optional comparisons) was deleted on
  29.10 (`bash scripts/setup_envs.sh qwen` makes it again). Everything: `bash scripts/install.sh`.
- Config: `stt_config.json` (shared, in git) with `stt_config.local.json` on top (personal, not in
  git: LLM through the opencode ChatGPT subscription with `router: ["orcarouter", "openrouter"]`
  as fallback, `paths.finished`). `voiceover_config.json`: voice-over details (export level
  `loudness_db: -20`).
- Keys in `.env` (OpenRouter, OrcaRouter, DeepSeek, HF): never print them.
- Videos: `/media/angrybear/large_storage/Survivor/Season 29 - San Juan Del Sur 1080p/`,
  `/media/angrybear/large_storage/Survivor (2000) RePack/Season 1..41/` (S20 = Heroes vs.
  Villains, no embedded subtitles), `/media/angrybear/large_storage/Survivor/Australia S10 1080p/`.
- Finished videos: `/media/angrybear/new_large_storage/Survivor/STT_<season folder>/<name>.ru.mkv`
  (`paths.finished` + `paths.finished_folder`). A folder `Season_51` there is the user's; not
  touched.
- Work folders: `output/stt_s20/` (S20, finished), `output/stt_s29_v2/` (S29, partly voiced),
  each with the season's glossary, cast, voice bank (`<folder>.voices.json`) and clips
  (`<folder>.references/`). `output/s20_subs/srt/` = English reference subtitles of S20 (CBS
  captions with `>>` speaker changes; AMZN as `.alt.srt`; `report.md`). The check scripts of 27.10
  were deleted with the other experiments on 29.10: `docs/experiments.md` (what helped, what not).
- Community knowledge: Telegram chat «Распознавание и синтез речи» exports in
  `~/.var/app/org.telegram.desktop/data/TelegramDesktop/tdata/temp_data/ChatExport_2026-10-23`
  and `ChatExport_2026-10-27` (one `messages.html`); 4pda TTS thread (see the user's memory).

## How to run a season (what worked for S20)

Batches of three episodes, every batch in all three voice-overs (cloned voices first), started
detached so a closed terminal does not stop it:

```bash
S="/media/.../Season 20"
for engine in f5 vosk teratts; do
  subs_env/bin/python -m stt.run "$S/Survivor - S20E0[1-3]*.mkv" --out-root output/stt_s20 --engine $engine
done
# detached: setsid nohup bash script.sh > log 2>&1 < /dev/null & disown
# stop: kill -TERM -<process group id>; start the same command again to resume
```

Finished stages are skipped; an interrupted voice-over keeps every finished line. A second run
of the same batches afterwards picks up whatever failed (S20: one OCR crash) and new voice clips.

- `stt.combine "<episode folders>" [--relevel] [--delete-separate]`: one video per episode from
  episodes voiced before the combined videos, levels tracks rendered before the levelling.
- Speech check results: `voiceover.<engine>/speech_check.json` (`summary`); speaker changes
  against the captions: `stt.turn_eval`.
- Since 29.10 the episode folders have no `voiceover.<engine>/export/track.wav` (deleted for disk
  space): an episode's next run re-renders its voice-over from the TTS cache (the `.m4a` files and
  the videos are kept). `voiceover.cache_clean --delete` (or the UI settings) drops cache takes no
  episode uses.

## State on 27.10.2026

- **S20**: done, 15 episodes, three tracks each, levelled. Recognition vs CBS captions: fusion
  6.5-12.9% of words differ; speech check ESpeech 1.45%, Vosk 0.66%, TeraTTS 0.55% character
  errors. E02/E03 cloned voices still use the fallback voice for Courtney and Danielle.
- **S29** (`output/stt_s29_v2`): E01, E02, E04 Vosk + TeraTTS; E03 all three; E05-E10 ESpeech
  only; E11 stopped at its glossary step (recognition done), E13-E15 not started (there is no E12
  file; E11 is 86 min, two episodes in one). Resume: `bash output/stt_s29_v2/run_s29_rest.sh`
  (detached, as below): E09-E11 in all three engines, then the missing tracks of E01-E08, then
  E13-E15; a file `output/stt_s29_v2/PAUSE` holds it between passes. The videos' own English
  subtitles (CBS roll-up captions) are extracted in `output/s29_subs/` for checks.
- **27-28.10 experiments** (DiariZen, FireRedTTS3, Seed-VC, stress, F5 / TeraTTS retakes, gender
  per line, the voice bank check): concluded, folders deleted on 29.10; results in
  `docs/experiments.md`. GPU experiments next to a season run: `scripts/gpu_window.sh <run log>
  <cmd>` waits while an episode is transcribing and stops the command when the next one starts.
- **28.10 (night)**: the spoken stress check (`voiceover.stress_listen`, on by default) and the
  one-command setup (`scripts/install.sh`, `stt.selftest`, `scripts/run_ui.sh`) with the UI tab
  "Настройки" (`ui/settings.py`, writes only differences into `stt_config.local.json`, keeps
  `.bak`). `install.sh` was tested only up to its checks on this machine (running it whole would
  update the existing environments); `stt.selftest --quick` passes here. The "picture" idea
  (active speaker detection + faces) is dropped by the user.
- 311 tests pass, 1 skipped (`subs_env/bin/python -m pytest tests -q`).
- Disk: `/home` 80 GB free after the cleanup of 29.10 (it was 22 GB).

## Working with this user

- Reply in Russian, plain language, explain terms; docs and code comments in English.
- Accuracy first, then UI; iterate and test each stage; commit periodically with small commits.
  Pushing `feat/stt-voiceover-pipeline` fast-forward to `origin/develop` was authorised; ask before
  anything else outward-facing. No force-pushes.
- LLM costs: the ChatGPT subscription through opencode first (free), cheap models for bulk
  steps, strong paid models only for small checks (Claude/Gemini stress votes are kept); say
  what something costs. Shared default model: OpenRouter gpt-6-luna.
- Voice-over: all three engines, the cloned voices (ESpeech) matter most; batches of ~3 episodes;
  one video per episode for the audience.
- TeraTTS female voice: the blend ru_f1+ru_f2 since 28.10, picked by the user by ear (ru_f1 alone
  sounded cartoonish).

## Pitfalls learned the hard way

- `pkill -f <pattern>` killed the agent's own shell: stop runs by process group
  (`kill -TERM -<pgid>`), never by pattern.
- Monitors expire after 30 min, background shell waits after 2 h: re-arm; keep runs detached.
- `stt.run` imports the transcription in its own process, so code changes apply to the next
  process; the voice-over and F5 worker run as subprocesses per episode and pick up new code.
- Bumping the F5 worker identity (`"worker"` in `voiceover/engine_f5.py`) invalidates the whole
  F5 cache: any re-run of an episode's voice-over stage then re-synthesises all its lines.
- A newer `<season>.references/references.json` makes every earlier F5 episode re-dub (new clips
  should be used); it is no longer rewritten when unchanged.
- The opencode ChatGPT token expires: runs fall back to OrcaRouter, then OpenRouter, silently;
  the user re-logs with `opencode auth login`.
- Reference subtitles of S20 run 0.5-3 s early, stepping at ad breaks: compare texts, not times.
- Old repacks can have damaged video packets (S20E06): OCR now skips them.

## Next steps

Everything planned is in `docs/roadmap.md`; in short, by the user's priorities:

1. Finish S29 (E09-E15 in all three voice-overs; ESpeech for E01, E02, E04; Vosk/TeraTTS for
   E05-E08) - batches of three, as above.
2. ~~One-command setup/run scripts and a "Настройки" page~~ (done 28.10); a test on a clean
   machine by a person is still to do.
3. Tests from the community (roadmap §6): FireRedTTS3 (new, 27.10), TeraTTS teacher sampler,
   another TeraTTS female voice, spoken stress check, RuVoice stress placer, Shmyrev's TTS
   metrics, DiariZen Large MD v2, newer diarization / speaker-embedding models, emotion models
   for picking clips, interruptions at speaker boundaries.
4. S20 follow-ups (roadmap §7): F5 first-word losses, speaker-change metric from the `>>` marks,
   E02/E03 cloned voices, a UI warning for an expired subscription login, TTS cache cleanup.
5. Vosk TTS 0.11 when it is published (roadmap §4); the translation check only if it measurably
   helps (roadmap §3b).

Open question to the user: the branch `origin/feat/developing-subtitle-shortening-skills` has 27
commits not in `develop`; keep or delete?
