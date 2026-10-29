# Handoff (27.10.2026, updated 29.10)

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
- **29.10**: cleanup (`docs/experiments.md`), `voiceover.cache_clean`, the season stress file,
  two lanes (`--engine f5,teratts,vosk`; the UI does it), `host_voice` `eng_m3+eng_m4` (picked by
  the user). The user's LLM rule (`stt_config.local.json`): a step with one model, sol or luna,
  goes through OpenCode; other models through OrcaRouter; OpenRouter only where OrcaRouter fails or
  lacks the model (gemini-4-argon, web search for the cast). OpenCode Go voters were tried and
  rejected by the user. Stress voters: sol + claude-sonnet-5.5 (OrcaRouter) + gemini-4-argon
  (OpenRouter); the shared default stays OpenRouter.
- **S51** (`output/stt`, videos in the user's `Season_51` folder: read only): E01 (86 min, the
  new season's first episode: cast from the web, glossary, voice bank and clips from scratch) in
  one detached run of `stt.run ... --engine f5,teratts,vosk`, 1 h 31 min, no failure, nothing done
  by hand: recognition 15 min, LLM steps 7 min, stress check 3 min, ESpeech 46 min with Vosk
  alongside (20 min), TeraTTS 10 min, checks and mixing ~5 min per track. Speech check: ESpeech
  1.06%, TeraTTS 0.82%, Vosk 0.95% character errors; 37 lines in the review list. Cristian Chavez
  and Ana Sani got no clean clip (TeraTTS voices in the ESpeech track); Ori Jean-Charles: voice
  and text say male, the web cast female (kept male, by design). GPU memory: peak 11.2 of 16 GB
  with ESpeech and Vosk (its host lines and the stress listener on the GPU) together; alone
  TeraTTS's voice-over process peaked at 6.7 GB, ESpeech's two processes at ~5.1 GB.
- The user's rule since 29.10: stability first, no re-voicing of S20 / S29 until the user says so.
- **29.10 afternoon**: (1) the season cast from its Wikipedia article (`stt/cast.py`: a cast model
  without `:online` reads the article the program fetched; the user's `llm.stages.cast` starts
  with sol through the subscription, free; S51, S29, S20, Australian S10 found that way).
  (2) A stress bug since the start: RUAccent got ten cues as one joined string, which left
  1.8-2.9% of the words with no stress mark (the first word of most cues, corrected names at a
  cue's start, words after "..."); the engines guessed them. Fixed (per cue, apart at ellipses,
  dictionary stress for bare words): 0.1-0.4% left. Checked by the text of S51E01 / S29E01 / S20E03
  and a 5-min S51E02 run in its own folder (all three tracks, 0 of 311 words bare; deleted after). S51E01's
  finished video still has the old stress (re-voice: `stt.run <S51E01> --engine f5,teratts,vosk
  --force stress`, the changed ~440 lines only). The user also heard wrong stresses where the
  vote itself is wrong (S51E01 "Айовы": sol and Claude for +Айовы).
- **29.10 evening**: line editing (`stt.subtitle_edit`, the UI tab "Проверка и правка реплик" that
  replaced "Проверка реплик", the tab "Справка" from `ui/help.md`): text / speaker / gender / "not
  voiced" per line with a length check, the Subtitle Edit round trip (ASS with Actor and gender
  styles, or SRT), typed stress `+` that wins and stays out of the subtitles. Checked end to end on
  5 min of S51E02 in its own folder (deleted after): first voicing 17 min; four edits (two texts,
  one gender in the UI, one speaker through an edited ASS) -> the same `stt.run` 7 min, 3 of 86
  lines synthesised again per engine (the speaker change kept the same fallback voice: cached).
- 318 tests pass, 1 skipped (`subs_env/bin/python -m pytest tests -q`).
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
- A wait loop `until ... ! pgrep -f "<pattern>"` never ends: pgrep finds the waiting shell itself,
  whose command line holds the pattern. Wait on a file the job writes at its end instead.
- Folder names with spaces ("...RAWR(Not Here to Get a Pedicure)", S51E05): never pass them
  through an unquoted `$(ls ...)` list.

## Next steps (updated 29.10)

Everything planned is in `docs/roadmap.md`. The user's rule: stability first, single test
episodes, no re-voicing of S20 / S29 until the user calls it stable.

1. Speaker stability (the plan of 29.10 evening in `docs/roadmap.md`, steps 1-4 done): S51E01-E05
   recognised up to the naming in `output/stt`, their AMZN subtitles with SDH speaker labels in
   `output/s51_subs/` (`stt.sdh_eval`, `stt.refine_speakers.sdh_score`). Base: 62.0% of 886
   labelled lines on their person's main speaker, 44 of 103 people merged, 65.7% under the right
   name. ReDimNet2 did not give a gain that holds on S10 too; ASR on the vocal stem no gain;
   numbers fixed (ru-normalizr). Next candidates: DiariZen's AHC threshold (people merged), words
   at speaker changes (DiCoW), the naming step itself (65.7% names vs 62.0% clusters).
2. S29 when the user says so: Vosk / TeraTTS for E05-E10, ESpeech for E01, E02, E04, then
   E11, E13-E15; S20E02 / E03 in cloned voices.
3. A clean-machine test of `scripts/install.sh` by a person; Vosk TTS 0.11 when it is out.

Open question to the user: the branch `origin/feat/developing-subtitle-shortening-skills` has 27
commits not in `develop`; keep or delete?
