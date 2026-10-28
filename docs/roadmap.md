# Roadmap

What comes after the current pipeline (`docs/stt_local.md`). Numbers are from S29E01/E02
(Survivor: San Juan del Sur, 64 and 43 min) unless noted.

## Next steps (28.10.2026, evening)

Done on 28.10 (details in the sections below and in `docs/stt_local.md`): DiariZen v2 as the
diarizer; the stress of the show's names, a second stress placer (silero-stress) and gpt-6.1-sol
as the free screener; ESpeech and TeraTTS retrying takes that lose a short word (ESpeech also a
dash and 0.3 s more length); the TeraTTS female voice ru_f1+ru_f2; TeraTTS voices for people
without a clip in the ESpeech track. Tried and dropped: FireRedTTS3, Seed-VC (accent), DiariZen
on the vocal stem, the TeraTTS teacher sampler, stress marks on one-syllable names.

Who speaks, where it stands (S29/S20/S10): speaker changes vs the captions' `>>` marks F1
0.81-0.85; SDH-labelled S10 lines on their person's speaker ~71%; ~3.7% of clear lines voiced
with the other gender (the line's own voice vs its speaker's gender, `output/stt_dz_sweep/
gender_check.py`); twins (Natalie / Nadiya) one voice. Not exhausted:

1. **Spoken stress check after synthesis** (done 28.10, `voiceover.stress_listen`, `listen_stress`):
   the words the stress check looked at are heard in every take by TeraAnaly and a take said with
   another stress is taken again (section 6). Small gain: the voices already say the marks.
2. **Gender per line** (done 28.10, `stt.line_gender`, `output/stt_linecheck/`): speakers whose gender
   stayed unknown (mostly mixed clusters: "camp voices", "challenge voices") get it from their
   lines' own voices: a confident line (p >= 0.9) keeps its own gender, the rest a vote of the
   speaker's lines (3+, 70%). S10 SDH-labelled lines (E01/E12/E16, 243): right gender 78.2% -> 84.8%;
   S29E01/E05-E08: 6 of 2756 lines changed. Trusting the line's voice on speakers of known gender
   did not pay (the classifier is often the one that is wrong: shouting men at challenges heard as
   women; on S29 47 of 81 such disagreements had the line clearly in its speaker's own voice).
3. **Each line against the season voice bank** (tested 28.10): move a line to another named person
   when its voice clearly sounds like them and not like its speaker (similarity >= 0.5 and 0.25
   above its own speaker, or the other gender with >= 0.4 and 0.15 above; never between relatives).
   S29: 20 of 2756 lines would move, some clearly right ("So, Missy, ..." said to Missy, now Jon's).
   S10 SDH labels: 7 moves in 2302 lines, of the 4 labelled ones 1 right, 2 wrong: names right
   63.6% before and after. Not used; would need a better voice model or a second signal (the
   picture, below).
4. ~~The picture: active speaker detection + faces~~: dropped (the user, 28.10: a dead end).
5. **Setup scripts and the "Настройки" page** (done 28.10, section 5).
6. **Re-voice S20 and S29 with all of it**, then finish S29 (E11, E13-E15, the missing tracks of
   E01-E08): once the improvements above are in (the user, 28.10: not after every small one).
7. Smaller: Vosk 0.11 when it is out (section 4), a TTS cache cleanup command, emotion models to
   pick calm / excited clips, newer diarization and speaker-embedding models, Shmyrev's TTS
   metrics, the translation check only if it measurably helps (section 3b).

## 1. Now: stable whole episodes on Vosk and TeraTTS

The first goal: any episode of a season, from the video to the mkv, without manual steps.

- S29E02 from scratch with Vosk: 28 min for a 43-min episode (voice-over on the CPU 11 min of
  it), no failures, 0.79% character errors in the Russian ASR check, 7 of 608 lines over 15%.
  The same script with TeraTTS: 6 min more (voice-over 4 min), 0.78%, 4 lines over 15%.
- S29E01-E04 in one run (23.10, gpt-6.1-sol through the ChatGPT subscription, voice bank on):
  Vosk 1 h 50 min for the four (23-31 min per episode; fresh ASR ~6 min, LLM steps ~10 min,
  Vosk voice-over 10-14 min on the CPU), then TeraTTS 29 min for the four (voice-over 4-5 min
  each). No failures, no fallback to OrcaRouter needed. Character errors: Vosk 0.57-1.06%,
  TeraTTS 0.50-0.97%; the voice bank found no naming conflicts across the four episodes.
- Done since (26.10): speaker refinement by WeSpeaker ResNet293 (voiced cues on their person's
  main speaker 63.5% -> 71.1% on S10 SDH labels), ducking only the original voices, the browser
  UI (processing, voice bank, review), the review list with corrections, ESpeech F5 engine.
- Done (27.10): Survivor US S20 (Heroes vs. Villains) whole season, 15 episodes in all three
  voice-overs, one video per episode with every voice-over as a track and the voiced Russian
  subtitles (`paths.finished`, `stt.combine`), every line levelled to -20 dBFS. ~16 h for the
  season with a second pass; numbers in `docs/experiments.md`.
- Known weak spot: diarization merges similar voices. S29E01 has one voice cluster with two
  women (Jaclyn and Baylor), so one of them gets the other's name and voice. The season
  speaker bank (below) is the planned fix.

## 2. Season speaker bank

Done (`stt.voice_bank`, see `docs/stt_local.md`): prints, matches as LLM evidence, naming of
unnamed clusters, conflicts, season re-check, clean reference clips (5-9 s, one voice, music
removed, Whisper transcript), UI tab with "another clip". Splitting a cluster into sub-voices
was tried on S10 and lost accuracy (one person's narration and on-location voice split too).

Done (26.10, idea of the user): **the season's cast list from the web** (`stt.cast`) as
evidence for the glossary and the speaker naming, and a check of the names after it. No
measurable gain in names on S10/S29 (they were mostly right already), one spelling per person,
$0.01-0.03 per season; the web lookup got one gender wrong of 45 people, so gender conflicts are
only logged. Worth watching on less known seasons.

Keep a voice print per named person across the episodes of a season:

- clusters of a new episode are matched to the bank, so names and gender stay the same in every
  episode and do not depend on the LLM finding them in the transcript;
- a cluster that matches two people is split, two clusters of one person are joined;
- candidates for the voice prints: the ones Shmyrev ranks highest in his Telegram comparison
  (voxblink2 ResNet-293) and the pyannote embedding the pipeline already computes;
- the bank also keeps, per person, the best clean samples: solo speech (no overlap, no music),
  8-12 s, with the exact transcript from the aligned words. Confessionals are the natural
  source. These samples are the references for voice cloning (section 3).

## 3. A voice per main character (voice cloning)

Done (26.10): engine "f5" (UI "ESpeech (голоса героев)"), clips from the voice bank, fallback
voices for people without a clip, leaked reference speech cut off. S29E03: 19 min per episode,
1.5% character errors (Vosk 0.6%, TeraTTS 0.5%); liked by ear. Open: several clips per person
(calm / excited) only if they can be picked reliably; licences of ESpeech and the clips.

History of the measurements:

Measured: ESpeech-TTS-1_RL-V2 (F5-TTS trained for Russian, accepts `+` stress marks),
reference = 8 s of an English confessional of Jon Misch + 1 s of silence and its transcript,
60 of his lines from S29E01, 32 sampling steps, on the RTX 4070 Ti SUPER:

- 32 sampling steps: 1.16 s per line including the GigaAM check and up to 3 takes (63 takes for
  60 lines), ~18 min for a 900-line episode; 3.1 GB VRAM; 1.0% character errors on first
  takes, 0.83% after retries (Vosk 0.72%, TeraTTS 0.71% on the whole episode);
- 7 steps (EPSS schedule, a 4pda tip; F5 applies it by itself for 5-7, 10, 12, 16 steps):
  0.50 s per line with the check (64 takes), ~7.5 min per episode, 0.77% after retries;
  16 steps: 0.71 s, 0.76%. By ear almost the same as 32 steps (maybe a little less emotion):
  32 steps for the normal voice-over, 7 as a fast mode;
- each line must get its exact length (`fix_duration` = reference + slot length). F5's default
  length estimate counts UTF-8 bytes, so Cyrillic text came out about twice too long and the end
  of the English reference leaked into the speech (31% character errors);
- voice similarity judged by ear: good. Den4ik (Telegram): ESpeech-F5 keeps the acoustics but
  not hoarseness or breathiness, and has no accent cross-lingually. (The listening samples of
  `output/demo/` were deleted on 27.10.)

First listening demo (23.10): three 75 s scenes of S29E02-E04 (23-30 lines, 6-7 people each)
voiced line by line in each speaker's cloned voice, in the TeraTTS slots and at their
loudness: 1.1 s per line with up to 3 takes, 2.0-2.5% character errors (short exclamations
and names are the hard part). References: the voice bank's cleaned clips (vocals only).
F5 runs in its own venv (f5-tts pins differ from subs_env); an engine would call it there.

Other engines discussed in the Telegram chat (Shmyrev's table of 20.10.2026, 1270 Russian
phrases; xRT = synthesis time / audio length, stress errors as he counts them):

| Engine | Size | CER % | stress errors | xRT | cloning | verdict for dubbing |
|---|---|---|---|---|---|---|
| Vosk TTS 0.10 (ours) | ~40M | 0.9 | 2.29 | 0.4 CPU, 0.07 GPU (torch) | no | in use |
| Vosk TTS 0.11 | ~40M | 0.8 | 1.78 | same | no | drop-in when released (not public on 23.10) |
| ESpeech F5 RL-V2 | 336M | 1.4 | 2.04 | 0.45 (4090) | yes | the clone engine to build on (above) |
| OmniVoice 0.6B | 600M | 1.2 | 3.13 | 1.5 (1080), ~0.3 (4090, official) | yes | livelier reading, more stress errors; stress fine-tunes only on Telegram |
| Pocket-TTS RU (xVibe, Storm) | ~90M | 1.9-2.1 | 2.44 | 0.17 (1080), 0.41 CPU | yes | fastest, but weak phrase starts, 7.6% lost endings (its own card), low similarity |
| Higgs TTS v3 | 4B | 0.8 | 1.81 | 0.55 (4090) | yes | best stress, but large |
| Qwen3-TTS 0.6B / 1.7B | | 1.1 / 0.9 | 5.36 / 3.53 | 0.91 / 2.25 (1080) | yes | too many stress errors |

TeraTTS's own cloning is weak (its author: not worth the effort); Supertonic-3 voices converted
to TeraTTS are unstable (azimutspb on 4pda); Fish S2 ignores RUAccent marks; Chatterbox is poor
in Russian.

F5 loses the first word of a line more often than Vosk/TeraTTS (S20E01-E09: the speech check
misses it in 5-7% of lines vs 2.5-3.5%). Two causes on 80 re-synthesized lines: the leaked-reference
cut fired on a name the ASR split in two ("Кортни" -> "корт ни", fixed 27.10: the cut is kept only
when it checks better), and F5 itself swallows a short first word ("Да! Так и есть" -> "так и
есть", "Роб, принеси" -> "принеси"), also over three takes. First take, 40 such lines / 40 normal
ones: a "— " before the text kept the word in 19/40 vs 14/40 but checked a little worse on the
normal lines; 0.3 s more length kept it in 17/40 and checked better on both. To measure on a few
hundred lines with the three takes before changing anything (`output/stt_s20/eval/f5_prefix_probe.py`).

Open problems: picking clean references automatically (reality TV is loud and emotional),
music under the reference (ClearerVoice separation?), licences of the models.

## 3b. Translation check before voicing

A pass of a strong model over the finished translation of an episode: names and terms as in the
glossary, gender agreement of every line with its speaker, lines that lost their meaning when
shortened. Only worth it if it measurably reduces errors heard in the voice-over (the LLM check
of line speakers did not, see `docs/stt_local.md`); measure on a few episodes first.

## 4. Vosk TTS 0.11

Shmyrev's table of 20.10.2026 lists Vosk TTS 0.11: same speed as 0.10, fewer stress errors
(1.78 vs 2.29) and 0.8% CER. Not published yet on 23.10 (no model file on alphacephei.com).
When it is out: download it, set `voiceover.engines.vosk.model_path` in `stt_config.json` to it, re-voice S29E01 and
compare the speech check and the stress errors with 0.10.

## 5. One-command setup and a settings page (done 28.10)

So that a new user only clones the branch, runs one script and then works in the browser:

- `scripts/install.sh`: checks Linux, the NVIDIA driver (CUDA 12.8+), GPU memory, free disk, git,
  curl and `uv` (offers to install it); creates subs_env, f5_env and diarizen_env (`--basic`:
  subs_env only; `scripts/setup_envs.sh` now keeps an existing environment and only updates it, so
  the script is also the update after `git pull`); downloads the Vosk voice model; runs `hf auth
  login` when there is no token and lists the two gated models (pyannote community-1, Cohere
  Transcribe) whose terms to accept; asks for the OpenRouter key without echo and appends it to
  `.env` (never printed); runs `stt.selftest`; offers a menu shortcut.
- `stt.selftest`: one line per check (GPU, ffmpeg, a Vosk line synthesised and heard by GigaAM,
  text preparation, the OpenRouter key against `/api/v1/key`, the subscription if configured;
  without `--quick` every model loaded or downloaded once: TeraTTS, TeraAnaly, Whisper, Cohere,
  pyannote, the voice models, wav2vec2, Demucs, ESpeech in f5_env, DiariZen in diarizen_env).
- `scripts/run_ui.sh`: starts the UI and opens the browser (only the browser when it already runs).
- UI tab "Настройки" (`ui/settings.py`): folders, engines ticked at start, recognition mode,
  diarizer, speaker refinement, line gender, OCR, cast, voice bank, LLM model and stress screener,
  stress check, heard stress, speech check, Vosk / TeraTTS voices (blends allowed, checked against
  the model's voices), ESpeech fallback and quality, subtitles, ducking, tracks, WAV keeping, crowd
  layer, review list. Saved into `stt_config.local.json` as differences from `stt_config.json`
  only, so other personal keys stay and the rest follows new defaults; the previous file is kept
  as `.bak`; "Сбросить" removes the page's keys. The "Обработка" form starts from these values.
  Not tested by a person on a clean machine yet.

## 6. To test (tips from Telegram and 4pda)

- **Spoken stress check** (done 28.10, on by default: `voiceover.listen_stress`). TeraAnaly's
  stress head gives every vowel of a word a score; a listened word is "heard wrong" when another
  vowel wins and the marked one gets under 0.25. Only the words the stress check looked at are
  heard (names, homographs, guessed and disputed words: S29E03 108 words, in ~260 of 640 lines).
  S29E03, current marks, 4 takes per line and engine (`output/tts_eval/stress_retake/`):

  | engine | lines heard wrong, 1st take | after 3 takes | wrong stress noticed* | false alarms |
  |---|---|---|---|---|
  | ESpeech F5 | 2 / 259 | 1 | 67% | 0.8% |
  | Vosk | 7 / 261 | 5 | 37% | 2.4% |
  | TeraTTS | 2 / 261 | 1 | 48% | 0.8% |

  \* the same lines with one listened word's mark moved to a neighbouring vowel, heard against the
  right marks (a voice that ignored the moved mark counts as not noticed, so these are lower
  bounds). ESpeech follows even a wrong mark (87%), so errors in the marks go straight to the audio
  and the stress check before synthesis stays what matters. Takes are retried like garbled ones
  (ESpeech, TeraTTS: up to 3 seeds; Vosk: its sampling noise differs on every run, up to 3 runs);
  cached takes are heard once (`"stress"` in the cache entry) and redone only when wrong; ~0.2 s
  per listened take. Stubborn: Vosk's "Н+атали" (wrong in every take), "Зд+орово" (all engines:
  maybe TeraAnaly). A listening file of the fixed lines: `listen_retakes.wav` + `listen_guide.txt`.
- **TeraTTS teacher sampler** (`diffusion_model: "teacher"`): recommended by the author for
  quality on a GPU; compare dropped words, character errors and speed with the distilled one.
  It changes the sampling network, not the voice.
- **Another female voice for TeraTTS** (done 28.10, `output/tts_eval/tera_female/`): ru_f1 sounded like
  an anime girl (median pitch 272 Hz). On 40 female S29 lines: ru_f2 0.46% character errors,
  204 Hz; eng_f3 0.70%, 186 Hz; eng_f5 5.2%; blends ru_f1+ru_f2 0.60%, 238 Hz;
  ru_f2+eng_f3 0.60%, 197 Hz; ru_f2+eng_f5 3.2%. All sound Russian to Whisper (P(ru) 0.999).
  Listening files `1_ru_f1.wav` ... `7_*.wav` there (the same 8 lines per voice).
  The user picked the blend ru_f1+ru_f2 by ear: the default since 28.10 (`style_dir` in
  `voiceover/engine_teratts.py` makes such blends).
- **RuVoice stress placer** (azimutspb's PC port of the Android RuVoice TTS, used by others as
  a stress placer): compare with RUAccent on the homographs and guessed words of an episode.
- **Shmyrev's TTS metrics** (intelligibility, stress, speaker similarity) to compare engines
  and cloned voices with one yardstick.
- **DiariZen Large MD v2** (tested 27-28.10, `"diarizer": "diarizen-v2"`, `diarizen_env`): best
  open model in Shmyrev's Russian telephony benchmark. On Survivor, with the ResNet293 refinement
  for both, after the whole speakers stage (`stt.turn_eval` against the captions' `>>` marks):
  speaker-change F1 S29E01 0.68 -> 0.83, E05 0.74 -> 0.81, E06 0.72 -> 0.85, E07 0.73 -> 0.81,
  E08 0.73 -> 0.85 (found changes +13-20 points, precision the same or better); diarization only,
  S20E01/E05/E10/E14 0.66-0.77 -> 0.80-0.88. Who is who is not worse: S10 SDH labels (251 lines,
  E01/E12/E16) 69.4% -> 71.3% on their person's speaker, people merged 22 -> 20 of 47; lines voiced
  with the other gender (each line's own voice vs its speaker's gender, S29E01/E05-E08, ~2300 clear
  lines) 3.7% both. Neither splits the twins Natalie / Nadiya. Cost: ~2.5 min per 43-min episode
  (pyannote ~0.5 min), ~4.5 GB VRAM, CC BY-NC weights. AHC threshold 0.5 (0.6 merged Wes into Alec
  in S29E05; a lower VBx Fb splits people into 25-150 clusters). S29E06 voiced both ways
  (`output/stt_dz_test/<E06>.AB.mkv`, notes next to it): 623 lines instead of 589, speech check
  1.54% vs 1.65%; 64 caption-confirmed speaker changes now get their own voice; challenge shouts
  that went to Jeff's voice now form an unnamed "camp voices" cluster (36 lines, a fallback
  voice), which also took a few of Julie's lines in the rain. Next: the user listens; possibly
  name such a crowd cluster's lines by their nearest bank voice when one is clear; then the
  default for new episodes.

- **FireRedTTS3** (tested 28.10, `models/fireredtts3/`, `output/tts_eval/firered/`): not usable
  for the cloned voices. With an English reference clip (all we have) it reads Russian with
  English sounds: 75% character errors on 64 S29E05 lines (ESpeech 0.9%); without the clip's
  transcript or with a language switch it is no better (290%, 83%). With a Russian reference
  (another ESpeech line of the person) it speaks perfectly (0% on 16 lines) but the voice moves
  away from the person (similarity to the clip 0.37 vs ESpeech's 0.45; with the English clip
  0.67, accent included). No length control (median 1.12x ESpeech's fitted length, a third of
  the lines over 1.3x), `+` marks read as "плюс", the acute accent does not steer the stress
  (homograph pairs 55%, chance 50%), ~1 s of synthesis per 1 s of speech, ~9 GB VRAM (backbone in
  bf16; fp32 12.3 GB). Worth a look again only for Russian-speaking shows.
- **Emotion for the cloned voices** (Telegram 27.10): GigaAM emo, LAION Empathic-Insight-Voice-Small,
  NVIDIA Audio Flamingo recognise emotion in speech; they could pick a calm and an excited clip per
  person (section 3, "several clips per person") and the clip matching the original line.
- **Interruptions at speaker boundaries** (Telegram 27.10, the same weak spot as ours): words at
  a turn change without a pause go to the wrong speaker. Ideas there: an overlap classifier, a
  speech separator on the overlapped stretches, punctuation / LLM at the boundary. Measure first
  on S20 against the CBS captions' `>>` speaker-change marks.
- **Look again on GitHub / Hugging Face** for diarization and speaker-embedding models newer
  than pyannote community-1 and WeSpeaker ResNet293 (the S10 SDH benchmark,
  `stt.refine_speakers sdh`, measures any of them in minutes).

## 7. Smaller follow-ups from the S20 run (27.10)

- **F5 swallowed words** (done 28.10, `output/tts_eval/f5_words/`): the speech check never retried a
  take that lost a word under 3 letters (и, в, да: ~240 of the voiced S20/S29 lines, 2-3x Vosk /
  TeraTTS). On 300 S29 lines (150 that had lost a word, 150 normal) through the production take
  loop: "— " before the text + 0.3 s of length + another take while a word is not heard: lines
  losing a word 40% -> 19% (normal lines 4% -> 2%), character errors 2.3% -> 0.85%, 1.57 takes per
  line instead of 1.09 (normal lines +10% time). The dash or the length alone made normal lines
  worse (9-10% lost). Now the F5 engine's default (`prefix`, `slack_s`, `retry_lost_words`).
  TeraTTS gets the retry too (`verify.retry_lost_words`): S29E03, 640 lines, words not heard
  93 -> 64, character errors 0.52% -> 0.37%, no slower. Its teacher sampler changed nothing (0.49%,
  94 words, same speed).
- **Short phrases split by a pause** are now kept in one cue (`--fragment-words`); S20 was
  voiced before that fix and is left as it is (the user, 27.10). New seasons get it.
- **Speaker changes against the S20 captions**: the CBS captions mark every speaker change with
  `>>` (and name only the host): a metric for turn boundaries, the diarization refinement and the
  interruption idea above.
- **S20E02 / E03 in cloned voices** were voiced before Courtney's and Danielle's clips existed,
  so these two speak with the fallback voice there (E01, E04, E05, E07 were re-voiced). A
  `stt.run ... --engine f5` of the two would give them their own voices.
- **A warning in the UI when the ChatGPT subscription login of opencode has expired**: runs fall
  back to the paid routers silently (26.10 the token expired mid-season).
- **TTS cache**: sets of older engine versions pile up in `output/tts_cache/` (40 of 45 were
  stale on 27.10); a cleanup command that keeps the sets the episode folders use.

## Not planned

- **MOSS-Transcribe-Diarize as a third input of the ASR fusion.** It would be one more full run
  (~1.5 min per 10 min of audio, 3.1 GB VRAM) and a third input did not help before (Qwen3-ASR:
  within the LLM's run-to-run spread). Alone it is close to the fusion of Cohere + Whisper
  (9.6% vs 9.1% WER on three 10-min clips of S10E01), not better, so it cannot replace them
  either. Its speaker separation was better than pyannote on one clip and worse on two.
- **Nemotron-3-Diarization**: at most 8 speakers; episodes have 16-21.
