# Roadmap

What comes after the current pipeline (`docs/stt_local.md`). Numbers are from S29E01/E02
(Survivor: San Juan del Sur, 64 and 43 min) unless noted.
The experiment folders named below (`output/stt_dz_*`, `output/tts_eval/*`, `models/fireredtts3`,
...) were deleted on 29.10.2026; what each showed is summed up in `docs/experiments.md`.

## Plan of 5.11.2026: one episode in about an hour; preparing a season; one subtitle base

Measured (S48, 64-min episodes, nothing cached): about 75 min per episode. Transcribe 12 (OCR 5,
diarization 2), glossary + speakers 3.5, translate + proofread + on-screen text 8.5, stress 3, ESpeech (F5)
voice-over 31, its check + mix 4.3, then TeraTTS voice-over + check + mix 10.5 (after F5, same GPU lane);
Vosk runs alongside on the CPU (20). Reviewed together with an independent GPT-6 Astra analysis (5.11).

1. **Faster without losing quality** (in progress, branch `perf/pipeline-speed`): find why F5 now takes
   1.94 s per line (1.16 s measured 28.10 with the same 32 steps: stress listening on every take, retries,
   reference reloading?) and win it back without dropping checks; start OCR at the beginning of transcribe,
   beside diarization/ASR/fusion; translate on-screen text beside the voice-over instead of before it;
   overlap stress screening with the whole-text proofreading; one Demucs separation of the original per
   episode and one final mux instead of one per engine; TeraTTS starts while F5 is mixed, not after;
   calibration of the voices cached across episodes. Expected 75 -> 62-66 min; measured stage by stage.
   Measured on S48E07 (64 min, nothing cached, 5.11): 65.7 min. Transcribe 470 s (was 709; OCR 322 s fully
   hidden), translate 254 s (was 513; on-screen text in the background), stress 142 s (was 191), F5 1761 s,
   TeraTTS started while F5 mixed. The deferred mux put the 312-s video compression at the end (mux 378 s):
   now started in the background when voicing starts (expected ~60 min). F5 per phase (869 lines): sampling
   1.04 s per attempt, 1.25 attempts per line (142 lines retried), GigaAM check 0.20 s, stress listening 0.31 s
   per line (326 lines): the extra time since 28.10 is the added checks, not slower synthesis.
2. **F5 at 16 steps** (needs the user's ear): 16 steps measured 0.71 s/line against 1.16 at 32, character
   errors 0.76% vs 0.83%, by ear almost the same (28.10). A 10-min A/B piece for listening; if accepted,
   about 55 min per episode.
3. **Prepare a whole season first** (user's idea, 5.11): a separate command and a UI button "recognise the
   whole season" (first pass of every episode: transcription, OCR, diarization, glossary, naming, season
   voices) with a clear message why: the season voices then use every episode and each later episode run
   only translates and voices (about 16 min less per episode). Then, optionally, overlap episodes: while the
   GPU voices episode N, the LLM stages of N+1 (translate, proofread, stress) run.
4. **One subtitle base for all seasons** (user's idea, 5.11, after 1-3): a subtitles stage with its own UI
   setting (a season subtitle folder can be given); pick the best file per episode from the hunt archive
   `/media/angrybear/new_large_storage/Survivor/subtitles_hunt` (names of speakers > `>>` marks > plain text,
   timing that matches the video) into one curated base, used by every season run (who speaks, captions,
   names).

### Status and queue (5.11 night)
- Done and in develop (1ab39c8): speed-up (S48E07 65.7 min, S01E01 34.5 min for 44 min), GPU fallback
  (`gpu.parallel: auto`), subtitle labels -> cast names, bank prints for every named person.
- To check: background video compression on a real 2 GB episode (S48, preset upload): log line
  "compressing the video in the background", mux without the 5-min encode.
- Now: (a) the narrator (host voice-over, 22% of S01E01, unnamed; the naming LLM marked the host card "same as S02");
  (b) a "Кто это?" panel per season with an episode filter: unnamed groups with audio snippets, pick a cast name;
  best used after preparing the season and before voicing; (c) a blind A/B piece of F5 at 32 vs 16 steps.
- Queue: season prepare command + UI button (item 3 above); references for new seasons (one episode rarely has a
  clean 5-9 s solo stretch: more episodes, or a relaxed rule for people without any clip); one subtitle base (item 4).
- Closed 5.11: the audio-LLM speaker experiment with references (`output/exp_audio_llm/report.md`): models 12-21% right
  vs our voice prints 64%, no use even for shouts (docs/experiments.md).

## Plan of 3.11.2026: the UI built around a season (agreed with the user)

The user found the UI confusing: four different "folder" fields (two hard-coded to `output/stt`), long actions with no
feedback, "season voices" (who speaks) and the voice bank (ESpeech clips) under similar names, and no way to give a
person's clip by hand. Agreed: rebuild it step by step; after each step the agent clicks through it in the built-in
browser (a spare port, never the user's UI on 7861), then the user tries it.

1. **Season and folders.** A season is two folders: the original episodes, and the results folder suggested next to
   it (`STT_<season>`, editable) that holds everything (episode work folders, season files, finished videos). Seasons
   are remembered; one season picker on top drives every tab; no other folder fields.
2. **Feedback for every action.** Quick actions answer at once; long ones run as a background job with stage,
   percent and "Остановить", and their button waits while they run.
3. **Tab "Голоса сезона"** (replaces "Банк голосов"): the people of the season, lines, clip yes/no with a plain
   reason; listen, find another automatically, give a clip by hand (episode + one or more `mm:ss-mm:ss` ranges; cut,
   cleaned, Whisper text shown and editable, silence added; never replaced by the automatic pick); then "re-voice
   ESpeech in N episodes" for only those people. Also "rebuild all clips" and "who speaks: assign again" with plain
   descriptions.
4. **Help** rewritten as the order of work.

Status: all four steps done 3.11.2026. Step 1: season bar, `ui/seasons.py`; folder dialog `ui/folder_pick.py` (zenity with a desktop, an in-page browser otherwise); a line on every tab saying what it writes where; settings per season in `<results>/season_settings.json` over one global settings file, `ui/season_settings.py`, UI state in `ui.local.json`. Steps 2-3: one task runner with Russian titles and a status strip under the season bar on every tab (`ui/dub_app.py` `Job`/`start_task`); every button answers (long ones are tasks, medium ones show "⏳ ..." on the button, quick ones a message); tab "Голоса сезона" (`ui/voices.py`): people table with the reason a clip is missing (`missing.json`), hand-made clips (`stt/manual_clip.py`, cut in a subprocess through `stt.voice_bank --manual-clip --preview --json`, kept in `manual.json`), re-voice banner derived from `speaker_voices.f5.json`, rebuild all clips, season voices recount, names check. Step 4: `ui/help.md` rewritten as the order of work. Checked by clicking through on a temp copy of season 48.

## Plan of 1.11.2026: train the voice model, use existing subtitles, the season's voices

Asked by the user after the voice-sample experiments closed. Scripts and logs: `output/exp_spk/`; data: `training_data_sdh/subs_train/` on
the external drive; subtitles found: `/media/angrybear/new_large_storage/Survivor/subtitles_hunt/` (`report.md`). Numbers: `docs/experiments.md`.

1. **Find subtitles** - done: every episode of S1-S51 has a timed English file (~4200 files): "NAME:" labels on S1, S2, S5, S6, S40-S47,
   most of S7-S9, part of S11, S13, S35; ">>" marks on nearly all of S3-S39; The Amazing Race has labels on S1, S7 and a few others.
2. **The rule in the pipeline** - done, on by default (`subtitles` in `stt_config.json`, `stt.subtitles`, `stt.subtitle_hints`): embedded -> next to
   the video -> archive -> online; named lines fix their words to the person, turn marks make a switch cheaper. Transcripts made before the rule stay as they are.
   Not yet run on real network or on a real mkv stream inside a full `stt.run`.
3. **Fine-tune ResNet293** - done: `resnet293-ft` (`models/speaker/resnet293-ft/`, run r3: 334 people of 20 seasons). Same/other error on S51 25.0% -> 20.6%;
   alone in the pipeline's cluster-based pass it is no better (66.2% -> 64.8%).
4. **Per-season fine-tuning (the user's "listener" idea, the network variant)** - done, dropped: no gain over r3 once the Viterbi runs, harmful with few labels (Australian S10).
5. **Season voices** - done, on by default since 2.11 (`stt.season_voices`, the user's idea, without training): words assigned over the people named in the whole season. Through the
   real assembly, r3 model: S51 lines on their person 68.4% -> 71.6% (read by name), S20 F1 0.879 -> 0.897, S29 0.860 -> 0.871. No LLM; GPU only for r3's word embeddings.

Listened to and approved by the user on 2.11.2026: this pipeline (season voices with `resnet293-ft`, existing subtitles, three voice-overs, the new voice level) became the default, with the presets `main`, `upload`, `fast`, `previous` (`stt.run --preset`, UI "Набор настроек"); README files added. Still not done: re-grouping the turns (below) and the remaining stress errors of other kinds (for example "+Избранный"). A neutral voice when unsure: dropped by the user on 2.11 (the voice must not switch between a hero's and a neutral one; `docs/experiments.md`).
State of the listening runs (config `output/exp_spk/listen/`, work `output/stt_listen/{s20,s51}`, videos `Survivor/STT_listen_2026-11-01/<season>/`, the earlier `STT_*` untouched):
S20E01 and S51E01 (finished 2.11.2026 10:24: season voices on, three voice-over tracks, 168 of 1434 cues moved to another person) were listened to (see above). Places to listen to: `subs_env/bin/python -I output/exp_spk/listen/moved_lines.py output/stt_listen/s51/<episode> 14`. Nothing is committed yet.
Added 2.11 at the user's request: tab "Звук и размер видео" (hear a piece with the voice level changed, `stt.mix_preview`; shrink a video to a size at the original resolution, `stt.video_compress`, off by default). A third listening run: Australian S10E01 (`output/exp_spk/listen/run_au10.sh`, voices from E01-E03), finished 2.11 with the new voice level and the stress rule; S20E01 and S51E01 keep the old ones until mixed or voiced again. Compression test: S51E01 7.07 -> 2.07 GB in 15 min on the graphics card, off by default.
Not done: re-grouping the turns with a bonus for DiariZen's clusters (B-cubed on S51 0.569 -> 0.613, 24 groups per episode, `output/exp_spk/cluster_quality.py`, not run through the pipeline; thresholds picked on the scored episodes).

## Plan of 31.10.2026: who speaks, told by a sample of the voice (closed by the user 1.11.2026)

The question changes from "which cluster is this piece like" to "is this person's voice here". Scripts, logs and
tables: `output/exp_tse/` and `output/exp_tsvad/` (`report_test.md`, `report_nearby.md`). The numbers are in
`docs/experiments.md`. Nothing went into the pipeline.

1. **A common test set** - done: S51E01-E05, 1048 SDH-labelled lines selected without looking at whether the pipeline was
   right (exchanges inside one cue, short interjections, ordinary lines as the control). Samples at least 120 s from the
   tested line: SDH lines from conversations ("field"), voice bank stretches, mostly confessionals ("interview"). Settings
   were picked on E01/E03, the verdict comes from E02/E04/E05.
2. **USEF-TSE, extracting a voice by its sample** - done, dropped: on real scenes the output hardly depends on the sample.
3. **TS-VAD (AED-TSVAD), the activity of a given voice** - done, not taken: the sample does steer it, but it is below
   the plain voice comparison made with the same samples and useless without a list of who is in the scene.
4. **What the tests showed instead**: with the scene's people from the SDH labels (an oracle, the real speaker is always
   among them) a plain ResNet293 comparison gets 78-81% of the hard lines against the pipeline's 53-54%; with the people the
   pipeline itself names nearby it gets 58-59% against 56%.
5. **Who is in the scene, without the labels of the moment** - done 1.11, nothing to build (`output/exp_tse/scene_people.py`,
   `cue_prior.py`, `phrase_prior.py`, `unnamed_sweep.py`, `unnamed_precision.py`; numbers in `docs/experiments.md`). Wider
   lists hold the real speaker more often (65% within 8 s, 79% within 60 s, E02/E04/E05), but the comparison gets no
   better (58-60%). A soft bonus for the pipeline's own name first looked strong on exact SDH lines (58.3% -> 65.6%), but
   nearly all of that came from lines without a name, which the metric cannot count as broken. On lines that already have
   a name the rule gains +1.6 / -0.4 points; on the pipeline's own cues and phrases +1-2 points on E02/E04/E05, a loss of
   about 1 point on E01/E03, and S29 F1 goes down. Moves of unnamed cues reach the right person only 20-54% of the time.
   So the mixed cluster stays with its generic voices: a wrong named voice is no better.

Closed: the user took these directions as finished on 1.11.2026, nothing went into the pipeline. An idea left unmeasured
when it closed: taking the speaker straight from the SDH label for the lines that have one, on the seasons with such
subtitles (S46, S48-S51, Australian S10; not S20).

## Plan agreed 31.10.2026: shouts, name cards, and where the ceiling is

The user picked two of the ideas of 30.10 (`docs/experiments.md`, 31.10 rows): a shout compensation of
the word embeddings and name-card conflicts as a sign of merged clusters; pyannote Precision-3 was
skipped (a new paid account, nothing on OpenRouter). Both measured on S51E01-E05 (4094 SDH-labelled
words, `output/exp_shout/`, `output/exp_cards/`), neither taken:

1. **Shouts** are not the cause: the wrong words are not louder or higher than the right ones, the
   "shout direction" of the embeddings holds 1% of the variance, and compensating it, clipping the
   windows at pauses or two centroids per cluster change nothing or lose. Measured on the way: a word
   may take only a speaker heard within 30 s (`locality_s`, off): S51 lines 63.5% -> 63.9%, S29 F1
   0.861 -> 0.852, S20 F1 0.878 -> 0.868. Not taken (code removed): it removes the far-away shouts the
   Viterbi finds.
2. **Name cards** catch 1 of the 50 merged pairs: the merged person's interviews sit in their own
   cluster, only their field speech lands in another's. Kept from it: a card shown in a pause now goes to
   the speaker of the next 10 s (`on_screen_hints`; 4 of 191 cards, all right).
3. **Where the ceiling is**: centroids from a person's own labelled words classify 79.8% of the words,
   the diarizer's clusters 61.9%; 30 clusters hold three or more people and 63% of the wrong words;
   re-clustering the turns with ResNet293 does not beat DiariZen. Centroids from the interview turns
   under the name cards reach 67.0% on the card persons (clusters 58.5%): interview speech is far from
   field speech (cosine 0.335 vs 0.428 to the own centroid), which is also why the voice bank gave
   nothing as Viterbi states. As extra states next to the clusters: +0.5 points of word accuracy, +0.9
   with the locality; replacing the cluster centroids: worse.

**Training data (31.10, evening, the user's go-ahead; since 1.11 on the external drive, `/media/angrybear/new_large_storage/Survivor/training_data_sdh/` with a README, `output/subs_train` and `output/exp_finetune` are symlinks to it; the trained weights were deleted):** the only speaker labels on this machine are the
SDH subtitle streams inside the Amazon WEB-DL files of seasons 46 (3 episodes), 48, 49, 50 and 51
(`output/subs_train/`: `inventory.py` lists the streams of every video file, `extract_subs.py` pulls the
English SDH streams to `<season>/<code>.sdh.srt`, `counts.json` has the labels per episode): 47 episodes,
57.8 h, 9865 labelled lines ("NAME:"), no `>>` marks. The RePack seasons 1-26, S45/S46 (RUS SUB) and The
Amazing Race have no subtitle streams; opensubtitles.org and subdl are behind Cloudflare for this machine,
subtitlecat shows the Amazon captions of S17-S33 as page text but without speaker labels. Training set: S46,
S48, S49, S50, S51E06 (~51 h, ~8.3k labelled lines); held out: S51E01-E05, S20, S29, Australian S10
(24 episodes with SDH, `/media/angrybear/large_storage/Survivor/Australia S10 1080p`, the `.srt` next to the
video; the `_synced.srt` files are ASS inside). `build_audio.py` decodes the audio and runs DiariZen (exclusive
layer) + the ResNet293 refinement per episode without ASR (~4 min each); `embed_labelled.py` embeds 1 s
windows inside the labelled lines for the embedding-side experiments (`output/exp_proj/`).

**Fine-tuning tried the same evening (two runs, `docs/experiments.md`):** no gain through the pipeline (S51 63.5% -> 63.8-64.0%, S20 F1 0.878 -> 0.866-0.870, S29 0.861 -> 0.859-0.860) and no movement on the corrected regions of the validation set; the user's reading, agreed: with the open models and this labelling, the hard off-screen lines stay at about 64%. Not pursued further. Status file of long jobs: `output/STATUS.md`, written every minute by `subs_env/bin/python -I scripts/status.py --loop 60` (start it with nohup when a long job runs).

Next, if the user wants more here: (a) the card persons as extra Viterbi states (about one point, a
day: the bank-state code of commit 4cfd261 is the template, the seeds come from `ocr.json` and the
refined turns); (b) the big lever stays domain adaptation of the segmentation model on Survivor audio
with the SDH / `>>` subtitles as weak labels (the clusters are the limit, not the embeddings;
`docs/experiments.md`, 30.10 ideas), 2-3 days of work and hours of GPU per variant.

## Plan agreed 30.10.2026: speaker changes without a pause

The user's biggest complaint: at a turn change without a pause one speaker gets the other's words,
and no diarizer fixed it. Where the errors sat (S51E01-E05, 886 SDH-labelled lines, 338 wrong): 22%
the neighbouring cue's speaker (a boundary off by a word or two), 68% a short line (median 1.4 s,
56% under 1.5 s) whose person's cluster is nowhere near: a shout the diarizer merged into a long
turn (70 of 231 went to Jeff Probst's cluster at challenges). S51E01: 63% of the speaker changes
have a pause of 0.2 s or less, 16% overlap. The S20 runs of 27.10 (pyannote) found 62% of the
captions' turn changes. The steps, in the order agreed with the user:

1. **DiariZen's exclusive layer and its median filter** (`"diarizer_options"`) - done 30.10: the
   exclusive layer (one speaker per frame by the soft segmentation) +0.015 F1 on S20 (0.841 ->
   0.856), neutral on S51; the median filter off: nothing. On by default. Since 30.10 one pass of the
   segmentation model gives both the hard and the soft output: same turns as two passes, bit for bit on
   S51E01, 253 -> 170 s, nothing added to DiariZen's time.
2. **Word-level relabelling** (`stt.boundaries`, `"word_speakers"`) - done 30.10: a Viterbi pass over
   1 s word embeddings (ResNet293) with a switch penalty and a bonus for the diarizer's own changes:
   S20 F1 0.856 -> 0.878 (recall 0.828 -> 0.873), S51 lines 61.7% -> 63.5%, people merged 41 -> 39;
   better on 9 of 9 test episodes (`docs/stt_local.md`, Models). On by default (switch 0.3, change
   bonus 0.15, no label prior). Side effect fixed on the way: the cue `overlap` flag now means two
   diarized speakers at once, not a label disagreement (the dub dropped relabelled shouts as
   "interjection over another speaker"); S51E01-E05 365 -> 205 flagged cues. Where the gain comes from
   (bench of 30.10, same episodes): only giving a voice to the words the diarizer left without a label
   (849 of 1006 words on S51) reaches S51 62.9%, S20 F1 0.863; moving labelled words adds the rest. So on
   S51 two thirds of the gain is the filling, on S20 two thirds is the moving.
3. **Text as the second signal at the boundary** - measured 30.10, not plugged in: gpt-6.1-sol through
   the subscription read S20 E01/E05/E10/E14 (~100 s per episode) and judged the 2180 sentence
   boundaries we give to one speaker, of which 12.5% are caption turns. Its claims are right 65-74%,
   and it finds 29-39% of those turns. Even if the acoustics then found the right person for every
   claim, F1 would go 0.878 -> 0.890 at most. A real but small signal (`output/exp_boundary/text/`).
4. **DiCoW / SE-DiCoW** - tried 30.10, dropped (`docs/experiments.md`): they do not judge who speaks,
   they follow the mask they are given.
5. Candidates after that, measured the same way, cheapest first:
   - DiariZen's AHC threshold 0.4 with this chain - measured 30.10, not taken: S51 64.6% vs 63.5%, S20
     F1 0.868 vs 0.878 (`docs/experiments.md`); the option stays, `"ahc_threshold"`, 0.5 by default;
   - season voice bank people as extra Viterbi states - measured 30.10, dropped (`docs/experiments.md`);
   - voting with other diarizers (pyannote, Sortformer v2.1, Nemotron-3-Diarization on windows) - measured
     30.10, dropped, no DOVER-Lap run (`docs/experiments.md`): each is far below our chain, and a change
     all three agree on and we lack is right only 34% of the time;
   - a two-voice separator (ClearerVoice MossFormer2) on overlapped stretches - measured 30.10, not taken
     (`docs/experiments.md`): within the noise, +60-200 s per episode and one more environment.
   Not worth trying: loudness levelling and the centre channel (`docs/experiments.md`), more audio LLMs
   on the joined places (step 3 of the night plan: none beat the pipeline).
   The settings of steps 1-2 were picked on the same 9 episodes they are scored on. Checked 30.10 on
   S29E01/E03/E07, not used for tuning: F1 0.824 -> 0.840 -> 0.861 (before, exclusive layer, Viterbi),
   every step better on each episode. S51's +1.8 points are close to the noise of a few lines.
   The acoustic candidates of this list are now used up; what is left is the text signal of step 3.
6. The user listened 30.10: speakers split a little better by ear, the mixed voice is heard more often.
   The chain stays the default; S20 / S29 are re-voiced only when the user says so. S51E01 was
   voiced 30.10 with all three engines (`STT_Season_51_speakers_v2/`, the old one stays in `STT_Season_51/`;
   list of the 283 of 1395 lines that changed person: `output/revoice_v2/listen_guide.md`). Waiting for the
   user's ear. Known side effect: the mixed cluster, voiced with generic voices, grew to 162 lines (12%)
   from 108 (8%), ordinary phrases among the shouts; SDH on this episode 65.3% -> 64.9%. If it bothers the
   ear, the fallback is the filling alone (`"prior"` 9: S51 62.9%, S20 F1 0.863; that cluster on S51E01: 96
   cues instead of 162). Sending such lines back is no fix: of 13 SDH-labelled lines the Viterbi moved into it,
   the diarizer's label was the person's main cluster for 3. The others had a wrong named voice before.

## Plan agreed 29.10.2026 (night): people merged, joined lines, audio LLMs

1. **DiariZen's AHC threshold** (0.5 now): lower / higher, scored like step 2 below (S51E01-E05,
   then S10). - stopped by the user 30.10 before the S10 check. S51E01-E05 after the refinement:
   0.5 (now) 63.4% / 45 of 101 merged; 0.4 64.5% / 45; 0.3 59.5% / 62; 0.5 with Fb 0.6 62.7% / 39.
2. **Where the speaker errors sit**: the share of wrong labelled lines inside a cue that joins two
   people's words (SDH: two differently labelled lines in one cue), and a set of such joined places.
   - done 29.10 (numbers in step 3)
3. **Audio LLMs on OpenRouter** (models that transcribe and tell speakers apart) on those joined
   places: the clip, the candidate speakers and our transcript in, who says what out; scored on
   the SDH labels, the cost stated first. This sends audio out of the machine (only these test
   clips). - done 30.10, no gain: on the 45 joined places (121 labelled lines) our pipeline 45.5%
   right, gemini-4-argon 45.5%, gpt-audio-mini 41.3%, gemini-3.8-flash 44.0% (12 broken answers),
   qwen3.8-omni-flash 36.4% (too slow, stopped); $0.64 in all. A third of those lines: the right
   person is not among the speakers our pipeline has nearby (merged into another voice), so no
   choice can fix them. Where the errors sit (step 2, done): of the 886 labelled lines 61.9%
   right; of the wrong ones 67% next to one of our speaker changes (<= 1 s), 16% in a cue joining
   two labelled people (45 such cues), 17% a whole turn on another voice.
4. **New speaker diarization models on Hugging Face** (2026): what is there, what is worth a test. - done
   29.10: nvidia/Nemotron-3-Diarization (Sept 2026, up to 8 speakers, OpenMDW licence: for short
   windows, not whole episodes), BUT-FIT SE-DiCoW / DiCoW v3.3 (Whisper that transcribes each
   speaker given DiariZen's output: the words at speaker changes), pyannote community-1 (in use as
   the fallback); nothing that beats DiariZen v2 on whole episodes.
5. **AGENTS.md** brought up to date (environments, RUAccent per cue, the new modules). - done 29.10
6. **Re-voice S51E01** with the stress fix and the number fix (`--force stress`). - not started (the user stopped the plan)

## Plan agreed 29.10.2026 (evening): stability of the speakers and the text, measured

Measured without voicing, free (LLM steps through the subscription). Status kept here.

1. **A measurement base**: S51E02-E05 recognised up to the speaker naming (`stt.run --until
   speakers`), their AMZN SDH subtitles (speaker labels, ~350 a episode) in `output/s51_subs/`;
   `stt.sdh_eval` on E01-E05. S51E01 baseline: diarization purity 66.3% (refined), labelled
   cues on their person's main speaker 65.3%, ~76% voiced under the right name. - done 29.10: E01-E05
   (886 labelled lines): on their person's main speaker 62.0%, 44 of 103 people merged with
   another, 65.7% voiced under the right name (E02 the worst: 56.8%, 13 of 21 merged).
2. **Speakers**: a newer speaker-embedding model (ReDimNet2) for the refinement (and the voice
   bank if it helps), and a speaker-count hint from the season cast for the diarizer; each
   against the base of step 1. - done 29.10, no default changed: ReDimNet2-B6 + VoxBlink2 (margin
   0.05) on S51 62.0% -> 63.1%, merged 44 -> 34, names 65.8% -> 66.5%, but on Australian S10
   61.1% -> 59.9-60.5% (merged 37 -> 39): within the noise of a few lines. The count hint does
   nothing: DiariZen's VBx ignores the speaker count. `docs/experiments.md`.
3. **ASR on the Demucs vocal stem** (Cohere + Whisper + fusion + alignment), WER against the
   episodes' own English subtitles, against the plain audio. - done 29.10, dropped: fusion WER
   13.84% plain vs 14.04% on the vocal stem (S51E01-E05).
4. **Number expansion**: ru-normalizr (MIT, pymorphy3) for numbers, our own rules for times,
   letter abbreviations and number ranges ("3-4 года" -> "три-четыре года"); on 64 test phrases
   rutextnorm made ~15 mistakes ("Мне 22 года" -> "двадцать второго года"), ru-normalizr 3 (the
   ranges). - done 29.10: numbers by ru-normalizr sentence by sentence, ranges and "A или B" by
   our rule, times / scores / letter abbreviations by rutextnorm as before; the 64 phrases right
   (only the counting form changed: "пять килограмм"); `tests/test_voiceover_normalize.py`.

Left out on purpose: DiCoW (only if step 2 shows the errors sit at speaker boundaries),
Sortformer v2.1 / multitalker-parakeet (at most 4 speakers), the translation check (paid second
model), The Amazing Race (after these), overlaps / chants / the crowd layer, short one-word lines,
OCR details, Vosk with 8 processes (no gain: Vosk ends long before ESpeech).

## Next steps (29.10.2026)

Done on 29.10: the cleanup (`docs/experiments.md`: what helped, what did not), the TTS cache
cleanup command and UI buttons, one stress per name for the season (`<season>.stress.json`),
the stress vote through the subscriptions (sol + gemini-4-argon and claude-haiku-5.5 of OpenCode
Go: S29E03 the same 27 corrections as the paid trio), Vosk in a CPU lane beside the GPU engines
(a 10-min piece in all three engines 25.4 -> 16.4 min), a TeraTTS voice for the host in the Vosk
and TeraTTS tracks (8 candidates in `output/host_voices/` for the user to pick). Dropped by the
user: several clips per person by the mood of the line (ESpeech; some clusters still mix people).

1. ~~**The host's voice**~~: the user picked `eng_m3+eng_m4` (29.10), now the default `host_voice`.
2. **Stability first** (the user, 29.10): no re-voicing of S20 / S29 and no big batches until the
   user calls the pipeline stable; single test episodes meanwhile. S51E01 (86 min, new season, all
   three engines in two lanes) ran on its own in 1 h 31 min without a failure (`docs/handoff.md`).
   Then re-voice S20 and S29 and finish S29 (E11, E13-E15, the missing tracks of E01-E08);
   afterwards `voiceover.cache_clean --delete` drops the old takes.
3. Smaller: a clean-machine test of `scripts/install.sh`, Vosk 0.11 when it is out, newer
   diarization / speaker-embedding models, the translation check only if it measurably helps.

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
- ~~**A warning in the UI when the ChatGPT subscription login of opencode has expired**~~: done
  (the check before a run, the UI status and the fallback warning; `docs/experiments.md`).
- ~~**TTS cache**~~: done 29.10 (`voiceover.cache_clean`, the buttons in "Настройки").

## Not planned

- **MOSS-Transcribe-Diarize as a third input of the ASR fusion.** It would be one more full run
  (~1.5 min per 10 min of audio, 3.1 GB VRAM) and a third input did not help before (Qwen3-ASR:
  within the LLM's run-to-run spread). Alone it is close to the fusion of Cohere + Whisper
  (9.6% vs 9.1% WER on three 10-min clips of S10E01), not better, so it cannot replace them
  either. Its speaker separation was better than pyannote on one clip and worse on two.
- **Nemotron-3-Diarization**: at most 8 speakers; episodes have 16-21.
