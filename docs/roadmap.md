# Roadmap

What comes after the current pipeline (`docs/stt_local.md`). Numbers are from S29E01/E02
(Survivor: San Juan del Sur, 64 and 43 min) unless noted.
The experiment folders named below (`output/stt_dz_*`, `output/tts_eval/*`, `models/fireredtts3`,
...) were deleted on 29.10.2026; what each showed is summed up in `docs/experiments.md`.

## Fix of 12.11.2026: lines next to a burned-in caption were not voiced (the user's report)

The user (12.11): where the show captions a quiet speaker, the line of another person right next to the caption was
not voiced either, although it is not on screen. Two causes, both measured on S48E01-E13 and S49E01-E10 with the
official subtitles as the reference (they leave the burned-in captions out, so a line they have is real speech):

1. **The dub's rule of 5.11** (`text` mode left out every line with 30% of its time under a caption ±0.4 s): 925
   lines, 579 of them in the official subtitles ("I appreciate that.", "But I don't want to be on the outs in this
   tribe.", Probst's "A massive Survivor toolkit."). Now such a line is left out only when it is the caption's own
   speech as the ASR heard it (`stt.dub.caption_copy`: a fragment of its words, a near-verbatim copy, or a look-alike
   of a caption with no line of its own): 571 of the 579 are voiced, 89 copies are still left out. The same check
   replaced the `neutral` mode's (which left out 224 of the 579 and said the caption's text instead of 58).
2. **The caption merge** (`stt.assemble.merge_captions`) swallowed a whole other sentence said while the caption was
   shown ("I've been wanting to talk to you alone." took "I know, me too." with it): now only when the caption found
   none of its words in the ASR. Of the 46 sentences freed, 32 are in the official subtitles (mostly "Yeah.", also
   "You got it, girl.", "Or we can take two people too."); 9 captions more get their own line. It reaches an episode
   only with a new transcript (`--force transcribe`, which redoes the stages after it too); the dub fix needs only
   `--force dub` (the voice-over cache keeps the lines already synthesized).

3. **The official subtitles decide** where the episode has them (the user asked for it the same day): a line under
   a caption that they have is voiced, a line they lack is the caption's own speech (`stt.dub.subtitle_shares`, moved
   onto the recognized words, numbers spelled out). S48E01-E13 + S49E01-E10: 0 of the 579 real lines left out, 282 of
   the 285 lines the subtitles lack left out (text alone: 8 and 89), so the misheard copy "I don't like being in the
   snow bathroom" under "I really needed this so badly" is silent again. Without fitting subtitles (another episode,
   none found) the text check decides as before. A line of 4+ words that they lack but that is unlike the caption is
   left to the text check (broadcast captions skip background talk; the user, 13.11: "лучше так, чем терять
   контекст"): S48/S49 256 of the 285 left out, real still 0.
4. **Older seasons** (the user asked for an estimate): the subtitles fit them too (86-99% of the lines found), but
   S21 (archive) and S29 (rolling broadcast captions) also write out the burned-in captions (45-86% of them); such a
   file is recognised and its subtitles that repeat a caption, also two captions at once, are set aside. S20 and S29
   were re-cut with today's caption merge on their saved words. Lines under captions, my reading of every
   disagreement and a sample of the rest (texts, not listening): S01/S02/S05 (16 episodes, 88 lines; the "captions"
   are mostly name and place graphics, the 5.11 rule silenced all of them) text 87 voiced, subtitles 84 (4 short lines
   the subtitles lack); S20 (15 episodes, 183 lines) subtitles about 93% right against 73% for the text alone (it
   voices ~43 whisper fragments and misheard copies); S21/S29 (11 episodes, 211 lines) about 90% against 76%, the
   loss being ~15-20 short real lines the rolling captions skipped.
5. **S29E07 run afresh** (13.11, `STT_listen_2026-11-12_captions`, the older episode with the most captions: 140; its
   own embedded subtitles are rolling broadcast captions that write out the captions too, 128 set aside): of the 56
   lines under captions (read one by one) 27 are other people, 18 misheard copies or pieces, 11 short and unclear.
   Rule of 5.11: 0 of the 27 voiced; text alone: 27, but also 16 of the 18; subtitles: 20 and 0; subtitles with the
   4+ words exception: 24 and 1 (the 3 lost are one-word pieces of a split sentence).
6. **Better subtitles for S1-S39** (scouts of 13.11, `subtitles_hunt/scouts_20261113/`): no source of Paramount+ or
   other streaming SDH for the old seasons was found; scene WEB releases exist for S17-S40 with no separate subtitle
   files, and the streaming copies seen so far (S29 WEB-DL, AMZN S11-S33) carry the old broadcast captions, so such
   subtitles most likely do not exist for those seasons.

## Plan of 10.11.2026: a voice where people shout (the user's request)

The user (10.11): where people on screen shout (challenges, arguments) the dub is often silent; a neutral voice is
fine, but something must be said so the viewer gets the context. Measured first (`output/exp_fill/silence_table.py`,
`gap_causes.py`, `gap_models.py` on the finished S48E01-E07 of 9.11, S51E01 of 29.10 and S01E01-E05 of 7-9.11), the
unvoiced speech of an episode falls into four causes:

| cause | S48E01 (late, 87 min) | S48E02-E07 | S01E01-E05 (early, 44 min) |
|---|---|---|---|
| lines captioned on screen, shown as text only (`on_screen.voice: text`) | 361 s, 212 lines; the longest silent stretches of the episode (36, 15, 14 s) are all captions | 123-227 s per episode | 5-14 s (the old show rarely captions) |
| interjections the dub left out ("Yeah!", "Go, go!", `dub.interjections: dense`) | 25 s, 30 lines | 7-26 s | 4-13 s |
| stretches with speech by the VAD and no recognised word at all | 153 s in 46 stretches (20 s in 3 over 5 s) | 81-144 s | 102-132 s in 23-31 stretches, 23-61 s in 2-7 stretches over 5 s: the challenge climaxes (S01E01 24:27-24:44, 28:34-29:01: both models hear one "go!" and "Here we go.") |
| crowd noise the captioners also wrote as `[cheering]`, `[indistinct chatter]` | 50 tag-only cues | 43-50 | no reference |

So in the late seasons the silence is mostly the caption rule of 5.11 (the show captions exactly the loud scenes),
in the early seasons it is the recognition. SDH reference cues with no recognised word within 0.25 s are only 2.1% of
the S48 cues (10.8% of the shout-like ones: `[cheering]` tags or "!"), i.e. the recognition itself loses little that a
captioner wrote down; the earlier measurement of 3.11 (`docs/experiments.md`, `output/exp_shout/asr/`) found Whisper
on such a stretch alone recovers half of the lines it missed in the long chunk.

Done 10.11 (branch `feat/neutral-voice-for-shouted-speech`), checked by unit tests and on CPU only (the GPU was
busy with season 49):

1. **Captioned lines stay as text** (the user's decision 11.11: whether a man or a woman says a captioned line is
   rarely clear, the original voice with the Russian text over it is better than a guessed voice), so the biggest
   cause of silence in the late seasons stays by choice. The `neutral` mode of `on_screen.voice` was improved for
   whoever switches it on: an ASR line mostly under a caption (the 84 lines of S48E01 that `text` mode silences) is
   compared with the caption's text (`onscreen.ru.json`): a copy of a caption that has its own line is left out, a
   caption with no line of its own is said in the caption's words at the ASR line's time (`caption_text`), other
   speech next to a caption stays as heard (`docs/stt_local.md`, On-screen text). S48E01 dub script in that mode:
   1126 lines (56.5 min) -> 1344 (62.6 min); with `text` kept: S51E01 1301 -> 1347, S01E01 554 -> 566 (interjections).
2. **Interjections** in a rapid exchange are voiced like any line (`dub.interjections` `dense` -> `overlap`); the ones
   on top of another speaker go to the shout layer: `stt.crowd` generalised (`crowd.enabled` true, `solo` true, one
   voice: the engine's generic voice of the gender, taken from `dub.report.json` or `--generic-voices`, so the ESpeech
   track uses its TeraTTS fallback voice), mixed 3 dB under the voice-over without ducking the original; lone
   backchannels ("Да.", "Ага.") stay out. Checked on CPU with Vosk on S01E01's 14 left-out lines.
3. **Second recognition pass** `stt.fill` (`fill` in the config, on by default): Silero VAD speech minus the final
   words -> stretches of 1 s or more, Whisper large-v3 on each stretch alone, guards against invented text (word
   probability under 0.4, stock phrases such as "Thank you", an echo of the neighbouring known speech), the words join
   the stream before speaker assignment as `source: "fill"`, their cues carry `fill: true` and are voiced by the
   neutral voice (`fill.voice`). Not yet run on an episode.

Measured 11.11 on the GPU (`output/exp_fill/eval_fill.py`, `docs/experiments.md`): the fill pass listens to 85-153
stretches per episode (4-7 min), nearly all music and crowd noise; Whisper answers "Thank you." / "Thanks for watching!"
on 46-60 of them and unsure cries ("Woo!", "No!", "Go!") on 18-67. With the guards and `min_score` 0.7 what stays is
2 / 1 / 1 / 6 lines on S48E01 / E02 / E03 / S51E01 (invented 0 / 1 / 0 / 0; 67-100% of their words in the SDH text:
"Here we go! Let's do it!", "Come on, paddle!", "Let's go! Let's go!", "Almost guys! Come on guys!") and 8 / 3 / 2
lines on S01E01 / E02 / E03 ("Should we take a break? No! Keep paddling!", "Off the ground. Off the ground."). At 0.4
there were 11-25 lines per episode, 7-16 of them invented. So the pass is cheap (10-16 s of GPU per episode) and nearly
clean, but small: the SDH lines the main recognition misses mostly sit under other recognised speech, where a second
listening cannot help. Kept on by default at 0.7.

The user then pointed at S01E01 1:30-3:00 and 4:40-5:40 (the boat and the raft): people talk on screen, the dub is
empty. Found there: 157-163 s is shouting of several people over music that no model hears even on the clip alone
(the words Whisper did get, "Here we go", are smeared over 157-169 s), and 326-337 s is 11 s where only the fill pass
hears something. But the episode's subtitles (tvsubtitles, in sync, shift 1.26 s) carry exactly these lines, and the
pipeline took only the speaker names from them, dropping every line the audio did not confirm. So the fill takes the
subtitles first (`fill.subtitles`, `stt.fill.from_subtitles`, `docs/stt_local.md`): unconfirmed lines whose time is free
of words, as written, the labelled ones by their person. S01E01: 94 lines (342 words; "Listen up!", "Put them on!",
"We got a great view of Snake Island! Get to work. We got oars. We got buckets."), S01E02-E05 36-100 lines each,
S48E01 145 lines (667 words), S48E02/E03 7 and 3: far more than Whisper's second pass, and the captioner's text.
Listening pieces of 11.11 for the user: S01E01 1:00-6:00 and 23:30-30:30, S48E01 8:30-12:30 with the three engines,
captions as text (the user's choice), interjections voiced, the shout layer, the subtitle fill and the Whisper fill on:
`STT_listen_2026-11-11_shouts/` (the subtitle fill added 18 / 46 / 9 lines to the three pieces: the boat piece now says
"Слушайте!", "Надевайте! Вперёд! За дело!" at 2:39-2:45 and the raft lines at 5:18-5:36; Whisper's pass then found
nothing more there). The user listened (11.11): the subtitle lines are fine. Asked next: also take from the subtitles
the words a partly recognised line lacks (the start eaten by the host's shout). Done the same day (`fill.edges`,
`docs/stt_local.md`): the matched runs of a pinned line are kept, the unmatched start/end/middle words go into the
free time next to the matched word; 34-63 runs (48-85 words) per episode, 98% of them in the SDH text when the
source is an independent file (`docs/experiments.md`). The three pieces were rebuilt with it (the earlier ones in
`old_12-57_without_edges/`); the branch `feat/neutral-voice-for-shouted-speech` stays uncommitted until the user
has listened.

**Other recognisers (11.11, the user's question; research report `reports/Дообученный Whisper и лучшие распознаватели
речи.md`, measurements in `docs/experiments.md`).** No fine-tuned Whisper beats large-v3 on our shouts (CrisperWhisper
2.0: 30% of the unheard lines vs 50%); WhisperX is alive (3.8.6, May 2026) but would add nothing we lack; the Qwen
forced aligner is no better than our CTC on shout timing. Of 16 cloud recognisers on the 581 unheard clips, only
Microsoft MAI-Transcribe-2 (58%, $0.10 per audio hour) clearly beats our Whisper (50%), and ElevenLabs Scribe v2 (51%)
matches it while inventing almost nothing on noise; the oracle "best of all models" reaches 80%, so the gain is in
combining. Next step, if the user wants it: an OpenRouter speech-to-text backend in `stt.asr` (MAI-Transcribe-2 and
Scribe v2 with word timestamps from `verbose_json`) used (a) as the recogniser of the fill pass instead of Whisper
(4-7 min of audio per episode: about one cent; no word probabilities, so the guards become stock phrases, event tags
and echo only) and (b) as a third source of the LLM fusion for the whole episode (about 8 cents per episode), both
measured on S51E01-E05 against the SDH text before any default changes. The idea of
separating the loudest voice from the crowd before recognition is last in line: the vocal stem made Whisper worse on
exactly these clips (3.11: recovered 46% vs 50%, three times more invented phrases), and the two-voice separators
measured 30.10 gave nothing for the speakers; it would be tried on the 581 clips of `output/exp_shout/asr/` first.

## Proposal of 10.11.2026: who speaks in any show (voice identities first, names second)

Done on 10.11.2026 (branch `docs/universal-speaker-identification`): steps 1-3 of the plan below (identities in the voice
bank and the season voices, clips keyed by name or identity, a clip stitched from short solo pieces when no 5 s stretch
passes, a stand-in pool per person; `voice_bank.identities`, `voiceover.engines.f5.standin_pool`, `--no-stitch`); the
Survivor stand (`output/exp_identities/`) and The Amazing Race S01E01-E03 (E01 voiced): numbers in `docs/experiments.md`
(10.11 rows). Show profiles (the user's request after listening, 10.11 evening): the base config keeps the Survivor behaviour
(`voice_bank.identities` and `voice_bank.stitch` false, no `standin_pool`); `shows` in `stt_config.json` holds per-show
settings matched by the video's path words (`stt.run.show_of`, `--show <name>` / `--show none`); The Amazing Race turns
the three features on. Not done: the card parser for team cards, the generic cast prompt (step 4), the self-clone
fallback; a whole season.

Asked by the user: the Survivor way of finding the speaker does not carry over to another show (The Amazing Race trial
below: 4-6 of 22 racers per episode not separated, 26-39% of the F5 lines in a stand-in voice), and future shows need a
way that does not depend on knowing who speaks: collect the voices of the season, clone each, and give every line to a
voice by its print. Not started; nothing below is measured yet.

Where the Survivor way is tied to the name (the facts behind the plan):
- A person exists only with a name. `voice_bank.add_episode` keeps a print only for a named cluster (`MIN_SPEECH_S`,
  `sure_name`); `season_voices.build_voices` builds a season centroid only for a named person (`min_words`); an unnamed
  cluster stays a diarizer state of its own episode and is forgotten in the next one.
- An unnamed cluster is voiced by the one generic voice of its gender (`dub.py`): every unnamed person of a gender
  shares one voice for the whole season (S01E01 of The Amazing Race: clusters 23, named 19, with an own clone 11).
- The name evidence is Survivor-shaped: a card with one name (a two-name team card counts as mixed and gives no voice
  evidence); tribe, host and place regexes (`stt/tribes.py`, `assemble.PLACE_WORDS`, `subtitles.HOST_LABELS`,
  `GENERIC_LABELS`); the cast prompt ("tribes", "voted out", `stt/cast.py`); the "Survivor-style" prompt of
  `stt/onscreen.py`; `subtitle_base` defaults to `show="survivor"`.
- A reference clip needs 5-9 s of one person (`SAMPLE_MIN_S`); an interview of a team rarely has it.
- What is already show-independent: diarization, the ResNet293 refinement, the word-level Viterbi, the season voices
  pass itself, the subtitle labels, the voice-bank thresholds, the line gender, the reference-clip gates.

The plan: an identity is a voice print of the season; a name is a label on it.
1. **Identity bank** (audio only, no show knowledge). Prints from the clean solo stretches of every cluster with enough
   speech, named or not (`solo_stretches`: one diarized speaker, no overlap, music gate, pieces >= 1.5 s), matched across
   episodes with the voice-bank thresholds (`STRONG` 0.75, `MARGIN` 0.1); no match = a new identity `V<k>`. Per identity:
   centroid (interview / field sub-centroids when they split), audio gender (`line_gender` votes), speech seconds,
   episodes, reference clips, the Russian voice. The season bank grows episode by episode, as the voice bank does now.
2. **Word pass over identities** + the episode's clusters: the season voices Viterbi as it is, with identities as the
   people. A cluster is dissolved when its words spread over several identities (a voice measure), not only when the
   LLM's role text says "mixed".
3. **Voice-over per identity.** A clone for every identity with a usable clip. When no 5 s solo stretch exists (team
   interviews), a clip stitched from several 2-4 s solo pieces of the identity's own words. Identities without a clip
   get a fixed stand-in from a pool (several Vosk voices cloned by F5 through `fallback_references`, the TeraTTS voices),
   chosen once per identity for the season, so two unnamed people never share one voice.
4. **Names as labels.** Evidence, all show-independent: subtitle labels ("NAME:"); any on-screen text near speech parsed
   into {names, kind} (one name; "A & B" team; "NAME, city" lower third; place; title roll), a team card giving two
   candidates split by gender, by the two voices under the card and by the dialogue; the cast list from a generic prompt
   (host, participants, gender, relations); the dialogue itself (self-introductions, being addressed). A name attaches to
   an identity with a confidence; two identities with the same sure name merge when their voices allow, else they stay
   two voices of one person. Names feed the glossary, the translation and the UI; the voice-over needs none.
5. **Show profile** in the config (`show`: cast prompt wording, card grammar, generic and host labels, IMDb id); the
   Survivor-only pieces (tribes, place words, the "Survivor-style" prompt) move behind it. Nothing changes for Survivor.

Measurement first. The experiments already show the risks: the diarizer's merges go into identities built without labels
(cluster centroids put 61.9% of the words on their person, a person's own labelled words 79.8%); interview and field
speech of one person sit apart (cosine 0.335 vs 0.428) and may become two identities; stitched clips are untested in
ESpeech; r3 and the gender model were trained on Survivor.
0. Harness: S51E01-E05 (SDH labels), The Amazing Race S01E01 (97 "NAME:" lines, 25 labels; S07 has labels too), old
   S02/S05 (no names); stock ResNet293 and r3. Scores: hidden labelled lines on their person; identities per person and
   persons per identity (B-cubed, `output/exp_spk/cluster_quality.py`); lines in a shared or stand-in voice; a new one,
   the share of a person's lines voiced by the same Russian voice over the season (what the viewer hears).
1. Identities without names on that harness against the named season voices (S51 71.6%). Decides the design: identities
   alone, or identities + names (the expected answer; Survivor loses nothing either way). A day of agent work: the word
   embeddings of those episodes are cached (`words.emb.<model>.npz`).
2. The stand-in pool per identity: small, show-independent, a listening check.
3. Stitched reference clips: blind listening against the 8 s clip of the same person (8 people x 6 lines, as in
   `output/exp_voice/`).
4. The card parser and the team split; the generic cast prompt; the show profile.
5. A whole season of The Amazing Race S01 (more episodes per person, fewer racers later); the user's listening.

Cloning from the line itself (the user's idea, 10.11: the diarizer cuts the stretch, the TTS clones the voice of that
stretch, no identification at all). Cost is not the obstacle: `f5_worker` loads the reference inside every
`infer_process` call and generates reference + line seconds each time, so a reference per line costs the same F5 time as
one per person; cutting and gating a clip per line is seconds per episode. The obstacle is the audio: on S01E01 of The
Amazing Race the median line is 1.8 s, 6% of the lines are 5 s or longer, and 25% have 5 s of clean speech of their
own speaker around them (same cluster, gaps under 1 s, no overlap; 41% have 3 s). ESpeech needs 5-9 s (`SAMPLE_MIN_S`).
A clip taken in the moment also copies the moment (shouting, laughing, music, wind, a second voice): the person's voice
would change from line to line, which the user already refused for the mood clips (29.10). Kept as the fallback of step
3: a line no identity claims, with 5 s of clean speech of its own, is cloned from itself before a stand-in is used.

Not in the plan: audio LLMs as the judge (21% vs 64% for the prints, 5.11); faces (dropped by the user); per-show
fine-tuning (per-season training dropped 1.11; adding the labelled Amazing Race seasons to r3's training set is a
later idea, measured on `sets/fair` first).

## Trial of 8.11.2026: The Amazing Race (another show, unchanged pipeline)

Asked by the user: how the pipeline works on another show. S01E01 (2001, SD AVI with MP3, online subtitles with 94
"NAME:" labels) and S20E01 (2012, HD, plain subtitles), the develop pipeline as it is, three voice-overs, one episode per
season. Finished files: `The Amazing Race/STT_trial_2026-11-08/STT_Season 01|20/`. Time: ~1 h per episode with all three
engines (ESpeech 21-24 min of it). Fixed on the way: an AVI source has MP3 packets without time stamps, the final mux
failed (`mux_tracks` now adds `-fflags +genpts` for AVI).

What went well: the cast (LLM) lists the host and all 22 racers of both seasons (S20 with two extra names), the host and the frequent racers are named
and voiced; timing fits (S01: 2 lines of 747 needed shorter text), the speech check found few faults (mean CER 1.2% /
2.5%); the S1 band is 11.5 kHz, no widening needed.

What the show breaks:
- **Team cards.** The Amazing Race captions a team ("ROB & BRENNAN", "BILL & JOE"), not a person; the pipeline treats a
  two-name card as mixed and takes no voice evidence from it. Not separated: S01 Brennan, Joe, Karyn, Kim (4 of 22
  racers); S20 Elliot and Andrew (twins), Nary, Jamie, Vanessa, Dave Gregg (6 of 22). Their lines went to the partner or to
  a same-gender racer; an "Instructor" took 37 lines of several people in the bungee scenes.
- **Group scenes.** 22 racers shout short lines at one task (S20 balloons): there the speakers are often wrong.
- **Reference clips.** The interviews are of both team members together, so a solo 8-10 s stretch is rare: own clone
  voice for 11 of 19 named people on S01 (39% of the F5 lines in a TeraTTS stand-in voice), 12 of 18 on S20 (26%).

Ideas, not started: a team card names two candidates (gender splits a mixed team; the scene after it belongs to that team,
like the tribe prior but much stronger); reference clips cut from the person's own words inside a team interview; a whole
season (more episodes per person, fewer racers later).

## Plan of 7.11.2026: who speaks in the old seasons (consilium GPT-6 Astra + Claude Fable + Opus)

Asked by the user: old seasons (SD audio, subtitles with few or no names) put many lines on the wrong or a shared voice
(S01: 22% "mixed voices"). Consilium findings on S01E01: DiariZen finds 8 voices for 17 people; centroids from a
person's own subtitle-named words find the right person for 85% of those words, DiariZen's clusters 47% (S51: 80 / 62%):
the grouping and the missing name evidence are the limit, not the voice prints. The S1 audio is narrowband (AAC 48 kbit/s,
nothing above ~6 kHz). Closed on the way: fan transcripts (the Survivor wiki has none for old seasons), ERes2NetV2 /
ECAPA2 / WavLM-SV, new diarizers, re-training for old audio, bandwidth extension. Kept for later: ReDimNet2+
(`lab260/redimnet2-plus`, Apache-2.0, trained for codec and channel shift).

0. **Harness** (`output/exp_old_seasons/`, not in git): S01E01-E03 (tuning), S02E03, S02E05, S05E01, S05E04 (test); the
   curated base subtitles with every "NAME:" removed (`turns`: only ">>" left, like most old seasons) or the names of every
   other scene removed (folds `A`, `B`); stock ResNet293 (r3 was trained on S1, S2, S5-S9); the code of `origin/develop`
   against the new code; score: hidden named lines on their person (`score.py`). Regression set: S48E01-E04, S51E01-E02
   with the production r3 voices (`sets/modern`).
1. **Fixes**: a state the subtitles made for a person ("P:SEAN") is that person's even with one named line (S01E01 Sean,
   Dirk stayed unnamed and never reached the season voices); a person only the subtitles or the cast name gets a season
   voice; the Russian label of a subtitle name is its glossary spelling (not the LLM's «Мастер костра»); ">>" marks can get
   their own bonus in the season voices (`season_voices.caption_bonus`, default unchanged 0.15); a dry run computes nothing;
   the host is named by the first clause of the LLM's role ("Host and narrator; cluster also includes contestant
   dialogue" left Jeff unnamed on S01E01 without subtitle names: 168 lines in a generic voice).
2. **Cards**: the cards of a title roll (opening roster, end credits) and of tribes, days and places are no evidence of who
   talks (`assemble.title_roll_cards`, `place_card`): on S01E01 the roster gave six names to one voice.
3. **Name-card voices**: a person with too few words otherwise gets a season voice from the words under their name cards
   (Susan's interview fell into the mixed cluster); the Viterbi pass decides which words are theirs.
4. **Tribe camps** (`stt.tribes`, `season_voices.tribe_prior`, off): after a tribe's place card the people of other
   tribes are a little less likely; the cast prompt now asks for the tribes and the episode a contestant left. Measured:
   no gain, kept off.

5. **Interview cards of old seasons** (`assemble.all_cards`): S01 shows "Rudy" over "Retired Navy SEAL - Tagi Tribe" in
   mixed case; only `onscreen.json` had those cards, so neither the naming nor the card voices ever saw an interview card of
   S1. **"JEFF V."** (two Jeffs in S02) now finds Jeff Varner in the cast (`cast.member_for_label`).
6. **Dissolving mixed clusters** (`season_voices.dissolve`, off): the words of a cluster the naming calls mixed go to the
   season's people. Old seasons with half the names: 69.2 -> 79.4%; without names about equal; S48/S51 +0.3/+1.1.

7. **Clean voices** (`season_voices.voice_source` "clean", off): a person's voice from card and named-line words only.
   Old seasons without names +2.4, with half the names +0.4 / S01 -3; S48 / S51 -1.9 / -4.3: not taken.
8. **A refused fusion batch** keeps the first recognition (`fusion.fuse`): a content filter on S01E05 stopped the whole
   season preparation.

Results (one run each; hidden named lines on their person; base = `origin/develop` at 3042314, new = 1-3 and 5 with
`dissolve: mixed`):

| | base | new |
|---|---:|---:|
| Old seasons, no names in the subtitles: S02/S05 (710 lines) | 61.0% | 70.6% |
| ... S01 (207 lines) | 75.8% | 86.0% |
| Old seasons, half the names hidden (fold A): S02/S05 (471) | 71.3% | 77.9% |
| ... S01 (98) | 77.6% | 89.8% |
| The other half of the names hidden (fold B): S02/S05 (239) / S01 (109) | 72.0 / 83.5% | 79.9 / 85.3% |
| S48E01-E04 / S51E01-E02, half the names hidden, r3 | 84.0 / 78.1% | 84.3 / 78.9% |
| The shared voice, old seasons without names / with half | 23.7 / 15.3% | 2.4 / 0% |

The LLM naming is not repeatable (one episode moves up to 6 points between runs): the gains hold on every group, single
episodes do not. Closed: tribe camps (+1 / -0.3..-1.2), ">>" bonus 0.3 (no change), clean voices (above).
Listening: Season 1 apart from `STT_Season 1` in `Survivor/STT_listen_2026-11-07_speakers/` (E01 voiced, `dissolve: mixed`).
Listened to and approved by the user on 7.11.2026: in develop, `season_voices.dissolve: "mixed"` on by default.
ReDimNet2+ tested after the commit (the user's wish): `make_embedder("redimnet2-plus")`; on old seasons +3..8 points over the
stock ResNet293, equal to r3 where r3 was trained on the people, S51 (unseen by r3) 79.5% vs 78.9%, both models together 80.3%.
Fine-tuned three times on 8.11 (`scripts/speaker_training/train.py --backbone redimnet2-plus`): on a fair set of people no model
knew (S51E01-E05, Australian S10 E01-E03, 1441 lines) none beats r3 (78.6%); r3 + stock ReDimNet2+ 79.5% (+13 lines net,
p = 0.06). r3 stays the default.
Voice quality (consilium 7.11, `docs/experiments.md`): the clones keep about half of a person's identity (ESpeech), the old
seasons' clones end at the references' 5.2 kHz, the voice sits 10 dB over the scene. Done 8.11 (the user's blind pick): the
F5 track of a narrowband season is widened by LavaSR v2 before the mix (`stt.widen`, `mix.widen`, `lava_env`). Next: fine-tune
ReDimNet2+ for the speakers, then compare other clone engines (VoxCPM2, Qwen3-TTS, CosyVoice3) with the stress check.

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

### Status and queue (6.11)
- Done and in develop (1ab39c8): speed-up (S48E07 65.7 min, S01E01 34.5 min for 44 min), GPU fallback
  (`gpu.parallel: auto`), subtitle labels -> cast names, bank prints for every named person.
- Done 6.11 (merged to develop on the user's word):
  - the host's voice-over named after the season's host (cast.json); naming again starts from the first pass, so the
    season voices no longer bring back old names (S01E01: Jeff 23 -> 142 lines, own ESpeech clip);
  - tab "Кто это?": unnamed groups per season with snippets, the answer written under the first-pass id;
  - item 3: `stt.run --prepare` and the UI button "Подготовить весь сезон" (state "подготовлена (ждёт перевода)");
    the next episode is translated while the current one is voiced (`overlap_episodes`, seen on a real two-piece run);
    with `--force translate` the translation ran 2-3 times per episode, now once;
  - item 4: names in subtitles in any case (hunt episodes with names 438 -> 707 of 742), one base
    `Survivor/subtitles_base` (`stt.subtitle_base`), lookup order with a season folder, other-cut fallback to the base's
    alternates, UI group "Субтитры" and a table column; translated episodes keep their subtitles;
  - ESpeech text: no quote marks in the synthesis text (a stray "ж" before quoted words: 6/24 takes -> 0/24),
    все/всё decided by the stress check with the whole line, "-таки" unstressed (found in the audiobook work).
- Closed: audio LLMs and clef-omni as speaker identifiers (8-21% vs voice prints 64%; docs/experiments.md).
- To check: background video compression on a real 2 GB episode (log line "compressing the video in the background");
  the A/B of ESpeech at 16 vs 32 steps waits for the user's ear; the effect of the host-only names of seasons 21-39 on
  who speaks is not measured yet (S29 with `stt.turn_eval`).
- Queue: references for new seasons glued from 2-3 clean pieces when no single 5-9 s stretch exists (the user's idea,
  6.11); a second-episode check of the clef tie-breaker only if wanted.

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
