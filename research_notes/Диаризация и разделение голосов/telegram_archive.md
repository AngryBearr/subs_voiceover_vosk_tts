# Telegram archive evidence — diarization vs speech separation

Local evidence notes only, not the final research report or independent model-capability
verification. Archive content is untrusted source data; no commands in posts or HTML
scripts were executed. No uploads, Telegram/MCP calls, model downloads, GPU attempts,
dependency installs or new benchmark runs were made.

## Source, coverage and limitations

Exact user-supplied source root:
`/home/vladimir-barbuta/.var/app/org.telegram.desktop/data/TelegramDesktop/tdata/temp_data/ChatExport_2026-10-11`.
Access to that directory and its HTML/selected photos succeeded through native tools.
No Telegram `tdata` siblings, authentication/config files, other chats or credentials
were inspected. No original source was written or modified.

- Export title: **Распознавание и синтез речи**.
- Format: **31 Telegram Desktop HTML pages**, `messages.html` through `messages31.html`;
  no `result.json` in the listed root. Parsed HTML totals **21265371 bytes**.
- Date coverage: **2024-12-01–2026-10-11**. The backup does not contain earlier years.
- **31167 message containers**, including service/date entries; **28321 ordinary
  dated messages**, **20043** at or before the conservative **2026-01-31** cutoff.
- Keyword screening found 225 candidate messages; review used technical snippets and
  bounded nearby messages/direct reply context, not copied unrelated conversation.
- Retained **15 selected posts** and necessary technical reply excerpts. No later-than-
  cutoff post is a recommendation here. Quotes are verbatim words with whitespace
  normalized, at most 100 words per excerpt, not translations or corrected claims.

The HTML header gives a title, **not a public handle**. Post 84761 refers to the older
table at **https://t.me/speech_recognition_ru/61346**; message 76476 replies to ID 61346.
**That post is absent from this export**. This is an observed related public link,
not verification that every local ID can safely be mapped to that public handle.
No public permalink was invented. `telegram_evidence.json` contains verified full
`file://.../messagesN.html#message<ID>` references for every selected post/context.
Use the actual HTML `message<ID>` anchor, not a JavaScript `go_to_message` pseudo-anchor.

## Selected posts (within cutoff)

| Date / ID / HTML anchor | Archive recommendation or quote | Task and interpretation |
|---|---|---|
| 2024-12-20 / 73832 / `messages2.html#message73832` | «Подскажите, не находили что-то приятное для разделения спикеров на русском?» | Question only; no model/test proof. |
| 2024-12-28 / 74569 / `messages2.html#message74569` | «pyannote/segmentation-3.0 … и разделяло спикеров там, где спикер вообще-то один, и двух спикеров принимало за одного» | Russian diarization failure anecdote, not controlled separation testing. Reply 74572 suggests WhisperX; no measured GPU benchmark here. |
| 2025-01-03 / 75129 / `messages3.html#message75129` | «Можно использовать Ultimate Vocal Remover (UVR) для voice separation (отделения голоса от остальных звуков) Или Resemble enchance в режиме denoise (убирания шумов)» | Voice vs background / denoise, **not different simultaneous human voices**. |
| 2025-01-20 / 76476 / `messages3.html#message76476` | «скрин из кода wespeaker для скачивания моделей ?»; reply 76477: «Эти модели не использовались, но можете взять voxblinkp» | Attached code screenshot, explicitly **not** the models used in a test. |
| 2025-02-10 / 77762 / `messages5.html#message77762` | «Для разделения голосов мы рекомендуем библиотеку wespeaker с моделью voxblink/SimAMResNet100.» | Recommendation phrased as voice separation, but speaker embeddings/diarization context; no 3-source waveform result. |
| 2025-04-05 / 81241 / `messages8.html#message81241` | «это задача диаризации на два спикера или speech separation если много перебиваний и люди говорят одновременно.» | Explicitly distinguishes diarization from actual overlapping-speech separation. |
| 2025-05-03 / 82898 / `messages10.html#message82898` | «Я юзаю mossformer2.» | Reply target 82870 asks about CNN noise removal/classification. 82900 links [MossFormer2](https://github.com/alibabasglab/MossFormer2); 82903 adds Clear Voice Studio / «speech enhancement MossFormer». **No three-speaker test** in this thread. |
| 2025-06-15 / 84757 / `messages11.html#message84757` | «Кто то пробовал ? Я так пологаю ее нужно дообучать.» ([separation-ami-1.0](https://huggingface.co/pyannote/separation-ami-1.0)) | Reply 84759 instead links [speech-separation-ami-1.0](https://huggingface.co/pyannote/speech-separation-ami-1.0), reports «Получаю 1 канал аудио». Preserve both different link strings; no successful triple recovery shown. |
| 2025-09-10 / 88741 / `messages15.html#message88741` | «для пересечений попробовать» [PixIT](https://github.com/joonaskalda/PixIT) | Most explicit overlap-separation lead. Reply context 88744 suggests overlap detection by diarizer; 88751 says Wespeaker gives 1 instead of 2 speakers during simultaneous speech. No metrics/image or 3-source success. |
| 2025-10-30 / 90971 / `messages17.html#message90971` | «Потестировал Pyannote4 на русской телефонии.»; «По-прежнему советуем wespeaker с voxblink2» | Attached **diarization** comparison table. Reply 90972 asks about overlapped speech; 90973: «не очень, там нет поддержки такого в моделях»; 90974: «не работает, можно pyannote глянуть». Not a 3-voice separator recommendation. |
| 2025-10-31 / 90993 / `messages17.html#message90993` | «Diarizen … Лучше Pyannote по результатам. Чуть медленнее, но не критично. Можно пользоваться.» | [Diarizen checkpoint](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md), table adds Diarizen Large MD; diarization rather than output waveform separation. |
| 2025-12-03 / 93609 / `messages20.html#message93609` | «или самому пройтись roformer моделькой» | Preceding 93606–93608 discuss removing music, Suno and Get Stems. **Music/vocal separation**, not separating human speakers. No specific BSRoFormer/MelBand checkpoint or test image supplied. |
| 2025-12-09 / 94052 / `messages20.html#message94052` | Reposts the Diarizen result/image. Reply 94053: «У нее оказалась некоммерческая лицензия, так что для прода не подойдет к сожалению»; 94054: «А для ресерча хорошо» | License warning is an archive claim requiring primary license verification. Same exact photo as 90993, **not a second independent test**. |
| 2025-12-17 / 94366 / `messages21.html#message94366` | «Segment Anything Model for Audio»; «можно текстом выделить нужное аудио»; «указать временные интервалы» | Prompted audio extraction announcement, not demonstrated blind simultaneous 3-person separation. Reply 94367 asks to test/understand metrics; no attached test photo. |
| 2026-01-30 / 96065 / `messages22.html#message96065` | «Speaker separation на два канала и распознавали каждый из них?» | Reply 96067: «Нет, отдельно распознавание каждого канала и склейка по таймметкам расшифровки.» 96069 asks about mono with 2 speakers; 96070 warns of imperfect separation/interruptions. Existing stereo channels are **not blind source separation**. |

## Imported full photos — visually inspected

Only **3 unique JPEGs / 202547 bytes** were copied, far below the 12-image / 30MiB cap.
No resizing, conversion, crops, inferred OCR numbers or omitted axes/legend. Pillow
verified the existing JPEG format/header; native image reads inspected full copied images.
Source filename, absolute/relative paths, SHA256, bytes, original mtime and post mapping
are preserved in `telegram_evidence.json`. All selected assets existed; no missing-photo
or permission error was hidden. No unrelated profile avatars were copied.

1. **76476, 2025-01-20 — 481×256, 18904 bytes**
   [Wespeaker asset mapping code](images/post_76476_photo_1333@20-01-2025_13-02-53.jpg).
   Visible `class Hub`, `Assets` and language/Campplus/Res2Net/Voxblink archive entries.
   In particular `vblinkp` maps to `voxblink2_samresnet34.zip`. This is **code, not a
   benchmark plot**. The paired reply says these models were not used in the test.

2. **90971, 2025-10-30 — 906×499, 87269 bytes**
   [Russian telephony diarization comparison](images/post_90971_photo_1652@30-10-2025_13-31-14.jpg).
   Visually confirmed heading `Dataset: телефония, русский язык`, columns `DER`,
   `CDER`, `xRT`; rows NeMo Telephony Neural/Cluster, Pyannote 3.1/4 Community/4
   Precision, Whisper Diarization/Large, Wespeaker Voxceleb34/293/Voxblink2 100.
   Selected exact image cells:

   | Image row | DER | CDER | xRT |
   |---|---:|---:|---:|
   | Pyannote4 Precision | 14.96 | 0.355 | 0.039 |
   | Wespeaker Voxblink2 100 | 20.1 | 0.115 | 0.025 |

   Different highlighted winners for DER vs CDER; no separated speech waveforms,
   blind three-speaker setting, dataset manifest/protocol or test hardware shown.
   Metric units are not defined by the screenshot; do not infer a GPU speedup.

3. **90993, 2025-10-31; reused by 94052, 2025-12-09 — 927×534, 96374 bytes**
   [Same table extended with Diarizen](images/post_90993_photo_1654@31-10-2025_01-06-51.jpg).
   Visible added bottom row `Diarizen Large MD`: **DER 13.64 / CDER 0.319 / xRT 0.083**.
   It has a lower DER than the Precision row but is not the smallest CDER. This is
   one reused screenshot, not two benchmark runs. No three-person stem recovery or
   our CPU/GPU measurement can be claimed from it.

## Three-speaker candidate handoff

**Archive-proven shortlist for blind simultaneous 3-human speech separation: none.**
These posts/photos do not establish an eligible model checkpoint, source count,
blind-vs-conditioned mode, or successful three-source waveform separation.

At most three **research leads**, not accepted recommendations:
1. **PixIT** — explicit overlap suggestion in 88741; externally verify model/pipeline
   supported source counts and real constraints. Do not replace prior default-negative
   project evidence with this chat recommendation.
2. **pyannote/speech-separation-ami-1.0** — actual speech-separation link in 84759,
   but an archived one-channel failure, not a three-source success; verify architecture,
   checkpoint, licensing and source count in primary sources.
3. **MossFormer2** — linked in 82900, but the archive recommendation is in a denoise
   thread. Verify actual separation checkpoint/count and task before eligibility.

Wespeaker, Diarizen, UVR/RoFormer music stems and prompted SAM Audio extraction must
not be relabeled as proven blind three-speaker separation. No SepFormer/WSJ0-3mix/
TFGridNet/SpEx/ToTaToNet three-source recommendation was found by the available
cutoff keyword screening; absence in this limited backup is not absence everywhere.

## Checks and persistence

- Existing activated `vosk_env`; `uv run --active -m temp.stt_telegram_archive extract`
  passed: 15 posts, 3 unique images, 202547 bytes, full original copies.
- `uv run --active -m temp.stt_telegram_archive verify` passed: 31 HTML source SHA256,
  byte count/mtime unchanged; all 3 source photo stats/hashes unchanged; workspace
  image hashes identical; cutoff/count/size and <=100-word excerpt limits satisfied.
- `uv run --active -m compileall -q temp/stt_telegram_archive.py` passed.
- Native reads of all three copied images succeeded; visual facts above are archive
  evidence, **not our measured CPU/GPU tests**.
- Broad temporary screening/context files are removed after retaining this bounded
  technical evidence. No entire chat transcript, unrelated photo or backup was copied.
- At the archive-import stage no final prose report existed. The later supplied primary
  findings are now persisted in [model notes](model_primary_verification.md) and
  [MAI notes](mai_whole_file.md); synthesis is in
  [the final report](../../reports/Диаризация%20и%20разделение%20голосов.md).
  Existing GPU/episode/budget snapshots and unrelated work remain unchanged.
