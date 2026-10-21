# MAI whole-file requests, async status and honest progress

Evidence basis: anonymous primary-source research findings supplied for this writing
assignment, dated **2026-10-12**. No new HTTP request, credential access or paid test
was performed while persisting these notes. Service limits are current findings,
not historical model-release eligibility evidence. Links below are primary references,
not a claim that this writer independently fetched them again.

## OpenRouter returns a final response, not an audio job

`POST /api/v1/audio/transcriptions` with `microsoft/mai-transcribe-2` documents final
JSON, not STT streaming, asynchronous submission, job polling or server processing
percentage. The upstream **60-second processing deadline** is not an audio-duration
limit; increasing client HTTP timeout cannot override it. MAI/Azure inline requests
remain under the **25MB** documented upload limit; the guide's larger-file exceptions
are for OpenAI/Groq, not MAI. The guide includes MAI `mp3` input: the service is not
WAV-only merely because our safety-bounded client uses complete PCM WAVs.
([STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt),
[transcription reference](https://openrouter.ai/docs/api/api-reference/stt/create-transcription)).

The completed episode is **3363.085375 seconds**, with 53809366 mono PCM16/16kHz
samples: approximately **107.6MB of PCM**, exceeding that limit. Compression can
reduce bytes, but does not remove the processing deadline. These facts do not prove
that every compressed whole-file request is impossible; **no whole-56-minute MAI
call was verified**. Our approved run used 57 <=60s chunks and already completed.
([Actual episode summary](../../temp/australian_survivor_e04_mai/summary.json)).

OpenRouter batches exclude `audio_transcriptions` and audio/video parts; generic
batch infrastructure is not a substitute MAI audio job. `X-Generation-Id` identifies
cost/debug tracking, not a pollable transcription job. `session_id` groups requests
but is not forwarded to providers and does not create persistent speaker identity.
([Batch quickstart](https://openrouter.ai/docs/batch-quickstart),
[STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt)).

## Direct Azure is a different credentialed synchronous route

The MAI guide uses multipart
`/speechtotext/transcriptions:transcribe?api-version=2025-10-15`, with
`enhancedMode.enabled=true` and `enhancedMode.model='MAI-Transcribe-2'`. It returns
completed results synchronously (HTTP 200), not a submitted job. It requires a
separate Azure Speech resource key/region; an OpenRouter key is not interchangeable.
([MAI guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)).

The versioned REST reference gives **<2 hours / <250MB**. Broader general Speech
quotas advertise larger figures (<5 hours/<500MB); do not substitute them as a
verified MAI guarantee. Our 56-minute/107.6MB audio arithmetically fits the smaller
REST limits, but the MAI guide limits diarization to shorter audio without giving
a numeric threshold; public preview has no SLA. Thus whole-file transcription
eligibility does **not** establish whole-file diarization reliability.
([Versioned REST](https://learn.microsoft.com/en-us/rest/api/speechtotext/transcriptions/transcribe?view=rest-speechtotext-2025-10-15),
[MAI guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)).

Generic Azure batch transcription has `NotStarted`, `Running` and `Succeeded`
statuses. MAI compatibility is not documented by the supplied batch findings, so
neither MAI batch support nor percent-complete can be promised.
([Batch creation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription-create),
[Batch status](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription-get)).

## Labels and progress must match observable evidence

Native phrases expose `speaker`, `offsetMilliseconds` and `durationMilliseconds`.
Speaker IDs are input-scoped: speaker 0 across different requests need not be the
same human. OpenRouter copying phrase speaker IDs onto words does not imply an
independent acoustic word-speaker classifier. MAI does not return native isolated
speaker WAV stems or guarantee complete overlap recovery. Cutting the mixed waveform
at speaker timestamps still leaves other overlapping voices. Generic Azure's 35-
speaker limit is not verified here as a MAI-specific limit; its fast-transcription
feature matrix does not support MAI stereo channel processing.
([MAI guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe),
[Fast feature matrix](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create),
[OpenRouter STT](https://openrouter.ai/docs/guides/overview/multimodal/stt)).

A single synchronous call can truthfully expose **uploading / waiting / elapsed /
done**, not inferred audio-processing percentage. Chunk progress can measure unique
completed input seconds / total seconds, or completed chunks / 57; our existing log
already reports completion after each real chunk. This writing task adds **no progress
UI, endpoint or additional call**. Completed episode: **57 new requests, 660 native
cues, $0.09344444444444445**; shared historical ledger **$0.1011395548888889 / 72
reservations, all costs known**. Request-local IDs and all-cue review remain required.
([Run log](../../temp/australian_survivor_e04_mai/run.log),
[Manifest](../../temp/australian_survivor_e04_mai/manifest.json)).
