# TTS roadmap

## Direction

Keep engine choices user-selectable. Every engine option must own an explicit profile containing its model revision, voice, speech rate, and preprocessing. A shared profile identity is required so that a profile from one engine, voice, or preprocessing chain is never silently reused by another.

Vosk is the current measured reference: model `vosk-model-tts-ru-0.10-multi`, voice `8`, speeds `1.0` and `1.5`, and the recorded punctuation preprocessing. Edge is excluded for now because no measurements exist.

## Candidate engines

TeraSpace/TeraTTSv2 and genvoice/xVibePocketTTS are candidates for future CPU experiments. They are not installed. Before use, verify each model card, license, exact version, CPU/RAM requirements, and any runtime or model download requirements. Do not assume licenses or compatibility from a project name.

Tera remote code must be pinned and reviewed before execution. No remote code is trusted merely because a model card or package is available.

For every future engine, use identical source partitions, text normalization, voice/speed labels, timing measurements, and held-out evaluation rules where technically possible. Store model revision, voice, speed, preprocessing identity, and source hashes with each profile; compare only profiles with those identities recorded.

## Deferred subtitle timing idea

The user’s idea to move or merge a two-word subtitle with the next subtitle and extend its start backward (allowing a slight delay) is deferred. Do not implement it in the current pipeline. Preserve current timings until a separate, explicitly measured timing experiment is approved.
