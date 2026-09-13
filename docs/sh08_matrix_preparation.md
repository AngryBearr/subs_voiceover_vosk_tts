# SH08 matrix preparation

**Status:** `PREPARATION_ONLY` / `NOT_EXECUTED`. This is an offline evidence
record and a concrete, unapproved protocol draft. No new model, network,
subscription, or paid calls are authorized. Local offline checks and use of
existing Vosk resources are permitted by the handoff, but no synthesis was
needed or performed during this continuation.

## Current offline inventory

The r2 inventory is a planner/corpus measurement, not a 24-case frozen
selection. It retains zero-capacity cases and keeps the invalid-case denominator
visible. Cases are excluded only after validated raw and canonical source hash
checks. The 48-case inventory has zero overlap with the three SH08 pilot files;
that is only an observation about these 48 cases, not proof of global training
independence.

| Policy | Cases | Invalid | Zero capacity | Positive capacity | Status counts |
|---|---:|---:|---:|---:|---|
| All / after pilot exclusions | 48 | 0 | 19 | 29 | 40 `needs_shortening` / 8 `short_timing` |
| Optional no `S44E11` | 42 | 0 | 15 | 27 | 35 `needs_shortening` / 7 `short_timing` |

The optional row removes the six `S44E11` reference cases; that policy choice is
not made here. A zero predicted character capacity does not prove that the
original speech fails to fit, and does not by itself prove that shortened speech
cannot fit. No WAV, TTS, or audio measurement was done in this continuation.
Historical observations remain separate: the prior 9-risk sample had 5 fits and
4 overflows; the prior 30 predicted-fit sample had 30/30 fits. Those are not 138
independent observations.

## Historical six-pair evidence

The canonical historical pair table is in
[`shortening_model_pairs.md`](shortening_model_pairs.md). The following is the
single requested comparison table; it does not rank models or establish truth.
The proposed settings are Luna `medium`, DeepSeek Flash `off`, and Sol
`medium`, all still unapproved. Historical stage B means accepted by Sol and
the old stage-B length check.

| Generator → arbiter | Proposed settings | Audited historical evidence |
|---|---|---|
| Luna → Luna | Luna medium → Luna medium | not found in audited artifacts |
| Luna → DeepSeek | Luna medium → Flash off | not found in audited artifacts |
| Luna → Sol | Luna medium → Sol medium | old stage B: 43/48 and 45/48 |
| DeepSeek → Luna | Flash off → Luna medium | not found in audited artifacts |
| DeepSeek → DeepSeek | Flash off → Flash off | not found in audited artifacts |
| DeepSeek → Sol | Flash off → Sol medium | old stage B: 38/48 and 31/48 |

Separately, historical Flash `low` had old stage-B counts 44/48 and 43/48;
this does not silently replace the proposed `off` variant. Each historical
48-case repetition used the same 48 cases: 48 twice is not 96 independent
cases. Historical counts are not comparable to a new rank and are not semantic
truth.

Historical limits were 13 characters per second, whereas the current comparison
uses the normalized calibrated profile. Luna and Sol are in the same model
family, so agreement between them is not independent confirmation.

## Evidence and verification boundaries

- r2 report: `output/sh08_matrix_inventory_20260914_r2/report.json`, SHA-256
  `ec231da712546367e5c5782965d2e5fd4544150a543cddb7e469ea19f6251b2b`.
- r2 helper: `output/sh08_matrix_inventory_20260914_r2/measurements.py`, SHA-256
  `b233e0c72a87b47f9ba34300e1128bc7e41d7db0b36f51aa19e9a7efa756d493`.
- Repeat report has the same payload/hash. The historical artifact is
  `output/shortening_verification_20260910/expanded_comparison_final.json`,
  SHA-256 `ccdc655d82f6eef0e5e760f498bcf24b2644a4807df6290092972985a8ffd0c7`.
- Source/profile links are [`sh08_pilot_manifest.json`](sh08_pilot_manifest.json)
  (SHA-256 `dc01852aff9032e713ab77da79babb6496057fdd97fdedd5323f93e5d9ade2f9`)
  and `output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json`
  (SHA-256 `eab509922c511ed0dbd2b3948d5a0cf55de78dc11ab6127e9c59443f4cabcb59`).
- The local baseline log is `/tmp/opencode/sh08-continuation-baseline-20260914.log`.
  It records `471 passed, 5 skipped in 42.59s`, compileall exit 0, and the
  `uv` no-sync warning outside a project. That warning is not an installation;
  no installation was done.
- The r2 repeat hashes are identical. Four mocked negative tamper checks fail
  before output writing. Outputs and models are Git-ignored and **MUST** be
  transferred separately; these links do not claim committed raw evidence.

## Draft recommendation (not authorization)

Retain six pairs with exact proposal IDs pending approval: generators
`openai/gpt-5.6-luna` (`medium`) and `deepseek/deepseek-flash` (`off`), and
arbiters `openai/gpt-5.6-luna` (`medium`), `deepseek/deepseek-flash` (`off`),
and `openai/gpt-5.6-sol` (`medium`). Use 24 total development cases. One
repetition is the economical option; two repetitions are the stability option.
Exclude the three SH08 pilot files mandatorily. Also recommend omitting
`S44E11` to reserve the historical reference, but that requires explicit
approval.

Sample from the full chosen pool, including zeros, by SHA-256 of UTF-8
`sh08-matrix-v1:20260914:<case_id>`, sorted by `(digest, case_id)`, taking the
first 24. Apply no manual semantic filtering. This ranking is proposed only: do
not generate or freeze IDs yet.
Count the whole chosen sample, including zeros; report model-eligible subset and
conditional metrics separately. Do not make futile requests for zeros. Existing
audio is reusable only on exact identity; otherwise record an unresolved reason
under a rule still to be agreed.

Keep source/window/context/current profile/prompt version/tools/token limits
identical across branches. Generate one first candidate from each generator per
eligible case—two candidates total—and share each saved candidate across the
respective three arbiters with an origin receipt. Allow at most one repair by the
same generator per branch and then at most one recheck. Keep raw receipts, costs,
and unknown states isolated; never mutate historical resumes.

Propose no tools and no implicit transport retries. Fail closed on missing
cost/quota data or an unknown request; a hard spend guard is required **before**
paid work. For the explicit one-case/request scenario, plan at most 20 requests
per eligible case per repetition: `2 initial + 6 judge + 6 repair + 6 recheck`.
Thus the conservative formula is at most 480 for 24 all-eligible cases in one
repetition, or 960 for two. These are planning formulas, not an executable cap,
approval, or current-runner request forecast; batching needs separate accounting.
Independent semantic audit and additional requests are excluded and require
separate authorization.

USD budget, token limits, subscription quota, and retry allowance are unknown.
Do not reuse `$0.5` or call a subscription free. Blinded human review may be an
option after approval, but is not a replacement for the pending independent
model checker or the SH08 pilot audit protocol (24 changed/file, maximum 72).

Decisions still required: model IDs and variants; pool, `S44E11` policy, sample
size, and repetitions; zero policy; independent judge or human-audit identity
and scope; dollar cap, subscription call quota, tokens, and retries; and explicit
run permission. Build the matrix adapter and spend guard only after these
decisions and plan review. Three full pilots require separate next authorization.
Do not add an SH06 control secretly.
