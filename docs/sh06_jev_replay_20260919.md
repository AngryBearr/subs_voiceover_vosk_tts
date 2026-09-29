# SH-06 Jev replay: 19 September 2026

## Result

The authorized bounded continuation completed all 47 remaining requests in the
existing replay directory. The final report is [the local HTML report](../output/jev_replay_20260919/shortening.html);
the offline extraction is [analysis.json](../output/jev_replay_20260919/analysis.json).
These artifacts are local/private and are not a public benchmark.

- Model: `jev-1.13.0`; mode: Choice selection plus per-candidate safety.
- Physical ledger: 48 total, 48 complete, 0 error, 0 unknown, 0 pending.
- Report: `complete`; 24 unique cases and 48 order descriptors.
- Records: 43 accepted and 5 unresolved; no report errors.
- Usage: 220,912 input tokens and 10,850 output tokens.
- Estimated cost: `$0.009278304` at `$0.042` per million input tokens. This is
  an estimate, not a provider-billed receipt; unknown reserves were `$0`.
- Observed receipt latency: median `0.9512775 s`, nearest-rank p95
  `1.122477 s`, over `N=48`; these are observed small-sample values, not an
  SLA or latency benchmark.

## Case-level outcomes

The denominator is the 24 unique cases, not 48 independent cases. Each case was
sent in two recorded orders of the same candidate set.

- Both orders accepted: 20 cases: 13 with the same selected ID and 7 with
  different selected IDs.
- Only one order accepted: 3 cases.
- Neither order accepted: 1 case.
- Raw best-selection ID was stable in 15/24 cases and unstable in 9/24.
- The combined set of safety-accepted candidate IDs was stable in 14/24 and
  unstable in 10/24. The stable set includes 1 case where both orders had an
  empty eligible set, so stability does not imply acceptance.

Candidate counts were 2 for 4 cues, 3 for 8, 4 for 4, and 5 for 8. Thus 16/24
were in the 2--4 range, while 8 actually had five candidates; the report does
not silently describe the run as exclusively 2--4.

These were historical pooled candidates from the saved low/high Muse runs, not
five new candidates from one generation call. The actual pool distribution was
4 cases with 2 candidates, 8 with 3, 4 with 4, and 8 with 5. The prepared
manifest freezes these pools and keeps `candidate_truncation_count` for pools
capped at five. The 24 selected cases were
the eligible cases with at least two safe-length candidates when available,
then a bounded selection; this is selection bias, not a representative sample
of all 60 cases or all subtitles.

There were 4 direct choice/safety conflicts: a candidate selected as `best` had
`safety=reject`. One further order chose `none_acceptable` while all candidate
safety choices were reject. With a descriptive 0.80 threshold applied to the
provider-reported best confidence, selected-choice probability, or selected
safety confidence, 0 of the 4 direct conflicts were high-confidence conflicts.
These values are not correctness scores.

## Screenshot cases

### `16e2689c67a5:cue-353`

Source: `16e2689c67a5`. Original target: “И так я питалась четыре с половиной
месяца.” The seven-cue source context was:

> с моими травмами / и поэтому у меня был такой назогастральный зонд… / Да. /
> И так я питалась четыре с половиной месяца. / Всё хорошо. Я просто каждый
> день радуюсь жизни. / Боже мой. / Потому что я сама не знаю почему. Например,
> мне дали второй шанс

Order 1 selected `c2`, “Питалась четыре с половиной мес.”; best-choice
probability was `0.52`, and the selected safety verdict was `accept`.
Order 2 selected the same text and ID; best-choice probability was `0.47`, and
the selected safety verdict was `accept`.

Historical Sol labels are shown only as descriptive provenance: `c1` low/rep1
generation `revise`, `c2` low/rep1 repair `accept`, `c3` low/rep2 generation
`revise`, `c4` low/rep2 repair `revise`, and `c5` high/rep1 generation `revise`.

### `16e2689c67a5:cue-94`

Source: `16e2689c67a5`. Original target: “но я знаю, что это так, чувствую это
кожей и сердцем.” The seven-cue source context was:

> хотят…хотят меня так сильно сохранить, э-э, / наверное, больше… они,
> наверное, больше смотрят на дружбу. / Я думаю, они не хотят признавать, что я
> становлюсь обузой, / но я знаю, что это так, чувствую это кожей и сердцем. /
> Да. / Хочу поесть йам ча! / Джонатан!

Order 1 selected `c4`, “но знаю, так, чувствую кожей, сердцем.”; best-choice
probability was `0.43`, and the selected safety verdict was `accept`.
Order 2 selected the same text and ID; best-choice probability was `0.54`, and
the selected safety verdict was `accept`.

Historical Sol labels are descriptive provenance only: `c1` low/rep1 generation
`accept`, `c2` low/rep2 generation `revise`, `c3` low/rep2 repair `accept`, `c4`
high/rep1 generation `revise`, and `c5` high/rep1 repair `revise`.

The historical labels are not an independent semantic reference, jury, or
quality score. No matched new Sol calls were made, so these results cannot show
that Jev knows Russian or is better than Sol.

The 43 operationally accepted records are not ground truth. Numeric
preservation was not independently validated, and no independent semantic,
audio-fit, or held-out assessment was run. The two recorded orders also
confound presentation order with stochastic model variation; they support a
descriptive stability result only, not a causal position-bias claim.

## Methods and limits

The replay used the already prepared manifest and its fixed 24-case, two-order
descriptor set. The actual run made 48 Jev judge requests: one authorized
canary and 47 continuation requests. It made no new generator or Sol calls, no
additional independent-judge calls, no fallback calls, and no retries.
Accounting was read from the durable ledger; latency came from actual network
receipt values. The cost formula used recorded input tokens multiplied by the
frozen `$0.042/1M` input rate. The HTML metadata sidecar has schema
`shortening_html_metadata.v1`, 48 rows, and status `complete`.

Recommended next checks, not launched or signed off:

1. **Offline identification:** identify the 4 direct choice/safety conflicts,
   the 3 one-order-only cases, and the 1 neither-order case, then record the
   exact historical Sol disagreements and provenance, including `cue-94`. This
   is record analysis, not independent semantic adjudication.
2. **Separately authorized independent Russian semantic review:** review that
   small set with another model family or, optionally, a user reviewer. This
   requires new inference; it cannot be called offline. Sol may be compared as
   descriptive evidence, but Sol itself is not gold.
3. **Controlled repeat if needed:** repeat the same candidate order and payload
   to separate stochastic model variability from presentation effects. If a
   mapping or consistency problem remains, use isolated one-candidate safety
   calls. These are recommendations only; none was launched or signed off.
