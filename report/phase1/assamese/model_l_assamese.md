# Phase 1 — Model L: Assamese

Assamese (অসমীয়া), `as` / `asm`, Bengali-Assamese script (U+0980–U+09FF).

| | |
|---|---|
| Training tokens | **559,359,252** |
| Manual training tokens | **312,112,673** (55.80%) |
| Tokenizer | SentencePiece Unigram, **8,000** pieces |
| Training documents | 392,097 |
| Training characters | 1,767,282,379 |

Both Phase 1 requirements are met: the corpus exceeds the ~500M token
target, and manual collection supplies 55.80% of training tokens against a
20% floor. No machine-translated or synthetic text is used.

---

## 1. Language selection

Assamese is taken from the assignment's permitted lower-resource list. It
is the majority language of Assam with roughly 15 million speakers, and it
is genuinely lower-resource in the sense the assignment intends: public
digitised text is scarce relative to the major Indian languages, and what
exists is fragmented across archives rather than concentrated in web
crawls.

### The problem that shaped everything downstream

Assamese does not have its own Unicode block. It shares the
Bengali-Assamese script with Bengali, which has roughly fifteen times more
digitised text. Every acceptance decision therefore has to answer a
question Telugu never faces: *is this Assamese, or is it Bengali?*

Metadata cannot answer it. Internet Archive language fields, Wikipedia
mirrors and web crawls all mislabel one as the other routinely. A filter
that only asks "is this the Bengali-Assamese script?" would accept Bengali
text without complaint, and the corpus would be substantially the wrong
language while looking perfectly healthy — perplexity would still converge,
so nothing downstream would reveal the error.

Assamese is distinguished orthographically by two letters absent from
standard Bengali:

```
U+09F0  ৰ  ASSAMESE LETTER RA   (Bengali writes র, U+09B0)
U+09F1  ৱ  ASSAMESE LETTER WA   (Bengali writes ব, U+09AC)
```

ৰ is the ordinary Assamese "ra", so it occurs throughout any genuine
Assamese text at a frequency comparable to "r" in English — it is the
second most frequent piece in the final tokenizer's vocabulary. Bengali
text contains it rarely or never.

Every source is therefore gated on **marker density**: the share of
in-script characters that are ৰ or ৱ, required to be ≥ 0.5%. The threshold
is deliberately conservative — the intent is to exclude text with
essentially no Assamese markers, not to demand a particular style.

**Measured across all 1,617 accepted Internet Archive books:**

| statistic | marker density |
|---|---|
| minimum | 0.919% |
| 5th percentile | 4.099% |
| **median** | **8.751%** |
| maximum | 12.649% |

| band | books | share |
|---|---|---|
| below 0.5% (rejected) | 0 | 0.0% |
| 0.5–1% | 1 | 0.1% |
| 1–3% | 28 | 1.7% |
| 3–6% | 84 | 5.2% |
| 6–10% | 1,340 | 82.9% |
| above 10% | 164 | 10.1% |

The accepted median sits 17× above the threshold and 93% of books exceed
6%, so the filter is separating two clearly distinct populations rather
than making close calls. The 445 documents rejected on language/script
grounds had a median marker density of 1.538% — an order of magnitude
lower.

---

## 2. Corpus sources

| Source | Group | Documents | Characters | Tokens | Share |
|---|---|---|---|---|---|
| Internet Archive | manual | 2,397 | 762,230,185 | **273,694,395** | 48.9% |
| Sangraha `verified/asm` | downloaded | 312,105 | 880,006,376 | 247,246,579 | 44.2% |
| Wikisource | manual | 53,654 | 64,832,583 | 21,084,715 | 3.8% |
| Wikipedia | manual | 23,941 | 60,213,235 | 17,333,563 | 3.1% |
| **Total (train)** | | **392,097** | **1,767,282,379** | **559,359,252** | |

### How each was collected

**Internet Archive** — the dominant source, and the one that decided
whether the project reached its target at all. 2,416 items harvested via
the `internetarchive` API, taking Archive's own `_djvu.txt` OCR output, so
no OCR pass of our own was required. Documents are whole scanned books
averaging ~318,000 characters.

The seam was found by enumerating every text item Archive holds in
Assamese. **This required searching two spellings, not one:**

```
language:(assamese) AND mediatype:(texts)   →   2,576 items
language:(asm)      AND mediatype:(texts)   →  15,385 items
both OR'd together                          →  17,961 items
```

The `asm` seam is where the digitised books live: Digital Library of India
scans (`in.ernet.dli.*`) tag language as the ISO 639-3 code, never the
English name. The first harvest searched only the English name and
recorded in a code comment that `language:"asm"` "returns nothing" — an
untested belief that hid seven eighths of the available items. Correcting
it is the single largest reason this corpus reaches its target.

Collection was cross-checked against the **Directorate of Library
Services, Assam** digitised-book catalogue (`digitized book list.xls`,
3,456 rows, 2,146 Assamese titles, 517,498 pages). Of those titles, 2,104
were located on Archive as DLI items — the catalogue and the seam
correspond almost one to one, which confirms the harvest is close to
exhaustive for this material.

Not all 17,961 items are usable. Sampling by identifier family showed:

| family | items | passed language/script test |
|---|---|---|
| DLI books | 2,181 | 93% |
| "other" | 1,165 | 43% |
| Assam legislature | 1,861 | 7% |
| Assam gazettes | 12,744 | **0%** |

The gazettes are 71% of the seam and contribute nothing: they are
bilingual government notices, English-dominant, and fail the 30%
script-ratio gate. Collection was therefore ordered books-first and the
gazettes dropped, which reduced the work list from 17,296 items to 4,067
without losing usable text.

**Sangraha** (`ai4bharat/sangraha`, subset `verified/asm`) — the only
downloaded source. All 3 available shards were processed; unlike Telugu,
where shards had to be rationed against the manual floor, Assamese has no
surplus to ration.

**Wikisource** (`aswikisource-20260801`) — 53,654 documents. The
`Page:` namespace (ns 104) **is** the corpus for this language: Assamese
Wikisource holds 82,410 page transcriptions against 2,999 namespace 0
pages, of which only 368 survived cleaning. The community transcribes but
rarely assembles. The Telugu pipeline excludes ns 104 because its
transcriptions duplicate assembled works; carrying that rule to Assamese
unexamined discarded 85,079,684 characters and left Wikisource
contributing 0.4M tokens instead of 21.1M. It is now a per-language flag.

**Wikipedia** (`aswiki-20260801`) — 23,941 articles after markup cleaning
with `mwparserfromhell`, removal of reference apparatus, and exclusion of
list/index/disambiguation pages.

---

## 3. Preprocessing

All sources pass through one pipeline
(`common/build_corpus_pipeline.py --language assamese`), so both models are
built by identical code and the Phase 3 comparison can attribute
differences to data rather than to implementation drift.

| Source | Read | Quality removed | Exact dupes | Near dupes | Kept |
|---|---|---|---|---|---|
| wikisource | 54,281 | 0 | 54 | 30 | 54,197 |
| wikipedia | 24,182 | 0 | 4 | 7 | 24,171 |
| archive | 2,547 | 100 | 30 | 1 | 2,416 |
| sangraha | 315,379 | 0 | 4 | 116 | 315,259 |

### 3.1 Collection-time filtering

Every document must have ≥100 characters, ≥50 in-script characters, ≥10%
of alphabetic characters in the Bengali-Assamese block, and ≥0.5% marker
density. Internet Archive additionally requires ≥1,000 characters and ≥30%
script ratio at download time.

### 3.2 Quality filter (OCR sources only)

Archive text is raw OCR and needs checks the wiki sources do not. Four
tests apply, in increasing order of cost:

| test | threshold | rejected |
|---|---|---|
| too short | < 1,000 chars | — |
| low script ratio | < 30% | — |
| repeated lines | > 50% duplicate lines | 1 |
| digit/symbol junk | > 65% of dense chars | 9 |
| **word purity** | **< 50%** | **90** |

**Word purity** is the share of whitespace-separated tokens composed
entirely of in-script characters after stripping punctuation. It was added
after measuring OCR quality across 1,238 accepted books:

| band | books | share |
|---|---|---|
| clean ≥90% | 274 | 22.1% |
| good 80–90% | 612 | 49.4% |
| moderate 65–80% | 204 | 16.5% |
| poor 50–65% | 55 | 4.4% |
| degraded <50% | 93 | 7.5% |

Median purity is 84.0%, median Latin intrusion 0.0% — in half the books
the OCR never hallucinates a Latin letter. But the bad tail is real: the
worst books bottom out at 6.8% purity and 69.7% Latin intrusion, and their
fragment rate (share of single-character tokens) reaches 30.7% against a
2.2% median.

Word fragments are exactly what inflates tokenizer fertility, so the gate
is set at the bottom of the distribution rather than the middle. Below 50%
purity a document has lost more than half its words and contributes
subword pieces that correspond to no morpheme. Between 50% and 80% the
text is degraded but still mostly words, and discarding it would cost real
tokens. The existing `OCR_GARBAGE` test caught 9 documents; word purity
caught 90.

**~12% of accepted books remain materially degraded.** This is reported
rather than engineered away: it is a realistic property of low-resource
corpora built from historical scans, and Section 4.5 shows its cost.

### 3.3 Exact deduplication

SHA-256 over NFC-normalised text, applied globally across all sources. 88
documents removed.

### 3.4 Near deduplication

MinHash LSH, 64 permutations, 5-word shingles, Jaccard threshold 0.90,
minimum 20 words. 154 documents removed.

Sources are processed **manual first, downloaded last**, so a document
appearing in both Internet Archive and Sangraha is credited to the manual
side. Sangraha lost nothing to archive (its kept count is identical
whether archive runs before it or not), which independently confirms the
two sources are disjoint — the same negative finding the Telugu report
reports.

### 3.5 Source mixing

Every source is kept whole (`keep_fraction = 1.0`). Assamese does not
over-deliver against any plausible budget, so nothing is trimmed. The
mixing policy exists to prevent a manual source being crowded out if
`--target-manual` is raised later.

### 3.6 Splits

Document-level, assigned by hashing the document id with seed 42, so
regeneration reproduces the identical split. Validation and test take 0.5%
each.

| Split | Documents | Characters | Tokens | Manual share |
|---|---|---|---|---|
| train | 392,097 | 1,767,282,379 | 559,359,252 | 55.80% |
| validation | 1,970 | 7,517,542 | 2,383,367 | 46.38% |
| test | 1,976 | 9,553,317 | 2,879,068 | 51.59% |

**A known limitation.** Internet Archive documents are whole books, so the
source contributes only 11 documents to validation and 8 to test despite
being 43.1% of the corpus by characters. Those 11 books carry 2,386,055
characters — ample text for aggregate fertility — but per-source archive
figures carry book-level variance, since OCR quality differs between
volumes.

A larger held-out fraction for archive was tried and rejected. At 0.5%,
archive is 43.1% of validation characters, exactly matching its 43.1%
share of the corpus: the split is already representative. Raising it to 2%
pushed archive to 73.7% of validation, which would have measured Phase 2
perplexity on a held-out set dominated by the noisiest part of the corpus
and broken comparability with Telugu, whose validation mirrors its own
corpus. Splitting books into pages would give thousands of documents and
is worse still: pages of one book would land in both train and test,
turning sampling noise into genuine leakage.

---

## 4. Tokenizer

### 4.1 Method

SentencePiece **Unigram**, trained from scratch on the training split
only — never on validation or test, so held-out fertility and Phase 2
perplexity are not optimistically biased.

| setting | value | reason |
|---|---|---|
| model type | unigram | probabilistic segmentation, better for morphologically rich scripts than greedy BPE |
| vocab size | 8,000 | see 4.2 |
| character coverage | 0.9995 | leaves a small tail to byte-fallback |
| byte_fallback | true | guarantees zero unknown tokens |
| max_sentence_length | 16,384 | see below |
| input_sentence_size | 3,000,000 | memory bound, sampled with `shuffle_input_sentence` |

`max_sentence_length` is measured in **bytes** and defaults to 4,192.
Bengali-Assamese costs three bytes per character, so the exporter's
2,000-character chunks reach ~6,000 bytes and would be dropped **silently**
— no warning, no error. Measured on Telugu, the default discarded 0.72% of
lines but 7.15% of the text, weighted towards the longest continuous
prose, which is exactly what subword learning depends on.

### 4.2 Vocabulary size selection

Candidates at 6k / 8k / 10k / 12k were trained on a 10% sample of the
training split and compared on held-out validation text.

| vocab | chars/tok | fertility | unk rate | byte fb | vocab used | embedding |
|---|---|---|---|---|---|---|
| 6,000 | 2.494 | 2.103 | 0.0000% | 0.3233% | 97.02% | 2,304,000 |
| **8,000** | **2.612** | **2.009** | 0.0000% | 0.3386% | 97.72% | **3,072,000** |
| 10,000 | 2.698 | 1.944 | 0.0000% | 0.3497% | 98.15% | 3,840,000 |
| 12,000 | 2.765 | 1.897 | 0.0000% | 0.3584% | 98.39% | 4,608,000 |

**Unknown-token rate is 0.000% at every size**, so it cannot discriminate
here — byte-fallback covers the tail by construction. The decision rests
on fertility and compression weighed against parameter cost.

**The vocabulary is bought out of depth.** Phase 2 fixes the model at ~25M
parameters, and the embedding matrix is inside that budget, not outside
it. At `d_model=384` with a tied output head, the embedding costs
`vocab × 384` and each Transformer block costs `12 × 384² = 1,769,472`:

| vocab | embedding | layers affordable within ~25M | embedding % |
|---|---|---|---|
| 6,000 | 2,304,000 | 12 | 10% |
| **8,000** | 3,072,000 | **12** | 13% |
| 10,000 | 3,840,000 | 11 | 16% |
| 12,000 | 4,608,000 | 11 | 19% |

**8,000 is chosen because the trade is a step function, not a smooth
curve.** 6k and 8k both afford twelve layers; 10k and 12k afford eleven.
Within twelve layers, 8k strictly dominates 6k — 4.7% better compression,
higher utilisation, identical depth — so 8k is the best tokenizer
available at no cost in layers. Going on to 10k buys 3.3% fertility for a
whole Transformer block, and 12k buys 2.5% more for 19% of the model spent
on a lookup table that performs no computation.

Vocabulary utilisation *rises* across the whole sweep (97.02% → 98.39%),
which normally indicates the corpus could support more pieces. It is not
taken as an argument for going larger: the binding constraint is the
parameter budget, not corpus capacity.

**10,000 is retained as a documented alternative** should Phase 2 find the
twelfth layer worth less than the compression.

### 4.3 Final tokenizer

Retrained at 8,000 on the **full** training split (1.77B characters,
4.55 GB export), 18.5 minutes.

| | sweep candidate (10% sample) | **final (full split)** |
|---|---|---|
| chars/token | 2.612 | **2.621** |
| fertility | 2.009 | **2.001** |
| vocabulary used | 97.72% | **97.91%** |
| byte-fallback rate | 0.3386% | 0.3936% |
| unknown rate | 0.0000% | **0.0000%** |

The full-data model is marginally better on every measure that matters,
and the sweep's ranking held — which is the assumption the
sweep-then-retrain design rests on.

### 4.4 Token-frequency statistics

| | |
|---|---|
| vocabulary size | 8,000 |
| distinct pieces used on held-out text | 7,833 (97.91%) |
| singleton pieces | 6 |
| top-100 coverage | 38.73% |
| top-500 coverage | 59.37% |
| top-1,000 coverage | 69.65% |
| top-5,000 coverage | 95.03% |

Six singleton pieces out of 8,000 means almost nothing in the vocabulary
is wasted. The most frequent pieces are the word-boundary marker `▁`,
then **ৰ** — the Assamese marker character — then the danda `।`, which is
exactly the distribution genuine Assamese prose should produce.

### 4.5 Fertility by source

Measured on held-out validation text with the final tokenizer:

| source | chars/token | fertility (tokens/word) | byte-fallback |
|---|---|---|---|
| **archive** | **2.067** | **2.408** | 0.0502% |
| wikisource | 2.615 | 1.915 | 0.6141% |
| wikipedia | 2.973 | 1.924 | 0.7028% |
| sangraha | 3.003 | 1.791 | 0.6082% |

**Internet Archive costs 34% more tokens per word than Sangraha.** This is
the OCR fertility penalty, and it is the price of the manual collection
that makes the corpus reach target. Two causes: OCR noise breaks words
into fragments the tokenizer cannot merge, and the books are older
literary and historical registers whose vocabulary is less represented in
the subword inventory than contemporary web prose.

The Telugu report measures the same effect on its own Archive source
(4.018 chars/token against Sangraha's 1.755 tokens per word, a 49%
penalty), so this is a property of Archive OCR rather than of Assamese.
It matters more here because archive is 48.9% of Model L's training
tokens against a much smaller share for Model H — a difference Phase 3
should expect to see in the comparison.

### 4.6 Tokenization examples

Examples are drawn from held-out text rather than written by hand: the
author does not read Assamese, and invented sentences could carry errors
the fertility table would then attribute to the tokenizer. See
`tokenizer_evaluation.json` for the per-source samples.

---

## 5. Requirements check

| # | Requirement | Result |
|---|---|---|
| 1 | ~500M training tokens | **559,359,252** ✅ |
| 2 | ≥20% manual tokens | **55.80%** ✅ |
| 3 | Separate corpus, no sharing with Model H | ✅ |
| 4 | Train/validation/test splits | 392,097 / 1,970 / 1,976 ✅ |
| 5 | Tokenizer trained from scratch | SentencePiece Unigram ✅ |
| 6 | Vocabulary chosen on held-out fertility | 6k/8k/10k/12k sweep ✅ |
| 7 | Unknown-token statistics | 0.0000% (byte-fallback) ✅ |
| 8 | Unicode normalisation | NFC throughout ✅ |
| 9 | Deduplication | exact + near (Jaccard 0.90) ✅ |
| 10 | No MT or synthetic text | none used ✅ |

---

## 6. Figures

| Figure | File |
|---|---|
| Source composition | `figures/corpus_source_composition.png` |
| Document length distribution | `figures/corpus_document_lengths.png` |
| Split sizes | `figures/corpus_split_sizes.png` |
| Vocabulary sweep | `figures/tokenizer_vocab_sweep.png` |
| Fertility by source | `figures/tokenizer_fertility_by_source.png` |

---

## 7. Notes on reproducibility

Everything is deterministic given seed 42. Split assignment is by content
hash, so regeneration reproduces the identical corpus — verified during
this build, when wikisource, wikipedia and sangraha each reproduced their
previous character counts exactly across three separate pipeline runs
while only the archive source changed.

### Defects found and corrected

Three silent defects in the original collection code are worth recording,
because each one failed without raising an error and the corpus looked
healthy throughout:

1. **The Internet Archive query searched one spelling of the language.**
   `language:"Assamese"` returns 2,576 items; `language:(asm)` returns
   15,385. A code comment asserted the latter "returns nothing," and the
   claim was never retested. Cost: seven eighths of the available items.

2. **The harvest stopped at 989 identifiers** with no error recorded — 38%
   of even the seam it could see. Nothing in the manifest indicated an
   incomplete run.

3. **`download_text()` deleted files it had just downloaded.** When
   Archive's filename matched `safe_name(identifier) + "_djvu.txt"` — the
   normal case for DLI books — the code treated the fresh download as a
   stale copy, unlinked it, and returned the path. The caller reported
   `READ_FAILED`, marked the identifier tried, and never retried. It
   destroyed the most valuable items specifically, because their filenames
   follow the identifier.

A fourth issue affected the sweep rather than the corpus:
`run_vocab_sweep.py` skips candidates whose model file already exists,
which is correct for resuming an interrupted sweep and wrong after the
corpus changes. A rerun silently retained an 8k model trained on the
earlier 289M-token corpus and compared it against candidates trained on
the current one. A `--force` flag was added, and the skip message now
prints the model's training date.

Together these took the corpus from 289,375,170 tokens (23.08% manual) to
559,359,252 tokens (55.80% manual), entirely from real Assamese text.
