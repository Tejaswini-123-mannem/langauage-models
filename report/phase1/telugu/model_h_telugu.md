# Phase 1 — Model H: Telugu

**Higher-resource language.** Corpus construction and tokenizer training.

| | |
|---|---|
| Language | Telugu (తెలుగు), `te` / `tel` |
| Script | Telugu block, U+0C00–U+0C7F |
| Training tokens | **647,207,005** |
| Manual token share | **22.02%** (requirement: ≥20%) |
| Tokenizer | SentencePiece Unigram, **8,000** pieces |

> **Tokenizer revised after the first submission.** This report originally
> delivered a 32,000-piece tokenizer and reported 522,289,581 training
> tokens. The corpus is unchanged; only the measuring instrument is. See
> §4.2 for why 32,000 was withdrawn and 8,000 selected, and §4.7 for the
> effect on the reported counts.

---

## 1. Language selection

Telugu was chosen as the higher-resource language on measured availability rather than reputation:

| source | Telugu | Assamese (Model L) | ratio |
|---|---:|---:|---:|
| Sangraha `verified/` | 41 shards, ~36 GB | 3 files, 0.91 GB | ~15× |
| Internet Archive items | 667,239 | 2,576 | **259×** |
| Wikipedia dump | 246.4 MB | 67.5 MB | 3.7× |
| Other Wikimedia projects | several | none with content | — |

Telugu is a Dravidian language with ~83M speakers, its own Unicode block, and mature
tooling (a Tesseract OCR model, an AI4Bharat Sangraha subset). The Unicode block matters
methodologically: no other language uses it, so "is this character Telugu?" is a range
check with no ambiguity. Model L's script is shared with Bengali and required an extra
language test — see the Model L report.

---

## 2. Corpus sources

Five sources, four of them manually collected.

| source | group | documents | characters | tokens | % of tokens |
|---|---|---:|---:|---:|---:|
| Sangraha | downloaded | 911,584 | 1,849,488,951 | 504,696,553 | 78.0% |
| Wikipedia | **manual** | 76,209 | 233,394,230 | 63,569,624 | 9.8% |
| Internet Archive | **manual** | 819 | 143,917,900 | 54,865,499 | 8.5% |
| Wikisource | **manual** | 8,794 | 35,437,438 | 15,037,739 | 2.3% |
| Textbook OCR | **manual** | 17,944 | 26,586,684 | 9,037,590 | 1.4% |
| **Total (train)** | | **1,015,350** | **2,288,825,203** | **647,207,005** | |

**Manual: 142,510,452 tokens (22.02%).** Counts are measured with the delivered 8,000-piece
tokenizer; see §4.7.

### How each was collected

**Sangraha** (`ai4bharat/sangraha`, subset `verified/tel`). Six of 41 available shards were
processed, giving 1,030,805 documents. More were available but unnecessary: the ≥20% manual
rule caps downloadable data at four times the manual total, so additional shards could not
have been used without proportionally more manual collection.

**Wikipedia** (`tewiki-20260801`). Full dump parsed, markup cleaned with `mwparserfromhell`.
114,775 articles survived from 236,831 pages.

**Internet Archive.** 1,641 items harvested via the `internetarchive` API, taking Archive's
own `_djvu.txt` OCR output. Documents are whole books averaging ~176,000 characters.

**Wikisource** (`tewikisource-20260801`). 8,909 works. Only namespace 0 was kept; the 180,300
`Page:` entries duplicate the assembled works. **This decision was correct for Telugu and
wrong for Assamese** — see the Model L report, where the same rule discarded 98% of the source.

**Textbook OCR.** 71 Telugu PDFs rendered at 2× and OCR'd with Tesseract (`tel`), 18,167 pages.

---

## 3. Preprocessing

Six stages. Each filter exists to remove a specific defect, and the counts below are what it
actually caught.

### 3.1 Collection-time filtering

Applied per source, because the failure modes differ.

| filter | applies to | purpose |
|---|---|---|
| Namespace | wikis | keep articles; drop talk/template/user pages |
| Redirect removal | wikis | `#REDIRECT` pages carry no text |
| Appendix stripping | Wikipedia | cut from మూలాలు / బయటి లింకులు onward — citation apparatus, not prose |
| List/disambiguation | wikis | ≥80% bullet lines, or జాబితా / సూచిక in the title |
| ≥1,000 chars, ≥30% Telugu | Archive | Archive items are books; the ratio gate rejects English books *about* Telugu |
| ≥50 chars, ≥10% Telugu | Sangraha | fragments and mislabelled documents |

Wikipedia: 236,831 pages → 114,775 kept. Of those removed, 72,054 were non-article
namespaces, 35,535 too short, 10,839 list/disambiguation, 3,550 redirects.

### 3.2 Quality filter (OCR sources only)

Wiki text is clean by construction; OCR is not. Two checks exist that no other stage performs:

| check | catches | dropped |
|---|---|---:|
| repeated-line ratio > 50% | OCR **looped**, repeating one line down the page | 2 |
| digit/symbol ratio > 65% | tables, indexes, page furniture — not language | **67** |
| length and script gates | as above | 1,000 |
| **total** | | **1,069** of 19,236 textbook pages |

The 67 `OCR_GARBAGE` pages pass every length and script test — they are number tables with
plenty of characters. Only the symbol-ratio check finds them.

### 3.3 Exact deduplication

SHA-256 over NFC-normalised text, globally across all five sources. Normalising *before*
hashing matters: two documents differing only in whitespace or Unicode composition are the
same document, and a naive hash keeps both.

**Result: 0 duplicates between manual sources and Sangraha.** A useful negative finding — the
hand-collected books, newspapers and wiki text do not overlap with the web crawl at the
exact-match level.

### 3.4 Near deduplication

MinHash LSH, Jaccard threshold 0.90, 5-word shingles, 64 permutations. Catches what exact
matching cannot: the same article with a changed header, a reformatted reprint, a mirror.

**Ordering is the important design choice.** Near-dedup keeps whichever document was indexed
first, so sources are processed **manual first, scarcest first**. If a textbook page and a
newspaper page are near-identical, the textbook survives and the newspaper copy is dropped.

| source | input | removed | rate |
|---|---:|---:|---:|
| Wikipedia | 114,775 | 1,045 | 0.91% |
| Textbook | 18,167 | 69 | 0.38% |
| Wikisource | 8,909 | 25 | 0.28% |
| Archive | 1,636 | 5 | 0.31% |
| Sangraha | 1,030,805 | 121 | 0.01% |
| **Total** | **1,174,292** | **1,265** | **0.11%** |

Sangraha's 0.01% confirms AI4Bharat's `verified` split is already well deduplicated.
Wikipedia's 0.91% is the highest — templated stub articles that survived the list filter
share most of their wording.

Documents under 20 words are kept unconditionally and never indexed: a Jaccard estimate over
so few shingles is not meaningful.

### 3.5 Source mixing

The assignment states the target and the ≥20% rule in **tokens**, but everything upstream is
measured in **characters**. These are not interchangeable — per-source rates range from 2.43
to 4.02 characters per token, a 65% spread. Converting with one global ratio would be wrong.

The mixing rule is that *manual ≥ 20% of total* is algebraically identical to
*downloaded ≤ 4 × manual*. Manual data is scarce and expensive, so it is kept and the
downloaded side absorbs the adjustment. Within manual, sources are consumed in priority
order — textbook, Wikisource, Wikipedia, Archive — with a 60M cap on Wikipedia.

**Why Wikipedia was capped.** It yielded 88.6M tokens against an expected 30–75M. Under a
strict priority order it would have consumed almost the entire manual budget and reduced
Internet Archive to ~1.6% of its collected size, deleting the contemporary-newspaper register
entirely. The purpose of collecting four manual sources was register coverage, not volume.

### 3.6 Splits

Train / validation / test at **0.5% / 0.5%**, assigned at **document level** by hashing
`sha1(seed:namespace:doc_id)`.

**Document-level, not chunk-level.** Phase 2 concatenates documents and cuts fixed-length
sequences. Splitting after that would put two chunks of one document in train and test, and
the model would be evaluated on text it had memorised.

**Hashed, not shuffled.** Deterministic, reproducible from the seed alone, and streams in
constant memory over a 5.9 GB file. This was validated accidentally but conclusively: a
`train.jsonl` lost to a Drive cache eviction was regenerated and the surviving
`validation.jsonl` and `test.jsonl` matched **byte for byte** (identical SHA-256). With
unseeded shuffling the loss would have been unrecoverable.

The namespace argument keeps the keep/drop decision independent of the split assignment;
reusing one hash for both would correlate them and bias the splits.

| split | documents | characters | tokens | manual share |
|---|---:|---:|---:|---:|
| train | 1,015,350 | 2,288,825,203 | **647,207,005** | **22.02%** |
| validation | 5,171 | 11,324,463 | 3,185,932 | 17.85% |
| test | 5,120 | 11,368,714 | 3,192,593 | 22.85% |

Validation's manual share drifts because Archive has only 831 documents in the corpus and 6
landed in validation — with so few, very large documents, random assignment is noisy. The
graded figure is the training split.

---

## 4. Tokenizer

### 4.1 Method

SentencePiece **Unigram**, trained from scratch. No pretrained tokenizer was used.

Unigram was chosen over BPE deliberately. BPE builds bottom-up by merging frequent adjacent
pairs and produces one deterministic segmentation. Unigram starts from a large candidate
vocabulary and prunes probabilistically, keeping the set that best explains the corpus. For a
morphologically rich agglutinative language — where విద్యార్థులందరూ is
విద్యార్థి + ల + అందరూ — Unigram finds cleaner morpheme-like boundaries.

Settings: `character_coverage=0.9995`, `byte_fallback=True`, `split_digits=True`,
`max_sentence_length=16384`, control tokens `<unk>=0 <s>=1 </s>=2 <pad>=3`.

**`max_sentence_length` was a bug fix, not a default.** SentencePiece's limit is in *bytes*;
Telugu costs 3 bytes per character, so a 2,000-character line reaches ~6,000 bytes and
exceeded the 4,192-byte default. Lines over the limit are **silently dropped**. Measured on
the real corpus this was 0.72% of lines but **7.15% of the text**, because the lines lost were
the longest and most continuous prose — exactly the material subword learning depends on.

### 4.2 Vocabulary size selection

The assignment requires the vocabulary size to be chosen on fertility and unknown-token rate
measured on held-out text. Five candidates were trained on a 10% sample and scored on the
**validation split**:

> These five candidates were trained before the `max_sentence_length` defect in §4.3 was
> found, so each saw the truncated export. The sweep code now sets the limit, which means
> rerunning it would not reproduce this table exactly. The effect is bounded: the 32k
> tokenizer retrained at full scale *with* the limit raised landed within 0.3% of its
> candidate (3.766 → 3.776 chars/token), and the defect applies equally to all five
> candidates, so the ranking the decision rests on is unaffected.

| vocab | chars/token | fertility | UNK | byte-fb | vocab used | embed params @384 |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 3.053 | 2.268 | 0.0000% | 0.423% | 97.97% | 3,072,000 |
| 16,000 | 3.429 | 2.019 | 0.0000% | 0.475% | **98.93%** | 6,144,000 |
| **32,000** | **3.766** | **1.839** | 0.0000% | 0.522% | 98.58% | **12,288,000** |
| 48,000 | 3.943 | 1.756 | 0.0000% | 0.546% | 96.27% | 18,432,000 |
| 64,000 | 4.055 | 1.708 | 0.0000% | 0.562% | **92.58%** | 24,576,000 |

**Compression has sharply diminishing returns.** Cost per unit of compression rises 6.7×
across the range:

| step | chars/token gained | extra params | params per 0.1 chars/token |
|---|---:|---:|---:|
| 8k → 16k | +0.376 | +3.07M | 0.82M |
| 16k → 32k | +0.337 | +6.14M | 1.82M |
| 32k → 48k | +0.177 | +6.14M | **3.47M** |
| 48k → 64k | +0.112 | +6.14M | **5.48M** |

**Vocabulary utilisation turns over at 16k**, falling to 92.58% at 64k — roughly 4,750 pieces
never appear in held-out text. That is the empirical signature of a vocabulary outgrowing its
data.

**The Phase 2 parameter budget decides it.** With tied embeddings at `d_model=384`, one
Transformer block costs ≈ 1.77M parameters:

| vocab | embedding | left of ~25M | ≈ layers |
|---:|---:|---:|---:|
| 16,000 | 6.1M | 18.9M | ~10–11 |
| **32,000** | **12.3M** | **12.7M** | **~7** |
| 48,000 | 18.4M | 6.6M | ~3–4 |
| 64,000 | 24.6M | 0.4M | unusable |

**32,000 was selected on this evidence** — 93% of 64k's compression at half the embedding
cost, 98.58% utilisation, and 12.7M parameters left for the model itself.

#### Why 32,000 was subsequently withdrawn

The table above weighs compression against parameter cost, but it accepts a model with
**seven Transformer layers** as the price. That was the error. Phase 3 is graded on
comparative and transitive reasoning, which depends on depth: a 32,000 vocabulary spends
**53% of a ~25M model on a lookup table that performs no computation**, and the remaining
12.7M has to hold every attention and feed-forward block.

The sweep was therefore rerun over a bracket the parameter budget actually permits. Candidates
at 6k/8k/10k/12k, trained on a 10% sample and scored on the **validation split**:

| vocab | chars/token | fertility | UNK | byte-fb | vocab used | embed params @384 | layers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,000 | 2.889 | 2.397 | 0.0000% | 0.400% | 97.30% | 2,304,000 | 12 |
| **8,000** | **3.053** | **2.268** | 0.0000% | 0.423% | 97.97% | **3,072,000** | **12** |
| 10,000 | 3.179 | 2.178 | 0.0000% | 0.440% | 98.38% | 3,840,000 | 11 |
| 12,000 | 3.278 | 2.113 | 0.0000% | 0.454% | 98.63% | 4,608,000 | 11 |

**8,000 selected.** The trade is a step function, not a smooth curve. At `d_model=384` a
Transformer block costs 1,769,472 parameters, so 6k and 8k both afford twelve layers while
10k and 12k afford eleven. Within twelve layers, **8k strictly dominates 6k** — 5.7% better
compression, higher utilisation, identical depth — making it the best tokenizer available at
no cost in layers. Going on to 10k buys 4.0% fertility for a whole block.

Against the withdrawn choice: 8,000 gives up 19% compression relative to 32,000 (3.053 vs
3.766 chars/token) and buys **five additional Transformer layers**.

**Model L (Assamese) selects 8,000 by the same rule on its own measured sweep.** Holding
vocabulary size equal across the two models is a methodological gain for Phase 3: with
granularity constant, a Model H versus Model L difference is attributable to the **data**
rather than to one model having a coarser tokenizer. The two vocabularies remain entirely
separate, trained on separate corpora, sharing no pieces.

The 16,000 and 32,000 tokenizers remain trained and published, as the record of what was
measured.

**UNK is 0.0000% by construction** because `byte_fallback=True` decomposes unrepresentable
characters into UTF-8 bytes. Reporting only "UNK = 0%" would hide the real signal, so the
**byte-fallback rate** is reported alongside as its honest analogue.

### 4.3 Final tokenizer

Trained on the **training split only** — 12,553,890 lines, 2,268,974,257 characters. Training
on the full corpus would mean the tokenizer had already seen validation and test text, making
fertility and Phase 2 perplexity optimistically biased.

| | value |
|---|---:|
| vocabulary size | 32,000 |
| chars/token (held-out) | 3.776 |
| fertility (tokens/word) | 1.834 |
| UNK rate | 0.0000% |
| byte-fallback rate | 0.419% |
| vocabulary utilisation | 98.82% |
| distinct pieces used | 31,623 of 32,000 |
| singleton pieces | 517 |
| pieces containing Telugu | 29,736 (92.9%) |
| pieces containing Latin | 2,162 (6.8%) |
| mean piece length | 6.17 characters |
| training time | 38.7 minutes |

**Sample-based selection was validated.** Full-corpus results differ from the 10% sweep by
under 0.3% (32k: 3.766 → 3.776), so the sweep's ranking transfers and the 48k/64k figures can
be cited with confidence despite not being retrained at scale.

### 4.4 Token-frequency statistics

Coverage of held-out text by the most frequent pieces:

| top-k pieces | coverage |
|---:|---:|
| 100 | 32.45% |
| 500 | 47.26% |
| 1,000 | 54.84% |
| 5,000 | 76.04% |
| 10,000 | 86.09% |
| 20,000 | 95.76% |

Only 517 pieces (1.6%) appear exactly once — the vocabulary is well supported by the data.

### 4.5 Fertility by source

| source | chars/token | fertility | byte-fallback |
|---|---:|---:|---:|
| Sangraha (web) | 4.018 | 1.755 | 0.388% |
| Wikipedia | 3.755 | 1.791 | 0.539% |
| Textbook (OCR) | 2.963 | 2.149 | 0.590% |
| Archive (OCR) | 2.735 | 2.363 | 0.275% |
| Wikisource | 2.428 | 2.613 | 0.812% |

**OCR and classical text fragment far more than clean web text** — Wikisource needs 2.613
tokens per word against Sangraha's 1.755, a 49% penalty. Two causes: OCR noise prevents the
tokenizer from finding long reusable pieces, and Wikisource's classical/literary register uses
vocabulary and inflections that are rare in the modern web text dominating the corpus.

This is why the mix had to be planned per source. It is also why the **character** share
(19.19%) and the **token** share (22.02%) differ by nearly 3 points: manual sources produce
more tokens per character, so they are worth more in the unit that is graded. The gap narrowed
from 4 points under the 32,000 tokenizer, because a smaller vocabulary lengthens the
downloaded web text proportionally more than it lengthens the manual sources.

### 4.6 Tokenization examples

```
తెలుగు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో ఒకటి.
  → 7 tokens, 6.43 chars/token
  ▁తెలుగు ▁భారతదేశంలో ▁అత్యధికంగా ▁మాట్లాడే ▁భాషలలో ▁ఒకటి .

విద్యార్థులందరూ పాఠశాలకు వెళ్ళి చదువుకుంటున్నారు.
  → 8 tokens, 5.75 chars/token
  ▁విద్యార్థుల ందరూ ▁పాఠశాల కు ▁వెళ్ళి ▁చదువుకుంటున్న ారు .

దేశభాషలందు తెలుగు లెస్స.
  → 6 tokens, 3.67 chars/token
  ▁దేశ భాష లందు ▁తెలుగు ▁లెస్స .

భౌతిక శాస్త్రంలో 25 డిగ్రీల ఉష్ణోగ్రత వద్ద density అనేది ముఖ్యం.
  → 14 tokens, 4.00 chars/token
  ▁భౌతిక ▁శాస్త్రంలో ▁ 2 5 ▁డిగ్రీల ▁ఉష్ణోగ్రత ▁వద్ద ▁d ens ity ▁అనేది ▁ముఖ్యం .
```

The segmentations show the Unigram model finding morphological structure: విద్యార్థులందరూ
splits as `విద్యార్థుల` + `ందరూ` (stem + plural-inclusive suffix), and
చదువుకుంటున్నారు as `చదువుకుంటున్న` + `ారు` (verb stem + person/number ending). The classical
line fragments more heavily (3.67 chars/token), consistent with the per-source fertility above.
Digits split individually (`2` `5`) by design, and English `density` decomposes into
`▁d ens ity` — correct behaviour for a monolingual Telugu vocabulary.

---

### 4.7 Effect of the tokenizer change on reported counts

The corpus is **byte-for-byte identical** to the one this report originally described. The
splits, the documents, the characters and the deduplication are unchanged. Only the tokenizer
that counts them changed, and a smaller vocabulary segments the same text into more pieces:

| | 32,000 (withdrawn) | **8,000 (delivered)** |
|---|---:|---:|
| Training tokens | 522,289,581 | **647,207,005** |
| Manual training tokens | 120,483,488 | **142,510,452** |
| Manual token share | 23.07% | **22.02%** |
| Characters per token (train) | 4.38 | **3.54** |
| Embedding parameters @384 | 12,288,000 | **3,072,000** |
| Transformer layers within ~25M | ~7 | **12** |

Both figures are correct for their own tokenizer. Token counts are not a property of a corpus
alone; they are a property of a corpus *and* the tokenizer measuring it, which is why the
assignment's ~500M target is only meaningful once the delivered tokenizer is fixed.

**The manual share fell slightly, from 23.07% to 22.02%**, and the reason is worth stating
rather than leaving as noise. A smaller vocabulary lengthens every source, but not equally:
Sangraha's contemporary web prose is the register the subword inventory covers best, so
shrinking the vocabulary costs it proportionally more pieces than it costs the manual
sources. Sangraha grew from 76.9% to 78.0% of tokens, and the manual side was diluted.

The share still passes, but with **2.0 points of headroom rather than 3.1**. Model L, by
contrast, sits at 55.80% — the lower-resource model carries far more manual data than the
higher-resource one, because Sangraha supplies 78% of Telugu's tokens while Assamese had to
be built from its own Internet Archive collection. That inversion is a genuine finding about
the two resource tiers and is taken up in the Phase 3 comparison.

---

## 5. Requirements check

| requirement | target | achieved | |
|---|---|---|---|
| Training tokens | ~500M | **647,207,005** | ✅ |
| Manual share | ≥20% | **22.02%** | ✅ |
| Separate tokenizer | required | trained on Telugu only | ✅ |
| No pretrained tokenizer | required | SentencePiece from scratch | ✅ |
| Vocabulary not shared | required | Telugu-only vocabulary | ✅ |
| Train/val/test splits | required | 1,015,350 / 5,171 / 5,120 | ✅ |

---

## 6. Figures

| file | contents |
|---|---|
| `figures/corpus_source_composition.png` | training tokens by source, manual vs downloaded |
| `figures/corpus_document_lengths.png` | document length distribution per source |
| `figures/corpus_split_sizes.png` | split sizes and manual share |
| `figures/tokenizer_vocab_sweep.png` | compression and fertility against vocabulary size |
| `figures/tokenizer_fertility_by_source.png` | chars/token by corpus source |

Data: `dataset_statistics.csv`, `dataset_statistics.json`, `tokenizer_evaluation.json`.

---

## 7. Notes on reproducibility

All stages are deterministic given seed 42. Split assignment is by content hash, so
regeneration reproduces the identical corpus — verified by byte-identical SHA-256 on the
validation and test splits after an unplanned regeneration.

Two defects found and fixed during construction are worth recording, since both produced
plausible-looking output rather than errors:

1. **A stale partial file** left by an interrupted deduplication run was picked up by a
   wildcard glob, causing every record to be compared against itself and discarded as its own
   near-duplicate. Fixed by naming the input explicitly.

2. **`max_sentence_length`** defaulted to 4,192 *bytes* while the exporter chunked at 2,000
   *characters* — 6,000 bytes for Telugu. 7.15% of the tokenizer's training text was being
   silently discarded, weighted toward the longest passages.

Both are the same class of failure: no exception, no warning, and output that looks entirely
reasonable.
