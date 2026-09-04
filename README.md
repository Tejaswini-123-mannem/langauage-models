# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Individual project, Language Models and Agents (Monsoon 2026).

Two completely independent decoder-only Transformer language models, built from
scratch: separate corpora, separate tokenizers, separate vocabularies, separate
weights. No pretrained models, no pretrained tokenizers, no HuggingFace
Transformer classes.

| | Model H | Model L |
|---|---|---|
| Language | **Telugu** (తెలుగు), `te` / `tel` | **Assamese** (অসমীয়া), `as` / `asm` |
| Script | Telugu, U+0C00–U+0C7F | Bengali-Assamese, U+0980–U+09FF |
| Training tokens | **647,207,005** | **559,359,252** |
| Manual share of training tokens | **22.02%** | **55.80%** |
| Training documents | 1,015,350 | 392,097 |
| Training characters | 2,288,825,203 | 1,767,282,379 |
| Tokenizer | SentencePiece Unigram, **8,000** pieces | SentencePiece Unigram, **8,000** pieces |

Both models exceed the ~500M token target and the ≥20% manual-collection
requirement. **No machine-translated or synthetic text is used in either
corpus.**

**Phase 1 reports:**
[Model H (Telugu)](report/phase1/telugu/model_h_telugu.md) ·
[Model L (Assamese)](report/phase1/assamese/model_l_assamese.md)

### Why both vocabularies are 8,000

The two vocabularies are trained separately on separate corpora and share no
pieces, as the assignment requires. They are the *same size* by independent
measurement, not by coupling.

Phase 2 fixes each model at ~25M parameters, and the embedding matrix is inside
that budget. At `d_model=384` with a tied output head the embedding costs
`vocab × 384`, while one Transformer block costs `12 × 384² = 1,769,472`. So
vocabulary is bought directly out of depth:

| vocab | embedding | layers within ~25M | embedding % |
|---|---|---|---|
| 8,000 | 3,072,000 | **12** | 13% |
| 16,000 | 6,144,000 | 10 | 30% |
| 32,000 | 12,288,000 | 7 | 53% |

Each language ran its own 6k/8k/10k/12k sweep and chose on held-out fertility.
Both landed on 8,000 for the same reason: 6k and 8k afford twelve layers while
10k and 12k afford eleven, so within twelve layers 8k strictly dominates 6k. A
side benefit for Phase 3 is that with granularity held constant, a Model H
versus Model L difference is attributable to the **data** rather than to one
model having a coarser tokenizer.

---

## Google Drive links

Large artifacts are not committed to git. Everything below is on Drive, shared
so it can be opened without requesting access.

| Artifact | Contents | Link |
|---|---|---|
| **Telugu** splits | `train.jsonl` (6.21 GB), `validation.jsonl`, `test.jsonl`, `split_statistics.csv`, `split_summary.json` | https://drive.google.com/drive/folders/1iIBbbT44FQG_PWLnkUaYRJKE--COGcD8?usp=sharing |
| **Telugu** tokenizers | `telugu_final_8000.{model,vocab}` (**delivered**), plus `telugu_final_16000` and `telugu_final_32000` retained as the record of the earlier selection | https://drive.google.com/drive/folders/1gbPBByd9VXcQgncJTV9I61TAfmKjI0vz?usp=sharing |
| **Telugu** vocabulary sweep | 8 candidates: 6k / 8k / 10k / 12k| https://drive.google.com/drive/folders/1Riuuf06Epv6osedd0N78n9603Ieuh_k7?usp=sharing |
| **Telugu** raw sources (8.0 GB) | 75 textbook PDFs, 3,106 Internet Archive `_djvu.txt`, 6 Sangraha parquet shards, `tewiki`/`tewikisource` dumps | https://drive.google.com/drive/folders/18Pdpv7-Ck76aNAHDtlq23Gfr4jnVNuxC?usp=sharing |
| **Assamese** splits | `train.jsonl` (4.66 GB), `validation.jsonl`, `test.jsonl`, `split_statistics.csv`, `split_summary.json` | https://drive.google.com/drive/folders/1YpxKdMwT4oaBXxu3y5par3okkihmh632?usp=sharing |
| **Assamese** tokenizer | `assamese_final_8000.{model,vocab}` with training metadata | https://drive.google.com/drive/folders/147l3I6VW_3iYj_kDnLjE531x3jhXzWsB?usp=sharing |
| **Assamese** vocabulary sweep | candidates at 6k / 8k / 10k / 12k, trained on a 10% sample | https://drive.google.com/drive/folders/1choNnxGTO9kij0FN4UdqmJO7o1RPKVnq?usp=sharing |
| **Assamese** raw sources (1.6 GB) | `archive.zip` (3,034 book texts, 514 MB), 3 Sangraha parquet shards, `aswiki`/`aswikisource` dumps | https://drive.google.com/drive/folders/1jCZRV1bhe1N4cLPsXlMimj9AjBdkrZEs?usp=sharing |
| **Telugu** checkpoints (Model H) | `best.pt`, `final.pt` (281 MB each), `train_log.jsonl`, `console.log` | https://drive.google.com/drive/folders/1QL21Wl45GYJA82YjVQyCQno366kZvgRY?usp=sharing |
| **Assamese** checkpoints (Model L) | `best.pt`, `final.pt` (281 MB each), `train_log.jsonl`, `console.log` | https://drive.google.com/drive/folders/18n4tms8r80zPj5YZatLq3iOVMlKCm_VZ?usp=sharing |



Drive layout:

```
LMA_Individual_Project/
    data/raw/telugu/                      textbooks/, internet_archive/, sangraha/,
                                          wikipedia/, wikisource/            (8.0 GB)
    data/raw/assamese/                    archive.zip, sangraha/,
                                          wikipedia/, wikisource/            (1.6 GB)
    data/processed/<language>/splits/     train.jsonl, validation.jsonl, test.jsonl
    tokenizer/<language>/final/           .model, .vocab, _metadata.json
    tokenizer/<language>/sweep/           vocabulary-size candidates
    checkpoints/<language>/               best.pt, final.pt, train_log.jsonl,
                                          console.log                    (1.1 GB)
```

Raw sources were originally stored per *source type* (`data/raw/manual/`,
`data/raw/internet_archive/`) from when Telugu was the only language. They are
now grouped per language, so the two models' inputs are separated the same way
everywhere and neither is the implicit default.

`python common/publish_deliverables.py --language <lang>` copies the splits,
both tokenizer directories (`final` and `sweep`), the statistics and the Phase 2
checkpoints; add `--include-raw` for the collected source files, `--dry-run` to
preview, and `--skip-checkpoints` when republishing only a tokenizer. The
checkpoints are staged at `<work>/<language>/checkpoints/` first — Phase 2
trains on Kaggle, so they arrive by download rather than from a local stage.

Two behaviours worth knowing:

- **Many small files are bundled into one zip each.** Assamese's 3,034 Internet
  Archive texts sync unreliably as loose files and cannot be downloaded in one
  action; as `archive.zip` they are 514 MB instead of 2,074 MB.
- **Staleness is decided by contents, not timestamps.** The `internetarchive`
  client preserves the server's modification times on download, so freshly
  collected books can carry timestamps older than the bundle holding them. An
  mtime check silently published 1,401 of 3,034 books while reporting "size
  matches"; the bundle's file list is now compared against the directory.

---

## Repository layout

```
common/                 pipeline stages shared by both languages, selected with --language
telugu/                 Model H: everything specific to Telugu
    collect/            Wikipedia and Wikisource dump collection
    preprocess/         Sangraha shard processing
    ocr/                textbook OCR (Tesseract), plus the Internet Archive harvest
    corpus/             corpus assembly, exact and near deduplication
    configs/            tokenizer training configurations
assamese/               Model L: everything specific to Assamese
    collect/            DLS Assam catalogue matcher
    configs/            tokenizer training configurations
notebooks/              Colab notebooks used for the Telugu preprocessing runs
report/phase1/          per-language reports, statistics, figures
digitized book list.xls DLS Assam digitised-book catalogue (2,146 Assamese titles)
```

### Why there is a `common/` directory

The two languages must be built by identical code, because Phase 3 has to
attribute the Model H / Model L difference to the **data**. If each language had
its own copy of the corpus and tokenizer stages, a bug fixed in one and not the
other would silently become part of that comparison.

So every stage from corpus assembly onward is written once and takes
`--language`. What differs between the languages is *configuration*, and it
lives in one place, [`common/language_config.py`](common/language_config.py):
script ranges, wiki markup patterns, source lists, mixing policy, tokenizer
sweep range. [`common/workspace.py`](common/workspace.py) resolves the paths.

The two models still share nothing that matters: separate corpora, separate
tokenizers, separate vocabularies, and in Phase 2 separate weights.

---

## Reproducing Phase 1

Everything is deterministic given seed 42. Split assignment is by content hash,
so regeneration reproduces the identical corpus — verified during the Assamese
rebuild, when Wikisource, Wikipedia and Sangraha each reproduced their previous
character counts exactly across three separate pipeline runs while only the
Internet Archive source changed.

### Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Telugu textbook OCR additionally needs the Tesseract binary with the `tel`
language pack. Assamese needs no OCR: Internet Archive supplies its own
`_djvu.txt`. Working files are written to `C:\lma_work\<language>\`, configured
in `common/workspace.py`.

### 1. Collect the sources

```bash
# Wikipedia and Wikisource dumps
python common/collect_wiki.py --language <lang> --wiki wikipedia  --download
python common/collect_wiki.py --language <lang> --wiki wikipedia  --extract --merge
python common/collect_wiki.py --language <lang> --wiki wikisource --download
python common/collect_wiki.py --language <lang> --wiki wikisource --extract --merge

# Sangraha (AI4Bharat), the only downloaded source
python common/collect_sangraha.py --language <lang> --download
python common/collect_sangraha.py --language <lang> --process

# Internet Archive
python common/collect_archive.py --language <lang> --search-only
python common/collect_archive.py --language <lang> --limit 20000 --workers 3
```

**Use 3 workers, not more.** Archive rate-limits by IP on concurrent
connections. At 32 workers the harvest was blocked at TCP level for hours; at 6
workers roughly half of requests timed out; at 3 workers 1,071 books were
collected with **zero errors** in 96 minutes. Lower concurrency was strictly
faster overall, because none of the work was wasted.

Assamese additionally cross-checks the DLS Assam catalogue, which reaches items
whose Archive language metadata is missing or wrong:

```bash
python assamese/collect/match_dls_catalogue.py --catalogue-only   # summarise
python assamese/collect/match_dls_catalogue.py                    # look titles up
python common/collect_archive.py --language assamese \
    --identifiers-file C:/lma_work/assamese/processed/archive/manifests/dls_candidates.txt
```

Telugu textbook OCR (Model H only):

```bash
python telugu/ocr/ocr_all_books.py
python telugu/ocr/normalize_ocr.py
python telugu/ocr/qualityfilter_ocr_archive.py
```

### 2. Assemble the corpus

```bash
python common/build_corpus_pipeline.py --language <lang>
```

Exact dedup (SHA-256 over NFC), near dedup (MinHash LSH, Jaccard 0.90), and an
OCR quality filter including a **word-purity gate** that rejects books whose OCR
broke more than half their words. Resumable via a checkpoint after each source.

> The checkpoint records which sources are complete. **Delete or move it if the
> collected data changed**, or the pipeline will skip a source that now has more
> documents than when the checkpoint was written.

### 3. Plan the mix, split, and train the tokenizer

```bash
# Provisional tokenizer, used only to measure characters-per-token per source
python common/train_tokenizer.py --config <lang>/configs/tokenizer_provisional.json

# Decide how much of each source to keep, in TOKENS
python common/plan_mix.py --language <lang> \
    --tokenizer "C:/lma_work/<lang>/tokenizer/provisional/<lang>_provisional.model" \
    --target-total 560000000 --target-manual 330000000

# Document-level train / validation / test split, assigned by hash
python common/make_splits.py --language <lang> --val-fraction 0.005 --test-fraction 0.005

# Vocabulary-size sweep on a 10% sample: 6k, 8k, 10k, 12k
python common/run_vocab_sweep.py --language <lang> --force

# Evaluate the candidates on HELD-OUT text and choose a size
python common/evaluate_tokenizer.py --language <lang> --models <sweep .model files>

# Train the chosen tokenizer on the training split only
python common/train_tokenizer.py --config <lang>/configs/tokenizer_final.json
```

> `--target-manual` **caps** manual tokens. Its 120M default was set when manual
> data was scarce; Assamese now has 312M, and the default would discard most of
> it. Set both targets above what you actually hold if nothing should be trimmed.

> `--force` on the sweep retrains candidates whose model file already exists.
> Required whenever the training split changed, because otherwise a stale
> candidate fitted to a different corpus is compared against fresh ones.

### 4. Statistics, figures, and publication

```bash
# Exact token counts and the graded manual token share
python common/corpus_stats.py --language <lang> \
    --tokenizer "C:/lma_work/<lang>/tokenizer/final/<lang>_final_8000.model"

# Final tokenizer evaluation, including the per-source fertility figure
python common/evaluate_tokenizer.py --language <lang> \
    --models <sweep .model files> "C:/lma_work/<lang>/tokenizer/final/<lang>_final_8000.model" \
    --primary "C:/lma_work/<lang>/tokenizer/final/<lang>_final_8000.model"

# Copy the deliverables to Drive
python common/publish_deliverables.py --language <lang> --include-raw --dry-run
python common/publish_deliverables.py --language <lang> --include-raw
```

Steps 3 and 4 write into `report/phase1/<language>/`.

---

## Phase 1 deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts | `common/collect_*.py`, `telugu/collect/`, `telugu/ocr/`, `assamese/collect/` |
| 2 | Preprocessing pipelines | `common/build_corpus_pipeline.py`, `telugu/corpus/`, `notebooks/` |
| 3 | Dataset statistics reports | `report/phase1/<lang>/model_*.md`, `dataset_statistics.{csv,json}` |
| 4 | Train / validation / test splits | Drive; definitions in `common/make_splits.py` |
| 5 | Tokenizer training code | `common/train_tokenizer.py`, `common/run_vocab_sweep.py` |
| 6 | Vocabulary files | Drive, `*.vocab` |
| 7 | Tokenizer model files | Drive, `*.model` |

Figures are in `report/phase1/<language>/figures/`.

---

## Phase 2: pretraining and evaluation

Two independent decoder-only Transformers, one per language. Nothing is shared
between them — separate corpora, tokenizers, vocabularies and weights — and
they are trained as two processes rather than one distributed job.

**Phase 2 report:** [report/phase2/phase2_report.md](report/phase2/phase2_report.md)
— architecture, training, evaluation, attention and the resource-level
comparison in one place. The sections below summarise it.

### Architecture

Identical for both models:

| | |
|---|---|
| Layers | 12 |
| Attention heads | 6 (d_k 64) |
| Model dimension | 384 |
| Feed-forward dimension | 1,536 |
| Context length | 512 tokens |
| Vocabulary | 8,000 |
| Dropout | 0.1 |
| Weight tying | input embedding = output projection |
| Parameters | 24,562,944 |

The two are identical on purpose. They differ only in the data they saw, so
every difference in the results below is a statement about the corpora rather
than about the architecture.

### Training

One pass over each language's training ribbon, both models trained
simultaneously on a Kaggle T4 x2, one per GPU. Batch is 24 micro × 5
accumulation = 61,440 tokens per step; AdamW (β 0.9/0.95, weight decay 0.1,
gradient clip 1.0), learning rate 6e-4 decayed by cosine to 6e-5 after 400
warmup steps, fp16 with a GradScaler.

| | Model H (Telugu) | Model L (Assamese) |
|---|---|---|
| Steps | 10,550 | 9,110 |
| Tokens seen | 648,192,000 | 559,718,400 |
| Tokens per parameter | 26 | 23 |
| Wall clock | 5:17:58 | 4:34:10 |
| Throughput | 34,030 tok/s | 34,079 tok/s |
| Final validation loss | 3.4234 | 3.9043 |
| Final validation perplexity | 30.67 | 49.62 |

**fp16 rather than bf16, on a card that claims to support both.**
`torch.cuda.is_bf16_supported()` returns True on a T4, which is misleading: a
T4 is Turing and has no bf16 hardware at all, so what is being reported is
emulation — slower than fp16, not faster.

### Test-set language modelling

Neither model has seen the test split, and neither tokenizer was trained on it.

| | Model H (Telugu) | Model L (Assamese) |
|---|---|---|
| Cross-entropy | 3.4891 nats/token | 3.8859 nats/token |
| Perplexity | 32.76 | 48.71 |
| Bits per token | 5.0337 | 5.6062 |
| **Bits per byte** | **0.5369** | **0.6509** |
| Test tokens | 3,197,440 | 2,881,024 |

**Bits per byte is the only figure here that compares the two models.**
Perplexity is per *token*, and a token means something different in each
model — the two tokenizers were trained on different corpora and segment at
different rates (3.54 vs 3.16 characters per token). Normalising to bytes
removes that, and on bytes Telugu is ahead by 0.114 bits, which is what the
larger and cleaner corpus buys.

Per source, on the first 2,000 test documents:

| Model H (Telugu) | docs | tokens | ppl | bits/byte |
|---|---|---|---|---|
| wikipedia | 395 | 157,439 | 10.88 | 0.3561 |
| sangraha | 400 | 139,568 | 28.27 | 0.4767 |
| textbook | 74 | 33,399 | 36.37 | 0.6521 |
| wikisource | 39 | 18,110 | 34.01 | 0.9219 |
| archive | 7 | 3,584 | 34.58 | 1.0084 |

| Model L (Assamese) | docs | tokens | ppl | bits/byte |
|---|---|---|---|---|
| wikipedia | 101 | 38,992 | 35.09 | 0.5405 |
| sangraha | 500 | 168,959 | 36.54 | 0.5463 |
| wikisource | 286 | 95,366 | 73.52 | 0.7722 |
| archive | 8 | 4,096 | 45.14 | 1.0914 |

Both models rank their sources the same way, and the spread within a language
is wider than the gap between the languages: Telugu Wikipedia costs 0.356
bits/byte while Telugu Archive costs 1.008, nearly three times as much. That
is a statement about scan quality rather than about either language — the
Archive rows are OCR output, and the model is paying for the errors.

Note that the perplexity and bits/byte columns disagree about which source is
hardest. Telugu `textbook` has the worst perplexity (36.37) but `archive` the
worst bits/byte (1.0084), because OCR noise fragments into many short tokens:
each individual token is easier to predict, and there are far more of them per
byte of text.

### Generation

200 test documents, a 64-token prefix continued for 64 tokens, under four
decoding settings.

| Model H (Telugu) | BLEU-4 | chrF2 | ROUGE-L | dist-1 | dist-2 | rep-4 |
|---|---|---|---|---|---|---|
| greedy | 5.351 | 15.675 | 0.0751 | 0.2185 | 0.2982 | 0.539 |
| t = 0.5 | **5.823** | 18.092 | 0.0826 | 0.3956 | 0.5922 | 0.234 |
| t = 1.0 | 4.114 | **22.726** | 0.0671 | 0.7277 | 0.9585 | 0.004 |
| t = 1.5 | 0.928 | 20.204 | 0.0225 | 0.7980 | 0.9974 | 0.000 |

| Model L (Assamese) | BLEU-4 | chrF2 | ROUGE-L | dist-1 | dist-2 | rep-4 |
|---|---|---|---|---|---|---|
| greedy | 1.420 | 13.765 | 0.0487 | 0.1732 | 0.2848 | 0.574 |
| t = 0.5 | **1.791** | 17.256 | 0.0612 | 0.3413 | 0.6418 | 0.173 |
| t = 1.0 | 0.606 | **19.022** | 0.0447 | 0.6358 | 0.9738 | 0.000 |
| t = 1.5 | 0.102 | 18.068 | 0.0204 | 0.6992 | 0.9976 | 0.000 |

**Greedy decoding degenerates.** A repetition rate of 0.539 means over half of
all 4-grams in the output are repeats, with distinct-1 down at 0.22 — the
samples in `report/phase2/<language>/generation_samples_test.md` show it
directly, as runs of a single token repeated to the length limit. This is the
expected failure at 24.6M parameters, not a fault in the decoder.

**The two metrics disagree, and chrF is the one to believe.** BLEU-4 peaks at
t=0.5 while chrF2 peaks at t=1.0. chrF is character-based, so it does not
depend on how these scripts are split into words; BLEU's `intl` tokenizer
splits on Unicode punctuation and leaves Indic runs intact, which is the best
option available but still a word-level assumption. At these values BLEU-4 is
also near its own noise floor — Model L's 1.79 rests on a 4-gram precision of
0.7%. Read by chrF, t=1.0 is the best decoding for both models and greedy is
the *worst*, despite greedy ranking second on BLEU.

Every score carries its sacrebleu signature in
`report/phase2/<language>/generation_test.json`, so the numbers can be
reproduced rather than merely quoted.

### Attention

8 test sequences of 128 tokens, all 12 layers, entropy normalised against a
uniform distribution over the positions each query is allowed to see.

Both models learned the same three-part structure: layers 0–1 attend broadly
and at long range (normalised entropy ≈ 0.9), the middle layers collapse to
sharp local attention — Telugu layers 3–5 look 8–12 positions back, entropy
bottoming at 0.399 — and the last four re-broaden to long range. Local syntax
in the middle of the stack, discourse-level context at the top.

Model L layer 8 is the one departure from the pattern: the sharpest attention
in either model (entropy 0.354) combined with the longest mean distance
(55.37). Sharp *and* long-range is the signature of a head doing retrieval
rather than smoothing, and it is the most interesting layer in either model.

### Reproducing Phase 2

Training runs on Kaggle, not locally. `notebooks/phase2_kaggle_pretrain.py` is
the source of the notebook, so the run is reproducible from this branch rather
than only from a Kaggle account.

Upload the code, configs and both tokenizers as the `lma-code` dataset and the
token bins as `lma-<language>-tokens`. Then, in the notebook: Accelerator
**GPU T4 x2**, Internet **On**, *Add Data* for all three datasets, and run with
**Save & Run All** — an interactive session that ends takes `/kaggle/working`
with it, checkpoints included.

Evaluation, against a downloaded checkpoint:

```bash
python common/evaluate.py  --checkpoint <best.pt> --split test --out report/phase2/<lang>
python common/generate.py  --checkpoint <best.pt> --split test --out report/phase2/<lang> --n-samples 200
python common/attention.py --checkpoint <best.pt> --split test --out report/phase2/<lang> --n-sequences 8
python common/plots.py --runs <runs directory> --out report/phase2
```

`common/test_train_resume.py` proves bit-for-bit that a resumed run continues
the run it interrupted: checkpoints carry model, optimizer, scheduler, step,
sampler position and RNG state, so step 4,001 after a resume is the step 4,001
the uninterrupted run would have produced.

### Phase 2 deliverables

| # | Deliverable | Where |
|---|---|---|
| 8 | Model implementation | `common/model.py`, `common/test_model.py` |
| 9 | Training code | `common/train.py`, `common/launch_both.py`, `<lang>/configs/` |
| 10 | Trained checkpoints | Drive, `best.pt` and `final.pt` per language |
| 11 | Perplexity evaluation | `common/evaluate.py`, `report/phase2/<lang>/eval_test.json` |
| 12 | Generation and metrics | `common/generate.py`, `report/phase2/<lang>/generation_*.json` |
| 13 | Attention analysis | `common/attention.py`, `report/phase2/<lang>/attention*.{json,md}` |
| 14 | Training curves | `report/phase2/figures/` |
| 15 | Training logs | `report/phase2/<lang>/train_log.jsonl`, `console.log` |

Figures are in `report/phase2/figures/` (loss, perplexity, learning rate,
throughput, loss against tokens) and `report/phase2/<language>/figures/`
(attention heatmaps for layers 0 and 11, entropy and distance by layer).

The logs are committed rather than only linked, since a grader reads the branch
and not Drive. `train_log.jsonl` is one JSON object per record — `train` lines
carry step, loss, perplexity, learning rate, tokens, throughput and elapsed
time, `eval` lines the validation loss and whether it was a new best, and a
final `done` line the totals — and it is what `common/plots.py` draws the curves
from. `console.log` is the run's stdout: the resolved configuration, the
parameter count, every validation evaluation and every checkpoint write.

---

## Results summary

| | Model H (Telugu) | Model L (Assamese) |
|---|---|---|
| Training tokens | 647,207,005 | 559,359,252 |
| Manual tokens | 142,510,452 | 312,112,673 |
| Manual share | 22.02% | **55.80%** |
| Largest source | Sangraha, 78.0% | Internet Archive, 48.9% |
| Internet Archive documents | 819 | **2,397** |
| Characters per token | 3.54 | 3.16 |
| Vocabulary utilisation (held-out) | 97.97% | 97.91% |
| Unknown-token rate | 0.0000% | 0.0000% |

**The resource tiers inverted on manual collection.** Model L carries 55.80%
manual tokens against Model H's 22.02%, and its Internet Archive collection is
nearly three times larger. Telugu could lean on Sangraha for 78% of its tokens;
Assamese had no such fallback and had to be built from historical book scans.
That difference — and the OCR fertility cost that comes with it, where Assamese
Archive text needs 34% more tokens per word than clean web prose — is the main
data-side asymmetry Phase 3 has to account for.

---

## Status

- **Phase 1, Model H (Telugu):** complete — 647,207,005 training tokens,
  22.02% manual, 8,000-piece Unigram tokenizer.
- **Phase 1, Model L (Assamese):** complete — 559,359,252 training tokens,
  55.80% manual, 8,000-piece Unigram tokenizer.
- **Phase 2, Model H (Telugu):** complete — 24.6M parameters, 10,550 steps
  over 648M tokens, test perplexity 32.76 at 0.5369 bits/byte.
- **Phase 2, Model L (Assamese):** complete — 24.6M parameters, 9,110 steps
  over 560M tokens, test perplexity 48.71 at 0.6509 bits/byte.

