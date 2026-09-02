"""
Phase 2, stage 0: turn the JSONL splits into flat uint16 token streams.

Why this stage exists
---------------------
Phase 1 delivered train/validation/test as JSONL, one document per line,
as text. A training loop cannot use that shape. Its inner loop is:

    pick a random offset i, return tokens[i : i + block_size + 1]

several thousand times a second. Doing that from JSONL would mean parsing
a JSON line and running SentencePiece's unigram Viterbi decoder inside the
loop, on the CPU, while the GPU waits. The GPU is the scarce resource --
this project has a 30 hour weekly Kaggle quota -- so tokenization happens
exactly once, here, and the training loop never sees text again.

The output format
-----------------
One flat little-endian uint16 array per split. Every document is encoded,
then a single EOS piece is appended, then the next document follows. The
result is one continuous ribbon of token ids with no padding and no
document alignment.

uint16 rather than uint32 because both vocabularies are 8,000 pieces and
uint16 holds 0..65535. That halves the file, and it is a direct dividend
of the Phase 1 vocabulary decision:

    telugu    ~647M tokens -> ~1.29 GB
    assamese  ~559M tokens -> ~1.12 GB

Small enough to upload to Kaggle as a dataset and np.memmap on the worker,
where only the pages actually touched are ever resident.

Why documents are joined by EOS and not padded
----------------------------------------------
A decoder-only LM trains on contiguous token sequences, so we concatenate
everything and cut fixed-length windows out of the ribbon. A window may
straddle a document boundary; that is intentional and is how GPT-2 onward
is trained. Padding to a fixed length instead would waste a large fraction
of every batch on documents shorter than the context, and would need an
attention mask on top of the causal one.

The EOS piece between documents is what carries the boundary information.
Without it the model would learn that a Wikipedia stub flows seamlessly
into a 1920s book scan, which is false and would degrade generation.

Why a sidecar .json per split
-----------------------------
The worst failure mode available in Phase 2 is training on a .bin that was
built with the wrong tokenizer. Nothing downstream reveals it: the loss
curve looks normal, the samples look like noise for a while and then like
text, and the reported perplexity is simply meaningless. So each .bin is
written beside a manifest recording the token count, the tokenizer file
and its SHA-256, and the source split's SHA-256. The trainer asserts
against it at startup rather than trusting a filename.

Why validation and test also get a document index
-------------------------------------------------
Perplexity can be measured on arbitrary windows, but the generation
evaluation cannot. Scoring BLEU or chrF needs a prefix that starts at a
real document start and a reference continuation from that same document
-- a window that begins mid-sentence in one book and continues into
another is not something a reference can be defined for. The index records
where each held-out document begins, how long it is, and which source it
came from, so the evaluation can sample prefixes across sources rather
than accidentally drawing all of them from Sangraha.

The index is not built for train: it would be a 1M-entry list nobody reads.

Usage
-----
    python common/build_token_bins.py --language telugu
    python common/build_token_bins.py --language assamese

    # smoke test on the small splits first
    python common/build_token_bins.py --language telugu --splits validation test

    # rebuild something that already exists
    python common/build_token_bins.py --language telugu --force
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================

SPLITS = ["train", "validation", "test"]

# Splits that get a per-document index. Generation evaluation draws its
# prefixes from held-out documents; train is only ever read as a ribbon.
INDEXED_SPLITS = ("validation", "test")

# Token id dtype. Asserted against the real vocabulary size at startup so
# that a future larger vocabulary fails loudly instead of wrapping around
# and silently producing a corpus of wrong token ids.
TOKEN_DTYPE = np.uint16

# Raw bytes of JSONL handed to one worker at a time. Chunking by BYTES
# rather than by line count is deliberate: Assamese Internet Archive
# documents are whole books averaging ~318,000 characters, so a chunk of
# "1000 documents" would be 300 MB in one language and 2 MB in the other.
CHUNK_BYTES = 4 * 1024 * 1024

# Progress line cadence, in chunks.
REPORT_EVERY = 64


# =============================================================================
# WORKER
# =============================================================================
#
# A SentencePieceProcessor cannot be pickled, so it cannot be passed as an
# argument to a pool task. Each worker loads its own copy once, in an
# initializer, and keeps it in a module-level global for the life of the
# process.
# =============================================================================

_WORKER = {}


def _init_worker(model_path, eos_id):
    """Load one SentencePiece model per worker process."""

    _WORKER["sp"] = spm.SentencePieceProcessor(model_file=str(model_path))
    _WORKER["eos"] = eos_id


def _encode_chunk(payload):
    """
    Encode one chunk of JSONL lines into a uint16 token array.

    Returns (tokens, doc_lengths, n_docs, n_chars, n_bytes, sources) where
    `tokens` already has one EOS appended per document, `doc_lengths`
    counts those EOS pieces, and `sources` is the per-document source_type
    used by the held-out document index.

    Documents are encoded independently, so the output does not depend on
    where the chunk boundaries happened to fall. That is what keeps the
    .bin byte-for-byte reproducible across runs and worker counts.
    """

    sp = _WORKER["sp"]
    eos = _WORKER["eos"]

    texts = []
    sources = []
    n_chars = 0
    n_bytes = 0

    for line in payload:
        n_bytes += len(line)
        record = json.loads(line)
        text = record.get("text") or ""
        if not text:
            continue
        texts.append(text)
        sources.append(record.get("source_type", "unknown"))
        n_chars += len(text)

    if not texts:
        empty = np.empty(0, dtype=TOKEN_DTYPE)
        return empty, [], 0, 0, n_bytes, []

    # encode_as_ids on a list batches the C++ side, which is measurably
    # faster than a Python loop over single documents.
    encoded = sp.encode(texts, out_type=int)

    doc_lengths = []
    total = 0
    for ids in encoded:
        total += len(ids) + 1          # +1 for the EOS separator
        doc_lengths.append(len(ids) + 1)

    tokens = np.empty(total, dtype=TOKEN_DTYPE)
    at = 0
    for ids in encoded:
        n = len(ids)
        tokens[at:at + n] = ids
        tokens[at + n] = eos
        at += n + 1

    return tokens, doc_lengths, len(texts), n_chars, n_bytes, sources


# =============================================================================
# COMPUTATION
# =============================================================================


def sha256_file(path, block=16 * 1024 * 1024):
    """SHA-256 of a file, read in blocks so a 6 GB split does not load."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block_bytes in iter(lambda: handle.read(block), b""):
            digest.update(block_bytes)

    return digest.hexdigest()


def iter_chunks(path, chunk_bytes=CHUNK_BYTES):
    """
    Yield lists of raw JSONL lines, each list about `chunk_bytes` long.

    Lines are yielded as bytes and decoded in the worker, so the main
    process never pays the UTF-8 decode cost for the whole corpus.
    """

    chunk = []
    size = 0

    with open(path, "rb") as handle:
        for line in handle:
            chunk.append(line)
            size += len(line)
            if size >= chunk_bytes:
                yield chunk
                chunk = []
                size = 0

    if chunk:
        yield chunk


def format_duration(seconds):
    """Seconds as H:MM:SS, for progress and ETA lines."""

    seconds = int(seconds)

    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def build_split(split, source_path, out_bin, model_path, eos_id,
                workers, want_index):
    """
    Tokenize one split into a flat uint16 .bin, and summarise it.

    The pool is driven with imap (not imap_unordered) so chunks are written
    in file order. Order matters twice over: the document index computed
    here has to point at real offsets in the .bin, and a reproducible build
    is what lets the manifest's hashes mean anything.
    """

    source_size = source_path.stat().st_size
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  {split}")
    print(f"    source : {source_path}  ({source_size / 1e9:.2f} GB)")
    print(f"    output : {out_bin}")

    n_tokens = 0
    n_docs = 0
    n_chars = 0
    bytes_done = 0
    documents = []          # [offset, n_tokens, source_type] for held-out splits

    started = time.time()

    with open(out_bin, "wb") as sink:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(model_path, eos_id),
        ) as pool:
            results = pool.imap(
                _encode_chunk,
                iter_chunks(source_path),
                chunksize=1,
            )

            for i, (tokens, doc_lengths, docs, chars, raw_bytes, sources) in enumerate(results, 1):
                if want_index:
                    at = n_tokens
                    for length, source in zip(doc_lengths, sources):
                        documents.append([at, length, source])
                        at += length

                sink.write(tokens.tobytes())

                n_tokens += int(tokens.size)
                n_docs += docs
                n_chars += chars
                bytes_done += raw_bytes

                if i % REPORT_EVERY == 0:
                    elapsed = time.time() - started
                    share = bytes_done / source_size
                    eta = elapsed / share - elapsed if share > 0 else 0
                    print(
                        f"      {share * 100:5.1f}%  "
                        f"{n_tokens / 1e6:8.1f}M tokens  "
                        f"{n_tokens / max(elapsed, 1e-9) / 1e3:6.1f}k tok/s  "
                        f"elapsed {format_duration(elapsed)}  "
                        f"eta {format_duration(eta)}",
                        flush=True,
                    )

    elapsed = time.time() - started

    summary = {
        "split": split,
        "documents": n_docs,
        "characters": n_chars,
        "tokens": n_tokens,
        "bytes": out_bin.stat().st_size,
        "chars_per_token": round(n_chars / n_tokens, 4) if n_tokens else 0.0,
        "build_seconds": round(elapsed, 1),
    }

    print(
        f"    done   : {n_tokens:,} tokens from {n_docs:,} documents "
        f"in {format_duration(elapsed)}"
    )

    return summary, documents


# =============================================================================
# ORCHESTRATION
# =============================================================================


def resolve_tokenizer(workspace, vocab_size, override):
    """
    Locate the tokenizer to encode with, and refuse to guess.

    Telugu's final/ directory holds three models: the delivered 8,000-piece
    tokenizer and the superseded 16,000 and 32,000 ones, kept as the record
    of the earlier selection. Picking by glob order or by file size would
    work today and break the moment a fourth appears, and the failure would
    be silent -- a corpus tokenized against the wrong vocabulary trains and
    reports perplexity perfectly happily.

    So the vocabulary size is named explicitly (default 8,000, which is what
    Phase 1 selected for both languages), and --tokenizer overrides it with
    a path for anything unusual.
    """

    if override:
        return workspace.require(Path(override), hint="Check the --tokenizer path.")

    return workspace.require(
        workspace.final_model(vocab_size),
        hint=(
            f"Run: python common/train_tokenizer.py "
            f"--config {workspace.slug}/configs/tokenizer_final.json\n"
            f"Available: "
            f"{[p.name for p in sorted(workspace.tokenizer_final.glob('*.model'))]}"
        ),
    )


def build_language(workspace, splits, workers, force, vocab_size, tokenizer):
    """
    Binarize every requested split for one language.

    Each split is skipped if its .bin and manifest already exist and the
    manifest agrees with the current tokenizer, unless --force. Rebuilding
    a 1.3 GB file to discover it is identical is 20 wasted minutes of a
    three-day budget.
    """

    model_path = resolve_tokenizer(workspace, vocab_size, tokenizer)

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()
    eos_id = sp.eos_id()

    if vocab_size > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"Vocabulary {vocab_size} does not fit in {TOKEN_DTYPE.__name__}; "
            f"widen TOKEN_DTYPE before rebuilding."
        )
    if eos_id < 0:
        raise ValueError(f"{model_path} has no EOS piece; cannot separate documents.")

    print("=" * 78)
    print(f"TOKEN BINS - {workspace.title}")
    print("=" * 78)
    print(f"  tokenizer : {model_path.name}  ({vocab_size:,} pieces, eos={eos_id})")
    print(f"  dtype     : {TOKEN_DTYPE.__name__}  ({np.dtype(TOKEN_DTYPE).itemsize} bytes/token)")
    print(f"  workers   : {workers}")
    print(f"  out dir   : {workspace.tokens}")

    print("\n  hashing tokenizer ...", end=" ", flush=True)
    tokenizer_sha = sha256_file(model_path)
    print(tokenizer_sha[:16])

    workspace.tokens.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for split in splits:
        source_path = workspace.require(
            workspace.split_file(split),
            hint=f"Run: python common/make_splits.py --language {workspace.slug}",
        )
        out_bin = workspace.token_file(split)
        out_meta = workspace.token_manifest(split)

        if out_bin.exists() and out_meta.exists() and not force:
            existing = json.loads(out_meta.read_text(encoding="utf-8"))
            if existing.get("tokenizer_sha256") == tokenizer_sha:
                print(f"\n  {split}: already built "
                      f"({existing['tokens']:,} tokens) - skipping, use --force to rebuild")
                summaries[split] = existing
                continue
            print(f"\n  {split}: manifest was built with a DIFFERENT tokenizer - rebuilding")

        print(f"\n  hashing {split}.jsonl ...", end=" ", flush=True)
        source_sha = sha256_file(source_path)
        print(source_sha[:16])

        summary, documents = build_split(
            split=split,
            source_path=source_path,
            out_bin=out_bin,
            model_path=model_path,
            eos_id=eos_id,
            workers=workers,
            want_index=split in INDEXED_SPLITS,
        )

        summary.update({
            "language": workspace.slug,
            "dtype": TOKEN_DTYPE.__name__,
            "vocab_size": vocab_size,
            "eos_id": eos_id,
            "tokenizer": str(model_path),
            "tokenizer_sha256": tokenizer_sha,
            "source": str(source_path),
            "source_sha256": source_sha,
            "bin": str(out_bin),
        })

        out_meta.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries[split] = summary

        if documents:
            index_path = workspace.token_index(split)
            index_path.write_text(
                json.dumps(
                    {
                        "split": split,
                        "language": workspace.slug,
                        "n_documents": len(documents),
                        "fields": ["offset", "n_tokens", "source_type"],
                        "documents": documents,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"    index  : {len(documents):,} documents -> {index_path.name}")

    return summaries


# =============================================================================
# REPORTING
# =============================================================================


def print_summary_table(workspace, summaries):
    """Print the per-split totals the Phase 2 report quotes."""

    print("\n" + "=" * 78)
    print(f"SUMMARY - {workspace.title}")
    print("=" * 78)
    print(f"  {'split':<12}{'documents':>14}{'tokens':>16}{'size':>12}{'chars/tok':>12}")
    print("  " + "-" * 64)

    for split, summary in summaries.items():
        print(
            f"  {split:<12}"
            f"{summary['documents']:>14,}"
            f"{summary['tokens']:>16,}"
            f"{summary['bytes'] / 1e9:>11.2f}G"
            f"{summary['chars_per_token']:>12.3f}"
        )

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize the Phase 1 splits into flat uint16 .bin files.",
    )
    add_language_argument(parser)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Which splits to build. Default: all three.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Encoder processes. Default: one fewer than the CPU count.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Which final tokenizer to encode with. Default: 8000, the size "
             "Phase 1 selected for both languages.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Explicit path to a .model file, overriding --vocab-size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild splits whose .bin already exists.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    workspace = Workspace(get_language(args.language))

    summaries = build_language(
        workspace=workspace,
        splits=args.splits,
        workers=args.workers,
        force=args.force,
        vocab_size=args.vocab_size,
        tokenizer=args.tokenizer,
    )

    print_summary_table(workspace, summaries)


if __name__ == "__main__":
    main()
