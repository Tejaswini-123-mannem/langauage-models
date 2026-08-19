"""
Tokenizer training for Model H (Telugu).

Trains a SentencePiece tokenizer from scratch. No pretrained tokenizer or
pretrained vocabulary is used anywhere in this project, which the
assignment forbids.

This same script is used twice:

  1. PROVISIONAL run, on a sample of the assembled corpus. Its only job is
     to give a characters-per-token estimate per source so the source mix
     (~400M Sangraha tokens / ~100M manual tokens) can be planned. A mix
     cannot be planned in characters, because the assignment states the
     target and the >=20% manual rule in TOKENS.

  2. FINAL run, on the training split only. Training the delivered
     tokenizer on the train split alone keeps validation and test text
     genuinely held out: a tokenizer fitted on the test set has already
     seen it, and the resulting perplexity and fertility numbers would be
     optimistic.

Design notes
------------
SentencePiece reads plain text, one sequence per line, so the JSONL corpus
is first exported to a line-oriented text file. Two details matter for
this corpus:

  * Archive documents are whole books, averaging ~185k characters.
    SentencePiece silently drops any line longer than max_sentence_length,
    so documents are split on newlines and over-long lines are further
    chunked. Without this, the entire Archive source would vanish from
    tokenizer training.

  * input_sentence_size with shuffle_input_sentence caps how many lines
    the trainer holds in memory while still sampling across the whole
    corpus rather than just the first N lines.

Usage
-----
    python train_tokenizer.py --config ../configs/tokenizer_provisional.json
    python train_tokenizer.py --config ../configs/tokenizer_final.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import sentencepiece as spm


# =============================================================================
# CORPUS EXPORT
# =============================================================================

# Longest line written to the tokenizer training text, in CHARACTERS.
#
# SentencePiece's own max_sentence_length is in BYTES, and Telugu costs
# 3 bytes per character in UTF-8, so 2000 characters can reach ~6000
# bytes. That is above SentencePiece's 4192-byte default, and lines over
# the limit are silently dropped: measured on the real corpus, that was
# 0.72% of lines but 7.15% of the text, because the lines lost are the
# longest and most continuous prose in the corpus.
#
# The limit is therefore raised on the trainer side (MAX_SENTENCE_BYTES
# below) rather than shortening lines here, since chopping sentences
# further would destroy exactly the long-range context that makes good
# subword units learnable.
MAX_LINE_CHARS = 2000

# Passed to SentencePiece as max_sentence_length. Sized to cover
# MAX_LINE_CHARS at 4 bytes per character, the worst case for any
# Unicode codepoint, with headroom.
MAX_SENTENCE_BYTES = 16384


def iter_documents(input_files):
    """
    Yield (text, source_type) for every document in the given JSONL files.
    """

    for path in input_files:

        with open(path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                text = record.get("text")

                if isinstance(text, str) and text.strip():
                    yield text, record.get("source_type", "unknown")


def split_into_lines(text):
    """
    Turn one document into tokenizer training lines.

    Splits on newlines first, since that preserves the natural sentence and
    verse boundaries in textbooks and Wikisource poetry. Any remaining
    over-long line (common in Archive books, where an entire page can be a
    single line) is hard-chunked so SentencePiece does not discard it.
    """

    for raw_line in text.split("\n"):

        line = raw_line.strip()

        if not line:
            continue

        if len(line) <= MAX_LINE_CHARS:
            yield line
            continue

        for start in range(0, len(line), MAX_LINE_CHARS):
            chunk = line[start:start + MAX_LINE_CHARS].strip()

            if chunk:
                yield chunk


def export_training_text(input_files, output_file, sample_rate, seed):
    """
    Write the line-oriented text file SentencePiece trains on.

    sample_rate < 1.0 keeps a random fraction of documents. The provisional
    run uses this to stay fast; the final run uses 1.0.

    Returns per-source line and character counts.
    """

    rng = random.Random(seed)

    stats = {}

    total_lines = 0
    total_chars = 0

    with open(output_file, "w", encoding="utf-8") as fout:

        for text, source_type in iter_documents(input_files):

            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue

            entry = stats.setdefault(
                source_type,
                {"documents": 0, "lines": 0, "characters": 0},
            )

            entry["documents"] += 1

            for line in split_into_lines(text):

                fout.write(line + "\n")

                entry["lines"] += 1
                entry["characters"] += len(line)

                total_lines += 1
                total_chars += len(line)

                if total_lines % 1_000_000 == 0:
                    print(
                        f"  exported {total_lines:,} lines "
                        f"({total_chars:,} chars)"
                    )

    return stats, total_lines, total_chars


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """
    Train one SentencePiece model from a configuration dictionary.
    """

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = output_dir / config["model_prefix"]

    input_files = [Path(p) for p in config["input_files"]]

    missing = [p for p in input_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing corpus files:\n" + "\n".join(str(p) for p in missing)
        )

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    # A sweep trains several vocabulary sizes over identical text, so the
    # export can be shared between them via "training_text" in the config.
    # Re-exporting the same sample once per candidate would read the
    # multi-gigabyte training split several times for no benefit.
    training_text = Path(
        config.get("training_text")
        or output_dir / f"{config['model_prefix']}_input.txt"
    )

    print()
    print("=" * 78)
    print("EXPORTING TOKENIZER TRAINING TEXT")
    print("=" * 78)
    print(f"Input files : {len(input_files)}")
    print(f"Sample rate : {config['sample_rate']}")
    print(f"Output      : {training_text}")

    start = time.perf_counter()

    # The export is deterministic given the same inputs, sample rate and
    # seed, so an existing file is reused rather than rebuilt. The sidecar
    # records what produced it, so a stale export from different settings
    # is detected instead of being silently reused.
    sidecar = training_text.with_suffix(".export.json")

    signature = {
        "input_files": [str(p) for p in input_files],
        "sample_rate": config["sample_rate"],
        "seed": config["seed"],
    }

    reusable = False

    if training_text.exists() and sidecar.exists():

        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                previous = json.load(f)

            reusable = previous.get("signature") == signature

        except (json.JSONDecodeError, OSError):
            reusable = False

    if reusable:

        with open(sidecar, "r", encoding="utf-8") as f:
            previous = json.load(f)

        stats = previous["stats"]
        total_lines = previous["total_lines"]
        total_chars = previous["total_characters"]

        print(
            f"  reusing existing export "
            f"({total_lines:,} lines, {total_chars:,} chars)"
        )

    else:

        stats, total_lines, total_chars = export_training_text(
            input_files,
            training_text,
            config["sample_rate"],
            config["seed"],
        )

        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signature": signature,
                    "stats": stats,
                    "total_lines": total_lines,
                    "total_characters": total_chars,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print()
    print(f"{'source':<12} {'documents':>12} {'lines':>14} {'characters':>16}")
    print("-" * 58)

    for source_type in sorted(stats):
        entry = stats[source_type]
        print(
            f"{source_type:<12} "
            f"{entry['documents']:>12,} "
            f"{entry['lines']:>14,} "
            f"{entry['characters']:>16,}"
        )

    print("-" * 58)
    print(f"{'TOTAL':<12} {'':>12} {total_lines:>14,} {total_chars:>16,}")
    print(f"Export time: {time.perf_counter() - start:.1f}s")

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING SENTENCEPIECE")
    print("=" * 78)
    print(f"model_type       : {config['model_type']}")
    print(f"vocab_size       : {config['vocab_size']:,}")
    print(f"character_coverage: {config['character_coverage']}")
    print(f"byte_fallback    : {config['byte_fallback']}")

    start = time.perf_counter()

    spm.SentencePieceTrainer.train(
        input=str(training_text),
        model_prefix=str(model_prefix),
        model_type=config["model_type"],
        vocab_size=config["vocab_size"],
        character_coverage=config["character_coverage"],

        # Reserved ids. A decoder-only LM needs at minimum a BOS to start
        # generation, an EOS to terminate it, and a PAD for batching.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,

        # byte_fallback expresses any character the vocabulary cannot cover
        # as UTF-8 bytes, so the model can never emit <unk>. Reported
        # separately from the true unknown rate during evaluation.
        byte_fallback=config["byte_fallback"],

        # Lines longer than this are dropped rather than truncated, so it
        # must exceed the longest line the export can produce. See the
        # note on MAX_LINE_CHARS.
        max_sentence_length=config.get(
            "max_sentence_length", MAX_SENTENCE_BYTES
        ),

        # Sample across the whole corpus instead of reading only the first
        # N lines, which would bias the vocabulary toward one source.
        input_sentence_size=config["input_sentence_size"],
        shuffle_input_sentence=True,

        # Digits are split individually. Telugu textbooks mix Telugu and
        # ASCII numerals, and a shared per-digit representation stops the
        # vocabulary from filling with one-off numbers.
        split_digits=True,

        # Keep whitespace recoverable so detokenisation is lossless.
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,

        num_threads=config.get("num_threads", 4),
        train_extremely_large_corpus=config.get(
            "train_extremely_large_corpus", False
        ),
    )

    elapsed = time.perf_counter() - start

    print()
    print(f"Training time: {elapsed / 60:.2f} minutes")
    print(f"Model : {model_prefix}.model")
    print(f"Vocab : {model_prefix}.vocab")

    # -------------------------------------------------------------------------
    # RECORD WHAT WAS RUN
    # -------------------------------------------------------------------------
    #
    # The assignment requires runs to be reproducible from the README, so
    # the exact configuration and the corpus it saw are saved next to the
    # model rather than living only in a shell history.
    # -------------------------------------------------------------------------

    metadata = {
        "config": config,
        "training_text": str(training_text),
        "export_stats": stats,
        "total_lines": total_lines,
        "total_characters": total_chars,
        "training_seconds": round(elapsed, 2),
    }

    metadata_file = output_dir / f"{config['model_prefix']}_metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata: {metadata_file}")

    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for Model H (Telugu)."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a tokenizer configuration JSON file.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    print()
    print("=" * 78)
    print(f"TOKENIZER CONFIG: {args.config}")
    print("=" * 78)

    for key, value in config.items():
        print(f"  {key:<26} {value}")

    train(config)


if __name__ == "__main__":
    main()
