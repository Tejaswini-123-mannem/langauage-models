"""
Vocabulary-size sweep for Model H (Telugu).

Phase 1 requires the vocabulary size to be chosen on fertility and
unknown-token rate measured on held-out text, rather than picked by
convention. This script trains one candidate tokenizer per vocabulary
size so that comparison can be made from evidence.

Why the candidates are trained on a sample
------------------------------------------
Training five tokenizers on the full ~2.2 GB training split would take
several hours and would not change the ranking. What the sweep measures
is how compression and vocabulary utilisation respond to vocabulary
size, and that relationship is already stable on a 10% sample: the
candidates all see the same text, so the comparison between them is
fair.

The chosen size is then retrained on the FULL training split for the
delivered tokenizer, because the delivered model should learn its
merges from all the data available.

What the sweep is trading off
-----------------------------
Larger vocabularies compress better, with diminishing returns, but cost
vocab_size x d_model parameters in the embedding matrix. Against Phase
2's ~25M parameter target that cost is not marginal: at d_model=384, a
64k vocabulary spends 24.6M parameters before any Transformer block
exists. The evaluation step reports embedding cost next to fertility so
the two are weighed together.

Usage
-----
    python run_vocab_sweep.py
    python run_vocab_sweep.py --vocab-sizes 8000 16000 32000
    python run_vocab_sweep.py --sample-rate 0.05
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# Splits and candidates live on local disk, not Drive. A 5.78 GB
# train.jsonl written into a paused Drive folder was evicted from Drive's
# local cache mid-sweep, and this script failed on its second candidate
# with "Missing corpus files". Reading and writing local removes that
# class of failure entirely, and is faster besides. publish_deliverables.py
# copies what is graded to Drive afterwards.
#
# Candidate vocabulary sizes come from the language: see
# LanguageConfig.vocab_sweep_sizes for why the bracket differs between a
# 522M-token corpus and a corpus a tenth that size.
# =============================================================================

DEFAULT_SAMPLE_RATE = 0.10

SEED = 42

# -----------------------------------------------------------------------------
# SENTENCE LENGTH LIMIT
# -----------------------------------------------------------------------------
#
# SentencePiece's max_sentence_length is measured in BYTES and defaults to
# 4,192. Lines longer than that are dropped SILENTLY: no warning, no
# error, just missing training text.
#
# Both languages here cost three bytes per character, so the exporter's
# 2,000-character chunks reach ~6,000 bytes and exceed the default. On the
# Telugu corpus that was 0.72% of lines but 7.15% of the text, weighted
# towards the longest continuous prose, which is exactly the material
# subword learning depends on.
#
# The Telugu sweep already reported in report/phase1/telugu ran before
# this was found, so its candidates were trained on the truncated export.
# The full-scale 32k tokenizer, trained with the limit raised, landed
# within 0.3% of its sweep figure (3.766 -> 3.776 chars/token), which
# bounds the effect on the ranking the sweep was used for.
# -----------------------------------------------------------------------------

MAX_SENTENCE_LENGTH = 16384

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_tokenizer.py"


# =============================================================================
# CANDIDATE TRAINING
# =============================================================================

def train_candidate(config):
    """
    Train one candidate in its OWN process.

    Candidates were originally trained in a single process, in a loop.
    That failed: the 48k candidate died with std::bad_alloc while the
    larger 64k candidate, which ran immediately afterwards, succeeded.

    The cause was ordering, not size. SentencePiece allocates a large
    suffix array over the training text for each candidate, and memory
    released by the C++ allocator is not necessarily returned to the
    operating system. Across successive candidates the heap fragmented
    until no sufficiently large contiguous block remained. When 48k
    failed its allocations were freed, which is precisely why 64k then
    had room.

    Running each candidate in a separate process makes the OS reclaim
    everything on exit. Every candidate then starts from an identical
    clean state, so results no longer depend on the order candidates
    happen to be trained in - which they never should have.
    """

    handle, config_path = tempfile.mkstemp(suffix=".json", text=True)

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--config", config_path],
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"training exited with code {result.returncode}"
            )

    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train candidate tokenizers at several vocabulary sizes."
    )

    add_language_argument(parser)

    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Defaults to the language's vocab_sweep_sizes.",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="Fraction of training documents used for the candidates.",
    )

    parser.add_argument(
        "--model-type",
        default="unigram",
        choices=["unigram", "bpe"],
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain candidates whose model file already exists. Required "
            "after the training split changes, since the existing models "
            "were fitted to different text and are not comparable with "
            "newly trained ones."
        ),
    )

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    vocab_sizes = args.vocab_sizes or list(language.vocab_sweep_sizes)

    train_file = workspace.require(
        workspace.split_file("train"),
        f"Run: python make_splits.py --language {workspace.slug}",
    )

    sweep_dir = workspace.tokenizer_sweep

    sweep_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"{workspace.title} - VOCABULARY SIZE SWEEP")
    print("=" * 78)
    print(f"Training split : {train_file}")
    print(f"Sample rate    : {args.sample_rate:.0%}")
    print(f"Model type     : {args.model_type}")
    print(f"Vocab sizes    : {vocab_sizes}")
    print(f"Output         : {sweep_dir}")

    results = []

    overall_start = time.perf_counter()

    for vocab_size in vocab_sizes:

        prefix = f"{workspace.slug}_sweep_{vocab_size}"

        model_path = sweep_dir / f"{prefix}.model"

        # ------------------------------------------------------------------
        # Skip work already done, so an interrupted sweep can be rerun
        # without retraining candidates that finished.
        #
        # This is safe ONLY while the training split is unchanged. When the
        # corpus is rebuilt the existing models were trained on different
        # text, and skipping them produces a sweep whose candidates are not
        # comparable -- the one thing a sweep exists to do.
        #
        # That happened here: an Assamese rerun after the corpus grew from
        # 289M to 518M tokens silently kept an 8k model trained on the old
        # corpus and compared it against 6k, 10k and 12k trained on the new
        # one. Nothing in the output said so; only the file timestamps did.
        #
        # Hence --force, and hence the timestamps printed below.
        # ------------------------------------------------------------------

        if model_path.exists() and not args.force:
            age = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(model_path.stat().st_mtime),
            )
            print()
            print(
                f"SKIPPING vocab {vocab_size:,}: model already exists "
                f"(trained {age})"
            )
            print(
                "         Pass --force if the training split has changed "
                "since then."
            )
            results.append(str(model_path))
            continue

        config = {
            "model_prefix": prefix,
            "output_dir": str(sweep_dir),
            "input_files": [str(train_file)],

            # All candidates train on identical text, so they share one
            # export. The first candidate writes it; the rest reuse it,
            # which avoids re-reading the multi-gigabyte training split
            # once per vocabulary size.
            "training_text": str(sweep_dir / "sweep_input.txt"),

            "sample_rate": args.sample_rate,
            "seed": SEED,
            "model_type": args.model_type,
            "vocab_size": vocab_size,
            "character_coverage": 0.9995,
            "byte_fallback": True,
            "max_sentence_length": MAX_SENTENCE_LENGTH,
            "input_sentence_size": 3_000_000,
            "num_threads": 4,
            "train_extremely_large_corpus": False,
        }

        print()
        print("#" * 78)
        print(f"CANDIDATE: vocab_size = {vocab_size:,}")
        print("#" * 78)

        try:
            train_candidate(config)

            if not model_path.exists():
                raise RuntimeError("training reported success but wrote no model")

            results.append(str(model_path))

        except Exception as exc:
            print(f"FAILED at vocab {vocab_size:,}: {exc}")

    elapsed = time.perf_counter() - overall_start

    print()
    print("=" * 78)
    print("SWEEP COMPLETE")
    print("=" * 78)
    print(f"Candidates trained: {len(results)}")
    print(f"Total time        : {elapsed / 60:.2f} minutes")

    manifest = sweep_dir / "sweep_models.json"

    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language.name,
                "sample_rate": args.sample_rate,
                "max_sentence_length": MAX_SENTENCE_LENGTH,
                "model_type": args.model_type,
                "seed": SEED,
                "models": results,
            },
            f,
            indent=2,
        )

    print(f"Manifest          : {manifest}")

    print()
    print("NEXT: compare the candidates on held-out text with")
    print(f"  python evaluate_tokenizer.py --language {workspace.slug} \\")
    print("      --models \\")

    for path in results:
        print(f"      \"{path}\" \\")

    print("=" * 78)


if __name__ == "__main__":
    main()
