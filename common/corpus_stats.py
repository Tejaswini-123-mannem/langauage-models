"""
Dataset statistics report, for either language.

Phase 1 requires a per-language statistics report covering corpus size,
sources, cleaning steps and the manual fraction. This script produces the
numbers and the figures; the written narrative lives in
report/phase1/<language>/ alongside them and cites what this writes.

The number that matters most here is the MANUAL TOKEN SHARE. Everything
upstream reasons in characters because that is what is cheap to measure,
but the assignment's >=20% rule is stated in tokens, and characters are a
poor proxy for it: OCR'd textbook text and clean Sangraha web text do
not tokenize at the same rate. This script settles the question by
tokenizing the actual splits with the actual delivered tokenizer.

Exact counts, not estimates
---------------------------
plan_mix.py extrapolated token counts from a sample in order to decide the
mix. That was the right trade-off for a planning decision. It is not good
enough for a graded figure, so every document in every split is tokenized
here. It is slower, and it is the number that goes in the report.

Usage
-----
    python corpus_stats.py --language telugu \
        --tokenizer <path to final .model>
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# Paths come from workspace.py. Splits live on local disk and are copied
# to Drive as deliverables only after syncing is running; see the note in
# make_splits.py.
# =============================================================================

SPLITS = ["train", "validation", "test"]

MIN_MANUAL_SHARE = 0.20


# =============================================================================
# COMPUTATION
# =============================================================================

def measure_split(sp, path, sample_rate=1.0):
    """
    Tokenize one split and accumulate per-source statistics.

    Document lengths are collected so the report can show the length
    distribution, which is what explains why a handful of Internet Archive
    books carry as many characters as thousands of textbook pages.
    """

    per_source = defaultdict(
        lambda: {
            "source_group": "",
            "documents": 0,
            "characters": 0,
            "non_space_characters": 0,
            "tokens": 0,
            "lengths": [],
        }
    )

    processed = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            source_type = record.get("source_type", "unknown")

            entry = per_source[source_type]

            entry["source_group"] = record.get("source_group", "unknown")

            token_count = len(sp.encode(text, out_type=int))

            entry["documents"] += 1
            entry["characters"] += len(text)
            entry["non_space_characters"] += sum(
                1 for c in text if not c.isspace()
            )
            entry["tokens"] += token_count
            entry["lengths"].append(token_count)

            processed += 1

            if processed % 50_000 == 0:
                print(f"    tokenized {processed:,} documents")

    return per_source


def summarise(per_source):
    """Collapse per-source statistics into split-level totals."""

    documents = sum(e["documents"] for e in per_source.values())
    characters = sum(e["characters"] for e in per_source.values())
    tokens = sum(e["tokens"] for e in per_source.values())

    manual_tokens = sum(
        e["tokens"] for e in per_source.values()
        if e["source_group"] == "manual"
    )

    return {
        "documents": documents,
        "characters": characters,
        "tokens": tokens,
        "manual_tokens": manual_tokens,
        "sangraha_tokens": tokens - manual_tokens,
        "manual_token_share": manual_tokens / tokens if tokens else 0.0,
        "chars_per_token": characters / tokens if tokens else 0.0,
    }


def percentile(values, fraction):
    """Return the value at a given fraction through a sorted list."""

    if not values:
        return 0

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        max(0, int(fraction * len(ordered))),
    )

    return ordered[index]


# =============================================================================
# VISUALISATION
# =============================================================================
#
# Separate from computation, per the assignment's code-design guidelines.
# =============================================================================

def plot_source_composition(train_stats, output_file, title_prefix):
    """Plot how many training tokens each source contributes."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted(
        train_stats,
        key=lambda s: train_stats[s]["tokens"],
        reverse=True,
    )

    tokens = [train_stats[s]["tokens"] / 1e6 for s in sources]

    colours = [
        "#C2472A" if train_stats[s]["source_group"] == "manual"
        else "#2A6FB5"
        for s in sources
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))

    bars = axis.bar(sources, tokens, color=colours)

    axis.set_xlabel("Corpus source")
    axis.set_ylabel("Training tokens (millions)")
    axis.set_title(
        f"{title_prefix}: training tokens by source\n"
        "red = manually collected, blue = downloaded public corpus"
    )
    axis.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, tokens):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.1f}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C2472A"),
        plt.Rectangle((0, 0), 1, 1, color="#2A6FB5"),
    ]

    axis.legend(handles, ["Manual collection", "Sangraha (downloaded)"])

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def plot_document_lengths(train_stats, output_file, title_prefix):
    """Plot the document length distribution of each source."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(9, 5.5))

    for source in sorted(train_stats):

        lengths = [
            length for length in train_stats[source]["lengths"]
            if length > 0
        ]

        if not lengths:
            continue

        axis.hist(
            np.log10(lengths),
            bins=60,
            alpha=0.55,
            label=source,
        )

    axis.set_xlabel("Document length in tokens (log base 10)")
    axis.set_ylabel("Number of documents")
    axis.set_title(
        f"{title_prefix}: document length distribution by source\n"
        "(training split)"
    )
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def plot_split_sizes(summaries, output_file, title_prefix):
    """Plot manual against downloaded tokens for each split."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    splits = [s for s in SPLITS if s in summaries]

    manual = [summaries[s]["manual_tokens"] / 1e6 for s in splits]
    sangraha = [summaries[s]["sangraha_tokens"] / 1e6 for s in splits]

    positions = np.arange(len(splits))

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        positions, manual,
        color="#C2472A", label="Manual collection",
    )

    axis.bar(
        positions, sangraha, bottom=manual,
        color="#2A6FB5", label="Sangraha (downloaded)",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(splits)
    axis.set_xlabel("Split")
    axis.set_ylabel("Tokens (millions)")
    axis.set_title(
        f"{title_prefix}: split sizes and manual share\n"
        "manual collection must be at least 20% of training tokens"
    )
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_yscale("log")

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Dataset statistics report for one language."
    )

    add_language_argument(parser)

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the FINAL SentencePiece .model file.",
    )

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    report_dir = workspace.report
    figure_dir = workspace.figures

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    print()
    print("=" * 78)
    title_prefix = f"{language.model_tag} ({language.name})"

    print(f"{workspace.title} - DATASET STATISTICS")
    print("=" * 78)
    print(f"Tokenizer  : {args.tokenizer}")
    print(f"Vocabulary : {sp.get_piece_size():,}")

    all_stats = {}
    summaries = {}

    for split in SPLITS:

        path = workspace.split_file(split)

        if not path.exists():
            print(f"WARNING: missing split {path.name}, skipped.")
            continue

        print()
        print(f"  Tokenizing {split}...")

        per_source = measure_split(sp, path)

        all_stats[split] = per_source
        summaries[split] = summarise(per_source)

    # -------------------------------------------------------------------------
    # PER-SOURCE TABLE
    # -------------------------------------------------------------------------

    rows = []

    for split in SPLITS:

        if split not in all_stats:
            continue

        for source_type, entry in sorted(all_stats[split].items()):

            rows.append({
                "split": split,
                "source_type": source_type,
                "source_group": entry["source_group"],
                "documents": entry["documents"],
                "characters": entry["characters"],
                "tokens": entry["tokens"],
                "chars_per_token": round(
                    entry["non_space_characters"] / entry["tokens"], 4
                ) if entry["tokens"] else 0.0,
                "median_doc_tokens": percentile(entry["lengths"], 0.50),
                "p95_doc_tokens": percentile(entry["lengths"], 0.95),
                "max_doc_tokens": max(entry["lengths"]) if entry["lengths"] else 0,
            })

    stats_file = report_dir / "dataset_statistics.csv"

    with open(stats_file, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "source_type", "source_group",
                "documents", "characters", "tokens", "chars_per_token",
                "median_doc_tokens", "p95_doc_tokens", "max_doc_tokens",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # PRINT
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("PER-SOURCE STATISTICS")
    print("=" * 78)

    header = (
        f"{'split':<11} {'source':<12} {'group':<9} {'docs':>10} "
        f"{'characters':>15} {'tokens':>14} {'c/tok':>7}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['split']:<11} {row['source_type']:<12} "
            f"{row['source_group']:<9} {row['documents']:>10,} "
            f"{row['characters']:>15,} {row['tokens']:>14,} "
            f"{row['chars_per_token']:>7.2f}"
        )

    print()
    print("=" * 78)
    print("SPLIT TOTALS")
    print("=" * 78)

    header = (
        f"{'split':<12} {'documents':>12} {'characters':>16} "
        f"{'tokens':>15} {'manual share':>14}"
    )

    print(header)
    print("-" * len(header))

    for split in SPLITS:

        if split not in summaries:
            continue

        entry = summaries[split]

        print(
            f"{split:<12} {entry['documents']:>12,} "
            f"{entry['characters']:>16,} {entry['tokens']:>15,} "
            f"{entry['manual_token_share']:>13.2%}"
        )

    # -------------------------------------------------------------------------
    # THE GRADED NUMBERS
    # -------------------------------------------------------------------------

    train = summaries.get("train", {})

    train_tokens = train.get("tokens", 0)
    manual_share = train.get("manual_token_share", 0.0)

    print()
    print("=" * 78)
    print("PHASE 1 REQUIREMENTS")
    print("=" * 78)

    print(f"Training tokens        : {train_tokens:,}")
    print(f"Manual training tokens : {train.get('manual_tokens', 0):,}")
    print(f"Manual token share     : {manual_share:.2%}")

    print()

    if manual_share >= MIN_MANUAL_SHARE:
        print(
            f"PASS  manual share {manual_share:.2%} meets the "
            f">= {MIN_MANUAL_SHARE:.0%} requirement."
        )
    else:
        print(
            f"FAIL  manual share {manual_share:.2%} is below "
            f"{MIN_MANUAL_SHARE:.0%}."
        )

    if train_tokens >= 450_000_000:
        print("PASS  training tokens are close to the ~500M target.")
    else:
        print(
            f"SHORT training tokens are {train_tokens / 1e6:,.0f}M against "
            f"the ~500M target."
        )
        print(
            "      The assignment permits this if the exact count is "
            "reported and the shortfall justified."
        )

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    if "train" in all_stats:

        composition_file = figure_dir / "corpus_source_composition.png"
        plot_source_composition(
            all_stats["train"], composition_file, title_prefix
        )

        lengths_file = figure_dir / "corpus_document_lengths.png"
        plot_document_lengths(
            all_stats["train"], lengths_file, title_prefix
        )

        print()
        print(f"Figure: {composition_file}")
        print(f"Figure: {lengths_file}")

    if summaries:

        splits_file = figure_dir / "corpus_split_sizes.png"
        plot_split_sizes(summaries, splits_file, title_prefix)
        print(f"Figure: {splits_file}")

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    summary_file = report_dir / "dataset_statistics.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language.name,
                "tokenizer": args.tokenizer,
                "vocab_size": sp.get_piece_size(),
                "splits": summaries,
                "per_source": rows,
                "requirements": {
                    "min_manual_share": MIN_MANUAL_SHARE,
                    "manual_share": manual_share,
                    "meets_manual_requirement": (
                        manual_share >= MIN_MANUAL_SHARE
                    ),
                    "training_tokens": train_tokens,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Statistics: {stats_file}")
    print(f"Summary   : {summary_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
