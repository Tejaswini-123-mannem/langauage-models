"""
Stage 3 of corpus assembly, for either language: subsample and split.

Reads the mix plan produced by plan_mix.py, trims Sangraha to the planned
fraction, and writes train / validation / test splits.

Two decisions drive this script.

1. Splits are made at DOCUMENT level, never at chunk level
--------------------------------------------------------
Phase 2 concatenates documents and cuts them into fixed-length sequences.
If splitting happened after that, two chunks of the same document could
land in train and test, and the model would be evaluated on text it had
already memorised. Validation perplexity would look far better than the
model deserves. Splitting whole documents makes held-out text genuinely
held out.

2. Assignment is by hash, not by shuffling
------------------------------------------
Each document's split is decided by hashing its id together with a seed.
This is deterministic, reproducible from the seed alone, and streams in
constant memory, which matters because the Sangraha file is several
gigabytes. It also means the split of any single document can be verified
later without replaying the whole run.

Because each source is streamed independently and the same fractions are
applied to each, the splits are stratified by source automatically: train,
validation and test all carry the same source composition. That is what
makes validation perplexity comparable to training loss.

Output
------
    corpus/splits/train.jsonl
    corpus/splits/validation.jsonl
    corpus/splits/test.jsonl
    corpus/splits/split_statistics.csv
    corpus/splits/split_summary.json

Usage
-----
    python make_splits.py --language telugu
    python make_splits.py --language assamese --val-fraction 0.01 \\
        --test-fraction 0.01

Paths come from workspace.py, and the list of sources from the language's
corpus_sources. The splits ARE a Phase 1 deliverable and have to reach
Google Drive with shareable links in the README, but they are written to
LOCAL DISK first and copied afterwards by publish_deliverables.py: a
5.78 GB train.jsonl written straight into a paused Drive folder was
evicted from Drive's local cache mid-run, and the tokenizer sweep failed
on its second candidate with "Missing corpus files".
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# =============================================================================
# DETERMINISTIC HASHING
# =============================================================================

def uniform_hash(seed, namespace, doc_id):
    """
    Map a document id to a uniform value in [0, 1).

    The namespace keeps independent decisions independent: the value used
    to decide whether a Sangraha document is kept must not correlate with
    the value used to decide which split it lands in. Reusing one hash for
    both would make the kept subset and the split assignment dependent,
    which would quietly bias the splits.
    """

    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")

    digest = hashlib.sha1(payload).digest()

    # 8 bytes of the digest as an unsigned integer, scaled to [0, 1).
    value = int.from_bytes(digest[:8], "big")

    return value / float(1 << 64)


def choose_split(seed, doc_id, val_fraction, test_fraction):
    """
    Assign one document to train, validation or test.

    The fractions are passed in rather than read from a constant because
    they vary by source: a source whose documents are whole books needs a
    larger held-out share to put a usable number of them in validation.
    See LanguageConfig.heldout_fractions.

    The hash is unchanged either way, so a document's position in the
    unit interval is fixed and only the thresholds move. Raising a
    source's held-out fraction therefore ADDS documents to validation and
    test rather than reshuffling the ones already there.
    """

    h = uniform_hash(seed, "split", doc_id)

    if h < val_fraction:
        return "validation"

    if h < val_fraction + test_fraction:
        return "test"

    return "train"


# =============================================================================
# PROCESSING
# =============================================================================

def process_source(
    path,
    source_type,
    source_group,
    keep_fraction,
    seed,
    val_fraction,
    test_fraction,
    writers,
    stats,
):
    """
    Stream one source file, apply subsampling, and route each document to
    its split.
    """

    if not path.exists():
        print(f"  WARNING: missing {path.name}, source skipped.")
        return

    print()
    print(f"  {source_type} (keep fraction {keep_fraction:.4f})")

    seen = 0
    dropped = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            seen += 1

            doc_id = record["doc_id"]

            # -----------------------------------------------------------------
            # SUBSAMPLING
            # -----------------------------------------------------------------
            # Keep fractions come from the mix plan. Sangraha is trimmed to
            # hit the target size and the 20% manual floor; a manual source
            # is trimmed only if manual collection over-delivered against
            # the manual target. Sources kept whole have a fraction of 1.0,
            # so this branch is a no-op for them.
            # -----------------------------------------------------------------

            if keep_fraction < 1.0:

                if uniform_hash(seed, "keep", doc_id) >= keep_fraction:
                    dropped += 1
                    continue

            split = choose_split(seed, doc_id, val_fraction, test_fraction)

            writers[split].write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            entry = stats[(split, source_type)]

            entry["source_group"] = source_group
            entry["documents"] += 1
            entry["characters"] += record.get(
                "char_count", len(record.get("text", ""))
            )

            if seen % 100_000 == 0:
                print(f"    read {seen:,} documents, dropped {dropped:,}")

    print(f"    total read {seen:,}, dropped {dropped:,}, kept {seen - dropped:,}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build train / validation / test splits for one language."
    )

    add_language_argument(parser)

    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    sources = language.corpus_sources

    mix_plan_file = workspace.require(
        workspace.mix_plan,
        f"Run: python plan_mix.py --language {workspace.slug} "
        f"--tokenizer <provisional .model>",
    )

    with open(mix_plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    decision = plan["decision"]

    # Older mix plans recorded only the Sangraha fraction. Fall back to
    # that shape so a plan written before per-source trimming still works.
    keep_fractions = decision.get("keep_fractions")

    if not keep_fractions:
        keep_fractions = {
            "sangraha": decision["sangraha_keep_fraction"]
        }

    split_dir = workspace.splits

    split_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"{workspace.title} - TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 78)
    print(f"Mix plan            : {mix_plan_file}")
    print(f"Validation fraction : {args.val_fraction:.2%}")
    print(f"Test fraction       : {args.test_fraction:.2%}")
    print(f"Seed                : {args.seed}")
    print(f"Output              : {split_dir}")
    print()
    print("Keep fractions from the mix plan:")

    for source_type, _ in sources:
        print(
            f"  {source_type:<12} "
            f"{keep_fractions.get(source_type, 1.0):>8.4f}"
        )

    stats = defaultdict(
        lambda: {"source_group": "", "documents": 0, "characters": 0}
    )

    writers = {}

    try:
        for split in ["train", "validation", "test"]:
            writers[split] = open(
                workspace.split_file(split), "w", encoding="utf-8"
            )

        for source_type, source_group in sources:

            keep_fraction = keep_fractions.get(source_type, 1.0)

            # A source whose documents are whole books contributes very few
            # of them at the default fraction, however large it is in
            # characters. See LanguageConfig.heldout_fractions.
            val_fraction, test_fraction = language.heldout_fractions.get(
                source_type, (args.val_fraction, args.test_fraction)
            )

            if (val_fraction, test_fraction) != (
                args.val_fraction, args.test_fraction
            ):
                print(
                    f"  {source_type}: held-out override "
                    f"{val_fraction:.1%} / {test_fraction:.1%} "
                    f"(default {args.val_fraction:.1%} / "
                    f"{args.test_fraction:.1%})"
                )

            process_source(
                workspace.source_file(source_type),
                source_type,
                source_group,
                keep_fraction,
                args.seed,
                val_fraction,
                test_fraction,
                writers,
                stats,
            )

    finally:
        for handle in writers.values():
            handle.close()

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    rows = []

    for (split, source_type), entry in sorted(stats.items()):
        rows.append({
            "split": split,
            "source_type": source_type,
            "source_group": entry["source_group"],
            "documents": entry["documents"],
            "characters": entry["characters"],
        })

    stats_file = split_dir / "split_statistics.csv"

    with open(stats_file, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "source_type", "source_group",
                "documents", "characters",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SPLITS COMPLETE")
    print("=" * 78)

    print(
        f"{'split':<12} {'source':<12} {'documents':>12} {'characters':>16}"
    )
    print("-" * 56)

    for row in rows:
        print(
            f"{row['split']:<12} {row['source_type']:<12} "
            f"{row['documents']:>12,} {row['characters']:>16,}"
        )

    summary = {}

    for split in ["train", "validation", "test"]:

        split_rows = [r for r in rows if r["split"] == split]

        documents = sum(r["documents"] for r in split_rows)
        characters = sum(r["characters"] for r in split_rows)

        manual_chars = sum(
            r["characters"] for r in split_rows
            if r["source_group"] == "manual"
        )

        summary[split] = {
            "documents": documents,
            "characters": characters,
            "manual_characters": manual_chars,
            "manual_char_share": (
                manual_chars / characters if characters else 0.0
            ),
        }

    print()
    print(
        f"{'split':<12} {'documents':>12} {'characters':>16} "
        f"{'manual share':>14}"
    )
    print("-" * 58)

    for split, entry in summary.items():
        print(
            f"{split:<12} {entry['documents']:>12,} "
            f"{entry['characters']:>16,} "
            f"{entry['manual_char_share']:>13.2%}"
        )

    summary_file = split_dir / "split_summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language.name,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "keep_fractions": keep_fractions,
                "splits": summary,
                "per_source": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Statistics: {stats_file}")
    print(f"Summary   : {summary_file}")
    print()
    print(
        "NOTE: shares above are character shares. Exact TOKEN counts and "
        "the graded manual token share are computed by corpus_stats.py "
        "once the final tokenizer has been trained on train.jsonl."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
