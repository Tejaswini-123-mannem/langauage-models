"""
Stage 1 of corpus assembly for Model H (Telugu).

Takes the output of the global near-duplicate filter and merges the four
sources into a single, uniformly-shaped corpus in which every document
carries an explicit source label.

Why this stage exists
---------------------
The four sources arrive in different record shapes:

    textbook    one record per OCR'd page   (fields: book, page, text, ...)
    wikisource  one record per wiki page    (fields: title, text, ...)
    archive     one record per book         (fields: doc_id, book, text, ...)
    sangraha    one record per web/pdf doc  (fields: doc_id, text, type)

Everything downstream (the source mix, the splits, the tokenizer, the
statistics report) needs to know which source each document came from and
how many characters it contributes. Normalising that here means no later
stage has to special-case a source.

Nothing is filtered or rewritten at this stage. Cleaning, quality
filtering, exact deduplication and near-deduplication have already run.
This is purely a merge-and-label pass.

Output
------
    corpus/all/corpus_<source_type>.jsonl   one file per source type
    corpus/all/corpus_manifest.csv          per-source record and char counts

Each output record has exactly these fields:

    doc_id         stable identifier, unique across the whole corpus
    text           the document text, unchanged
    source_group   "manual" or "sangraha"   (drives the >=20% manual rule)
    source_type    "textbook" | "wikisource" | "archive" | "sangraha"
    char_count     len(text)
    origin         where it came from (book name / title / sangraha type)
"""

import csv
import hashlib
import json
import sys
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

# Intermediate corpus data lives on local disk. Only the final splits and
# the tokenizer are written to Drive, because only those are deliverables.
WORK = Path(r"C:\lma_work")

NEAR_DEDUP_DIR = WORK / "global_near_dedup"

MANUAL_INPUT = NEAR_DEDUP_DIR / "manual"

SANGRAHA_INPUT = NEAR_DEDUP_DIR / "sangraha"

OUTPUT_DIR = WORK / "corpus" / "all"

MANIFEST_FILE = OUTPUT_DIR / "corpus_manifest.csv"


# =============================================================================
# SOURCE DEFINITIONS
# =============================================================================
#
# source_type -> (source_group, input directory)
#
# The group is what the assignment's ">=20% manual" rule is computed over.
# Sangraha is the only downloaded public corpus; the other three were
# collected by hand (OCR, dump parsing, Archive harvesting).
# =============================================================================

SOURCES = [
    ("textbook",   "manual",   MANUAL_INPUT / "textbook"),
    ("wikisource", "manual",   MANUAL_INPUT / "wikisource"),
    ("wikipedia",  "manual",   MANUAL_INPUT / "wikipedia"),
    ("archive",    "manual",   MANUAL_INPUT / "archive"),
    ("sangraha",   "sangraha", SANGRAHA_INPUT),
]


# =============================================================================
# READING
# =============================================================================

def read_jsonl(path):
    """
    Yield parsed records from a JSONL file, skipping unreadable lines.

    A single malformed line must not abort a merge over thousands of
    files, so parse failures are counted by the caller rather than raised.
    """

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError:
                yield None


def discover_jsonl_files(directory):
    """Return every .jsonl file under a directory, sorted for determinism."""

    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.rglob("*.jsonl")
        if p.is_file()
    )


# =============================================================================
# FIELD EXTRACTION
# =============================================================================

def get_origin(record, source_type):
    """
    Describe where a document came from, in a form that is meaningful
    for that source.

    This is kept for the statistics report and for error analysis later:
    when a weird document turns up, we want to know which book or which
    Sangraha subset produced it.
    """

    if source_type == "textbook":

        book = record.get("book") or "unknown_book"
        page = record.get("page")

        return f"{book}#page{page}" if page else book

    if source_type in ("wikisource", "wikipedia"):
        return record.get("title") or "unknown_title"

    if source_type == "archive":
        return record.get("book") or record.get("doc_id") or "unknown_item"

    # Sangraha labels each document web / pdf / speech.
    return record.get("type") or "unknown_type"


def make_doc_id(source_type, text, fallback_index):
    """
    Build an identifier that is unique across the whole corpus.

    The source types use different and sometimes absent id schemes, so a
    content hash prefixed with the source type is used instead. Prefixing
    guarantees that two identical strings from two different sources stay
    distinguishable, which matters because the manual sources are
    deliberately never deduplicated against each other.
    """

    if text:
        digest = hashlib.sha1(
            text.encode("utf-8")
        ).hexdigest()[:20]

    else:
        digest = f"empty{fallback_index}"

    return f"{source_type}_{digest}"


# =============================================================================
# PER-SOURCE MERGE
# =============================================================================

def build_source(source_type, source_group, input_dir, output_dir):
    """
    Merge every JSONL file of one source into a single labelled file.

    Returns a statistics dictionary for the manifest.
    """

    files = discover_jsonl_files(input_dir)

    print()
    print("=" * 78)
    print(f"SOURCE: {source_type.upper()}  ({source_group})")
    print("=" * 78)
    print(f"Input : {input_dir}")
    print(f"Files : {len(files):,}")

    if not files:
        print("WARNING: no input files found; source skipped.")
        return None

    output_file = output_dir / f"corpus_{source_type}.jsonl"

    records = 0
    characters = 0
    skipped_invalid = 0
    skipped_empty = 0

    seen_ids = set()
    duplicate_ids = 0

    with open(output_file, "w", encoding="utf-8") as fout:

        for file_index, input_file in enumerate(files, start=1):

            for record in read_jsonl(input_file):

                if record is None:
                    skipped_invalid += 1
                    continue

                text = record.get("text")

                if not isinstance(text, str) or not text.strip():
                    skipped_empty += 1
                    continue

                doc_id = make_doc_id(source_type, text, records)

                # Identical text within one source can survive the earlier
                # stages (near-dedup keeps documents under 20 words). Count
                # them so the number is visible in the report rather than
                # silently changing the corpus.
                if doc_id in seen_ids:
                    duplicate_ids += 1

                seen_ids.add(doc_id)

                output_record = {
                    "doc_id": doc_id,
                    "text": text,
                    "source_group": source_group,
                    "source_type": source_type,
                    "char_count": len(text),
                    "origin": get_origin(record, source_type),
                }

                fout.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )

                records += 1
                characters += len(text)

            if file_index % 100 == 0 or file_index == len(files):
                print(
                    f"  [{file_index:5d}/{len(files):5d}] "
                    f"records={records:,} chars={characters:,}"
                )

    print(f"Output: {output_file}")
    print(f"  records          : {records:,}")
    print(f"  characters       : {characters:,}")
    print(f"  skipped invalid  : {skipped_invalid:,}")
    print(f"  skipped empty    : {skipped_empty:,}")
    print(f"  repeated doc_ids : {duplicate_ids:,}")

    return {
        "source_type": source_type,
        "source_group": source_group,
        "files": len(files),
        "records": records,
        "characters": characters,
        "skipped_invalid": skipped_invalid,
        "skipped_empty": skipped_empty,
        "repeated_doc_ids": duplicate_ids,
        "output_file": str(output_file),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print("MODEL H (TELUGU) - CORPUS ASSEMBLY")
    print("=" * 78)
    print(f"Input : {NEAR_DEDUP_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if not NEAR_DEDUP_DIR.exists():
        print()
        print("ERROR: near-dedup output not found.")
        print(NEAR_DEDUP_DIR)
        sys.exit(1)

    all_stats = []

    for source_type, source_group, input_dir in SOURCES:

        stats = build_source(
            source_type,
            source_group,
            input_dir,
            OUTPUT_DIR,
        )

        if stats:
            all_stats.append(stats)

    # -------------------------------------------------------------------------
    # MANIFEST
    # -------------------------------------------------------------------------

    fieldnames = [
        "source_type",
        "source_group",
        "files",
        "records",
        "characters",
        "skipped_invalid",
        "skipped_empty",
        "repeated_doc_ids",
        "output_file",
    ]

    with open(MANIFEST_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in all_stats:
            writer.writerow(stats)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    total_records = sum(s["records"] for s in all_stats)
    total_chars = sum(s["characters"] for s in all_stats)

    manual_records = sum(
        s["records"] for s in all_stats
        if s["source_group"] == "manual"
    )

    manual_chars = sum(
        s["characters"] for s in all_stats
        if s["source_group"] == "manual"
    )

    print()
    print("=" * 78)
    print("CORPUS ASSEMBLY COMPLETE")
    print("=" * 78)

    print(f"{'source':<12} {'records':>12} {'characters':>16} {'share':>8}")
    print("-" * 52)

    for stats in all_stats:

        share = (
            stats["characters"] / total_chars
            if total_chars else 0
        )

        print(
            f"{stats['source_type']:<12} "
            f"{stats['records']:>12,} "
            f"{stats['characters']:>16,} "
            f"{share:>7.2%}"
        )

    print("-" * 52)
    print(f"{'TOTAL':<12} {total_records:>12,} {total_chars:>16,}")

    print()
    print(f"Manual records    : {manual_records:,}")
    print(f"Manual characters : {manual_chars:,}")

    if total_chars:
        print(
            f"Manual share      : "
            f"{manual_chars / total_chars:.2%}  "
            f"(assignment requires >= 20% of final training TOKENS)"
        )

    print()
    print(f"Manifest: {MANIFEST_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
