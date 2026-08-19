"""
Clean and exact-deduplicate additional Sangraha shards for Model H (Telugu).

Shards 00-03 were processed inside the Colab notebook
01_Sangraha_Preprocessing_ipynb.ipynb. This script does the same work for
further shards from the command line, so the corpus can be grown without
returning to Colab.

Why the cleaning function is copied verbatim
--------------------------------------------
clean_text and telugu_ratio below are ported unchanged from cells 29-31 of
that notebook. They must stay identical. If shards 04+ were cleaned even
slightly differently from 00-03 the corpus would be internally
inconsistent: the tokenizer would learn two different whitespace and
punctuation conventions from what is supposed to be one source, and the
dataset statistics would describe a corpus that does not exist. Any future
change to the cleaning rules has to be applied to every shard, by
reprocessing all of them.

Why shards are appended rather than regenerated
------------------------------------------------
sangraha_deduplicated.jsonl already holds 681,635 records and
dedup_hashes.txt already holds their 681,635 hashes. The notebook built
these by streaming shards in order against a growing hash set, so
appending shard 04 against the existing hash set produces exactly the file
the notebook would have produced had it looped to 04. Rewriting 3.5 GB on
a network drive to reach the same result is not worth the time.

Crash safety
------------
This pipeline has already been bitten once by a partially-written file
left behind by an interrupted run, which later got mistaken for real
input. So each stage writes to a temporary file and renames it into place
only on success, and completed shards are recorded in a state file. Re-run
the script after an interruption and it resumes without duplicating work.

Usage
-----
    python process_sangraha_shards.py --shards 4 5
    python process_sangraha_shards.py --shards 4 5 --dry-run
"""

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(r"G:\My Drive\LMA_Individual_Project")

ARROW_DIR = (
    PROJECT
    / "hf_cache" / "datasets" / "ai4bharat___sangraha"
    / "default-275b029f4210df27" / "0.0.0"
    / "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
)

PROCESSED_DIR = PROJECT / "data" / "processed" / "sangraha"

DEDUP_FILE = PROCESSED_DIR / "sangraha_deduplicated.jsonl"

HASH_FILE = PROCESSED_DIR / "dedup_hashes.txt"

STATE_FILE = PROCESSED_DIR / "processed_shards.json"


# Filters, matching the notebook exactly.
MIN_CHARACTERS = 50
MIN_TELUGU_RATIO = 0.10


# =============================================================================
# CLEANING  (ported verbatim from notebook 01, cell 29)
# =============================================================================

def clean_text(text):
    """
    Conservative normalisation of one Sangraha document.

    Deliberately light: it fixes encoding and whitespace inconsistencies
    and strips the lecture-slide metadata that Sangraha's speech subset
    carries, but does not attempt to repair or rewrite the language.
    """

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove speech/video timestamp metadata
    text = re.sub(
        r'\(Refer Slide Time:\s*\d+:\d+\)',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # 3. Remove "Slide N" metadata and optional |
    text = re.sub(
        r'\bSlide\s+\d+\s*\|?',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # 4. Normalize spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Normalize excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Remove spaces before common punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    # 7. Strip
    text = text.strip()

    return text


def telugu_ratio(text):
    """
    Telugu letters as a fraction of all alphabetic characters.

    Alphabetic characters are the denominator, not all characters, so that
    digits, punctuation and whitespace cannot drag the ratio down and cause
    a legitimately Telugu document to be discarded.
    """

    total_letters = 0
    telugu_letters = 0

    for ch in text:
        if ch.isalpha():
            total_letters += 1

            if '\u0C00' <= ch <= '\u0C7F':
                telugu_letters += 1

    if total_letters == 0:
        return 0.0

    return telugu_letters / total_letters


# =============================================================================
# STATE
# =============================================================================

def load_state():
    """Return the set of shard indices already fully processed."""

    if not STATE_FILE.exists():
        # Shards 0-3 were completed in the notebook, before this script
        # existed, so they are the implicit starting state.
        return {"cleaned": [0, 1, 2, 3], "deduplicated": [0, 1, 2, 3]}

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    """Write the state file atomically."""

    temp = STATE_FILE.with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    temp.replace(STATE_FILE)


# =============================================================================
# ARROW READING
# =============================================================================

def iter_arrow_records(path):
    """
    Yield {doc_id, text, type} dictionaries from one HuggingFace .arrow shard.

    Read with pyarrow directly rather than the datasets library: the file
    is a plain Arrow IPC stream, and pyarrow reads it in record batches, so
    memory stays flat regardless of the shard being ~0.9 GB.
    """

    import pyarrow as pa

    with pa.memory_map(str(path), "rb") as source:

        try:
            reader = pa.ipc.open_stream(source)

        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_file(source)

        for batch in reader:

            columns = {
                name: batch.column(name).to_pylist()
                for name in batch.schema.names
            }

            row_count = batch.num_rows

            for i in range(row_count):
                yield {
                    name: columns[name][i]
                    for name in columns
                }


# =============================================================================
# STAGE 1: CLEAN ONE SHARD
# =============================================================================

def clean_shard(shard_index, dry_run=False):
    """
    Clean one raw Arrow shard into sangraha_shard_NN.jsonl.

    Returns a statistics dictionary.
    """

    arrow_file = ARROW_DIR / (
        f"sangraha-train-{shard_index:05d}-of-00041.arrow"
    )

    if not arrow_file.exists():
        raise FileNotFoundError(f"Shard not found: {arrow_file}")

    output_file = PROCESSED_DIR / f"sangraha_shard_{shard_index:02d}.jsonl"

    temp_file = output_file.with_suffix(".tmp")

    print()
    print("=" * 78)
    print(f"CLEANING SHARD {shard_index:02d}")
    print("=" * 78)
    print(f"Input : {arrow_file.name}")
    print(f"Output: {output_file.name}")

    if dry_run:
        print("DRY RUN: nothing written.")
        return None

    original = 0
    kept = 0
    removed_empty = 0
    removed_short = 0
    removed_language = 0

    start = time.perf_counter()

    with open(temp_file, "w", encoding="utf-8") as fout:

        for row in iter_arrow_records(arrow_file):

            original += 1

            cleaned = clean_text(row.get("text") or "")

            if not cleaned:
                removed_empty += 1
                continue

            if len(cleaned) < MIN_CHARACTERS:
                removed_short += 1
                continue

            if telugu_ratio(cleaned) < MIN_TELUGU_RATIO:
                removed_language += 1
                continue

            fout.write(
                json.dumps(
                    {
                        "doc_id": row.get("doc_id"),
                        "text": cleaned,
                        "type": row.get("type"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1

            if original % 50_000 == 0:
                print(f"  read {original:,}, kept {kept:,}")

    # Rename only after a complete, successful write.
    temp_file.replace(output_file)

    elapsed = time.perf_counter() - start

    print(f"  original         : {original:,}")
    print(f"  kept             : {kept:,}")
    print(f"  removed empty    : {removed_empty:,}")
    print(f"  removed short    : {removed_short:,}")
    print(f"  removed language : {removed_language:,}")
    print(
        f"  retention        : "
        f"{kept / original * 100:.3f}%" if original else "  retention: n/a"
    )
    print(f"  time             : {elapsed / 60:.2f} minutes")
    print(f"  size             : {output_file.stat().st_size / 1024**3:.3f} GB")

    return {
        "shard": shard_index,
        "original": original,
        "kept": kept,
        "removed_empty": removed_empty,
        "removed_short": removed_short,
        "removed_language": removed_language,
    }


# =============================================================================
# STAGE 2: EXACT DEDUPLICATE ONE SHARD AGAINST THE EXISTING CORPUS
# =============================================================================

def load_hashes():
    """Load the existing SHA-256 text hash set."""

    if not HASH_FILE.exists():
        return set()

    print(f"Loading existing hashes from {HASH_FILE.name} ...")

    with open(HASH_FILE, "r", encoding="utf-8") as f:
        hashes = {line.strip() for line in f if line.strip()}

    print(f"  {len(hashes):,} hashes loaded")

    return hashes


def deduplicate_shard(shard_index, seen_hashes, dry_run=False):
    """
    Append the unique records of one cleaned shard to the deduplicated
    corpus, and extend the hash file.

    New records are staged in a temporary file first. Only once that file
    is complete are the records appended to the corpus and their hashes
    appended to the hash file, in that order. If the run dies partway
    through, the staged file is discarded and the shard is simply
    reprocessed, because the state file was never updated.
    """

    shard_file = PROCESSED_DIR / f"sangraha_shard_{shard_index:02d}.jsonl"

    if not shard_file.exists():
        raise FileNotFoundError(f"Cleaned shard missing: {shard_file}")

    print()
    print("=" * 78)
    print(f"EXACT DEDUPLICATING SHARD {shard_index:02d}")
    print("=" * 78)

    if dry_run:
        print("DRY RUN: nothing written.")
        return None

    staged = PROCESSED_DIR / f"_staged_{shard_index:02d}.jsonl"
    staged_hashes = PROCESSED_DIR / f"_staged_{shard_index:02d}.hashes"

    records = 0
    unique = 0
    duplicates = 0

    start = time.perf_counter()

    with open(staged, "w", encoding="utf-8") as fout, \
         open(staged_hashes, "w", encoding="utf-8") as fhash:

        with open(shard_file, "r", encoding="utf-8") as fin:

            for line in fin:

                if not line.strip():
                    continue

                try:
                    row = json.loads(line)

                except json.JSONDecodeError:
                    continue

                records += 1

                text = row["text"]

                digest = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()

                if digest in seen_hashes:
                    duplicates += 1
                    continue

                seen_hashes.add(digest)

                fout.write(line if line.endswith("\n") else line + "\n")
                fhash.write(digest + "\n")

                unique += 1

    # Commit: append staged content to the live files.
    with open(DEDUP_FILE, "a", encoding="utf-8") as fout:
        with open(staged, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)

    with open(HASH_FILE, "a", encoding="utf-8") as fout:
        with open(staged_hashes, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)

    staged.unlink()
    staged_hashes.unlink()

    elapsed = time.perf_counter() - start

    print(f"  records processed : {records:,}")
    print(f"  unique appended   : {unique:,}")
    print(f"  duplicates removed: {duplicates:,}")
    print(f"  hash set size     : {len(seen_hashes):,}")
    print(f"  time              : {elapsed / 60:.2f} minutes")

    return {
        "shard": shard_index,
        "records": records,
        "unique": unique,
        "duplicates": duplicates,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Clean and deduplicate additional Sangraha shards."
    )

    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        required=True,
        help="Shard indices to process, e.g. --shards 4 5",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing anything.",
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("SANGRAHA SHARD PROCESSING (Model H, Telugu)")
    print("=" * 78)
    print(f"Arrow directory : {ARROW_DIR}")
    print(f"Processed dir   : {PROCESSED_DIR}")
    print(f"Shards requested: {args.shards}")

    state = load_state()

    print(f"Already cleaned : {sorted(state['cleaned'])}")
    print(f"Already deduped : {sorted(state['deduplicated'])}")

    # -------------------------------------------------------------------------
    # STAGE 1
    # -------------------------------------------------------------------------

    for shard_index in args.shards:

        if shard_index in state["cleaned"]:
            print(f"\nShard {shard_index:02d}: already cleaned, skipping.")
            continue

        clean_shard(shard_index, args.dry_run)

        if not args.dry_run:
            state["cleaned"].append(shard_index)
            save_state(state)

    # -------------------------------------------------------------------------
    # STAGE 2
    # -------------------------------------------------------------------------

    pending = [
        s for s in args.shards
        if s not in state["deduplicated"]
    ]

    if pending and not args.dry_run:

        seen_hashes = load_hashes()

        for shard_index in pending:

            deduplicate_shard(shard_index, seen_hashes, args.dry_run)

            state["deduplicated"].append(shard_index)
            save_state(state)

    elif args.dry_run:
        for shard_index in pending:
            deduplicate_shard(shard_index, set(), True)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SANGRAHA SHARD PROCESSING COMPLETE")
    print("=" * 78)

    if not args.dry_run and DEDUP_FILE.exists():

        size_gb = DEDUP_FILE.stat().st_size / 1024**3

        print(f"Deduplicated corpus: {DEDUP_FILE}")
        print(f"  size  : {size_gb:.3f} GB")

        print(f"Shards cleaned     : {sorted(state['cleaned'])}")
        print(f"Shards deduplicated: {sorted(state['deduplicated'])}")

        print()
        print(
            "NEXT: re-run global_dedup.py then global_near_dedup.py so the "
            "enlarged Sangraha corpus is deduplicated against the manual "
            "sources."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
