from pathlib import Path
from collections import Counter
import hashlib
import json
import csv
import time
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed"
)


# ============================================================
# MANUAL DATA
# ============================================================

TEXTBOOK_DIR = (
    BASE
    / "quality_filtered"
    / "textbooks"
)

WIKISOURCE_DIR = (
    BASE
    / "manual"
    / "wikisource"
    / "final"
)

ARCHIVE_DIR = (
    BASE
    / "quality_filtered"
    / "internet_archive"
)

# Telugu Wikipedia, collected by telugu/collect/collect_wikipedia.py.
# Added after Internet Archive collection stalled: the language:"tel"
# seam ran dry and the remaining queries returned English-language books
# about Telugu, leaving manual collection short of its token target.
# Wikipedia also supplies encyclopedic prose, a register absent from the
# other three manual sources.
WIKIPEDIA_DIR = (
    BASE
    / "manual"
    / "wikipedia"
    / "final"
)


# ============================================================
# SANGRAHA
# ============================================================
#
# IMPORTANT:
#
# We use sangraha_deduplicated.jsonl, NOT the shards.
#
# Sangraha's own preprocessing already removed internal
# exact duplicates.
# ============================================================

SANGRAHA_FILE = (
    BASE
    / "sangraha"
    / "sangraha_deduplicated.jsonl"
)

SANGRAHA_HASH_FILE = (
    BASE
    / "sangraha"
    / "dedup_hashes.txt"
)


# ============================================================
# OUTPUT
# ============================================================
#
# Inputs are read from Google Drive, but output goes to LOCAL DISK.
#
# This stage and the near-duplicate stage after it produce roughly
# 15 GB of intermediate data. Writing that to Drive has two costs:
# every write goes through Drive's sync layer, and the result then
# has to upload, which saturates the connection for hours.
#
# None of this output is a deliverable. It is fully reproducible by
# re-running this script over the quality-filtered inputs, so the
# assignment's requirement to put datasets and checkpoints on Drive
# does not apply to it. The train/validation/test splits and the
# tokenizer, which ARE deliverables, are still written to Drive.
#
# WORK can be pointed anywhere with a lot of free space.
# ============================================================

WORK = Path(r"C:\lma_work")

OUT_DIR = (
    WORK
    / "global_dedup"
)

MANUAL_OUT = (
    OUT_DIR
    / "manual"
)

SANGRAHA_OUT = (
    OUT_DIR
    / "sangraha"
)

REPORT_DIR = (
    OUT_DIR
    / "reports"
)


# ============================================================
# REPORT FILES
# ============================================================

SUMMARY_FILE = (
    REPORT_DIR
    / "global_exact_dedup_summary.csv"
)

DUPLICATE_LEDGER = (
    REPORT_DIR
    / "sangraha_duplicates_removed.csv"
)


# ============================================================
# CONFIG
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jsonl",
    ".json",
}


# ============================================================
# HASH FUNCTION
# ============================================================
#
# IMPORTANT:
#
# This MUST match the Sangraha preprocessing code.
#
# Sangraha used:
#
# hashlib.sha256(text.encode("utf-8")).hexdigest()
#
# Therefore we do NOT normalize whitespace here.
#
# ============================================================

def text_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    MANUAL_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    SANGRAHA_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DISCOVER MANUAL FILES
# ============================================================

def discover_files(directory):

    if not directory.exists():

        print()
        print(
            "ERROR: directory does not exist:"
        )

        print(
            directory
        )

        sys.exit(1)

    files = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in (
            SUPPORTED_EXTENSIONS
        ):

            files.append(
                path
            )

    return sorted(
        files
    )


# ============================================================
# READ JSONL
# ============================================================

def read_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.rstrip(
                "\r\n"
            )

            if not line.strip():
                continue

            try:

                record = json.loads(
                    line
                )

            except Exception:

                yield (
                    None,
                    line_number,
                    "INVALID_JSON"
                )

                continue

            yield (
                record,
                line_number,
                None
            )


# ============================================================
# GET TEXT
# ============================================================

def get_text(record):

    if not isinstance(
        record,
        dict
    ):

        return ""

    text = record.get(
        "text",
        ""
    )

    if not isinstance(
        text,
        str
    ):

        return ""

    return text


# ============================================================
# WRITE RECORD
# ============================================================
#
# IMPORTANT:
#
# We write the ORIGINAL record.
#
# We do NOT remove or alter:
#
# source
# book
# page
# doc_id
# type
# labels
# etc.
#
# ============================================================

def write_record(
    fout,
    record
):

    fout.write(
        json.dumps(
            record,
            ensure_ascii=False
        )
        + "\n"
    )


# ============================================================
# PROCESS MANUAL SOURCE
# ============================================================
#
# Manual data is ALWAYS kept.
#
# We only build a global hash set from it.
#
# Even if the same text appears in two Manual sources,
# neither Manual record is removed.
#
# ============================================================

def process_manual_source(
    source_name,
    input_dir,
    output_dir,
    manual_hashes,
    stats
):

    files = discover_files(
        input_dir
    )

    print()
    print("=" * 80)
    print(
        f"MANUAL SOURCE: "
        f"{source_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Files: {len(files):,}"
    )

    start_time = time.time()

    for file_index, input_file in enumerate(
        files,
        start=1
    ):

        # Preserve relative structure.
        relative = input_file.relative_to(
            input_dir
        )

        output_file = (
            output_dir
            / relative
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Always output JSONL.
        if output_file.suffix.lower() != ".jsonl":

            output_file = (
                output_file.with_suffix(
                    ".jsonl"
                )
            )

        kept = 0
        invalid = 0
        characters = 0

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as fout:

            for (
                record,
                record_number,
                error
            ) in read_jsonl(
                input_file
            ):

                stats["records"] += 1

                if error:

                    invalid += 1
                    stats["invalid"] += 1

                    continue

                text = get_text(
                    record
                )

                if not text.strip():

                    invalid += 1
                    stats["invalid"] += 1

                    continue

                digest = text_hash(
                    text
                )

                # ------------------------------------------------
                # Add to global Manual hash set.
                #
                # We DO NOT remove duplicate Manual records.
                # ------------------------------------------------

                manual_hashes.add(
                    digest
                )

                write_record(
                    fout,
                    record
                )

                kept += 1

                characters += len(
                    text
                )

                stats["kept"] += 1

                stats["characters"] += len(
                    text
                )

        stats["files"] += 1

        print(
            f"[{file_index:4d}/{len(files):4d}] "
            f"{input_file.name} "
            f"| kept={kept:,} "
            f"| invalid={invalid:,} "
            f"| chars={characters:,}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"{source_name} completed in "
        f"{elapsed / 60:.2f} minutes"
    )


# ============================================================
# CHECK SANGRAHA HASH FILE
# ============================================================

def check_sangraha_hash_file():

    if not SANGRAHA_HASH_FILE.exists():

        print()
        print(
            "WARNING:"
        )

        print(
            "Sangraha dedup_hashes.txt not found."
        )

        print(
            "We will calculate hashes from text."
        )

        return False

    return True


# ============================================================
# PROCESS SANGRAHA
# ============================================================
#
# Sangraha has ALREADY been internally exact-deduplicated.
#
# We only remove records whose content already exists in
# Manual data.
#
# ============================================================

def process_sangraha(
    manual_hashes,
    stats,
    ledger
):

    if not SANGRAHA_FILE.exists():

        print()
        print(
            "ERROR: Sangraha deduplicated file not found:"
        )

        print(
            SANGRAHA_FILE
        )

        sys.exit(1)

    print()
    print("=" * 80)
    print("SANGRAHA / CROSS-SOURCE EXACT DEDUPLICATION")
    print("=" * 80)

    print()
    print(
        f"Input: {SANGRAHA_FILE}"
    )

    print(
        f"Manual hashes available: "
        f"{len(manual_hashes):,}"
    )

    # --------------------------------------------------------
    # Use existing Sangraha hashes if available.
    # --------------------------------------------------------

    use_existing_hashes = (
        check_sangraha_hash_file()
    )

    hash_file = None

    if use_existing_hashes:

        hash_file = open(
            SANGRAHA_HASH_FILE,
            "r",
            encoding="utf-8"
        )

        print(
            "Using existing Sangraha SHA-256 hashes."
        )

    else:

        print(
            "Calculating Sangraha SHA-256 hashes."
        )

    output_file = (
        SANGRAHA_OUT
        / "sangraha_global_exact_dedup.jsonl"
    )

    start_time = time.time()

    processed = 0
    kept = 0
    removed = 0
    invalid = 0

    kept_chars = 0
    removed_chars = 0

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as fout:

        for (
            record,
            record_number,
            error
        ) in read_jsonl(
            SANGRAHA_FILE
        ):

            if error:

                invalid += 1
                stats["invalid"] += 1

                continue

            text = get_text(
                record
            )

            if not text.strip():

                invalid += 1
                stats["invalid"] += 1

                continue

            # ------------------------------------------------
            # Obtain Sangraha's existing hash.
            # ------------------------------------------------

            existing_hash = None

            if hash_file is not None:

                hash_line = (
                    hash_file.readline()
                )

                if hash_line:

                    hash_line = (
                        hash_line.strip()
                    )

                    if len(hash_line) == 64:

                        existing_hash = (
                            hash_line
                        )

            # ------------------------------------------------
            # Fallback:
            # calculate if hash file is unavailable or
            # malformed.
            # ------------------------------------------------

            if existing_hash is None:

                existing_hash = text_hash(
                    text
                )

            digest = existing_hash

            # ------------------------------------------------
            # Cross-source duplicate?
            # ------------------------------------------------

            if digest in manual_hashes:

                removed += 1

                removed_chars += len(
                    text
                )

                stats[
                    "duplicates_removed"
                ] += 1

                stats[
                    "duplicate_characters"
                ] += len(
                    text
                )

                ledger.writerow([
                    digest,
                    record_number,
                    record.get(
                        "doc_id",
                        ""
                    ),
                    len(text),
                ])

            else:

                write_record(
                    fout,
                    record
                )

                kept += 1

                kept_chars += len(
                    text
                )

                stats[
                    "kept"
                ] += 1

                stats[
                    "characters"
                ] += len(
                    text
                )

            processed += 1
            stats[
                "records"
            ] += 1

            # ------------------------------------------------
            # Progress every 100,000 records.
            # ------------------------------------------------

            if processed % 100000 == 0:

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{processed:,} "
                    f"| kept={kept:,} "
                    f"| removed={removed:,} "
                    f"| invalid={invalid:,} "
                    f"| {rate:,.0f} records/sec"
                )

    if hash_file is not None:

        hash_file.close()

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        "Sangraha completed."
    )

    print(
        f"Records processed : "
        f"{processed:,}"
    )

    print(
        f"Kept              : "
        f"{kept:,}"
    )

    print(
        f"Removed            : "
        f"{removed:,}"
    )

    print(
        f"Invalid            : "
        f"{invalid:,}"
    )

    print(
        f"Characters kept   : "
        f"{kept_chars:,}"
    )

    print(
        f"Characters removed: "
        f"{removed_chars:,}"
    )

    print(
        f"Time               : "
        f"{elapsed / 60:.2f} minutes"
    )

    stats["files"] = 1


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("GLOBAL EXACT DEDUPLICATION")
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Manual data is protected."
    )

    print(
        "Only Sangraha copies of Manual duplicates are removed."
    )

    print(
        "Original input files will NOT be modified."
    )

    print()

    create_directories()

    # ========================================================
    # REMOVE ONLY OUR PREVIOUS OUTPUT
    # ========================================================
    #
    # We intentionally do NOT delete source data.
    #
    # ========================================================

    old_sangraha_output = (
        SANGRAHA_OUT
        / "sangraha_global_exact_dedup.jsonl"
    )

    if old_sangraha_output.exists():

        print(
            "Removing previous incomplete output:"
        )

        print(
            old_sangraha_output
        )

        old_sangraha_output.unlink()

    # ========================================================
    # GLOBAL MANUAL HASH SET
    # ========================================================

    manual_hashes = set()

    manual_stats = {
        "textbook": {
            "files": 0,
            "records": 0,
            "kept": 0,
            "invalid": 0,
            "characters": 0,
        },

        "wikisource": {
            "files": 0,
            "records": 0,
            "kept": 0,
            "invalid": 0,
            "characters": 0,
        },

        "wikipedia": {
            "files": 0,
            "records": 0,
            "kept": 0,
            "invalid": 0,
            "characters": 0,
        },

        "archive": {
            "files": 0,
            "records": 0,
            "kept": 0,
            "invalid": 0,
            "characters": 0,
        },
    }

    # ========================================================
    # 1. TEXTBOOK
    # ========================================================

    process_manual_source(
        "textbook",
        TEXTBOOK_DIR,
        MANUAL_OUT / "textbook",
        manual_hashes,
        manual_stats["textbook"]
    )

    # ========================================================
    # 2. WIKISOURCE
    # ========================================================

    process_manual_source(
        "wikisource",
        WIKISOURCE_DIR,
        MANUAL_OUT / "wikisource",
        manual_hashes,
        manual_stats["wikisource"]
    )

    # ========================================================
    # 3. WIKIPEDIA
    # ========================================================

    process_manual_source(
        "wikipedia",
        WIKIPEDIA_DIR,
        MANUAL_OUT / "wikipedia",
        manual_hashes,
        manual_stats["wikipedia"]
    )

    # ========================================================
    # 4. INTERNET ARCHIVE
    # ========================================================

    process_manual_source(
        "archive",
        ARCHIVE_DIR,
        MANUAL_OUT / "archive",
        manual_hashes,
        manual_stats["archive"]
    )

    # ========================================================
    # MANUAL SUMMARY
    # ========================================================

    manual_files = sum(
        x["files"]
        for x in manual_stats.values()
    )

    manual_records = sum(
        x["kept"]
        for x in manual_stats.values()
    )

    manual_chars = sum(
        x["characters"]
        for x in manual_stats.values()
    )

    print()
    print("=" * 80)
    print("MANUAL SOURCES COMPLETE")
    print("=" * 80)

    print(
        f"Files              : "
        f"{manual_files:,}"
    )

    print(
        f"Records            : "
        f"{manual_records:,}"
    )

    print(
        f"Unique Manual hashes: "
        f"{len(manual_hashes):,}"
    )

    print(
        f"Characters         : "
        f"{manual_chars:,}"
    )

    # ========================================================
    # 4. SANGRAHA
    # ========================================================

    sangraha_stats = {
        "files": 0,
        "records": 0,
        "kept": 0,
        "characters": 0,
        "invalid": 0,
        "duplicates_removed": 0,
        "duplicate_characters": 0,
    }

    # --------------------------------------------------------
    # Ledger
    # --------------------------------------------------------

    ledger_file = open(
        DUPLICATE_LEDGER,
        "w",
        encoding="utf-8",
        newline=""
    )

    ledger = csv.writer(
        ledger_file
    )

    ledger.writerow([
        "sha256",
        "sangraha_record_number",
        "sangraha_doc_id",
        "characters_removed",
    ])

    process_sangraha(
        manual_hashes,
        sangraha_stats,
        ledger
    )

    ledger_file.close()

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "source_group",
            "source",
            "files",
            "records_processed",
            "records_kept",
            "duplicates_removed",
            "invalid",
            "characters_kept",
            "duplicate_characters_removed",
        ])

        for source_name in [
            "textbook",
            "wikisource",
            "wikipedia",
            "archive",
        ]:

            s = manual_stats[
                source_name
            ]

            writer.writerow([
                "manual",
                source_name,
                s["files"],
                s["records"],
                s["kept"],
                0,
                s["invalid"],
                s["characters"],
                0,
            ])

        writer.writerow([
            "sangraha",
            "sangraha",
            sangraha_stats["files"],
            sangraha_stats["records"],
            sangraha_stats["kept"],
            sangraha_stats[
                "duplicates_removed"
            ],
            sangraha_stats["invalid"],
            sangraha_stats["characters"],
            sangraha_stats[
                "duplicate_characters"
            ],
        ])

    # ========================================================
    # FINAL NUMBERS
    # ========================================================

    final_manual_chars = (
        manual_chars
    )

    final_sangraha_chars = (
        sangraha_stats["characters"]
    )

    final_chars = (
        final_manual_chars
        + final_sangraha_chars
    )

    total_records = (
        manual_records
        + sangraha_stats["kept"]
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("GLOBAL EXACT DEDUPLICATION COMPLETE")
    print("=" * 80)

    print()

    print("MANUAL")
    print("-" * 40)

    print(
        f"Textbook records    : "
        f"{manual_stats['textbook']['kept']:,}"
    )

    print(
        f"Wikisource records  : "
        f"{manual_stats['wikisource']['kept']:,}"
    )

    print(
        f"Wikipedia records   : "
        f"{manual_stats['wikipedia']['kept']:,}"
    )

    print(
        f"Archive records     : "
        f"{manual_stats['archive']['kept']:,}"
    )

    print(
        f"Manual total records: "
        f"{manual_records:,}"
    )

    print(
        f"Manual characters   : "
        f"{manual_chars:,}"
    )

    print()

    print("SANGRAHA")
    print("-" * 40)

    print(
        f"Input records       : "
        f"{sangraha_stats['records']:,}"
    )

    print(
        f"Kept records        : "
        f"{sangraha_stats['kept']:,}"
    )

    print(
        f"Manual duplicates   : "
        f"{sangraha_stats['duplicates_removed']:,}"
    )

    print(
        f"Characters removed  : "
        f"{sangraha_stats['duplicate_characters']:,}"
    )

    print(
        f"Characters kept     : "
        f"{sangraha_stats['characters']:,}"
    )

    print()

    print("FINAL CORPUS")
    print("-" * 40)

    print(
        f"Total records       : "
        f"{total_records:,}"
    )

    print(
        f"Manual characters   : "
        f"{final_manual_chars:,}"
    )

    print(
        f"Sangraha characters : "
        f"{final_sangraha_chars:,}"
    )

    print(
        f"Total characters    : "
        f"{final_chars:,}"
    )

    if final_chars > 0:

        print(
            f"Manual proportion   : "
            f"{final_manual_chars / final_chars:.2%}"
        )

        print(
            f"Sangraha proportion : "
            f"{final_sangraha_chars / final_chars:.2%}"
        )

    print()

    print("OUTPUT")
    print("-" * 40)

    print(
        f"Manual corpus:"
    )

    print(
        MANUAL_OUT
    )

    print()

    print(
        f"Sangraha corpus:"
    )

    print(
        SANGRAHA_OUT
    )

    print()

    print(
        f"Duplicate ledger:"
    )

    print(
        DUPLICATE_LEDGER
    )

    print()

    print(
        f"Summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()