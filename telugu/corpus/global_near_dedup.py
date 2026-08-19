from pathlib import Path
import json
import csv
import time
import sys
import re
import pickle

import numpy as np

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print()
    print("=" * 80)
    print("ERROR: datasketch is not installed")
    print("=" * 80)
    print()
    print("Run:")
    print()
    print("pip install datasketch")
    print()
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed"
)

# Intermediate corpus data lives on local disk rather than Drive.
# See the OUTPUT note in global_dedup.py for why: this stage reads and
# writes several gigabytes that are reproducible and are not
# deliverables, so routing them through Drive's sync layer costs hours
# for no benefit.
WORK = Path(r"C:\lma_work")

# ============================================================
# INPUT
# ============================================================
#
# This is the result of GLOBAL EXACT DEDUPLICATION.
#
# We DO NOT read:
#
#   data/raw/
#   sangraha_shard_*.jsonl
#
# ============================================================

GLOBAL_DEDUP_DIR = (
    WORK / "global_dedup"
)

MANUAL_INPUT = (
    GLOBAL_DEDUP_DIR / "manual"
)

SANGRAHA_INPUT = (
    GLOBAL_DEDUP_DIR / "sangraha"
)

# ------------------------------------------------------------
# The Sangraha input is named EXPLICITLY rather than globbed.
#
# The output directory of global_dedup.py also contains
# sangraha_deduplicated.jsonl, which is a partial leftover
# (195,475 of 681,635 records) written by an older version of
# that script. Globbing *.jsonl picked up both files, so every
# record present in the leftover was compared against itself
# and discarded as a near-duplicate of itself.
#
# Only the file below is a real global-exact-dedup output.
# ------------------------------------------------------------

SANGRAHA_INPUT_FILE = (
    SANGRAHA_INPUT / "sangraha_global_exact_dedup.jsonl"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    WORK / "global_near_dedup"
)

MANUAL_OUTPUT = (
    OUTPUT_DIR / "manual"
)

SANGRAHA_OUTPUT = (
    OUTPUT_DIR / "sangraha"
)

REPORT_DIR = (
    OUTPUT_DIR / "reports"
)


REMOVED_LEDGER = (
    REPORT_DIR /
    "near_duplicates_removed.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "global_near_dedup_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Previous experimental threshold.
NEAR_DUP_THRESHOLD = 0.90

# Number of MinHash permutations.
NUM_PERM = 64

# Number of LSH bands is internally determined by datasketch.
# threshold=0.90 gives relatively conservative candidate
# generation.

# Minimum number of words.
MIN_WORDS = 20

# Shingle size.
SHINGLE_SIZE = 5

# Progress frequency.
PROGRESS_EVERY = 10000


# ============================================================
# CHECKPOINTING
# ============================================================
#
# This stage takes about three hours and holds all of its state in
# memory: one MinHash LSH index plus a signature for every document
# kept so far. A previous run was killed after two hours with nothing
# recoverable, because the index cannot be rebuilt from the output
# files without re-shingling every document.
#
# So the index, the signature store and the progress markers are
# written to disk periodically. On restart the script reloads them and
# continues from the last completed block instead of from zero.
#
# Granularity differs by source:
#
#   manual sources   checkpointed when the source completes. They take
#                    2 to 22 minutes each, so losing one is cheap.
#
#   sangraha         checkpointed every CHECKPOINT_EVERY records, and
#                    its output is written in numbered blocks. A block
#                    is only counted as done once its checkpoint is
#                    saved, so a partial block is simply rewritten on
#                    resume rather than being appended to twice.
# ============================================================

# Records between Sangraha checkpoints.
#
# At the observed ~130 records/sec this is roughly six minutes of work,
# which is the most a kill can cost. Writing the checkpoint costs a few
# seconds once the signature store is large, so a smaller interval would
# start to pay a noticeable overhead for little extra safety.
CHECKPOINT_EVERY = 50_000

# Files between checkpoints inside a multi-file manual source.
# Internet Archive's 1,641 files run at roughly 100 files/minute, so this
# caps the loss from a kill at about two minutes.
CHECKPOINT_FILES = 200

CHECKPOINT_FILE = None  # set in main(), lives under OUTPUT_DIR


def save_checkpoint(path, lsh, candidate_stats, state, all_stats):
    """
    Persist the deduplication state atomically.

    Written to a temporary file and renamed into place, so a kill during
    the write cannot leave a half-written checkpoint that would be
    loaded as valid on the next run.
    """

    payload = {
        "lsh": lsh,
        "signatures": candidate_stats["signatures"],
        "candidates": candidate_stats["candidates"],
        "state": state,
        "all_stats": all_stats,
        "settings": {
            "threshold": NEAR_DUP_THRESHOLD,
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "min_words": MIN_WORDS,
        },
    }

    temp = path.with_suffix(".tmp")

    start = time.time()

    with open(temp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    temp.replace(path)

    print(
        f"  checkpoint saved: {len(candidate_stats['signatures']):,} "
        f"signatures, {path.stat().st_size / 1024**2:,.0f} MB, "
        f"{time.time() - start:.1f}s"
    )


def load_checkpoint(path):
    """
    Restore a previous run's state, or return None if there is none.

    The dedup settings are compared against the current ones. Resuming a
    run that used a different threshold or permutation count would mix
    two incompatible index configurations in a single output, so that is
    refused rather than silently accepted.
    """

    if not path.exists():
        return None

    print()
    print(f"Loading checkpoint: {path}")

    start = time.time()

    with open(path, "rb") as f:
        payload = pickle.load(f)

    settings = payload.get("settings", {})

    current = {
        "threshold": NEAR_DUP_THRESHOLD,
        "num_perm": NUM_PERM,
        "shingle_size": SHINGLE_SIZE,
        "min_words": MIN_WORDS,
    }

    if settings != current:

        print()
        print("ERROR: checkpoint was written with different settings.")
        print(f"  checkpoint: {settings}")
        print(f"  current   : {current}")
        print()
        print(
            "Delete the checkpoint to start a fresh run, or restore the "
            "original settings to resume."
        )

        sys.exit(1)

    print(
        f"  restored {len(payload['signatures']):,} signatures in "
        f"{time.time() - start:.1f}s"
    )

    print(f"  completed sources : {payload['state']['completed_sources']}")
    print(f"  sangraha records  : {payload['state']['sangraha_records']:,}")
    print(f"  sangraha blocks   : {payload['state']['sangraha_blocks']}")

    return payload


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    MANUAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    SANGRAHA_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_jsonl_files(directory):

    if not directory.exists():

        print()
        print("ERROR: directory not found:")
        print(directory)
        print()

        sys.exit(1)

    files = sorted(
        p
        for p in directory.rglob("*.jsonl")
        if p.is_file()
    )

    return files


# ============================================================
# JSONL READER
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
                    True
                )

                continue

            yield (
                record,
                line_number,
                False
            )


# ============================================================
# TEXT EXTRACTION
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
# TOKENIZATION FOR SHINGLES
# ============================================================
#
# We use Unicode-aware word extraction.
#
# Telugu Unicode characters are retained.
#
# We are NOT modifying the actual corpus text.
# This is only used for near-duplicate comparison.
# ============================================================

WORD_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)


def get_words(text):

    return WORD_RE.findall(
        text.lower()
    )


# ============================================================
# SHINGLES
# ============================================================

def get_shingles(
    text,
    k=SHINGLE_SIZE
):

    words = get_words(
        text
    )

    if len(words) < k:

        return set(words)

    return {
        " ".join(
            words[i:i + k]
        )
        for i in range(
            len(words) - k + 1
        )
    }


# ============================================================
# MINHASH
# ============================================================

def make_minhash(
    shingles
):

    mh = MinHash(
        num_perm=NUM_PERM
    )

    for shingle in shingles:

        mh.update(
            shingle.encode(
                "utf-8"
            )
        )

    return mh


def jaccard(
    hashvalues_a,
    hashvalues_b
):
    """
    Estimate the Jaccard similarity of two documents from their
    MinHash permutation vectors.

    This is the same estimator datasketch's MinHash.jaccard uses:
    the fraction of permutations on which the two signatures agree.
    It is reimplemented here so the signature store can hold plain
    numpy arrays instead of full MinHash objects.
    """

    return (
        np.count_nonzero(
            hashvalues_a == hashvalues_b
        )
        / float(NUM_PERM)
    )


# ============================================================
# OUTPUT RECORD
# ============================================================
#
# Original fields are preserved.
#
# We only ADD provenance fields.
# ============================================================

def write_record(
    fout,
    record,
    source_group,
    source_type
):

    output = dict(
        record
    )

    output[
        "_near_dedup_source_group"
    ] = source_group

    output[
        "_near_dedup_source_type"
    ] = source_type

    fout.write(
        json.dumps(
            output,
            ensure_ascii=False
        )
        + "\n"
    )


# ============================================================
# ITERATE MANUAL FILES
# ============================================================

def get_manual_files():

    textbook_dir = (
        MANUAL_INPUT / "textbook"
    )

    wikisource_dir = (
        MANUAL_INPUT / "wikisource"
    )

    wikipedia_dir = (
        MANUAL_INPUT / "wikipedia"
    )

    archive_dir = (
        MANUAL_INPUT / "archive"
    )

    sources = []

    # Order matters. Manual sources are processed before Sangraha so that
    # manual records become the protected representatives in the LSH
    # index, and within manual the scarcer sources go first for the same
    # reason: if a textbook page and a newspaper page are near-duplicates,
    # the textbook page is the one that survives.
    for name, directory in [
        (
            "textbook",
            textbook_dir
        ),
        (
            "wikisource",
            wikisource_dir
        ),
        (
            "wikipedia",
            wikipedia_dir
        ),
        (
            "archive",
            archive_dir
        ),
    ]:

        if not directory.exists():

            print(
                f"WARNING: {name} directory missing:"
            )

            print(
                directory
            )

            continue

        files = discover_jsonl_files(
            directory
        )

        sources.append(
            (
                "manual",
                name,
                directory,
                files
            )
        )

    return sources


# ============================================================
# ITERATE SANGRAHA
# ============================================================

def get_sangraha_files():

    if not SANGRAHA_INPUT_FILE.exists():

        print()
        print(
            "ERROR: Sangraha global-dedup output missing:"
        )

        print(
            SANGRAHA_INPUT_FILE
        )

        sys.exit(1)

    files = [
        SANGRAHA_INPUT_FILE
    ]

    return [
        (
            "sangraha",
            "sangraha",
            SANGRAHA_INPUT,
            files
        )
    ]


# ============================================================
# SUMMARY STRUCTURE
# ============================================================

def new_stats():

    return {
        "files": 0,
        "records": 0,
        "kept": 0,
        "removed": 0,
        "invalid": 0,
        "characters_kept": 0,
        "characters_removed": 0,
    }


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(
    source_group,
    source_type,
    source_dir,
    files,
    output_root,
    lsh,
    stats,
    ledger,
    candidate_stats,
    state=None,
    checkpoint_path=None,
    all_stats=None,
):

    print()
    print("=" * 80)

    print(
        f"PROCESSING "
        f"{source_group.upper()} / "
        f"{source_type.upper()}"
    )

    print("=" * 80)

    print(
        f"Files: {len(files):,}"
    )

    # ------------------------------------------------------------------
    # FILE-LEVEL RESUME
    # ------------------------------------------------------------------
    # Internet Archive is 1,641 files taking ~15 minutes. Checkpointing
    # only when the whole source finishes means a kill part-way through
    # discards every file processed so far, and if kills arrive more
    # often than the source takes, it can never complete.
    #
    # So progress is recorded every CHECKPOINT_FILES files. The file list
    # is sorted deterministically by discover_jsonl_files, so a count is
    # enough to identify where to resume.
    #
    # `processed` must be advanced over skipped records as well, because
    # it forms the LSH key and those keys are already in the restored
    # index.
    # ------------------------------------------------------------------

    files_done = 0

    if state is not None:
        files_done = state.get("source_progress", {}).get(source_type, 0)

    if files_done:
        print(f"Resuming after {files_done:,} files already processed")

    start_time = time.time()

    processed = state.get("source_records", {}).get(source_type, 0) \
        if state is not None else 0

    for file_index, input_file in enumerate(
        files,
        start=1
    ):

        # Already handled before the last checkpoint.
        if file_index <= files_done:
            continue

        relative = input_file.relative_to(
            source_dir
        )

        output_file = (
            output_root /
            source_type /
            relative
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        kept_file = 0
        removed_file = 0
        invalid_file = 0

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as fout:

            for (
                record,
                record_number,
                invalid
            ) in read_jsonl(
                input_file
            ):

                processed += 1

                stats[
                    "records"
                ] += 1

                if invalid:

                    stats[
                        "invalid"
                    ] += 1

                    invalid_file += 1

                    continue

                text = get_text(
                    record
                )

                if not text.strip():

                    stats[
                        "invalid"
                    ] += 1

                    invalid_file += 1

                    continue

                characters = len(
                    text
                )

                words = get_words(
                    text
                )

                # ------------------------------------------------
                # Very short documents don't provide reliable
                # near-duplicate evidence.
                #
                # They are retained.
                # ------------------------------------------------

                if len(words) < MIN_WORDS:

                    write_record(
                        fout,
                        record,
                        source_group,
                        source_type
                    )

                    stats[
                        "kept"
                    ] += 1

                    stats[
                        "characters_kept"
                    ] += characters

                    kept_file += 1

                    # ------------------------------------------------
                    # Documents below MIN_WORDS are kept unconditionally
                    # and are NOT indexed.
                    #
                    # Indexing them previously caused two problems:
                    #
                    #   1. They were inserted into the LSH index but
                    #      never recorded in the signature store, so a
                    #      later query returning one of their keys hit
                    #      a KeyError and killed the run.
                    #
                    #   2. Indexing adds candidates rather than
                    #      removing them. A document of fewer than 20
                    #      words cannot reach Jaccard 0.90 against a
                    #      normal-length document anyway, so the entry
                    #      only costs memory and comparisons.
                    # ------------------------------------------------

                    continue

                shingles = get_shingles(
                    text
                )

                if not shingles:

                    write_record(
                        fout,
                        record,
                        source_group,
                        source_type
                    )

                    stats[
                        "kept"
                    ] += 1

                    stats[
                        "characters_kept"
                    ] += characters

                    kept_file += 1

                    continue

                mh = make_minhash(
                    shingles
                )

                # ------------------------------------------------
                # Candidate generation
                # ------------------------------------------------

                candidates = lsh.query(
                    mh
                )

                candidate_stats[
                    "candidates"
                ] += len(
                    candidates
                )

                duplicate = False

                best_similarity = 0.0

                best_candidate = None

                # ------------------------------------------------
                # MinHash similarity verification
                # ------------------------------------------------

                for candidate_key in candidates:

                    candidate_signature = (
                        candidate_stats[
                            "signatures"
                        ].get(candidate_key)
                    )

                    # A key in the LSH index with no stored
                    # signature cannot happen now that short
                    # documents are no longer indexed, but a
                    # missing signature must never abort a run
                    # that takes this long.
                    if candidate_signature is None:
                        continue

                    similarity = jaccard(
                        mh.hashvalues,
                        candidate_signature
                    )

                    if similarity > best_similarity:

                        best_similarity = (
                            similarity
                        )

                        best_candidate = (
                            candidate_key
                        )

                    if (
                        similarity
                        >= NEAR_DUP_THRESHOLD
                    ):

                        duplicate = True

                        break

                # ------------------------------------------------
                # DUPLICATE
                # ------------------------------------------------

                if duplicate:

                    stats[
                        "removed"
                    ] += 1

                    stats[
                        "characters_removed"
                    ] += characters

                    removed_file += 1

                    ledger.writerow([
                        source_group,
                        source_type,
                        str(input_file),
                        record_number,

                        record.get(
                            "doc_id",
                            ""
                        ),

                        characters,

                        best_candidate,
                        f"{best_similarity:.6f}",
                    ])

                    continue

                # ------------------------------------------------
                # KEEP
                # ------------------------------------------------

                write_record(
                    fout,
                    record,
                    source_group,
                    source_type
                )

                stats[
                    "kept"
                ] += 1

                stats[
                    "characters_kept"
                ] += characters

                kept_file += 1

                # ------------------------------------------------
                # Add ONLY KEPT records to LSH.
                #
                # This is important:
                # if a record is discarded, it should not become
                # the representative for later records.
                # ------------------------------------------------

                key = (
                    f"{source_type}:"
                    f"{processed}"
                )

                lsh.insert(
                    key,
                    mh
                )

                # Store the raw permutation vector rather than the
                # MinHash object. It carries the same information for
                # a Jaccard estimate at a fraction of the memory,
                # which matters across ~700k documents.
                candidate_stats[
                    "signatures"
                ][key] = mh.hashvalues.copy()

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    processed
                    % PROGRESS_EVERY
                    == 0
                ):

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
                        f"{source_type}: "
                        f"{processed:,} records | "
                        f"kept={stats['kept']:,} | "
                        f"removed={stats['removed']:,} | "
                        f"candidates={candidate_stats['candidates']:,} | "
                        f"{rate:,.0f} records/sec"
                    )

        stats[
            "files"
        ] += 1

        print(
            f"[{file_index:4d}/{len(files):4d}] "
            f"{input_file.name} "
            f"| kept={kept_file:,} "
            f"| removed={removed_file:,} "
            f"| invalid={invalid_file:,}"
        )

        # --------------------------------------------------------------
        # PERIODIC CHECKPOINT
        # --------------------------------------------------------------
        # Recorded after the file's output is closed, so the checkpoint
        # never claims progress that is not on disk. Output files are
        # opened with "w", so a file interrupted before its checkpoint is
        # simply rewritten on resume rather than duplicated.
        # --------------------------------------------------------------

        if (
            state is not None
            and checkpoint_path is not None
            and file_index % CHECKPOINT_FILES == 0
            and file_index < len(files)
        ):

            state.setdefault("source_progress", {})[source_type] = file_index
            state.setdefault("source_records", {})[source_type] = processed

            if all_stats is not None:
                all_stats[source_type] = stats

            save_checkpoint(
                checkpoint_path, lsh, candidate_stats, state, all_stats or {}
            )

    elapsed = (
        time.time()
        - start_time
    )

    # The source is finished, so its partial progress markers are no
    # longer meaningful and are cleared to keep the state readable.
    if state is not None:
        state.setdefault("source_progress", {}).pop(source_type, None)
        state.setdefault("source_records", {}).pop(source_type, None)

    print()

    print(
        f"{source_type} completed in "
        f"{elapsed / 60:.2f} minutes"
    )


# ============================================================
# PROCESS SANGRAHA (RESUMABLE)
# ============================================================
#
# Sangraha is handled separately from the manual sources because it is
# a single 1.03 million record file that takes over two hours. Writing
# it as one output file means a kill loses everything; writing it in
# numbered blocks with a checkpoint after each means a kill loses at
# most one block.
#
# The manual path above is deliberately left unchanged. Those sources
# complete in 2 to 22 minutes each and already work.
# ============================================================

def process_sangraha(
    input_file,
    output_root,
    lsh,
    stats,
    ledger,
    candidate_stats,
    state,
    checkpoint_path,
    all_stats,
):

    print()
    print("=" * 80)
    print("PROCESSING SANGRAHA / SANGRAHA")
    print("=" * 80)

    output_dir = output_root / "sangraha"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_records = state["sangraha_records"]
    block_index = state["sangraha_blocks"]

    if resume_records:
        print(
            f"Resuming after {resume_records:,} records "
            f"({block_index} blocks already written)"
        )

    start_time = time.time()

    processed = 0
    since_checkpoint = 0

    fout = None

    def open_block(index):
        path = output_dir / f"sangraha_block_{index:05d}.jsonl"
        return open(path, "w", encoding="utf-8"), path

    try:

        for record, record_number, invalid in read_jsonl(input_file):

            processed += 1

            # ----------------------------------------------------------
            # RESUME
            # ----------------------------------------------------------
            # Records up to the last checkpoint were already written and
            # already inserted into the restored index, so they are
            # skipped without being reprocessed.
            # ----------------------------------------------------------

            if processed <= resume_records:
                continue

            if fout is None:
                fout, _ = open_block(block_index)

            stats["records"] += 1

            if invalid:
                stats["invalid"] += 1
                continue

            text = get_text(record)

            if not text.strip():
                stats["invalid"] += 1
                continue

            characters = len(text)
            words = get_words(text)

            if len(words) < MIN_WORDS:
                write_record(fout, record, "sangraha", "sangraha")
                stats["kept"] += 1
                stats["characters_kept"] += characters
                since_checkpoint += 1
                continue

            shingles = get_shingles(text)

            if not shingles:
                write_record(fout, record, "sangraha", "sangraha")
                stats["kept"] += 1
                stats["characters_kept"] += characters
                since_checkpoint += 1
                continue

            mh = make_minhash(shingles)

            candidates = lsh.query(mh)

            candidate_stats["candidates"] += len(candidates)

            duplicate = False
            best_similarity = 0.0
            best_candidate = None

            for candidate_key in candidates:

                signature = candidate_stats["signatures"].get(candidate_key)

                if signature is None:
                    continue

                similarity = jaccard(mh.hashvalues, signature)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_candidate = candidate_key

                if similarity >= NEAR_DUP_THRESHOLD:
                    duplicate = True
                    break

            if duplicate:

                stats["removed"] += 1
                stats["characters_removed"] += characters

                ledger.writerow([
                    "sangraha",
                    "sangraha",
                    str(input_file),
                    record_number,
                    record.get("doc_id", ""),
                    characters,
                    best_candidate,
                    f"{best_similarity:.6f}",
                ])

                since_checkpoint += 1
                continue

            write_record(fout, record, "sangraha", "sangraha")

            stats["kept"] += 1
            stats["characters_kept"] += characters

            key = f"sangraha:{processed}"

            lsh.insert(key, mh)

            candidate_stats["signatures"][key] = mh.hashvalues.copy()

            since_checkpoint += 1

            # ----------------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------------

            if stats["records"] % PROGRESS_EVERY == 0:

                elapsed = time.time() - start_time

                rate = (
                    stats["records"] / elapsed if elapsed > 0 else 0
                )

                print(
                    f"sangraha: {processed:,} records | "
                    f"kept={stats['kept']:,} | "
                    f"removed={stats['removed']:,} | "
                    f"candidates={candidate_stats['candidates']:,} | "
                    f"{rate:,.0f} records/sec"
                )

            # ----------------------------------------------------------
            # CHECKPOINT
            # ----------------------------------------------------------
            # The block is closed first, so the checkpoint is only
            # written once its output is safely on disk. If the process
            # dies between the two, the block is simply rewritten on
            # resume, because the checkpoint still names the earlier
            # record count.
            # ----------------------------------------------------------

            if since_checkpoint >= CHECKPOINT_EVERY:

                fout.close()
                fout = None

                block_index += 1

                state["sangraha_records"] = processed
                state["sangraha_blocks"] = block_index
                state["current_source"] = "sangraha"

                all_stats["sangraha"] = stats

                save_checkpoint(
                    checkpoint_path, lsh, candidate_stats, state, all_stats
                )

                since_checkpoint = 0

    finally:
        if fout is not None:
            fout.close()

    # Final block and final checkpoint.
    if since_checkpoint > 0:
        block_index += 1

    stats["files"] = 1

    state["sangraha_records"] = processed
    state["sangraha_blocks"] = block_index
    state["completed_sources"].append("sangraha")
    state["current_source"] = None

    all_stats["sangraha"] = stats

    save_checkpoint(
        checkpoint_path, lsh, candidate_stats, state, all_stats
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"sangraha completed in {elapsed / 60:.2f} minutes "
        f"({block_index} blocks)"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 80)
    print("GLOBAL NEAR-DUPLICATE FILTER")
    print("=" * 80)

    print()

    print(
        f"Threshold          : "
        f"{NEAR_DUP_THRESHOLD}"
    )

    print(
        f"MinHash permutations: "
        f"{NUM_PERM}"
    )

    print(
        f"Shingle size       : "
        f"{SHINGLE_SIZE} words"
    )

    print(
        f"Minimum words      : "
        f"{MIN_WORDS}"
    )

    print()

    print(
        "INPUT:"
    )

    print(
        GLOBAL_DEDUP_DIR
    )

    print()

    print(
        "OUTPUT:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    # ========================================================
    # CHECKPOINT / RESUME
    # ========================================================

    global CHECKPOINT_FILE

    CHECKPOINT_FILE = OUTPUT_DIR / "near_dedup_checkpoint.pkl"

    checkpoint = load_checkpoint(CHECKPOINT_FILE)

    if checkpoint:

        lsh = checkpoint["lsh"]

        candidate_stats = {
            "candidates": checkpoint["candidates"],
            "signatures": checkpoint["signatures"],
        }

        state = checkpoint["state"]
        all_stats = checkpoint["all_stats"]

        # Checkpoints written before file-level resume existed lack these
        # keys, so they are backfilled rather than crashing the resume.
        state.setdefault("source_progress", {})
        state.setdefault("source_records", {})

        print("  RESUMING from checkpoint")

        if state["source_progress"]:
            print(f"  partial sources   : {state['source_progress']}")

    else:

        print()
        print("Creating MinHash LSH index...")

        lsh = MinHashLSH(
            threshold=NEAR_DUP_THRESHOLD,
            num_perm=NUM_PERM
        )

        # ----------------------------------------------------
        # Signatures are kept alongside the index because
        # MinHashLSH does not retain them, and they are needed
        # to verify each candidate rather than trusting the
        # banding. 64 permutations keeps this affordable.
        # ----------------------------------------------------

        candidate_stats = {
            "candidates": 0,
            "signatures": {}
        }

        state = {
            "completed_sources": [],
            "current_source": None,
            "sangraha_records": 0,
            "sangraha_blocks": 0,

            # Files completed within a manual source that is still in
            # progress, and the record counter that forms its LSH keys.
            "source_progress": {},
            "source_records": {},
        }

        all_stats = {}

    # ========================================================
    # LEDGER
    # ========================================================
    #
    # Appended when resuming, so the removals recorded before
    # the checkpoint are not lost.
    # ========================================================

    resuming = checkpoint is not None

    ledger_file = open(
        REMOVED_LEDGER,
        "a" if resuming else "w",
        encoding="utf-8",
        newline=""
    )

    ledger = csv.writer(
        ledger_file
    )

    if not resuming:

        ledger.writerow([
            "removed_source_group",
            "removed_source_type",
            "removed_file",
            "removed_record_number",
            "removed_doc_id",
            "removed_characters",
            "representative_key",
            "minhash_similarity",
        ])

    total_start = time.time()

    # ========================================================
    # IMPORTANT:
    #
    # Manual FIRST.
    #
    # This means Manual records become the protected
    # representatives.
    # ========================================================

    manual_sources = (
        get_manual_files()
    )

    for (
        source_group,
        source_type,
        source_dir,
        files
    ) in manual_sources:

        # A source finished before the checkpoint is already in the
        # index and already written to disk, so it is skipped rather
        # than reprocessed.
        if source_type in state["completed_sources"]:

            print()
            print(
                f"SKIPPING {source_type.upper()}: "
                f"already completed before checkpoint "
                f"({all_stats.get(source_type, {}).get('kept', 0):,} kept)"
            )

            continue

        # Carry forward the counts from a source that was interrupted
        # part-way through, so a resumed run reports totals for the whole
        # source rather than only for the files handled after the last
        # checkpoint.
        #
        # The lookup has to happen BEFORE all_stats is reassigned:
        # writing a fresh new_stats() into all_stats first would mean
        # reading straight back the empty dict just written.
        stats = all_stats.get(source_type) or new_stats()

        all_stats[source_type] = stats

        process_source(
            source_group,
            source_type,
            source_dir,
            files,
            MANUAL_OUTPUT,
            lsh,
            stats,
            ledger,
            candidate_stats,
            state,
            CHECKPOINT_FILE,
            all_stats,
        )

        state["completed_sources"].append(source_type)

        save_checkpoint(
            CHECKPOINT_FILE, lsh, candidate_stats, state, all_stats
        )

    # ========================================================
    # SANGRAHA SECOND
    # ========================================================
    #
    # Handled by its own resumable routine: it is a single file of
    # 1.03 million records taking over two hours, so it is written in
    # blocks with a checkpoint after each.
    # ========================================================

    if "sangraha" in state["completed_sources"]:

        print()
        print(
            f"SKIPPING SANGRAHA: already completed before checkpoint "
            f"({all_stats.get('sangraha', {}).get('kept', 0):,} kept)"
        )

    else:

        if not SANGRAHA_INPUT_FILE.exists():

            print()
            print("ERROR: Sangraha input missing:")
            print(SANGRAHA_INPUT_FILE)
            sys.exit(1)

        stats = all_stats.get("sangraha") or new_stats()

        all_stats["sangraha"] = stats

        process_sangraha(
            SANGRAHA_INPUT_FILE,
            SANGRAHA_OUTPUT,
            lsh,
            stats,
            ledger,
            candidate_stats,
            state,
            CHECKPOINT_FILE,
            all_stats,
        )

    # ========================================================
    # CLOSE LEDGER
    # ========================================================

    ledger_file.close()

    total_elapsed = (
        time.time()
        - total_start
    )

    # ========================================================
    # SUMMARY
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
            "source_type",
            "files",
            "records_processed",
            "records_kept",
            "near_duplicates_removed",
            "invalid",
            "characters_kept",
            "characters_removed",
        ])

        for source_type, stats in (
            all_stats.items()
        ):

            source_group = (
                "manual"
                if source_type
                != "sangraha"
                else "sangraha"
            )

            writer.writerow([
                source_group,
                source_type,
                stats["files"],
                stats["records"],
                stats["kept"],
                stats["removed"],
                stats["invalid"],
                stats["characters_kept"],
                stats["characters_removed"],
            ])

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("GLOBAL NEAR-DUPLICATE FILTER COMPLETE")
    print("=" * 80)

    total_records = 0
    total_kept = 0
    total_removed = 0
    total_chars = 0
    total_removed_chars = 0

    manual_chars = 0
    sangraha_chars = 0

    manual_records = 0
    sangraha_records = 0

    for source_type, stats in (
        all_stats.items()
    ):

        source_group = (
            "MANUAL"
            if source_type
            != "sangraha"
            else "SANGRAHA"
        )

        print()
        print(
            f"{source_group} / "
            f"{source_type.upper()}"
        )

        print("-" * 40)

        print(
            f"Files              : "
            f"{stats['files']:,}"
        )

        print(
            f"Records processed  : "
            f"{stats['records']:,}"
        )

        print(
            f"Records kept       : "
            f"{stats['kept']:,}"
        )

        print(
            f"Near duplicates    : "
            f"{stats['removed']:,}"
        )

        print(
            f"Invalid            : "
            f"{stats['invalid']:,}"
        )

        print(
            f"Characters kept    : "
            f"{stats['characters_kept']:,}"
        )

        print(
            f"Characters removed : "
            f"{stats['characters_removed']:,}"
        )

        total_records += (
            stats["records"]
        )

        total_kept += (
            stats["kept"]
        )

        total_removed += (
            stats["removed"]
        )

        total_chars += (
            stats["characters_kept"]
        )

        total_removed_chars += (
            stats["characters_removed"]
        )

        if source_type == "sangraha":

            sangraha_records += (
                stats["kept"]
            )

            sangraha_chars += (
                stats["characters_kept"]
            )

        else:

            manual_records += (
                stats["kept"]
            )

            manual_chars += (
                stats["characters_kept"]
            )

    print()
    print("=" * 80)

    print(
        "FINAL CORPUS"
    )

    print("-" * 40)

    print(
        f"Input records      : "
        f"{total_records:,}"
    )

    print(
        f"Final records      : "
        f"{total_kept:,}"
    )

    print(
        f"Near duplicates    : "
        f"{total_removed:,}"
    )

    print(
        f"Characters removed : "
        f"{total_removed_chars:,}"
    )

    print(
        f"Characters retained: "
        f"{total_chars:,}"
    )

    print()

    print(
        "SOURCE ACCOUNTING"
    )

    print("-" * 40)

    print(
        f"Manual records     : "
        f"{manual_records:,}"
    )

    print(
        f"Manual characters  : "
        f"{manual_chars:,}"
    )

    print(
        f"Sangraha records   : "
        f"{sangraha_records:,}"
    )

    print(
        f"Sangraha characters: "
        f"{sangraha_chars:,}"
    )

    if total_chars > 0:

        print(
            f"Manual proportion  : "
            f"{manual_chars / total_chars:.2%}"
        )

        print(
            f"Sangraha proportion: "
            f"{sangraha_chars / total_chars:.2%}"
        )

    print()

    print(
        f"Candidate comparisons: "
        f"{candidate_stats['candidates']:,}"
    )

    print(
        f"Elapsed time          : "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "OUTPUT"
    )

    print("-" * 40)

    print(
        f"Manual   : {MANUAL_OUTPUT}"
    )

    print(
        f"Sangraha : {SANGRAHA_OUTPUT}"
    )

    print(
        f"Ledger   : {REMOVED_LEDGER}"
    )

    print(
        f"Summary  : {SUMMARY_FILE}"
    )

    print()
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()