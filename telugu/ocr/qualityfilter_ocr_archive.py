from pathlib import Path
import json
import re
import csv
import unicodedata
import hashlib
import os

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed"
)

# Internet Archive JSONL
ARCHIVE_DIR = (
    BASE
    / "internet_archive"
    / "jsonl"
)

# Final textbook OCR JSONL
TEXTBOOK_DIR = (
    BASE
    / "manual"
    / "textbooks"
    / "ocr_final"
)

# New quality-filtered output
OUT_DIR = (
    BASE
    / "quality_filtered"
)

ARCHIVE_OUT = (
    OUT_DIR
    / "internet_archive"
)

TEXTBOOK_OUT = (
    OUT_DIR
    / "textbooks"
)

REPORT_DIR = (
    OUT_DIR
    / "reports"
)


# ============================================================
# SETTINGS
# ============================================================

# Internet Archive:
# Your original collection used 30% Telugu as the acceptance
# threshold, so we keep the same quality threshold here.
ARCHIVE_MIN_TELUGU = 0.30

# Textbook OCR:
# Textbooks legitimately contain English terms, numbers,
# headings, equations, etc.
TEXTBOOK_MIN_TELUGU = 0.20

# Minimum useful text.
ARCHIVE_MIN_CHARS = 1000
TEXTBOOK_MIN_CHARS = 300

# If more than this proportion of lines are repeated,
# consider the record suspicious.
MAX_REPEATED_LINE_RATIO = 0.50

# Existing textbook OCR metadata.
MAX_DIGIT_SYMBOL_RATIO = 0.65

# Windows multiprocessing.
WORKERS = 4


# ============================================================
# REGEX
# ============================================================

TELUGU_RE = re.compile(
    r"[\u0C00-\u0C7F]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)

DIGIT_RE = re.compile(
    r"\d"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    ARCHIVE_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    TEXTBOOK_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Normalize newlines
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove control characters except
    # newline and tab.
    text = "".join(
        c
        for c in text
        if (
            c in "\n\t"
            or not unicodedata.category(c).startswith("C")
        )
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{4,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CALCULATE TEXT STATISTICS
# ============================================================

def calculate_stats(text):

    text = normalize_text(text)

    nonspace = [
        c
        for c in text
        if not c.isspace()
    ]

    total = len(nonspace)

    if total == 0:

        return {
            "characters": 0,
            "telugu_characters": 0,
            "telugu_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "repeated_line_ratio": 0.0,
        }

    telugu = len(
        TELUGU_RE.findall(text)
    )

    latin = len(
        LATIN_RE.findall(text)
    )

    digits = len(
        DIGIT_RE.findall(text)
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    repeated_ratio = 0.0

    if len(lines) >= 10:

        counts = Counter(lines)

        repeated_lines = sum(
            count
            for count in counts.values()
            if count >= 3
        )

        repeated_ratio = (
            repeated_lines
            / len(lines)
        )

    return {
        "characters": total,
        "telugu_characters": telugu,
        "telugu_ratio": telugu / total,
        "latin_ratio": latin / total,
        "digit_ratio": digits / total,
        "repeated_line_ratio": repeated_ratio,
    }


# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(
    record,
    source
):

    # --------------------------------------------------------
    # Valid record
    # --------------------------------------------------------

    if not isinstance(
        record,
        dict
    ):

        return (
            False,
            "INVALID_RECORD",
            {}
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = record.get(
        "text",
        ""
    )

    if not isinstance(
        text,
        str
    ):

        return (
            False,
            "NO_TEXT",
            {}
        )

    text = normalize_text(text)

    if not text:

        return (
            False,
            "EMPTY",
            {}
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_stats(
        text
    )

    characters = stats[
        "characters"
    ]

    telugu_ratio = stats[
        "telugu_ratio"
    ]

    # --------------------------------------------------------
    # Minimum character threshold
    # --------------------------------------------------------

    if source == "archive":

        if characters < ARCHIVE_MIN_CHARS:

            return (
                False,
                "TOO_SHORT",
                stats
            )

    else:

        if characters < TEXTBOOK_MIN_CHARS:

            return (
                False,
                "TOO_SHORT",
                stats
            )

    # --------------------------------------------------------
    # Telugu ratio
    # --------------------------------------------------------

    if source == "archive":

        threshold = ARCHIVE_MIN_TELUGU

    else:

        threshold = TEXTBOOK_MIN_TELUGU

    if telugu_ratio < threshold:

        return (
            False,
            "LOW_TELUGU",
            stats
        )

    # --------------------------------------------------------
    # Repeated lines
    # --------------------------------------------------------

    if (
        stats["repeated_line_ratio"]
        > MAX_REPEATED_LINE_RATIO
    ):

        return (
            False,
            "HIGH_REPETITION",
            stats
        )

    # --------------------------------------------------------
    # Textbook OCR digit/symbol metadata
    # --------------------------------------------------------

    digit_symbol_ratio = record.get(
        "digit_symbol_ratio"
    )

    if digit_symbol_ratio is not None:

        try:

            digit_symbol_ratio = float(
                digit_symbol_ratio
            )

        except (
            ValueError,
            TypeError
        ):

            digit_symbol_ratio = None

    # Only reject if it is both symbol/digit-heavy
    # AND has low Telugu.
    #
    # This protects legitimate textbook pages
    # containing equations/numbers.
    if (
        digit_symbol_ratio is not None
        and
        digit_symbol_ratio
        > MAX_DIGIT_SYMBOL_RATIO
        and
        telugu_ratio < 0.30
    ):

        return (
            False,
            "OCR_GARBAGE",
            stats
        )

    # --------------------------------------------------------
    # Keep
    # --------------------------------------------------------

    record["text"] = text

    return (
        True,
        "KEEP",
        stats
    )


# ============================================================
# PROCESS ONE JSONL FILE
# ============================================================

def process_file(
    input_path,
    output_path,
    source
):

    kept = 0
    dropped = 0

    total_chars = 0
    total_telugu = 0

    rows = []

    # Duplicates within this file.
    seen_hashes = set()

    error = ""

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as fin, open(
            output_path,
            "w",
            encoding="utf-8"
        ) as fout:

            for line_number, line in enumerate(
                fin,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                # ------------------------------------------------
                # JSON parsing
                # ------------------------------------------------

                try:

                    record = json.loads(
                        line
                    )

                except Exception:

                    dropped += 1

                    rows.append({
                        "source": source,
                        "file": input_path.name,
                        "line": line_number,
                        "status": "DROP",
                        "reason": "INVALID_JSON",
                        "characters": 0,
                        "telugu_characters": 0,
                        "telugu_ratio": 0.0,
                        "latin_ratio": 0.0,
                        "digit_ratio": 0.0,
                        "repeated_line_ratio": 0.0,
                    })

                    continue

                # ------------------------------------------------
                # Quality check
                # ------------------------------------------------

                keep, reason, stats = quality_check(
                    record,
                    source
                )

                # ------------------------------------------------
                # Exact duplicate within file
                # ------------------------------------------------

                if keep:

                    text = record[
                        "text"
                    ]

                    text_hash = hashlib.sha1(
                        text.encode(
                            "utf-8"
                        )
                    ).hexdigest()

                    if text_hash in seen_hashes:

                        keep = False
                        reason = "DUPLICATE"

                    else:

                        seen_hashes.add(
                            text_hash
                        )

                # ------------------------------------------------
                # Keep
                # ------------------------------------------------

                if keep:

                    fout.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    kept += 1

                    total_chars += stats[
                        "characters"
                    ]

                    total_telugu += stats[
                        "telugu_characters"
                    ]

                # ------------------------------------------------
                # Drop
                # ------------------------------------------------

                else:

                    dropped += 1

                # ------------------------------------------------
                # Report row
                # ------------------------------------------------

                rows.append({
                    "source": source,
                    "file": input_path.name,
                    "line": line_number,
                    "status": (
                        "KEEP"
                        if keep
                        else "DROP"
                    ),
                    "reason": reason,
                    **stats,
                })

    except Exception as e:

        error = str(e)

    return {
        "source": source,
        "file": input_path.name,
        "error": error,
        "rows": rows,
        "kept": kept,
        "dropped": dropped,
        "characters": total_chars,
        "telugu": total_telugu,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 80)
    print("QUALITY FILTER")
    print("=" * 80)

    # ========================================================
    # DISCOVER ARCHIVE FILES
    # ========================================================

    archive_files = sorted(
        ARCHIVE_DIR.glob(
            "*.jsonl"
        )
    )

    # ========================================================
    # DISCOVER TEXTBOOK FILES
    # ========================================================

    textbook_files = sorted(
        TEXTBOOK_DIR.glob(
            "*.jsonl"
        )
    )

    print(
        f"Archive JSONL files : "
        f"{len(archive_files)}"
    )

    print(
        f"Textbook JSONL files: "
        f"{len(textbook_files)}"
    )

    total_files = (
        len(archive_files)
        + len(textbook_files)
    )

    print(
        f"Total files        : "
        f"{total_files}"
    )

    print(
        f"Workers            : "
        f"{WORKERS}"
    )

    print()

    if total_files == 0:

        raise RuntimeError(
            "No JSONL files found."
        )

    # ========================================================
    # BUILD JOBS
    # ========================================================

    jobs = []

    for path in archive_files:

        output_path = (
            ARCHIVE_OUT
            / path.name
        )

        jobs.append({
            "input": path,
            "output": output_path,
            "source": "archive",
        })

    for path in textbook_files:

        output_path = (
            TEXTBOOK_OUT
            / path.name
        )

        jobs.append({
            "input": path,
            "output": output_path,
            "source": "textbook",
        })

    # ========================================================
    # RUN MULTIPROCESSING
    # ========================================================

    results = []

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        future_map = {}

        for job in jobs:

            future = executor.submit(
                process_file,
                job["input"],
                job["output"],
                job["source"]
            )

            future_map[
                future
            ] = job

        completed = 0

        for future in as_completed(
            future_map
        ):

            job = future_map[
                future
            ]

            completed += 1

            try:

                result = future.result()

                results.append(
                    result
                )

                if result["error"]:

                    print(
                        f"[{completed}/{total_files}] "
                        f"ERROR   "
                        f"{job['source']:9s} "
                        f"{job['input'].name} "
                        f"-> "
                        f"{result['error']}"
                    )

                else:

                    print(
                        f"[{completed}/{total_files}] "
                        f"{job['source']:9s} "
                        f"{job['input'].name} "
                        f"| "
                        f"kept={result['kept']} "
                        f"dropped={result['dropped']}"
                    )

            except Exception as e:

                print(
                    f"[{completed}/{total_files}] "
                    f"ERROR   "
                    f"{job['source']:9s} "
                    f"{job['input'].name} "
                    f"-> {e}"
                )

    # ========================================================
    # REPORT
    # ========================================================

    report_path = (
        REPORT_DIR
        / "quality_filter_report.csv"
    )

    fieldnames = [
        "source",
        "file",
        "line",
        "status",
        "reason",
        "characters",
        "telugu_characters",
        "telugu_ratio",
        "latin_ratio",
        "digit_ratio",
        "repeated_line_ratio",
    ]

    with open(
        report_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for result in results:

            for row in result["rows"]:

                writer.writerow(
                    row
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    reason_counts = Counter()

    archive_kept = 0
    archive_dropped = 0
    archive_chars = 0
    archive_telugu = 0

    textbook_kept = 0
    textbook_dropped = 0
    textbook_chars = 0
    textbook_telugu = 0

    errors = []

    for result in results:

        if result["error"]:

            errors.append(
                (
                    result["source"],
                    result["file"],
                    result["error"]
                )
            )

        for row in result["rows"]:

            reason_counts[
                row["reason"]
            ] += 1

        if result["source"] == "archive":

            archive_kept += (
                result["kept"]
            )

            archive_dropped += (
                result["dropped"]
            )

            archive_chars += (
                result["characters"]
            )

            archive_telugu += (
                result["telugu"]
            )

        else:

            textbook_kept += (
                result["kept"]
            )

            textbook_dropped += (
                result["dropped"]
            )

            textbook_chars += (
                result["characters"]
            )

            textbook_telugu += (
                result["telugu"]
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("QUALITY FILTER COMPLETE")
    print("=" * 80)

    # --------------------------------------------------------
    # Archive
    # --------------------------------------------------------

    print()
    print("INTERNET ARCHIVE")
    print("-" * 40)

    print(
        f"Files processed     : "
        f"{len(archive_files):,}"
    )

    print(
        f"Records kept        : "
        f"{archive_kept:,}"
    )

    print(
        f"Records dropped     : "
        f"{archive_dropped:,}"
    )

    print(
        f"Characters          : "
        f"{archive_chars:,}"
    )

    print(
        f"Telugu characters   : "
        f"{archive_telugu:,}"
    )

    if archive_chars > 0:

        print(
            f"Telugu ratio        : "
            f"{archive_telugu / archive_chars:.2%}"
        )

    # --------------------------------------------------------
    # Textbooks
    # --------------------------------------------------------

    print()
    print("TEXTBOOK OCR")
    print("-" * 40)

    print(
        f"Files processed     : "
        f"{len(textbook_files):,}"
    )

    print(
        f"Records kept        : "
        f"{textbook_kept:,}"
    )

    print(
        f"Records dropped     : "
        f"{textbook_dropped:,}"
    )

    print(
        f"Characters          : "
        f"{textbook_chars:,}"
    )

    print(
        f"Telugu characters   : "
        f"{textbook_telugu:,}"
    )

    if textbook_chars > 0:

        print(
            f"Telugu ratio        : "
            f"{textbook_telugu / textbook_chars:.2%}"
        )

    # --------------------------------------------------------
    # Drop reasons
    # --------------------------------------------------------

    print()
    print("DROP REASONS")
    print("-" * 40)

    for reason, count in (
        reason_counts.most_common()
    ):

        print(
            f"{reason:22s}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    print()
    print(
        f"Processing errors   : "
        f"{len(errors):,}"
    )

    if errors:

        print()
        print("ERROR FILES")
        print("-" * 40)

        for source, filename, error in errors[:20]:

            print(
                f"{source:10s} "
                f"{filename}: "
                f"{error}"
            )

        if len(errors) > 20:

            print(
                f"... and "
                f"{len(errors) - 20} more"
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("OUTPUT")
    print("-" * 40)

    print(
        f"Archive filtered : "
        f"{ARCHIVE_OUT}"
    )

    print(
        f"Textbook filtered: "
        f"{TEXTBOOK_OUT}"
    )

    print(
        f"Report           : "
        f"{report_path}"
    )

    print()
    print("=" * 80)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()