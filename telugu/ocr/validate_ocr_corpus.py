import csv
import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"G:\My Drive\LMA_Individual_Project"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manual"
    / "textbooks"
)

FINAL_DIR = (
    PROCESSED_DIR
    / "ocr_final"
)

REPORT_DIR = (
    PROCESSED_DIR
    / "ocr_validation"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

PAGE_REPORT = (
    REPORT_DIR
    / "ocr_page_quality.csv"
)

BOOK_REPORT = (
    REPORT_DIR
    / "ocr_validation_report.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "ocr_quality_summary.txt"
)


# ============================================================
# REGEX
# ============================================================

TELUGU_RE = re.compile(
    r"[\u0C00-\u0C7F]"
)

DIGIT_RE = re.compile(
    r"[0-9౦-౯]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)

WHITESPACE_RE = re.compile(
    r"\s"
)

PUNCT_RE = re.compile(
    r"[^\w\s]",
    re.UNICODE
)


# ============================================================
# ANALYZE TEXT
# ============================================================

def analyze_text(text):

    chars = len(text)

    if chars == 0:

        return {
            "char_count": 0,
            "telugu_chars": 0,
            "telugu_ratio": 0.0,
            "digit_count": 0,
            "digit_ratio": 0.0,
            "latin_count": 0,
            "latin_ratio": 0.0,
            "punctuation_count": 0,
            "punctuation_ratio": 0.0,
            "whitespace_count": 0,
            "whitespace_ratio": 0.0,
        }

    telugu = len(
        TELUGU_RE.findall(text)
    )

    digits = len(
        DIGIT_RE.findall(text)
    )

    latin = len(
        LATIN_RE.findall(text)
    )

    punctuation = len(
        PUNCT_RE.findall(text)
    )

    whitespace = len(
        WHITESPACE_RE.findall(text)
    )

    return {
        "char_count": chars,

        "telugu_chars": telugu,

        "telugu_ratio": round(
            telugu / chars,
            6
        ),

        "digit_count": digits,

        "digit_ratio": round(
            digits / chars,
            6
        ),

        "latin_count": latin,

        "latin_ratio": round(
            latin / chars,
            6
        ),

        "punctuation_count": punctuation,

        "punctuation_ratio": round(
            punctuation / chars,
            6
        ),

        "whitespace_count": whitespace,

        "whitespace_ratio": round(
            whitespace / chars,
            6
        ),
    }


# ============================================================
# BASIC PAGE QUALITY LABEL
#
# IMPORTANT:
# These are NOT final cleaning rules.
# They are only diagnostic labels.
# ============================================================

def diagnostic_label(stats):

    chars = stats[
        "char_count"
    ]

    telugu_ratio = stats[
        "telugu_ratio"
    ]

    digit_ratio = stats[
        "digit_ratio"
    ]

    if chars == 0:

        return "EMPTY"

    if chars < 50:

        return "VERY_LOW_TEXT"

    if (
        telugu_ratio >= 0.70
    ):

        return "GOOD_TELUGU"

    if (
        telugu_ratio >= 0.40
    ):

        return "MIXED"

    if (
        digit_ratio >= 0.35
        and telugu_ratio < 0.40
    ):

        return "DIGIT_SYMBOL_HEAVY"

    if (
        telugu_ratio < 0.20
    ):

        return "LOW_TELUGU"

    return "REVIEW"


# ============================================================
# LOAD ONE JSONL
# ============================================================

def load_jsonl(path):

    records = []

    malformed = 0

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():

                continue

            try:

                record = json.loads(
                    line
                )

                records.append(
                    record
                )

            except Exception:

                malformed += 1

    return records, malformed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "TELUGU OCR CORPUS VALIDATION"
    )
    print("=" * 80)

    print(
        f"Input directory:\n"
        f"{FINAL_DIR}"
    )

    print(
        f"Report directory:\n"
        f"{REPORT_DIR}"
    )

    # --------------------------------------------------------
    # Find JSONL files
    # --------------------------------------------------------

    jsonl_files = sorted(
        FINAL_DIR.glob(
            "*.jsonl"
        )
    )

    print()
    print(
        f"JSONL files found: "
        f"{len(jsonl_files)}"
    )

    if not jsonl_files:

        print(
            "ERROR: No JSONL files found."
        )

        return

    # --------------------------------------------------------
    # Report rows
    # --------------------------------------------------------

    page_rows = []

    book_rows = []

    total_pages = 0

    total_chars = 0

    total_telugu = 0

    total_digits = 0

    total_malformed = 0

    global_labels = Counter()

    # --------------------------------------------------------
    # Process books
    # --------------------------------------------------------

    for index, jsonl_path in enumerate(
        jsonl_files,
        start=1
    ):

        print()
        print(
            f"[{index:02d}/"
            f"{len(jsonl_files):02d}] "
            f"{jsonl_path.name}"
        )

        records, malformed = (
            load_jsonl(
                jsonl_path
            )
        )

        total_malformed += malformed

        pages = len(records)

        chars = 0

        telugu_chars = 0

        digits = 0

        empty_pages = 0

        very_low_pages = 0

        good_pages = 0

        mixed_pages = 0

        low_telugu_pages = 0

        digit_heavy_pages = 0

        review_pages = 0

        page_numbers = []

        for record in records:

            page = record.get(
                "page"
            )

            text = record.get(
                "text",
                ""
            )

            stats = analyze_text(
                text
            )

            label = diagnostic_label(
                stats
            )

            page_numbers.append(
                page
            )

            chars += stats[
                "char_count"
            ]

            telugu_chars += stats[
                "telugu_chars"
            ]

            digits += stats[
                "digit_count"
            ]

            total_pages += 1

            total_chars += stats[
                "char_count"
            ]

            total_telugu += stats[
                "telugu_chars"
            ]

            total_digits += stats[
                "digit_count"
            ]

            global_labels[
                label
            ] += 1

            if label == "EMPTY":
                empty_pages += 1

            elif label == "VERY_LOW_TEXT":
                very_low_pages += 1

            elif label == "GOOD_TELUGU":
                good_pages += 1

            elif label == "MIXED":
                mixed_pages += 1

            elif label == "LOW_TELUGU":
                low_telugu_pages += 1

            elif label == "DIGIT_SYMBOL_HEAVY":
                digit_heavy_pages += 1

            elif label == "REVIEW":
                review_pages += 1

            page_rows.append(
                {
                    "book": record.get(
                        "book",
                        jsonl_path.stem
                    ),
                    "page": page,
                    "char_count":
                        stats[
                            "char_count"
                        ],
                    "telugu_chars":
                        stats[
                            "telugu_chars"
                        ],
                    "telugu_ratio":
                        stats[
                            "telugu_ratio"
                        ],
                    "digit_count":
                        stats[
                            "digit_count"
                        ],
                    "digit_ratio":
                        stats[
                            "digit_ratio"
                        ],
                    "latin_count":
                        stats[
                            "latin_count"
                        ],
                    "latin_ratio":
                        stats[
                            "latin_ratio"
                        ],
                    "punctuation_ratio":
                        stats[
                            "punctuation_ratio"
                        ],
                    "diagnostic_label":
                        label,
                    "status":
                        record.get(
                            "status",
                            ""
                        )
                }
            )

        # ----------------------------------------------------
        # Missing pages
        # ----------------------------------------------------

        missing_pages = 0

        if page_numbers:

            page_numbers_clean = [
                int(p)
                for p in page_numbers
                if p is not None
            ]

            minimum = min(
                page_numbers_clean
            )

            maximum = max(
                page_numbers_clean
            )

            expected = set(
                range(
                    minimum,
                    maximum + 1
                )
            )

            actual = set(
                page_numbers_clean
            )

            missing_pages = len(
                expected - actual
            )

        else:

            minimum = ""

            maximum = ""

        # ----------------------------------------------------
        # Book-level ratios
        # ----------------------------------------------------

        telugu_ratio = (
            telugu_chars / chars
            if chars
            else 0
        )

        digit_ratio = (
            digits / chars
            if chars
            else 0
        )

        book_rows.append(
            {
                "filename":
                    jsonl_path.name,

                "pages":
                    pages,

                "minimum_page":
                    minimum,

                "maximum_page":
                    maximum,

                "missing_pages":
                    missing_pages,

                "malformed_records":
                    malformed,

                "characters":
                    chars,

                "telugu_characters":
                    telugu_chars,

                "overall_telugu_ratio":
                    round(
                        telugu_ratio,
                        6
                    ),

                "digits":
                    digits,

                "digit_ratio":
                    round(
                        digit_ratio,
                        6
                    ),

                "empty_pages":
                    empty_pages,

                "very_low_text_pages":
                    very_low_pages,

                "good_telugu_pages":
                    good_pages,

                "mixed_pages":
                    mixed_pages,

                "low_telugu_pages":
                    low_telugu_pages,

                "digit_symbol_heavy_pages":
                    digit_heavy_pages,

                "review_pages":
                    review_pages
            }
        )

        print(
            f"  Pages : {pages:,}"
        )

        print(
            f"  Chars : {chars:,}"
        )

        print(
            f"  Telugu: "
            f"{telugu_ratio * 100:.2f}%"
        )

        print(
            f"  Good  : {good_pages:,}"
        )

        print(
            f"  Review: "
            f"{(
                mixed_pages
                + low_telugu_pages
                + digit_heavy_pages
                + review_pages
            ):,}"
        )

    # ========================================================
    # SAVE PAGE REPORT
    # ========================================================

    page_fields = [
        "book",
        "page",
        "char_count",
        "telugu_chars",
        "telugu_ratio",
        "digit_count",
        "digit_ratio",
        "latin_count",
        "latin_ratio",
        "punctuation_ratio",
        "diagnostic_label",
        "status"
    ]

    with open(
        PAGE_REPORT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=page_fields
        )

        writer.writeheader()

        writer.writerows(
            page_rows
        )

    # ========================================================
    # SAVE BOOK REPORT
    # ========================================================

    book_fields = [
        "filename",
        "pages",
        "minimum_page",
        "maximum_page",
        "missing_pages",
        "malformed_records",
        "characters",
        "telugu_characters",
        "overall_telugu_ratio",
        "digits",
        "digit_ratio",
        "empty_pages",
        "very_low_text_pages",
        "good_telugu_pages",
        "mixed_pages",
        "low_telugu_pages",
        "digit_symbol_heavy_pages",
        "review_pages"
    ]

    with open(
        BOOK_REPORT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=book_fields
        )

        writer.writeheader()

        writer.writerows(
            book_rows
        )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    global_telugu_ratio = (
        total_telugu / total_chars
        if total_chars
        else 0
    )

    global_digit_ratio = (
        total_digits / total_chars
        if total_chars
        else 0
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "TELUGU OCR CORPUS "
            "VALIDATION SUMMARY\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"JSONL files              : "
            f"{len(jsonl_files):,}\n"
        )

        f.write(
            f"Total pages              : "
            f"{total_pages:,}\n"
        )

        f.write(
            f"Total characters         : "
            f"{total_chars:,}\n"
        )

        f.write(
            f"Total Telugu characters  : "
            f"{total_telugu:,}\n"
        )

        f.write(
            f"Overall Telugu ratio     : "
            f"{global_telugu_ratio * 100:.2f}%\n"
        )

        f.write(
            f"Total digits             : "
            f"{total_digits:,}\n"
        )

        f.write(
            f"Overall digit ratio      : "
            f"{global_digit_ratio * 100:.2f}%\n"
        )

        f.write(
            f"Malformed JSONL records  : "
            f"{total_malformed:,}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "PAGE DIAGNOSTIC LABELS\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for label, count in (
            global_labels.most_common()
        ):

            percentage = (
                count / total_pages * 100
                if total_pages
                else 0
            )

            f.write(
                f"{label:25s}"
                f"{count:10,}"
                f" "
                f"{percentage:7.2f}%\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "IMPORTANT\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            "The diagnostic labels in this "
            "report are NOT final cleaning rules.\n"
        )

        f.write(
            "They are used only to understand "
            "the OCR quality distribution.\n"
        )

        f.write(
            "No source JSONL files were modified "
            "by this validation process.\n"
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VALIDATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"JSONL files       : "
        f"{len(jsonl_files):,}"
    )

    print(
        f"Total pages       : "
        f"{total_pages:,}"
    )

    print(
        f"Total characters  : "
        f"{total_chars:,}"
    )

    print(
        f"Telugu characters : "
        f"{total_telugu:,}"
    )

    print(
        f"Overall Telugu    : "
        f"{global_telugu_ratio * 100:.2f}%"
    )

    print()
    print(
        "PAGE LABELS"
    )

    print(
        "-" * 80
    )

    for label, count in (
        global_labels.most_common()
    ):

        percentage = (
            count / total_pages * 100
            if total_pages
            else 0
        )

        print(
            f"{label:25s}"
            f"{count:10,}"
            f" "
            f"{percentage:7.2f}%"
        )

    print()
    print(
        "Reports:"
    )

    print(
        PAGE_REPORT
    )

    print(
        BOOK_REPORT
    )

    print(
        SUMMARY_FILE
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()