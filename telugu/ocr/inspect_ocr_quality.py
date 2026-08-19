import csv
import json
import random
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"G:\My Drive\LMA_Individual_Project"
)

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manual"
    / "textbooks"
)

FINAL_DIR = BASE_DIR / "ocr_final"
VALIDATION_DIR = BASE_DIR / "ocr_validation"

PAGE_REPORT = (
    VALIDATION_DIR
    / "ocr_page_quality.csv"
)

OUTPUT_FILE = (
    VALIDATION_DIR
    / "suspicious_pages_review.txt"
)

SAMPLE_PER_CATEGORY = 10

RANDOM_SEED = 42


# ============================================================
# CATEGORIES WE WANT TO INSPECT
# ============================================================

CATEGORIES = [
    "EMPTY",
    "VERY_LOW_TEXT",
    "LOW_TELUGU",
    "DIGIT_SYMBOL_HEAVY",
    "REVIEW",
    "MIXED",
]


# ============================================================
# LOAD PAGE QUALITY REPORT
# ============================================================

def load_quality_report():

    rows = []

    with open(
        PAGE_REPORT,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["page"] = int(
                row["page"]
            )

            row["char_count"] = int(
                row["char_count"]
            )

            row["telugu_ratio"] = float(
                row["telugu_ratio"]
            )

            row["digit_ratio"] = float(
                row["digit_ratio"]
            )

            rows.append(row)

    return rows


# ============================================================
# FIND CORRESPONDING JSONL
# ============================================================

def find_jsonl(book_name):

    # Most records use normalized filenames.
    # We first try direct stem matching.

    candidate = (
        FINAL_DIR
        / (Path(book_name).stem + ".jsonl")
    )

    if candidate.exists():

        return candidate

    # Otherwise search by normalized stem.

    target = normalize_name(
        Path(book_name).stem
    )

    for path in FINAL_DIR.glob(
        "*.jsonl"
    ):

        if normalize_name(
            path.stem
        ) == target:

            return path

    return None


# ============================================================
# NORMALIZE FILENAMES
# ============================================================

def normalize_name(name):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        name.lower()
    )


# ============================================================
# LOAD ALL JSONL RECORDS
# ============================================================

def load_records():

    cache = {}

    for path in FINAL_DIR.glob(
        "*.jsonl"
    ):

        records = {}

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():

                    continue

                try:

                    record = json.loads(
                        line
                    )

                    page = record.get(
                        "page"
                    )

                    if page is not None:

                        records[int(page)] = record

                except Exception:

                    continue

        cache[path] = records

    return cache


# ============================================================
# CLEAN TEXT FOR DISPLAY
# ============================================================

def clean_for_display(text):

    if not text:

        return "[EMPTY TEXT]"

    # Keep Unicode intact but avoid enormous whitespace blocks.

    text = re.sub(
        r"\n{4,}",
        "\n\n",
        text
    )

    text = text.strip()

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "TELUGU OCR — SUSPICIOUS PAGE INSPECTION"
    )
    print("=" * 80)

    print(
        f"Quality report:\n{PAGE_REPORT}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    if not PAGE_REPORT.exists():

        print(
            "\nERROR: Validation report not found."
        )

        return

    quality_rows = (
        load_quality_report()
    )

    print(
        f"\nQuality rows: "
        f"{len(quality_rows):,}"
    )

    # --------------------------------------------------------
    # Group by category
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for row in quality_rows:

        category = row[
            "diagnostic_label"
        ]

        grouped[
            category
        ].append(row)

    # --------------------------------------------------------
    # Load JSONL records
    # --------------------------------------------------------

    print(
        "\nLoading OCR JSONL records..."
    )

    record_cache = load_records()

    print(
        f"JSONL files loaded: "
        f"{len(record_cache)}"
    )

    # --------------------------------------------------------
    # Sample pages
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    selected = {}

    for category in CATEGORIES:

        rows = grouped.get(
            category,
            []
        )

        if len(rows) <= SAMPLE_PER_CATEGORY:

            sample = rows

        else:

            sample = random.sample(
                rows,
                SAMPLE_PER_CATEGORY
            )

        # Sort by book/page for easier reading.

        sample.sort(
            key=lambda x: (
                x["book"],
                x["page"]
            )
        )

        selected[
            category
        ] = sample

    # --------------------------------------------------------
    # Write inspection file
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        out.write(
            "=" * 80
            + "\n"
        )

        out.write(
            "TELUGU OCR — SUSPICIOUS PAGE "
            "INSPECTION\n"
        )

        out.write(
            "=" * 80
            + "\n\n"
        )

        out.write(
            "Purpose:\n"
        )

        out.write(
            "Inspect representative OCR pages "
            "before defining final cleaning rules.\n"
        )

        out.write(
            "No source JSONL files are modified.\n\n"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        out.write(
            "CATEGORY COUNTS\n"
        )

        out.write(
            "-" * 80
            + "\n"
        )

        for category in CATEGORIES:

            count = len(
                grouped.get(
                    category,
                    []
                )
            )

            out.write(
                f"{category:25s}"
                f"{count:8,}\n"
            )

        out.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        for category in CATEGORIES:

            samples = selected[
                category
            ]

            out.write(
                "=" * 80
                + "\n"
            )

            out.write(
                f"CATEGORY: {category}\n"
            )

            out.write(
                f"TOTAL PAGES: "
                f"{len(grouped.get(category, [])):,}\n"
            )

            out.write(
                f"SAMPLES SHOWN: "
                f"{len(samples):,}\n"
            )

            out.write(
                "=" * 80
                + "\n\n"
            )

            if not samples:

                out.write(
                    "No pages in this category.\n\n"
                )

                continue

            for i, row in enumerate(
                samples,
                start=1
            ):

                book = row[
                    "book"
                ]

                page = row[
                    "page"
                ]

                out.write(
                    "-" * 80
                    + "\n"
                )

                out.write(
                    f"SAMPLE {i}\n"
                )

                out.write(
                    f"Book          : "
                    f"{book}\n"
                )

                out.write(
                    f"Page          : "
                    f"{page}\n"
                )

                out.write(
                    f"Characters    : "
                    f"{row['char_count']:,}\n"
                )

                out.write(
                    f"Telugu ratio  : "
                    f"{float(row['telugu_ratio']) * 100:.2f}%\n"
                )

                out.write(
                    f"Digit ratio   : "
                    f"{float(row['digit_ratio']) * 100:.2f}%\n"
                )

                out.write(
                    f"Label         : "
                    f"{category}\n"
                )

                # --------------------------------------------
                # Find JSONL
                # --------------------------------------------

                jsonl_path = find_jsonl(
                    book
                )

                record = None

                if jsonl_path is not None:

                    records = record_cache.get(
                        jsonl_path,
                        {}
                    )

                    record = records.get(
                        page
                    )

                if record is None:

                    # Try searching all cached records
                    # using the book field.

                    for path, records in (
                        record_cache.items()
                    ):

                        candidate = records.get(
                            page
                        )

                        if (
                            candidate is not None
                            and candidate.get(
                                "book"
                            ) == book
                        ):

                            record = candidate

                            break

                if record is None:

                    out.write(
                        "\n[OCR RECORD NOT FOUND]\n"
                    )

                    continue

                text = record.get(
                    "text",
                    ""
                )

                out.write(
                    "\nOCR TEXT\n"
                )

                out.write(
                    "-" * 40
                    + "\n"
                )

                out.write(
                    clean_for_display(
                        text
                    )
                )

                out.write(
                    "\n\n"
                )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "INSPECTION FILE CREATED"
    )
    print("=" * 80)

    for category in CATEGORIES:

        total = len(
            grouped.get(
                category,
                []
            )
        )

        shown = len(
            selected.get(
                category,
                []
            )
        )

        print(
            f"{category:25s}"
            f"total={total:6,}"
            f"  samples={shown:2d}"
        )

    print()
    print(
        f"Review file:\n"
        f"{OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()