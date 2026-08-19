import csv
import json
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"G:\My Drive\LMA_Individual_Project"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manual"
    / "textbooks"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manual"
    / "textbooks"
)

MANIFEST_FILE = (
    PROCESSED_DIR
    / "textbook_processing_manifest.csv"
)

CHECKPOINT_DIR = (
    PROCESSED_DIR
    / "ocr_checkpoints"
)

FINAL_DIR = (
    PROCESSED_DIR
    / "ocr_final"
)

LOG_DIR = (
    PROCESSED_DIR
    / "ocr_logs"
)

NUM_WORKERS = 4

BATCH_SIZE = 20

OCR_LANGUAGE = "tel"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_name(filename):

    name = Path(filename).stem

    name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name
    )

    return name.strip("_")


def checkpoint_path(pdf_name):

    return (
        CHECKPOINT_DIR
        / f"{safe_name(pdf_name)}.jsonl"
    )


def final_path(pdf_name):

    return (
        FINAL_DIR
        / f"{safe_name(pdf_name)}.jsonl"
    )


# ============================================================
# TEXT STATISTICS
# ============================================================

def analyze_text(text):

    char_count = len(text)

    if char_count == 0:

        return (
            0,
            0.0,
            0.0
        )

    telugu_chars = len(
        re.findall(
            r"[\u0C00-\u0C7F]",
            text
        )
    )

    digit_symbol_chars = len(
        re.findall(
            r"[0-9౦-౯\s\W_]",
            text,
            flags=re.UNICODE
        )
    )

    telugu_ratio = (
        telugu_chars
        / char_count
    )

    digit_symbol_ratio = (
        digit_symbol_chars
        / char_count
    )

    return (
        char_count,
        round(
            telugu_ratio,
            6
        ),
        round(
            digit_symbol_ratio,
            6
        )
    )


# ============================================================
# LOAD EXISTING CHECKPOINT
#
# IMPORTANT:
# Existing Colab checkpoints use:
#
# status = CANDIDATE
#
# We consider a page processed when it has:
#
# page + text
#
# regardless of whether it came from Colab or laptop.
# ============================================================

def load_checkpoint(pdf_name):

    path = checkpoint_path(
        pdf_name
    )

    records = {}

    if not path.exists():

        return records

    try:

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

                    page = int(
                        record["page"]
                    )

                    text = record.get(
                        "text",
                        ""
                    )

                    # A page is considered
                    # successfully OCR'd if
                    # a page record exists.
                    if (
                        page >= 1
                        and isinstance(
                            text,
                            str
                        )
                    ):

                        records[page] = record

                except Exception:

                    continue

    except Exception as e:

        print(
            f"WARNING: Could not read "
            f"checkpoint:\n{path}\n{e}"
        )

    return records


# ============================================================
# OCR WORKER
# ============================================================

def process_page(args):

    pdf_path, page_number = args

    start = time.perf_counter()

    try:

        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

        doc = pymupdf.open(
            pdf_path
        )

        page = doc[
            page_number - 1
        ]

        # ----------------------------------------------------
        # Render page
        # ----------------------------------------------------

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(
                2,
                2
            ),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [
                pix.width,
                pix.height
            ],
            pix.samples
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE
        )

        doc.close()

        text = text.strip()

        (
            char_count,
            telugu_ratio,
            digit_symbol_ratio
        ) = analyze_text(
            text
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # USE SAME FORMAT AS COLAB
        # ----------------------------------------------------

        return {
            "doc_id": None,
            "source": "manual_textbooks",
            "book": Path(
                pdf_path
            ).name,
            "page": page_number,
            "text": text,
            "char_count": char_count,
            "telugu_ratio": telugu_ratio,
            "digit_symbol_ratio": digit_symbol_ratio,
            "status": "CANDIDATE",
            "ocr_time_seconds": round(
                elapsed,
                4
            )
        }

    except Exception as e:

        return {
            "doc_id": None,
            "source": "manual_textbooks",
            "book": Path(
                pdf_path
            ).name,
            "page": page_number,
            "text": "",
            "char_count": 0,
            "telugu_ratio": 0.0,
            "digit_symbol_ratio": 0.0,
            "status": "FAILED",
            "error": repr(e)
        }


# ============================================================
# SAVE RECORDS
# ============================================================

def append_records(
    pdf_name,
    records
):

    path = checkpoint_path(
        pdf_name
    )

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# CREATE FINAL JSONL
# ============================================================

def create_final_file(
    pdf_name,
    records,
    expected_pages
):

    output_path = final_path(
        pdf_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for page in range(
            1,
            expected_pages + 1
        ):

            if page not in records:

                continue

            record = records[
                page
            ]

            # Remove runtime-only timing
            # from final corpus if present.
            final_record = dict(
                record
            )

            final_record.pop(
                "ocr_time_seconds",
                None
            )

            f.write(
                json.dumps(
                    final_record,
                    ensure_ascii=False
                )
                + "\n"
            )

    return output_path


# ============================================================
# PROCESS ONE BOOK
# ============================================================

def process_book(
    pdf_name,
    expected_pages
):

    pdf_path = (
        RAW_DIR
        / pdf_name
    )

    print()
    print("=" * 80)
    print(
        f"BOOK: {pdf_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not pdf_path.exists():

        print(
            "ERROR: PDF not found:"
        )

        print(
            pdf_path
        )

        return {
            "filename": pdf_name,
            "status": "SOURCE_MISSING",
            "pages": expected_pages,
            "processed": 0,
            "failed": expected_pages
        }

    # --------------------------------------------------------
    # Get actual PDF pages
    # --------------------------------------------------------

    try:

        doc = pymupdf.open(
            str(pdf_path)
        )

        actual_pages = len(
            doc
        )

        doc.close()

    except Exception as e:

        print(
            f"ERROR opening PDF: {e}"
        )

        return {
            "filename": pdf_name,
            "status": "PDF_ERROR",
            "pages": expected_pages,
            "processed": 0,
            "failed": expected_pages
        }

    if actual_pages != expected_pages:

        print(
            f"WARNING:"
        )

        print(
            f"Manifest pages = "
            f"{expected_pages}"
        )

        print(
            f"Actual PDF pages = "
            f"{actual_pages}"
        )

        expected_pages = actual_pages

    # --------------------------------------------------------
    # Existing checkpoint
    # --------------------------------------------------------

    records = load_checkpoint(
        pdf_name
    )

    processed_pages = set(
        records.keys()
    )

    print(
        f"Expected pages : "
        f"{expected_pages:,}"
    )

    print(
        f"Existing pages : "
        f"{len(processed_pages):,}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    if len(processed_pages) >= expected_pages:

        print(
            "STATUS: COMPLETE"
        )

        print(
            "No OCR required."
        )

        output = create_final_file(
            pdf_name,
            records,
            expected_pages
        )

        print(
            f"Final JSONL:\n{output}"
        )

        return {
            "filename": pdf_name,
            "status": "ALREADY_COMPLETE",
            "pages": expected_pages,
            "processed": expected_pages,
            "failed": 0
        }

    # --------------------------------------------------------
    # Remaining pages
    # --------------------------------------------------------

    remaining_pages = [
        page
        for page in range(
            1,
            expected_pages + 1
        )
        if page not in processed_pages
    ]

    print(
        f"Remaining pages: "
        f"{len(remaining_pages):,}"
    )

    if processed_pages:

        print(
            f"Resume point: "
            f"{min(remaining_pages)}"
        )

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    total_start = (
        time.perf_counter()
    )

    total_batches = (
        (
            len(remaining_pages)
            + BATCH_SIZE
            - 1
        )
        // BATCH_SIZE
    )

    for batch_index, batch_start in enumerate(
        range(
            0,
            len(remaining_pages),
            BATCH_SIZE
        ),
        start=1
    ):

        batch_pages = remaining_pages[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        print()
        print("-" * 80)

        print(
            f"Batch "
            f"{batch_index}/"
            f"{total_batches}"
        )

        print(
            f"Pages "
            f"{batch_pages[0]}-"
            f"{batch_pages[-1]}"
        )

        jobs = [
            (
                str(pdf_path),
                page
            )
            for page in batch_pages
        ]

        batch_results = []

        batch_start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Parallel processing
        # ----------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    process_page,
                    job
                ): job[1]
                for job in jobs
            }

            completed = 0

            for future in as_completed(
                futures
            ):

                result = future.result()

                batch_results.append(
                    result
                )

                completed += 1

                if (
                    result["status"]
                    == "CANDIDATE"
                ):

                    print(
                        f"[{completed:02d}/"
                        f"{len(batch_pages):02d}] "
                        f"Page "
                        f"{result['page']:04d} | "
                        f"chars="
                        f"{result['char_count']:5,} | "
                        f"Telugu="
                        f"{result['telugu_ratio'] * 100:6.2f}% | "
                        f"time="
                        f"{result['ocr_time_seconds']:.2f}s"
                    )

                else:

                    print(
                        f"[{completed:02d}/"
                        f"{len(batch_pages):02d}] "
                        f"Page "
                        f"{result['page']:04d} | "
                        f"FAILED"
                    )

        # ----------------------------------------------------
        # Sort results
        # ----------------------------------------------------

        batch_results.sort(
            key=lambda x: x["page"]
        )

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        append_records(
            pdf_name,
            batch_results
        )

        # ----------------------------------------------------
        # Update memory
        # ----------------------------------------------------

        for result in batch_results:

            if (
                result["status"]
                == "CANDIDATE"
            ):

                records[
                    result["page"]
                ] = result

        batch_elapsed = (
            time.perf_counter()
            - batch_start_time
        )

        successful = len(
            records
        )

        print()
        print(
            f"Batch time      : "
            f"{batch_elapsed:.2f}s"
        )

        print(
            f"Checkpoint saved"
        )

        print(
            f"Progress        : "
            f"{successful}/"
            f"{expected_pages}"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    successful_pages = len(
        records
    )

    failed_pages = (
        expected_pages
        - successful_pages
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    if failed_pages == 0:

        final_output = create_final_file(
            pdf_name,
            records,
            expected_pages
        )

        status = "COMPLETE"

        print()
        print(
            "=" * 80
        )

        print(
            f"BOOK COMPLETE: "
            f"{pdf_name}"
        )

        print(
            f"Pages: "
            f"{successful_pages}/"
            f"{expected_pages}"
        )

        print(
            f"Time: "
            f"{total_elapsed / 60:.2f} minutes"
        )

        print(
            f"Final JSONL:\n"
            f"{final_output}"
        )

    else:

        status = "PARTIAL"

        print()
        print(
            "BOOK PARTIAL"
        )

        print(
            f"Processed: "
            f"{successful_pages}/"
            f"{expected_pages}"
        )

        print(
            "The checkpoint is safe."
        )

    return {
        "filename": pdf_name,
        "status": status,
        "pages": expected_pages,
        "processed": successful_pages,
        "failed": failed_pages
    }


# ============================================================
# LOAD PROCESSING MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{MANIFEST_FILE}\n"
        )

    books = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            process_value = (
                row.get(
                    "process",
                    ""
                )
                .strip()
                .upper()
            )

            if process_value != "YES":

                continue

            filename = (
                row.get(
                    "filename",
                    ""
                )
                .strip()
            )

            if not filename:

                continue

            try:

                pages = int(
                    row.get(
                        "pages",
                        0
                    )
                )

            except:

                pages = 0

            books.append(
                {
                    "filename": filename,
                    "pages": pages,
                    "source_group":
                        row.get(
                            "source_group",
                            ""
                        )
                }
            )

    return books


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "TELUGU LOW-RESOURCE CORPUS"
    )
    print(
        "RESUMABLE 71-BOOK OCR"
    )
    print("=" * 80)

    print(
        f"Raw directory:\n"
        f"{RAW_DIR}"
    )

    print(
        f"Checkpoint directory:\n"
        f"{CHECKPOINT_DIR}"
    )

    print(
        f"Final directory:\n"
        f"{FINAL_DIR}"
    )

    print(
        f"Workers     : "
        f"{NUM_WORKERS}"
    )

    print(
        f"Batch size  : "
        f"{BATCH_SIZE}"
    )

    print(
        f"OCR language: "
        f"{OCR_LANGUAGE}"
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    books = load_manifest()

    print()
    print(
        f"Books selected from manifest: "
        f"{len(books)}"
    )

    # --------------------------------------------------------
    # Production results
    # --------------------------------------------------------

    results = []

    overall_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Process books sequentially
    # --------------------------------------------------------

    for index, book in enumerate(
        books,
        start=1
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"BOOK {index}/"
            f"{len(books)}"
        )

        print(
            book["filename"]
        )

        print(
            "#" * 80
        )

        result = process_book(
            book["filename"],
            book["pages"]
        )

        result[
            "source_group"
        ] = book[
            "source_group"
        ]

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save production manifest
    # --------------------------------------------------------

    production_manifest = (
        PROCESSED_DIR
        / "ocr_production_manifest.csv"
    )

    with open(
        production_manifest,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        fieldnames = [
            "filename",
            "source_group",
            "status",
            "pages",
            "processed",
            "failed"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    field:
                    result.get(
                        field,
                        ""
                    )
                    for field in fieldnames
                }
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    complete = sum(
        1
        for r in results
        if r["status"]
        in (
            "COMPLETE",
            "ALREADY_COMPLETE"
        )
    )

    partial = sum(
        1
        for r in results
        if r["status"]
        == "PARTIAL"
    )

    missing = sum(
        1
        for r in results
        if r["status"]
        == "SOURCE_MISSING"
    )

    print()
    print("=" * 80)
    print(
        "PRODUCTION OCR RUN COMPLETE"
    )
    print("=" * 80)

    print(
        f"Books selected : "
        f"{len(books)}"
    )

    print(
        f"Books complete : "
        f"{complete}"
    )

    print(
        f"Books partial  : "
        f"{partial}"
    )

    print(
        f"Missing PDFs   : "
        f"{missing}"
    )

    print(
        f"Wall time      : "
        f"{elapsed / 3600:.2f} hours"
    )

    print()
    print(
        "Production manifest:"
    )

    print(
        production_manifest
    )

    print("=" * 80)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()