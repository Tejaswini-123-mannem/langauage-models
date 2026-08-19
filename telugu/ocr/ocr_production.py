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

NUM_WORKERS = 4
BATCH_SIZE = 20
OCR_LANGUAGE = "tel"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
# WORKER FUNCTION
# ============================================================

def process_page(args):

    pdf_path, page_index = args

    try:

        start = time.perf_counter()

        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

        doc = pymupdf.open(pdf_path)

        page = doc[page_index]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE
        )

        doc.close()

        elapsed = time.perf_counter() - start

        text = text.strip()

        chars = len(text)

        telugu_chars = len(
            re.findall(
                r"[\u0C00-\u0C7F]",
                text
            )
        )

        telugu_ratio = (
            telugu_chars / chars
            if chars > 0
            else 0
        )

        return {
            "page": page_index + 1,
            "text": text,
            "chars": chars,
            "telugu_chars": telugu_chars,
            "telugu_ratio": round(
                telugu_ratio,
                4
            ),
            "ocr_time_seconds": round(
                elapsed,
                3
            ),
            "status": "success",
            "error": None
        }

    except Exception as e:

        return {
            "page": page_index + 1,
            "text": "",
            "chars": 0,
            "telugu_chars": 0,
            "telugu_ratio": 0,
            "ocr_time_seconds": 0,
            "status": "failed",
            "error": repr(e)
        }


# ============================================================
# SAFE NAME
# ============================================================

def safe_name(filename):

    name = Path(filename).stem

    name = re.sub(
        r"[^\w.-]+",
        "_",
        name,
        flags=re.UNICODE
    )

    return name.strip("_")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    base_dir = Path(__file__).resolve().parent

    input_dir = base_dir / "input"
    checkpoint_dir = base_dir / "checkpoints"
    final_dir = base_dir / "final"
    log_dir = base_dir / "logs"

    for directory in [
        checkpoint_dir,
        final_dir,
        log_dir
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # FIND PDF
    # --------------------------------------------------------

    pdf_files = sorted(
        input_dir.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF found in:\n{input_dir}"
        )

    if len(pdf_files) > 1:

        print(
            "WARNING: Multiple PDFs found."
        )

        print(
            "Using:",
            pdf_files[0].name
        )

    pdf_path = pdf_files[0]

    book_name = safe_name(
        pdf_path.name
    )

    book_checkpoint_dir = (
        checkpoint_dir / book_name
    )

    book_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # PDF INFO
    # --------------------------------------------------------

    doc = pymupdf.open(pdf_path)

    total_pages = len(doc)

    doc.close()

    print()
    print("=" * 80)
    print("PRODUCTION OCR — ONE BOOK TEST")
    print("=" * 80)

    print(
        f"Book           : {pdf_path.name}"
    )

    print(
        f"Pages          : {total_pages:,}"
    )

    print(
        f"Workers        : {NUM_WORKERS}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Checkpoint dir : {book_checkpoint_dir}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # FIND EXISTING CHECKPOINTS
    # --------------------------------------------------------

    completed_pages = set()

    for checkpoint_file in sorted(
        book_checkpoint_dir.glob(
            "batch_*.jsonl"
        )
    ):

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                record = json.loads(line)

                if record["status"] == "success":

                    completed_pages.add(
                        record["page"]
                    )

    print(
        f"Already completed: "
        f"{len(completed_pages):,} pages"
    )

    # --------------------------------------------------------
    # OCR BATCHES
    # --------------------------------------------------------

    overall_start = time.perf_counter()

    for batch_start in range(
        0,
        total_pages,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_pages
        )

        batch_number = (
            batch_start // BATCH_SIZE
        )

        checkpoint_file = (
            book_checkpoint_dir
            / f"batch_{batch_number:04d}.jsonl"
        )

        # Existing completed checkpoint
        if checkpoint_file.exists():

            print(
                f"Batch {batch_number:04d} "
                f"| already exists | SKIP"
            )

            continue

        jobs = []

        for page_index in range(
            batch_start,
            batch_end
        ):

            page_number = page_index + 1

            if page_number not in completed_pages:

                jobs.append(
                    (
                        str(pdf_path),
                        page_index
                    )
                )

        if not jobs:
            continue

        print()
        print(
            f"Batch {batch_number:04d} "
            f"| pages "
            f"{batch_start + 1}-{batch_end}"
        )

        batch_start_time = time.perf_counter()

        results = []

        # ----------------------------------------------------
        # PARALLEL OCR
        # ----------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    process_page,
                    job
                ): job[1]
                for job in jobs
            }

            for future in as_completed(
                future_map
            ):

                result = future.result()

                results.append(result)

                if result["status"] == "success":

                    print(
                        f"  Page "
                        f"{result['page']:04d} "
                        f"| chars="
                        f"{result['chars']:,} "
                        f"| Telugu="
                        f"{result['telugu_ratio']:.2%}"
                    )

                else:

                    print(
                        f"  Page "
                        f"{result['page']:04d} "
                        f"| FAILED"
                    )

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        results.sort(
            key=lambda x: x["page"]
        )

        # ----------------------------------------------------
        # ATOMIC CHECKPOINT
        # ----------------------------------------------------

        temp_file = checkpoint_file.with_suffix(
            ".tmp"
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            for result in results:

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        temp_file.replace(
            checkpoint_file
        )

        batch_elapsed = (
            time.perf_counter()
            - batch_start_time
        )

        successful = sum(
            r["status"] == "success"
            for r in results
        )

        print(
            f"Batch complete | "
            f"{successful}/{len(results)} successful | "
            f"{batch_elapsed:.2f}s"
        )

    # --------------------------------------------------------
    # MERGE CHECKPOINTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MERGING CHECKPOINTS")
    print("=" * 80)

    all_records = []

    for checkpoint_file in sorted(
        book_checkpoint_dir.glob(
            "batch_*.jsonl"
        )
    ):

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.strip():

                    all_records.append(
                        json.loads(line)
                    )

    all_records.sort(
        key=lambda x: x["page"]
    )

    # --------------------------------------------------------
    # FINAL JSONL
    # --------------------------------------------------------

    final_file = (
        final_dir
        / f"{book_name}.jsonl"
    )

    with open(
        final_file,
        "w",
        encoding="utf-8"
    ) as f:

        for record in all_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # FAILED PAGES
    # --------------------------------------------------------

    failed_file = (
        log_dir
        / f"{book_name}_failed.jsonl"
    )

    with open(
        failed_file,
        "w",
        encoding="utf-8"
    ) as f:

        for record in all_records:

            if record["status"] == "failed":

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    successful_pages = sum(
        r["status"] == "success"
        for r in all_records
    )

    failed_pages = sum(
        r["status"] == "failed"
        for r in all_records
    )

    total_chars = sum(
        r["chars"]
        for r in all_records
    )

    total_telugu_chars = sum(
        r["telugu_chars"]
        for r in all_records
    )

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 80)
    print("PRODUCTION OCR COMPLETE")
    print("=" * 80)

    print(
        f"Book              : "
        f"{pdf_path.name}"
    )

    print(
        f"Total pages       : "
        f"{total_pages:,}"
    )

    print(
        f"Successful pages  : "
        f"{successful_pages:,}"
    )

    print(
        f"Failed pages      : "
        f"{failed_pages:,}"
    )

    print(
        f"Total characters  : "
        f"{total_chars:,}"
    )

    print(
        f"Telugu characters: "
        f"{total_telugu_chars:,}"
    )

    print(
        f"Total time        : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"Final JSONL:"
    )

    print(final_file)

    print()
    print(
        f"Failed pages:"
    )

    print(failed_file)

    print("=" * 80)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()