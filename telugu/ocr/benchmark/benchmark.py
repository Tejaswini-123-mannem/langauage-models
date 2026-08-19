import time
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    r"G:\My Drive\LMA_Individual_Project\data\raw\manual\textbooks\Copy of 10 telugu fl 2025-26.pdf"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_PAGES = 20
NUM_WORKERS = 4

OCR_LANGUAGE = "tel"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# OCR WORKER
# ============================================================

def process_page(args):

    pdf_path, page_index = args

    try:

        start = time.perf_counter()

        # Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

        # Open PDF
        doc = pymupdf.open(pdf_path)

        # Get page
        page = doc[page_index]

        # Render page
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )

        # Convert to PIL image
        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE
        )

        doc.close()

        elapsed = (
            time.perf_counter()
            - start
        )

        text = text.strip()

        chars = len(text)

        telugu_chars = len(
            re.findall(
                r"[\u0C00-\u0C7F]",
                text
            )
        )

        telugu_percentage = (
            (telugu_chars / chars) * 100
            if chars > 0
            else 0
        )

        return {
            "page": page_index + 1,
            "text": text,
            "chars": chars,
            "telugu_chars": telugu_chars,
            "telugu_percentage": telugu_percentage,
            "time_seconds": elapsed,
            "error": None
        }

    except Exception as e:

        return {
            "page": page_index + 1,
            "text": "",
            "chars": 0,
            "telugu_chars": 0,
            "telugu_percentage": 0,
            "time_seconds": 0,
            "error": repr(e)
        }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"\nPDF not found:\n{PDF_PATH}\n"
        )

    # --------------------------------------------------------
    # PDF INFORMATION
    # --------------------------------------------------------

    doc = pymupdf.open(
        str(PDF_PATH)
    )

    total_pdf_pages = len(doc)

    doc.close()

    pages_to_process = min(
        NUM_PAGES,
        total_pdf_pages
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TELUGU OCR — GOOGLE DRIVE 4 WORKER BENCHMARK")
    print("=" * 80)

    print(
        f"PDF              : {PDF_PATH.name}"
    )

    print(
        f"PDF location     : G:\\My Drive"
    )

    print(
        f"Total PDF pages  : {total_pdf_pages:,}"
    )

    print(
        f"Pages to process : {pages_to_process:,}"
    )

    print(
        f"OCR language     : {OCR_LANGUAGE}"
    )

    print(
        f"Workers          : {NUM_WORKERS}"
    )

    print(
        f"Tesseract        : {TESSERACT_PATH}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print()
    print("Starting parallel OCR...")
    print("=" * 80)

    # --------------------------------------------------------
    # CREATE JOBS
    # --------------------------------------------------------

    jobs = [
        (
            str(PDF_PATH),
            page_index
        )
        for page_index in range(
            pages_to_process
        )
    ]

    results = []

    benchmark_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # PARALLEL OCR
    # --------------------------------------------------------

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

            results.append(result)

            completed += 1

            if result["error"]:

                print(
                    f"[{completed:02d}/{pages_to_process}] "
                    f"Page {result['page']:02d} "
                    f"FAILED: "
                    f"{result['error']}"
                )

            else:

                print(
                    f"[{completed:02d}/{pages_to_process}] "
                    f"Page {result['page']:02d} | "
                    f"time={result['time_seconds']:6.2f}s | "
                    f"chars={result['chars']:6,} | "
                    f"Telugu={result['telugu_percentage']:6.2f}%"
                )

    benchmark_elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    # --------------------------------------------------------
    # SORT PAGE ORDER
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["page"]
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    successful_results = [
        r
        for r in results
        if r["error"] is None
    ]

    failed_results = [
        r
        for r in results
        if r["error"] is not None
    ]

    successful_pages = len(
        successful_results
    )

    total_chars = sum(
        r["chars"]
        for r in successful_results
    )

    total_telugu_chars = sum(
        r["telugu_chars"]
        for r in successful_results
    )

    overall_telugu_percentage = (
        (
            total_telugu_chars
            / total_chars
        ) * 100
        if total_chars > 0
        else 0
    )

    pages_per_minute = (
        successful_pages
        / (benchmark_elapsed / 60)
        if benchmark_elapsed > 0
        else 0
    )

    pages_per_hour = (
        successful_pages
        / (benchmark_elapsed / 3600)
        if benchmark_elapsed > 0
        else 0
    )

    # --------------------------------------------------------
    # SAVE OCR TEXT
    # --------------------------------------------------------

    output_text = (
        OUTPUT_DIR
        / "drive_benchmark_20pages_4workers.txt"
    )

    with open(
        output_text,
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:

            f.write("\n")
            f.write("=" * 80)
            f.write("\n")

            f.write(
                f"PAGE {result['page']}\n"
            )

            f.write("=" * 80)
            f.write("\n")

            if result["error"]:

                f.write(
                    "OCR ERROR:\n"
                )

                f.write(
                    result["error"]
                )

                f.write("\n")

            else:

                f.write(
                    result["text"]
                )

                f.write("\n")

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_file = (
        OUTPUT_DIR
        / "drive_benchmark_4workers_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write(
            "TELUGU OCR — GOOGLE DRIVE "
            "4 WORKER BENCHMARK\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"PDF: {PDF_PATH.name}\n"
        )

        f.write(
            f"Workers: {NUM_WORKERS}\n"
        )

        f.write(
            f"Pages requested: "
            f"{pages_to_process}\n"
        )

        f.write(
            f"Pages successful: "
            f"{successful_pages}\n"
        )

        f.write(
            f"Pages failed: "
            f"{len(failed_results)}\n"
        )

        f.write(
            f"Total time: "
            f"{benchmark_elapsed:.2f} seconds\n"
        )

        f.write(
            f"Pages/minute: "
            f"{pages_per_minute:.2f}\n"
        )

        f.write(
            f"Pages/hour: "
            f"{pages_per_hour:.2f}\n"
        )

        f.write(
            f"Total characters: "
            f"{total_chars:,}\n"
        )

        f.write(
            f"Telugu characters: "
            f"{total_telugu_chars:,}\n"
        )

        f.write(
            f"Overall Telugu: "
            f"{overall_telugu_percentage:.2f}%\n"
        )

        if failed_results:

            f.write("\n")
            f.write(
                "FAILED PAGES\n"
            )
            f.write("-" * 80 + "\n")

            for result in failed_results:

                f.write(
                    f"Page {result['page']}: "
                    f"{result['error']}\n"
                )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GOOGLE DRIVE BENCHMARK COMPLETE")
    print("=" * 80)

    print(
        f"Pages processed       : "
        f"{successful_pages:,}"
    )

    print(
        f"Failed pages          : "
        f"{len(failed_results):,}"
    )

    print(
        f"Total time            : "
        f"{benchmark_elapsed:.2f} seconds"
    )

    print(
        f"Pages/minute          : "
        f"{pages_per_minute:.2f}"
    )

    print(
        f"Pages/hour            : "
        f"{pages_per_hour:.2f}"
    )

    print(
        f"Total characters      : "
        f"{total_chars:,}"
    )

    print(
        f"Telugu characters     : "
        f"{total_telugu_chars:,}"
    )

    print(
        f"Overall Telugu        : "
        f"{overall_telugu_percentage:.2f}%"
    )

    print()
    print(
        "OCR output:"
    )
    print(output_text)

    print()
    print(
        "Benchmark report:"
    )
    print(report_file)

    print("=" * 80)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()