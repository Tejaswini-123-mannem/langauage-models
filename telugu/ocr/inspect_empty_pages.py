from pathlib import Path
import csv
import re
import fitz  # PyMuPDF


# ============================================================
# PATHS
# ============================================================

DRIVE_ROOT = Path(
    r"G:\My Drive\LMA_Individual_Project"
)

BASE_DIR = (
    DRIVE_ROOT
    / "data"
    / "processed"
    / "manual"
    / "textbooks"
)

RAW_DIR = (
    DRIVE_ROOT
    / "data"
    / "raw"
    / "manual"
    / "textbooks"
)

VALIDATION_DIR = (
    BASE_DIR
    / "ocr_validation"
)

QUALITY_CSV = (
    VALIDATION_DIR
    / "ocr_page_quality.csv"
)

OUTPUT_DIR = (
    VALIDATION_DIR
    / "empty_page_inspection"
)

IMAGE_DIR = (
    OUTPUT_DIR
    / "images"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "empty_pages_visual_inspection.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RENDER_DPI = 100


# ============================================================
# FIND PDF
# ============================================================

def find_pdf(book_name):

    exact = RAW_DIR / book_name

    if exact.exists():
        return exact

    target = book_name.lower()

    for pdf in RAW_DIR.glob("*.pdf"):

        if pdf.name.lower() == target:
            return pdf

    return None


# ============================================================
# RENDER PAGE
# ============================================================

def render_page(
    pdf_path,
    page_number,
    output_path
):

    doc = fitz.open(pdf_path)

    try:

        index = page_number - 1

        if index < 0 or index >= len(doc):

            return False, (
                f"Page {page_number} "
                f"outside PDF range "
                f"(1-{len(doc)})"
            )

        page = doc[index]

        zoom = RENDER_DPI / 72

        matrix = fitz.Matrix(
            zoom,
            zoom
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        pix.save(
            output_path
        )

        return True, ""

    finally:

        doc.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "TELUGU OCR — ALL EMPTY PAGE "
        "VISUAL INSPECTION"
    )
    print("=" * 80)

    print(
        f"Quality CSV:\n{QUALITY_CSV}"
    )

    print(
        f"Raw PDF directory:\n{RAW_DIR}"
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not QUALITY_CSV.exists():

        raise FileNotFoundError(
            f"Quality report not found:\n"
            f"{QUALITY_CSV}"
        )

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Raw PDF directory not found:\n"
            f"{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read ALL EMPTY pages
    # --------------------------------------------------------

    empty_pages = []

    with open(
        QUALITY_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if (
                row["diagnostic_label"]
                == "EMPTY"
            ):

                empty_pages.append(
                    {
                        "book": row["book"],
                        "page": int(
                            row["page"]
                        ),
                        "characters": int(
                            row["char_count"]
                        ),
                        "telugu_ratio": float(
                            row["telugu_ratio"]
                        ),
                    }
                )

    print()
    print(
        f"ALL EMPTY pages found: "
        f"{len(empty_pages)}"
    )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    results = []

    for i, item in enumerate(
        empty_pages,
        start=1
    ):

        book = item["book"]

        page_number = item["page"]

        print()
        print(
            f"[{i:02d}/{len(empty_pages):02d}] "
            f"{book}"
        )

        print(
            f"    Page: {page_number}"
        )

        pdf_path = find_pdf(
            book
        )

        if pdf_path is None:

            print(
                "    PDF NOT FOUND"
            )

            results.append(
                {
                    "book": book,
                    "page": page_number,
                    "pdf_found": False,
                    "pdf_pages": "",
                    "image_path": "",
                    "visual_label": "",
                    "status": "PDF_NOT_FOUND",
                }
            )

            continue

        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            Path(book).stem
        )

        image_name = (
            f"{safe_name}"
            f"_page_{page_number:04d}.png"
        )

        image_path = (
            IMAGE_DIR
            / image_name
        )

        try:

            doc = fitz.open(
                pdf_path
            )

            total_pages = len(
                doc
            )

            doc.close()

            success, error = (
                render_page(
                    pdf_path,
                    page_number,
                    image_path
                )
            )

            if not success:

                print(
                    f"    ERROR: {error}"
                )

                results.append(
                    {
                        "book": book,
                        "page": page_number,
                        "pdf_found": True,
                        "pdf_pages": total_pages,
                        "image_path": "",
                        "visual_label": "",
                        "status": "PAGE_ERROR",
                    }
                )

                continue

            print(
                f"    Rendered: "
                f"{image_name}"
            )

            results.append(
                {
                    "book": book,
                    "page": page_number,
                    "pdf_found": True,
                    "pdf_pages": total_pages,
                    "image_path": str(
                        image_path
                    ),
                    "visual_label": "",
                    "status": "RENDERED",
                }
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

            results.append(
                {
                    "book": book,
                    "page": page_number,
                    "pdf_found": True,
                    "pdf_pages": "",
                    "image_path": "",
                    "visual_label": "",
                    "status": "ERROR",
                }
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        fields = [
            "book",
            "page",
            "pdf_found",
            "pdf_pages",
            "image_path",
            "visual_label",
            "status",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    rendered = sum(
        r["status"] == "RENDERED"
        for r in results
    )

    missing = sum(
        r["status"] == "PDF_NOT_FOUND"
        for r in results
    )

    errors = sum(
        r["status"] in {
            "ERROR",
            "PAGE_ERROR"
        }
        for r in results
    )

    print()
    print("=" * 80)
    print(
        "ALL EMPTY PAGE INSPECTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"EMPTY pages       : "
        f"{len(empty_pages)}"
    )

    print(
        f"Pages rendered    : "
        f"{rendered}"
    )

    print(
        f"PDFs missing      : "
        f"{missing}"
    )

    print(
        f"Errors            : "
        f"{errors}"
    )

    print()
    print(
        "Images:"
    )

    print(
        IMAGE_DIR
    )

    print()
    print(
        "CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 80)


if __name__ == "__main__":

    main()