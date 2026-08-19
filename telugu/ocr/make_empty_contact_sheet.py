from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import math


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

VALIDATION_DIR = (
    BASE_DIR
    / "ocr_validation"
)

IMAGE_DIR = (
    VALIDATION_DIR
    / "empty_page_inspection"
    / "images"
)

CSV_FILE = (
    VALIDATION_DIR
    / "empty_page_inspection"
    / "empty_pages_visual_inspection.csv"
)

OUTPUT_FILE = (
    VALIDATION_DIR
    / "empty_page_inspection"
    / "empty_pages_contact_sheet.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

THUMB_WIDTH = 260
THUMB_HEIGHT = 340

COLUMNS = 4

LABEL_HEIGHT = 55

MARGIN = 10


# ============================================================
# FONT
# ============================================================

try:
    FONT = ImageFont.truetype(
        "arial.ttf",
        12
    )
except Exception:
    FONT = ImageFont.load_default()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EMPTY PAGE CONTACT SHEET")
    print("=" * 80)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_FILE}"
        )

    rows = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["status"] == "RENDERED":

                rows.append(row)

    print(
        f"Rendered pages: {len(rows)}"
    )

    if not rows:
        print("No rendered pages.")
        return

    # --------------------------------------------------------
    # Calculate sheet size
    # --------------------------------------------------------

    rows_count = math.ceil(
        len(rows) / COLUMNS
    )

    cell_width = (
        THUMB_WIDTH
        + MARGIN * 2
    )

    cell_height = (
        THUMB_HEIGHT
        + LABEL_HEIGHT
        + MARGIN * 2
    )

    sheet_width = (
        COLUMNS
        * cell_width
    )

    sheet_height = (
        rows_count
        * cell_height
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    # --------------------------------------------------------
    # Add thumbnails
    # --------------------------------------------------------

    for index, row in enumerate(
        rows
    ):

        image_path = Path(
            row["image_path"]
        )

        if not image_path.exists():

            print(
                f"Missing image: "
                f"{image_path}"
            )

            continue

        try:

            image = Image.open(
                image_path
            )

            image.thumbnail(
                (
                    THUMB_WIDTH,
                    THUMB_HEIGHT
                )
            )

            col = index % COLUMNS

            r = index // COLUMNS

            x = (
                col * cell_width
                + MARGIN
                + (
                    THUMB_WIDTH
                    - image.width
                ) // 2
            )

            y = (
                r * cell_height
                + MARGIN
            )

            sheet.paste(
                image,
                (
                    x,
                    y
                )
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label_x = (
                col * cell_width
                + MARGIN
            )

            label_y = (
                r * cell_height
                + THUMB_HEIGHT
                + MARGIN
            )

            book = Path(
                row["book"]
            ).stem

            page = row["page"]

            # Shorten long book names
            if len(book) > 30:
                book = book[:27] + "..."

            label = (
                f"{index + 1}. "
                f"{book}\n"
                f"page {page}"
            )

            draw.text(
                (
                    label_x,
                    label_y
                ),
                label,
                fill="black",
                font=FONT
            )

        except Exception as e:

            print(
                f"Error opening "
                f"{image_path}: {e}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    sheet.save(
        OUTPUT_FILE,
        quality=90
    )

    print()
    print(
        "CONTACT SHEET CREATED"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


if __name__ == "__main__":
    main()