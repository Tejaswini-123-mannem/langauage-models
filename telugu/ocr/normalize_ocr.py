import json
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\manual\textbooks\ocr_final"
)

OUTPUT_DIR = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\manual\textbooks\cleaned"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "cleaning_report.txt"

# Minimum amount of text we consider useful.
MIN_CHARS = 50

# Minimum Telugu-script proportion.
MIN_TELUGU_RATIO = 0.10


# =============================================================================
# TELUGU CHARACTER DETECTION
# =============================================================================

def is_telugu_char(ch):
    """
    Telugu Unicode block:
    U+0C00 - U+0C7F
    """
    return "\u0C00" <= ch <= "\u0C7F"


def calculate_telugu_ratio(text):
    """
    Telugu characters / alphabetic characters.

    We use alphabetic characters as denominator so that
    punctuation, spaces and formatting do not artificially reduce
    the Telugu percentage.
    """

    alphabetic = [
        ch for ch in text
        if ch.isalpha()
    ]

    if not alphabetic:
        return 0.0

    telugu = sum(
        1 for ch in alphabetic
        if is_telugu_char(ch)
    )

    return telugu / len(alphabetic)


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text):
    """
    Conservative normalization.

    IMPORTANT:
    We are not aggressively rewriting OCR output.
    We only normalize Unicode and obvious whitespace problems.
    """

    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize different newline types
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove zero-width characters that can appear in OCR
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    # Remove spaces at beginning/end of lines
    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Final surrounding whitespace
    text = text.strip()

    return text


# =============================================================================
# TEXT FINGERPRINT
# =============================================================================

def text_hash(text):
    """
    SHA-256 fingerprint used for exact duplicate detection.
    """

    normalized = text.encode("utf-8")

    return hashlib.sha256(normalized).hexdigest()


# =============================================================================
# PROCESS ONE JSONL FILE
# =============================================================================

def process_file(
    input_file,
    output_file,
    global_hashes,
    stats
):

    kept_records = []

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                stats["empty_json_lines"] += 1
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                stats["invalid_json"] += 1
                continue

            original_text = record.get("text", "")

            if original_text is None:
                original_text = ""

            stats["input_records"] += 1

            # -------------------------------------------------------------
            # NORMALIZE
            # -------------------------------------------------------------

            text = normalize_text(original_text)

            # -------------------------------------------------------------
            # EMPTY
            # -------------------------------------------------------------

            if not text:

                stats["removed_empty"] += 1
                continue

            # -------------------------------------------------------------
            # VERY SHORT
            # -------------------------------------------------------------

            if len(text) < MIN_CHARS:

                stats["removed_short"] += 1
                continue

            # -------------------------------------------------------------
            # TELUGU RATIO
            # -------------------------------------------------------------

            telugu_ratio = calculate_telugu_ratio(text)

            # -------------------------------------------------------------
            # LOW TELUGU
            # -------------------------------------------------------------

            if telugu_ratio < MIN_TELUGU_RATIO:

                stats["removed_low_telugu"] += 1
                continue

            # -------------------------------------------------------------
            # EXACT DUPLICATE
            # -------------------------------------------------------------

            fingerprint = text_hash(text)

            if fingerprint in global_hashes:

                stats["removed_duplicate"] += 1
                continue

            global_hashes.add(fingerprint)

            # -------------------------------------------------------------
            # BUILD CLEAN RECORD
            # -------------------------------------------------------------

            clean_record = {
                "doc_id": record.get("doc_id"),
                "source": record.get(
                    "source",
                    "manual_textbooks"
                ),
                "book": record.get("book"),
                "page": record.get("page"),
                "text": text,
                "char_count": len(text),
                "telugu_ratio": round(
                    telugu_ratio,
                    6
                ),
                "status": "CLEAN"
            }

            kept_records.append(clean_record)

            stats["kept_records"] += 1

    # -------------------------------------------------------------
    # WRITE OUTPUT
    # -------------------------------------------------------------

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in kept_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("TELUGU LOW-RESOURCE CORPUS")
    print("NORMALIZATION + BASIC CLEANING")
    print("=" * 80)

    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Minimum chars   : {MIN_CHARS}")
    print(f"Minimum Telugu  : {MIN_TELUGU_RATIO:.0%}")

    if not INPUT_DIR.exists():

        print("\nERROR: Input directory does not exist.")
        print(INPUT_DIR)
        return

    input_files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    print(f"\nJSONL files found: {len(input_files)}")

    if not input_files:

        print("ERROR: No JSONL files found.")
        return

    # -------------------------------------------------------------
    # GLOBAL STATISTICS
    # -------------------------------------------------------------

    stats = Counter()

    global_hashes = set()

    # -------------------------------------------------------------
    # PROCESS FILES
    # -------------------------------------------------------------

    for index, input_file in enumerate(
        input_files,
        start=1
    ):

        output_file = OUTPUT_DIR / input_file.name

        print("\n" + "-" * 80)
        print(
            f"[{index}/{len(input_files)}] "
            f"{input_file.name}"
        )

        process_file(
            input_file=input_file,
            output_file=output_file,
            global_hashes=global_hashes,
            stats=stats
        )

        print(
            f"Kept so far      : {stats['kept_records']:,}"
        )

        print(
            f"Duplicates removed: "
            f"{stats['removed_duplicate']:,}"
        )

    # =============================================================================
    # REPORT
    # =============================================================================

    print("\n")
    print("=" * 80)
    print("CLEANING COMPLETE")
    print("=" * 80)

    total_removed = (
        stats["removed_empty"]
        + stats["removed_short"]
        + stats["removed_low_telugu"]
        + stats["removed_duplicate"]
    )

    print(
        f"Input records          : "
        f"{stats['input_records']:,}"
    )

    print(
        f"Kept records           : "
        f"{stats['kept_records']:,}"
    )

    print(
        f"Removed empty          : "
        f"{stats['removed_empty']:,}"
    )

    print(
        f"Removed <50 chars     : "
        f"{stats['removed_short']:,}"
    )

    print(
        f"Removed low Telugu     : "
        f"{stats['removed_low_telugu']:,}"
    )

    print(
        f"Removed exact duplicate: "
        f"{stats['removed_duplicate']:,}"
    )

    print(
        f"Total removed          : "
        f"{total_removed:,}"
    )

    print(
        f"Invalid JSON lines     : "
        f"{stats['invalid_json']:,}"
    )

    print(
        f"Unique text hashes     : "
        f"{len(global_hashes):,}"
    )

    print("=" * 80)

    # -------------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------------

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TELUGU LOW-RESOURCE CORPUS - CLEANING REPORT\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write(
            f"Input directory: {INPUT_DIR}\n"
        )

        f.write(
            f"Output directory: {OUTPUT_DIR}\n\n"
        )

        f.write(
            f"Input records          : "
            f"{stats['input_records']:,}\n"
        )

        f.write(
            f"Kept records           : "
            f"{stats['kept_records']:,}\n"
        )

        f.write(
            f"Removed empty          : "
            f"{stats['removed_empty']:,}\n"
        )

        f.write(
            f"Removed <50 chars     : "
            f"{stats['removed_short']:,}\n"
        )

        f.write(
            f"Removed low Telugu     : "
            f"{stats['removed_low_telugu']:,}\n"
        )

        f.write(
            f"Removed exact duplicate: "
            f"{stats['removed_duplicate']:,}\n"
        )

        f.write(
            f"Invalid JSON lines     : "
            f"{stats['invalid_json']:,}\n"
        )

        f.write(
            f"Unique text hashes     : "
            f"{len(global_hashes):,}\n"
        )

    print(f"\nReport:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()