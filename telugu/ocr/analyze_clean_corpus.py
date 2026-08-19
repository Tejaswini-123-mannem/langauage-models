import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict


# =============================================================================
# PATHS
# =============================================================================

INPUT_DIR = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\manual\textbooks\cleaned"
)

OUTPUT_DIR = INPUT_DIR / "analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = OUTPUT_DIR / "corpus_statistics.txt"
BOOK_STATS_FILE = OUTPUT_DIR / "book_statistics.csv"


# =============================================================================
# CHARACTER HELPERS
# =============================================================================

def is_telugu(ch):
    return "\u0C00" <= ch <= "\u0C7F"


def is_english(ch):
    return (
        ("A" <= ch <= "Z")
        or ("a" <= ch <= "z")
    )


def is_digit(ch):
    return ch.isdigit()


def character_statistics(text):

    total = len(text)

    telugu = sum(
        is_telugu(ch)
        for ch in text
    )

    english = sum(
        is_english(ch)
        for ch in text
    )

    digits = sum(
        is_digit(ch)
        for ch in text
    )

    whitespace = sum(
        ch.isspace()
        for ch in text
    )

    punctuation = sum(
        ch in ".,;:!?\"'()[]{}<>-_=+*/\\|@#$%^&₹"
        for ch in text
    )

    symbols = (
        total
        - telugu
        - english
        - digits
        - whitespace
    )

    alphabetic = sum(
        ch.isalpha()
        for ch in text
    )

    if alphabetic:
        telugu_ratio = telugu / alphabetic
        english_ratio = english / alphabetic
    else:
        telugu_ratio = 0.0
        english_ratio = 0.0

    if total:
        digit_ratio = digits / total
        symbol_ratio = symbols / total
    else:
        digit_ratio = 0.0
        symbol_ratio = 0.0

    return {
        "chars": total,
        "telugu": telugu,
        "english": english,
        "digits": digits,
        "whitespace": whitespace,
        "punctuation": punctuation,
        "symbols": symbols,
        "alphabetic": alphabetic,
        "telugu_ratio": telugu_ratio,
        "english_ratio": english_ratio,
        "digit_ratio": digit_ratio,
        "symbol_ratio": symbol_ratio,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("TELUGU LOW-RESOURCE CORPUS")
    print("CLEAN CORPUS STATISTICS")
    print("=" * 80)

    print(
        f"Input: {INPUT_DIR}"
    )

    files = sorted(
        INPUT_DIR.glob("*.jsonl")
    )

    if not files:

        print("ERROR: No JSONL files found.")
        return

    print(
        f"JSONL files: {len(files)}"
    )

    # -------------------------------------------------------------------------
    # Global counters
    # -------------------------------------------------------------------------

    total_records = 0
    total_chars = 0
    total_telugu = 0
    total_english = 0
    total_digits = 0
    total_whitespace = 0
    total_symbols = 0
    total_punctuation = 0

    char_lengths = []
    telugu_ratios = []
    digit_ratios = []
    symbol_ratios = []

    short_pages = 0
    long_pages = 0
    suspicious_symbol_pages = 0
    suspicious_digit_pages = 0

    book_stats = defaultdict(
        lambda: {
            "pages": 0,
            "chars": 0,
            "telugu": 0,
            "english": 0,
            "digits": 0,
            "telugu_ratios": [],
        }
    )

    # -------------------------------------------------------------------------
    # Read corpus
    # -------------------------------------------------------------------------

    for file_index, jsonl_file in enumerate(
        files,
        start=1
    ):

        print(
            f"[{file_index}/{len(files)}] "
            f"{jsonl_file.name}"
        )

        with jsonl_file.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                text = record.get(
                    "text",
                    ""
                )

                if not text:
                    continue

                stats = character_statistics(
                    text
                )

                total_records += 1

                total_chars += stats["chars"]
                total_telugu += stats["telugu"]
                total_english += stats["english"]
                total_digits += stats["digits"]
                total_whitespace += stats["whitespace"]
                total_symbols += stats["symbols"]
                total_punctuation += stats["punctuation"]

                char_lengths.append(
                    stats["chars"]
                )

                telugu_ratios.append(
                    stats["telugu_ratio"]
                )

                digit_ratios.append(
                    stats["digit_ratio"]
                )

                symbol_ratios.append(
                    stats["symbol_ratio"]
                )

                # -------------------------------------------------------------
                # Page-level flags
                # -------------------------------------------------------------

                if stats["chars"] < 100:
                    short_pages += 1

                if stats["chars"] > 5000:
                    long_pages += 1

                if stats["symbol_ratio"] > 0.30:
                    suspicious_symbol_pages += 1

                if stats["digit_ratio"] > 0.30:
                    suspicious_digit_pages += 1

                # -------------------------------------------------------------
                # Book statistics
                # -------------------------------------------------------------

                book = record.get(
                    "book",
                    jsonl_file.stem
                )

                b = book_stats[book]

                b["pages"] += 1
                b["chars"] += stats["chars"]
                b["telugu"] += stats["telugu"]
                b["english"] += stats["english"]
                b["digits"] += stats["digits"]

                b["telugu_ratios"].append(
                    stats["telugu_ratio"]
                )

    # =============================================================================
    # SORTED STATISTICS
    # =============================================================================

    char_lengths.sort()
    telugu_ratios.sort()
    digit_ratios.sort()
    symbol_ratios.sort()

    def percentile(values, p):

        if not values:
            return 0

        index = (
            (len(values) - 1)
            * p
        )

        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return values[lower]

        weight = index - lower

        return (
            values[lower]
            * (1 - weight)
            + values[upper]
            * weight
        )

    def mean(values):

        if not values:
            return 0

        return sum(values) / len(values)

    # =============================================================================
    # GLOBAL RATIOS
    # =============================================================================

    if total_chars:

        global_telugu_char_ratio = (
            total_telugu
            / total_chars
        )

        global_digit_ratio = (
            total_digits
            / total_chars
        )

        global_symbol_ratio = (
            total_symbols
            / total_chars
        )

    else:

        global_telugu_char_ratio = 0
        global_digit_ratio = 0
        global_symbol_ratio = 0

    # =============================================================================
    # ESTIMATED TOKENS
    # =============================================================================
    #
    # This is only an estimate.
    #
    # Telugu tokenization depends heavily on the tokenizer.
    # We deliberately report several rough estimates instead of pretending
    # there is one exact token count.
    #

    estimated_tokens_char_4 = (
        total_chars / 4
    )

    estimated_tokens_char_3 = (
        total_chars / 3
    )

    # =============================================================================
    # REPORT
    # =============================================================================

    report = []

    report.append(
        "=" * 80
    )

    report.append(
        "TELUGU LOW-RESOURCE CORPUS STATISTICS"
    )

    report.append(
        "=" * 80
    )

    report.append("")

    report.append(
        f"JSONL files                 : {len(files):,}"
    )

    report.append(
        f"Records / pages             : {total_records:,}"
    )

    report.append(
        f"Total characters            : {total_chars:,}"
    )

    report.append(
        f"Telugu characters           : {total_telugu:,}"
    )

    report.append(
        f"English characters          : {total_english:,}"
    )

    report.append(
        f"Digits                      : {total_digits:,}"
    )

    report.append(
        f"Whitespace                  : {total_whitespace:,}"
    )

    report.append(
        f"Symbols                     : {total_symbols:,}"
    )

    report.append(
        f"Punctuation                 : {total_punctuation:,}"
    )

    report.append("")

    report.append(
        f"Global Telugu/total ratio   : "
        f"{global_telugu_char_ratio:.2%}"
    )

    report.append(
        f"Global digit ratio          : "
        f"{global_digit_ratio:.2%}"
    )

    report.append(
        f"Global symbol ratio         : "
        f"{global_symbol_ratio:.2%}"
    )

    report.append("")

    report.append(
        "PAGE LENGTH"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Mean characters/page       : "
        f"{mean(char_lengths):,.1f}"
    )

    report.append(
        f"Median characters/page     : "
        f"{percentile(char_lengths, 0.50):,.1f}"
    )

    report.append(
        f"P10 characters/page        : "
        f"{percentile(char_lengths, 0.10):,.1f}"
    )

    report.append(
        f"P25 characters/page        : "
        f"{percentile(char_lengths, 0.25):,.1f}"
    )

    report.append(
        f"P75 characters/page        : "
        f"{percentile(char_lengths, 0.75):,.1f}"
    )

    report.append(
        f"P90 characters/page        : "
        f"{percentile(char_lengths, 0.90):,.1f}"
    )

    report.append(
        f"P95 characters/page        : "
        f"{percentile(char_lengths, 0.95):,.1f}"
    )

    report.append(
        f"Maximum characters/page    : "
        f"{max(char_lengths):,}"
    )

    report.append("")

    report.append(
        "TELUGU RATIO"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Mean Telugu ratio           : "
        f"{mean(telugu_ratios):.2%}"
    )

    report.append(
        f"Median Telugu ratio         : "
        f"{percentile(telugu_ratios, 0.50):.2%}"
    )

    report.append(
        f"P10 Telugu ratio            : "
        f"{percentile(telugu_ratios, 0.10):.2%}"
    )

    report.append(
        f"P25 Telugu ratio            : "
        f"{percentile(telugu_ratios, 0.25):.2%}"
    )

    report.append(
        f"P75 Telugu ratio            : "
        f"{percentile(telugu_ratios, 0.75):.2%}"
    )

    report.append(
        f"P90 Telugu ratio            : "
        f"{percentile(telugu_ratios, 0.90):.2%}"
    )

    report.append(
        f"P95 Telugu ratio            : "
        f"{percentile(telugu_ratios, 0.95):.2%}"
    )

    report.append("")

    report.append(
        "SUSPICIOUS PAGE COUNTS"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"<100 characters             : "
        f"{short_pages:,}"
    )

    report.append(
        f">5000 characters            : "
        f"{long_pages:,}"
    )

    report.append(
        f">30% symbols                : "
        f"{suspicious_symbol_pages:,}"
    )

    report.append(
        f">30% digits                 : "
        f"{suspicious_digit_pages:,}"
    )

    report.append("")

    report.append(
        "TOKEN ESTIMATES"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"chars / 4 estimate          : "
        f"{estimated_tokens_char_4:,.0f}"
    )

    report.append(
        f"chars / 3 estimate          : "
        f"{estimated_tokens_char_3:,.0f}"
    )

    report.append(
        "NOTE: Actual token count depends on the tokenizer."
    )

    report.append("")

    report.append(
        "=" * 80
    )

    report_text = "\n".join(
        report
    )

    print()
    print(report_text)

    # =============================================================================
    # BOOK CSV
    # =============================================================================

    import csv

    with BOOK_STATS_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "book",
                "pages",
                "characters",
                "telugu_characters",
                "english_characters",
                "digits",
                "telugu_ratio",
                "avg_chars_per_page",
            ]
        )

        for book in sorted(
            book_stats
        ):

            b = book_stats[book]

            ratio = (
                b["telugu"]
                / (
                    b["chars"]
                    if b["chars"]
                    else 1
                )
            )

            avg_chars = (
                b["chars"]
                / b["pages"]
                if b["pages"]
                else 0
            )

            writer.writerow(
                [
                    book,
                    b["pages"],
                    b["chars"],
                    b["telugu"],
                    b["english"],
                    b["digits"],
                    f"{ratio:.6f}",
                    f"{avg_chars:.2f}",
                ]
            )

    # =============================================================================
    # SAVE REPORT
    # =============================================================================

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        f"Report:\n{REPORT_FILE}"
    )

    print(
        f"\nBook statistics:\n{BOOK_STATS_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()