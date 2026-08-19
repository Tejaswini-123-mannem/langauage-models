"""
Match the DLS Assam digitised-book catalogue against Internet Archive.

Why this stage exists
---------------------
The Directorate of Library Services, Assam, digitised roughly 4,000 rare
and copyright-free titles and published the catalogue as a spreadsheet
(`digitized book list.xls`, 3,456 rows, 2,148 of them Assamese, 517,609
pages). The catalogue is only a catalogue: it lists title, creator,
language, year, pages and the DVD the scan sits on. It carries no URL,
and the DVDs are not distributed online.

The scans themselves are not lost, though. Most Indian public-library
digitisation of that era was deposited into the Digital Library of India
and mirrored onto Internet Archive as `in.ernet.dli.*` items, which carry
Archive's own `_djvu.txt` OCR. So the catalogue is usable as a WORK LIST:
for each Assamese title, ask Archive whether it holds that book.

Why the language query does not make this redundant
---------------------------------------------------
`collect_archive.py` finds items by language metadata. That misses any
item whose language field is blank, wrong, or set to the English name
when the harvest asked for the code (the mistake that cost this project
seven eighths of the seam — see `language_config.archive_queries`).

A title lookup does not depend on the language field at all, so it
reaches exactly the items metadata search cannot. The two passes overlap
heavily and that is fine: identifiers are deduplicated against the
collector's own `processed_identifiers.txt`, so an item found twice is
downloaded once.

What this script does NOT do
----------------------------
It does not download anything and it does not decide what is Assamese.
It emits candidate identifiers. Acceptance stays where it belongs, in
`collect_archive.py`, where the marker-character test (ৰ / ৱ) is applied
to the TEXT rather than to anyone's metadata claim. A title match is a
hint, not evidence: "Assam Buranji" exists on Archive in Assamese,
Bengali and English editions, and only the text can tell them apart.

Usage
-----
    # See what the catalogue holds, without touching the network
    python assamese/collect/match_dls_catalogue.py --catalogue-only

    # Look every Assamese title up on Archive (resumable, ~2,148 lookups)
    python assamese/collect/match_dls_catalogue.py

    # Then hand the new identifiers to the ordinary collector
    python common/collect_archive.py --language assamese \
        --identifiers-file C:/lma_work/assamese/processed/archive/manifests/dls_candidates.txt
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from language_config import get_language, use_utf8_output
from workspace import Workspace

use_utf8_output()

try:
    import xlrd

except ImportError:
    sys.exit(
        "xlrd is not installed (needed to read the .xls catalogue).\n\n"
        "Run:\n    pip install xlrd\n"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

REPO = Path(__file__).resolve().parents[2]

CATALOGUE = REPO / "digitized book list.xls"

# The catalogue's own language column. Rows are matched case-insensitively
# because the sheet contains both "Sanskrit" and "sanskrit".
TARGET_LANGUAGE = "assamese"

SEARCH_URL = "https://archive.org/advancedsearch.php"

USER_AGENT = "lma-individual-project/1.0 (student coursework; contact via GitHub)"

# Archive rate-limits aggressively on sustained scripted search. One
# second between lookups keeps a 2,148-title pass under an hour and has
# not triggered throttling; lower values did.
REQUEST_DELAY = 1.0

RETRIES = 3

# A title has to be this many characters before it is worth searching.
# The catalogue contains a handful of one-word titles ("Anjali") that
# match hundreds of unrelated items.
MIN_TITLE_LENGTH = 6

FIELDS = [
    "catalogue_row", "title", "creator", "year", "pages",
    "query", "num_found", "identifiers", "status",
]


# =============================================================================
# CATALOGUE
# =============================================================================

def read_catalogue(path, target_language):
    """
    Read the DLS spreadsheet and return its target-language rows.

    The workbook holds the same 3,456 books twice, as "Title wise" and
    "DVD wise" sheets ordered differently, so rows are deduplicated on
    (title, creator, year). Both sheets carry two banner rows and a
    header row before the data starts.
    """

    book = xlrd.open_workbook(str(path))

    seen = set()
    rows = []

    for sheet in book.sheets():

        for index in range(sheet.nrows):

            values = [str(cell.value).strip() for cell in sheet.row(index)]

            # Data rows have a title in column 1 and a language in
            # column 3; the banner and header rows do not.
            if len(values) < 4 or not values[1]:
                continue

            title, creator, language = values[1], values[2], values[3]

            if language.strip().lower() != target_language:
                continue

            year = values[4] if len(values) > 4 else ""
            pages = values[5] if len(values) > 5 else ""

            key = (title.lower(), creator.lower(), year)

            if key in seen:
                continue

            seen.add(key)

            rows.append(
                {
                    "catalogue_row": len(rows) + 1,
                    "title": title,
                    "creator": creator,
                    "year": clean_number(year),
                    "pages": clean_number(pages),
                }
            )

    return rows


def clean_number(value):
    """
    Render a spreadsheet number as a plain integer string.

    xlrd returns every numeric cell as a float, so a year reads "1891.0"
    and a page count "28.0". Both are reported in the CSV and compared by
    eye against Archive metadata, so the noise is worth removing.
    """

    value = str(value).strip()

    if re.fullmatch(r"\d+\.0", value):
        return value[:-2]

    return value


# =============================================================================
# SEARCH
# =============================================================================

def clean_title(title):
    """
    Reduce a catalogue title to something worth searching.

    Catalogue titles carry edition and volume apparatus that Archive's
    own titles usually omit:

        "A Brief History Of India Part. 1, Ed. 1st"  ->  "A Brief History Of India"
        "Assamese Larapatha Part. 2 Ed. 5th"         ->  "Assamese Larapatha"

    Searching the full string finds nothing; searching the stem finds the
    volume. Volume numbers are dropped rather than kept because Archive
    splits multi-volume works inconsistently, and a stem match returns
    every volume, which is what we want.
    """

    text = title

    # Trailing edition / part / volume apparatus, in the spellings the
    # catalogue actually uses.
    text = re.sub(
        r"[,\s]*\b(?:ed|edn|edition)\b\.?\s*\d*\s*(?:st|nd|rd|th)?\.?\s*$",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[,\s]*\b(?:part|pt|vol|volume|no)\b\.?\s*\d*\.?\s*$",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[,\s]*\b(?:ed|edn|edition)\b\.?\s*\d*\s*(?:st|nd|rd|th)?\.?\s*$",
        "", text, flags=re.IGNORECASE,
    )

    # Collapse punctuation that Archive's index does not carry anyway.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def search_title(title, retries=RETRIES):
    """
    Ask Archive for text items whose title matches, as an exact phrase.

    Returns (num_found, [identifier, ...]) or (None, []) on failure.

    The phrase is quoted so that "Assam Buranji" does not return every
    item with "Assam" in the title. mediatype is constrained to texts:
    the catalogue is books, and Archive holds a great deal of Assamese
    audio that would otherwise match.
    """

    query = f'title:("{title}") AND mediatype:(texts)'

    params = {
        "q": query,
        "rows": "20",
        "page": "1",
        "output": "json",
    }

    url = (
        SEARCH_URL + "?" + urllib.parse.urlencode(params)
        + "&fl[]=identifier&fl[]=title&fl[]=language&fl[]=year"
    )

    for attempt in range(1, retries + 1):

        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )

            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)

            block = payload.get("response", {})

            docs = block.get("docs", [])

            return (
                block.get("numFound", 0),
                [d.get("identifier") for d in docs if d.get("identifier")],
                query,
            )

        except Exception:

            if attempt < retries:
                # Back off: sustained scripted search is what triggers
                # Archive's rate limiter, and hammering makes it worse.
                time.sleep(5 * attempt)

    return None, [], query


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Match the DLS Assam catalogue against Internet Archive.",
    )

    parser.add_argument(
        "--language", default="assamese",
        help="Language whose workspace the output is written to.",
    )
    parser.add_argument(
        "--catalogue", type=Path, default=CATALOGUE,
        help="Path to the DLS .xls catalogue.",
    )
    parser.add_argument(
        "--catalogue-only", action="store_true",
        help="Summarise the catalogue and exit, without searching.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Search at most this many titles (0 = all). Useful for a trial run.",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help="Seconds between Archive lookups.",
    )

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    manifests = workspace.processed / "archive" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    report_file = manifests / "dls_catalogue_matches.csv"
    candidates_file = manifests / "dls_candidates.txt"

    print()
    print("=" * 78)
    print(f"DLS ASSAM CATALOGUE -> INTERNET ARCHIVE   ({language.name})")
    print("=" * 78)

    if not args.catalogue.exists():
        sys.exit(f"Catalogue not found: {args.catalogue}")

    rows = read_catalogue(args.catalogue, TARGET_LANGUAGE)

    total_pages = sum(int(r["pages"]) for r in rows if r["pages"].isdigit())

    print(f"Catalogue      : {args.catalogue.name}")
    print(f"{language.name} titles : {len(rows):,}")
    print(f"Catalogue pages: {total_pages:,}")
    print(f"Report         : {report_file}")
    print(f"Candidates     : {candidates_file}")

    if args.catalogue_only:
        print()
        print("First 10 titles:")
        for row in rows[:10]:
            print(
                f"   {row['catalogue_row']:>5}  {row['title'][:52]:<52} "
                f"{row['creator'][:24]:<24} {row['year']:>6} {row['pages']:>6}p"
            )
        return

    # ---- resume ------------------------------------------------------------
    #
    # A full pass is ~2,148 network lookups at one second each. It will be
    # interrupted, so completed titles are read back and skipped.

    done = set()

    if report_file.exists():
        with report_file.open("r", encoding="utf-8", newline="") as f:
            for record in csv.DictReader(f):
                done.add(record["title"])

    print(f"Already looked up: {len(done):,}")

    pending = [r for r in rows if r["title"] not in done]

    if args.limit:
        pending = pending[: args.limit]

    print(f"To look up     : {len(pending):,}")
    print()

    # ---- search ------------------------------------------------------------

    new_file = not report_file.exists()

    found_identifiers = set()

    if candidates_file.exists():
        found_identifiers = {
            line.strip()
            for line in candidates_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    matched = 0
    searched = 0

    with report_file.open("a", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        for row in pending:

            stem = clean_title(row["title"])

            if len(stem) < MIN_TITLE_LENGTH:
                writer.writerow(
                    {**row, "query": "", "num_found": 0,
                     "identifiers": "", "status": "TITLE_TOO_SHORT"}
                )
                continue

            num_found, identifiers, query = search_title(stem)

            searched += 1

            if num_found is None:
                status = "SEARCH_FAILED"

            elif not identifiers:
                status = "NO_MATCH"

            else:
                status = "MATCH"
                matched += 1
                found_identifiers.update(identifiers)

            writer.writerow(
                {
                    **row,
                    "query": query,
                    "num_found": num_found if num_found is not None else "",
                    "identifiers": " ".join(identifiers),
                    "status": status,
                }
            )

            f.flush()

            print(
                f"[{searched:>5}/{len(pending)}] {stem[:46]:<46} "
                f"{status:<14} {len(identifiers)} id(s)"
            )

            # Write the candidate list as we go, so an interrupted run
            # still leaves something the collector can consume.
            candidates_file.write_text(
                "\n".join(sorted(found_identifiers)) + "\n", encoding="utf-8"
            )

            time.sleep(args.delay)

    # ---- summary -----------------------------------------------------------

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Titles searched      : {searched:,}")
    print(f"Titles with a match  : {matched:,}")
    print(f"Distinct identifiers : {len(found_identifiers):,}")
    print()
    print("These are CANDIDATES. Whether each is genuinely Assamese is")
    print("decided from its text by collect_archive.py, not from its")
    print("catalogue entry. Next:")
    print()
    print("    python common/collect_archive.py --language assamese \\")
    print(f"        --identifiers-file {candidates_file}")


if __name__ == "__main__":
    main()
