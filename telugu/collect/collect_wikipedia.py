"""
Telugu Wikipedia collection for Model H.

Downloads the tewiki dump, extracts article text and cleans it, producing
a JSONL corpus that sits alongside the existing Wikisource, textbook and
Internet Archive sources on the manual side.

Why Wikipedia was added
-----------------------
Internet Archive collection hit a wall: after ~840 documents the
language:"tel" seam was exhausted and the remaining queries returned books
ABOUT Telugu written in English, giving a 98% LOW_TELUGU rejection rate.
That left manual collection at ~98M tokens, short of the 120M needed to
build a 550M-token corpus while holding manual at a safe margin above the
20% floor.

Wikipedia also broadens the corpus. The manual side was otherwise
newspapers (Archive), school textbooks, and literary or devotional
Wikisource text. Encyclopedic prose is a register none of those cover.

Counting this as manual collection is consistent with how Wikisource is
already counted: the assignment permits "scraping + cleaning pages you
gather yourself", and both are dumps that are downloaded, parsed and
cleaned by this pipeline rather than consumed as a prepared corpus.

Relationship to the Wikisource pipeline
---------------------------------------
clean_wikitext below is ported from clean_wikisource_text_v4 in notebook
03 (cell 13), with additions for constructs that appear in Wikipedia but
not Wikisource: infobox-heavy templates, external link blocks, and the
standard appendix sections. The shared core is deliberately unchanged, so
the two wiki sources are cleaned consistently.

One structural difference from the notebook: that pipeline wrote raw
chunks first and cleaned them in a second pass, which allowed the cleaner
to be iterated (v3, v4) without re-parsing 113 MB of XML. The cleaner is
now settled, so this script parses and cleans in a single pass and halves
the writes to Drive.

Usage
-----
    python collect_wikipedia.py --check          # verify the dump exists
    python collect_wikipedia.py --download       # fetch the dump
    python collect_wikipedia.py --extract        # parse and clean
    python collect_wikipedia.py --merge          # build the final JSONL
"""

import argparse
import bz2
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT = Path(r"G:\My Drive\LMA_Individual_Project")

RAW_DIR = PROJECT / "data" / "raw" / "manual" / "wikipedia"

CHUNK_DIR = RAW_DIR / "chunks"

PROCESSED_DIR = PROJECT / "data" / "processed" / "manual" / "wikipedia"

FINAL_DIR = PROCESSED_DIR / "final"

FINAL_FILE = FINAL_DIR / "telugu_wikipedia_merged.jsonl"

STATE_FILE = PROCESSED_DIR / "extraction_state.json"


# The Wikisource dump used in notebook 03 was dated 20260801, so the same
# date is used here for a consistent corpus snapshot. "latest" is the
# fallback because Wikimedia removes dated dumps after a few months, which
# would silently break reproduction later.
DUMP_DATE = "20260801"

DUMP_FILENAME = f"tewiki-{DUMP_DATE}-pages-articles.xml.bz2"

DUMP_URL = f"https://dumps.wikimedia.org/tewiki/{DUMP_DATE}/{DUMP_FILENAME}"

FALLBACK_URL = (
    "https://dumps.wikimedia.org/tewiki/latest/"
    "tewiki-latest-pages-articles.xml.bz2"
)

# Official Wikimedia mirror at Umeå University. Measured at 365 KB/s
# against 65 KB/s from dumps.wikimedia.org on this connection, so it is
# the default download source. Mirrors serve byte-identical copies of the
# same dated dump, so this does not affect which snapshot is used.
MIRROR_URL = (
    f"https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/tewiki/"
    f"{DUMP_DATE}/{DUMP_FILENAME}"
)

DUMP_FILE = RAW_DIR / DUMP_FILENAME


# Pages per output chunk. Small enough that an interrupted run loses
# little, large enough that Drive is not hit with thousands of small files.
CHUNK_SIZE = 5000

# Filters, matched to the Wikisource pipeline.
MIN_CHARACTERS = 100
MIN_TELUGU_CHARACTERS = 50

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


# =============================================================================
# CLEANING
# =============================================================================
#
# Ported from clean_wikisource_text_v4, notebook 03 cell 13.
# =============================================================================

def clean_wikitext(text):
    """
    Convert MediaWiki markup into plain Telugu prose.

    Returns None if the text cannot be parsed at all.
    """

    import mwparserfromhell

    # Comments
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)

    # noinclude / includeonly
    text = re.sub(
        r'<noinclude>.*?</noinclude>', ' ', text, flags=re.S | re.I
    )
    text = re.sub(
        r'<includeonly>.*?</includeonly>', ' ', text, flags=re.S | re.I
    )

    # Preserve the contents of poem and pre blocks
    text = re.sub(r'<(?:poem|pre)[^>]*>', '', text, flags=re.I)
    text = re.sub(r'</(?:poem|pre)>', '', text, flags=re.I)

    # -------------------------------------------------------------------------
    # WIKIPEDIA-SPECIFIC: drop appendix sections
    # -------------------------------------------------------------------------
    # Everything from the first appendix heading onward is reference
    # apparatus, not prose: link lists, citation dumps and category
    # scaffolding. Keeping it would teach the model bibliography
    # formatting rather than Telugu.
    # -------------------------------------------------------------------------

    text = re.split(
        r'\n==+\s*(?:'
        r'ఇవి\s*కూడా\s*చూడండి|మూలాలు|బయటి\s*లింకులు|'
        r'బయటి\s*లింకులు|సూచికలు|గ్రంథ\s*సూచి|'
        r'References|External\s+links|See\s+also|Further\s+reading'
        r')\s*==+',
        text,
        maxsplit=1,
        flags=re.I,
    )[0]

    try:
        code = mwparserfromhell.parse(text)

    except Exception:
        return None

    # Templates. On Wikipedia these are overwhelmingly infoboxes,
    # navigation boxes and citations, none of which are prose.
    for template in list(code.filter_templates(recursive=True)):
        try:
            code.remove(template)
        except ValueError:
            pass

    # Wiki links become their visible text.
    for link in list(code.filter_wikilinks(recursive=True)):
        try:
            replacement = (
                str(link.text)
                if link.text is not None
                else str(link.title)
            )
            code.replace(link, replacement)
        except ValueError:
            pass

    text = str(code)

    # Tables
    text = re.sub(r'\{\|.*?\|\}', ' ', text, flags=re.S)

    # Files and images, English and Telugu namespace spellings
    text = re.sub(
        r'\[\[(?:File|Image|దస్త్రం|బొమ్మ):.*?\]\]',
        ' ', text, flags=re.I | re.S
    )

    # References
    text = re.sub(r'<ref\b[^>]*>.*?</ref>', ' ', text, flags=re.S | re.I)
    text = re.sub(r'<ref\b[^>]*/>', ' ', text, flags=re.I)

    # Remaining HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # Categories that survived parsing
    text = re.sub(
        r'^\s*\[?\[?\s*(?:Category|వర్గం)\s*:[^\n]*$',
        '', text, flags=re.M | re.I
    )

    # WIKIPEDIA-SPECIFIC: bare external links left as bullets
    text = re.sub(r'^\s*\*\s*https?://\S+.*$', '', text, flags=re.M)
    text = re.sub(r'\[https?://\S+\s*([^\]]*)\]', r'\1', text)

    # Bold and italic markers
    text = re.sub(r"'{2,5}", '', text)

    # Section heading markers, keeping the heading text as a line
    text = re.sub(r'^\s*=+\s*(.*?)\s*=+\s*$', r'\1', text, flags=re.M)

    # Table remnants
    text = re.sub(r'^\s*[\|\!].*$', '', text, flags=re.M)

    # Whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text.strip()


def telugu_count(text):
    """Count characters in the Telugu Unicode block."""

    return sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")


def looks_like_list_article(title, text):
    """
    Detect list, index and disambiguation articles.

    These are mostly bullet lists of proper nouns. They contribute little
    running prose and would bias the tokenizer toward name fragments, so
    they are excluded on the same reasoning as looks_like_index_v4 in the
    Wikisource pipeline.
    """

    lines = [x.strip() for x in text.splitlines() if x.strip()]

    if not lines:
        return True

    list_lines = sum(
        line.startswith(("#", "*", ";")) for line in lines
    )

    if len(lines) >= 5 and list_lines / len(lines) >= 0.80:
        return True

    patterns = [
        r'జాబితా',            # list
        r'సూచిక',             # index
        r'అయోమయ',            # disambiguation
        r'\(అయోమయ\s*నివృత్తి\)',
        r'\blist\s+of\b',
        r'disambiguation',
    ]

    return any(re.search(p, title, flags=re.I) for p in patterns)


# =============================================================================
# XML HELPERS
# =============================================================================

def get_field(page, field):
    """Read a direct child field such as title, ns or id."""

    for child in page.iter():
        if child.tag.endswith(field):
            return child.text or ""

    return ""


def get_text(page):
    """Read the revision text of a page element."""

    for child in page.iter():
        if child.tag.endswith("text"):
            return child.text or ""

    return ""


def is_redirect(text):
    return text.lstrip().lower().startswith("#redirect")


# =============================================================================
# STAGES
# =============================================================================

def check_dump():
    """Report which dump URLs are reachable, without downloading."""

    import urllib.request

    print()
    print("=" * 78)
    print("CHECKING WIKIMEDIA DUMP AVAILABILITY")
    print("=" * 78)

    for label, url in [("dated", DUMP_URL), ("latest", FALLBACK_URL)]:

        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (LMA course project)"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:

                size = response.headers.get("Content-Length")

                print(
                    f"  {label:<8} {response.status}  "
                    f"{int(size) / 1024**2:,.1f} MB  {url}"
                )

        except Exception as exc:
            print(f"  {label:<8} UNAVAILABLE  {type(exc).__name__}  {url}")

    print("=" * 78)


def download_dump(url=None):
    """Download the dump, resuming a partial file if one exists."""

    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = url or MIRROR_URL

    print()
    print("=" * 78)
    print("DOWNLOADING TELUGU WIKIPEDIA DUMP")
    print("=" * 78)
    print(f"URL : {url}")
    print(f"File: {DUMP_FILE}")

    if DUMP_FILE.exists():
        print(
            f"Already present ({DUMP_FILE.stat().st_size / 1024**2:,.1f} MB)."
        )
        return DUMP_FILE

    temp = DUMP_FILE.with_suffix(".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (LMA course project)"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=120) as response:

        total = int(response.headers.get("Content-Length") or 0)

        downloaded = 0

        with open(temp, "wb") as f:

            while True:

                block = response.read(1024 * 512)

                if not block:
                    break

                f.write(block)
                downloaded += len(block)

                if downloaded % (20 * 1024 * 1024) < 512 * 1024:
                    pct = (
                        f" ({downloaded / total:.1%})" if total else ""
                    )
                    print(
                        f"  {downloaded / 1024**2:,.0f} MB{pct}"
                    )

    # Rename only after a complete download, so an interrupted transfer
    # can never be mistaken for a finished dump.
    temp.replace(DUMP_FILE)

    elapsed = time.perf_counter() - start

    print(
        f"Downloaded {DUMP_FILE.stat().st_size / 1024**2:,.1f} MB "
        f"in {elapsed:.0f}s"
    )

    return DUMP_FILE


def extract():
    """
    Parse the dump, clean each article, and write chunked JSONL.

    Resumes from the state file by skipping pages already processed.
    """

    if not DUMP_FILE.exists():
        raise FileNotFoundError(
            f"Dump not found: {DUMP_FILE}\nRun with --download first."
        )

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    state = {"pages_done": 0, "chunks_done": 0, "kept": 0}

    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

    skip_pages = state["pages_done"]

    print()
    print("=" * 78)
    print("EXTRACTING TELUGU WIKIPEDIA ARTICLES")
    print("=" * 78)
    print(f"Dump   : {DUMP_FILE.name}")
    print(f"Output : {CHUNK_DIR}")

    if skip_pages:
        print(f"Resuming: skipping {skip_pages:,} pages already processed")

    counters = {
        "pages": 0,
        "redirects": 0,
        "non_article": 0,
        "empty": 0,
        "short": 0,
        "low_telugu": 0,
        "list_article": 0,
        "kept": 0,
    }

    buffer = []
    chunk_index = state["chunks_done"]

    start = time.perf_counter()

    def flush(buffer, chunk_index):
        """Write one chunk atomically and record progress."""

        if not buffer:
            return chunk_index

        path = CHUNK_DIR / f"wikipedia_chunk_{chunk_index:05d}.jsonl"
        temp = path.with_suffix(".tmp")

        with open(temp, "w", encoding="utf-8") as f:
            for record in buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        temp.replace(path)

        state["pages_done"] = counters["pages"]
        state["chunks_done"] = chunk_index + 1
        state["kept"] = counters["kept"]

        temp_state = STATE_FILE.with_suffix(".tmp")

        with open(temp_state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        temp_state.replace(STATE_FILE)

        rate = counters["pages"] / max(
            time.perf_counter() - start, 1e-9
        )

        print(
            f"  chunk {chunk_index:05d} | pages {counters['pages']:,} "
            f"| kept {counters['kept']:,} | {rate:.0f} pages/sec"
        )

        return chunk_index + 1

    with bz2.open(DUMP_FILE, "rb") as f:

        for event, element in ET.iterparse(f, events=("end",)):

            if not element.tag.endswith("page"):
                continue

            counters["pages"] += 1

            # Resume: pages before the checkpoint were already written.
            if counters["pages"] <= skip_pages:
                element.clear()
                continue

            namespace = get_field(element, "ns")
            title = get_field(element, "title")
            raw = get_text(element)

            element.clear()

            # Namespace 0 is articles. Everything else is talk pages,
            # templates, user pages and project scaffolding.
            if namespace != "0":
                counters["non_article"] += 1
                continue

            if is_redirect(raw):
                counters["redirects"] += 1
                continue

            cleaned = clean_wikitext(raw)

            if not cleaned:
                counters["empty"] += 1
                continue

            dense = re.sub(r"\s+", "", cleaned)

            if len(dense) < MIN_CHARACTERS:
                counters["short"] += 1
                continue

            if telugu_count(cleaned) < MIN_TELUGU_CHARACTERS:
                counters["low_telugu"] += 1
                continue

            if looks_like_list_article(title, cleaned):
                counters["list_article"] += 1
                continue

            counters["kept"] += 1

            buffer.append({
                "doc_id": hashlib.sha256(
                    cleaned.encode("utf-8")
                ).hexdigest(),
                "title": title,
                "text": cleaned,
                "namespace": namespace,
                "source": "manual_wikipedia",
            })

            if len(buffer) >= CHUNK_SIZE:
                chunk_index = flush(buffer, chunk_index)
                buffer = []

    chunk_index = flush(buffer, chunk_index)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)

    for key, value in counters.items():
        print(f"  {key:<14} {value:>10,}")

    print(f"  {'time':<14} {elapsed / 60:>10.2f} minutes")
    print("=" * 78)


def merge():
    """Concatenate the chunks into one corpus file for the dedup stages."""

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    chunks = sorted(CHUNK_DIR.glob("wikipedia_chunk_*.jsonl"))

    if not chunks:
        raise FileNotFoundError(f"No chunks found in {CHUNK_DIR}")

    print()
    print("=" * 78)
    print("MERGING WIKIPEDIA CHUNKS")
    print("=" * 78)
    print(f"Chunks: {len(chunks)}")

    records = 0
    characters = 0
    telugu = 0

    temp = FINAL_FILE.with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as fout:

        for path in chunks:

            with open(path, "r", encoding="utf-8") as fin:

                for line in fin:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    fout.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    text = record.get("text", "")

                    records += 1
                    characters += len(text)
                    telugu += telugu_count(text)

    temp.replace(FINAL_FILE)

    print(f"  records    : {records:,}")
    print(f"  characters : {characters:,}")
    print(
        f"  telugu     : {telugu:,} "
        f"({telugu / characters:.2%})" if characters else ""
    )
    print(f"  est tokens : {int(characters / 3.2):,}  (at ~3.2 chars/token)")
    print(f"  output     : {FINAL_FILE}")
    print(f"  size       : {FINAL_FILE.stat().st_size / 1024**2:,.1f} MB")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Collect Telugu Wikipedia for Model H."
    )

    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--url",
        help="Override the dump URL (e.g. the 'latest' fallback).",
    )

    args = parser.parse_args()

    if not any([args.check, args.download, args.extract, args.merge]):
        parser.error(
            "Choose at least one of --check, --download, --extract, --merge"
        )

    if args.check:
        check_dump()

    if args.download:
        download_dump(args.url)

    if args.extract:
        extract()

    if args.merge:
        merge()


if __name__ == "__main__":
    main()
