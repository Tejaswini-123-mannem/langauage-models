"""
Language-agnostic MediaWiki dump collector.

Downloads a Wikipedia or Wikisource dump for any configured language,
extracts the article text, cleans the markup, and writes chunked JSONL.

Relationship to telugu/collect/collect_wikipedia.py
---------------------------------------------------
That script collected Telugu Wikipedia and is kept unchanged as the
record of how that corpus was produced. This module generalises it: the
cleaning logic is the same, but everything language-specific — dump
names, script range, reference-section headings, file and category
namespaces, list-article titles — now comes from a LanguageConfig.

The generalisation matters beyond convenience. Phase 3 has to attribute
the difference between Model H and Model L to the data rather than to
the pipeline, and that argument is only sound if both corpora were built
by the same code.

Wikipedia and Wikisource in one collector
-----------------------------------------
The two wikis differ in ways the cleaner has to handle, so it applies the
union of both sets of rules:

  Wikisource   proofreading metadata (PageQuality), <poem> and <pre>
               blocks holding verse that must be preserved, and a large
               Page: namespace holding per-scan transcriptions, which is
               excluded by keeping only namespace 0.

  Wikipedia    infobox and citation templates, reference tags, bare
               external links, and appendix sections that are citation
               apparatus rather than prose.

Applying both to either wiki is harmless: rules that do not match simply
do nothing.

Usage
-----
    python collect_wiki.py --language assamese --wiki wikipedia --check
    python collect_wiki.py --language assamese --wiki wikipedia --download
    python collect_wiki.py --language assamese --wiki wikipedia --extract --merge
"""

import argparse
import bz2
import hashlib
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================

WORK = Path(r"C:\lma_work")

# Dumps are dated so the corpus is reproducible. Wikimedia removes dated
# dumps after a few months, hence the "latest" fallback.
DUMP_DATE = "20260801"

# Official Wikimedia mirror. Measured at 3.2 MB/s against 65 KB/s from
# dumps.wikimedia.org on this connection.
MIRROR = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps"

CANONICAL = "https://dumps.wikimedia.org"

USER_AGENT = "Mozilla/5.0 (LMA course project)"

CHUNK_SIZE = 5000

MAX_LINE_CHARS = 2000

# Wikisource proofreading state, kept per document so corpus quality
# can be reported by level: 0 without text, 1 not proofread,
# 2 problematic, 3 proofread, 4 validated.
PAGEQUALITY_RE = re.compile(r'<pagequality\s+level="(\d)"', re.I)


# =============================================================================
# PATHS
# =============================================================================

def paths_for(language, wiki):
    """Return the directory layout for one language and wiki."""

    root = WORK / language.name.lower() / "raw" / wiki

    processed = WORK / language.name.lower() / "processed" / wiki

    return {
        "raw": root,
        "chunks": root / "chunks",
        "processed": processed,
        "final": processed / "final",
        "state": processed / "extraction_state.json",
        "merged": processed / "final" / f"{language.code_2}_{wiki}_merged.jsonl",
    }


def dump_name(language, wiki, date=DUMP_DATE):
    prefix = (
        language.wikipedia_prefix if wiki == "wikipedia"
        else language.wikisource_prefix
    )

    return prefix, f"{prefix}-{date}-pages-articles.xml.bz2"


# =============================================================================
# CLEANING
# =============================================================================

def build_patterns(language):
    """
    Precompile the language-specific regexes once per run.

    Compiling inside the per-article cleaner would recompile them for
    every one of a hundred thousand articles.
    """

    appendix = (
        re.compile(
            r"\n==+\s*(?:" + "|".join(language.wiki_appendix_headings) + r")\s*==+",
            re.IGNORECASE,
        )
        if language.wiki_appendix_headings else None
    )

    files = re.compile(
        r"\[\[(?:" + "|".join(
            re.escape(n) for n in language.wiki_file_namespaces
        ) + r"):.*?\]\]",
        re.IGNORECASE | re.DOTALL,
    )

    categories = re.compile(
        r"^\s*\[?\[?\s*(?:" + "|".join(
            re.escape(n) for n in language.wiki_category_names
        ) + r")\s*:[^\n]*$",
        re.MULTILINE | re.IGNORECASE,
    )

    list_titles = (
        re.compile(
            "|".join(language.wiki_list_title_patterns), re.IGNORECASE
        )
        if language.wiki_list_title_patterns else None
    )

    return {
        "appendix": appendix,
        "files": files,
        "categories": categories,
        "list_titles": list_titles,
    }


def clean_wikitext(text, patterns):
    """Convert MediaWiki markup to plain prose, or None if unparseable."""

    import mwparserfromhell

    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)

    text = re.sub(
        r"<noinclude>.*?</noinclude>", " ", text, flags=re.S | re.I
    )
    text = re.sub(
        r"<includeonly>.*?</includeonly>", " ", text, flags=re.S | re.I
    )

    # Wikisource proofreading metadata.
    text = re.sub(
        r"\{\{\s*PageQuality\s*\|\s*[0-9]+\s*%\s*\}\}", " ", text, flags=re.I
    )
    text = re.sub(r"<pagequality[^>]*/?>", " ", text, flags=re.I)

    # Keep the CONTENTS of poem and pre blocks: on Wikisource these hold
    # verse, where the line breaks are the content.
    text = re.sub(r"<(?:poem|pre)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"</(?:poem|pre)>", "", text, flags=re.I)

    # Everything from the first appendix heading is reference apparatus.
    if patterns["appendix"]:
        text = patterns["appendix"].split(text, maxsplit=1)[0]

    try:
        code = mwparserfromhell.parse(text)

    except Exception:
        return None

    for template in list(code.filter_templates(recursive=True)):
        try:
            code.remove(template)
        except ValueError:
            pass

    for link in list(code.filter_wikilinks(recursive=True)):
        try:
            code.replace(
                link,
                str(link.text) if link.text is not None else str(link.title),
            )
        except ValueError:
            pass

    text = str(code)

    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)      # tables
    text = patterns["files"].sub(" ", text)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<ref\b[^>]*/>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)                       # html
    text = patterns["categories"].sub("", text)

    text = re.sub(r"^\s*\*\s*https?://\S+.*$", "", text, flags=re.M)
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)

    text = re.sub(r"'{2,5}", "", text)                          # bold/italic
    text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^\s*[\|\!].*$", "", text, flags=re.M)       # table scraps

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def looks_like_list_article(title, text, patterns):
    """Detect list, index and disambiguation pages."""

    lines = [x.strip() for x in text.splitlines() if x.strip()]

    if not lines:
        return True

    bullets = sum(line.startswith(("#", "*", ";")) for line in lines)

    if len(lines) >= 5 and bullets / len(lines) >= 0.80:
        return True

    if patterns["list_titles"] and patterns["list_titles"].search(title):
        return True

    return False


# =============================================================================
# XML HELPERS
# =============================================================================

def get_field(page, field):
    for child in page.iter():
        if child.tag.endswith(field):
            return child.text or ""
    return ""


def is_redirect(text):
    return text.lstrip().lower().startswith("#redirect")


# =============================================================================
# STAGES
# =============================================================================

def check(language, wiki):
    """Report dump availability and size without downloading."""

    prefix, filename = dump_name(language, wiki)

    print()
    print("=" * 78)
    print(f"DUMP AVAILABILITY: {language.name} {wiki} ({prefix})")
    print("=" * 78)

    for label, base in [("mirror", MIRROR), ("canonical", CANONICAL)]:

        url = f"{base}/{prefix}/{DUMP_DATE}/{filename}"

        try:
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": USER_AGENT}
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                size = int(response.headers.get("Content-Length") or 0)
                print(
                    f"  {label:<10} {response.status}  "
                    f"{size / 1024**2:>9,.1f} MB  {url}"
                )

        except Exception as exc:
            print(f"  {label:<10} UNAVAILABLE  {type(exc).__name__}")

    print("=" * 78)


def download(language, wiki):
    """Fetch the dump from the mirror, resuming is not attempted."""

    p = paths_for(language, wiki)
    p["raw"].mkdir(parents=True, exist_ok=True)

    prefix, filename = dump_name(language, wiki)

    target = p["raw"] / filename

    if target.exists():
        print(
            f"Already downloaded: {target} "
            f"({target.stat().st_size / 1024**2:,.1f} MB)"
        )
        return target

    url = f"{MIRROR}/{prefix}/{DUMP_DATE}/{filename}"

    print()
    print("=" * 78)
    print(f"DOWNLOADING {language.name} {wiki}")
    print("=" * 78)
    print(f"URL : {url}")
    print(f"File: {target}")

    temp = target.with_suffix(".part")

    start = time.perf_counter()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=120) as response:

        total = int(response.headers.get("Content-Length") or 0)
        got = 0

        with open(temp, "wb") as f:

            while True:
                block = response.read(1024 * 512)

                if not block:
                    break

                f.write(block)
                got += len(block)

                if got % (20 * 1024 * 1024) < 512 * 1024:
                    pct = f" ({got / total:.0%})" if total else ""
                    print(f"  {got / 1024**2:,.0f} MB{pct}")

    # Renamed only on completion, so a truncated transfer is never
    # mistaken for a finished dump.
    temp.replace(target)

    elapsed = time.perf_counter() - start

    print(
        f"Downloaded {target.stat().st_size / 1024**2:,.1f} MB "
        f"in {elapsed:.0f}s ({target.stat().st_size / elapsed / 1024**2:.1f} MB/s)"
    )

    return target


def extract(language, wiki):
    """Parse the dump, clean each article, write chunked JSONL."""

    p = paths_for(language, wiki)

    prefix, filename = dump_name(language, wiki)

    dump = p["raw"] / filename

    if not dump.exists():
        raise FileNotFoundError(f"Dump not found: {dump}\nRun --download first.")

    p["chunks"].mkdir(parents=True, exist_ok=True)
    p["processed"].mkdir(parents=True, exist_ok=True)

    patterns = build_patterns(language)

    state = {"pages_done": 0, "chunks_done": 0, "kept": 0}

    if p["state"].exists():
        with open(p["state"], "r", encoding="utf-8") as f:
            state = json.load(f)

    skip = state["pages_done"]

    print()
    print("=" * 78)
    print(f"EXTRACTING {language.name.upper()} {wiki.upper()}")
    print("=" * 78)
    print(f"Dump  : {dump.name}")
    print(f"Output: {p['chunks']}")

    if skip:
        print(f"Resuming: skipping {skip:,} pages already processed")

    counters = {
        "pages": 0, "redirects": 0, "non_article": 0, "empty": 0,
        "short": 0, "low_script": 0, "wrong_language": 0,
        "list_article": 0, "page_namespace": 0, "kept": 0,
    }

    buffer = []
    chunk_index = state["chunks_done"]
    start = time.perf_counter()

    def flush(buffer, chunk_index):

        if not buffer:
            return chunk_index

        path = p["chunks"] / f"{wiki}_chunk_{chunk_index:05d}.jsonl"
        temp = path.with_suffix(".tmp")

        with open(temp, "w", encoding="utf-8") as f:
            for record in buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        temp.replace(path)

        state["pages_done"] = counters["pages"]
        state["chunks_done"] = chunk_index + 1
        state["kept"] = counters["kept"]

        temp_state = p["state"].with_suffix(".tmp")

        with open(temp_state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        temp_state.replace(p["state"])

        rate = counters["pages"] / max(time.perf_counter() - start, 1e-9)

        print(
            f"  chunk {chunk_index:05d} | pages {counters['pages']:,} "
            f"| kept {counters['kept']:,} | {rate:.0f} pages/sec"
        )

        return chunk_index + 1

    with bz2.open(dump, "rb") as f:

        for event, element in ET.iterparse(f, events=("end",)):

            if not element.tag.endswith("page"):
                continue

            counters["pages"] += 1

            if counters["pages"] <= skip:
                element.clear()
                continue

            namespace = get_field(element, "ns")
            title = get_field(element, "title")

            raw = ""
            for child in element.iter():
                if child.tag.endswith("text"):
                    raw = child.text or ""

            element.clear()

            # ----------------------------------------------------------
            # NAMESPACE FILTER
            # ----------------------------------------------------------
            # Namespace 0 holds articles and assembled works and is
            # always kept. Namespace 104 ("Page:") holds per-scan
            # transcriptions on Wikisource, and whether it is corpus or
            # duplication depends on the language. See the note on
            # wikisource_include_page_namespace.
            # ----------------------------------------------------------

            keep_page_ns = (
                wiki == "wikisource"
                and language.wikisource_include_page_namespace
            )

            if namespace == "0":
                pass

            elif namespace == "104" and keep_page_ns:
                counters["page_namespace"] += 1

            else:
                counters["non_article"] += 1
                continue

            if is_redirect(raw):
                counters["redirects"] += 1
                continue

            # Read the proofreading level before cleaning strips it.
            pq_match = PAGEQUALITY_RE.search(raw)
            pagequality = int(pq_match.group(1)) if pq_match else None

            cleaned = clean_wikitext(raw, patterns)

            if not cleaned:
                counters["empty"] += 1
                continue

            dense = re.sub(r"\s+", "", cleaned)

            if len(dense) < language.min_characters:
                counters["short"] += 1
                continue

            if language.script_count(cleaned) < language.min_script_characters:
                counters["low_script"] += 1
                continue

            # Marker-character test. For a shared script such as
            # Bengali-Assamese this is what separates the target language
            # from its neighbour; for an unambiguous block it passes
            # trivially.
            if language.required_chars:
                if (
                    language.required_char_density(cleaned)
                    < language.required_char_ratio
                ):
                    counters["wrong_language"] += 1
                    continue

            if looks_like_list_article(title, cleaned, patterns):
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
                "source": f"manual_{wiki}",
                "language": language.code_3,
                "pagequality": pagequality,
            })

            if len(buffer) >= CHUNK_SIZE:
                chunk_index = flush(buffer, chunk_index)
                buffer = []

    flush(buffer, chunk_index)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)

    for key, value in counters.items():
        print(f"  {key:<16} {value:>10,}")

    print(f"  {'time':<16} {elapsed / 60:>10.2f} minutes")
    print("=" * 78)


def merge(language, wiki):
    """Concatenate the chunks into one corpus file."""

    p = paths_for(language, wiki)

    p["final"].mkdir(parents=True, exist_ok=True)

    chunks = sorted(p["chunks"].glob(f"{wiki}_chunk_*.jsonl"))

    if not chunks:
        raise FileNotFoundError(f"No chunks in {p['chunks']}")

    print()
    print("=" * 78)
    print(f"MERGING {language.name.upper()} {wiki.upper()} CHUNKS")
    print("=" * 78)
    print(f"Chunks: {len(chunks)}")

    records = characters = script = 0

    temp = p["merged"].with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as fout:

        for path in chunks:

            with open(path, "r", encoding="utf-8") as fin:

                for line in fin:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                    text = record.get("text", "")

                    records += 1
                    characters += len(text)
                    script += language.script_count(text)

    temp.replace(p["merged"])

    print(f"  records    : {records:,}")
    print(f"  characters : {characters:,}")

    if characters:
        print(f"  in-script  : {script:,} ({script / characters:.2%})")
        print(f"  est tokens : {int(characters / 3.5):,}  (at ~3.5 chars/token)")

    print(f"  output     : {p['merged']}")
    print(f"  size       : {p['merged'].stat().st_size / 1024**2:,.1f} MB")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Collect a Wikipedia or Wikisource dump for any language."
    )

    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--wiki", required=True, choices=["wikipedia", "wikisource"]
    )

    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()

    if not any([args.check, args.download, args.extract, args.merge]):
        parser.error("Choose at least one of --check/--download/--extract/--merge")

    language = get_language(args.language)

    print()
    print(f"Language : {language.name} ({language.endonym}) [{language.code_3}]")
    print(
        f"Script   : {language.script_name} "
        f"U+{ord(language.script_start):04X}-U+{ord(language.script_end):04X}"
    )

    if language.required_chars:
        print(
            f"Markers  : {' '.join(language.required_chars)} "
            f"(min density {language.required_char_ratio:.2%})"
        )

    if args.check:
        check(language, args.wiki)

    if args.download:
        download(language, args.wiki)

    if args.extract:
        extract(language, args.wiki)

    if args.merge:
        merge(language, args.wiki)


if __name__ == "__main__":
    main()
