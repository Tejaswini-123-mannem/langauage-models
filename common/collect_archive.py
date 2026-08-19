"""
Internet Archive OCR-text collection for any configured language.

Searches Internet Archive for texts in the target language, downloads the
OCR output that Archive already produces for scanned books, and writes
JSONL after filtering.

Relationship to textbook_ocr/archive_collection/collect_archive_text.py
-----------------------------------------------------------------------
That script collected the Telugu Archive corpus and is kept unchanged as
the record of how it was produced. This module generalises it: the
retry, resume and manifest logic is the same, but the search queries,
script test and output paths now come from a LanguageConfig.

Why the language test matters far more here than for Wikipedia
--------------------------------------------------------------
Assamese Wikipedia is Assamese by construction. An Internet Archive item
tagged "Assamese" is a metadata claim made by whoever uploaded it, and
Archive's language fields are frequently wrong.

For Assamese the risk is acute. Bengali shares the script and has far
more digitised material, so a mislabelled Bengali book passes any test
that only asks "is this the Bengali-Assamese script?". The corpus would
then be substantially the wrong language while looking perfectly healthy,
and nothing downstream would reveal it.

So acceptance requires the language's marker characters (ৰ and ৱ for
Assamese) at a minimum density, not merely script membership. For a
language with an unambiguous block, such as Telugu, that test passes
trivially and behaviour is unchanged.

The check is applied PER DOCUMENT rather than per item, because scanned
volumes are often multilingual.

Usage
-----
    python collect_archive.py --language assamese --search-only
    python collect_archive.py --language assamese --limit 3000
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output

use_utf8_output()

try:
    import internetarchive

except ImportError:
    sys.exit(
        "internetarchive is not installed.\n\nRun:\n    pip install internetarchive\n"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

WORK = Path(r"C:\lma_work")

MAX_WORKERS = 4

SEARCH_ROWS = 500

# Sent only on the direct-HTTP fallback in download_text(). Archive asks that
# scripted clients identify themselves; the internetarchive client already
# does this on the requests it makes itself.
DIRECT_USER_AGENT = (
    "lma-individual-project/1.0 "
    "(student coursework; Assamese corpus collection)"
)

# Matches the Telugu collection's acceptance gates.
MIN_CHARACTERS = 1000
MIN_SCRIPT_RATIO = 0.30


def paths_for(language):

    root = WORK / language.name.lower()

    return {
        "raw": root / "raw" / "archive",
        "jsonl": root / "processed" / "archive" / "jsonl",
        "manifests": root / "processed" / "archive" / "manifests",
    }


FIELDS = [
    "identifier", "title", "creator", "date", "status", "source_file",
    "raw_text", "jsonl", "pages", "characters", "script_characters",
    "script_ratio", "marker_density", "error",
]


manifest_lock = threading.Lock()
seen_lock = threading.Lock()


# =============================================================================
# MANIFEST AND RESUME
# =============================================================================

def ensure_manifest(path):

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def write_manifest(path, row):

    ensure_manifest(path)

    data = {field: row.get(field, "") for field in FIELDS}

    with manifest_lock:
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(data)


def load_seen(path):
    """Identifiers already attempted, so a rerun resumes rather than repeats."""

    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_seen(path, identifier):

    with seen_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(identifier + "\n")


def safe_name(value):

    value = re.sub(r'[<>:"/\\|?*]', "_", str(value))
    value = re.sub(r"\s+", "_", value)

    return value[:180]


# =============================================================================
# ARCHIVE HELPERS
# =============================================================================

def find_djvu_txt(item):
    """Return the largest _djvu.txt file, which holds Archive's own OCR."""

    candidates = []

    try:
        for info in item.get_files():

            name = str(info.name)

            if not name.lower().endswith("_djvu.txt"):
                continue

            try:
                size = int(getattr(info, "size", 0) or 0)
            except (TypeError, ValueError):
                size = 0

            candidates.append((size, name))

    except Exception:
        return None

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


def download_text(item, filename, identifier, raw_dir):

    local = raw_dir / (safe_name(identifier) + "_djvu.txt")

    if local.exists():
        return local

    for attempt in range(1, 4):

        try:
            item.download(
                files=[filename], destdir=str(raw_dir),
                no_directory=True, verbose=False,
            )

            downloaded = raw_dir / Path(filename).name

            if downloaded.exists():

                # ----------------------------------------------------------
                # The download and the cache path can be THE SAME FILE.
                #
                # `local` is safe_name(identifier) + "_djvu.txt", and for
                # most Digital Library of India items Archive's own file is
                # named exactly that ("dli.language.1234_djvu.txt"), so
                # `downloaded == local`.
                #
                # The unguarded branch below then read `local.exists()` as
                # "a previous copy is in the way", unlinked it, and returned
                # a path to the file it had just deleted. The caller reported
                # READ_FAILED, and the identifier was marked as tried, so the
                # item was lost and never retried.
                #
                # It struck the DLI books hardest — the most valuable part of
                # the seam — because their filenames follow the identifier.
                # Compare paths before touching anything.
                # ----------------------------------------------------------

                if downloaded.resolve() == local.resolve():
                    return local

                if local.exists():
                    downloaded.unlink()
                else:
                    downloaded.rename(local)

                return local

        except Exception as exc:
            # Neither will become a 200 on retry: 403 is a blocked file,
            # 401 is a lending-restricted item whose text is visible in the
            # file listing but not downloadable without a borrow. Both are
            # common in the `asm` seam and retrying them three times each
            # costs minutes across a 15,000-item harvest.
            if "403" in str(exc) or "401" in str(exc):
                return None

            if attempt < 3:
                time.sleep(3 * attempt)

    # ----------------------------------------------------------------------
    # FALLBACK: fetch the file over plain HTTP.
    #
    # The internetarchive client sometimes returns without raising and
    # without leaving a file on disk -- observed on `dli.language.*` items
    # whose metadata advertises a perfectly ordinary 1.3 MB derivative,
    # more often as worker count rises. The client swallows whatever went
    # wrong, so there is nothing to branch on.
    #
    # /download/<identifier>/<filename> is the same bytes without the
    # client in the way. Every such item retried here has succeeded, so
    # the alternative to this fallback is discarding them silently.
    # ----------------------------------------------------------------------

    url = (
        "https://archive.org/download/"
        + urllib.parse.quote(identifier)
        + "/"
        + urllib.parse.quote(filename)
    )

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": DIRECT_USER_AGENT}
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()

        # A zero-byte body is a failure dressed as a success; writing it
        # would cache the emptiness and the item would never be retried.
        if not payload:
            return None

        local.write_bytes(payload)

        return local

    except Exception:
        return None


def split_pages(text):
    return text.split("\f") if "\f" in text else [text]


# =============================================================================
# ONE ITEM
# =============================================================================

def process_item(identifier, position, total, language, p, manifest, seen_file):

    start = time.perf_counter()

    prefix = f"[{position}/{total}]"

    def finish(status, **extra):
        write_manifest(
            manifest,
            {"identifier": identifier, "status": status, **extra},
        )
        mark_seen(seen_file, identifier)
        return {"status": status, **extra}

    try:
        item = internetarchive.get_item(
            identifier, request_kwargs={"timeout": (15, 60)}
        )

        metadata = item.metadata

        meta = {
            "title": metadata.get("title", ""),
            "creator": metadata.get("creator", ""),
            "date": metadata.get("date", ""),
        }

        djvu = find_djvu_txt(item)

        if not djvu:
            print(f"{prefix} {identifier} | NO_DJVU")
            return finish("NO_DJVU_TXT", **meta)

        raw_file = download_text(item, djvu, identifier, p["raw"])

        if raw_file is None:
            print(f"{prefix} {identifier} | DOWNLOAD_FAILED")
            return finish("DOWNLOAD_FAILED", source_file=djvu, **meta)

        try:
            text = raw_file.read_text(encoding="utf-8", errors="replace")

        except Exception as exc:
            print(f"{prefix} {identifier} | READ_FAILED")
            return finish("READ_FAILED", error=str(exc), **meta)

        characters = len(text)

        script_chars = language.script_count(text)

        ratio = language.script_ratio(text)

        markers = language.required_char_density(text)

        stats = {
            "characters": characters,
            "script_characters": script_chars,
            "script_ratio": f"{ratio:.6f}",
            "marker_density": f"{markers:.6f}",
            "source_file": djvu,
            "raw_text": str(raw_file),
            **meta,
        }

        elapsed = time.perf_counter() - start

        if characters < MIN_CHARACTERS:
            print(f"{prefix} {identifier} | TOO_SHORT | {elapsed:.1f}s")
            return finish("TOO_SHORT", **stats)

        if ratio < MIN_SCRIPT_RATIO:
            print(
                f"{prefix} {identifier} | chars={characters:,} "
                f"| script={ratio:.1%} | LOW_SCRIPT | {elapsed:.1f}s"
            )
            return finish("LOW_SCRIPT", **stats)

        # ------------------------------------------------------------------
        # LANGUAGE TEST
        # ------------------------------------------------------------------
        # Where the script is shared, this is what separates the target
        # language from its neighbour. Archive's language metadata is not
        # evidence; the text is.
        # ------------------------------------------------------------------

        if language.required_chars and markers < language.required_char_ratio:
            print(
                f"{prefix} {identifier} | chars={characters:,} "
                f"| script={ratio:.1%} | markers={markers:.3%} "
                f"| WRONG_LANGUAGE | {elapsed:.1f}s"
            )
            return finish("WRONG_LANGUAGE", **stats)

        # ------------------------------------------------------------------
        # ACCEPTED
        # ------------------------------------------------------------------

        output = p["jsonl"] / (safe_name(identifier) + ".jsonl")

        if not output.exists():

            pages = split_pages(text)

            with output.open("w", encoding="utf-8") as f:

                for number, page_text in enumerate(pages, start=1):

                    page_text = page_text.strip()

                    f.write(
                        json.dumps(
                            {
                                "doc_id": identifier,
                                "source": "internet_archive",
                                "book": meta["title"],
                                "page": number,
                                "text": page_text,
                                "char_count": len(page_text),
                                "script_characters":
                                    language.script_count(page_text),
                                "language": language.code_3,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        print(
            f"{prefix} {identifier} | chars={characters:,} "
            f"| script={ratio:.1%} | markers={markers:.2%} "
            f"| ACCEPTED | {elapsed:.1f}s"
        )

        return finish(
            "COMPLETE", jsonl=str(output),
            pages=len(split_pages(text)), **stats,
        )

    except Exception as exc:
        print(f"{prefix} {identifier} | ERROR {type(exc).__name__}")
        return finish("ERROR", error=str(exc))


# =============================================================================
# SEARCH
# =============================================================================

def search(language, limit, seen):

    identifiers = []
    local_seen = set()

    print()
    print("=" * 78)
    print(f"SEARCHING INTERNET ARCHIVE: {language.name}")
    print("=" * 78)

    for query in language.archive_queries:

        print()
        print(f"Query: {query}")

        page = 1

        while len(identifiers) < limit:

            try:
                results = internetarchive.search_items(
                    query,
                    fields=["identifier", "title", "creator", "date"],
                    params={"rows": SEARCH_ROWS, "page": page},
                )

                before = len(identifiers)

                for result in results:

                    identifier = result.get("identifier")

                    if not identifier:
                        continue

                    if identifier in seen or identifier in local_seen:
                        continue

                    local_seen.add(identifier)
                    identifiers.append(identifier)

                    if len(identifiers) >= limit:
                        break

                added = len(identifiers) - before

                print(f"  page {page}: +{added} new (total {len(identifiers)})")

                if added == 0 or len(identifiers) >= limit:
                    break

                page += 1

            except Exception as exc:
                print(f"  search error: {type(exc).__name__}")
                break

    return identifiers


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Collect Internet Archive OCR text for a language."
    )

    parser.add_argument("--language", required=True)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--search-only", action="store_true")

    # ------------------------------------------------------------------
    # Identifiers supplied from outside the language-metadata search.
    #
    # Metadata search can only find items whose language field is set and
    # correct. Items reached another way -- by looking a printed
    # catalogue's titles up, as assamese/collect/match_dls_catalogue.py
    # does -- are fed in here instead. Everything downstream is
    # unchanged: the same download, the same marker-character test, the
    # same manifest, the same resume file. Only the way the identifier
    # was discovered differs.
    # ------------------------------------------------------------------

    parser.add_argument(
        "--identifiers-file", type=Path, default=None,
        help=(
            "Read candidate identifiers from a file, one per line, "
            "instead of searching by language metadata."
        ),
    )

    args = parser.parse_args()

    language = get_language(args.language)

    p = paths_for(language)

    for directory in p.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest = p["manifests"] / "archive_collection_manifest.csv"
    seen_file = p["manifests"] / "processed_identifiers.txt"

    print()
    print("=" * 78)
    print(f"INTERNET ARCHIVE COLLECTION: {language.name} ({language.endonym})")
    print("=" * 78)
    print(f"Queries        : {len(language.archive_queries)}")
    print(f"Limit          : {args.limit:,} new identifiers")
    print(f"Min characters : {MIN_CHARACTERS:,}")
    print(f"Min script     : {MIN_SCRIPT_RATIO:.0%}")

    if language.required_chars:
        print(
            f"Marker test    : {' '.join(language.required_chars)} "
            f"at >= {language.required_char_ratio:.2%} of in-script chars"
        )
        print(
            "                 (Archive language metadata is unreliable; "
            "this is the real language test)"
        )

    print(f"Output         : {p['jsonl']}")

    ensure_manifest(manifest)

    if not seen_file.exists():
        seen_file.touch()

    seen = load_seen(seen_file)

    print(f"Already tried  : {len(seen):,} identifiers")

    if args.identifiers_file:

        supplied = [
            line.strip()
            for line in args.identifiers_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        # Deduplicate against previous runs, so feeding in a candidate
        # list that overlaps the metadata harvest re-downloads nothing.
        identifiers = []
        local_seen = set()

        for identifier in supplied:
            if identifier in seen or identifier in local_seen:
                continue
            local_seen.add(identifier)
            identifiers.append(identifier)

        print(f"Supplied file  : {args.identifiers_file}")
        print(
            f"                 {len(supplied):,} listed, "
            f"{len(identifiers):,} not yet tried"
        )

        identifiers = identifiers[: args.limit]

    else:
        identifiers = search(language, args.limit, seen)

    print()
    print(f"NEW IDENTIFIERS: {len(identifiers):,}")

    if args.search_only or not identifiers:
        return

    wall_start = time.perf_counter()

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(
                process_item, identifier, position, len(identifiers),
                language, p, manifest, seen_file,
            ): identifier
            for position, identifier in enumerate(identifiers, start=1)
        }

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"worker failed: {type(exc).__name__}")

    elapsed = time.perf_counter() - wall_start

    counts = {}

    for r in results:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1

    accepted_chars = sum(
        int(r.get("characters", 0) or 0)
        for r in results if r.get("status") == "COMPLETE"
    )

    print()
    print("=" * 78)
    print("COLLECTION COMPLETE")
    print("=" * 78)

    for status, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {status:<18} {count:>7,}")

    print()
    print(f"  accepted characters : {accepted_chars:,}")
    print(f"  est tokens          : {int(accepted_chars / 3.5):,}")
    print(f"  wall time           : {elapsed / 60:.1f} min")
    print(f"  manifest            : {manifest}")
    print("=" * 78)


if __name__ == "__main__":
    main()
