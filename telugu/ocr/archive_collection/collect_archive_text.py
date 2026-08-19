# ============================================================
# TELUGU LOW-RESOURCE CORPUS
# INTERNET ARCHIVE OCR-TEXT COLLECTION
# 4-WORKER BENCHMARK
# ============================================================

import csv
import json
import re
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import internetarchive


# ============================================================
# CONFIGURATION
# ============================================================

# Number of NEW identifiers to process in this run. Identifiers already
# listed in processed_identifiers.txt are skipped before this cap applies,
# so the run resumes rather than repeating earlier work.
#
# Raised from 2000 to 4000 to grow the manual side of the corpus. Manual
# collection must supply at least 20% of final training tokens, and at
# 949 accepted documents it was supplying only 73M of the ~100M tokens
# needed for a 500M-token corpus.
DOCUMENT_LIMIT = 4000

MAX_WORKERS = 4

SEARCH_ROWS = 500

MIN_CHARACTERS = 1000

MIN_TELUGU_RATIO = 0.30


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\internet_archive"
)

RAW_TEXT_DIR = Path(
    r"G:\My Drive\LMA_Individual_Project\data\raw\internet_archive\ocr_text"
)

JSONL_DIR = BASE_DIR / "jsonl"

MANIFEST_DIR = BASE_DIR / "manifests"


RAW_TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

JSONL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MANIFEST_FILE = (
    MANIFEST_DIR
    / "archive_text_collection_manifest.csv"
)

SEEN_FILE = (
    MANIFEST_DIR
    / "processed_identifiers.txt"
)


# ============================================================
# SEARCH QUERIES
# ============================================================

SEARCH_QUERIES = [

    'language:"tel" AND mediatype:texts',

    'subject:"Telugu" AND mediatype:texts',

    'title:"Telugu" AND mediatype:texts',

    'creator:"Telugu" AND mediatype:texts',

]


# ============================================================
# MANIFEST
# ============================================================

FIELDS = [

    "identifier",
    "title",
    "creator",
    "date",
    "status",
    "source_file",
    "raw_text",
    "jsonl",
    "pages",
    "characters",
    "telugu_characters",
    "telugu_ratio",
    "error",

]


manifest_lock = threading.Lock()


def ensure_manifest():

    if MANIFEST_FILE.exists():
        return

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()


def write_manifest(row):

    ensure_manifest()

    data = {
        field: row.get(
            field,
            ""
        )
        for field in FIELDS
    }

    with manifest_lock:

        with MANIFEST_FILE.open(
            "a",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=FIELDS
            )

            writer.writerow(data)


# ============================================================
# PROCESSED IDENTIFIERS
# ============================================================

seen_lock = threading.Lock()


def load_seen():

    if not SEEN_FILE.exists():
        return set()

    with SEEN_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        return {
            line.strip()
            for line in f
            if line.strip()
        }


def mark_seen(identifier):

    with seen_lock:

        with SEEN_FILE.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                identifier
                + "\n"
            )


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_name(value):

    value = str(value)

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value
    )

    value = re.sub(
        r"\s+",
        "_",
        value
    )

    return value[:180]


# ============================================================
# TELUGU CHARACTER
# ============================================================

def is_telugu(ch):

    return (
        "\u0C00"
        <= ch
        <= "\u0C7F"
    )


def count_telugu(text):

    return sum(
        1
        for ch in text
        if is_telugu(ch)
    )


# ============================================================
# FIND ARCHIVE OCR FILE
# ============================================================

def find_djvu_txt(item):

    candidates = []

    try:

        for file_info in item.get_files():

            name = str(
                file_info.name
            )

            if not name.lower().endswith(
                "_djvu.txt"
            ):
                continue

            size = getattr(
                file_info,
                "size",
                0
            )

            try:
                size = int(size)
            except Exception:
                size = 0

            candidates.append(
                (
                    size,
                    name
                )
            )

    except Exception as exc:

        print(
            "File inspection failed:",
            type(exc).__name__,
            exc
        )

        return None


    if not candidates:
        return None


    candidates.sort(
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# DOWNLOAD OCR
# ============================================================

def download_text(
    item,
    filename,
    identifier
):

    local_path = (
        RAW_TEXT_DIR
        /
        (
            safe_name(identifier)
            + "_djvu.txt"
        )
    )


    # --------------------------------------------------------
    # Already downloaded
    # --------------------------------------------------------

    if local_path.exists():

        return local_path


    # --------------------------------------------------------
    # Retry
    # --------------------------------------------------------

    for attempt in range(1, 4):

        try:

            item.download(
                files=[
                    filename
                ],
                destdir=str(
                    RAW_TEXT_DIR
                ),
                no_directory=True,
                verbose=False
            )


            downloaded = (
                RAW_TEXT_DIR
                /
                Path(filename).name
            )


            if downloaded.exists():

                # Rename to identifier-based name
                # to prevent collisions.

                if local_path.exists():

                    downloaded.unlink()

                else:

                    downloaded.rename(
                        local_path
                    )

                return local_path


        except Exception as exc:

            error_text = str(exc)


            # 403 should not be repeatedly retried.

            if "403" in error_text:

                return None


            if attempt < 3:

                time.sleep(
                    3 * attempt
                )


    return None


# ============================================================
# SPLIT INTO PAGES
# ============================================================

def split_pages(text):

    if "\f" in text:

        return text.split(
            "\f"
        )

    return [
        text
    ]


# ============================================================
# CREATE JSONL
# ============================================================

def convert_to_jsonl(
    identifier,
    title,
    raw_file
):

    output_file = (
        JSONL_DIR
        /
        (
            safe_name(identifier)
            + ".jsonl"
        )
    )


    # Already converted.

    if output_file.exists():

        return output_file


    try:

        text = raw_file.read_text(
            encoding="utf-8",
            errors="replace"
        )

    except Exception:

        return None


    pages = split_pages(
        text
    )


    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for page_number, page_text in enumerate(
            pages,
            start=1
        ):

            page_text = page_text.strip()

            chars = len(
                page_text
            )

            telugu = count_telugu(
                page_text
            )


            record = {

                "doc_id":
                    identifier,

                "source":
                    "internet_archive",

                "book":
                    title,

                "page":
                    page_number,

                "text":
                    page_text,

                "char_count":
                    chars,

                "telugu_characters":
                    telugu,

            }


            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


    return output_file


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_item(
    identifier,
    position,
    total
):

    start_time = time.perf_counter()


    prefix = (
        f"[{position:02d}/{total:02d}]"
    )


    try:

        # ----------------------------------------------------
        # GET METADATA
        # ----------------------------------------------------

        item = internetarchive.get_item(
            identifier,
            request_kwargs={
                "timeout": (
                    15,
                    60
                )
            }
        )


        metadata = item.metadata


        title = metadata.get(
            "title",
            ""
        )


        creator = metadata.get(
            "creator",
            ""
        )


        date = metadata.get(
            "date",
            ""
        )


        # ----------------------------------------------------
        # FIND OCR
        # ----------------------------------------------------

        djvu_file = find_djvu_txt(
            item
        )


        if not djvu_file:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                "| NO_DJVU",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "NO_DJVU_TXT",

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "NO_DJVU_TXT",

                "time":
                    elapsed
            }


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        raw_file = download_text(
            item,
            djvu_file,
            identifier
        )


        if raw_file is None:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                "| DOWNLOAD_FAILED",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "DOWNLOAD_FAILED",

                "source_file":
                    djvu_file,

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "DOWNLOAD_FAILED",

                "time":
                    elapsed
            }


        # ----------------------------------------------------
        # READ OCR
        # ----------------------------------------------------

        try:

            text = raw_file.read_text(
                encoding="utf-8",
                errors="replace"
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                "| READ_FAILED",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "READ_FAILED",

                "error":
                    str(exc),

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "READ_FAILED",

                "time":
                    elapsed
            }


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_chars = len(
            text
        )

        total_telugu = count_telugu(
            text
        )


        ratio = (

            total_telugu
            /
            total_chars

            if total_chars
            else 0

        )


        # ----------------------------------------------------
        # CHARACTER GATE
        # ----------------------------------------------------

        if total_chars < MIN_CHARACTERS:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                f"| chars={total_chars:,}",
                f"| Telugu={ratio:.2%}",
                "| TOO_SHORT",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "TOO_SHORT",

                "source_file":
                    djvu_file,

                "raw_text":
                    str(raw_file),

                "characters":
                    total_chars,

                "telugu_characters":
                    total_telugu,

                "telugu_ratio":
                    f"{ratio:.6f}",

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "TOO_SHORT",

                "characters":
                    total_chars,

                "time":
                    elapsed
            }


        # ----------------------------------------------------
        # TELUGU GATE
        # ----------------------------------------------------

        if ratio < MIN_TELUGU_RATIO:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                f"| chars={total_chars:,}",
                f"| Telugu={ratio:.2%}",
                "| LOW_TELUGU",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "LOW_TELUGU",

                "source_file":
                    djvu_file,

                "raw_text":
                    str(raw_file),

                "characters":
                    total_chars,

                "telugu_characters":
                    total_telugu,

                "telugu_ratio":
                    f"{ratio:.6f}",

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "LOW_TELUGU",

                "characters":
                    total_chars,

                "time":
                    elapsed
            }


        # ----------------------------------------------------
        # ACCEPTED
        # ----------------------------------------------------

        jsonl_file = convert_to_jsonl(
            identifier,
            title,
            raw_file
        )


        if jsonl_file is None:

            elapsed = (
                time.perf_counter()
                - start_time
            )


            print(
                prefix,
                identifier,
                "| CONVERSION_FAILED",
                f"| {elapsed:.2f}s"
            )


            write_manifest({

                "identifier":
                    identifier,

                "title":
                    title,

                "creator":
                    creator,

                "date":
                    date,

                "status":
                    "CONVERSION_FAILED",

                "source_file":
                    djvu_file,

            })


            mark_seen(
                identifier
            )


            return {
                "status":
                    "CONVERSION_FAILED",

                "time":
                    elapsed
            }


        pages = len(
            split_pages(text)
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        print(
            prefix,
            identifier,
            f"| pages={pages}",
            f"| chars={total_chars:,}",
            f"| Telugu={ratio:.2%}",
            "| ACCEPTED",
            f"| {elapsed:.2f}s"
        )


        write_manifest({

            "identifier":
                identifier,

            "title":
                title,

            "creator":
                creator,

            "date":
                date,

            "status":
                "COMPLETE",

            "source_file":
                djvu_file,

            "raw_text":
                str(raw_file),

            "jsonl":
                str(jsonl_file),

            "pages":
                pages,

            "characters":
                total_chars,

            "telugu_characters":
                total_telugu,

            "telugu_ratio":
                f"{ratio:.6f}",

        })


        mark_seen(
            identifier
        )


        return {
            "status":
                "COMPLETE",

            "characters":
                total_chars,

            "telugu":
                total_telugu,

            "time":
                elapsed
        }


    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )


        print(
            prefix,
            identifier,
            "| ERROR",
            type(exc).__name__,
            f"| {elapsed:.2f}s"
        )


        write_manifest({

            "identifier":
                identifier,

            "status":
                "ERROR",

            "error":
                str(exc),

        })


        mark_seen(
            identifier
        )


        return {
            "status":
                "ERROR",

            "time":
                elapsed
        }


# ============================================================
# SEARCH ARCHIVE WITH PAGINATION
# ============================================================

def search_archive():

    seen = load_seen()

    identifiers = []

    local_seen = set()


    print()
    print("=" * 80)
    print("SEARCHING INTERNET ARCHIVE")
    print("=" * 80)


    for query in SEARCH_QUERIES:

        print()
        print("Query:", query)


        page = 1


        while len(identifiers) < DOCUMENT_LIMIT:

            print(
                f"  Search page {page}..."
            )


            try:

                results = internetarchive.search_items(

                    query,

                    fields=[
                        "identifier",
                        "title",
                        "creator",
                        "date",
                    ],

                    params={
                        "rows":
                            SEARCH_ROWS,

                        "page":
                            page,
                    }

                )


                count_before = len(
                    identifiers
                )


                for result in results:

                    identifier = result.get(
                        "identifier"
                    )


                    if not identifier:
                        continue


                    if identifier in seen:
                        continue


                    if identifier in local_seen:
                        continue


                    local_seen.add(
                        identifier
                    )

                    identifiers.append(
                        identifier
                    )


                    if len(identifiers) >= DOCUMENT_LIMIT:
                        break


                added = (
                    len(identifiers)
                    - count_before
                )


                print(
                    f"  Added {added} "
                    f"new identifiers."
                )


                if len(identifiers) >= DOCUMENT_LIMIT:
                    break


                # If nothing new was returned,
                # move to the next query.

                if added == 0:
                    break


                page += 1


            except Exception as exc:

                print(
                    "Search error:",
                    type(exc).__name__,
                    exc
                )

                break


    return identifiers


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_manifest()


    print("=" * 80)
    print("TELUGU LOW-RESOURCE CORPUS")
    print("INTERNET ARCHIVE OCR-TEXT COLLECTION")
    print("4-WORKER BENCHMARK")
    print("=" * 80)


    print(
        "Document limit:",
        DOCUMENT_LIMIT
    )

    print(
        "Workers:",
        MAX_WORKERS
    )

    print(
        "Minimum characters:",
        f"{MIN_CHARACTERS:,}"
    )

    print(
        "Minimum Telugu ratio:",
        f"{MIN_TELUGU_RATIO:.0%}"
    )

    print(
        "Raw OCR:",
        RAW_TEXT_DIR
    )

    print(
        "JSONL:",
        JSONL_DIR
    )

    print(
        "Manifest:",
        MANIFEST_FILE
    )


    # ========================================================
    # SEARCH
    # ========================================================

    identifiers = search_archive()


    print()
    print("=" * 80)

    print(
        "NEW DOCUMENTS FOUND:",
        len(identifiers)
    )

    print("=" * 80)


    if not identifiers:

        print(
            "No new documents found."
        )

        return


    # ========================================================
    # PARALLEL PROCESSING
    # ========================================================

    wall_start = time.perf_counter()


    results = []


    print()
    print(
        "Starting parallel Archive collection..."
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print("=" * 80)


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        future_map = {

            executor.submit(
                process_item,
                identifier,
                position,
                len(identifiers)
            ):
                identifier

            for position, identifier
            in enumerate(
                identifiers,
                start=1
            )

        }


        for future in as_completed(
            future_map
        ):

            identifier = future_map[
                future
            ]


            try:

                result = future.result()

                results.append(
                    result
                )

            except Exception as exc:

                print(
                    "Worker failed:",
                    identifier,
                    type(exc).__name__,
                    exc
                )


    wall_time = (
        time.perf_counter()
        - wall_start
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    completed = sum(
        1
        for r in results
        if r.get("status")
        == "COMPLETE"
    )


    too_short = sum(
        1
        for r in results
        if r.get("status")
        == "TOO_SHORT"
    )


    low_telugu = sum(
        1
        for r in results
        if r.get("status")
        == "LOW_TELUGU"
    )


    no_djvu = sum(
        1
        for r in results
        if r.get("status")
        == "NO_DJVU_TXT"
    )


    failed = sum(
        1
        for r in results
        if r.get("status")
        in {
            "DOWNLOAD_FAILED",
            "READ_FAILED",
            "CONVERSION_FAILED",
            "ERROR",
        }
    )


    total_chars = sum(
        r.get(
            "characters",
            0
        )
        for r in results
    )


    total_telugu = sum(
        r.get(
            "telugu",
            0
        )
        for r in results
    )


    overall_ratio = (

        total_telugu
        /
        total_chars

        if total_chars
        else 0

    )


    docs_per_minute = (

        len(identifiers)
        /
        (wall_time / 60)

        if wall_time
        else 0

    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("4-WORKER ARCHIVE BENCHMARK COMPLETE")
    print("=" * 80)


    print(
        "Documents processed :",
        len(identifiers)
    )

    print(
        "Accepted             :",
        completed
    )

    print(
        "Too short            :",
        too_short
    )

    print(
        "Low Telugu           :",
        low_telugu
    )

    print(
        "No _djvu.txt         :",
        no_djvu
    )

    print(
        "Failed               :",
        failed
    )


    print()

    print(
        "Total wall time      :",
        f"{wall_time:.2f} seconds"
    )

    print(
        "Wall time             :",
        f"{wall_time / 60:.2f} minutes"
    )

    print(
        "Documents/minute     :",
        f"{docs_per_minute:.2f}"
    )


    print()

    print(
        "Accepted characters  :",
        f"{total_chars:,}"
    )

    print(
        "Accepted Telugu      :",
        f"{total_telugu:,}"
    )

    print(
        "Overall Telugu       :",
        f"{overall_ratio:.2%}"
    )


    print()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_FILE
    )


    print()
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()