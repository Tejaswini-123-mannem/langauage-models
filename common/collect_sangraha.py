"""
Sangraha collection for any configured language.

Downloads the parquet files of a Sangraha `verified/<lang>` subset and
runs the same cleaning and filtering the Telugu pipeline used, so the two
corpora are comparable.

Relationship to telugu/preprocess/process_sangraha_shards.py
------------------------------------------------------------
Telugu's Sangraha data was already in the local HuggingFace cache as
Arrow shards, downloaded by notebook 01, so that script reads Arrow and
never needed a download step. Assamese is not cached, so this module
fetches the parquet files directly over HTTP.

The cleaning is deliberately identical: clean_text below is the same
function, ported from notebook 01 cells 29-31 and used unchanged for
Telugu shards 4 and 5. Only the language test differs, and that comes
from the LanguageConfig rather than being hardcoded.

Why direct HTTP rather than the datasets library
------------------------------------------------
`datasets` would pull in a large dependency tree and its own cache
layout, for the sake of downloading three files whose URLs are already
known. HuggingFace serves them over plain HTTPS, and pyarrow reads
parquet directly, so a request and a loop are enough.

Usage
-----
    python collect_sangraha.py --language assamese --list
    python collect_sangraha.py --language assamese --download
    python collect_sangraha.py --language assamese --process
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================

WORK = Path(r"C:\lma_work")

REPO = "ai4bharat/sangraha"

API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"

RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

USER_AGENT = "Mozilla/5.0 (LMA course project)"

# Matches the Telugu pipeline exactly.
MIN_CHARACTERS = 50
MIN_SCRIPT_RATIO = 0.10


def paths_for(language):

    root = WORK / language.name.lower()

    return {
        "raw": root / "raw" / "sangraha",
        "processed": root / "processed" / "sangraha",
        "cleaned": root / "processed" / "sangraha" / "cleaned.jsonl",
        "hashes": root / "processed" / "sangraha" / "dedup_hashes.txt",
        "state": root / "processed" / "sangraha" / "state.json",
    }


# =============================================================================
# CLEANING  (ported verbatim from notebook 01, cell 29)
# =============================================================================

def clean_text(text):
    """
    Conservative normalisation of one Sangraha document.

    Identical to the function used for the Telugu corpus. It fixes
    encoding and whitespace inconsistencies and strips the lecture-slide
    metadata carried by Sangraha's speech subset, but does not attempt to
    repair or rewrite the language.

    Any change here would have to be applied to both languages and both
    corpora reprocessed, or Model H and Model L would no longer be
    comparable.
    """

    text = unicodedata.normalize("NFC", text)

    text = re.sub(
        r"\(Refer Slide Time:\s*\d+:\d+\)", " ", text, flags=re.IGNORECASE
    )

    text = re.sub(r"\bSlide\s+\d+\s*\|?", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# =============================================================================
# LISTING AND DOWNLOAD
# =============================================================================

def list_files(language):
    """Return (filename, size) for the language's Sangraha subset."""

    url = f"{API}/{language.sangraha_subset}"

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(4):

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                entries = json.load(response)

            return [
                (e["path"].split("/")[-1], e.get("size") or 0)
                for e in entries
                if e["type"] == "file" and e["path"].endswith(".parquet")
            ]

        except Exception as exc:
            # HuggingFace rate-limits anonymous API calls, so back off
            # rather than failing the run.
            print(f"  attempt {attempt + 1}: {type(exc).__name__}")
            time.sleep(15 * (attempt + 1))

    raise RuntimeError("Could not list Sangraha files after 4 attempts")


def download(language):
    """Fetch every parquet file of the subset that is not already present."""

    p = paths_for(language)
    p["raw"].mkdir(parents=True, exist_ok=True)

    files = list_files(language)

    total = sum(size for _, size in files)

    print()
    print("=" * 78)
    print(f"DOWNLOADING SANGRAHA {language.sangraha_subset}")
    print("=" * 78)
    print(f"Files: {len(files)}   Total: {total / 1024**3:,.2f} GB")
    print(f"Into : {p['raw']}")

    for name, size in files:

        target = p["raw"] / name

        if target.exists() and target.stat().st_size == size:
            print(f"  {name:<24} already present ({size / 1024**2:,.1f} MB)")
            continue

        url = f"{RESOLVE}/{language.sangraha_subset}/{name}"

        temp = target.with_suffix(".part")

        print(f"  {name:<24} downloading {size / 1024**2:,.1f} MB ...")

        start = time.perf_counter()

        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )

        with urllib.request.urlopen(request, timeout=180) as response:

            got = 0

            with open(temp, "wb") as f:

                while True:
                    block = response.read(1024 * 512)

                    if not block:
                        break

                    f.write(block)
                    got += len(block)

        # Renamed only on completion, so an interrupted transfer cannot be
        # mistaken for a finished file.
        temp.replace(target)

        elapsed = time.perf_counter() - start

        print(
            f"  {name:<24} done {got / 1024**2:,.1f} MB in {elapsed:.0f}s "
            f"({got / elapsed / 1024**2:.2f} MB/s)"
        )

    print("=" * 78)


# =============================================================================
# PROCESSING
# =============================================================================

def iter_parquet_rows(path):
    """Yield row dictionaries from a parquet file, batch by batch."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(batch_size=2000):

        columns = {
            name: batch.column(name).to_pylist()
            for name in batch.schema.names
        }

        for i in range(batch.num_rows):
            yield {name: columns[name][i] for name in columns}


def process(language):
    """
    Clean, language-filter and exact-deduplicate the downloaded parquet.

    Deduplication is by SHA-256 of the cleaned text, matching the Telugu
    pipeline, and the hash set is persisted so the stage is resumable.
    """

    p = paths_for(language)

    p["processed"].mkdir(parents=True, exist_ok=True)

    files = sorted(p["raw"].glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files in {p['raw']}\nRun --download first."
        )

    print()
    print("=" * 78)
    print(f"PROCESSING SANGRAHA {language.name.upper()}")
    print("=" * 78)
    print(f"Files : {len(files)}")
    print(f"Output: {p['cleaned']}")

    counters = {
        "rows": 0, "empty": 0, "short": 0,
        "wrong_script": 0, "wrong_language": 0,
        "duplicate": 0, "kept": 0,
    }

    seen = set()

    characters = 0

    start = time.perf_counter()

    temp = p["cleaned"].with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as fout:

        for path in files:

            print(f"  reading {path.name} ...")

            for row in iter_parquet_rows(path):

                counters["rows"] += 1

                cleaned = clean_text(row.get("text") or "")

                if not cleaned:
                    counters["empty"] += 1
                    continue

                if len(cleaned) < MIN_CHARACTERS:
                    counters["short"] += 1
                    continue

                if language.script_ratio(cleaned) < MIN_SCRIPT_RATIO:
                    counters["wrong_script"] += 1
                    continue

                # Separates the target language from a neighbour sharing
                # the same script. A no-op for unambiguous scripts.
                if language.required_chars:
                    if (
                        language.required_char_density(cleaned)
                        < language.required_char_ratio
                    ):
                        counters["wrong_language"] += 1
                        continue

                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

                if digest in seen:
                    counters["duplicate"] += 1
                    continue

                seen.add(digest)

                fout.write(
                    json.dumps(
                        {
                            "doc_id": row.get("doc_id"),
                            "text": cleaned,
                            "type": row.get("type"),
                            "language": language.code_3,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                counters["kept"] += 1
                characters += len(cleaned)

                if counters["rows"] % 50_000 == 0:
                    print(
                        f"    {counters['rows']:,} rows, "
                        f"{counters['kept']:,} kept"
                    )

    temp.replace(p["cleaned"])

    with open(p["hashes"], "w", encoding="utf-8") as f:
        for digest in seen:
            f.write(digest + "\n")

    elapsed = time.perf_counter() - start

    print()
    print("=" * 78)
    print("SANGRAHA PROCESSING COMPLETE")
    print("=" * 78)

    for key, value in counters.items():
        print(f"  {key:<16} {value:>12,}")

    print(f"  {'characters':<16} {characters:>12,}")

    if counters["rows"]:
        print(f"  {'retention':<16} {counters['kept'] / counters['rows']:>12.2%}")

    print(f"  {'time':<16} {elapsed / 60:>12.2f} min")
    print()
    print(f"  est tokens      : {int(characters / 4.0):,}  (at ~4.0 chars/token)")
    print(f"  output          : {p['cleaned']}")
    print(f"  size            : {p['cleaned'].stat().st_size / 1024**3:,.2f} GB")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Download and process a Sangraha language subset."
    )

    parser.add_argument("--language", required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--process", action="store_true")

    args = parser.parse_args()

    if not any([args.list, args.download, args.process]):
        parser.error("Choose at least one of --list/--download/--process")

    language = get_language(args.language)

    print()
    print(f"Language : {language.name} ({language.endonym}) [{language.code_3}]")
    print(f"Subset   : {language.sangraha_subset}")

    if language.required_chars:
        print(
            f"Markers  : {' '.join(language.required_chars)} "
            f"(min density {language.required_char_ratio:.2%})"
        )

    if args.list:
        files = list_files(language)
        total = sum(s for _, s in files)
        print()
        print(f"{len(files)} files, {total / 1024**3:,.2f} GB")
        for name, size in files:
            print(f"  {name:<28} {size / 1024**2:>9,.1f} MB")

    if args.download:
        download(language)

    if args.process:
        process(language)


if __name__ == "__main__":
    main()
