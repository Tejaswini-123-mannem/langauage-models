"""
Rebuild the global near-duplicate summary from the output files.

Why this exists
---------------
The near-dedup run was interrupted and resumed several times. Its
in-memory counters only covered work done after the last checkpoint, so
the summary it wrote understated Internet Archive: 1,238 records and
163.0M characters against the 1,631 records and 272.2M characters
actually on disk.

The corpus itself is complete and correct; only the report was wrong.
The underlying counter bug is fixed in global_near_dedup.py, but
re-running three hours of deduplication purely to regenerate a summary
CSV would be wasteful, so the summary is recomputed from the artefacts
instead.

Counting the output directly is also the more trustworthy method for a
figure that goes into the Phase 1 report: it describes the corpus that
exists, rather than what a counter believed it had produced.

Input counts come from the global exact-dedup summary, which was written
in a single uninterrupted run and is therefore reliable.
"""

import csv
import json
import sys
from pathlib import Path


WORK = Path(r"C:\lma_work")

NEAR_DEDUP_DIR = WORK / "global_near_dedup"

EXACT_SUMMARY = WORK / "global_dedup" / "reports" / "global_exact_dedup_summary.csv"

REPORT_DIR = NEAR_DEDUP_DIR / "reports"

SUMMARY_FILE = REPORT_DIR / "global_near_dedup_summary.csv"

LEDGER_FILE = REPORT_DIR / "near_duplicates_removed.csv"


SOURCES = [
    ("manual", "textbook", NEAR_DEDUP_DIR / "manual" / "textbook"),
    ("manual", "wikisource", NEAR_DEDUP_DIR / "manual" / "wikisource"),
    ("manual", "wikipedia", NEAR_DEDUP_DIR / "manual" / "wikipedia"),
    ("manual", "archive", NEAR_DEDUP_DIR / "manual" / "archive"),
    ("sangraha", "sangraha", NEAR_DEDUP_DIR / "sangraha"),
]


def count_output(directory):
    """Return (files, records, characters) actually present on disk."""

    files = sorted(directory.rglob("*.jsonl"))

    records = 0
    characters = 0

    for path in files:

        with open(path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                records += 1
                characters += len(record.get("text", "") or "")

    return len(files), records, characters


def load_input_counts():
    """Read per-source input counts from the exact-dedup summary."""

    counts = {}

    if not EXACT_SUMMARY.exists():
        print(f"WARNING: {EXACT_SUMMARY} not found; inputs unknown.")
        return counts

    with open(EXACT_SUMMARY, "r", encoding="utf-8") as f:

        for row in csv.DictReader(f):

            counts[row["source"]] = {
                "records": int(row["records_kept"]),
                "characters": int(row["characters_kept"]),
            }

    return counts


def load_removed_characters():
    """Sum removed characters per source from the removal ledger."""

    removed = {}

    if not LEDGER_FILE.exists():
        return removed

    with open(LEDGER_FILE, "r", encoding="utf-8") as f:

        for row in csv.DictReader(f):

            source = row.get("removed_source_type")

            if not source:
                continue

            try:
                chars = int(row.get("removed_characters") or 0)

            except ValueError:
                chars = 0

            entry = removed.setdefault(
                source, {"records": 0, "characters": 0}
            )

            entry["records"] += 1
            entry["characters"] += chars

    return removed


def main():

    if not NEAR_DEDUP_DIR.exists():
        sys.exit(f"Near-dedup output not found: {NEAR_DEDUP_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_input_counts()
    removed = load_removed_characters()

    print()
    print("=" * 78)
    print("REBUILDING NEAR-DEDUP SUMMARY FROM OUTPUT")
    print("=" * 78)

    rows = []

    header = (
        f"{'source':<12} {'files':>7} {'in':>12} {'kept':>12} "
        f"{'removed':>9} {'characters':>16}"
    )

    print(header)
    print("-" * len(header))

    for source_group, source_type, directory in SOURCES:

        if not directory.exists():
            print(f"{source_type:<12} MISSING")
            continue

        files, records, characters = count_output(directory)

        in_records = inputs.get(source_type, {}).get("records", 0)

        removed_records = (
            in_records - records if in_records else
            removed.get(source_type, {}).get("records", 0)
        )

        removed_chars = removed.get(source_type, {}).get("characters", 0)

        print(
            f"{source_type:<12} {files:>7,} {in_records:>12,} "
            f"{records:>12,} {removed_records:>9,} {characters:>16,}"
        )

        rows.append({
            "source_group": source_group,
            "source_type": source_type,
            "files": files,
            "records_processed": in_records,
            "records_kept": records,
            "near_duplicates_removed": removed_records,
            "invalid": 0,
            "characters_kept": characters,
            "characters_removed": removed_chars,
        })

    with open(SUMMARY_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_group", "source_type", "files",
                "records_processed", "records_kept",
                "near_duplicates_removed", "invalid",
                "characters_kept", "characters_removed",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    total_in = sum(r["records_processed"] for r in rows)
    total_kept = sum(r["records_kept"] for r in rows)
    total_removed = sum(r["near_duplicates_removed"] for r in rows)
    total_chars = sum(r["characters_kept"] for r in rows)

    manual_chars = sum(
        r["characters_kept"] for r in rows
        if r["source_group"] == "manual"
    )

    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {'':>7} {total_in:>12,} {total_kept:>12,} "
        f"{total_removed:>9,} {total_chars:>16,}"
    )

    print()
    print(f"Near-duplicate removal rate : {total_removed / total_in:.4%}")
    print(f"Manual characters           : {manual_chars:,}")
    print(
        f"Manual character share      : {manual_chars / total_chars:.2%}"
    )
    print()
    print(f"Summary written: {SUMMARY_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
