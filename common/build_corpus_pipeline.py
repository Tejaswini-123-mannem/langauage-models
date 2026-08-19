"""
Corpus pipeline for any configured language: filter, deduplicate, label.

Runs the four stages the Telugu corpus went through, in the same order,
producing the same per-stage statistics:

    1. quality filter      OCR-specific checks on Internet Archive text
    2. exact deduplication  SHA-256 over normalised text, global
    3. near deduplication   MinHash LSH at Jaccard 0.90, manual first
    4. corpus assembly      one labelled file per source

Relationship to the Telugu pipeline
-----------------------------------
Telugu ran these as four scripts writing intermediates to disk, because
its 1.17M documents and 2.8B characters made a single pass unwieldy and
because the near-duplicate stage alone took three hours and had to be
resumable.

Assamese has 394k documents. The stages are therefore run in one process
here, but they remain separate functions applied in the same order, and
the operations are identical. Only the intermediate persistence differs,
which does not affect any output. Keeping the reported statistics in the
same shape lets the two corpora be compared directly.

Why manual sources are deduplicated first
-----------------------------------------
Near-duplicate removal keeps whichever document was indexed first and
discards later matches. Manual sources are therefore processed before the
downloaded corpus, so a hand-collected document always survives and the
web crawl's copy is the one dropped. Within manual, the scarcer sources
go first for the same reason.

Usage
-----
    python build_corpus_pipeline.py --language assamese
    python build_corpus_pipeline.py --language assamese --skip-near-dedup
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import pickle
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output

use_utf8_output()

try:
    from datasketch import MinHash, MinHashLSH
    import numpy as np

except ImportError:
    sys.exit("Requires datasketch and numpy.\n\n    pip install datasketch numpy\n")


WORK = Path(r"C:\lma_work")


# =============================================================================
# SETTINGS  (identical to the Telugu pipeline)
# =============================================================================

NEAR_DUP_THRESHOLD = 0.90
NUM_PERM = 64
SHINGLE_SIZE = 5
MIN_WORDS = 20

# Quality filter, applied only to OCR sources.
OCR_MIN_CHARACTERS = 1000
OCR_MIN_SCRIPT_RATIO = 0.30
MAX_REPEATED_LINE_RATIO = 0.50
MAX_DIGIT_SYMBOL_RATIO = 0.65

# -----------------------------------------------------------------------------
# WORD PURITY
# -----------------------------------------------------------------------------
#
# MAX_DIGIT_SYMBOL_RATIO only catches catastrophic OCR: a document has to
# be more than 65% digits and symbols to fail it. That is the right bar
# for scan artefacts and index pages, and on the Assamese Archive books
# it rejected 5 documents out of 984.
#
# It does not catch a book the OCR read badly but not hopelessly. Measured
# over 1,238 accepted Assamese books, using the share of whitespace-
# separated tokens composed entirely of in-script characters:
#
#     clean      >=90%      274   22.1%
#     good     80-90%       612   49.4%
#     moderate 65-80%       204   16.5%
#     poor     50-65%        55    4.4%
#     degraded   <50%        93    7.5%     <- rejected by this gate
#
#     median purity 84.0%, p10 56.6%, min 6.8%
#
# The tail matters more than its size suggests. Purity falls because
# conjuncts break apart, and the same books carry a fragment rate (share
# of single-character tokens) reaching 30.7% against a 2.2% median. Word
# fragments are exactly what inflates tokenizer fertility: the Telugu
# report measured a 49% fertility penalty on its Archive source and
# attributed it to precisely this.
#
# So the gate is set at the bottom of the distribution, not the middle.
# Below 50% purity a document has lost more than half its words and
# contributes noise the tokenizer would learn as vocabulary. Between 50%
# and 80% the text is degraded but still mostly words, and discarding it
# would cost real tokens in a corpus that is already short of target.
#
# Cost: ~93 of 1,238 books, roughly 8-10M tokens.
# -----------------------------------------------------------------------------

MIN_WORD_PURITY = 0.50

# Characters stripped before judging whether a token is pure. Sentence
# punctuation sits outside the script block but is not OCR damage, and
# counting it as such would penalise well-scanned prose for being
# punctuated.
PURITY_STRIP = "।৷,.;:!?\"'()[]{}—–-…*"

# Purity is measured on a prefix rather than the whole document. Archive
# books reach 670,000 characters, the measure is stable well before that,
# and this runs once per document across the whole corpus.
PURITY_SAMPLE = 300_000

PROGRESS_EVERY = 20000


# =============================================================================
# SOURCES
# =============================================================================
#
# Order matters: manual before downloaded, scarcest manual first. See the
# module docstring.
# =============================================================================

def sources_for(language):

    root = WORK / language.name.lower() / "processed"

    return [
        # (source_type, source_group, path, is_ocr)
        ("wikisource", "manual", root / "wikisource" / "final", False),
        ("wikipedia", "manual", root / "wikipedia" / "final", False),
        ("archive", "manual", root / "archive" / "jsonl", True),
        ("sangraha", "sangraha", root / "sangraha" / "cleaned.jsonl", False),
    ]


def output_for(language):

    root = WORK / language.name.lower() / "corpus"

    return {
        "all": root / "all",
        "reports": root / "reports",
    }


# =============================================================================
# TEXT UTILITIES
# =============================================================================

WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalise(text):
    """NFC normalisation and whitespace tidying, for hashing and filtering."""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)

    return text.strip()


def get_words(text):
    return WORD_RE.findall(text.lower())


def get_shingles(text, k=SHINGLE_SIZE):
    words = get_words(text)

    if len(words) < k:
        return set(words)

    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def make_minhash(shingles):
    mh = MinHash(num_perm=NUM_PERM)

    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))

    return mh


def jaccard(a, b):
    """Estimate Jaccard similarity from two MinHash permutation vectors."""

    return np.count_nonzero(a == b) / float(NUM_PERM)


# =============================================================================
# STAGE 1: QUALITY FILTER  (OCR sources only)
# =============================================================================

def word_purity(text, language):
    """
    Share of tokens made entirely of in-script characters.

    A proxy for how well the OCR read the page. Clean Assamese prose
    scores 0.84 at the median; a book the engine could not resolve
    sprays Latin letters, digits and bracket characters through its
    words and falls far below that.

    Returns 1.0 for text with too few tokens to judge, so the decision
    is left to the length and script tests rather than being made on a
    handful of words.
    """

    tokens = text[:PURITY_SAMPLE].split()

    if len(tokens) < 50:
        return 1.0

    pure = 0

    for token in tokens:
        stripped = token.strip(PURITY_STRIP)

        if stripped and all(
            language.is_script_char(c) for c in stripped
        ):
            pure += 1

    return pure / len(tokens)


def quality_verdict(text, language):
    """
    Judge one OCR document.

    Wiki sources are filtered at collection time, where markup has to be
    cleaned before quality can be assessed at all. Internet Archive text
    is raw OCR and needs checks those sources do not: pages where OCR
    looped and repeated a line, pages that are mostly digits and
    punctuation such as tables and indexes, and books the OCR read so
    poorly that most of their words came out broken.
    """

    if len(text) < OCR_MIN_CHARACTERS:
        return "TOO_SHORT"

    if language.script_ratio(text) < OCR_MIN_SCRIPT_RATIO:
        return "LOW_SCRIPT"

    if language.required_chars:
        if language.required_char_density(text) < language.required_char_ratio:
            return "WRONG_LANGUAGE"

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if len(lines) >= 10:
        unique = len(set(lines))

        if 1 - (unique / len(lines)) > MAX_REPEATED_LINE_RATIO:
            return "HIGH_REPETITION"

    dense = [c for c in text if not c.isspace()]

    if dense:
        junk = sum(
            1 for c in dense
            if c.isdigit()
            or (not c.isalpha() and not language.is_script_char(c))
        )

        if junk / len(dense) > MAX_DIGIT_SYMBOL_RATIO:
            return "OCR_GARBAGE"

    # Last, because it is the most expensive test and the cheap ones
    # above have already removed the documents it would agree with.
    if word_purity(text, language) < MIN_WORD_PURITY:
        return "LOW_PURITY"

    return "KEEP"


# =============================================================================
# READING
# =============================================================================

def iter_source(path):
    """Yield records from a file or every .jsonl under a directory."""

    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))

    for file_path in files:

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                try:
                    yield json.loads(line)

                except json.JSONDecodeError:
                    yield None


def make_doc_id(source_type, text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
    return f"{source_type}_{digest}"


def get_origin(record, source_type):

    if source_type in ("wikisource", "wikipedia"):
        return record.get("title") or "unknown"

    if source_type == "archive":
        book = record.get("book") or record.get("doc_id") or "unknown"
        page = record.get("page")
        return f"{book}#page{page}" if page else book

    return record.get("type") or "unknown"



# =============================================================================
# CHECKPOINTING
# =============================================================================
#
# This pipeline holds all of its deduplication state in memory: one
# MinHash LSH index plus a signature and hash for every document kept so
# far. A run over Assamese takes roughly two hours.
#
# An earlier run was stopped 35 minutes in with nothing recoverable,
# because the index cannot be rebuilt from the output files without
# re-shingling every document. The output .jsonl files look like progress
# but a restart reopens them with "w" and overwrites them.
#
# So the state is written to disk after each source completes. A source
# already recorded as done is skipped on the next run and its output file
# left untouched. Sources are the natural granularity here: the largest
# is Sangraha, and everything before it finishes in minutes.
# =============================================================================

def checkpoint_path(language):
    return output_for(language)["reports"] / "pipeline_checkpoint.pkl"


def save_checkpoint(language, lsh, signatures, seen_hashes, stats, key_counter):
    """Persist pipeline state atomically after a source completes."""

    path = checkpoint_path(language)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "lsh": lsh,
        "signatures": signatures,
        "seen_hashes": seen_hashes,
        "stats": stats,
        "key_counter": key_counter,
        "settings": {
            "threshold": NEAR_DUP_THRESHOLD,
            "num_perm": NUM_PERM,
            "shingle_size": SHINGLE_SIZE,
            "min_words": MIN_WORDS,
        },
    }

    temp = path.with_suffix(".tmp")

    start = time.time()

    with open(temp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    temp.replace(path)

    print(
        f"    checkpoint saved: {len(signatures):,} signatures, "
        f"{path.stat().st_size / 1024**2:,.0f} MB, {time.time()-start:.1f}s"
    )


def load_checkpoint(language):
    """Restore state from a previous run, or None."""

    path = checkpoint_path(language)

    if not path.exists():
        return None

    print()
    print(f"Loading checkpoint: {path}")

    start = time.time()

    with open(path, "rb") as f:
        payload = pickle.load(f)

    current = {
        "threshold": NEAR_DUP_THRESHOLD,
        "num_perm": NUM_PERM,
        "shingle_size": SHINGLE_SIZE,
        "min_words": MIN_WORDS,
    }

    if payload.get("settings") != current:
        print("Checkpoint was written with different dedup settings.")
        print(f"  checkpoint: {payload.get('settings')}")
        print(f"  current   : {current}")
        print("Delete the checkpoint to start fresh, or restore the settings.")
        sys.exit(1)

    print(
        f"  restored {len(payload['signatures']):,} signatures in "
        f"{time.time()-start:.1f}s"
    )
    print(f"  sources already done: {sorted(payload['stats'])}")

    return payload


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(language, skip_near_dedup=False):

    sources = sources_for(language)
    out = output_for(language)

    out["all"].mkdir(parents=True, exist_ok=True)
    out["reports"].mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"CORPUS PIPELINE: {language.name} ({language.endonym})")
    print("=" * 78)
    print(f"  near-dup threshold : {NEAR_DUP_THRESHOLD}")
    print(f"  minhash perms      : {NUM_PERM}")
    print(f"  shingle size       : {SHINGLE_SIZE} words")

    if language.required_chars:
        print(
            f"  marker test        : {' '.join(language.required_chars)} "
            f"at >= {language.required_char_ratio:.2%}"
        )

    checkpoint = load_checkpoint(language)

    if checkpoint:
        lsh = checkpoint["lsh"]
        signatures = checkpoint["signatures"]
        seen_hashes = checkpoint["seen_hashes"]
        stats = checkpoint["stats"]
        key_counter = checkpoint["key_counter"]
        print("  RESUMING - completed sources will be skipped")
    else:
        lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=NUM_PERM)
        signatures = {}
        seen_hashes = set()
        stats = {}
        key_counter = 0

    ledger_path = out["reports"] / "removed_documents.csv"

    resuming = checkpoint is not None

    ledger_file = open(
        ledger_path, "a" if resuming else "w", encoding="utf-8", newline=""
    )
    ledger = csv.writer(ledger_file)

    if not resuming:
        ledger.writerow([
            "source_type", "stage", "reason", "doc_id",
            "characters", "matched_key", "similarity",
        ])

    try:

        for source_type, source_group, path, is_ocr in sources:

            if not path.exists():
                print(f"\n  SKIP {source_type}: not found at {path}")
                continue

            print()
            print("-" * 78)
            print(f"{source_type.upper()}  ({source_group}{', OCR' if is_ocr else ''})")
            print("-" * 78)

            s = {
                "read": 0, "invalid": 0, "quality_removed": 0,
                "exact_removed": 0, "near_removed": 0,
                "kept": 0, "characters": 0,
            }

            quality_reasons = {}

            output_file = out["all"] / f"corpus_{source_type}.jsonl"

            start = time.time()

            with open(output_file, "w", encoding="utf-8") as fout:

                for record in iter_source(path):

                    if record is None:
                        s["invalid"] += 1
                        continue

                    s["read"] += 1

                    text = record.get("text")

                    if not isinstance(text, str) or not text.strip():
                        s["invalid"] += 1
                        continue

                    text = normalise(text)

                    if not text:
                        s["invalid"] += 1
                        continue

                    # ------------------------------------------------------
                    # STAGE 1: QUALITY FILTER (OCR only)
                    # ------------------------------------------------------

                    if is_ocr:
                        verdict = quality_verdict(text, language)

                        if verdict != "KEEP":
                            s["quality_removed"] += 1
                            quality_reasons[verdict] = (
                                quality_reasons.get(verdict, 0) + 1
                            )
                            continue

                    # ------------------------------------------------------
                    # STAGE 2: EXACT DEDUPLICATION
                    # ------------------------------------------------------

                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

                    if digest in seen_hashes:
                        s["exact_removed"] += 1
                        ledger.writerow([
                            source_type, "exact", "DUPLICATE",
                            record.get("doc_id", ""), len(text), "", "",
                        ])
                        continue

                    seen_hashes.add(digest)

                    # ------------------------------------------------------
                    # STAGE 3: NEAR DEDUPLICATION
                    # ------------------------------------------------------

                    key_counter += 1

                    words = get_words(text)

                    indexable = (
                        not skip_near_dedup and len(words) >= MIN_WORDS
                    )

                    if indexable:

                        shingles = get_shingles(text)

                        if shingles:

                            mh = make_minhash(shingles)

                            best_sim = 0.0
                            best_key = None
                            duplicate = False

                            for candidate in lsh.query(mh):

                                sig = signatures.get(candidate)

                                if sig is None:
                                    continue

                                sim = jaccard(mh.hashvalues, sig)

                                if sim > best_sim:
                                    best_sim, best_key = sim, candidate

                                if sim >= NEAR_DUP_THRESHOLD:
                                    duplicate = True
                                    break

                            if duplicate:
                                s["near_removed"] += 1
                                ledger.writerow([
                                    source_type, "near", "NEAR_DUPLICATE",
                                    record.get("doc_id", ""), len(text),
                                    best_key, f"{best_sim:.6f}",
                                ])
                                continue

                            key = f"{source_type}:{key_counter}"
                            lsh.insert(key, mh)
                            signatures[key] = mh.hashvalues.copy()

                    # ------------------------------------------------------
                    # STAGE 4: LABEL AND WRITE
                    # ------------------------------------------------------

                    fout.write(
                        json.dumps(
                            {
                                "doc_id": make_doc_id(source_type, text),
                                "text": text,
                                "source_group": source_group,
                                "source_type": source_type,
                                "char_count": len(text),
                                "origin": get_origin(record, source_type),
                                "language": language.code_3,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    s["kept"] += 1
                    s["characters"] += len(text)

                    if s["read"] % PROGRESS_EVERY == 0:
                        rate = s["read"] / max(time.time() - start, 1e-9)
                        print(
                            f"    {s['read']:>8,} read | "
                            f"{s['kept']:>8,} kept | {rate:,.0f}/sec"
                        )

            s["seconds"] = round(time.time() - start, 1)
            s["quality_reasons"] = quality_reasons
            s["source_group"] = source_group

            stats[source_type] = s

            save_checkpoint(
                language, lsh, signatures, seen_hashes, stats, key_counter
            )

            print(f"    read            : {s['read']:>10,}")
            print(f"    invalid         : {s['invalid']:>10,}")

            if is_ocr:
                print(f"    quality removed : {s['quality_removed']:>10,}")
                for reason, count in sorted(
                    quality_reasons.items(), key=lambda x: -x[1]
                ):
                    print(f"        {reason:<16} {count:>8,}")

            print(f"    exact dupes     : {s['exact_removed']:>10,}")
            print(f"    near dupes      : {s['near_removed']:>10,}")
            print(f"    KEPT            : {s['kept']:>10,}")
            print(f"    characters      : {s['characters']:>10,}")
            print(f"    time            : {s['seconds']:>10,.1f}s")

    finally:
        ledger_file.close()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    summary_path = out["reports"] / "pipeline_summary.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as f:

        w = csv.writer(f)
        w.writerow([
            "source_type", "source_group", "read", "invalid",
            "quality_removed", "exact_removed", "near_removed",
            "kept", "characters", "seconds",
        ])

        for source_type, s in stats.items():
            w.writerow([
                source_type, s["source_group"], s["read"], s["invalid"],
                s["quality_removed"], s["exact_removed"], s["near_removed"],
                s["kept"], s["characters"], s["seconds"],
            ])

    total_kept = sum(s["kept"] for s in stats.values())
    total_chars = sum(s["characters"] for s in stats.values())
    total_read = sum(s["read"] for s in stats.values())

    manual_chars = sum(
        s["characters"] for s in stats.values()
        if s["source_group"] == "manual"
    )

    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)

    print(f"{'source':<12} {'kept':>10} {'characters':>16} {'share':>8}")
    print("-" * 50)

    for source_type, s in stats.items():
        share = s["characters"] / total_chars if total_chars else 0
        print(
            f"{source_type:<12} {s['kept']:>10,} "
            f"{s['characters']:>16,} {share:>7.2%}"
        )

    print("-" * 50)
    print(f"{'TOTAL':<12} {total_kept:>10,} {total_chars:>16,}")

    print()
    print(f"  documents read     : {total_read:,}")
    print(f"  documents kept     : {total_kept:,}")
    print(f"  manual characters  : {manual_chars:,}")

    if total_chars:
        print(f"  manual char share  : {manual_chars / total_chars:.2%}")

    print()
    print(f"  corpus  : {out['all']}")
    print(f"  summary : {summary_path}")
    print(f"  ledger  : {ledger_path}")
    print("=" * 78)


def main():

    parser = argparse.ArgumentParser(
        description="Filter, deduplicate and label a language corpus."
    )

    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--skip-near-dedup",
        action="store_true",
        help="Run quality and exact dedup only (much faster).",
    )

    args = parser.parse_args()

    run(get_language(args.language), args.skip_near_dedup)


if __name__ == "__main__":
    main()
