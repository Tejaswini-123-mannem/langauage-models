"""
Stage 2 of corpus assembly, for either language: plan the source mix.

The assignment states two things in TOKENS, not characters:

    * a target of approximately 500M training tokens per language
    * at least 20% of final training tokens must come from manual
      collection

The assembled corpus is measured in characters, so this script converts
one to the other using the provisional tokenizer, then decides how much
Sangraha to keep.

The mixing rule
---------------
Manual data (textbook OCR, Wikisource, Internet Archive) is scarce and was
expensive to collect, so all of it is kept. Sangraha is abundant, so it is
the side that gets trimmed:

    manual_used    = every manual token available
    sangraha_used  = min(TARGET_SANGRAHA, MANUAL_MULTIPLIER * manual_used)

The multiplier is 4, because manual = 20% of the total is the same
statement as sangraha = 4 x manual. So:

    * If manual reaches the ~100M target, Sangraha caps at ~400M and the
      corpus lands near 500M tokens with manual at or above 20%.

    * If manual falls short, Sangraha is reduced to four times whatever
      manual actually yielded. The corpus is then smaller than 500M, but
      the 20% manual floor still holds, which is the requirement that is
      graded pass/fail rather than approximately.

Estimating tokens without tokenizing everything
-----------------------------------------------
Tokenizing 1.6 billion characters just to count them would be wasteful.
Characters-per-token is very stable within a source, so it is measured on
a random document sample per source and the total is extrapolated. The
sample size is reported so the estimate can be sanity-checked, and the
true counts are recomputed exactly in make_splits.py when the corpus is
actually written.

Which sources exist, the order the manual budget is spent in, and any
per-source ceiling are properties of the language, not of this script, so
they live in language_config.py. See LanguageConfig.corpus_sources,
manual_priority and manual_source_caps, each with the reasoning for the
value it holds.

Usage
-----
    python plan_mix.py --language telugu --tokenizer <path to .model>
    python plan_mix.py --language assamese --tokenizer <path to .model> \\
        --target-total 200e6 --target-manual 120e6
"""

import argparse
import json
import random
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Manual must be at least 20% of final training tokens, and
# 20% manual is equivalent to sangraha = 4 x manual.
MANUAL_MULTIPLIER = 4.0

MIN_MANUAL_SHARE = 0.20

# Documents sampled per source when measuring characters-per-token.
SAMPLE_DOCUMENTS = 3000

# Cap on characters actually tokenized per source, so one source of whole
# books does not dominate the measurement time.
SAMPLE_CHAR_BUDGET = 20_000_000


# =============================================================================
# MEASUREMENT
# =============================================================================

def count_source(path):
    """
    Return (documents, characters) for one corpus file.

    A full pass is cheap relative to tokenization and gives exact totals,
    which is what the extrapolation is applied to.
    """

    documents = 0
    characters = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get("text") or ""

            documents += 1
            characters += len(text)

    return documents, characters


def sample_documents(path, total_documents, sample_size, seed):
    """
    Yield a roughly uniform random sample of documents from a corpus file.

    Uses independent per-document acceptance rather than reservoir
    sampling, so the file is streamed once and nothing large is buffered.
    """

    if total_documents == 0:
        return

    rate = min(1.0, sample_size / total_documents)

    rng = random.Random(seed)

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            if rng.random() > rate:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get("text")

            if isinstance(text, str) and text.strip():
                yield text


def measure_chars_per_token(sp, path, total_documents, seed):
    """
    Estimate characters-per-token for one source.

    Returns (chars_per_token, sampled_documents, sampled_characters,
    sampled_tokens).
    """

    sampled_chars = 0
    sampled_tokens = 0
    sampled_docs = 0

    for text in sample_documents(
        path, total_documents, SAMPLE_DOCUMENTS, seed
    ):

        # Whole books are truncated for the measurement. Fertility does not
        # drift meaningfully within a document, and this keeps one Archive
        # book from consuming the entire sample budget.
        if len(text) > 200_000:
            text = text[:200_000]

        sampled_chars += len(text)
        sampled_tokens += len(sp.encode(text, out_type=int))
        sampled_docs += 1

        if sampled_chars >= SAMPLE_CHAR_BUDGET:
            break

    if sampled_tokens == 0:
        return 0.0, sampled_docs, sampled_chars, sampled_tokens

    return (
        sampled_chars / sampled_tokens,
        sampled_docs,
        sampled_chars,
        sampled_tokens,
    )


# =============================================================================
# ALLOCATION
# =============================================================================

def allocate_manual(measurements, target_manual, manual_priority, caps):
    """
    Decide how much of each manual source to keep.

    Sources are consumed in the language's manual_priority order until the
    manual budget runs out. Two limits apply to each source:

      * its own ceiling in the language's manual_source_caps, if it has one
      * whatever remains of the overall manual budget

    The source that straddles the remaining budget is kept fractionally,
    and anything after it gets nothing.

    Keep fractions are expressed against the tokens a source ACTUALLY
    holds, not against its capped allowance, because make_splits.py
    applies them to the real corpus files.

    Returns (tokens_used, {source_type: keep_fraction}).
    """

    keep_fractions = {}

    remaining = target_manual
    used = 0

    for source_type in manual_priority:

        measurement = measurements.get(source_type)

        if not measurement:
            continue

        available = measurement["estimated_tokens"]

        if available <= 0:
            keep_fractions[source_type] = 0.0
            continue

        if remaining <= 0:
            # Budget already exhausted by higher-priority sources.
            keep_fractions[source_type] = 0.0
            continue

        # A source never contributes more than its own ceiling, so an
        # unexpectedly large source cannot crowd out the ones below it.
        allowance = min(available, caps.get(source_type, available))

        take = min(allowance, remaining)

        keep_fractions[source_type] = take / available

        used += take
        remaining -= take

    return int(used), keep_fractions


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plan the downloaded / manual token mix for one language."
    )

    add_language_argument(parser)

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the provisional SentencePiece .model file.",
    )

    parser.add_argument(
        "--target-total",
        type=float,
        default=550_000_000,
        help="Target total training tokens, treated as a ceiling "
             "(default: 550M).",
    )

    parser.add_argument(
        "--target-manual",
        type=float,
        default=120_000_000,
        help="Target manual training tokens (default: 120M). Manual is "
             "capped at this value rather than being kept whole, so the "
             "corpus does not overshoot the total.",
    )

    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        metavar="SOURCE=TOKENS",
        help="Override a per-source ceiling, e.g. --cap wikipedia=60e6. "
             "Repeatable. Use TOKENS=0 to remove a source's ceiling. "
             "Defaults come from the language's manual_source_caps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    sources = language.corpus_sources
    manual_priority = language.manual_priority

    # -------------------------------------------------------------------------
    # PER-SOURCE CEILINGS
    # -------------------------------------------------------------------------

    caps = dict(language.manual_source_caps)

    for override in args.cap:

        if "=" not in override:
            parser.error(f"--cap expects SOURCE=TOKENS, got {override!r}")

        source_type, _, value = override.partition("=")

        source_type = source_type.strip()

        try:
            tokens = int(float(value))

        except ValueError:
            parser.error(f"--cap value must be a number, got {value!r}")

        if tokens <= 0:
            caps.pop(source_type, None)
        else:
            caps[source_type] = tokens

    target_sangraha = args.target_total - args.target_manual

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    print()
    print("=" * 78)
    print(f"{workspace.title} - SOURCE MIX PLANNING")
    print("=" * 78)
    print(f"Corpus           : {workspace.corpus_all}")
    print(f"Tokenizer        : {args.tokenizer}")
    print(f"Vocabulary size  : {sp.get_piece_size():,}")
    print(f"Target total     : {args.target_total / 1e6:,.0f}M tokens")
    print(f"Target manual    : {args.target_manual / 1e6:,.0f}M tokens")
    print(f"Target sangraha  : {target_sangraha / 1e6:,.0f}M tokens")

    # -------------------------------------------------------------------------
    # MEASURE EACH SOURCE
    # -------------------------------------------------------------------------

    measurements = {}

    print()
    print("Measuring sources...")

    for source_type, source_group in sources:

        path = workspace.source_file(source_type)

        if not path.exists():
            print(f"  WARNING: missing {path.name}, source skipped.")
            continue

        documents, characters = count_source(path)

        cpt, s_docs, s_chars, s_tokens = measure_chars_per_token(
            sp, path, documents, args.seed
        )

        estimated_tokens = (
            int(characters / cpt) if cpt > 0 else 0
        )

        measurements[source_type] = {
            "source_group": source_group,
            "documents": documents,
            "characters": characters,
            "chars_per_token": round(cpt, 4),
            "estimated_tokens": estimated_tokens,
            "sampled_documents": s_docs,
            "sampled_characters": s_chars,
            "sampled_tokens": s_tokens,
        }

        print(
            f"  {source_type:<12} "
            f"docs={documents:>9,}  "
            f"chars={characters:>15,}  "
            f"chars/token={cpt:5.2f}  "
            f"est tokens={estimated_tokens:>13,}"
        )

    # -------------------------------------------------------------------------
    # APPLY THE MIXING RULE
    # -------------------------------------------------------------------------

    manual_available = sum(
        m["estimated_tokens"] for m in measurements.values()
        if m["source_group"] == "manual"
    )

    sangraha_available = measurements.get(
        "sangraha", {}
    ).get("estimated_tokens", 0)

    # -------------------------------------------------------------------------
    # MANUAL SIDE
    # -------------------------------------------------------------------------
    # Manual is capped at the target so the corpus does not overshoot the
    # requested total. The budget is spent in manual_priority order, so the
    # scarce hand-collected sources are taken whole and only the abundant
    # tail source is trimmed.
    # -------------------------------------------------------------------------

    manual_used, keep_fractions = allocate_manual(
        measurements, args.target_manual, manual_priority, caps
    )

    # -------------------------------------------------------------------------
    # SANGRAHA SIDE
    # -------------------------------------------------------------------------
    # Three separate ceilings, all of which must hold:
    #
    #   target_sangraha            the requested size
    #   MANUAL_MULTIPLIER x manual the >=20% manual floor
    #   sangraha_available         what actually exists
    # -------------------------------------------------------------------------

    sangraha_used = int(
        min(
            target_sangraha,
            MANUAL_MULTIPLIER * manual_used,
            sangraha_available,
        )
    )

    sangraha_keep_fraction = (
        sangraha_used / sangraha_available
        if sangraha_available > 0 else 0.0
    )

    keep_fractions["sangraha"] = sangraha_keep_fraction

    total_tokens = manual_used + sangraha_used

    manual_share = (
        manual_used / total_tokens if total_tokens > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("PLANNED MIX")
    print("=" * 78)

    print(f"Manual tokens available  : {manual_available:>15,}")
    print(f"Sangraha tokens available: {sangraha_available:>15,}")
    print()
    print(f"Manual tokens used       : {manual_used:>15,}")
    print(f"Sangraha tokens used     : {sangraha_used:>15,}")
    print()

    print(
        f"  {'source':<12} {'available':>13} {'used':>13} "
        f"{'keep':>8}   reason"
    )
    print("  " + "-" * 60)

    for source_type, _ in sources:

        fraction = keep_fractions.get(source_type)

        if fraction is None:
            continue

        available = measurements[source_type]["estimated_tokens"]

        if fraction >= 1.0:
            reason = "kept whole"

        elif fraction <= 0.0:
            reason = "budget exhausted"

        elif source_type in caps and (
            abs(available * fraction - caps[source_type]) < 1_000_000
        ):
            reason = f"capped at {caps[source_type] / 1e6:,.0f}M"

        else:
            reason = "trimmed to fit budget"

        print(
            f"  {source_type:<12} {available:>13,} "
            f"{int(available * fraction):>13,} "
            f"{fraction:>8.4f}   {reason}"
        )

    # -------------------------------------------------------------------------
    # GUARD: a source dropped entirely
    # -------------------------------------------------------------------------
    #
    # Priority allocation spends the manual budget in order, so if the
    # budget is smaller than the sources ahead of it, a source can receive
    # a keep fraction of exactly zero and disappear from the corpus.
    #
    # That is a serious outcome and an easy one to miss, because a missing
    # source looks identical to a source that was never configured. It is
    # almost always a sign that --target-manual is set too low or a cap is
    # too generous, so it is called out loudly rather than left to be
    # discovered in the statistics table after the splits are built.
    # -------------------------------------------------------------------------

    dropped = [
        source_type
        for source_type in manual_priority
        if source_type in keep_fractions
        and keep_fractions[source_type] <= 0.0
        and measurements.get(source_type, {}).get("estimated_tokens", 0) > 0
    ]

    if dropped:

        print()
        print("!" * 78)
        print("WARNING: these manual sources were dropped entirely:")

        for source_type in dropped:
            available = measurements[source_type]["estimated_tokens"]
            print(f"  {source_type:<12} {available:>15,} tokens discarded")

        print()
        print(
            "  The manual budget was exhausted by higher-priority sources."
        )
        print(
            f"  Raise --target-manual (currently "
            f"{args.target_manual / 1e6:,.0f}M) or lower a cap in "
            f"the language's manual_source_caps."
        )
        print("!" * 78)

    print()
    print(f"TOTAL training tokens    : {total_tokens:>15,}")
    print(f"Manual share             : {manual_share:>15.2%}")

    print()

    if manual_share >= MIN_MANUAL_SHARE:
        print(f"PASS: manual share meets the >= {MIN_MANUAL_SHARE:.0%} requirement.")
    else:
        print(f"FAIL: manual share is below {MIN_MANUAL_SHARE:.0%}.")

    shortfall = args.target_total - total_tokens

    if shortfall > 1_000_000:

        print(
            f"NOTE: {shortfall / 1e6:,.0f}M tokens below the "
            f"{args.target_total / 1e6:,.0f}M target."
        )

        # Name the binding constraint, so it is obvious which side to grow.
        manual_short = args.target_manual - manual_used
        sangraha_short = target_sangraha - sangraha_used

        if manual_short > 1_000_000:
            print(
                f"      Manual is {manual_short / 1e6:,.0f}M short. "
                f"Internet Archive is the cheapest way to close this."
            )

        if sangraha_short > 1_000_000:

            if sangraha_used >= sangraha_available:
                print(
                    f"      Sangraha is {sangraha_short / 1e6:,.0f}M short "
                    f"and fully consumed. Process more shards."
                )
            else:
                print(
                    f"      Sangraha is {sangraha_short / 1e6:,.0f}M short, "
                    f"limited by the 20% manual floor, not by supply. "
                    f"Growing manual would also raise the Sangraha ceiling."
                )

        print(
            "      The assignment allows a shortfall if the exact token "
            "count is reported and justified."
        )

    # -------------------------------------------------------------------------
    # SAVE THE PLAN
    # -------------------------------------------------------------------------

    plan = {
        "language": language.name,
        "tokenizer": args.tokenizer,
        "vocab_size": sp.get_piece_size(),
        "seed": args.seed,
        "targets": {
            "total_tokens": args.target_total,
            "manual_tokens": args.target_manual,
            "sangraha_tokens": target_sangraha,
            "min_manual_share": MIN_MANUAL_SHARE,
        },
        "measurements": measurements,
        "decision": {
            "manual_tokens_available": manual_available,
            "manual_tokens_used": manual_used,
            "sangraha_tokens_available": sangraha_available,
            "sangraha_tokens_used": sangraha_used,
            "sangraha_keep_fraction": sangraha_keep_fraction,

            # Consumed by make_splits.py. Every source has an entry, so the
            # split stage never has to infer which sources were trimmed.
            "keep_fractions": keep_fractions,

            "manual_priority": list(manual_priority),
            "manual_source_caps": caps,
            "estimated_total_tokens": total_tokens,
            "estimated_manual_share": manual_share,
            "meets_manual_requirement": manual_share >= MIN_MANUAL_SHARE,
            "sources_dropped_entirely": dropped,
        },
    }

    output_file = workspace.mix_plan

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print()
    print(f"Mix plan: {output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
