"""
Tokenizer evaluation, for either language.

Produces the tokenizer statistics Phase 1 asks for:

    * vocabulary size
    * token-frequency statistics
    * average characters per token
    * tokenization examples
    * unknown-token statistics

and the evidence for the vocabulary-size choice, which the assignment
requires to be made on fertility and unknown-token rate measured on
HELD-OUT text.

All measurements run on the validation split, never on train. A tokenizer
scored on its own training text reports optimistic fertility, because
every frequent sequence in that text already earned a vocabulary entry.

Metric definitions used here
----------------------------
characters per token
    non-whitespace characters / tokens. The headline compression number.
    Higher means each token carries more text, so a fixed context window
    covers more of the document.

fertility (tokens per word)
    tokens / whitespace-delimited words. The standard cross-lingual
    comparison metric, and what makes Model H and Model L comparable in
    Phase 3 despite having different vocabularies.

unknown rate
    fraction of emitted tokens equal to <unk>. With byte_fallback enabled
    this is zero by construction, so the byte-fallback rate below is the
    metric that actually carries information.

byte-fallback rate
    fraction of emitted tokens that are raw UTF-8 byte pieces. These are
    characters the vocabulary could not represent natively. This is the
    honest analogue of an unknown rate for a byte-fallback tokenizer, and
    it is reported separately rather than being hidden behind a zero.

Note on vocabulary size and the Phase 2 parameter budget
--------------------------------------------------------
Vocabulary size is not a free choice. With tied input and output
embeddings, the embedding matrix costs vocab_size x d_model parameters.
Against the ~25M parameter target in Phase 2, a 32k vocabulary at
d_model=384 already spends ~12.3M parameters, roughly half the budget,
before a single Transformer block exists. The sweep therefore reports the
embedding cost alongside fertility so the two are traded off explicitly
rather than accidentally.

Usage
-----
    python evaluate_tokenizer.py --language telugu --models <a.model> ...

Results are written to report/phase1/<language>/, so the two models'
tables and figures cannot be confused with one another.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import Workspace, add_language_argument

use_utf8_output()


# Characters tokenized per source when measuring. Enough for the ratios to
# be stable, small enough to keep the sweep interactive.
CHAR_BUDGET_PER_SOURCE = 5_000_000

# Byte-fallback pieces are written by SentencePiece as <0x41> etc.
BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

WORD_RE = re.compile(r"\S+")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_by_source(path, char_budget):
    """
    Read the validation split into per-source text samples.

    Held-out text is grouped by source because fertility differs sharply
    between clean web text and OCR output, and a single blended number
    would hide that. The per-source table is what explains the Model H
    versus Model L gap in the Phase 3 report.
    """

    by_source = {}

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            source_type = record.get("source_type", "unknown")
            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            bucket = by_source.setdefault(source_type, [])

            current = sum(len(t) for t in bucket)

            if current >= char_budget:
                continue

            bucket.append(text[: char_budget - current])

    return {
        source: "\n".join(texts)
        for source, texts in by_source.items()
    }


# =============================================================================
# COMPUTATION
# =============================================================================

def evaluate_text(sp, text):
    """
    Compute tokenizer statistics for one block of text.

    Returns a dictionary of raw counts and derived ratios.
    """

    pieces = sp.encode(text, out_type=str)

    if not pieces:
        return None

    tokens = len(pieces)

    characters = len(text)

    non_space_characters = sum(
        1 for c in text if not c.isspace()
    )

    words = len(WORD_RE.findall(text))

    unk_piece = sp.id_to_piece(sp.unk_id())

    unknown_tokens = sum(1 for p in pieces if p == unk_piece)

    byte_tokens = sum(1 for p in pieces if BYTE_PIECE_RE.match(p))

    return {
        "characters": characters,
        "non_space_characters": non_space_characters,
        "words": words,
        "tokens": tokens,
        "chars_per_token": non_space_characters / tokens,
        "fertility_tokens_per_word": tokens / words if words else 0.0,
        "unknown_tokens": unknown_tokens,
        "unknown_rate": unknown_tokens / tokens,
        "byte_fallback_tokens": byte_tokens,
        "byte_fallback_rate": byte_tokens / tokens,
        "piece_counter": Counter(pieces),
    }


def token_frequency_statistics(counter, vocab_size):
    """
    Summarise the token-frequency distribution on held-out text.

    Coverage by top-k answers a concrete question: how much of real text
    is carried by the most common few thousand pieces. A vocabulary whose
    tail is never used is spending Phase 2 embedding parameters on nothing.
    """

    total = sum(counter.values())

    if total == 0:
        return {}

    ranked = counter.most_common()

    coverage = {}

    for k in [100, 500, 1000, 5000, 10000, 20000]:

        if k > vocab_size:
            continue

        covered = sum(count for _, count in ranked[:k])

        coverage[f"top_{k}_coverage"] = covered / total

    used = len(counter)

    # Pieces appearing exactly once on held-out text are a sign the
    # vocabulary is larger than the data supports.
    singletons = sum(1 for _, count in ranked if count == 1)

    return {
        "total_tokens": total,
        "distinct_pieces_used": used,
        "vocabulary_utilisation": used / vocab_size,
        "singleton_pieces": singletons,
        "coverage": coverage,
        "top_30": ranked[:30],
    }


def evaluate_model(model_path, samples):
    """
    Evaluate one tokenizer across every source, plus a combined figure.
    """

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    vocab_size = sp.get_piece_size()

    per_source = {}

    combined_counter = Counter()

    combined = {
        "non_space_characters": 0,
        "words": 0,
        "tokens": 0,
        "unknown_tokens": 0,
        "byte_fallback_tokens": 0,
    }

    for source_type, text in sorted(samples.items()):

        result = evaluate_text(sp, text)

        if result is None:
            continue

        combined_counter.update(result["piece_counter"])

        for key in combined:
            combined[key] += result[key]

        result.pop("piece_counter")

        per_source[source_type] = result

    tokens = combined["tokens"]

    combined_ratios = {
        "chars_per_token": (
            combined["non_space_characters"] / tokens if tokens else 0.0
        ),
        "fertility_tokens_per_word": (
            combined["tokens"] / combined["words"]
            if combined["words"] else 0.0
        ),
        "unknown_rate": (
            combined["unknown_tokens"] / tokens if tokens else 0.0
        ),
        "byte_fallback_rate": (
            combined["byte_fallback_tokens"] / tokens if tokens else 0.0
        ),
    }

    frequency = token_frequency_statistics(combined_counter, vocab_size)

    return {
        "model": str(model_path),
        "vocab_size": vocab_size,
        "per_source": per_source,
        "combined": {**combined, **combined_ratios},
        "frequency": frequency,
    }


def choose_primary(results, requested=None):
    """
    Pick the model the per-source figure should describe.

    Order of preference:

      1. an explicit --primary path, when one is given
      2. the single model whose filename says "final", when exactly one
         final tokenizer was evaluated
      3. the largest vocabulary, as a last resort

    Rule 2 exists because a run usually evaluates the sweep candidates
    alongside the delivered tokenizer, and "largest vocabulary" would then
    pick the 64k candidate: a model that was trained on a 10% sample and
    is not what Phase 2 loads.
    """

    if requested:

        wanted = Path(requested).resolve()

        for result in results:
            if Path(result["model"]).resolve() == wanted:
                return result

        raise ValueError(
            f"--primary {requested} was not among the evaluated models"
        )

    finals = [r for r in results if "final" in Path(r["model"]).stem]

    if len(finals) == 1:
        return finals[0]

    return max(results, key=lambda r: r["vocab_size"])


def embedding_parameters(vocab_size, d_model):
    """
    Parameters spent on a tied embedding / output matrix.

    Reported so the vocabulary choice is made against the Phase 2 budget
    rather than on fertility alone.
    """

    return vocab_size * d_model


# =============================================================================
# TOKENIZATION EXAMPLES
# =============================================================================

SENTENCE_END_RE = re.compile(r"(?<=[.!?।॥])\s+")

# Bounds for a sentence drawn from held-out text: long enough to show
# real morphology, short enough to print in a table.
EXAMPLE_MIN_CHARACTERS = 40
EXAMPLE_MAX_CHARACTERS = 140


def sample_example_sentences(samples, per_source=1):
    """
    Draw tokenization examples from held-out text, one per source.

    Used when a language declares no hand-written example_sentences.
    Telugu's four examples were written deliberately to cover everyday,
    agglutinated, classical and mixed-script registers. That is only
    possible for a language the author reads: invented Assamese sentences
    could carry errors, and the fertility they produced would then be
    read as a property of the tokenizer rather than of the sentence.

    Held-out text has neither problem. It is real language, it is not
    training data, and taking one sentence per source reproduces the
    register coverage the hand-written set was chosen for, since the
    sources ARE the registers.

    Selection is deterministic - the first sentence in each source's
    sample that falls inside the length bounds - so the table is stable
    across runs.
    """

    examples = []

    for source in sorted(samples):

        taken = 0

        for sentence in SENTENCE_END_RE.split(samples[source]):

            sentence = " ".join(sentence.split())

            if not (
                EXAMPLE_MIN_CHARACTERS
                <= len(sentence)
                <= EXAMPLE_MAX_CHARACTERS
            ):
                continue

            examples.append((source, sentence))
            taken += 1

            if taken >= per_source:
                break

    return examples


def tokenization_examples(model_path, sentences):
    """
    Tokenize a set of sentences for the qualitative table.

    Takes (origin, sentence) pairs, where origin is the corpus source a
    held-out sentence came from, or "hand-written" for a sentence the
    language config supplies. Recording it keeps the table honest about
    which examples were chosen and which were drawn.
    """

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    examples = []

    for origin, sentence in sentences:

        pieces = sp.encode(sentence, out_type=str)

        examples.append({
            "origin": origin,
            "text": sentence,
            "pieces": pieces,
            "token_count": len(pieces),
            "characters": len(sentence),
            "chars_per_token": (
                sum(1 for c in sentence if not c.isspace()) / len(pieces)
                if pieces else 0.0
            ),
        })

    return examples


# =============================================================================
# VISUALISATION
# =============================================================================
#
# Kept separate from the computation above, as the assignment's code-design
# guidelines require. Every plot carries a title, axis labels and a legend
# where more than one series is drawn.
# =============================================================================

def plot_vocab_sweep(results, output_file, language_name):
    """Plot fertility and byte-fallback rate against vocabulary size."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r["vocab_size"])

    vocab_sizes = [r["vocab_size"] for r in results]
    chars_per_token = [r["combined"]["chars_per_token"] for r in results]
    fertility = [
        r["combined"]["fertility_tokens_per_word"] for r in results
    ]

    figure, axis_left = plt.subplots(figsize=(9, 5.5))

    line_a = axis_left.plot(
        vocab_sizes, chars_per_token,
        marker="o", color="#2A6FB5", label="Characters per token",
    )

    axis_left.set_xlabel("Vocabulary size (pieces)")
    axis_left.set_ylabel("Characters per token")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()

    line_b = axis_right.plot(
        vocab_sizes, fertility,
        marker="s", color="#C2472A", label="Fertility (tokens per word)",
    )

    axis_right.set_ylabel("Fertility (tokens per word)")

    axis_left.set_title(
        f"{language_name} tokenizer: compression against vocabulary size\n"
        "(measured on the held-out validation split)"
    )

    lines = line_a + line_b

    axis_left.legend(
        lines,
        [line.get_label() for line in lines],
        loc="center right",
    )

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def plot_per_source_fertility(result, output_file, language_name):
    """Plot characters per token for each source of the corpus."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_source = result["per_source"]

    sources = sorted(per_source)
    values = [per_source[s]["chars_per_token"] for s in sources]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(sources, values, color="#2A6FB5")

    axis.axhline(
        result["combined"]["chars_per_token"],
        color="#C2472A",
        linestyle="--",
        label="Corpus average",
    )

    axis.set_xlabel("Corpus source")
    axis.set_ylabel("Characters per token")
    axis.set_title(
        f"{language_name} tokenizer compression by source "
        f"(vocabulary {result['vocab_size']:,})\n"
        "measured on the held-out validation split"
    )
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate one language's tokenizers on held-out text."
    )

    add_language_argument(parser)

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more SentencePiece .model files.",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=384,
        help="Planned Phase 2 model dimension, for embedding cost.",
    )

    parser.add_argument(
        "--char-budget",
        type=int,
        default=CHAR_BUDGET_PER_SOURCE,
        help="Held-out characters tokenized per source. Lower it only for "
             "a smoke test, and expect the numbers to move: the report "
             "figures are produced at the default.",
    )

    parser.add_argument(
        "--primary",
        default=None,
        help="Model the per-source fertility figure describes. Defaults "
             "to the one final tokenizer among --models.",
    )

    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)

    validation_file = workspace.require(
        workspace.split_file("validation"),
        f"Run: python make_splits.py --language {workspace.slug}",
    )

    report_dir = workspace.report
    figure_dir = workspace.figures

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"{workspace.title} - TOKENIZER EVALUATION")
    print("=" * 78)
    print(f"Held-out file: {validation_file}")

    samples = load_validation_by_source(validation_file, args.char_budget)

    # -------------------------------------------------------------------------
    # EXAMPLE SENTENCES
    # -------------------------------------------------------------------------
    # Hand-written where the language config supplies them, drawn from
    # held-out text otherwise. See sample_example_sentences.
    # -------------------------------------------------------------------------

    if language.example_sentences:
        example_pairs = [
            ("hand-written", sentence)
            for sentence in language.example_sentences
        ]
    else:
        example_pairs = sample_example_sentences(samples)

    print(f"Examples     : {len(example_pairs)} sentences")

    print()
    print("Held-out sample per source:")

    for source_type, text in sorted(samples.items()):
        print(f"  {source_type:<12} {len(text):>12,} characters")

    # -------------------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------------------

    results = []

    for model_path in args.models:

        path = Path(model_path)

        if not path.exists():
            print(f"WARNING: missing model {path}, skipped.")
            continue

        print()
        print("-" * 78)
        print(f"Model: {path.name}")

        result = evaluate_model(path, samples)

        result["embedding_parameters"] = embedding_parameters(
            result["vocab_size"], args.d_model
        )

        result["examples"] = tokenization_examples(path, example_pairs)

        results.append(result)

        combined = result["combined"]

        print(f"  vocabulary size    : {result['vocab_size']:,}")
        print(f"  characters/token   : {combined['chars_per_token']:.3f}")
        print(
            f"  fertility (tok/wd) : "
            f"{combined['fertility_tokens_per_word']:.3f}"
        )
        print(f"  unknown rate       : {combined['unknown_rate']:.6%}")
        print(
            f"  byte-fallback rate : "
            f"{combined['byte_fallback_rate']:.6%}"
        )
        print(
            f"  vocabulary used    : "
            f"{result['frequency'].get('vocabulary_utilisation', 0):.2%}"
        )
        print(
            f"  embedding params   : "
            f"{result['embedding_parameters']:,} "
            f"(d_model={args.d_model}, tied)"
        )

    # -------------------------------------------------------------------------
    # COMPARISON TABLE
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("VOCABULARY SIZE COMPARISON (held-out validation)")
    print("=" * 78)

    header = (
        f"{'vocab':>8} {'chars/tok':>11} {'fertility':>11} "
        f"{'unk rate':>11} {'byte fb':>11} {'vocab used':>11} "
        f"{'embed params':>14}"
    )

    print(header)
    print("-" * len(header))

    for result in sorted(results, key=lambda r: r["vocab_size"]):

        combined = result["combined"]

        print(
            f"{result['vocab_size']:>8,} "
            f"{combined['chars_per_token']:>11.3f} "
            f"{combined['fertility_tokens_per_word']:>11.3f} "
            f"{combined['unknown_rate']:>10.4%} "
            f"{combined['byte_fallback_rate']:>10.4%} "
            f"{result['frequency'].get('vocabulary_utilisation', 0):>10.2%} "
            f"{result['embedding_parameters']:>14,}"
        )

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # The two figures answer different questions, so they are drawn from
    # different subsets of the models evaluated:
    #
    #   sweep figure       how compression trades against vocabulary size.
    #                      Candidates only. A final tokenizer trained on the
    #                      full split shares an x position with the
    #                      candidate at that size, and plotting both would
    #                      put two points on one vocabulary size and make
    #                      the curve read as noise.
    #
    #   per-source figure  how the DELIVERED tokenizer behaves across
    #                      registers. The primary model only, since that is
    #                      the one Phase 2 will use.
    # -------------------------------------------------------------------------

    candidates = [r for r in results if "final" not in Path(r["model"]).stem]

    if len(candidates) > 1:

        sweep_file = figure_dir / "tokenizer_vocab_sweep.png"
        plot_vocab_sweep(candidates, sweep_file, language.name)
        print()
        print(f"Figure: {sweep_file}")

    if results:

        primary = choose_primary(results, args.primary)

        source_file = figure_dir / "tokenizer_fertility_by_source.png"
        plot_per_source_fertility(primary, source_file, language.name)
        print(f"Figure: {source_file}  (model: {Path(primary['model']).name})")

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    output_file = report_dir / "tokenizer_evaluation.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language.name,
                "d_model": args.d_model,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Results: {output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
