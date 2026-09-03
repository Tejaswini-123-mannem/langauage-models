"""
Generation quality: BLEU, chrF, ROUGE-L, and diversity diagnostics.

    python common/generate.py --checkpoint <path> --split test

What is being measured
----------------------
For each held-out document the first `prefix_tokens` tokens are fed to the
model as context, and it continues for `continuation_tokens`. The true
continuation from the same document is the reference. Decoding is greedy
plus temperatures 0.5, 1.0 and 1.5, as the assignment specifies.

Prefixes come from the document index built in stage 0, so every prefix
starts at a real document start. A window cut at an arbitrary offset would
begin mid-sentence, and "the reference continuation" would not be a
well-defined object.

Why these metrics are mostly uninformative here, and why that is worth saying
----------------------------------------------------------------------------
The assignment asks for BLEU, chrF and ROUGE-L and then asks whether each
is informative for open-ended generation. The honest answer is that two of
the three are close to meaningless in this setting, and saying so with
numbers is a better answer than reporting them as if they were scores.

BLEU-4 counts exact 4-gram overlap against ONE reference. A continuation
can be fluent, on-topic and grammatical while sharing almost no 4-gram
with the particular continuation that document happened to have; there are
thousands of valid futures for any prefix and BLEU credits one. Telugu
makes it worse: it is agglutinative, so a single differing suffix changes
the whole word token and destroys every n-gram containing it. Expect
values near zero for a 25M model, and read a rise from 0.1 to 0.3 as
noise, not progress.

chrF compares character n-grams, so a shared stem with a different
inflection still earns partial credit. For two morphologically rich
languages written in abugida scripts this is much the most informative of
the three, and it is also the only one that is not hostage to how words
are tokenized.

ROUGE-L is longest common subsequence, so the matched tokens need not be
contiguous. That makes it more forgiving than BLEU but it is still bound
to a single reference.

The reference-free diagnostics are what actually characterise a 25M model.
Small undertrained language models fail by collapsing into loops, and
repetition rate and Distinct-1/2 detect that immediately, at every
temperature, without needing a reference at all.

Tokenization for the metrics
----------------------------
BLEU is computed with sacrebleu's 'intl' tokenizer, which splits on
Unicode punctuation and leaves Indic script runs intact. The default '13a'
tokenizer is built around Latin script conventions. sacrebleu also ships an
'spm' tokenizer, which is not used here: it loads a pretrained
SentencePiece model, and this project's whole premise is that no
pretrained tokenizer is involved.

chrF needs no tokenizer, being character-based, which is part of why it is
the metric to trust here.

ROUGE-L is implemented directly rather than imported. The available
packages tokenize for English -- stemming, stopwords, \\w+ regexes that
treat a Telugu word as one blob or split it at combining marks -- and
would silently measure something other than what is claimed.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TokenDataset
from evaluate import load_checkpoint, load_tokenizer
from language_config import get_language, use_utf8_output
from workspace import Workspace

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]

# Decoding settings the assignment names explicitly: greedy, then three
# temperatures. Greedy is temperature 0 by convention here.
DECODINGS = [
    ("greedy", 0.0),
    ("t0.5", 0.5),
    ("t1.0", 1.0),
    ("t1.5", 1.5),
]


# =============================================================================
# METRICS
# =============================================================================


def lcs_length(a, b):
    """
    Length of the longest common subsequence of two token lists.

    Row-rolled dynamic programming: only the previous row of the table is
    needed, so memory is O(len(b)) rather than O(len(a) * len(b)).
    """

    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    return previous[-1]


def rouge_l(hypothesis, reference):
    """
    ROUGE-L F1 over whitespace tokens.

    Precision is LCS/len(hypothesis), recall is LCS/len(reference), and the
    score is their F1. Reported rather than recall alone so that a model
    which pads its output cannot inflate the number.
    """

    hyp = hypothesis.split()
    ref = reference.split()

    if not hyp or not ref:
        return 0.0

    lcs = lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0

    precision = lcs / len(hyp)
    recall = lcs / len(ref)

    return 2 * precision * recall / (precision + recall)


def distinct_n(token_lists, n):
    """
    Distinct-n: unique n-grams divided by total n-grams, pooled over samples.

    Pooled rather than averaged per sample: a model that produces the same
    sentence for every prompt scores well per-sample (each sentence is
    internally varied) and badly pooled, and the pooled number is the one
    that reflects the failure.
    """

    total = 0
    unique = set()

    for tokens in token_lists:
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i:i + n]))
            total += 1

    return len(unique) / total if total else 0.0


def repetition_rate(tokens, n=4):
    """
    Fraction of n-grams in one sample that are not their first occurrence.

    This is the loop detector. A model stuck repeating a phrase produces
    the same 4-gram over and over, and the rate approaches 1; fluent text
    in either language sits low, since exact 4-word repeats are rare.
    """

    if len(tokens) < n:
        return 0.0

    seen = set()
    repeated = 0
    total = 0

    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        total += 1
        if gram in seen:
            repeated += 1
        seen.add(gram)

    return repeated / total if total else 0.0


def require_sacrebleu():
    """
    Check for sacrebleu before generating rather than after.

    The metrics import lives inside score_corpus(), which does not run
    until the first decoding has finished, so a missing sacrebleu would
    otherwise surface several minutes of GPU time into the run. Kaggle's
    image does not ship it.
    """

    try:
        import sacrebleu  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "sacrebleu is not installed, and BLEU and chrF cannot "
            "be computed without it. Install it with:\n\n"
            "    pip install sacrebleu==2.6.0"
        )


def score_corpus(hypotheses, references):
    """
    Corpus BLEU-4 and chrF2 via sacrebleu, plus mean ROUGE-L.

    The class-based API is used rather than the corpus_bleu() shortcut
    because the signature -- which records the tokenizer, smoothing and
    version the score was produced with -- hangs off the metric object.
    Reporting a BLEU without its signature is reporting a number that
    cannot be reproduced, and sacrebleu exists largely to stop people
    doing that.
    """

    from sacrebleu.metrics import BLEU, CHRF

    # tokenize="intl" splits on Unicode punctuation and leaves Indic script
    # runs intact; the default "13a" encodes Latin-script conventions.
    bleu_metric = BLEU(tokenize="intl", force=True)
    bleu = bleu_metric.corpus_score(hypotheses, [references])

    # word_order=0 is chrF proper (character n-grams only). chrF++ would be
    # word_order=2, which reintroduces the word-tokenization sensitivity
    # that makes chrF the trustworthy metric here in the first place.
    chrf_metric = CHRF(word_order=0)
    chrf = chrf_metric.corpus_score(hypotheses, [references])

    rouge = [rouge_l(h, r) for h, r in zip(hypotheses, references)]

    return {
        "bleu4": round(bleu.score, 4),
        "bleu_signature": str(bleu_metric.get_signature()),
        "bleu_precisions": [round(p, 3) for p in bleu.precisions],
        "chrf2": round(chrf.score, 4),
        "chrf_signature": str(chrf_metric.get_signature()),
        "rouge_l": round(float(np.mean(rouge)) if rouge else 0.0, 4),
    }


def score_diversity(generated_texts, generated_tokens):
    """Reference-free fluency and diversity diagnostics."""

    word_lists = [text.split() for text in generated_texts]

    return {
        "distinct_1": round(distinct_n(word_lists, 1), 4),
        "distinct_2": round(distinct_n(word_lists, 2), 4),
        "distinct_1_tokens": round(distinct_n(generated_tokens, 1), 4),
        "distinct_2_tokens": round(distinct_n(generated_tokens, 2), 4),
        "repetition_rate_4gram": round(
            float(np.mean([repetition_rate(w, 4) for w in word_lists])), 4
        ),
        "repetition_rate_4gram_tokens": round(
            float(np.mean([repetition_rate(t, 4) for t in generated_tokens])), 4
        ),
        "mean_words": round(float(np.mean([len(w) for w in word_lists])), 2),
    }


# =============================================================================
# SAMPLING
# =============================================================================


def choose_documents(index, n_samples, min_tokens, seed=7):
    """
    Pick held-out documents, spread across sources rather than by frequency.

    A uniform sample of Telugu test documents would be roughly 90% Sangraha,
    because that is the corpus composition -- and the interesting question
    is precisely how the model does on the OTHER sources. Sampling evenly
    across sources and reporting the source alongside each sample keeps
    the rarer ones visible.
    """

    rng = np.random.default_rng(seed)

    by_source = defaultdict(list)
    for offset, length, source in index["documents"]:
        if length >= min_tokens:
            by_source[source].append((offset, length, source))

    if not by_source:
        raise ValueError(
            f"No held-out document has {min_tokens} tokens; "
            f"lower --prefix-tokens or --continuation-tokens."
        )

    sources = sorted(by_source)
    per_source = max(1, n_samples // len(sources))

    chosen = []
    for source in sources:
        pool = by_source[source]
        take = min(per_source, len(pool))
        picked = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in picked)

    # Top up from the largest pool if integer division left us short.
    if len(chosen) < n_samples:
        biggest = max(by_source.values(), key=len)
        extra = rng.choice(len(biggest), size=min(n_samples - len(chosen), len(biggest)),
                           replace=False)
        chosen.extend(biggest[i] for i in extra)

    return chosen[:n_samples]


@torch.no_grad()
def generate_batch(model, prefixes, n_new, temperature, device, batch_size=16,
                   eos_id=None):
    """
    Continue a list of equal-length prefixes.

    Batched because every prefix is the same length, so they pack into one
    tensor with no padding and no mask -- generation for 200 prompts costs
    little more than for one, and the evaluation runs in minutes.

    eos_id is deliberately NOT passed through to model.generate: the
    continuation must be exactly n_new tokens long to line up with a
    reference of that length. A sample that stops early would be scored
    against a longer reference and penalised for the wrong reason.
    """

    outputs = []

    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start:start + batch_size]
        idx = torch.tensor(np.stack(chunk), dtype=torch.long, device=device)
        out = model.generate(idx, n_new, temperature=temperature)
        outputs.extend(out[:, idx.shape[1]:].cpu().numpy())

    return outputs


# =============================================================================
# REPORTING
# =============================================================================


def print_table(results):
    print(f"\n  {'decoding':<10}{'BLEU-4':>9}{'chrF2':>9}{'ROUGE-L':>10}"
          f"{'dist-1':>9}{'dist-2':>9}{'rep-4':>8}")
    print("  " + "-" * 64)

    for name, scores in results.items():
        print(f"  {name:<10}{scores['bleu4']:>9.3f}{scores['chrf2']:>9.3f}"
              f"{scores['rouge_l']:>10.4f}{scores['distinct_1']:>9.4f}"
              f"{scores['distinct_2']:>9.4f}"
              f"{scores['repetition_rate_4gram']:>8.3f}")


def write_samples(path, samples, n_show):
    """Write qualitative examples as Markdown for the report."""

    lines = [
        "# Generated samples",
        "",
        "Prefixes are the opening tokens of held-out documents. The reference",
        "is that document's true continuation. Newlines are absent because the",
        "tokenizer's nmt_nfkc normalizer maps them to spaces; see stage 0.",
        "",
    ]

    for sample in samples[:n_show]:
        lines += [
            f"## {sample['source']} (document at offset {sample['offset']})",
            "",
            "**Prefix**",
            "",
            f"> {sample['prefix']}",
            "",
            "**Reference continuation**",
            "",
            f"> {sample['reference']}",
            "",
        ]
        for name, _ in DECODINGS:
            lines += [f"**{name}**", "", f"> {sample['generations'][name]}", ""]
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generation quality and diversity on held-out prefixes.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-show", type=int, default=12,
                        help="Qualitative examples written to the Markdown file.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    require_sacrebleu()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, config, state = load_checkpoint(args.checkpoint, device)
    workspace = Workspace(get_language(config.language))
    data_dir = Path(args.data) if args.data else workspace.tokens

    dataset = TokenDataset(data_dir / f"{args.split}.bin", config.block_size)
    tokenizer = load_tokenizer(workspace, config.vocab_size)

    index_path = data_dir / f"{args.split}_documents.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    need = args.prefix_tokens + args.continuation_tokens
    documents = choose_documents(index, args.n_samples, need, seed=args.seed)

    print("=" * 78)
    print(f"GENERATE - {workspace.title} - {args.split}")
    print("=" * 78)
    print(f"  checkpoint : {Path(args.checkpoint).name} (step {state.get('step', '?'):,})")
    print(f"  device     : {device}")
    print(f"  samples    : {len(documents)} documents, "
          f"{args.prefix_tokens} prefix -> {args.continuation_tokens} continuation")
    print(f"  decodings  : {', '.join(name for name, _ in DECODINGS)}")

    # ---- build prefixes and references --------------------------------------
    prefixes, references, sources, offsets = [], [], [], []
    for offset, length, source in documents:
        span = np.asarray(dataset.tokens[offset:offset + need], dtype=np.int64)
        prefixes.append(span[:args.prefix_tokens])
        references.append(tokenizer.decode(span[args.prefix_tokens:].tolist()))
        sources.append(source)
        offsets.append(int(offset))

    # ---- generate under each decoding ---------------------------------------
    results = {}
    per_sample = [
        {"source": s, "offset": o,
         "prefix": tokenizer.decode(p.tolist()), "reference": r,
         "generations": {}}
        for s, o, p, r in zip(sources, offsets, prefixes, references)
    ]

    for name, temperature in DECODINGS:
        print(f"\n  {name} (temperature {temperature}) ...", end=" ", flush=True)
        started = time.time()

        generated = generate_batch(
            model, prefixes, args.continuation_tokens, temperature, device,
            batch_size=args.batch_size,
        )
        texts = [tokenizer.decode(g.tolist()) for g in generated]

        for sample, text in zip(per_sample, texts):
            sample["generations"][name] = text

        scores = score_corpus(texts, references)
        scores.update(score_diversity(texts, [g.tolist() for g in generated]))
        results[name] = scores

        print(f"{time.time() - started:.1f}s  "
              f"BLEU {scores['bleu4']:.2f}  chrF {scores['chrf2']:.2f}  "
              f"rep {scores['repetition_rate_4gram']:.3f}", flush=True)

    print_table(results)

    # ---- write --------------------------------------------------------------
    out_dir = Path(args.out) if args.out else (
        REPO / "report" / "phase2" / workspace.slug
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"generation_{args.split}.json").write_text(
        json.dumps({
            "language": config.language,
            "model": config.name,
            "checkpoint": str(args.checkpoint),
            "step": state.get("step"),
            "split": args.split,
            "n_samples": len(documents),
            "prefix_tokens": args.prefix_tokens,
            "continuation_tokens": args.continuation_tokens,
            "metrics": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / f"generation_samples_{args.split}.json").write_text(
        json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_samples(out_dir / f"generation_samples_{args.split}.md",
                  per_sample, args.n_show)

    print(f"\n  wrote {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
