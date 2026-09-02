"""
Intrinsic language-modelling evaluation: cross-entropy, perplexity, BPB.

    python common/evaluate.py --checkpoint <path> --split test

Why bits-per-byte is the number that matters here
-------------------------------------------------
Perplexity is per TOKEN, and Model H and Model L do not have the same
token. Phase 1 measured 3.531 characters per token for Telugu and 3.157
for Assamese, so an Assamese token commits the model to predicting less
text than a Telugu one. A model that is genuinely worse can therefore post
a lower perplexity purely because its tokenizer chopped the text finer.
Comparing the two perplexities directly would be measuring the tokenizers.

Bits-per-byte re-denominates the same total loss over the UTF-8 bytes of
the underlying text, which no tokenizer can change:

    BPB = total_loss_in_nats / (ln 2 * n_bytes)

Both models are then quoted in the same unit -- bits of information per
byte of Telugu or Assamese -- and the comparison is about the models and
their corpora rather than their segmentation. This is the figure Phase 3's
resource-tier argument rests on.

Which bytes go in the denominator
---------------------------------
The bytes of the text the model actually modelled, i.e. what the token
stream decodes to -- not the bytes of the original JSONL.

SentencePiece's nmt_nfkc normalizer maps newlines to spaces, so a decoded
document differs from its source in whitespace. Measured over 400 held-out
documents per language the byte count moves by -0.02% for Telugu and
-0.03% for Assamese: negligible in size, but it differs BETWEEN the two
languages, and the whole point of BPB here is a fair cross-language
comparison. Charging a model for newline bytes it was never shown, at a
rate that differs by language, would put a thumb on exactly the scale
being read.

Per-source results
------------------
The held-out document index from build_token_bins.py records each
document's source, so the same forward pass yields a per-source breakdown
for free. That is where the interesting Phase 3 evidence is: Assamese is
48.9% Internet Archive OCR against Telugu's 78.0% clean Sangraha web text,
and "Model L is worse" is a much weaker claim than "Model L is worse on
OCR text specifically, by this much".
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TokenDataset
from language_config import get_language, use_utf8_output
from model import GPT, GPTConfig
from workspace import Workspace

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# LOADING
# =============================================================================


def load_checkpoint(path, device):
    """
    Rebuild a model from a checkpoint.

    The architecture comes from the checkpoint's own model_config, not from
    a config file on disk. A checkpoint has to be self-describing: the
    config file may have been edited since the run, and loading weights
    into a differently-shaped model either raises or -- worse, if only
    dropout or block_size changed -- silently evaluates something other
    than what was trained.
    """

    state = torch.load(path, map_location=device, weights_only=False)

    config = GPTConfig(**state["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    return model, config, state


def load_tokenizer(workspace, vocab_size):
    """The language's final SentencePiece model, for byte accounting."""

    import sentencepiece as spm

    path = workspace.final_model(vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"Missing tokenizer {path}")

    return spm.SentencePieceProcessor(model_file=str(path))


# =============================================================================
# COMPUTATION
# =============================================================================


@torch.no_grad()
def evaluate_windows(model, dataset, device, batch_windows, max_windows=None,
                     progress_every=200):
    """
    Total cross-entropy over consecutive windows of a split.

    Returns (total_nats, n_predicted_tokens).

    Windows are taken in order and cover the split exactly once. Unlike
    training there is no shuffling to do: every window is visited, so the
    order cannot bias the result, and going in order keeps the memmap reads
    sequential.

    The sum of per-token losses is accumulated rather than a mean of batch
    means. The two differ whenever the final batch is short, and a
    perplexity that silently depends on the batch size is not a measurement.
    """

    n_windows = dataset.n_windows if max_windows is None \
        else min(dataset.n_windows, max_windows)

    total_nats = 0.0
    total_tokens = 0
    started = time.time()

    for start in range(0, n_windows, batch_windows):
        ids = np.arange(start, min(start + batch_windows, n_windows))
        x, y = dataset.windows(ids, device=device)

        logits, _, _ = model(x)

        # reduction="sum" over every predicted position, so short final
        # batches contribute exactly their own token count.
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_nats += loss.item()
        total_tokens += y.numel()

        if progress_every and (start // batch_windows) % progress_every == 0:
            done = min(start + batch_windows, n_windows)
            rate = done / max(time.time() - started, 1e-9)
            print(f"      {done:,}/{n_windows:,} windows  "
                  f"{rate:.0f} win/s", flush=True)

    return total_nats, total_tokens


def stratify_documents(index, max_documents, seed=3):
    """
    Take documents evenly across sources, not the first N in the file.

    The document index is written in ribbon order, and the ribbon is
    ordered by source -- so the first N entries of the Assamese validation
    index are all wikisource, and the first N of Telugu's are all textbook
    OCR. Truncating to the first N therefore produces a "per-source
    breakdown" containing exactly one source, which is worse than useless
    because it looks like a result.

    Sampling evenly per source also over-represents the rare ones relative
    to the corpus, which is the point: the interesting question is how the
    model handles OCR text versus clean web text, and a proportional sample
    would be ~90% Sangraha for Telugu.
    """

    if not max_documents:
        return index["documents"]

    by_source = defaultdict(list)
    for document in index["documents"]:
        by_source[document[2]].append(document)

    rng = np.random.default_rng(seed)
    per_source = max(1, max_documents // max(1, len(by_source)))

    chosen = []
    for source in sorted(by_source):
        pool = by_source[source]
        take = min(per_source, len(pool))
        picked = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in picked)

    return chosen


@torch.no_grad()
def evaluate_by_document(model, dataset, index, tokenizer, device,
                         max_documents=None):
    """
    Loss and decoded byte count per held-out document, grouped by source.

    Documents are evaluated one at a time, truncated to block_size, because
    a document is the unit that has a source label and a decodable byte
    count. This is slower than the windowed pass and is used for the
    per-source breakdown rather than the headline number.

    A document's first token has no context and is not predicted, so it
    contributes block-1 predictions; the byte count is taken over the same
    decoded span so loss and bytes describe the same text.
    """

    block = dataset.block_size
    documents = stratify_documents(index, max_documents)

    per_source = defaultdict(lambda: {"nats": 0.0, "tokens": 0, "bytes": 0,
                                      "documents": 0})

    for offset, length, source in documents:
        span = min(int(length), block + 1)
        if span < 2:
            continue

        chunk = np.asarray(dataset.tokens[offset:offset + span], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(chunk[1:]).unsqueeze(0).to(device)

        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )

        # Bytes of the text these predictions actually covered: the decoded
        # form of the predicted tokens, which is what the model was scored
        # on. See the module docstring for why not the raw source bytes.
        text = tokenizer.decode(chunk[1:].tolist())

        bucket = per_source[source]
        bucket["nats"] += loss.item()
        bucket["tokens"] += int(y.numel())
        bucket["bytes"] += len(text.encode("utf-8"))
        bucket["documents"] += 1

    return per_source


def decoded_bytes(tokenizer, tokens, start, count, chunk=200_000):
    """
    UTF-8 byte length of the text a token span decodes to.

    Decoded in chunks because a full test split is millions of ids and
    handing them to SentencePiece in one call builds the entire string in
    memory for no reason.

    Chunking can shift the count by a byte or two at the boundaries, where
    SentencePiece would have emitted a word-initial space differently.
    Against ~7 million bytes that is far below the precision anything here
    is quoted to, and the alternative -- materialising the whole decoded
    corpus as one Python string -- costs hundreds of megabytes.
    """

    total = 0

    for begin in range(start, start + count, chunk):
        span = np.asarray(
            tokens[begin:min(begin + chunk, start + count)], dtype=np.int64
        )
        if span.size == 0:
            break
        total += len(tokenizer.decode(span.tolist()).encode("utf-8"))

    return total


def summarise(total_nats, total_tokens, total_bytes):
    """Cross-entropy, perplexity and bits-per-byte from raw totals."""

    cross_entropy = total_nats / max(total_tokens, 1)

    return {
        "cross_entropy_nats_per_token": round(cross_entropy, 6),
        "perplexity": round(math.exp(min(cross_entropy, 20)), 4),
        "bits_per_token": round(cross_entropy / math.log(2), 6),
        "bits_per_byte": (
            round(total_nats / (math.log(2) * total_bytes), 6)
            if total_bytes else None
        ),
        "tokens": int(total_tokens),
        "bytes": int(total_bytes),
    }


# =============================================================================
# REPORTING
# =============================================================================


def print_headline(title, metrics):
    print(f"\n  {title}")
    print(f"    cross-entropy : {metrics['cross_entropy_nats_per_token']:.4f} nats/token")
    print(f"    perplexity    : {metrics['perplexity']:.2f}")
    print(f"    bits/token    : {metrics['bits_per_token']:.4f}")
    if metrics["bits_per_byte"] is not None:
        print(f"    bits/byte     : {metrics['bits_per_byte']:.4f}")
    print(f"    tokens        : {metrics['tokens']:,}")


def print_per_source(per_source):
    print(f"\n  BY SOURCE")
    print(f"    {'source':<14}{'docs':>8}{'tokens':>12}{'ppl':>10}{'bits/byte':>12}")
    print("    " + "-" * 56)

    for source in sorted(per_source, key=lambda s: -per_source[s]["tokens"]):
        bucket = per_source[source]
        metrics = summarise(bucket["nats"], bucket["tokens"], bucket["bytes"])
        print(f"    {source:<14}{bucket['documents']:>8,}{bucket['tokens']:>12,}"
              f"{metrics['perplexity']:>10.2f}{metrics['bits_per_byte']:>12.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perplexity and bits-per-byte on a held-out split.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file.")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--data", default=None, help="Directory holding the .bin files.")
    parser.add_argument("--out", default=None, help="Where to write the JSON result.")
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Evaluate only the first N windows. For a quick check, not a report.",
    )
    parser.add_argument(
        "--max-documents", type=int, default=2000,
        help="Documents used for the per-source breakdown, spread evenly "
             "across sources rather than taken from the head of the index.",
    )
    parser.add_argument(
        "--no-per-source", action="store_true",
        help="Skip the per-source pass.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, config, state = load_checkpoint(args.checkpoint, device)
    workspace = Workspace(get_language(config.language))
    data_dir = Path(args.data) if args.data else workspace.tokens

    dataset = TokenDataset(data_dir / f"{args.split}.bin", config.block_size)

    print("=" * 78)
    print(f"EVALUATE - {workspace.title} - {args.split}")
    print("=" * 78)
    print(f"  checkpoint : {Path(args.checkpoint).name} (step {state.get('step', '?'):,})")
    print(f"  device     : {device}")
    print(f"  parameters : {model.num_parameters():,}")
    print(f"  split      : {dataset.n_tokens:,} tokens, {dataset.n_windows:,} windows")

    # ---- headline: windowed pass over the whole split -----------------------
    print(f"\n  scoring {args.split} ...")
    total_nats, total_tokens = evaluate_windows(
        model, dataset, device, args.batch_windows, args.max_windows
    )

    # Bytes for the headline BPB come from decoding the same token span the
    # loss was computed over.
    tokenizer = load_tokenizer(workspace, config.vocab_size)
    total_bytes = decoded_bytes(tokenizer, dataset.tokens, 1, int(total_tokens))

    headline = summarise(total_nats, total_tokens, total_bytes)
    print_headline(f"{args.split.upper()} (all windows)", headline)

    result = {
        "language": config.language,
        "model": config.name,
        "checkpoint": str(args.checkpoint),
        "step": state.get("step"),
        "split": args.split,
        "block_size": config.block_size,
        "parameters": model.num_parameters(),
        "overall": headline,
    }

    # ---- per-source ---------------------------------------------------------
    index_path = data_dir / f"{args.split}_documents.json"
    if not args.no_per_source and index_path.exists():
        print(f"\n  per-source breakdown "
              f"(first {args.max_documents:,} documents) ...")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        per_source = evaluate_by_document(
            model, dataset, index, tokenizer, device, args.max_documents
        )
        print_per_source(per_source)

        result["by_source"] = {
            source: {
                **summarise(b["nats"], b["tokens"], b["bytes"]),
                "documents": b["documents"],
            }
            for source, b in per_source.items()
        }

    # ---- write --------------------------------------------------------------
    out_path = Path(args.out) if args.out else (
        REPO / "report" / "phase2" / workspace.slug / f"eval_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"\n  wrote {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
