"""
Attention analysis: heatmaps, per-head entropy, mean attention distance.

    python common/attention.py --checkpoint <path>

This is possible only because CausalSelfAttention computes softmax(QK^T/
sqrt(d_k) + M)V explicitly and can hand back the (B, h, T, T) probability
tensor. Had the forward pass delegated to a fused kernel the probabilities
would never be materialised, and none of the measurements below would
exist to make.

Computation and plotting are kept in separate functions throughout, as the
assignment's code-design guidelines require: every metric here is computed
by a function that returns numbers, and drawn by a function that takes
those numbers and knows nothing about models.

The three measurements
----------------------
HEATMAP. For one head, the (T, T) lower-triangular matrix of attention
weights: row i is the distribution query i placed over keys 0..i. The
structures worth naming are visible by eye -- a bright diagonal band is a
local head reading the previous few tokens; a bright first column is an
attention sink, where heads dump probability mass on position 0 when they
have nothing to attend to; scattered bright cells are content-based
matching.

ENTROPY. H_i = -sum_j p_ij log p_ij for query i, averaged over queries.
Low entropy means the head commits to a few positions; high means it
averages diffusely over its context.

    Raw entropy is not comparable across query positions, because query i
    can only attend to i+1 keys and so has a hard ceiling of log(i+1)
    nats -- query 0 has entropy exactly 0 no matter what the model
    learned. So NORMALISED entropy, H_i / log(i+1), is reported alongside:
    it reads as "what fraction of the maximum possible diffuseness this
    head uses", and is comparable across positions, layers and the two
    models.

DISTANCE. D_i = sum_j p_ij * (i - j), the mean number of positions a query
looks back, averaged over queries.

    This has the same position artefact and it is worse: query 3 cannot
    have a mean distance above 3, so a genuinely long-range head is
    indistinguishable from a local one over the first few positions, and
    averaging over all queries drags every head's number down by the same
    irrelevant amount. The headline figure is therefore computed over the
    SECOND HALF of the sequence only, where every query has at least T/2
    positions of history available and heads can actually differ. The
    all-position figure is reported too, for comparison.

Why this matters for Phase 3
----------------------------
Model L's corpus is 48.9% Internet Archive OCR against Model H's 78.0%
clean Sangraha web text, and OCR noise breaks exactly the long-range
regularities a head would need in order to specialise on them. Whether
Model L's heads stay more local than Model H's is a concrete, falsifiable
version of "the data quality difference shows up in the model", and mean
attention distance is how it gets measured.

A note on axis labels
---------------------
Heatmap axes are labelled with POSITION INDICES, not with the tokens
themselves. Telugu and Bengali-Assamese glyphs require a font with those
blocks -- Nirmala UI on Windows, generally absent on the Linux images
Kaggle runs -- and matplotlib silently renders missing glyphs as empty
boxes, which produces a figure that looks broken and communicates nothing.
The tokenized text is written to the accompanying Markdown file instead,
so position i can be looked up. --token-labels forces token labels if a
suitable font is found.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")             # no display on Kaggle or in a terminal
import matplotlib.pyplot as plt

from data import TokenDataset
from evaluate import load_checkpoint, load_tokenizer
from language_config import get_language, use_utf8_output
from workspace import Workspace

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]

# Fonts that cover Telugu and Bengali-Assamese, tried in order when
# --token-labels is requested.
SCRIPT_FONTS = ["Nirmala UI", "Noto Sans Telugu", "Noto Sans Bengali",
                "Lohit Telugu", "Lohit Assamese", "FreeSerif"]


# =============================================================================
# COMPUTATION
# =============================================================================


@torch.no_grad()
def collect_attention(model, tokens, device):
    """
    Run one sequence and return its attention tensors.

    Returns a list of n_layer arrays, each (n_head, T, T), on the CPU as
    float32. Batch dimension is dropped: one sequence at a time keeps the
    memory bounded and every downstream measurement is per-sequence anyway.
    """

    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    _, _, attentions = model(idx, return_attention=True)

    return [a[0].float().cpu().numpy() for a in attentions]


def attention_entropy(attention, normalise=True):
    """
    Mean attention entropy per head, in nats.

    attention: (n_head, T, T), rows summing to 1 over allowed keys.

    Masked positions hold exactly 0 and contribute 0 to -sum p log p, which
    is the correct limit of p log p as p -> 0 but evaluates to nan in
    floating point. They are excluded rather than clamped, so the result is
    the true entropy of the distribution and not a value nudged by an
    epsilon.

    With normalise, each query's entropy is divided by log(i+1), its
    maximum possible value given that query i sees i+1 keys. Query 0 is
    skipped: its entropy is 0 out of a maximum of 0, which is not a
    meaningful ratio.
    """

    n_head, T, _ = attention.shape
    out = np.zeros(n_head)

    for head in range(n_head):
        per_query = []
        for i in range(1, T):                 # query 0 carries no information
            row = attention[head, i, : i + 1]
            row = row[row > 0]
            if row.size == 0:
                continue
            entropy = float(-(row * np.log(row)).sum())
            if normalise:
                entropy /= math.log(i + 1)
            per_query.append(entropy)
        out[head] = float(np.mean(per_query)) if per_query else 0.0

    return out


def attention_distance(attention, from_position=None):
    """
    Mean attention distance per head, in token positions.

    D_i = sum_j p_ij * (i - j), averaged over queries i >= from_position.

    from_position defaults to T // 2. Restricting to the second half is not
    cosmetic: query i can look back at most i positions, so a long-range
    head and a local head are literally indistinguishable at small i, and
    including those queries compresses every head towards the same small
    number. Over the second half every query has at least T/2 positions of
    history and the heads are free to differ.
    """

    n_head, T, _ = attention.shape
    start = T // 2 if from_position is None else int(from_position)
    start = max(1, min(start, T - 1))

    offsets = np.arange(T)
    out = np.zeros(n_head)

    for head in range(n_head):
        per_query = []
        for i in range(start, T):
            row = attention[head, i, : i + 1]
            distance = float((row * (i - offsets[: i + 1])).sum())
            per_query.append(distance)
        out[head] = float(np.mean(per_query)) if per_query else 0.0

    return out


def summarise_heads(attentions, from_position=None):
    """
    Per-layer, per-head entropy and distance for one sequence.

    Returns arrays of shape (n_layer, n_head).
    """

    entropy_norm = np.stack([attention_entropy(a, True) for a in attentions])
    entropy_raw = np.stack([attention_entropy(a, False) for a in attentions])
    distance_half = np.stack([attention_distance(a, from_position) for a in attentions])
    distance_all = np.stack([attention_distance(a, 1) for a in attentions])

    return {
        "entropy_normalised": entropy_norm,
        "entropy_nats": entropy_raw,
        "distance_second_half": distance_half,
        "distance_all_positions": distance_all,
    }


def average_over_sequences(summaries):
    """Mean of per-sequence (n_layer, n_head) arrays, key by key."""

    return {
        key: np.mean([s[key] for s in summaries], axis=0)
        for key in summaries[0]
    }


def classify_heads(entropy, distance, n_layer, n_head):
    """
    A coarse label per head, to anchor the report's discussion.

    Deliberately crude -- the thresholds are medians over this model's own
    heads, so the labels are relative to the model rather than absolute
    claims about what a head "is". They exist to point the written analysis
    at the heads worth looking at, not to substitute for looking.
    """

    entropy_median = float(np.median(entropy))
    distance_median = float(np.median(distance))

    labels = {}
    for layer in range(n_layer):
        for head in range(n_head):
            sharp = entropy[layer, head] < entropy_median
            local = distance[layer, head] < distance_median

            if sharp and local:
                label = "local, focused"
            elif sharp and not local:
                label = "long-range, focused"
            elif not sharp and local:
                label = "local, diffuse"
            else:
                label = "long-range, diffuse"

            labels[f"L{layer}H{head}"] = label

    return labels, entropy_median, distance_median


# =============================================================================
# VISUALISATION
# =============================================================================


def find_script_font():
    """Name of an installed font covering Indic blocks, or None."""

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in SCRIPT_FONTS:
        if name in available:
            return name

    return None


def plot_head_grid(attention, layer, pieces, path, language, font=None,
                   max_positions=48):
    """
    Heatmap of every head in one layer: query position vs key position.

    Cropped to the first max_positions tokens. A 512x512 heatmap printed at
    figure size is a grey smear -- the diagonal bands and sink columns that
    the analysis is about are only legible over a few dozen positions.
    """

    n_head = attention.shape[0]
    T = min(max_positions, attention.shape[1])

    columns = min(3, n_head)
    rows = math.ceil(n_head / columns)

    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.9 * rows), squeeze=False
    )

    for head in range(n_head):
        axis = axes[head // columns][head % columns]
        image = axis.imshow(
            attention[head, :T, :T],
            cmap="viridis", aspect="equal",
            vmin=0.0, vmax=float(attention[head, :T, :T].max()),
        )
        axis.set_title(f"Layer {layer}, head {head}", fontsize=10)
        axis.set_xlabel("Key position (attended to)")
        axis.set_ylabel("Query position (attending)")

        if font and pieces:
            labels = pieces[:T]
            step = max(1, T // 16)
            axis.set_xticks(range(0, T, step))
            axis.set_xticklabels(labels[::step], rotation=90,
                                 fontname=font, fontsize=6)
            axis.set_yticks(range(0, T, step))
            axis.set_yticklabels(labels[::step], fontname=font, fontsize=6)

        figure.colorbar(image, ax=axis, fraction=0.046, label="Attention weight")

    for spare in range(n_head, rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    figure.suptitle(
        f"{language}: attention weights, layer {layer} "
        f"(first {T} positions; upper triangle is masked)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_layer_head_matrix(values, title, colorbar_label, path, cmap="magma"):
    """Layer x head matrix as a labelled heatmap with per-cell values."""

    n_layer, n_head = values.shape

    figure, axis = plt.subplots(figsize=(1.0 * n_head + 3.5, 0.42 * n_layer + 2.5))
    image = axis.imshow(values, cmap=cmap, aspect="auto")

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Head")
    axis.set_ylabel("Layer (0 = closest to the embeddings)")
    axis.set_xticks(range(n_head))
    axis.set_yticks(range(n_layer))

    for layer in range(n_layer):
        for head in range(n_head):
            axis.text(head, layer, f"{values[layer, head]:.2f}",
                      ha="center", va="center", fontsize=7,
                      color="white" if values[layer, head] < values.max() * 0.6
                      else "black")

    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_by_layer(entropy, distance, title, path):
    """Per-layer means of both metrics, on twin axes with a legend."""

    n_layer = entropy.shape[0]
    layers = np.arange(n_layer)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    twin = axis.twinx()

    line_e = axis.plot(layers, entropy.mean(axis=1), "o-",
                       color="#3b6ea5", label="Normalised entropy")
    line_d = twin.plot(layers, distance.mean(axis=1), "s--",
                       color="#b5533c", label="Mean attention distance")

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Layer (0 = closest to the embeddings)")
    axis.set_ylabel("Normalised attention entropy (0 = sharp, 1 = uniform)",
                    color="#3b6ea5")
    twin.set_ylabel("Mean attention distance (token positions)", color="#b5533c")
    axis.set_xticks(layers)
    axis.grid(alpha=0.3)

    lines = line_e + line_d
    axis.legend(lines, [line.get_label() for line in lines], loc="best")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# ORCHESTRATION
# =============================================================================


def write_notes(path, language, sequences, labels, entropy_median,
                distance_median, summary):
    """Markdown accompanying the figures: the example text and head labels."""

    n_layer, n_head = summary["entropy_normalised"].shape

    lines = [
        f"# {language}: attention analysis",
        "",
        "Heatmap axes are labelled by POSITION, not token, because matplotlib",
        "renders missing Indic glyphs as empty boxes on images without a font",
        "covering the script. The tokenized sequences below let position i be",
        "looked up.",
        "",
        f"Head labels are relative to this model's own medians "
        f"(normalised entropy {entropy_median:.3f}, "
        f"distance {distance_median:.2f} positions), not absolute claims.",
        "",
        "## Example sequences",
        "",
    ]

    for i, sequence in enumerate(sequences):
        lines += [
            f"### Sequence {i} ({sequence['source']})",
            "",
            f"> {sequence['text']}",
            "",
            "Tokens by position:",
            "",
            "```",
            " ".join(f"{j}:{piece}" for j, piece in enumerate(sequence["pieces"][:48])),
            "```",
            "",
        ]

    lines += ["## Head classification", "", "| Head | Label | Norm. entropy | Distance |",
              "|---|---|---|---|"]

    for layer in range(n_layer):
        for head in range(n_head):
            key = f"L{layer}H{head}"
            lines.append(
                f"| {key} | {labels[key]} | "
                f"{summary['entropy_normalised'][layer, head]:.3f} | "
                f"{summary['distance_second_half'][layer, head]:.2f} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention heatmaps, entropy and distance for one model.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-sequences", type=int, default=8,
                        help="Sequences averaged for the entropy/distance tables.")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Tokens per analysed sequence.")
    parser.add_argument("--heatmap-positions", type=int, default=48)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to draw head grids for. Default: first and last.")
    parser.add_argument("--token-labels", action="store_true",
                        help="Label heatmap axes with tokens if a font supports them.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default=None)

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
    tokenizer = load_tokenizer(workspace, config.vocab_size)

    index = json.loads(
        (data_dir / f"{args.split}_documents.json").read_text(encoding="utf-8")
    )

    seq_len = min(args.seq_len, config.block_size)
    layers = args.layers or [0, config.n_layer - 1]

    out_dir = Path(args.out) if args.out else (
        REPO / "report" / "phase2" / workspace.slug
    )
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    font = find_script_font() if args.token_labels else None

    print("=" * 78)
    print(f"ATTENTION - {workspace.title}")
    print("=" * 78)
    print(f"  checkpoint : {Path(args.checkpoint).name} (step {state.get('step', '?'):,})")
    print(f"  device     : {device}")
    print(f"  layers     : {config.n_layer}, heads {config.n_head}, "
          f"d_k {config.d_head}")
    print(f"  sequences  : {args.n_sequences} x {seq_len} tokens from {args.split}")
    print(f"  heatmaps   : layers {layers}, first {args.heatmap_positions} positions")
    print(f"  font       : {font or 'position labels (no Indic font requested/found)'}")

    # ---- choose sequences, spread across sources ---------------------------
    rng = np.random.default_rng(args.seed)
    eligible = [d for d in index["documents"] if d[1] >= seq_len]
    picked = rng.choice(len(eligible), size=min(args.n_sequences, len(eligible)),
                        replace=False)

    sequences = []
    summaries = []
    first_attention = None

    print(f"\n  analysing ...")
    for n, i in enumerate(picked):
        offset, _, source = eligible[i]
        tokens = np.asarray(dataset.tokens[offset:offset + seq_len], dtype=np.int64)

        attentions = collect_attention(model, tokens, device)
        summaries.append(summarise_heads(attentions))

        sequences.append({
            "source": source,
            "offset": int(offset),
            "text": tokenizer.decode(tokens.tolist()),
            "pieces": [tokenizer.id_to_piece(int(t)) for t in tokens],
        })

        if first_attention is None:
            first_attention = attentions

        print(f"    {n + 1}/{len(picked)}  {source}", flush=True)

    summary = average_over_sequences(summaries)
    labels, entropy_median, distance_median = classify_heads(
        summary["entropy_normalised"], summary["distance_second_half"],
        config.n_layer, config.n_head,
    )

    # ---- figures ------------------------------------------------------------
    title_language = workspace.title

    for layer in layers:
        plot_head_grid(
            first_attention[layer], layer, sequences[0]["pieces"],
            figures / f"attention_heatmap_layer{layer}.png",
            title_language, font, args.heatmap_positions,
        )

    plot_layer_head_matrix(
        summary["entropy_normalised"],
        f"{title_language}: normalised attention entropy per head\n"
        f"(0 = sharp and decisive, 1 = uniform over available context)",
        "Normalised entropy",
        figures / "attention_entropy.png",
    )
    plot_layer_head_matrix(
        summary["distance_second_half"],
        f"{title_language}: mean attention distance per head\n"
        f"(token positions looked back, queries in the second half only)",
        "Mean distance (positions)",
        figures / "attention_distance.png",
        cmap="viridis",
    )
    plot_by_layer(
        summary["entropy_normalised"], summary["distance_second_half"],
        f"{title_language}: attention behaviour by depth",
        figures / "attention_by_layer.png",
    )

    # ---- tables -------------------------------------------------------------
    result = {
        "language": config.language,
        "model": config.name,
        "checkpoint": str(args.checkpoint),
        "step": state.get("step"),
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "seq_len": seq_len,
        "n_sequences": len(sequences),
        "medians": {
            "entropy_normalised": entropy_median,
            "distance_second_half": distance_median,
        },
        "per_head": {
            key: values.tolist() for key, values in summary.items()
        },
        "head_labels": labels,
    }
    (out_dir / "attention.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_notes(out_dir / "attention_notes.md", title_language, sequences,
                labels, entropy_median, distance_median, summary)

    # ---- console summary ----------------------------------------------------
    print(f"\n  MEAN BY LAYER")
    print(f"    {'layer':<8}{'norm entropy':>15}{'distance (2nd half)':>22}")
    print("    " + "-" * 45)
    for layer in range(config.n_layer):
        print(f"    {layer:<8}"
              f"{summary['entropy_normalised'][layer].mean():>15.3f}"
              f"{summary['distance_second_half'][layer].mean():>22.2f}")

    print(f"\n  wrote {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
