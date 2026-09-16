"""
Attention before and after reasoning finetuning, on reasoning prompts.

    python common/reasoning_attention.py \
        --pretrained C:/lma_work/telugu/checkpoints/best.pt \
        --finetuned  C:/lma_work/telugu/runs/model_h_reasoning/best.pt

Reuses the Phase 2 toolkit -- common/attention.py's collect_attention,
entropy and distance functions -- unchanged, so a pretrained-versus-
finetuned difference cannot be an artefact of measuring the two checkpoints
with different code. What is new is the input and one measurement.

Input: reasoning prompts, not corpus text
-----------------------------------------
Phase 2 measured attention on 128-token windows of held-out prose. The
question here is whether finetuning changed how the model reads a
comparison, so the sequences are test-split reasoning prompts, identical
for both checkpoints. Chain and numeric prompts are summarised separately:
a chain prompt can only be answered by relating names across sentences,
a numeric prompt by relating numbers, and they may well recruit different
heads.

Distance is measured over ALL query positions here, not only the second
half as in Phase 2. The second-half rule existed because query i cannot
look further back than i, and on 128-token windows the first half was
dominated by that ceiling. Reasoning prompts are ~40-60 tokens, so the
second half would discard the premises entirely -- the part of the prompt
the analysis is about. Both checkpoints are measured the same way, so the
comparison is unaffected.

New measurement: where the answer position looks
------------------------------------------------
The last prompt position (the answer prefix's colon) is the query that
produces the answer. For each layer and head, the share of its attention
that lands on pieces belonging to ENTITY NAMES, and separately on DIGITS,
is recorded. If finetuning teaches the model to answer by retrieving a name
from the premises, the late layers' answer-position mass on names should
rise; if it learns a shortcut that ignores the premises, it will not.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attention import attention_distance, attention_entropy, collect_attention
from evaluate import load_checkpoint
from language_config import get_language, use_utf8_output
from reasoning_task import encode_examples, load_examples, tokenizer_for
from workspace import REPO, Workspace

use_utf8_output()

CHAIN_TASKS = {"chain_pair", "chain_extreme", "chain_yesno"}


# =============================================================================
# COMPUTATION
# =============================================================================


def piece_spans(tokenizer, ids):
    """
    Character span (start, end) of each piece within the decoded prompt.

    Measured as the length of the decoded prefix after each piece. Slower
    than the offsets SentencePiece can return, but that API has changed
    shape between releases (immutable_proto was removed), and the Kaggle
    image and this machine run different versions. Prompts are ~50 pieces,
    so the quadratic cost is irrelevant.
    """

    spans, previous = [], 0
    for k in range(len(ids)):
        end = len(tokenizer.decode(ids[: k + 1]))
        spans.append((previous, end))
        previous = end

    return spans


def position_masks(example, tokenizer):
    """
    Boolean masks over prompt positions: pieces inside entity names, pieces
    containing digits.
    """

    ids = example["prompt_ids"]
    text = tokenizer.decode(ids)
    spans = piece_spans(tokenizer, ids)

    # Entities are the example's name/item choices; for yes/no and relation
    # tasks the choices are labels, so names are recovered from the prompt
    # by matching the full name lists in the lexicon instead.
    names = example.get("entities") or []
    char_is_entity = np.zeros(len(text) + 1, dtype=bool)
    for name in names:
        for match in re.finditer(re.escape(name), text):
            char_is_entity[match.start():match.end()] = True

    entity = np.array([char_is_entity[s:e].any() if e > s else False for s, e in spans])
    digit = np.array([bool(re.search(r"\d", text[s:e])) for s, e in spans])

    return entity, digit


def measure(model, tokenizer, examples, device):
    """
    Per-example attention summaries for one checkpoint.

    Returns (n_layer, n_head) arrays averaged over the examples, for all
    examples and for the chain and numeric subsets separately, plus the
    raw attention of the first example for the heatmaps.
    """

    rows = []
    first = None

    for e in examples:
        attentions = collect_attention(model, e["prompt_ids"], device)
        if first is None:
            first = attentions

        entity, digit = position_masks(e, tokenizer)
        last = len(e["prompt_ids"]) - 1

        rows.append({
            "chain": e["task"] in CHAIN_TASKS,
            "entropy": np.stack([attention_entropy(a, True) for a in attentions]),
            "distance": np.stack([attention_distance(a, 1) for a in attentions]),
            # attention[head, last, :] is the answer position's distribution.
            "entity_mass": np.stack([a[:, last, :][:, entity].sum(axis=1) for a in attentions]),
            "digit_mass": np.stack([a[:, last, :][:, digit].sum(axis=1) for a in attentions]),
        })

    def mean(subset):
        keys = ("entropy", "distance", "entity_mass", "digit_mass")
        return {k: np.mean([r[k] for r in subset], axis=0) for k in keys} if subset else None

    return {
        "all": mean(rows),
        "chain": mean([r for r in rows if r["chain"]]),
        "numeric": mean([r for r in rows if not r["chain"]]),
    }, first


# =============================================================================
# VISUALISATION
# =============================================================================


def plot_heatmap_comparison(pre, post, layer, title, path):
    """Rows: pretrained, finetuned. Columns: every head of one layer."""

    n_head, T, _ = pre[layer].shape
    figure, axes = plt.subplots(2, n_head, figsize=(2.9 * n_head, 6.4), squeeze=False)

    for row, (label, attn) in enumerate((("Pretrained", pre), ("Finetuned", post))):
        for head in range(n_head):
            axis = axes[row][head]
            image = axis.imshow(attn[layer][head], cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"{label}, head {head}", fontsize=9)
            axis.set_xlabel("Key position", fontsize=8)
            axis.set_ylabel("Query position", fontsize=8)
            axis.tick_params(labelsize=6)

    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, label="Attention weight")
    figure.suptitle(f"{title}: layer {layer}, pretrained vs finetuned, same reasoning prompt",
                    fontsize=12)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_delta_matrix(delta, title, colorbar_label, path):
    """Layer x head difference (finetuned - pretrained) on a diverging scale."""

    n_layer, n_head = delta.shape
    bound = float(np.abs(delta).max()) or 1.0

    figure, axis = plt.subplots(figsize=(n_head + 3.5, 0.42 * n_layer + 2.5))
    image = axis.imshow(delta, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for layer in range(n_layer):
        for head in range(n_head):
            axis.text(head, layer, f"{delta[layer, head]:+.2f}", ha="center",
                      va="center", fontsize=7)

    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Head")
    axis.set_ylabel("Layer (0 = closest to the embeddings)")
    axis.set_xticks(range(n_head))
    axis.set_yticks(range(n_layer))
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_by_layer(series, ylabel, title, path):
    """Per-layer lines; `series` maps legend label -> (n_layer,) array."""

    figure, axis = plt.subplots(figsize=(8, 4.5))

    # Marker shape separates the prompt type, colour the checkpoint, so the
    # pretrained/finetuned contrast the figure is about reads at a glance.
    # Colours are the first slots of the same validated categorical palette
    # the other Phase 3 figures use.
    styles = [("o-", "#2a78d6"), ("o--", "#eb6834"), ("s-", "#1baf7a"), ("s--", "#eda100")]

    for (label, values), (style, colour) in zip(series.items(), styles):
        axis.plot(np.arange(len(values)), values, style, color=colour, label=label)

    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Layer (0 = closest to the embeddings)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(next(iter(series.values())))))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# ORCHESTRATION
# =============================================================================


def attach_entities(examples, lexicon_names):
    """Record which lexicon names/items occur in each prompt."""

    for e in examples:
        e["entities"] = [n for n in lexicon_names if n in e["prompt"]]

    return examples


def pick_examples(examples, n):
    """First n examples, which round-robin generation balances over tasks."""

    return examples[:n]


def main():
    parser = argparse.ArgumentParser(description="Pretrained vs finetuned attention.")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-examples", type=int, default=120)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    pre_model, config, _ = load_checkpoint(args.pretrained, device)
    post_model, post_config, post_state = load_checkpoint(args.finetuned, device)
    assert config.language == post_config.language, "checkpoints are different languages"

    workspace = Workspace(get_language(config.language))
    tokenizer = tokenizer_for(workspace.slug, config.vocab_size)
    lexicon = json.loads((REPO / workspace.slug / "configs" / "reasoning.json")
                         .read_text(encoding="utf-8"))
    names = sorted(lexicon["people"] + lexicon["items"], key=len, reverse=True)

    data_dir = Path(args.data) if args.data else workspace.root / "reasoning"
    out_dir = Path(args.out) if args.out else REPO / "report" / "phase3" / workspace.slug
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    examples = load_examples(data_dir / f"{args.split}.jsonl", limit=args.n_examples)
    examples = attach_entities(encode_examples(examples, tokenizer), names)

    # The heatmap example: the first chain prompt, since that is where a
    # change in cross-sentence attention would be visible.
    chain_first = next(i for i, e in enumerate(examples) if e["task"] in CHAIN_TASKS)
    examples.insert(0, examples.pop(chain_first))

    print(f"ATTENTION COMPARISON - {workspace.title}: {len(examples)} {args.split} prompts")
    pre, pre_first = measure(pre_model, tokenizer, examples, device)
    post, post_first = measure(post_model, tokenizer, examples, device)

    title = workspace.title
    layers = args.layers or [0, config.n_layer - 1]

    for layer in layers:
        plot_heatmap_comparison(pre_first, post_first, layer, title,
                                figures / f"finetune_attention_heatmap_layer{layer}.png")

    plot_delta_matrix(
        post["all"]["entropy"] - pre["all"]["entropy"],
        f"{title}: change in normalised attention entropy after finetuning\n"
        f"(finetuned minus pretrained; blue = sharper, red = more diffuse)",
        "Delta normalised entropy", figures / "finetune_entropy_delta.png")
    plot_delta_matrix(
        post["all"]["distance"] - pre["all"]["distance"],
        f"{title}: change in mean attention distance after finetuning\n"
        f"(finetuned minus pretrained; red = longer range, blue = more local)",
        "Delta mean distance (positions)", figures / "finetune_distance_delta.png")

    plot_by_layer(
        {"Pretrained, entropy": pre["all"]["entropy"].mean(axis=1),
         "Finetuned, entropy": post["all"]["entropy"].mean(axis=1)},
        "Normalised attention entropy (0 = sharp, 1 = uniform)",
        f"{title}: attention entropy by layer on reasoning prompts",
        figures / "finetune_entropy_by_layer.png")
    plot_by_layer(
        {"Pretrained, distance": pre["all"]["distance"].mean(axis=1),
         "Finetuned, distance": post["all"]["distance"].mean(axis=1)},
        "Mean attention distance (token positions)",
        f"{title}: attention distance by layer on reasoning prompts",
        figures / "finetune_distance_by_layer.png")
    plot_by_layer(
        {"Pretrained, chain prompts": pre["chain"]["entity_mass"].mean(axis=1),
         "Finetuned, chain prompts": post["chain"]["entity_mass"].mean(axis=1),
         "Pretrained, numeric prompts": pre["numeric"]["entity_mass"].mean(axis=1),
         "Finetuned, numeric prompts": post["numeric"]["entity_mass"].mean(axis=1)},
        "Answer-position attention on entity-name pieces",
        f"{title}: how much the answer position attends to names",
        figures / "finetune_entity_attention_by_layer.png")

    def to_lists(group):
        return {k: v.tolist() for k, v in group.items()} if group else None

    result = {
        "language": workspace.slug,
        "pretrained": str(args.pretrained),
        "finetuned": str(args.finetuned),
        "finetuned_step": post_state.get("step"),
        "n_examples": len(examples),
        "distance_positions": "all query positions (see module docstring)",
        "per_layer_mean": {
            which: {
                group: {k: v.mean(axis=1).round(4).tolist() for k, v in data[group].items()}
                for group in ("all", "chain", "numeric") if data[group]
            }
            for which, data in (("pretrained", pre), ("finetuned", post))
        },
        "per_head": {"pretrained": {g: to_lists(pre[g]) for g in pre},
                     "finetuned": {g: to_lists(post[g]) for g in post}},
        "heatmap_example": {
            "task": examples[0]["task"],
            "prompt": examples[0]["prompt"],
            "pieces": [f"{i}:{tokenizer.id_to_piece(t)}"
                       for i, t in enumerate(examples[0]["prompt_ids"])],
        },
    }
    (out_dir / "attention_finetune_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {'layer':<6}{'entropy pre':>12}{'post':>8}{'dist pre':>10}{'post':>8}"
          f"{'name@ans pre':>14}{'post':>8}")
    for layer in range(config.n_layer):
        print(f"  {layer:<6}{pre['all']['entropy'][layer].mean():>12.3f}"
              f"{post['all']['entropy'][layer].mean():>8.3f}"
              f"{pre['all']['distance'][layer].mean():>10.2f}"
              f"{post['all']['distance'][layer].mean():>8.2f}"
              f"{pre['chain']['entity_mass'][layer].mean():>14.3f}"
              f"{post['chain']['entity_mass'][layer].mean():>8.3f}")
    print(f"\n  wrote {out_dir}")


if __name__ == "__main__":
    main()
