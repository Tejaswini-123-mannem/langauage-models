"""
Encoding, batching and scoring for the reasoning task.

Shared by finetune.py (training loss, validation accuracy) and
reasoning_eval.py (the test-set report), so the two cannot disagree about
what an example looks like to the model or what counts as a correct
answer.

What the model sees
-------------------
One example is encoded as

    [prompt pieces] [answer pieces] [EOS]

where the prompt ends at the answer prefix ("సమాధానం:" / "উত্তৰ:"). Prompt
and answer are encoded separately and concatenated. SentencePiece gives
the answer its own leading ▁, which is exactly the space that separates
it from the prefix, and doing it this way guarantees the answer's pieces
are identical whether the answer is being trained on, generated, or
scored as a choice. Encoding "prompt answer" as one string instead lets
the tokenizer merge across the boundary, and the answer's first piece
would then differ between training and scoring.

EOS after the answer is what lets greedy decoding stop by itself. It is
the same EOS that separated documents in pretraining, so the model already
knows it as "the text ends here".

Loss on the answer only
-----------------------
Targets are -100 everywhere except the answer pieces and the EOS, and
cross_entropy ignores -100 by default, so model.forward needs no change.
Training on the prompt too would spend most of the gradient on predicting
the templated context -- names drawn uniformly at random and numbers,
which are unpredictable by construction -- and dilute the signal from the
one position that requires reasoning.

Two accuracies
--------------
EXACT MATCH. Greedy-decode after the prompt until EOS, and compare the
decoded string with the answer. This is the real task, and it is what the
finetuned model is judged on.

CHOICE ACCURACY. Score each candidate answer by the log-probability of its
pieces given the prompt, and pick the highest. This needs no generation
and no knowledge of the answer format -- the candidates are supplied --
so it is the fair way to measure a PRETRAINED model, which has never seen
"సమాధానం:" followed by a one-word answer and would score ~0 exact match
for reasons unrelated to reasoning. Chance is 1/len(choices).

Summed rather than length-normalised log-probability. Candidates here are
names or single labels of one to four pieces; normalising by length would
favour long names, and summing is the quantity the model actually assigns
to the string.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

IGNORE = -100


# =============================================================================
# DATA
# =============================================================================


def load_examples(path, limit=None):
    """Read one split's .jsonl into a list of dicts."""

    examples = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break

    return examples


def encode_examples(examples, tokenizer):
    """
    Attach token ids to each example, in place.

    prompt_ids  pieces of the prompt, ending at the answer prefix
    answer_ids  pieces of the answer followed by EOS
    choice_ids  pieces of each candidate, without EOS
    """

    eos = tokenizer.eos_id()

    for e in examples:
        e["prompt_ids"] = tokenizer.encode(e["prompt"])
        e["answer_ids"] = tokenizer.encode(e["answer"]) + [eos]
        e["choice_ids"] = [tokenizer.encode(c) for c in e["choices"]]

    return examples


def collate(batch, pad_id, device, block_size):
    """
    Right-padded (input, target) tensors for a list of encoded examples.

    Right padding, not left: positions are learned absolute embeddings, so
    left padding would shift every real token to a different position row
    than it had in pretraining. Padding on the right leaves the prompt at
    positions 0.. exactly as a document start was, and the pad positions
    are never attended to by any real token because of the causal mask.
    """

    sequences = [e["prompt_ids"] + e["answer_ids"] for e in batch]
    length = min(block_size + 1, max(len(s) for s in sequences))

    x = torch.full((len(batch), length - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), length - 1), IGNORE, dtype=torch.long)

    for row, (e, seq) in enumerate(zip(batch, sequences)):
        seq = seq[:length]
        n = len(seq)
        x[row, : n - 1] = torch.tensor(seq[:-1])

        # Supervise only the answer and EOS: target position t predicts
        # token t+1, so the first answer piece is the target at position
        # len(prompt) - 1.
        start = len(e["prompt_ids"]) - 1
        y[row, start : n - 1] = torch.tensor(seq[start + 1 : n])

    return x.to(device), y.to(device)


# =============================================================================
# SCORING
# =============================================================================


@torch.no_grad()
def greedy_answer(model, prompt_ids, eos_id, max_new_tokens=12, device="cpu"):
    """
    Greedy-decode after one prompt until EOS. Returns the new token ids.

    One example at a time: batching generation would need left padding
    for the prompts to end at the same column, which the learned absolute
    positions rule out (see collate). Answers are a few pieces long, so
    the cost is a handful of forward passes per example.
    """

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []

    for _ in range(max_new_tokens):
        logits, _, _ = model(idx[:, -model.config.block_size:])
        token = int(logits[0, -1].argmax())
        if token == eos_id:
            break
        out.append(token)
        idx = torch.cat([idx, torch.tensor([[token]], device=device)], dim=1)

    return out


@torch.no_grad()
def choice_logprobs(model, prompt_ids, choice_ids, pad_id, device="cpu"):
    """
    Summed log p(choice | prompt) for every candidate, in one forward pass.

    All candidates share the prompt, so they are stacked as rows of one
    right-padded batch; each row's score reads only its own choice pieces.
    """

    rows = [prompt_ids + c for c in choice_ids]
    length = max(len(r) for r in rows)

    idx = torch.full((len(rows), length), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        idx[i, : len(r)] = torch.tensor(r)
    idx = idx.to(device)

    logits, _, _ = model(idx)
    logp = F.log_softmax(logits.float(), dim=-1)

    scores = []
    p = len(prompt_ids)
    for i, c in enumerate(choice_ids):
        # Piece c[k] sits at position p + k and is predicted from p + k - 1.
        positions = torch.arange(p - 1, p - 1 + len(c), device=device)
        targets = torch.tensor(c, device=device)
        scores.append(float(logp[i, positions, targets].sum()))

    return scores


def normalise_answer(text):
    """Strip whitespace and trailing punctuation a model may add."""

    return text.strip().strip(".।?!,:;").strip()


def score_example(model, tokenizer, example, device, prompt_ids=None,
                  generate=True):
    """
    Exact-match and choice results for one encoded example.

    prompt_ids overrides the example's own, which is how few-shot prompts
    are scored: the demonstrations are prepended, the choices are not.
    """

    prompt_ids = prompt_ids or example["prompt_ids"]
    result = {"id": example["id"], "task": example["task"],
              "hops": example["hops"], "attribute": example["attribute"],
              "answer": example["answer"]}

    scores = choice_logprobs(model, prompt_ids, example["choice_ids"],
                             tokenizer.pad_id(), device)
    chosen = example["choices"][max(range(len(scores)), key=scores.__getitem__)]
    result["choice"] = chosen
    result["choice_correct"] = chosen == example["answer"]

    if generate:
        ids = greedy_answer(model, prompt_ids, tokenizer.eos_id(), device=device)
        text = normalise_answer(tokenizer.decode(ids))
        result["generated"] = text
        result["exact_match"] = text == example["answer"]

    return result


def build_fewshot_prompt(example, demonstrations, tokenizer, block_size,
                         reserve=16):
    """
    Prompt ids with solved demonstrations prepended, EOS between each.

    Demonstrations are dropped from the front if the whole prompt would not
    leave `reserve` positions for the answer inside block_size.
    """

    eos = tokenizer.eos_id()
    parts = [d["prompt_ids"] + d["answer_ids"] for d in demonstrations]

    while parts and sum(map(len, parts)) + len(example["prompt_ids"]) + reserve > block_size:
        parts.pop(0)

    ids = []
    for p in parts:
        ids.extend(p)                          # answer_ids already ends in EOS
    ids.extend(example["prompt_ids"])

    return ids


def tokenizer_for(language_slug, vocab_size):
    """The language's final SentencePiece model."""

    import sentencepiece as spm
    from workspace import Workspace

    workspace = Workspace(language_slug)
    path = workspace.final_model(vocab_size)
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing tokenizer {path}")

    return spm.SentencePieceProcessor(model_file=str(path))
