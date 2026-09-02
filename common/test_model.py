"""
Correctness tests for the from-scratch Transformer.

Why these specific tests
------------------------
Three of them are graded evidence rather than developer hygiene.

The assignment asks, in so many words, to "verify empirically that the
model cannot see the future -- for example, show that changing token t+1
does not change the logits at position t". test_causality does exactly
that, and its output is quoted in the Phase 2 report.

It also asks for the exact parameter count per model. test_parameter_count
asserts it, so a config typo cannot silently produce a 40M model whose
reported count was copied from a table rather than measured.

The rest exist because the alternative is discovering the problem four
hours into a Kaggle session, and the weekly GPU quota is 30 hours for two
pretraining runs plus Phase 3. A broken backward pass costs a run;
test_overfit_single_batch catches it in about twenty seconds on a CPU.

Run
---
    python common/test_model.py
    python common/test_model.py --config telugu/configs/model_h.json
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import use_utf8_output
from model import GPT, GPTConfig, build_model

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]

# Small model for the fast structural tests. Same code path as the real
# thing; a 12-layer 384-wide model on a CPU makes the suite tedious enough
# that it stops being run.
TINY = GPTConfig(
    name="tiny", language="test",
    vocab_size=97, block_size=16, n_layer=2, n_head=4, d_model=32, d_ff=128,
    dropout=0.0,
)


# =============================================================================
# HELPERS
# =============================================================================


def report(name, passed, detail=""):
    """Print one test result and return whether it passed."""

    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))

    return passed


# =============================================================================
# TESTS
# =============================================================================


def test_shapes():
    """Every tensor in the forward pass has the shape the report claims."""

    torch.manual_seed(0)
    model = GPT(TINY).eval()

    B, T = 3, 12
    idx = torch.randint(0, TINY.vocab_size, (B, T))
    targets = torch.randint(0, TINY.vocab_size, (B, T))

    logits, loss, attentions = model(idx, targets, return_attention=True)

    checks = [
        (logits.shape == (B, T, TINY.vocab_size), f"logits {tuple(logits.shape)}"),
        (loss.ndim == 0, f"loss ndim {loss.ndim}"),
        (len(attentions) == TINY.n_layer, f"{len(attentions)} attention tensors"),
        (
            all(a.shape == (B, TINY.n_head, T, T) for a in attentions),
            f"attention {tuple(attentions[0].shape)}",
        ),
    ]

    ok = all(c for c, _ in checks)

    return report(
        "shapes",
        ok,
        f"logits (B,T,V)={tuple(logits.shape)}, "
        f"attention (B,h,T,T)={tuple(attentions[0].shape)}, "
        f"{len(attentions)} layers",
    )


def test_attention_is_a_distribution():
    """
    Attention rows sum to 1, and are exactly 0 on masked positions.

    Exactly zero, not merely small: the mask is additive -inf applied
    before the softmax, so exp(-inf) is 0 by construction. A tolerance
    here would hide a mask applied after the softmax, which would leave
    rows not summing to 1 and leak the future into the normaliser.
    """

    torch.manual_seed(0)
    model = GPT(TINY).eval()

    T = 12
    idx = torch.randint(0, TINY.vocab_size, (2, T))
    _, _, attentions = model(idx, return_attention=True)

    sums_to_one = True
    future_is_zero = True

    for att in attentions:
        sums_to_one &= torch.allclose(
            att.sum(dim=-1), torch.ones_like(att.sum(dim=-1)), atol=1e-5
        )
        # Strictly upper triangle: key position > query position.
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        future_is_zero &= bool((att[..., upper] == 0).all())

    return report(
        "attention is a causal distribution",
        sums_to_one and future_is_zero,
        f"rows sum to 1: {sums_to_one}; future weight exactly 0: {future_is_zero}",
    )


def test_causality():
    """
    THE causal-mask proof the assignment asks for.

    Take a sequence, change exactly one token at position t+1, and confirm
    the logits at every position <= t are bit-identical. If the mask leaked,
    changing a future token would perturb earlier logits and the maximum
    absolute difference would be non-zero.

    Compared with torch.equal rather than a tolerance, because the claim is
    that those positions never touched the changed token at all -- not that
    they touched it a little.
    """

    torch.manual_seed(0)
    model = GPT(TINY).eval()

    T = 12
    t = 6                                        # position we check
    a = torch.randint(0, TINY.vocab_size, (1, T))
    b = a.clone()

    # Change token t+1 to something guaranteed different.
    b[0, t + 1] = (a[0, t + 1] + 1) % TINY.vocab_size

    with torch.no_grad():
        logits_a, _, _ = model(a)
        logits_b, _, _ = model(b)

    prefix_identical = torch.equal(logits_a[:, : t + 1], logits_b[:, : t + 1])
    max_prefix_delta = (logits_a[:, : t + 1] - logits_b[:, : t + 1]).abs().max().item()

    # The change must actually reach somewhere, or the test proves nothing:
    # a model that ignored its input entirely would also pass the above.
    suffix_changed = not torch.equal(logits_a[:, t + 1:], logits_b[:, t + 1:])
    max_suffix_delta = (logits_a[:, t + 1:] - logits_b[:, t + 1:]).abs().max().item()

    return report(
        "causality (changing token t+1 does not move logits at <= t)",
        prefix_identical and suffix_changed,
        f"max |delta| at positions 0..{t} = {max_prefix_delta:.2e}; "
        f"at positions {t + 1}..{T - 1} = {max_suffix_delta:.4f}",
    )


def test_no_position_leak_through_padding():
    """
    A shorter sequence gives the same logits as the prefix of a longer one.

    This is causality stated the other way round. Running the model on the
    first t+1 tokens alone must reproduce, exactly, what it produced for
    those positions inside a full-length sequence -- because nothing after
    position t was ever visible to them. A mask that was off by one along
    the diagonal passes test_causality (which only perturbs t+1) but fails
    here.
    """

    torch.manual_seed(0)
    model = GPT(TINY).eval()

    T, short = 12, 7
    idx = torch.randint(0, TINY.vocab_size, (1, T))

    with torch.no_grad():
        full, _, _ = model(idx)
        part, _, _ = model(idx[:, :short])

    delta = (full[:, :short] - part).abs().max().item()

    return report(
        "prefix invariance (short run == prefix of long run)",
        delta < 1e-5,
        f"max |delta| = {delta:.2e}",
    )


def test_weight_tying():
    """lm_head shares storage with the token embedding, and is counted once."""

    tied = GPT(GPTConfig(**{**TINY.to_dict(), "tie_weights": True}))
    untied = GPT(GPTConfig(**{**TINY.to_dict(), "tie_weights": False}))

    shared = tied.lm_head.weight is tied.token_embedding.weight
    saving = untied.num_parameters() - tied.num_parameters()
    expected = TINY.vocab_size * TINY.d_model

    return report(
        "weight tying",
        shared and saving == expected,
        f"shared storage: {shared}; saves {saving:,} params "
        f"(vocab x d_model = {expected:,})",
    )


def test_parameter_count(config_path, expected=None):
    """
    Exact parameter count for a real config, with the per-component split.

    Reported rather than only asserted: the breakdown is the table the
    Phase 2 report prints, and printing it from the live model is what
    keeps the report honest.
    """

    model, config = build_model(config_path)
    total = model.num_parameters()
    breakdown = model.parameter_breakdown()

    print(f"\n    {Path(config_path).name}  ({config.name}, {config.language})")
    print(f"      d_model {config.d_model}, {config.n_layer} layers, "
          f"{config.n_head} heads (d_k={config.d_head}), d_ff {config.d_ff}, "
          f"block {config.block_size}, vocab {config.vocab_size:,}")
    for key in ("token_embedding", "position_embedding", "blocks",
                "final_layernorm", "lm_head"):
        print(f"      {key:<20}{breakdown[key]:>12,}")
    print(f"      {'block_each':<20}{breakdown['block_each']:>12,}")
    print(f"      {'TOTAL':<20}{total:>12,}")

    consistent = breakdown["total"] == total
    on_budget = 20_000_000 <= total <= 30_000_000
    matches = expected is None or total == expected

    return report(
        f"parameter count {Path(config_path).name}",
        consistent and on_budget and matches,
        f"{total:,} trainable ({total / 1e6:.2f}M), within ~25M budget",
    )


def test_ablation_removes_positions():
    """The bonus ablation switch removes the table rather than zeroing it."""

    ablated = GPT(GPTConfig(**{**TINY.to_dict(), "use_positional_embedding": False}))
    normal = GPT(TINY)

    gone = ablated.position_embedding is None
    saving = normal.num_parameters() - ablated.num_parameters()
    expected = TINY.block_size * TINY.d_model

    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    runs = ablated(idx)[0].shape == (1, 8, TINY.vocab_size)

    return report(
        "positional-embedding ablation switch",
        gone and saving == expected and runs,
        f"table removed: {gone}; frees {saving:,} params; forward pass runs",
    )


def test_overfit_single_batch():
    """
    The model can memorise one batch, so the backward pass works.

    A model that cannot drive the loss on 8 x 16 tokens towards zero has a
    broken gradient path -- a detached tensor, a mis-shaped loss, an
    optimizer stepping the wrong parameters. Twenty seconds here is cheaper
    than four hours into a Kaggle session.

    Dropout is off: with it on, the loss floor is not near zero and the
    test would have no clean threshold.
    """

    torch.manual_seed(0)
    config = GPTConfig(**{**TINY.to_dict(), "dropout": 0.0})
    model = GPT(config).train()

    idx = torch.randint(0, config.vocab_size, (8, config.block_size))
    targets = torch.randint(0, config.vocab_size, (8, config.block_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    _, first, _ = model(idx, targets)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(idx, targets)
        loss.backward()
        optimizer.step()

    return report(
        "overfits a single batch (backward pass is intact)",
        loss.item() < 0.1,
        f"loss {first.item():.4f} -> {loss.item():.4f} in 300 steps",
    )


def test_generation():
    """Greedy decoding is deterministic; temperature sampling is not."""

    torch.manual_seed(0)
    model = GPT(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (1, 4))

    greedy_a = model.generate(prompt, 8, temperature=0.0)
    greedy_b = model.generate(prompt, 8, temperature=0.0)

    torch.manual_seed(1)
    sampled_a = model.generate(prompt, 8, temperature=1.0)
    torch.manual_seed(2)
    sampled_b = model.generate(prompt, 8, temperature=1.0)

    deterministic = torch.equal(greedy_a, greedy_b)
    stochastic = not torch.equal(sampled_a, sampled_b)
    length = greedy_a.shape[1] == prompt.shape[1] + 8
    prompt_kept = torch.equal(greedy_a[:, :4], prompt)

    return report(
        "generation",
        deterministic and stochastic and length and prompt_kept,
        f"greedy deterministic: {deterministic}; sampling varies: {stochastic}; "
        f"length {prompt.shape[1]} -> {greedy_a.shape[1]}",
    )


def test_block_size_is_enforced():
    """Exceeding block_size raises, rather than silently indexing garbage."""

    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 1))

    try:
        model(idx)
        raised = False
        message = "no exception"
    except ValueError as error:
        raised = True
        message = str(error).split("\n")[0][:60]

    return report("block_size is enforced", raised, message)


# =============================================================================
# ENTRY POINT
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Test the Transformer implementation.")
    parser.add_argument(
        "--config",
        nargs="*",
        default=[
            str(REPO / "telugu" / "configs" / "model_h.json"),
            str(REPO / "assamese" / "configs" / "model_l.json"),
        ],
        help="Model configs to count parameters for.",
    )
    parser.add_argument(
        "--expect-params",
        type=int,
        default=24_562_944,
        help="Assert this exact trainable parameter count for each config.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("TRANSFORMER IMPLEMENTATION TESTS")
    print("=" * 78)
    print(f"  torch {torch.__version__}, device cpu\n")

    results = [
        test_shapes(),
        test_attention_is_a_distribution(),
        test_causality(),
        test_no_position_leak_through_padding(),
        test_block_size_is_enforced(),
        test_weight_tying(),
        test_ablation_removes_positions(),
        test_generation(),
        test_overfit_single_batch(),
    ]

    print("\n  PARAMETER COUNTS")
    for config_path in args.config:
        results.append(test_parameter_count(config_path, args.expect_params))

    passed = sum(results)
    print("\n" + "=" * 78)
    print(f"  {passed}/{len(results)} passed")
    print("=" * 78)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
