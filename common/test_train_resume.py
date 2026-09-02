"""
Proof that an interrupted pretraining run resumes exactly.

Why this test exists
--------------------
The assignment makes checkpointing mandatory and says a run that cannot
resume after interruption will lose marks. It is also the practical
constraint on this project: training happens on Kaggle, whose sessions end
at a hard 12 hour limit and can end sooner, against a 30 hour weekly GPU
quota that must also cover Phase 3 finetuning.

"It resumes" is easy to believe and easy to get wrong. A run that reloads
weights and optimizer state but redraws its data order looks completely
healthy -- the step counter continues, the loss curve is smooth, nothing
warns -- while the model trains twice on some windows and never on others,
and the report's claim of one clean pass over the corpus is false.

So the test is not "does it start again". It is:

    train 12 steps straight through            -> weights A
    train 6 steps, throw the trainer away,
    build a new one, resume, train 6 more      -> weights B

    assert A == B, bit for bit

Bitwise equality is the right bar. Anything less would pass with a
plausible-looking but different run, which is exactly the failure being
tested for. Passing requires the checkpoint to carry the sampler position,
the scheduler position, the optimizer moments AND the RNG states -- dropout
is deliberately left on at 0.1 here, so a resumed run that restored
everything except the torch generator would draw different masks and
diverge in the last decimal places.

Self-contained: it writes its own synthetic corpus into a temporary
directory, so it can be run without the 1.3 GB token bins present.

    python common/test_train_resume.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import use_utf8_output
from train import TrainConfig, Trainer

use_utf8_output()

VOCAB = 8000
BLOCK = 64
TRAIN_TOKENS = 200_000
VAL_TOKENS = 20_000
TOTAL_STEPS = 12
INTERRUPT_AT = 6


# =============================================================================
# FIXTURE
# =============================================================================


def write_corpus(directory):
    """
    Write a synthetic train/validation ribbon in the real .bin format.

    Random token ids, not real text: the test is about the mechanics of
    resuming, and a model that cannot learn anything from noise still has
    to reproduce its own trajectory exactly.
    """

    rng = np.random.default_rng(0)

    for split, n in (("train", TRAIN_TOKENS), ("validation", VAL_TOKENS)):
        tokens = rng.integers(0, VOCAB, size=n, dtype=np.uint16)
        (directory / f"{split}.bin").write_bytes(tokens.tobytes())
        (directory / f"{split}.json").write_text(
            json.dumps({
                "split": split, "tokens": int(n), "vocab_size": VOCAB,
                "dtype": "uint16", "eos_id": 2,
                "tokenizer_sha256": "synthetic-fixture",
            }),
            encoding="utf-8",
        )


def write_configs(directory):
    """A deliberately tiny model, so the whole test runs on a CPU in seconds."""

    model_path = directory / "model_tiny.json"
    model_path.write_text(json.dumps({
        "name": "tiny", "language": "telugu",
        "vocab_size": VOCAB, "block_size": BLOCK,
        "n_layer": 2, "n_head": 4, "d_model": 64, "d_ff": 256,
        # Dropout stays ON. It is what makes this a test of RNG restoration
        # and not only of weight restoration.
        "dropout": 0.1,
        "bias": True, "tie_weights": True, "use_positional_embedding": True,
    }), encoding="utf-8")

    return model_path


def make_config(model_path, name):
    return TrainConfig(
        name=name,
        language="telugu",
        model_config=str(model_path),
        micro_batch=4,
        grad_accum=2,
        max_steps=TOTAL_STEPS,
        warmup_steps=3,
        learning_rate=1e-3,
        min_lr=1e-4,
        log_every=1000,            # quiet; the test asserts, it does not narrate
        eval_every=1000,
        eval_batches=2,
        checkpoint_every=INTERRUPT_AT,
        seed=42,
    )


# =============================================================================
# TEST
# =============================================================================


def compare_state_dicts(a, b):
    """Return (n_tensors, n_identical, largest absolute difference)."""

    worst = 0.0
    identical = 0

    for key in a:
        x, y = a[key].float(), b[key].float()
        if torch.equal(a[key], b[key]):
            identical += 1
        worst = max(worst, (x - y).abs().max().item())

    return len(a), identical, worst


def run(root):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    write_corpus(data_dir)
    model_path = write_configs(root)

    print("=" * 78)
    print("CHECKPOINT RESUME TEST")
    print("=" * 78)
    print(f"  synthetic corpus : {TRAIN_TOKENS:,} train tokens, block {BLOCK}")
    print(f"  schedule         : {TOTAL_STEPS} steps, interrupt after {INTERRUPT_AT}")
    print(f"  dropout          : 0.1 (so RNG state is exercised)\n")

    # ---- A: straight through ------------------------------------------------
    print(f"  [A] training {TOTAL_STEPS} steps uninterrupted ...")
    trainer_a = Trainer(
        make_config(model_path, "uninterrupted"),
        out_dir=root / "run_a", data_dir=data_dir, device="cpu",
    )
    trainer_a.print_banner = lambda: None
    trainer_a.train()
    state_a = torch.load(root / "run_a" / "final.pt", weights_only=False)

    # ---- B: interrupted and resumed -----------------------------------------
    #
    # A fresh Trainer, exactly as a new Kaggle session would build one: it
    # knows nothing except what is in the checkpoint file.
    print(f"  [B] training {INTERRUPT_AT} steps, then resuming in a new trainer ...")
    trainer_b = Trainer(
        make_config(model_path, "interrupted"),
        out_dir=root / "run_b", data_dir=data_dir, device="cpu",
    )
    trainer_b.print_banner = lambda: None
    trainer_b.config.max_steps = INTERRUPT_AT
    trainer_b.max_steps = INTERRUPT_AT
    trainer_b.train()

    checkpoint = root / "run_b" / "latest.pt"
    del trainer_b

    trainer_c = Trainer(
        make_config(model_path, "resumed"),
        out_dir=root / "run_c", data_dir=data_dir, device="cpu",
    )
    trainer_c.print_banner = lambda: None
    trainer_c.train(resume=checkpoint)
    state_c = torch.load(root / "run_c" / "final.pt", weights_only=False)

    # ---- compare ------------------------------------------------------------
    print("\n  RESULTS")

    results = []

    n, identical, worst = compare_state_dicts(state_a["model"], state_c["model"])
    weights_match = identical == n
    results.append(weights_match)
    print(f"    [{'PASS' if weights_match else 'FAIL'}] model weights identical"
          f"  -- {identical}/{n} tensors bit-for-bit, max |delta| {worst:.3e}")

    step_match = state_a["step"] == state_c["step"] == TOTAL_STEPS
    results.append(step_match)
    print(f"    [{'PASS' if step_match else 'FAIL'}] step counter"
          f"  -- A {state_a['step']}, resumed {state_c['step']}")

    tokens_match = state_a["tokens_seen"] == state_c["tokens_seen"]
    results.append(tokens_match)
    print(f"    [{'PASS' if tokens_match else 'FAIL'}] tokens seen"
          f"  -- {state_a['tokens_seen']:,} both")

    sampler_match = state_a["sampler"] == state_c["sampler"]
    results.append(sampler_match)
    print(f"    [{'PASS' if sampler_match else 'FAIL'}] data sampler position"
          f"  -- A {state_a['sampler']}, resumed {state_c['sampler']}")

    lr_a = state_a["scheduler"]["_last_lr"]
    lr_c = state_c["scheduler"]["_last_lr"]
    sched_match = lr_a == lr_c
    results.append(sched_match)
    print(f"    [{'PASS' if sched_match else 'FAIL'}] scheduler position"
          f"  -- last lr {lr_a[0]:.6e} both")

    opt_a = state_a["optimizer"]["state"]
    opt_c = state_c["optimizer"]["state"]
    moments_match = all(
        torch.equal(opt_a[k]["exp_avg"], opt_c[k]["exp_avg"])
        and torch.equal(opt_a[k]["exp_avg_sq"], opt_c[k]["exp_avg_sq"])
        for k in opt_a
    )
    results.append(moments_match)
    print(f"    [{'PASS' if moments_match else 'FAIL'}] AdamW moment estimates"
          f"  -- {len(opt_a)} parameter tensors")

    required = {"model", "optimizer", "scheduler", "step", "train_config",
                "model_config", "sampler", "torch_rng"}
    complete = required <= set(state_a)
    results.append(complete)
    print(f"    [{'PASS' if complete else 'FAIL'}] checkpoint completeness"
          f"  -- {sorted(required)}")

    print("\n" + "=" * 78)
    print(f"  {sum(results)}/{len(results)} passed")
    print("=" * 78)

    return 0 if all(results) else 1


def main():
    root = Path(tempfile.mkdtemp(prefix="lma_resume_"))
    try:
        return run(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
