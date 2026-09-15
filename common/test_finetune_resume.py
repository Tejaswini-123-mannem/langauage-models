"""
Proof that an interrupted finetune resumes into the run it interrupted.

    python common/test_finetune_resume.py --language assamese

The assignment requires finetuned checkpoints in the same resume-capable
format as pretraining, and common/test_train_resume.py proved that for
pretraining. This is the same test for finetune.py:

    A. train for 2N steps without stopping
    B. train for N steps, stop, build a FRESH Finetuner, resume from
       latest.pt, train N more

and require B's final weights to equal A's exactly. Equality, not
closeness: the checkpoint carries model, optimizer, scheduler, scaler,
sampler position and every RNG state, so a correct resume reproduces the
same dropout masks on the same examples and the weights match bit for bit
on CPU. Any missing piece of state -- the sampler's epoch position is the
usual one -- shows up as a mismatch.

Runs on CPU with a handful of steps and a 4-example validation pass, so it
takes about a minute.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune import Finetuner, FinetuneConfig
from language_config import use_utf8_output
from workspace import REPO

use_utf8_output()

CONFIGS = {"telugu": "telugu/configs/finetune_h.json",
           "assamese": "assamese/configs/finetune_l.json"}


def make_config(language, steps, checkpoint_every):
    config = FinetuneConfig.from_json(REPO / CONFIGS[language])
    config.max_steps = steps
    config.warmup_steps = 2
    config.batch_size = 8
    config.eval_examples = 4
    config.eval_every = 10_000          # only the end-of-run evaluation
    config.log_every = 1
    config.checkpoint_every = checkpoint_every

    return config


def weights(finetuner):
    return {k: v.detach().clone() for k, v in finetuner.model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="assamese", choices=sorted(CONFIGS))
    parser.add_argument("--steps", type=int, default=3, help="N; runs are 2N steps.")
    args = parser.parse_args()

    n = args.steps
    root = Path(tempfile.mkdtemp(prefix="finetune_resume_"))

    try:
        print(f"A: {2 * n} uninterrupted steps")
        a = Finetuner(make_config(args.language, 2 * n, 10_000), device="cpu",
                      out_dir=root / "a")
        a.train()
        final_a = weights(a)

        print(f"\nB1: {n} steps, then stop")
        b1 = Finetuner(make_config(args.language, 2 * n, n), device="cpu",
                       out_dir=root / "b")
        # Stop at step n by running the loop with a lowered ceiling. The
        # schedule was built for 2n steps, as the interrupted run's was.
        b1.max_steps = n
        b1.train()
        del b1

        print(f"\nB2: fresh process state, resume from latest.pt, {n} more steps")
        b2 = Finetuner(make_config(args.language, 2 * n, 10_000), device="cpu",
                       out_dir=root / "b")
        b2.train(resume=root / "b" / "latest.pt")
        final_b = weights(b2)

        mismatched = [k for k in final_a if not torch.equal(final_a[k], final_b[k])]
        if mismatched:
            print(f"\nFAIL: {len(mismatched)} tensors differ, e.g. {mismatched[:3]}")
            sys.exit(1)

        print(f"\nPASS: all {len(final_a)} tensors identical after resume "
              f"at step {n} of {2 * n}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
