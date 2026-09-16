"""
Reasoning finetuning for one pretrained model.

    python common/finetune.py --config telugu/configs/finetune_h.json
    python common/finetune.py --config telugu/configs/finetune_h.json --resume auto

Protocol, as the assignment sets it
-----------------------------------
- Start from that language's OWN pretrained checkpoint. The config names
  it, and the script refuses a checkpoint whose model_config.language is
  not the config's language -- initialising Model L from Model H's weights
  is the one mistake that would silently invalidate the whole comparison.
- Keep that language's tokenizer and vocabulary fixed. The tokenizer is
  loaded from the Phase 1 location and its vocabulary size is checked
  against the checkpoint; no pieces are added and the embedding is not
  resized.
- Document all hyperparameters: FinetuneConfig below, serialised into
  every checkpoint and echoed at the top of the console log.
- Save checkpoints in the same resume-capable format as pretraining:
  model, optimizer, scheduler, scaler, sampler position, step and RNG
  states, written through the same temp-file-and-rename path.

What changes from pretraining, and why
--------------------------------------
LEARNING RATE. 1e-4 peak rather than 6e-4. The weights already encode
the language; the finetune should adapt them to a narrow format without
overwriting that. A pretraining-scale rate on 24,000 short templated
examples moves a 25M model far enough that it forgets general text, and
the Phase 3 attention comparison would then be measuring damage rather
than specialisation.

LOSS. Answer pieces and EOS only; see reasoning_task.py.

BATCHING. Examples, not fixed windows. Each example is one right-padded
row, because an example is a unit with a question and an answer, and
packing several into a 512-token window would let attention read the
previous example's answer.

SELECTION. The best checkpoint is chosen on VALIDATION exact match, not
validation loss. Loss rewards putting probability on the right answer;
exact match rewards the right answer being the argmax, and only the second
is what the test set measures. Validation names are disjoint from both
train and test, so selecting on it does not leak test information.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curriculum import CurriculumSampler
from data import PermutationSampler
from evaluate import load_checkpoint
from language_config import get_language, use_utf8_output
from reasoning_data import load_lexicon
from reasoning_task import (collate, encode_examples, load_examples,
                            score_example, tokenizer_for)
from train import build_optimizer, format_duration, make_lr_lambda
from workspace import Workspace

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FinetuneConfig:
    """Every hyperparameter of one finetuning run."""

    name: str = "finetune"
    language: str = "telugu"

    # The pretrained checkpoint this run starts from.
    init_checkpoint: str = None

    # Directory holding train/validation/test.jsonl from reasoning_data.py.
    data_dir: str = None
    out_dir: str = None

    batch_size: int = 32
    grad_accum: int = 1
    epochs: float = 2.0
    max_steps: int = None           # overrides epochs when set

    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    log_every: int = 20
    eval_every: int = 250
    eval_examples: int = 300        # validation examples scored per eval
    checkpoint_every: int = 250

    seed: int = 42
    dtype: str = "fp16"

    # Run 3: train on a worked answer ("<rationale> <therefore> <answer>")
    # instead of the bare answer. See reasoning_data.py for the rationales.
    use_rationale: bool = False

    # Run 5: sample tasks under a curriculum that moves from easy to hard
    # instead of uniformly. See common/curriculum.py.
    curriculum: bool = False
    easy_tasks: tuple = ("numeric_pair", "numeric_relation", "numeric_extreme")
    hard_tasks: tuple = ("chain_pair", "chain_extreme", "chain_yesno")
    curriculum_easy_start: float = 3.0
    curriculum_easy_end: float = 1.0
    curriculum_hard_start: float = 1.0
    curriculum_hard_end: float = 3.0

    # Run 4: use only the first N training examples. The whole 120,000 at
    # run 3's schedule cost the model its language: test perplexity went
    # from 13.8 to 6,709 (Model H) and 61.9 to 868,303 (Model L). Fewer
    # examples and fewer steps are one of the two levers against that.
    train_limit: int = None

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {k: v for k, v in raw.items() if not k.startswith("_")}

        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")

        return cls(**fields)

    def to_dict(self):
        return asdict(self)


# =============================================================================
# FINETUNER
# =============================================================================


class Finetuner:
    """Owns one finetuning run: data, model, optimizer, checkpoints, logs."""

    def __init__(self, config, device=None, init_checkpoint=None, out_dir=None,
                 data_dir=None):
        self.config = config
        self.language = get_language(config.language)
        self.workspace = Workspace(self.language)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.init_checkpoint = Path(init_checkpoint or config.init_checkpoint)
        self.data_dir = Path(data_dir or config.data_dir or self.workspace.root / "reasoning")
        self.out_dir = Path(
            out_dir or config.out_dir or self.workspace.root / "runs" / config.name
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "finetune_log.jsonl"

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._build_model()
        self._build_data()
        self._build_optimizer()

        self.step = 0
        self.examples_seen = 0
        self.best_val_accuracy = -1.0

    # ---- construction -------------------------------------------------------

    def _build_model(self):
        self.model, self.model_config, init_state = load_checkpoint(
            self.init_checkpoint, self.device
        )

        if self.model_config.language.lower() != self.workspace.slug:
            raise ValueError(
                f"{self.init_checkpoint} is a {self.model_config.language} "
                f"model, but this run finetunes {self.workspace.slug}. Each "
                f"model must start from its own language's checkpoint."
            )

        self.init_step = init_state.get("step")
        self.pretrain_manifest = init_state.get("data_manifest")

        # The finetune keeps pretraining's dropout: the dataset is small and
        # templated, which is exactly where a model overfits.
        self.model.train()

        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                          "fp32": torch.float32}[self.config.dtype]
        self.use_amp = self.device.type == "cuda" and self.config.dtype != "fp32"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16
        )

    def _build_data(self):
        self.tokenizer = tokenizer_for(self.workspace.slug, self.model_config.vocab_size)

        if self.tokenizer.get_piece_size() != self.model_config.vocab_size:
            raise ValueError(
                f"Tokenizer has {self.tokenizer.get_piece_size()} pieces, "
                f"checkpoint expects {self.model_config.vocab_size}."
            )

        self.therefore = (load_lexicon(self.language)["therefore"]
                          if self.config.use_rationale else None)

        self.train_examples = encode_examples(
            load_examples(self.data_dir / "train.jsonl", limit=self.config.train_limit),
            self.tokenizer, self.therefore
        )
        val = load_examples(self.data_dir / "validation.jsonl")
        self.val_examples = encode_examples(val[: self.config.eval_examples], self.tokenizer,
                                            self.therefore)

        batch = self.config.batch_size * self.config.grad_accum

        if self.config.curriculum:
            # max_steps has to be known before the sampler is built, since
            # the weight schedule is defined over the length of the run.
            self.max_steps = self.config.max_steps or int(
                math.ceil(self.config.epochs * len(self.train_examples) / batch)
            )
            self.sampler = CurriculumSampler(
                tasks=[e["task"] for e in self.train_examples],
                batch_size=batch,
                max_steps=self.max_steps,
                easy_tasks=self.config.easy_tasks,
                hard_tasks=self.config.hard_tasks,
                easy_start=self.config.curriculum_easy_start,
                easy_end=self.config.curriculum_easy_end,
                hard_start=self.config.curriculum_hard_start,
                hard_end=self.config.curriculum_hard_end,
                seed=self.config.seed,
            )
        else:
            self.sampler = PermutationSampler(
                n_windows=len(self.train_examples),
                batch_windows=batch,
                seed=self.config.seed,
            )
            self.max_steps = self.config.max_steps or int(
                math.ceil(self.config.epochs * self.sampler.batches_per_epoch)
            )

    def _build_optimizer(self):
        self.optimizer, self.n_decay, self.n_no_decay = build_optimizer(
            self.model, self.config
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_lr_lambda(self.config, self.max_steps)
        )

    def _autocast(self):
        if not self.use_amp:
            import contextlib
            return contextlib.nullcontext()

        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    # ---- checkpointing ------------------------------------------------------

    def checkpoint_state(self):
        """Same keys as a pretraining checkpoint, plus the finetune lineage."""

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.state_dict(),
            "step": self.step,
            "examples_seen": self.examples_seen,
            "best_val_accuracy": self.best_val_accuracy,
            "max_steps": self.max_steps,
            "model_config": self.model_config.to_dict(),
            "train_config": self.config.to_dict(),
            "stage": "reasoning_finetune",
            "therefore": self.therefore,
            "init_checkpoint": str(self.init_checkpoint),
            "init_step": self.init_step,
            "data_manifest": self.pretrain_manifest,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng": np.random.get_state(),
        }

    def save_checkpoint(self, tag):
        path = self.out_dir / f"{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(self.checkpoint_state(), tmp)
        tmp.replace(path)

        return path

    def load_checkpoint(self, path):
        """Resume this run. Refuses a checkpoint from a different run."""

        state = torch.load(path, map_location=self.device, weights_only=False)

        if state.get("stage") != "reasoning_finetune":
            raise ValueError(
                f"{path} is not a finetuning checkpoint. To START a finetune "
                f"from a pretrained model, set init_checkpoint instead."
            )

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.sampler.load_state_dict(state["sampler"])

        self.step = state["step"]
        self.examples_seen = state["examples_seen"]
        self.best_val_accuracy = state.get("best_val_accuracy", -1.0)

        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda_rng"]])
        if state.get("numpy_rng") is not None:
            np.random.set_state(state["numpy_rng"])

        return state

    def find_resume(self):
        latest = self.out_dir / "latest.pt"

        return latest if latest.exists() else None

    # ---- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self):
        """Validation answer loss, exact match and choice accuracy."""

        self.model.eval()

        losses = []
        for lo in range(0, len(self.val_examples), self.config.batch_size):
            batch = self.val_examples[lo:lo + self.config.batch_size]
            x, y = collate(batch, self.tokenizer.pad_id(), self.device,
                           self.model_config.block_size)
            with self._autocast():
                _, loss, _ = self.model(x, y)
            losses.append(loss.item())

        results = [score_example(self.model, self.tokenizer, e, self.device,
                                 therefore=self.therefore)
                   for e in self.val_examples]

        self.model.train()

        n = max(1, len(results))
        return {
            "val_answer_loss": float(np.mean(losses)),
            "val_exact_match": sum(r["exact_match"] for r in results) / n,
            "val_choice_accuracy": sum(r["choice_correct"] for r in results) / n,
        }

    def log(self, record):
        with open(self.log_path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")

    # ---- the loop -----------------------------------------------------------

    def train(self, resume=None):
        if resume:
            self.load_checkpoint(resume)
            print(f"  resumed from {Path(resume).name} at step {self.step:,}")

        self.print_banner()

        if self.step == 0:
            # Step 0 is the pretrained model on the finetune's own validation
            # set, so the curve starts from a measured baseline.
            metrics = self.evaluate()
            self.log({"type": "eval", "step": 0, **metrics})
            print(f"  step      0  VAL loss {metrics['val_answer_loss']:.4f}  "
                  f"EM {metrics['val_exact_match']:.3f}  "
                  f"choice {metrics['val_choice_accuracy']:.3f}", flush=True)

        started = time.time()
        window_start, window_loss, window_steps, window_tokens = time.time(), 0.0, 0, 0
        batch_size = self.config.batch_size

        while self.step < self.max_steps:
            lr = self.scheduler.get_last_lr()[0]
            self.optimizer.zero_grad(set_to_none=True)
            ids = self.sampler.next_batch()
            step_loss = 0.0

            for micro in range(self.config.grad_accum):
                batch = [self.train_examples[int(i)]
                         for i in ids[micro * batch_size:(micro + 1) * batch_size]]
                x, y = collate(batch, self.tokenizer.pad_id(), self.device,
                               self.model_config.block_size)
                window_tokens += int(x.numel())

                with self._autocast():
                    _, loss, _ = self.model(x, y)
                    loss = loss / self.config.grad_accum

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            self.examples_seen += len(ids)
            window_loss += step_loss
            window_steps += 1

            if self.step % self.config.log_every == 0:
                elapsed = time.time() - window_start
                mean_loss = window_loss / window_steps
                rate = window_steps / max(elapsed, 1e-9)

                self.log({
                    "type": "train", "step": self.step, "lr": lr,
                    "loss": round(mean_loss, 5),
                    "examples": self.examples_seen,
                    "tokens_per_sec": round(window_tokens / max(elapsed, 1e-9), 1),
                    "elapsed": round(time.time() - started, 1),
                })
                print(f"  step {self.step:>6,}/{self.max_steps:,}  loss {mean_loss:6.4f}  "
                      f"lr {lr:.2e}  {window_tokens / max(elapsed, 1e-9):7.0f} tok/s  "
                      f"eta {format_duration((self.max_steps - self.step) / max(rate, 1e-9))}",
                      flush=True)
                window_start, window_loss, window_steps, window_tokens = time.time(), 0.0, 0, 0

            if self.step % self.config.eval_every == 0 or self.step == self.max_steps:
                metrics = self.evaluate()
                is_best = metrics["val_exact_match"] > self.best_val_accuracy
                self.log({"type": "eval", "step": self.step, "best": is_best, **metrics})
                print(f"  {'':>6}  VAL loss {metrics['val_answer_loss']:.4f}  "
                      f"EM {metrics['val_exact_match']:.3f}  "
                      f"choice {metrics['val_choice_accuracy']:.3f}"
                      f"{'   <- best' if is_best else ''}", flush=True)
                if is_best:
                    self.best_val_accuracy = metrics["val_exact_match"]
                    self.save_checkpoint("best")

            if self.step % self.config.checkpoint_every == 0:
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        self.save_checkpoint("latest")

        elapsed = time.time() - started
        self.log({"type": "done", "step": self.step, "examples": self.examples_seen,
                  "best_val_exact_match": self.best_val_accuracy,
                  "seconds": round(elapsed, 1)})
        print(f"\n  finished {self.max_steps:,} steps in {format_duration(elapsed)}")
        print(f"  best validation exact match {self.best_val_accuracy:.3f}")
        print(f"  checkpoints in {self.out_dir}")

    def print_banner(self):
        print("=" * 78)
        print(f"FINETUNE - {self.workspace.title} - {self.config.name}")
        print("=" * 78)
        print(f"  device       : {self.device}")
        print(f"  precision    : {self.config.dtype if self.use_amp else 'fp32'}")
        print(f"  init from    : {self.init_checkpoint} (pretrain step {self.init_step})")
        print(f"  parameters   : {self.model.num_parameters():,}")
        print(f"  tokenizer    : {self.tokenizer.get_piece_size():,} pieces (fixed)")
        print(f"  train        : {len(self.train_examples):,} examples")
        print(f"  validation   : {len(self.val_examples):,} examples scored per eval")
        print(f"  batch        : {self.config.batch_size} x {self.config.grad_accum}")
        if self.config.curriculum:
            easy_first, hard_first = self.sampler.mix(0)
            easy_last, hard_last = self.sampler.mix(self.max_steps - 1)
            print(f"  curriculum   : easy {easy_first:.0%} -> {easy_last:.0%}, "
                  f"hard {hard_first:.0%} -> {hard_last:.0%} of each batch")
        print(f"  steps        : {self.max_steps:,} "
              f"({self.max_steps * self.config.batch_size * self.config.grad_accum / len(self.train_examples):.2f} epochs)")
        print(f"  lr           : {self.config.learning_rate:.1e} -> {self.config.min_lr:.1e}, "
              f"{self.config.warmup_steps} warmup, cosine")
        print(f"  weight decay : {self.config.weight_decay}")
        print(f"  out          : {self.out_dir}")
        print("  config       : " + json.dumps(self.config.to_dict()))
        print("=" * 78, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Finetune one model on reasoning data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'auto'.")
    parser.add_argument("--init", default=None, help="Override init_checkpoint.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads; defaults to torch's choice.")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = REPO / args.config
    config = FinetuneConfig.from_json(config_path)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    finetuner = Finetuner(config, device=args.device, init_checkpoint=args.init,
                          out_dir=args.out, data_dir=args.data)

    resume = args.resume
    if resume == "auto":
        resume = finetuner.find_resume()
        if resume is None:
            print("  --resume auto: no checkpoint found, starting from the pretrained model")

    finetuner.train(resume=resume)


if __name__ == "__main__":
    main()
