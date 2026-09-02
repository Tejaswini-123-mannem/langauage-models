"""
Pretraining loop for one model.

    python common/train.py --config telugu/configs/train_h.json
    python common/train.py --config telugu/configs/train_h.json --resume auto

Both models are trained by this one script with different configs. As in
Phase 1, that is deliberate: Phase 3 has to attribute the Model H versus
Model L difference to the DATA, and identical training code is what makes
that a claim about the corpora rather than about two training loops that
drifted apart.

Checkpoint-resume is the design constraint
------------------------------------------
The assignment makes checkpointing mandatory and says a run that cannot
resume after interruption loses marks. It is also a practical necessity:
this trains on Kaggle, whose sessions end at a hard 12 hour limit and can
end sooner, against a 30 hour weekly GPU quota that must also cover Phase
3. A run that cannot resume is a run that has to start over, and there is
not enough quota for many of those.

So a checkpoint carries five things, and the fifth is the one usually
missed:

    1. model weights
    2. optimizer state       (Adam's two moment estimates per parameter)
    3. scheduler state       (the LR schedule's position)
    4. training step         (and tokens seen)
    5. DATA SAMPLER state    (seed, epoch, position within the permutation)
       plus the torch and numpy RNG states

Without (5) a resumed run redraws its window order. The step counter keeps
climbing, the loss curve looks continuous, and the model quietly trains
twice on some windows and never on others -- while the report claims one
clean pass over the corpus. Nothing downstream reveals it.

Mixed precision on a T4, specifically
-------------------------------------
The T4 is Turing. It has fp16 tensor cores and NO bfloat16 support, so
this uses fp16 with a GradScaler rather than the bf16 that an A100 or
newer notebook would use. bf16 has the same exponent range as fp32 and
needs no scaler; fp16 does not, and gradients below roughly 6e-8 flush to
zero. GradScaler multiplies the loss by a large factor before backward and
divides it out before the optimizer step, moving the gradients into fp16's
representable range. Without it, training on a T4 in fp16 stalls with a
loss that falls briefly and then flattens.

The autocast/scaler path is skipped entirely on CPU, where fp16 is slower
than fp32 rather than faster.
"""

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import FixedEvalSampler, PermutationSampler, TokenDataset
from language_config import get_language, use_utf8_output
from model import GPT, GPTConfig
from workspace import Workspace

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainConfig:
    """
    Everything about one pretraining run except the architecture.

    Serialised into every checkpoint, so a checkpoint file is enough to
    say how it was produced.
    """

    name: str = "run"
    language: str = "telugu"
    model_config: str = "telugu/configs/model_h.json"

    # ---- batch -------------------------------------------------------------
    # micro_batch is what fits in GPU memory at once; grad_accum micro
    # batches are accumulated before one optimizer step. The gradient the
    # optimizer sees is therefore computed over
    #     micro_batch x grad_accum x block_size
    # tokens, which is the number that matters -- memory decides only how
    # it is split up.
    micro_batch: int = 24
    grad_accum: int = 5

    # ---- schedule ----------------------------------------------------------
    # max_steps None means "one pass over the corpus", computed from the
    # ribbon length once the dataset is open.
    max_steps: int = None
    warmup_steps: int = 400
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- cadence -----------------------------------------------------------
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 40
    checkpoint_every: int = 1000

    # ---- housekeeping ------------------------------------------------------
    seed: int = 42
    keep_last: int = 2
    compile_model: bool = False
    dtype: str = "fp16"          # fp16 | bf16 | fp32

    # Set by --out on Kaggle, where checkpoints must land in /kaggle/working.
    out_dir: str = None
    data_dir: str = None

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {k: v for k, v in raw.items() if not k.startswith("_")}

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(fields) - known
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")

        return cls(**fields)

    def to_dict(self):
        return asdict(self)


# =============================================================================
# SCHEDULE
# =============================================================================


def make_lr_lambda(config, max_steps):
    """
    Linear warmup, then cosine decay to min_lr.

    Warmup exists because Adam's second-moment estimate v is an average
    over the steps taken so far, so for the first few hundred steps it is
    an average of almost nothing and the update m/sqrt(v) is wildly
    oversized. Those first updates land on a randomly initialised model and
    can move it somewhere it does not recover from. Ramping the learning
    rate from ~0 keeps them small while v fills in.

    Cosine decay then spends most of the run at a high rate -- exploring --
    and anneals smoothly to min_lr so the final weights settle rather than
    bouncing around a minimum.

    Returned as a multiplier on the base learning rate, for LambdaLR. A
    torch scheduler object is used rather than setting the LR by hand
    because its state_dict is what the assignment means by "scheduler
    state" in a checkpoint.
    """

    warmup = max(1, int(config.warmup_steps))
    floor = config.min_lr / config.learning_rate

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return floor + (1.0 - floor) * cosine

    return lr_lambda


def build_optimizer(model, config):
    """
    AdamW with weight decay on matmul weights only.

    The 'W' is decoupled weight decay. Plain Adam adds the L2 penalty to
    the gradient, so it then passes through the same 1/sqrt(v) division as
    everything else -- parameters with small gradients get decayed HARDER,
    which is backwards. AdamW subtracts lr * wd * w from the weight
    directly, so the decay is the same for every parameter.

    Biases and LayerNorm gains are excluded. They are per-feature offsets
    and scales, not directions in a weight space, and shrinking them
    towards zero is not regularisation -- it just biases the network
    towards zero activations. Embeddings are excluded for a related
    reason: decay applies every step regardless of whether a row was used,
    so a rare token's embedding is pulled towards zero purely for being
    rare.

    beta2 is 0.95 rather than the 0.999 default. At 0.999 the second
    moment averages over ~1000 steps, which is a large fraction of a
    ~10,000 step run; 0.95 adapts faster and is what GPT-style runs of
    this length use.
    """

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # fused AdamW keeps the whole update on the GPU as one kernel instead
    # of a Python loop over parameter tensors. Available on CUDA only.
    fused = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        **({"fused": True} if fused else {}),
    )

    return optimizer, len(decay), len(no_decay)


# =============================================================================
# TRAINER
# =============================================================================


class Trainer:
    """Owns one pretraining run: data, model, optimizer, checkpoints, logs."""

    def __init__(self, config, out_dir=None, data_dir=None, device=None):
        self.config = config
        self.workspace = Workspace(get_language(config.language))

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Kaggle mounts datasets read-only under /kaggle/input and requires
        # output under /kaggle/working, so both roots are overridable.
        self.data_dir = Path(data_dir or config.data_dir or self.workspace.tokens)
        self.out_dir = Path(
            out_dir or config.out_dir or (self.workspace.root / "runs" / config.name)
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._build_data()
        self._build_model()
        self._build_optimizer()

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        self.log_path = self.out_dir / "train_log.jsonl"

    # ---- construction -------------------------------------------------------

    def _build_data(self):
        """Open the train and validation ribbons and their samplers."""

        model_config_path = REPO / self.config.model_config
        self.model_config = GPTConfig.from_json(model_config_path)

        self.train_data = TokenDataset(
            self.data_dir / "train.bin", self.model_config.block_size
        )
        self.val_data = TokenDataset(
            self.data_dir / "validation.bin", self.model_config.block_size
        )

        declared_vocab = self.train_data.vocab_size
        if declared_vocab and declared_vocab != self.model_config.vocab_size:
            raise ValueError(
                f"train.bin was tokenized with a {declared_vocab:,}-piece "
                f"vocabulary but {model_config_path.name} declares "
                f"{self.model_config.vocab_size:,}."
            )

        batch_windows = self.config.micro_batch * self.config.grad_accum

        self.sampler = PermutationSampler(
            n_windows=self.train_data.n_windows,
            batch_windows=batch_windows,
            seed=self.config.seed,
        )
        self.eval_sampler = FixedEvalSampler(
            n_windows=self.val_data.n_windows,
            batch_windows=self.config.micro_batch,
            n_batches=self.config.eval_batches,
        )

        # One pass over the corpus, unless the config says otherwise.
        self.max_steps = self.config.max_steps or self.sampler.batches_per_epoch
        self.tokens_per_step = batch_windows * self.model_config.block_size

    def _build_model(self):
        self.model = GPT(self.model_config).to(self.device)

        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # fp16 needs the loss scaler; bf16 and fp32 do not. A T4 has no
        # bf16 support at all, so fp16 is the only mixed-precision option
        # available on the hardware this actually runs on.
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.config.dtype]

        self.use_amp = self.device.type == "cuda" and self.config.dtype != "fp32"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16
        )

    def _build_optimizer(self):
        self.optimizer, n_decay, n_no_decay = build_optimizer(self.model, self.config)
        self.n_decay, self.n_no_decay = n_decay, n_no_decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_lr_lambda(self.config, self.max_steps)
        )

    def _autocast(self):
        if not self.use_amp:
            return contextlib.nullcontext()

        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    # ---- checkpointing ------------------------------------------------------

    def checkpoint_state(self):
        """
        Everything needed to resume this run bit-for-bit.

        The RNG states matter alongside the sampler position: dropout masks
        are drawn from the torch generator, so restoring the sampler but
        not the generator resumes the right data in a different noise
        stream. Not fatal, but it makes a resumed run non-reproducible, and
        the assignment asks for reproducible experiments.
        """

        model = self.model
        if hasattr(model, "_orig_mod"):        # unwrap torch.compile
            model = model._orig_mod

        return {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "max_steps": self.max_steps,
            "model_config": self.model_config.to_dict(),
            "train_config": self.config.to_dict(),
            "data_manifest": {
                "train": self.train_data.manifest,
                "validation": self.val_data.manifest,
            },
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng": np.random.get_state(),
        }

    def save_checkpoint(self, tag):
        path = self.out_dir / f"{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")

        # Written to a temporary name and renamed. A Kaggle session killed
        # midway through torch.save leaves a truncated file, and a
        # truncated checkpoint that overwrote the last good one is the
        # whole run lost.
        torch.save(self.checkpoint_state(), tmp)
        tmp.replace(path)

        return path

    def load_checkpoint(self, path):
        """Restore a run. Refuses a checkpoint from a different corpus."""

        state = torch.load(path, map_location=self.device, weights_only=False)

        saved_sha = (
            state.get("data_manifest", {}).get("train", {}).get("tokenizer_sha256")
        )
        current_sha = self.train_data.manifest.get("tokenizer_sha256")
        if saved_sha and current_sha and saved_sha != current_sha:
            raise ValueError(
                f"{path} was trained on a corpus tokenized with "
                f"{saved_sha[:16]}, but train.bin here is {current_sha[:16]}."
            )

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(state["model"])

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.sampler.load_state_dict(state["sampler"])

        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in state["cuda_rng"]]
            )
        if state.get("numpy_rng") is not None:
            np.random.set_state(state["numpy_rng"])

        return state

    def find_resume(self):
        """Most recent resumable checkpoint in out_dir, or None."""

        candidates = sorted(
            self.out_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        latest = self.out_dir / "latest.pt"

        if latest.exists():
            return latest

        return candidates[-1] if candidates else None

    def prune_checkpoints(self):
        """Keep only the newest keep_last step checkpoints, plus best/latest."""

        steps = sorted(
            self.out_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for stale in steps[: max(0, len(steps) - self.config.keep_last)]:
            stale.unlink(missing_ok=True)

    # ---- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self):
        """Mean cross-entropy over the fixed validation subset."""

        self.model.eval()
        total, count = 0.0, 0

        for window_ids in self.eval_sampler:
            x, y = self.val_data.windows(window_ids, device=self.device)
            with self._autocast():
                _, loss, _ = self.model(x, y)
            total += loss.item()
            count += 1

        self.model.train()

        return total / max(1, count)

    # ---- logging ------------------------------------------------------------

    def log(self, record):
        """Append one JSON record to the run log, and echo a short line."""

        with open(self.log_path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")

    # ---- the loop -----------------------------------------------------------

    def train(self, resume=None):
        """Run training to max_steps, resuming from `resume` if given."""

        if resume:
            self.load_checkpoint(resume)
            print(f"  resumed from {Path(resume).name} at step {self.step:,} "
                  f"({self.tokens_seen / 1e6:.1f}M tokens seen)")

        self.model.train()
        self.print_banner()

        started = time.time()
        window_start = time.time()
        window_loss, window_steps = 0.0, 0

        while self.step < self.max_steps:
            lr = self.scheduler.get_last_lr()[0]

            # ---- one optimizer step over grad_accum micro-batches ----------
            #
            # The loss of each micro-batch is divided by grad_accum before
            # backward, so the accumulated gradient equals the gradient of
            # the mean loss over the full effective batch. Without the
            # division the effective learning rate would scale with
            # grad_accum.
            self.optimizer.zero_grad(set_to_none=True)
            batch_windows = self.sampler.next_batch()
            step_loss = 0.0

            for micro in range(self.config.grad_accum):
                lo = micro * self.config.micro_batch
                ids = batch_windows[lo:lo + self.config.micro_batch]

                x, y = self.train_data.windows(ids, device=self.device)

                with self._autocast():
                    _, loss, _ = self.model(x, y)
                    loss = loss / self.config.grad_accum

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # Unscale before clipping: the gradients currently carry the
            # scaler's factor, and clipping them at that scale would clip
            # at the wrong threshold entirely.
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            self.tokens_seen += self.tokens_per_step
            window_loss += step_loss
            window_steps += 1

            # ---- logging ----------------------------------------------------
            if self.step % self.config.log_every == 0:
                elapsed = time.time() - window_start
                mean_loss = window_loss / max(1, window_steps)
                throughput = window_steps * self.tokens_per_step / max(elapsed, 1e-9)
                remaining = (self.max_steps - self.step) / max(
                    window_steps / max(elapsed, 1e-9), 1e-9
                )

                self.log({
                    "type": "train", "step": self.step, "lr": lr,
                    "loss": round(mean_loss, 5),
                    "perplexity": round(math.exp(min(mean_loss, 20)), 3),
                    "tokens": self.tokens_seen,
                    "tokens_per_sec": round(throughput, 1),
                    "elapsed": round(time.time() - started, 1),
                })
                print(
                    f"  step {self.step:>6,}/{self.max_steps:,}  "
                    f"loss {mean_loss:6.4f}  ppl {math.exp(min(mean_loss, 20)):8.2f}  "
                    f"lr {lr:.2e}  {throughput / 1e3:6.1f}k tok/s  "
                    f"eta {format_duration(remaining)}",
                    flush=True,
                )

                window_loss, window_steps = 0.0, 0
                window_start = time.time()

            # ---- validation -------------------------------------------------
            if self.step % self.config.eval_every == 0 or self.step == self.max_steps:
                val_loss = self.evaluate()
                is_best = val_loss < self.best_val_loss

                self.log({
                    "type": "eval", "step": self.step,
                    "val_loss": round(val_loss, 5),
                    "val_perplexity": round(math.exp(min(val_loss, 20)), 3),
                    "tokens": self.tokens_seen, "best": is_best,
                })
                print(
                    f"  {'':>6}  VAL loss {val_loss:6.4f}  "
                    f"ppl {math.exp(min(val_loss, 20)):8.2f}"
                    f"{'   <- best' if is_best else ''}",
                    flush=True,
                )

                if is_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

            # ---- checkpoint --------------------------------------------------
            if self.step % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"step_{self.step:07d}")
                self.save_checkpoint("latest")
                self.prune_checkpoints()

        # ---- done -----------------------------------------------------------
        val_loss = self.evaluate()
        self.best_val_loss = min(self.best_val_loss, val_loss)
        self.save_checkpoint("final")
        self.save_checkpoint("latest")

        elapsed = time.time() - started
        print(f"\n  finished {self.max_steps:,} steps in {format_duration(elapsed)}")
        print(f"  final val loss {val_loss:.4f}  "
              f"ppl {math.exp(min(val_loss, 20)):.2f}")
        print(f"  checkpoints in {self.out_dir}")

        self.log({
            "type": "done", "step": self.step, "tokens": self.tokens_seen,
            "val_loss": round(val_loss, 5),
            "best_val_loss": round(self.best_val_loss, 5),
            "seconds": round(elapsed, 1),
        })

        return val_loss

    # ---- reporting ----------------------------------------------------------

    def print_banner(self):
        """Print the run's configuration, so a log file is self-describing."""

        params = self.model.num_parameters() if not hasattr(self.model, "_orig_mod") \
            else self.model._orig_mod.num_parameters()

        print("=" * 78)
        print(f"PRETRAIN - {self.workspace.title} - {self.config.name}")
        print("=" * 78)
        print(f"  device       : {self.device} "
              f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
        # Report what is actually in effect, not what was requested: on CPU
        # autocast is skipped entirely, and a log claiming fp16 for a run
        # that executed in fp32 would misreport its own throughput numbers.
        if self.use_amp:
            precision = self.config.dtype + (
                " + GradScaler" if self.scaler.is_enabled() else " (no scaler needed)"
            )
        else:
            precision = f"fp32 (requested {self.config.dtype}; " \
                        f"mixed precision is CUDA-only)"
        print(f"  precision    : {precision}")
        print(f"  parameters   : {params:,}")
        print(f"  architecture : d_model {self.model_config.d_model}, "
              f"{self.model_config.n_layer}L, {self.model_config.n_head}H, "
              f"block {self.model_config.block_size}")
        print(f"  train data   : {self.train_data.n_tokens:,} tokens, "
              f"{self.train_data.n_windows:,} windows")
        print(f"  batch        : {self.config.micro_batch} x {self.config.grad_accum} "
              f"= {self.tokens_per_step:,} tokens/step")
        print(f"  steps        : {self.max_steps:,} "
              f"({self.max_steps * self.tokens_per_step / 1e6:.1f}M tokens, "
              f"{self.max_steps * self.tokens_per_step / self.train_data.n_tokens:.2f} epochs)")
        print(f"  lr           : {self.config.learning_rate:.2e} -> "
              f"{self.config.min_lr:.2e}, {self.config.warmup_steps} warmup, cosine")
        print(f"  weight decay : {self.config.weight_decay} on {self.n_decay} tensors, "
              f"0.0 on {self.n_no_decay}")
        print(f"  out          : {self.out_dir}")
        print("=" * 78, flush=True)


# =============================================================================
# HELPERS
# =============================================================================


def format_duration(seconds):
    seconds = int(max(0, seconds))

    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain one model.")
    parser.add_argument("--config", required=True, help="Training config JSON.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, or 'auto' to continue the newest in out_dir.",
    )
    parser.add_argument("--out", default=None, help="Override the output directory.")
    parser.add_argument("--data", default=None, help="Directory holding the .bin files.")
    parser.add_argument("--device", default=None, help="cuda, cuda:1, cpu.")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the step count. Useful for a short smoke run.",
    )
    # micro_batch and grad_accum are overridable from the command line
    # because the only way to find a T4's memory ceiling is to hit it. When
    # a run OOMs, halving micro_batch and doubling grad_accum leaves the
    # effective batch -- and therefore the run being reported -- unchanged,
    # and that should not require editing a config file mid-session.
    parser.add_argument(
        "--micro-batch", type=int, default=None,
        help="Override micro_batch. Halve it on OOM and double --grad-accum.",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None,
        help="Override grad_accum.",
    )
    parser.add_argument(
        "--dtype", default=None, choices=["fp16", "bf16", "fp32"],
        help="Override the precision. fp16 for a T4, bf16 for Ampere or newer.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO / args.config

    config = TrainConfig.from_json(config_path)
    for field_name, value in (
        ("max_steps", args.max_steps),
        ("micro_batch", args.micro_batch),
        ("grad_accum", args.grad_accum),
        ("dtype", args.dtype),
    ):
        if value is not None:
            setattr(config, field_name, value)

    trainer = Trainer(
        config, out_dir=args.out, data_dir=args.data, device=args.device
    )

    resume = args.resume
    if resume == "auto":
        resume = trainer.find_resume()
        if resume is None:
            print("  --resume auto: no checkpoint found, starting from scratch")

    trainer.train(resume=resume)


if __name__ == "__main__":
    main()
