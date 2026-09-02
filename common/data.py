"""
Batching from the tokenized corpus.

What this module is for
-----------------------
build_token_bins.py produced one flat uint16 ribbon per split. This module
turns that ribbon into (x, y) batches, and does two things that are easy
to get subtly wrong.

Why the windows must be shuffled
--------------------------------
The training ribbon is ordered by SOURCE, not interleaved. Measured on the
actual splits:

    telugu    5 contiguous runs: textbook(17,944) wikisource(8,794)
              wikipedia(76,209) archive(819) sangraha(296,234+)
    assamese  4 contiguous runs: wikisource(53,654) wikipedia(23,941)
              archive(2,397) sangraha(312,105)

Telugu's archive run is only 819 documents but 143.9M characters, all
adjacent. Reading the ribbon in order would train the model on a sequence
of distribution shifts -- every textbook, then every Wikisource page, then
819 whole books, then 900,000 web documents -- and the final weights would
be dominated by whichever source happened to be last. Perplexity on a
mixed held-out set would be poor and the reason would not be visible.

So the ribbon is cut into fixed windows and the WINDOW ORDER is permuted.
Each window is still contiguous text, which is what a causal LM needs; it
is only the order in which windows are visited that is randomised.

Why a permutation and not random offsets
----------------------------------------
The common shortcut is to draw a fresh random offset per batch. It is one
line, but it samples with replacement: over a run some windows are seen
three times and others never, and there is no meaningful notion of "one
pass over the corpus", which is exactly what this project claims to do.

A permutation of window indices gives a genuine single pass, and it is
exactly resumable: the sampler's whole state is (seed, epoch, position).
Restoring those three integers reproduces the remaining order precisely,
so a run killed by a Kaggle session limit continues instead of silently
re-training on windows it already saw while reporting a clean epoch.

Memory: 1.27M windows for Telugu is a 5 MB int32 array. Nothing here
needs to be clever.

Why x and y overlap by one token
--------------------------------
A window of block_size+1 tokens gives x = w[:-1] and y = w[1:]. The
causal LM objective is to predict token t+1 from tokens <= t, so the
targets are the inputs shifted one left, and every one of the block_size
positions supplies a training signal. Adjacent windows therefore share a
single token at their boundary, which is correct and not double counting:
that token is an input in one window and a target in the other.
"""

import json
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """
    A memory-mapped uint16 token ribbon, cut into fixed-length windows.

    The file is np.memmap'd rather than read: the training loop touches a
    few hundred kilobytes per batch out of a 1.3 GB file, and the OS pages
    in only what is actually indexed. Loading it would cost 1.3 GB of RAM
    on a Kaggle worker for no benefit.
    """

    def __init__(self, bin_path, block_size, manifest_path=None):
        self.bin_path = Path(bin_path)
        self.block_size = int(block_size)

        if manifest_path is None:
            manifest_path = self.bin_path.with_suffix(".json")
        self.manifest_path = Path(manifest_path)

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Missing {self.bin_path}\n"
                f"Run: python common/build_token_bins.py --language <lang>"
            )

        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {}
        )

        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self.tokens.shape[0])

        # A manifest token count that disagrees with the file means the .bin
        # was truncated -- a killed build, or a partial upload to Kaggle.
        # Training on a truncated ribbon works and reports nothing unusual.
        declared = self.manifest.get("tokens")
        if declared is not None and declared != self.n_tokens:
            raise ValueError(
                f"{self.bin_path} holds {self.n_tokens:,} tokens but its "
                f"manifest declares {declared:,}. The file is truncated or "
                f"the manifest is stale; rebuild with --force."
            )

        # Windows are non-overlapping in their INPUTS, and each needs one
        # extra token for its final target, hence (n - 1) // block_size.
        self.n_windows = (self.n_tokens - 1) // self.block_size
        if self.n_windows < 1:
            raise ValueError(
                f"{self.bin_path} holds {self.n_tokens:,} tokens, too few "
                f"for even one window of {self.block_size}."
            )

    # ---- integrity ----------------------------------------------------------

    def assert_matches_tokenizer(self, tokenizer_sha256):
        """
        Refuse to train against a .bin built with a different tokenizer.

        This is the failure mode with no symptoms: token ids from tokenizer
        A fed to a model whose embedding was sized for tokenizer B still
        train, still produce a falling loss curve, and still report a
        perplexity. The number is meaningless and nothing says so.
        """

        recorded = self.manifest.get("tokenizer_sha256")
        if recorded and tokenizer_sha256 and recorded != tokenizer_sha256:
            raise ValueError(
                f"{self.bin_path} was built with tokenizer {recorded[:16]} "
                f"but the run expects {tokenizer_sha256[:16]}."
            )

    @property
    def vocab_size(self):
        return self.manifest.get("vocab_size")

    # ---- batching -----------------------------------------------------------

    def windows(self, window_ids, device=None):
        """
        Materialise one batch from a list of window indices.

        Returns (x, y), both (len(window_ids), block_size) int64.

        The uint16 slice is copied into an int64 tensor because
        nn.Embedding requires a long index and torch has no uint16 dtype.
        The copy is small -- one batch, not the corpus -- and is what keeps
        the memmap read-only.
        """

        block = self.block_size
        xs = np.empty((len(window_ids), block), dtype=np.int64)
        ys = np.empty((len(window_ids), block), dtype=np.int64)

        for row, window in enumerate(window_ids):
            start = int(window) * block
            chunk = self.tokens[start:start + block + 1]
            xs[row] = chunk[:-1]
            ys[row] = chunk[1:]

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)

        if device is not None:
            # pin_memory + non_blocking overlaps the host-to-device copy
            # with compute. Only meaningful on CUDA.
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)

        return x, y

    def __repr__(self):
        return (
            f"TokenDataset({self.bin_path.name}, {self.n_tokens:,} tokens, "
            f"{self.n_windows:,} windows of {self.block_size})"
        )


class PermutationSampler:
    """
    Visits every window exactly once per epoch, in a seeded random order.

    State is three integers -- seed, epoch, position -- and that is the
    whole of what a checkpoint needs to resume the data stream exactly.
    Saving model and optimizer state but not this is the quiet way to
    corrupt a "single pass over the corpus" claim: the run restarts, the
    order is redrawn, and the model sees some windows twice and others not
    at all while the step counter says everything is fine.
    """

    def __init__(self, n_windows, batch_windows, seed=42, epoch=0, position=0):
        self.n_windows = int(n_windows)
        self.batch_windows = int(batch_windows)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.position = int(position)

        self._order = self._build_order(self.epoch)

    def _build_order(self, epoch):
        """
        Permutation for one epoch.

        Seeded by (seed, epoch) so epoch 1 is a different order from epoch
        0 while both are reproducible from the checkpoint alone.
        """

        rng = np.random.default_rng([self.seed, epoch])

        return rng.permutation(self.n_windows)

    @property
    def batches_per_epoch(self):
        """Complete batches only; a short tail batch is dropped."""

        return self.n_windows // self.batch_windows

    def next_batch(self):
        """Return the next batch_windows window indices, wrapping epochs."""

        if self.position + self.batch_windows > self.n_windows:
            self.epoch += 1
            self.position = 0
            self._order = self._build_order(self.epoch)

        batch = self._order[self.position:self.position + self.batch_windows]
        self.position += self.batch_windows

        return batch

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.position = int(state["position"])

        if int(state["epoch"]) != self.epoch:
            self.epoch = int(state["epoch"])
            self._order = self._build_order(self.epoch)

    def __repr__(self):
        return (
            f"PermutationSampler(epoch {self.epoch}, "
            f"{self.position:,}/{self.n_windows:,} windows, "
            f"{self.batches_per_epoch:,} batches/epoch)"
        )


class FixedEvalSampler:
    """
    A fixed, seeded subset of a held-out split, identical at every eval.

    Validation loss is read as a curve, so the measurement has to be the
    same measurement each time. Drawing fresh random windows per evaluation
    adds sampling noise to the curve that looks like real movement -- the
    model appears to improve or regress between evals when only the sample
    changed.

    Seeded rather than "the first N windows" because the ribbon is ordered
    by source: the first N windows of Telugu validation are all textbook
    OCR, which is neither the hardest nor the most representative text in
    the split.
    """

    def __init__(self, n_windows, batch_windows, n_batches, seed=1234):
        self.batch_windows = int(batch_windows)

        rng = np.random.default_rng(seed)
        wanted = min(int(n_batches) * self.batch_windows, int(n_windows))
        chosen = rng.choice(int(n_windows), size=wanted, replace=False)

        self.batches = [
            chosen[i:i + self.batch_windows]
            for i in range(0, wanted - self.batch_windows + 1, self.batch_windows)
        ]

    @property
    def n_batches(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __repr__(self):
        return (
            f"FixedEvalSampler({self.n_batches} batches "
            f"x {self.batch_windows} windows)"
        )
