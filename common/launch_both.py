"""
Run Model H and Model L at the same time, one per GPU.

Why this exists
---------------
Kaggle offers a T4 x2 accelerator, and the two pretraining runs are
completely independent -- separate corpora, separate tokenizers, separate
weights, and by design no communication between them. So they are not a
distributed job; they are two single-GPU jobs that happen to fit on one
machine. Running them concurrently halves the wall-clock for the phase and
costs the same GPU quota, because Kaggle bills the session, not the card.

This is NOT torch.distributed. There is no gradient synchronisation and
there must not be: the assignment forbids sharing anything between the two
models, and a DDP wrapper would average their gradients into one model.
Each process is given exactly one GPU through CUDA_VISIBLE_DEVICES and
never learns the other exists.

Usage
-----
    python common/launch_both.py \
        --data-root /kaggle/input/lma-tokens \
        --out-root  /kaggle/working/runs \
        --resume auto

Output from both runs is interleaved on one stream, so every line is
prefixed with its model tag. Each run also writes its own train_log.jsonl,
which is what the loss curves are built from -- the console is for
watching, the JSONL is the record.

Single-GPU fallback
-------------------
With one GPU (or none) the runs are executed one after the other rather
than refusing to start, so the same command works on a T4 x1 session, on
Colab, and on a CPU for a smoke test.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import use_utf8_output

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]

# (tag, training config, language). The tags are what prefix the console
# lines and name the output subdirectories.
RUNS = [
    ("H", "telugu/configs/train_h.json", "telugu"),
    ("L", "assamese/configs/train_l.json", "assamese"),
]


def stream_output(process, tag, sink):
    """Forward one child's stdout to ours, prefixed and flushed per line."""

    for line in iter(process.stdout.readline, ""):
        text = f"[{tag}] {line.rstrip()}"
        print(text, flush=True)
        sink.write(text + "\n")
        sink.flush()

    process.stdout.close()


def build_command(config, out_dir, data_dir, resume, extra):
    """The argv for one train.py invocation."""

    command = [
        sys.executable, "-u", str(REPO / "common" / "train.py"),
        "--config", str(REPO / config),
        "--out", str(out_dir),
        "--data", str(data_dir),
    ]
    if resume:
        command += ["--resume", resume]

    return command + list(extra)


def resolve_data_dir(root, language):
    """
    Find the directory holding this language's train.bin.

    Kaggle nests uploaded datasets differently depending on whether they
    were pushed as a directory tree, as a zip, or through the web uploader,
    and the local workspace has its own layout again. Rather than assume
    one, the candidates are tried in order and the first that actually
    contains train.bin wins. Guessing wrong here fails four lines into a
    session that took ten minutes to start.
    """

    root = Path(root)
    candidates = [
        root / language / "tokens",     # local workspace layout
        root / language,                # dataset uploaded per language
        root / "tokens" / language,
        root / language.upper(),
        root,                           # dataset already at the language level
    ]

    for candidate in candidates:
        if (candidate / "train.bin").exists():
            return candidate

    # Last resort: search, so a deeply nested Kaggle unpack still works.
    for found in root.rglob("train.bin"):
        if language in str(found).lower():
            return found.parent

    raise FileNotFoundError(
        f"No train.bin for {language} under {root}. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def launch(tag, config, language, args, gpu, extra):
    """Start one training process pinned to one GPU."""

    out_dir = Path(args.out_root) / f"model_{tag.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_root, language)

    environment = os.environ.copy()
    if gpu is not None:
        # The child sees exactly one device, numbered 0 from its point of
        # view, so train.py needs no notion of which card it is on.
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

    command = build_command(config, out_dir, data_dir, args.resume, extra)

    print(f"[{tag}] GPU {gpu if gpu is not None else 'default'}  "
          f"data {data_dir}  out {out_dir}", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    log_path = out_dir / "console.log"
    sink = open(log_path, "a", encoding="utf-8")
    thread = threading.Thread(
        target=stream_output, args=(process, tag, sink), daemon=True
    )
    thread.start()

    return process, thread, sink


def count_gpus():
    """Number of visible CUDA devices, without importing torch here."""

    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Model H and Model L concurrently, one per GPU.",
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Directory holding <language>/train.bin, e.g. /kaggle/input/lma-tokens",
    )
    parser.add_argument(
        "--out-root", required=True,
        help="Where checkpoints go, e.g. /kaggle/working/runs",
    )
    parser.add_argument(
        "--resume", default="auto",
        help="Passed through to train.py. 'auto' continues the newest checkpoint.",
    )
    parser.add_argument(
        "--only", choices=["H", "L"], default=None,
        help="Run just one of the two models.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Force one after the other even if two GPUs are present.",
    )

    return parser.parse_known_args()


def main():
    args, extra = parse_args()

    runs = [r for r in RUNS if args.only is None or r[0] == args.only]
    n_gpus = count_gpus()

    print("=" * 78)
    print("PRETRAIN BOTH MODELS")
    print("=" * 78)
    print(f"  visible GPUs : {n_gpus}")
    print(f"  runs         : {', '.join(tag for tag, _, _ in runs)}")
    print(f"  data root    : {args.data_root}")
    print(f"  out root     : {args.out_root}")
    print(f"  resume       : {args.resume}")
    if extra:
        print(f"  passthrough  : {' '.join(extra)}")

    concurrent = n_gpus >= len(runs) and not args.sequential and len(runs) > 1
    print(f"  mode         : {'concurrent, one model per GPU' if concurrent else 'sequential'}")
    print("=" * 78, flush=True)

    started = time.time()
    failures = []

    if concurrent:
        live = [
            launch(tag, config, language, args, gpu, extra)
            for gpu, (tag, config, language) in enumerate(runs)
        ]
        for (process, thread, sink), (tag, _, _) in zip(live, runs):
            code = process.wait()
            thread.join(timeout=10)
            sink.close()
            if code != 0:
                failures.append((tag, code))
    else:
        for tag, config, language in runs:
            gpu = 0 if n_gpus else None
            process, thread, sink = launch(tag, config, language, args, gpu, extra)
            code = process.wait()
            thread.join(timeout=10)
            sink.close()
            if code != 0:
                failures.append((tag, code))

    elapsed = time.time() - started
    print("\n" + "=" * 78)
    print(f"  wall clock: {int(elapsed) // 3600}:{(int(elapsed) % 3600) // 60:02d}:"
          f"{int(elapsed) % 60:02d}")

    if failures:
        for tag, code in failures:
            print(f"  MODEL {tag} FAILED with exit code {code}")
        print("=" * 78)

        return 1

    print(f"  both runs completed; checkpoints under {args.out_root}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
