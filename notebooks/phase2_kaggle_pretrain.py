"""
Kaggle notebook source: pretrain Model H and Model L on T4 x2.

This is the .py source of the Kaggle notebook, kept in the repository so
the run is reproducible from the branch rather than only from a Kaggle
account. Paste the cells below into a Kaggle notebook, or use
`kaggle kernels push`.

SESSION SETUP (do this in the Kaggle UI before running anything)
----------------------------------------------------------------
    Settings -> Accelerator          : GPU T4 x2
    Settings -> Internet             : off (nothing here downloads)
    Add Data -> Your Datasets        : lma-code
                                       lma-telugu-tokens
                                       lma-assamese-tokens

Internet is deliberately OFF. Every input is a mounted dataset, and a
notebook with no network cannot silently pip-install a package that
changes behaviour between runs. It also removes the possibility of
accidentally replacing Kaggle's CUDA build of torch with a CPU wheel,
which fails by training slowly rather than by erroring.

HOW A RUN THAT OUTLIVES ITS SESSION WORKS
-----------------------------------------
Kaggle sessions end at 12 hours, and can end sooner. The pattern is:

    1. Run this notebook with "Save & Run All (Commit)". It executes on
       Kaggle's servers with the browser closed.
    2. When it finishes -- or is killed -- /kaggle/working is saved as the
       notebook's OUTPUT.
    3. To continue: Add Data -> Notebook Output -> this notebook's previous
       version, then set RESUME_FROM below to that mounted path.

Checkpoints carry model, optimizer, scheduler, step, sampler position and
RNG state, so step 4,001 after a resume is the step 4,001 the
uninterrupted run would have produced. common/test_train_resume.py proves
this bit-for-bit.
"""

# %% [markdown]
# # Phase 2 pretraining: Model H (Telugu) and Model L (Assamese)
#
# Two independent ~24.6M-parameter decoder-only Transformers, one per GPU,
# trained simultaneously. Separate corpora, tokenizers, vocabularies and
# weights -- nothing is shared between them, and this notebook runs them as
# two separate processes rather than one distributed job.

# %% [code]
# ---- Cell 1: verify the environment before spending any quota -------------
#
# Every failure this catches is one that would otherwise appear several
# minutes into a run, or worse, not appear at all: a CPU-only torch trains
# without complaining, just fifty times slower.

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  cuda:{i}  {name}  {total:.1f} GB")

assert torch.cuda.is_available(), (
    "No GPU. Settings -> Accelerator -> GPU T4 x2. If the option is greyed "
    "out, phone verification is not complete."
)

# T4 is Turing: fp16 tensor cores, no bf16. The configs specify fp16 with a
# GradScaler for exactly this reason.
if torch.cuda.is_available():
    print("bf16 supported:", torch.cuda.is_bf16_supported())

INPUT = Path("/kaggle/input")
print("\nmounted datasets:")
for entry in sorted(INPUT.glob("*")):
    print(" ", entry.name)

# %% [code]
# ---- Cell 2: unpack the code and point the workspace at it ----------------
#
# lma-code holds one zip: common/, telugu/configs/, assamese/configs/, and
# work/<lang>/tokenizer/final/. Kaggle sometimes auto-extracts a zip in a
# dataset and sometimes does not, so handle both.

REPO = Path("/kaggle/working/repo")
REPO.mkdir(parents=True, exist_ok=True)

code_zip = next(INPUT.glob("lma-code/*.zip"), None)
if code_zip is not None:
    with zipfile.ZipFile(code_zip) as archive:
        archive.extractall(REPO)
    print("extracted", code_zip.name)
else:
    subprocess.run(["cp", "-r", str(INPUT / "lma-code" / "."), str(REPO)], check=True)
    print("copied pre-extracted code")

# LMA_WORK relocates Workspace's root, which is where the evaluation
# scripts find each language's tokenizer. One variable instead of a
# --tokenizer flag on every script.
os.environ["LMA_WORK"] = str(REPO / "work")

print("\nrepo contents:")
for entry in sorted(REPO.rglob("*"))[:40]:
    if entry.is_file():
        print(" ", entry.relative_to(REPO))

# %% [code]
# ---- Cell 3: locate the token bins and check they arrived intact ----------
#
# A dataset upload that was interrupted leaves a short .bin. Training on it
# works and reports a perfectly ordinary loss curve, so the manifest's
# declared token count is checked against the file on disk here rather than
# discovered later. TokenDataset raises on a mismatch, but confirming it
# now costs a second.

def resolve_tokens(language):
    """
    Find a language's .bin files, whether the dataset holds them loose or
    as a zip.

    The two datasets are packaged differently and it is worth knowing why.
    Telugu uploaded as eight loose files. Assamese failed twice at the same
    file with a misreported authentication error, and went up as a single
    zip instead -- one file is one upload call, which removes whatever the
    eighth call was hitting. It also transfers less: 1.13 GB of token ids
    compresses to 0.73 GB, because an 8,000-entry vocabulary leaves the top
    bits of every uint16 zero.

    A zip is extracted to /kaggle/working, since /kaggle/input is read-only.
    """

    root = next(INPUT.glob(f"lma-{language}-tokens"), None)
    assert root is not None, f"Dataset lma-{language}-tokens is not mounted."

    if (root / "train.bin").exists():
        return root

    archive = next(root.glob("*.zip"), None)
    assert archive is not None, (
        f"{root} holds neither train.bin nor a zip: {list(root.iterdir())}"
    )

    target = Path("/kaggle/working/tokens") / language
    if not (target / "train.bin").exists():
        target.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {archive.name} ...", flush=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)

    return target


DATA = {}
for language in ("telugu", "assamese"):
    root = resolve_tokens(language)
    DATA[language] = root

    print(f"\n{language}: {root}")
    for split in ("train", "validation", "test"):
        binary = root / f"{split}.bin"
        manifest = json.loads((root / f"{split}.json").read_text())
        on_disk = binary.stat().st_size // 2          # uint16
        status = "OK" if on_disk == manifest["tokens"] else "MISMATCH"
        print(f"  {split:<11}{manifest['tokens']:>14,} tokens  "
              f"{binary.stat().st_size / 1e9:>6.2f} GB  [{status}]")
        assert on_disk == manifest["tokens"], (
            f"{binary} holds {on_disk:,} tokens, manifest says "
            f"{manifest['tokens']:,}. The upload is incomplete; re-upload "
            f"the dataset."
        )

# %% [code]
# ---- Cell 4: a 30-step smoke run, and a resume, before the real thing -----
#
# This is the cheapest insurance available. It measures actual throughput
# on this session's GPUs -- which is what the time estimate should be based
# on, not arithmetic -- and it proves resume works HERE, on Kaggle, before
# hours depend on it.
#
# Set SMOKE = False once it has passed to skip straight to the real run.

SMOKE = True
SMOKE_DIR = Path("/kaggle/working/smoke")

if SMOKE:
    subprocess.run(
        [sys.executable, "-u", str(REPO / "common" / "launch_both.py"),
         "--data-root", str(INPUT),
         "--out-root", str(SMOKE_DIR),
         "--resume", "",
         "--max-steps", "30"],
        check=True, env={**os.environ},
    )

    # Resume the smoke run for 30 more steps: if this continues rather than
    # restarting, the mechanism works on this machine.
    subprocess.run(
        [sys.executable, "-u", str(REPO / "common" / "launch_both.py"),
         "--data-root", str(INPUT),
         "--out-root", str(SMOKE_DIR),
         "--resume", "auto",
         "--max-steps", "60"],
        check=True, env={**os.environ},
    )
    print("\nSmoke run complete. Read the tok/s figure above: at ~60k tok/s "
          "Model H's 10,550 steps take about 3 hours.")

# %% [code]
# ---- Cell 5: the real run -------------------------------------------------
#
# One pass over each corpus: Model H 10,550 steps over 648M tokens, Model L
# 9,110 steps over 560M. Both run concurrently, one per GPU.
#
# RESUME_FROM: leave None for a fresh start. To continue an earlier session,
# add that notebook version's output under Add Data and put its path here --
# the checkpoints are copied into /kaggle/working first, because
# /kaggle/input is read-only and the trainer writes new checkpoints beside
# the one it resumed from.

RESUME_FROM = None          # e.g. Path("/kaggle/input/pretrain-run-v1/runs")
RUNS = Path("/kaggle/working/runs")
RUNS.mkdir(parents=True, exist_ok=True)

if RESUME_FROM is not None:
    subprocess.run(["cp", "-r", f"{RESUME_FROM}/.", str(RUNS)], check=True)
    print("seeded from previous session:")
    for checkpoint in sorted(RUNS.rglob("*.pt")):
        print(" ", checkpoint.relative_to(RUNS),
              f"{checkpoint.stat().st_size / 1e6:.0f} MB")

subprocess.run(
    [sys.executable, "-u", str(REPO / "common" / "launch_both.py"),
     "--data-root", str(INPUT),
     "--out-root", str(RUNS),
     "--resume", "auto"],
    check=True, env={**os.environ},
)

# %% [code]
# ---- Cell 6: evaluate both models -----------------------------------------
#
# Runs against the test split, which neither model has seen and which the
# tokenizer was never trained on either.

REPORT = Path("/kaggle/working/report")

for tag, language in (("h", "telugu"), ("l", "assamese")):
    checkpoint = RUNS / f"model_{tag}" / "best.pt"
    if not checkpoint.exists():
        print(f"skipping model_{tag}: no checkpoint yet")
        continue

    out = REPORT / language
    for script, extra in (
        ("evaluate.py", ["--split", "test"]),
        ("generate.py", ["--split", "test", "--n-samples", "200"]),
        ("attention.py", ["--split", "test", "--n-sequences", "8"]),
    ):
        print(f"\n=== {language}: {script} ===", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(REPO / "common" / script),
             "--checkpoint", str(checkpoint),
             "--data", str(DATA[language]),
             "--out", str(out)] + extra,
            check=True, env={**os.environ},
        )

# Loss curves from both runs.
subprocess.run(
    [sys.executable, "-u", str(REPO / "common" / "plots.py"),
     "--runs", str(RUNS), "--out", str(REPORT)],
    check=True, env={**os.environ},
)

# %% [code]
# ---- Cell 7: package the outputs -----------------------------------------
#
# /kaggle/working is saved as the notebook's output, but browsing it file by
# file is tedious and the checkpoints are what have to reach Google Drive
# for the README links. Two archives: one small enough to download quickly
# and commit to the branch, one holding the weights.

import shutil

shutil.make_archive("/kaggle/working/phase2_report", "zip", REPORT)
print("report bundle:",
      Path("/kaggle/working/phase2_report.zip").stat().st_size / 1e6, "MB")

for tag in ("h", "l"):
    source = RUNS / f"model_{tag}"
    if source.exists():
        # best.pt and final.pt only. The step_*.pt rolling checkpoints are
        # for resume within a session and are not worth the download.
        keep = Path(f"/kaggle/working/ckpt_model_{tag}")
        keep.mkdir(exist_ok=True)
        for name in ("best.pt", "final.pt", "train_log.jsonl", "console.log"):
            if (source / name).exists():
                shutil.copy2(source / name, keep / name)
        shutil.make_archive(f"/kaggle/working/checkpoints_model_{tag}", "zip", keep)
        print(f"model {tag.upper()} checkpoints:",
              Path(f"/kaggle/working/checkpoints_model_{tag}.zip")
              .stat().st_size / 1e6, "MB")

print("\nDownload these from the notebook's Output tab:")
print("  phase2_report.zip        -> unzip into report/phase2/ on the branch")
print("  checkpoints_model_h.zip  -> Google Drive, link in the README")
print("  checkpoints_model_l.zip  -> Google Drive, link in the README")
