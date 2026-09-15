"""
Phase 3 on Kaggle: reasoning finetune and evaluation for both models.

Pushed as a SCRIPT kernel with `kaggle kernels push`, so it runs as a
committed version from start to finish. That is the lesson Phase 2 paid
for: an interactive session that ends takes /kaggle/working with it, and
only a committed run leaves its output behind.

Inputs (private datasets):
    lma-phase3        one zip: common/, telugu/configs/, assamese/configs/,
                      work/<lang>/tokenizer/final/*.model,
                      work/<lang>/reasoning/{train,validation,test}.jsonl
    lma-checkpoints   model_h/best.pt, model_l/best.pt from Phase 2

Stages:
    1. check the GPUs and locate both datasets
    2. unpack the code; point LMA_WORK at it
    3. finetune Model H on cuda:0 and Model L on cuda:1, concurrently
    4. evaluate each model pretrained (zero-shot and 3-shot) and finetuned
    5. drop the redundant latest.pt copies to keep the output small

The two models are separate processes on separate GPUs and never share
anything but the machine, exactly as in Phase 2's launch_both.py.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

# ---- 1. environment --------------------------------------------------------

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "| GPUs", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
assert torch.cuda.is_available(), "No GPU attached to this kernel."

INPUT = Path("/kaggle/input")


def find_dataset(name, root=INPUT, max_depth=4):
    """
    Locate a mounted dataset by name. This account mounts under
    /kaggle/input/datasets/<user>/<slug>, not the documented
    /kaggle/input/<slug>, so the name is searched for rather than assumed.
    """

    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth) + ("/" if depth else "") + name
        for candidate in root.glob(pattern):
            if candidate.is_dir():
                return candidate

    return None


CODE = find_dataset("lma-phase3")
CKPT = find_dataset("lma-checkpoints")
print("lma-phase3      :", CODE)
print("lma-checkpoints :", CKPT)
assert CODE and CKPT, f"datasets missing; present: {[str(p) for p in INPUT.rglob('*')][:30]}"

# ---- 2. code ----------------------------------------------------------------

REPO = Path("/kaggle/working/repo")
REPO.mkdir(parents=True, exist_ok=True)

archive = next(CODE.rglob("*.zip"), None)
if archive:
    with zipfile.ZipFile(archive) as z:
        z.extractall(REPO)
else:
    shutil.copytree(CODE, REPO, dirs_exist_ok=True)

os.environ["LMA_WORK"] = str(REPO / "work")
os.environ["PYTHONUNBUFFERED"] = "1"

MODELS = {
    "telugu":   {"tag": "model_h", "config": "telugu/configs/finetune_h.json", "gpu": "0"},
    "assamese": {"tag": "model_l", "config": "assamese/configs/finetune_l.json", "gpu": "1"},
}

for lang, m in MODELS.items():
    m["init"] = next(CKPT.rglob(f"{m['tag']}/best.pt"))
    m["runs"] = REPO / "work" / lang / "runs" / f"{m['tag']}_reasoning"
    m["report"] = Path("/kaggle/working/report/phase3") / lang
    for needed in (REPO / m["config"],
                   REPO / "work" / lang / "tokenizer" / "final" / f"{lang}_final_8000.model",
                   REPO / "work" / lang / "reasoning" / "train.jsonl"):
        assert needed.exists(), f"missing {needed}"
    print(f"{lang:<9} init {m['init']}")

if torch.cuda.device_count() < 2:
    MODELS["assamese"]["gpu"] = "0"


def launch(lang, args, log_name):
    """Start one command for one language on its GPU, logging to a file."""

    m = MODELS[lang]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=m["gpu"])
    log = open(m["runs"].parent / log_name, "w", encoding="utf-8")

    return subprocess.Popen([sys.executable, "-X", "utf8", *args], cwd=REPO, env=env,
                            stdout=log, stderr=subprocess.STDOUT), log


def run_parallel(stage, commands):
    """Run one command per language concurrently; fail loudly on any error."""

    procs = []
    for lang, (args, log_name) in commands.items():
        MODELS[lang]["runs"].mkdir(parents=True, exist_ok=True)
        procs.append((lang, log_name, *launch(lang, args, log_name)))

    failed = []
    for lang, log_name, proc, log in procs:
        code = proc.wait()
        log.close()
        text = (MODELS[lang]["runs"].parent / log_name).read_text(encoding="utf-8")
        print(f"\n===== {stage} / {lang} (exit {code}) =====")
        print(text[-6000:])
        if code:
            failed.append(lang)

    assert not failed, f"{stage} failed for {failed}"


# ---- 3. finetune --------------------------------------------------------------

run_parallel("finetune", {
    lang: (["common/finetune.py", "--config", m["config"], "--init", str(m["init"]),
            "--out", str(m["runs"]), "--resume", "auto"],
           f"{m['tag']}_finetune_console.log")
    for lang, m in MODELS.items()
})

for lang, m in MODELS.items():
    shutil.copy(m["runs"].parent / f"{m['tag']}_finetune_console.log", m["runs"] / "console.log")

# ---- 4. evaluate --------------------------------------------------------------
#
# Pretrained and finetuned run back to back on each language's GPU, the two
# languages concurrently.

eval_commands = {}
for lang, m in MODELS.items():
    m["report"].mkdir(parents=True, exist_ok=True)
    script = (
        f"common/reasoning_eval.py --checkpoint {m['init']} --label pretrained "
        f"--fewshot 3 --fewshot-limit 600 --out {m['report']} && "
        f"common/reasoning_eval.py --checkpoint {m['runs'] / 'best.pt'} --label finetuned "
        f"--out {m['report']}"
    )
    eval_commands[lang] = script

procs = []
for lang, script in eval_commands.items():
    m = MODELS[lang]
    first, second = script.split(" && ")
    cmd = (f"{sys.executable} -X utf8 {first} && {sys.executable} -X utf8 {second}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=m["gpu"])
    log_path = m["report"] / "reasoning_eval_console.log"
    log = open(log_path, "w", encoding="utf-8")
    procs.append((lang, log_path, log,
                  subprocess.Popen(cmd, shell=True, cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)))

failed = []
for lang, log_path, log, proc in procs:
    code = proc.wait()
    log.close()
    print(f"\n===== evaluate / {lang} (exit {code}) =====")
    print(log_path.read_text(encoding="utf-8")[-5000:])
    if code:
        failed.append(lang)
assert not failed, f"evaluation failed for {failed}"

# ---- 5. tidy ------------------------------------------------------------------
#
# latest.pt duplicates final.pt once a run has finished. best.pt and
# final.pt are both kept, in the full resume-capable format.

for lang, m in MODELS.items():
    (m["runs"] / "latest.pt").unlink(missing_ok=True)
    shutil.copytree(m["runs"], Path("/kaggle/working/checkpoints") / lang, dirs_exist_ok=True)

shutil.rmtree(REPO / "work", ignore_errors=True)
print("\ndone. outputs:")
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size / 1e6:9.1f} MB  {p}")
