"""
Figures for the Phase 2 report: loss curves and the H-versus-L comparison.

    python common/plots.py --runs <run dir> <run dir> --out report/phase2

Every plot carries a title, an x label, a y label, and a legend where more
than one series is drawn. The assignment states that plots missing those
may receive zero marks for that component, so the labelling here is part
of the deliverable rather than presentation.

Computation and drawing are separate, as the code-design guidelines ask:
the load_* functions return numbers and know nothing about matplotlib, the
plot_* functions take numbers and know nothing about training runs.

Where the data comes from
-------------------------
train.py appends one JSON object per log interval to train_log.jsonl,
tagged "train", "eval" or "done". That file is the record of a run -- the
console output scrolls away, and on Kaggle it is gone when the session
ends. Curves are rebuilt from the JSONL, so a run that was interrupted and
resumed three times still plots as one continuous history.

A resumed run writes into the same log, so steps can repeat where a
checkpoint was rolled back to. The loaders keep the LAST record for any
repeated step, which is the one that actually contributed to the final
weights.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from language_config import use_utf8_output

use_utf8_output()

REPO = Path(__file__).resolve().parents[1]

# One colour per model, used consistently across every figure so the two
# are never confused between plots.
COLOURS = {"H": "#3b6ea5", "L": "#b5533c"}


# =============================================================================
# COMPUTATION
# =============================================================================


def load_log(path):
    """
    Read one train_log.jsonl into per-record-type lists.

    Later records win for a repeated step: a run resumed from a checkpoint
    replays the steps between the checkpoint and where it was killed, and
    the second pass is the one whose weights survived.
    """

    train, evaluation, done = {}, {}, None

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")

            if kind == "train":
                train[record["step"]] = record
            elif kind == "eval":
                evaluation[record["step"]] = record
            elif kind == "done":
                done = record

    return {
        "train": [train[s] for s in sorted(train)],
        "eval": [evaluation[s] for s in sorted(evaluation)],
        "done": done,
    }


def series(records, key):
    """Two aligned arrays: step, and the requested field."""

    steps = np.array([r["step"] for r in records], dtype=float)
    values = np.array([r[key] for r in records], dtype=float)

    return steps, values


def smooth(values, window):
    """
    Centred moving average, for the training-loss curve only.

    The raw per-interval training loss is noisy enough at 20-step intervals
    that the trend is hard to read. The raw series is always drawn
    underneath at low opacity so the smoothing cannot hide anything.
    """

    if window <= 1 or values.size < window:
        return values

    kernel = np.ones(window) / window

    return np.convolve(values, kernel, mode="same")


def discover_runs(roots):
    """
    Find (tag, name, log path) for every run directory given.

    Accepts either a run directory containing train_log.jsonl, or a parent
    holding several of them, since launch_both.py writes model_h/ and
    model_l/ side by side under one root.
    """

    found = []

    for root in roots:
        root = Path(root)
        candidates = (
            [root] if (root / "train_log.jsonl").exists()
            else sorted(p.parent for p in root.rglob("train_log.jsonl"))
        )

        for directory in candidates:
            name = directory.name
            tag = "H" if name.lower().endswith("_h") or "model_h" in name.lower() \
                else "L" if name.lower().endswith("_l") or "model_l" in name.lower() \
                else name
            found.append((tag, name, directory / "train_log.jsonl"))

    return found


# =============================================================================
# VISUALISATION
# =============================================================================


def add_legend(axis, **kwargs):
    """
    Draw a legend only when something is labelled.

    matplotlib warns when legend() finds no labelled artists, which happens
    for a run whose log holds no records of the plotted type. The figure is
    still valid -- it just has nothing to label -- and a warning printed
    into a report run is noise.
    """

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(**kwargs)


def plot_loss_curves(runs, path, smooth_window=9):
    """Training and validation loss against step, both models on one axis."""

    figure, axis = plt.subplots(figsize=(9, 5))

    for tag, name, log in runs:
        colour = COLOURS.get(tag, None)

        steps, loss = series(log["train"], "loss")
        axis.plot(steps, loss, color=colour, alpha=0.18, linewidth=0.8)
        axis.plot(steps, smooth(loss, smooth_window), color=colour, linewidth=1.6,
                  label=f"Model {tag} - training loss")

        if log["eval"]:
            val_steps, val_loss = series(log["eval"], "val_loss")
            axis.plot(val_steps, val_loss, "o--", color=colour, markersize=4,
                      linewidth=1.2, label=f"Model {tag} - validation loss")

    axis.set_title("Pretraining loss: Model H (Telugu) and Model L (Assamese)",
                   fontsize=12)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss (nats per token)")
    axis.grid(alpha=0.3)
    add_legend(axis, loc="upper right", fontsize=9)

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_loss_vs_tokens(runs, path):
    """
    Validation loss against tokens seen, not steps.

    The two models have different corpus sizes -- 648M against 560M -- so
    they run for different numbers of steps. Plotting against steps makes
    the shorter run look like it stopped early; plotting against tokens
    puts both on the axis that actually describes how much data each model
    consumed, which is the comparison the report is making.
    """

    figure, axis = plt.subplots(figsize=(9, 5))

    for tag, name, log in runs:
        if not log["eval"]:
            continue
        tokens = np.array([r["tokens"] for r in log["eval"]], dtype=float) / 1e6
        val_loss = np.array([r["val_loss"] for r in log["eval"]], dtype=float)
        axis.plot(tokens, val_loss, "o-", color=COLOURS.get(tag), markersize=4,
                  label=f"Model {tag}")

    axis.set_title("Validation loss against training tokens consumed", fontsize=12)
    axis.set_xlabel("Training tokens seen (millions)")
    axis.set_ylabel("Validation cross-entropy (nats per token)")
    axis.grid(alpha=0.3)
    add_legend(axis, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_perplexity(runs, path):
    """Validation perplexity on a log scale, which is how PPL should be read."""

    figure, axis = plt.subplots(figsize=(9, 5))

    for tag, name, log in runs:
        if not log["eval"]:
            continue
        steps, ppl = series(log["eval"], "val_perplexity")
        axis.plot(steps, ppl, "o-", color=COLOURS.get(tag), markersize=4,
                  label=f"Model {tag}")

    axis.set_yscale("log")
    axis.set_title("Validation perplexity during pretraining", fontsize=12)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation perplexity (log scale)")
    axis.grid(alpha=0.3, which="both")
    add_legend(axis, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_learning_rate(runs, path):
    """The schedule as it actually ran, warmup and cosine decay visible."""

    figure, axis = plt.subplots(figsize=(9, 4))

    for tag, name, log in runs:
        steps, lr = series(log["train"], "lr")
        axis.plot(steps, lr, color=COLOURS.get(tag), linewidth=1.5,
                  label=f"Model {tag}")

    axis.set_title("Learning-rate schedule: linear warmup, then cosine decay",
                   fontsize=12)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Learning rate")
    axis.grid(alpha=0.3)
    add_legend(axis, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_throughput(runs, path):
    """
    Tokens per second over the run.

    Included because it is the evidence behind the compute-budget claims in
    the report, and because a sag in it is how a Kaggle session that has
    started thermally throttling announces itself.
    """

    figure, axis = plt.subplots(figsize=(9, 4))

    for tag, name, log in runs:
        records = [r for r in log["train"] if "tokens_per_sec" in r]
        if not records:
            continue
        steps, rate = series(records, "tokens_per_sec")
        axis.plot(steps, rate / 1e3, color=COLOURS.get(tag), alpha=0.35,
                  linewidth=0.8)
        axis.plot(steps, smooth(rate, 9) / 1e3, color=COLOURS.get(tag),
                  linewidth=1.6, label=f"Model {tag}")

    axis.set_title("Training throughput", fontsize=12)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Thousands of tokens per second")
    axis.grid(alpha=0.3)
    add_legend(axis, loc="lower right")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# ORCHESTRATION
# =============================================================================


def summarise(runs):
    """Final numbers per run, for the report's headline table."""

    rows = []

    for tag, name, log in runs:
        final = log["done"]
        best = min((r["val_loss"] for r in log["eval"]), default=None)
        rows.append({
            "model": tag,
            "run": name,
            "steps": log["train"][-1]["step"] if log["train"] else 0,
            "tokens": log["train"][-1]["tokens"] if log["train"] else 0,
            "final_val_loss": final["val_loss"] if final else None,
            "best_val_loss": best,
            "best_val_perplexity": (
                round(math.exp(min(best, 20)), 3) if best is not None else None
            ),
            "seconds": final["seconds"] if final else None,
            "mean_tokens_per_sec": (
                round(float(np.mean([r["tokens_per_sec"] for r in log["train"]
                                     if "tokens_per_sec" in r])), 1)
                if log["train"] else None
            ),
        })

    return rows


def print_summary(rows):
    print(f"\n  {'model':<8}{'steps':>10}{'tokens':>16}{'val loss':>11}"
          f"{'val ppl':>11}{'tok/s':>11}")
    print("  " + "-" * 67)

    for row in rows:
        print(f"  {row['model']:<8}{row['steps']:>10,}{row['tokens']:>16,}"
              f"{row['best_val_loss'] or float('nan'):>11.4f}"
              f"{row['best_val_perplexity'] or float('nan'):>11.2f}"
              f"{row['mean_tokens_per_sec'] or float('nan'):>11.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Loss curves and training figures for the Phase 2 report.",
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories, or a parent containing them.")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: report/phase2/figures")

    return parser.parse_args()


def main():
    args = parse_args()

    found = discover_runs(args.runs)
    if not found:
        raise SystemExit(f"No train_log.jsonl found under {args.runs}")

    runs = [(tag, name, load_log(path)) for tag, name, path in found]

    out_dir = Path(args.out) if args.out else (REPO / "report" / "phase2")
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAINING FIGURES")
    print("=" * 78)
    for tag, name, log in runs:
        print(f"  Model {tag:<3} {name:<24} "
              f"{len(log['train']):>5} train records, "
              f"{len(log['eval']):>4} evals")

    plot_loss_curves(runs, figures / "pretrain_loss_curves.png")
    plot_loss_vs_tokens(runs, figures / "pretrain_loss_vs_tokens.png")
    plot_perplexity(runs, figures / "pretrain_perplexity.png")
    plot_learning_rate(runs, figures / "pretrain_learning_rate.png")
    plot_throughput(runs, figures / "pretrain_throughput.png")

    rows = summarise(runs)
    (out_dir / "pretrain_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print_summary(rows)

    print(f"\n  wrote 5 figures to {figures}")
    print("=" * 78)


if __name__ == "__main__":
    main()
