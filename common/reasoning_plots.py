"""
Phase 3 figures and summary tables, for both models side by side.

    python common/reasoning_plots.py \
        --runs telugu=<finetune run dir> assamese=<finetune run dir>

Reads each run's finetune_log.jsonl and report/phase3/<language>/
reasoning_{pretrained,finetuned}.json, and writes report/phase3/figures/
plus report/phase3/reasoning_summary.md.

Computation (load_*, *_table) and drawing (plot_*) are separate functions,
as in the Phase 2 plotting code. Colour follows the MODEL throughout --
Model H is always blue and Model L always orange -- so a reader never has
to re-learn the legend between figures; the measurement is told apart by
line style or by the x-axis instead.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reasoning_data import TASKS
from workspace import REPO

# First two slots of a colourblind-validated categorical palette, in order.
COLOURS = {"telugu": "#2a78d6", "assamese": "#eb6834"}
NAMES = {"telugu": "Model H (Telugu)", "assamese": "Model L (Assamese)"}
INK_MUTED = "#898781"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUTED, "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 2, "font.size": 10,
})


# =============================================================================
# COMPUTATION
# =============================================================================


def load_log(run_dir):
    records = [json.loads(line) for line in
               (Path(run_dir) / "finetune_log.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]

    return ([r for r in records if r["type"] == "train"],
            [r for r in records if r["type"] == "eval"])


def load_reports(language):
    folder = REPO / "report" / "phase3" / language

    return {label: json.loads((folder / f"reasoning_{label}.json").read_text(encoding="utf-8"))
            for label in ("pretrained", "finetuned")}


def accuracy_rows(reports):
    """
    The comparison table for one model: rows are tasks plus overall.

    Columns: chance, pretrained zero-shot choice, pretrained 3-shot EM and
    choice, finetuned EM and choice. Pretrained zero-shot EM is omitted from
    the headline because it is ~0 by construction (see reasoning_eval.py).
    """

    pre, post = reports["pretrained"], reports["finetuned"]
    few = pre.get("few_shot_3")

    def cell(summary, task, metric):
        if summary is None:
            return None
        block = summary["overall"] if task == "overall" else summary["by_task"].get(task)
        return None if block is None else block[metric]

    rows = []
    for task in list(TASKS) + ["overall"]:
        chance = (pre["zero_shot"]["chance_overall"] if task == "overall"
                  else pre["zero_shot"]["chance_by_task"][task])
        rows.append({
            "task": task,
            "chance": chance,
            "pre_em": cell(pre["zero_shot"], task, "exact_match"),
            "pre_choice": cell(pre["zero_shot"], task, "choice_correct"),
            "few_em": cell(few, task, "exact_match"),
            "few_choice": cell(few, task, "choice_correct"),
            "ft_em": cell(post["zero_shot"], task, "exact_match"),
            "ft_choice": cell(post["zero_shot"], task, "choice_correct"),
        })

    return rows


# =============================================================================
# VISUALISATION
# =============================================================================


def plot_loss(logs, path):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for language, (train, _) in logs.items():
        axis.plot([r["step"] for r in train], [r["loss"] for r in train],
                  color=COLOURS[language], label=NAMES[language])

    axis.set_yscale("log")
    axis.set_title("Reasoning finetune: training loss on answer tokens")
    axis.set_xlabel("Optimizer step (32 examples per step)")
    axis.set_ylabel("Cross-entropy on answer + EOS (nats, log scale)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_validation(logs, path):
    """Validation exact match (solid) and choice accuracy (dashed) per model."""

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for language, (_, evals) in logs.items():
        steps = [r["step"] for r in evals]
        axis.plot(steps, [r["val_exact_match"] for r in evals], "o-",
                  color=COLOURS[language], label=f"{NAMES[language]}, exact match")
        axis.plot(steps, [r["val_choice_accuracy"] for r in evals], "s--",
                  color=COLOURS[language], label=f"{NAMES[language]}, choice accuracy")

    axis.set_ylim(0, 1.02)
    axis.set_title("Reasoning finetune: validation accuracy (held-out names)")
    axis.set_xlabel("Optimizer step (step 0 = pretrained model)")
    axis.set_ylabel("Accuracy on 300 validation examples")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_by_task(tables, path):
    """
    One panel per model: pretrained 3-shot choice vs finetuned exact match
    per task, with chance as a tick.
    """

    figure, axes = plt.subplots(1, len(tables), figsize=(6.5 * len(tables), 4.8),
                                sharey=True, squeeze=False)

    for axis, (language, rows) in zip(axes[0], tables.items()):
        rows = [r for r in rows if r["task"] != "overall"]
        x = np.arange(len(rows))
        width = 0.38
        colour = COLOURS[language]

        axis.bar(x - width / 2 - 0.01, [r["few_choice"] or 0 for r in rows], width,
                 color=colour, alpha=0.35, label="Pretrained, 3-shot choice accuracy")
        axis.bar(x + width / 2 + 0.01, [r["ft_em"] for r in rows], width,
                 color=colour, label="Finetuned, exact match")
        axis.scatter(x, [r["chance"] for r in rows], marker="_", s=900,
                     color="#0b0b0b", zorder=3, label="Chance")

        axis.set_xticks(x)
        axis.set_xticklabels([r["task"].replace("_", "\n") for r in rows], fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(NAMES[language])
        axis.set_xlabel("Task")
        axis.set_ylabel("Test accuracy")
        axis.legend(fontsize=8, loc="upper left")

    figure.suptitle("Synthetic reasoning test set: pretrained vs finetuned, by task")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_by_hops(reports_by_language, path):
    """Finetuned exact match against inference hops, both models on one axis."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for language, reports in reports_by_language.items():
        by_hops = reports["finetuned"]["zero_shot"]["by_hops"]
        hops = sorted(by_hops, key=int)
        axis.plot([int(h) for h in hops], [by_hops[h]["exact_match"] for h in hops], "o-",
                  color=COLOURS[language], label=NAMES[language])
        for h in hops:
            axis.annotate(f"n={by_hops[h]['n']}", (int(h), by_hops[h]["exact_match"]),
                          textcoords="offset points", xytext=(0, 8), ha="center",
                          fontsize=7, color=INK_MUTED)

    axis.set_xticks([1, 2, 3])
    axis.set_ylim(0, 1.05)
    axis.set_title("Finetuned exact match by number of inference hops")
    axis.set_xlabel("Hops (1 = values stated; 2-3 = chained comparisons)")
    axis.set_ylabel("Test exact match")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# =============================================================================
# TABLES
# =============================================================================


def fmt(value):
    return "–" if value is None else f"{value:.3f}"


def summary_markdown(tables, reports_by_language):
    lines = ["# Phase 3 reasoning results", "",
             "Generated by `common/reasoning_plots.py`. Test split, 2,400 examples",
             "per language (3-shot columns: first 600).", ""]

    for language, rows in tables.items():
        lines += [f"## {NAMES[language]}", "",
                  "| task | chance | pretrained choice | pretrained 3-shot EM | "
                  "pretrained 3-shot choice | **finetuned EM** | finetuned choice |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            name = f"**{r['task']}**" if r["task"] == "overall" else r["task"]
            lines.append(f"| {name} | {fmt(r['chance'])} | {fmt(r['pre_choice'])} | "
                         f"{fmt(r['few_em'])} | {fmt(r['few_choice'])} | "
                         f"**{fmt(r['ft_em'])}** | {fmt(r['ft_choice'])} |")
        lines.append("")

        post = reports_by_language[language]["finetuned"]["zero_shot"]
        lines += ["| hops | n | finetuned EM |", "|---|---|---|"]
        for h, row in sorted(post["by_hops"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {h} | {row['n']} | {fmt(row['exact_match'])} |")
        lines += ["", "| attribute | n | finetuned EM |", "|---|---|---|"]
        for a, row in post["by_attribute"].items():
            lines.append(f"| {a} | {row['n']} | {fmt(row['exact_match'])} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 figures and tables.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="language=run_dir pairs, e.g. telugu=C:/lma_work/...")
    args = parser.parse_args()

    runs = dict(item.split("=", 1) for item in args.runs)
    out = REPO / "report" / "phase3"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    logs = {language: load_log(run) for language, run in runs.items()}
    reports = {language: load_reports(language) for language in runs}
    tables = {language: accuracy_rows(r) for language, r in reports.items()}

    plot_loss(logs, figures / "finetune_loss.png")
    plot_validation(logs, figures / "finetune_validation_accuracy.png")
    plot_by_task(tables, figures / "reasoning_accuracy_by_task.png")
    plot_by_hops(reports, figures / "reasoning_accuracy_by_hops.png")

    (out / "reasoning_summary.md").write_text(summary_markdown(tables, reports), encoding="utf-8")
    print(f"wrote {figures} and {out / 'reasoning_summary.md'}")


if __name__ == "__main__":
    main()
