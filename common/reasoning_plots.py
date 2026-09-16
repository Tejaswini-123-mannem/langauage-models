"""
Phase 3 figures and summary tables, for both models side by side.

    python common/reasoning_plots.py \
        --runs telugu=<finetune run dir> assamese=<finetune run dir>

Reads each run's finetune_log.jsonl and report/phase3/<language>/
reasoning_{pretrained,finetuned_worked}.json, and writes report/phase3/figures/
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

    # *_seen are the test_seen evaluations (training names, unseen prompts);
    # the unsuffixed pair is the held-out-name test set.
    #
    # The finetuned model reported is run 5, the one kept: 8,000 examples
    # for 6,000 steps at 2e-5 under an easy-to-hard task curriculum. The heavier runs are kept under
    # report/phase3/<language>/run*/ as evidence and are not read here --
    # run 3 in particular scored far higher on reasoning while destroying the
    # language model, which is what language_model_table below records.
    files = {"pretrained": "reasoning_pretrained.json",
             "pretrained_seen": "reasoning_pretrained_seen.json",
             "finetuned": "reasoning_finetuned_curriculum.json",
             "finetuned_seen": "reasoning_finetuned_curriculum_seen.json"}

    return {label: json.loads((folder / name).read_text(encoding="utf-8"))
            for label, name in files.items() if (folder / name).exists()}


def accuracy_rows(reports):
    """
    The comparison table for one model: rows are tasks plus overall.

    Columns: chance, pretrained zero-shot choice, pretrained 3-shot EM and
    choice, finetuned EM and choice. Pretrained zero-shot EM is omitted from
    the headline because it is ~0 by construction (see reasoning_eval.py).
    """

    pre, post = reports["pretrained"], reports["finetuned"]
    few = pre.get("few_shot_3")
    pre_seen = reports.get("pretrained_seen", {}).get("zero_shot")
    post_seen = reports.get("finetuned_seen", {}).get("zero_shot")

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
            "seen_pre_choice": cell(pre_seen, task, "choice_correct"),
            "seen_ft_em": cell(post_seen, task, "exact_match"),
            "seen_ft_choice": cell(post_seen, task, "choice_correct"),
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
    axis.set_xlabel("Optimizer step (64 examples per step)")
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
    axis.set_ylabel("Accuracy on the validation sample")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_by_task(tables, path):
    """
    One panel per model, per task: pretrained 3-shot choice accuracy,
    finetuned exact match on held-out names, and finetuned exact match on
    seen names (test_seen), with chance as a tick.
    """

    figure, axes = plt.subplots(1, len(tables), figsize=(7 * len(tables), 4.8),
                                sharey=True, squeeze=False)

    for axis, (language, rows) in zip(axes[0], tables.items()):
        rows = [r for r in rows if r["task"] != "overall"]
        x = np.arange(len(rows))
        width = 0.27
        colour = COLOURS[language]

        axis.bar(x - width - 0.02, [r["few_choice"] or 0 for r in rows], width,
                 color=colour, alpha=0.25, label="Pretrained, 3-shot choice accuracy")
        axis.bar(x, [r["ft_em"] for r in rows], width,
                 color=colour, alpha=0.6, label="Finetuned EM, held-out names")
        axis.bar(x + width + 0.02, [r["seen_ft_em"] or 0 for r in rows], width,
                 color=colour, label="Finetuned EM, seen names (unseen prompts)")
        axis.scatter(x, [r["chance"] for r in rows], marker="_", s=900,
                     color="#0b0b0b", zorder=3, label="Chance")


        # A bar at exactly 0 draws nothing, and several tasks sit there on
        # held-out names. Label them so the reader sees a measured zero
        # rather than a missing bar.
        for position, value in zip(x, [r["ft_em"] for r in rows]):
            if value == 0:
                axis.text(position, 0.012, "0", ha="center", va="bottom",
                          fontsize=7, color=INK_MUTED)
        axis.set_xticks(x)
        axis.set_xticklabels([r["task"].replace("_", "\n") for r in rows], fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(NAMES[language])
        axis.set_xlabel("Task")
        axis.set_ylabel("Test accuracy")

    # One legend for both panels, below them, so it cannot sit over a bar.
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Synthetic reasoning test set: pretrained vs finetuned, by task")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_by_hops(reports_by_language, path):
    """
    Finetuned exact match against inference hops, both models: solid on
    seen names (test_seen), dashed on held-out names (test).
    """

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for language, reports in reports_by_language.items():
        for label, style, suffix in (("finetuned_seen", "o-", "seen names"),
                                     ("finetuned", "s--", "held-out names")):
            if label not in reports:
                continue
            by_hops = reports[label]["zero_shot"]["by_hops"]
            hops = sorted(by_hops, key=int)
            axis.plot([int(h) for h in hops], [by_hops[h]["exact_match"] for h in hops], style,
                      color=COLOURS[language], label=f"{NAMES[language]}, {suffix}")

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


def language_model_rows(language):
    """
    Perplexity on the Phase 2 test text, before and after finetuning.

    The check a reasoning finetune most needs and most often skips: a model
    that answers comparison questions but can no longer predict ordinary
    text has been broken, not taught. Written by common/evaluate.py into
    lm_eval_{pretrained,finetuned}.json, with the heavier runs' numbers in
    their own subdirectories.
    """

    folder = REPO / "report" / "phase3" / language
    rows = []
    for label, path in (("pretrained", folder / "lm_eval_pretrained.json"),
                        ("finetuned (kept: run 5 curriculum)", folder / "lm_eval_finetuned.json"),
                        ("run 4 small", folder / "run4_small" / "lm_eval_finetuned.json"),
                        ("run 4 gentle", folder / "run4_gentle" / "lm_eval_finetuned.json"),
                        ("run 3 (8,000 steps at 3e-4)", folder / "run3_overfit" / "lm_eval_finetuned.json")):
        if path.exists():
            overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
            rows.append((label, overall["perplexity"], overall["bits_per_byte"]))

    return rows


def summary_markdown(tables, reports_by_language):
    lines = ["# Phase 3 reasoning results", "",
             "Generated by `common/reasoning_plots.py`. Test split, 2,400 examples",
             "per language (3-shot columns: first 600).", ""]

    for language, rows in tables.items():
        lines += [f"## {NAMES[language]}", "",
                  "Held-out names (`test`):", "",
                  "| task | chance | pretrained choice | pretrained 3-shot EM | "
                  "pretrained 3-shot choice | finetuned EM | finetuned choice |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            name = f"**{r['task']}**" if r["task"] == "overall" else r["task"]
            lines.append(f"| {name} | {fmt(r['chance'])} | {fmt(r['pre_choice'])} | "
                         f"{fmt(r['few_em'])} | {fmt(r['few_choice'])} | "
                         f"{fmt(r['ft_em'])} | {fmt(r['ft_choice'])} |")
        lines += ["", "Seen names, unseen prompts (`test_seen`):", "",
                  "| task | chance | pretrained choice | finetuned EM | finetuned choice |",
                  "|---|---|---|---|---|"]
        for r in rows:
            name = f"**{r['task']}**" if r["task"] == "overall" else r["task"]
            lines.append(f"| {name} | {fmt(r['chance'])} | {fmt(r['seen_pre_choice'])} | "
                         f"{fmt(r['seen_ft_em'])} | {fmt(r['seen_ft_choice'])} |")
        lines.append("")

        reports = reports_by_language[language]
        post = reports["finetuned"]["zero_shot"]
        seen = reports.get("finetuned_seen", {}).get("zero_shot")

        def seen_cell(group, key):
            return fmt(seen[group][key]["exact_match"]) if seen and key in seen[group] else "–"

        lines += ["| hops | n | finetuned EM, held-out | finetuned EM, seen |", "|---|---|---|---|"]
        for h, row in sorted(post["by_hops"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {h} | {row['n']} | {fmt(row['exact_match'])} | {seen_cell('by_hops', h)} |")
        lines += ["", "| attribute | n | finetuned EM, held-out | finetuned EM, seen |",
                  "|---|---|---|---|"]
        for a, row in post["by_attribute"].items():
            lines.append(f"| {a} | {row['n']} | {fmt(row['exact_match'])} | {seen_cell('by_attribute', a)} |")

        lm = language_model_rows(language)
        if lm:
            lines += ["", "Language modelling on the Phase 2 test split "
                          "(600 windows, 307,200 tokens):", "",
                      "| checkpoint | perplexity | bits/byte |", "|---|---|---|"]
            for label, perplexity, bits in lm:
                lines.append(f"| {label} | {perplexity:,.2f} | {bits:.4f} |")
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
