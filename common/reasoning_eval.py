"""
Reasoning accuracy for one checkpoint, pretrained or finetuned.

    python common/reasoning_eval.py --checkpoint <best.pt> --label pretrained --fewshot 3
    python common/reasoning_eval.py --checkpoint <finetuned best.pt> --label finetuned

Writes report/phase3/<language>/reasoning_<label>.json and
reasoning_<label>_examples.md.

Three measurements, and which model each is fair to
---------------------------------------------------
exact_match         greedy answer == gold. The real task. Fair to the
                    finetuned model; a pretrained model scores ~0 because
                    it has never seen the answer format, which says nothing
                    about reasoning.

choice_accuracy     highest log-probability candidate == gold. Needs no
                    format knowledge, so it is the fair zero-shot measure
                    for the pretrained model, and it is reported for the
                    finetuned model too so the two can be compared on one
                    scale.

fewshot_*           both of the above with k solved training examples of
                    the same task prepended. The strongest measurement
                    available for a pretrained model without updating its
                    weights: the demonstrations teach the format in context.
                    Demonstrations come from TRAIN, whose names are
                    disjoint from test, so they cannot reveal an answer.

Every figure is broken down by task, hop count and attribute, with the
per-task chance rate beside it. Accuracy on chain tasks at 3 hops against
2 hops is the most direct evidence of whether the model composes
inequalities or pattern-matches the premises.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_checkpoint
from language_config import get_language, use_utf8_output
from reasoning_data import TASKS, load_lexicon
from reasoning_task import (build_fewshot_prompt, encode_examples, load_examples,
                            score_example, tokenizer_for)
from workspace import REPO, Workspace

use_utf8_output()


# =============================================================================
# COMPUTATION
# =============================================================================


def aggregate(results, key):
    """Mean of each boolean metric, grouped by one field of the results."""

    metrics = [m for m in ("exact_match", "choice_correct") if m in results[0]]
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)

    return {
        str(group): {
            "n": len(rows),
            **{m: round(sum(r[m] for r in rows) / len(rows), 4) for m in metrics},
        }
        for group, rows in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def summarise(results, chance):
    metrics = [m for m in ("exact_match", "choice_correct") if m in results[0]]

    return {
        "n": len(results),
        "overall": {m: round(sum(r[m] for r in results) / len(results), 4) for m in metrics},
        "by_task": aggregate(results, "task"),
        "by_hops": aggregate(results, "hops"),
        "by_attribute": aggregate(results, "attribute"),
        "chance_by_task": chance,
        "chance_overall": round(sum(chance[r["task"]] for r in results) / len(results), 4),
    }


def run(model, tokenizer, examples, device, fewshot=0, demos_by_task=None,
        block_size=512, seed=0, progress_every=200, therefore=None):
    """Score every example; with fewshot > 0, prepend same-task demonstrations."""

    rng = random.Random(seed)
    results = []
    started = time.time()

    for n, e in enumerate(examples, 1):
        prompt_ids = None
        if fewshot:
            demos = rng.sample(demos_by_task[e["task"]], fewshot)
            prompt_ids = build_fewshot_prompt(e, demos, tokenizer, block_size)

        result = score_example(model, tokenizer, e, device, prompt_ids=prompt_ids,
                               therefore=therefore)
        result["prompt"] = e["prompt"]
        results.append(result)

        if n % progress_every == 0:
            em = sum(r["exact_match"] for r in results) / n
            ch = sum(r["choice_correct"] for r in results) / n
            print(f"    {n:>5}/{len(examples)}  EM {em:.3f}  choice {ch:.3f}  "
                  f"({time.time() - started:.0f}s)", flush=True)

    return results


# =============================================================================
# REPORTING
# =============================================================================


def write_examples(path, title, results, per_task=3):
    """Successes and failures per task, for the qualitative analysis."""

    lines = [f"# {title}", "",
             "Greedy answers. `choice` is the highest-likelihood",
             "candidate, scored separately from generation.", ""]

    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)

    for task in TASKS:
        rows = by_task.get(task, [])
        if not rows:
            continue
        right = [r for r in rows if r["exact_match"]]
        wrong = [r for r in rows if not r["exact_match"]]
        lines += [f"## {task}  (exact match {len(right)}/{len(rows)})", ""]

        for heading, subset in (("Correct", right), ("Incorrect", wrong)):
            lines += [f"**{heading}**", ""]
            for r in subset[:per_task]:
                lines += [
                    f"- ({r['attribute']}, {r['hops']} hop) {r['prompt']}",
                    f"  gold **{r['answer']}** · generated **{r['generated'] or '∅'}** "
                    f"· choice {r['choice']}",
                ]
            if not subset:
                lines.append("- none")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Reasoning accuracy for one checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="e.g. pretrained, finetuned")
    parser.add_argument("--split", default="test", choices=["validation", "test", "test_seen"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fewshot", type=int, default=0,
                        help="Also score with this many same-task demonstrations.")
    parser.add_argument("--fewshot-limit", type=int, default=600,
                        help="Examples scored in the few-shot pass (it is the slow one).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, state = load_checkpoint(args.checkpoint, device)
    workspace = Workspace(get_language(config.language))
    tokenizer = tokenizer_for(workspace.slug, config.vocab_size)

    data_dir = Path(args.data) if args.data else workspace.root / "reasoning"
    out_dir = Path(args.out) if args.out else REPO / "report" / "phase3" / workspace.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # A checkpoint trained on worked answers records its marker word; score
    # it the way it was trained, and everything else on the bare answer.
    therefore = state.get("therefore")

    examples = encode_examples(load_examples(data_dir / f"{args.split}.jsonl"), tokenizer,
                               therefore)
    if args.limit:
        # Round-robin generation means a prefix is still balanced over tasks.
        examples = examples[: args.limit]

    chance = defaultdict(list)
    for e in examples:
        chance[e["task"]].append(1 / len(e["choices"]))
    chance = {t: round(sum(v) / len(v), 4) for t, v in chance.items()}

    print("=" * 78)
    print(f"REASONING EVAL - {workspace.title} - {args.label}")
    print("=" * 78)
    print(f"  checkpoint : {args.checkpoint} (stage {state.get('stage', 'pretrain')}, "
          f"step {state.get('step')})")
    print(f"  examples   : {len(examples):,} from {args.split}")
    print(f"  device     : {device}")

    print("\n  zero-shot")
    results = run(model, tokenizer, examples, device, block_size=config.block_size,
                  therefore=therefore)
    report = {
        "language": workspace.slug,
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "stage": state.get("stage", "pretrain"),
        "step": state.get("step"),
        "split": args.split,
        "worked_answers": therefore is not None,
        "zero_shot": summarise(results, chance),
    }

    if args.fewshot:
        train = encode_examples(load_examples(data_dir / "train.jsonl", limit=6000), tokenizer)
        demos_by_task = defaultdict(list)
        for e in train:
            demos_by_task[e["task"]].append(e)

        subset = examples[: args.fewshot_limit]
        print(f"\n  {args.fewshot}-shot on {len(subset)} examples")
        few = run(model, tokenizer, subset, device, fewshot=args.fewshot,
                  demos_by_task=demos_by_task, block_size=config.block_size, seed=7)
        report[f"few_shot_{args.fewshot}"] = summarise(few, chance)

    (out_dir / f"reasoning_{args.label}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"reasoning_{args.label}_predictions.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k != "prompt"},
                             ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    write_examples(out_dir / f"reasoning_{args.label}_examples.md",
                   f"{workspace.title}: {args.label} reasoning examples", results)

    print("\n  RESULTS")
    for name in [k for k in report if k.endswith("_shot") or k.startswith("few_shot")]:
        overall = report[name]["overall"]
        print(f"    {name:<12} EM {overall['exact_match']:.3f}  "
              f"choice {overall['choice_correct']:.3f}  "
              f"(chance {report[name]['chance_overall']:.3f})")
        for task, row in report[name]["by_task"].items():
            print(f"      {task:<18} EM {row['exact_match']:.3f}  choice {row['choice_correct']:.3f}")
    print(f"\n  wrote {out_dir}")


if __name__ == "__main__":
    main()

