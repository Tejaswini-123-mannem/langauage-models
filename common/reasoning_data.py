"""
Synthetic comparative-reasoning dataset, one per language.

    python common/reasoning_data.py --language telugu
    python common/reasoning_data.py --language assamese

Writes <work>/<language>/reasoning/{train,validation,test}.jsonl, plus a
statistics file and a sample of rendered examples into
report/phase3/<language>/.

Why generated rather than downloaded
------------------------------------
The assignment requires the finetuning set to be built programmatically so
the ground truth is known. It also makes the set identical in STRUCTURE for
Model H and Model L: the same six task types, the same hop counts, the same
answer balance, the same sizes. Only the surface language differs, and it
comes from <language>/configs/reasoning.json. So a Model H versus Model L
accuracy gap on this data is a statement about the pretrained models, not
about one language getting an easier dataset.

The six tasks
-------------
Every example is a short context followed by a question whose answer is a
single word: a name from the context, or one of five fixed labels.

    task               context                    question            hops
    numeric_pair       two stated values          who has more/less      1
    numeric_relation   two stated values          more, less or equal    1
    numeric_extreme    three stated values        who has most/least     1
    chain_pair         2-3 comparisons, shuffled  endpoints: who more   2-3
    chain_extreme      2-3 comparisons, shuffled  who has most/least    2-3
    chain_yesno        2-3 comparisons, shuffled  is X more than Y?     2-3

The numeric tasks need the model to compare digit strings. The chain tasks
state no numbers at all -- only "A's height is more than B's" -- so the
answer can only come from composing the inequalities, which is the
transitive reasoning the assignment asks about. Their premises are shuffled
and each is rendered randomly in its "more" or "less" direction, so neither
the order of premises nor the order in which names first appear gives the
answer away.

Answer balance is enforced, not hoped for: yes/no is drawn 50/50 before the
example is built, relation labels 1/3 each, and the position of the correct
name in the context is uniform. A model that learned "always answer the
first name" or "always answer yes" scores chance, and the per-task chance
rate is written into the statistics so the report can quote it.

Leakage
-------
Names and items are partitioned into disjoint train / validation / test
pools before any example is generated. No test example contains a name or
item seen in training, so the test set cannot be passed by memorising
(name, attribute) associations -- a real risk here, since a 25M model can
easily memorise that "Ravi" usually answers height questions in training.
Exact duplicate prompts within a split are removed, and the script asserts
that no prompt appears in two splits.

Numbers, attributes and templates are shared across splits deliberately.
Holding out a template would test phrasing transfer, a different question
from whether the model can compare; the report says so.
"""

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from language_config import get_language, use_utf8_output
from workspace import REPO, Workspace, add_language_argument

use_utf8_output()


TASKS = (
    "numeric_pair",
    "numeric_relation",
    "numeric_extreme",
    "chain_pair",
    "chain_extreme",
    "chain_yesno",
)

SPLITS = ("train", "validation", "test")

# Share of the entity pools given to each split. With 156 names that is
# 110 / 23 / 23 and with 64 items 40 / 8 / 8.
#
# The pools were 60 names and 24 items at first, and that was too few: with
# 42 training names both models learned to answer with a memorised training
# name instead of copying one from the prompt, and scored 0 on name answers
# for the held-out test names. A larger pool makes that shortcut stop paying.
POOL_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


# =============================================================================
# LEXICON
# =============================================================================


def nfc(text):
    """
    NFC-normalise, as every Phase 1 corpus stage did.

    Matters for Assamese specifically: য় can be typed as one code point
    (U+09DF) or as য + nukta (U+09AF U+09BC), and NFC maps both to the
    decomposed form the tokenizer was trained on. Un-normalised text would
    tokenize into different pieces from the same word in pretraining.
    """

    return unicodedata.normalize("NFC", text)


def lexicon_path(language):
    return REPO / language.name.lower() / "configs" / "reasoning.json"


def load_lexicon(language):
    """Load and sanity-check one language's reasoning.json."""

    path = lexicon_path(language)
    raw = json.loads(path.read_text(encoding="utf-8"))
    lexicon = {k: v for k, v in raw.items() if not k.startswith("_")}

    for kind in ("people", "items"):
        names = [nfc(n) for n in lexicon[kind]]
        if len(set(names)) != len(names):
            dupes = [n for n, c in Counter(names).items() if c > 1]
            raise ValueError(f"{path}: duplicate {kind} {dupes}")
        lexicon[kind] = names

    # An answer label that is also an entity name would make exact match
    # ambiguous.
    clash = set(lexicon["labels"].values()) & set(lexicon["people"] + lexicon["items"])
    if clash:
        raise ValueError(f"{path}: labels collide with entity names {clash}")

    return lexicon


def partition_pool(names, seed):
    """
    Split one entity pool into disjoint train / validation / test lists.

    Seeded shuffle of a sorted copy, so the partition depends only on the
    names and the seed, not on the order they happen to be listed in.
    """

    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_val = max(4, round(n * POOL_FRACTIONS["validation"]))
    n_test = max(4, round(n * POOL_FRACTIONS["test"]))

    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


# =============================================================================
# RENDERING
# =============================================================================


class Renderer:
    """Turns abstract facts and questions into one language's sentences."""

    def __init__(self, lexicon):
        self.lex = lexicon
        self.templates = lexicon["templates"]
        self.labels = lexicon["labels"]

    def slots(self, attribute, rng, **names):
        """The template variables for one attribute, with phrasing variants drawn."""

        kind = self.lex["kinds"][attribute["kind"]]

        return {
            "noun": attribute["noun"],
            "than": attribute["than"],
            "who": kind["who"],
            "most": kind["most"],
            "of_two": rng.choice(kind["of_two"]),
            "of_all": rng.choice(kind["of_all"]),
            "of_pair": kind["of_pair"].format(a=names.get("a", ""), b=names.get("b", "")),
            **names,
        }

    def fact(self, attribute, entity, value):
        return attribute["fact"].format(e=entity, n=value)

    def comparison(self, attribute, greater, lesser, rng):
        """
        'greater has more attribute than lesser', in a random direction.

        Half the time rendered as 'lesser has LESS than greater', so the
        surface word in a premise does not predict which entity is larger.
        """

        if rng.random() < 0.5:
            template = rng.choice(self.templates["compare_more"])
            a, b = greater, lesser
        else:
            template = rng.choice(self.templates["compare_less"])
            a, b = lesser, greater

        return template.format(**self.slots(attribute, rng, a=a, b=b))

    def question(self, key, attribute, rng, **names):
        return self.templates[key].format(**self.slots(attribute, rng, **names))

    def prompt(self, context_sentences, question):
        """Full model input, ending at the answer prefix."""

        body = " ".join(context_sentences + [question])

        return nfc(f"{self.lex['prompt_prefix']} {body} {self.lex['answer_prefix']}")


# =============================================================================
# TASK GENERATORS
# =============================================================================
#
# Each returns (context_sentences, question, answer, choices, hops, n_entities)
# where `choices` is the closed set the answer is drawn from. The evaluator
# uses it for likelihood-ranked accuracy, which is the only fair way to score
# a pretrained model that has never seen the answer format.


def distinct_values(attribute, k, rng):
    low, high = attribute["range"]

    return rng.sample(range(low, high + 1), k)


def task_numeric_pair(render, attribute, entities, rng):
    a, b = entities[:2]
    va, vb = distinct_values(attribute, 2, rng)

    context = [render.fact(attribute, a, va), render.fact(attribute, b, vb)]
    ask_more = rng.random() < 0.5
    question = render.question("ask_more" if ask_more else "ask_less", attribute, rng)

    answer = a if (va > vb) == ask_more else b

    return context, question, answer, [a, b], 1, 2


def task_numeric_relation(render, attribute, entities, rng):
    a, b = entities[:2]
    relation = rng.choice(["more", "less", "equal"])

    if relation == "equal":
        va = vb = distinct_values(attribute, 1, rng)[0]
    else:
        high, low = sorted(distinct_values(attribute, 2, rng), reverse=True)
        va, vb = (high, low) if relation == "more" else (low, high)

    context = [render.fact(attribute, a, va), render.fact(attribute, b, vb)]
    question = render.question("ask_relation", attribute, rng, a=a, b=b)

    labels = render.labels
    choices = [labels["more"], labels["less"], labels["equal"]]

    return context, question, labels[relation], choices, 1, 2


def task_numeric_extreme(render, attribute, entities, rng):
    names = entities[:3]
    values = distinct_values(attribute, 3, rng)

    context = [render.fact(attribute, n, v) for n, v in zip(names, values)]
    ask_most = rng.random() < 0.5
    question = render.question("ask_most" if ask_most else "ask_least", attribute, rng)

    pick = max if ask_most else min
    answer = names[values.index(pick(values))]

    return context, question, answer, list(names), 1, 3


def chain_premises(render, attribute, ordered, rng):
    """
    Comparisons between ADJACENT entities of a strict order, shuffled.

    `ordered` runs from greatest to least. Only neighbours are stated, so
    any question about a non-adjacent pair needs at least two premises
    chained together.
    """

    premises = [
        render.comparison(attribute, ordered[i], ordered[i + 1], rng)
        for i in range(len(ordered) - 1)
    ]
    rng.shuffle(premises)

    return premises


def task_chain_pair(render, attribute, entities, rng):
    n = rng.choice([3, 4])
    ordered = list(entities[:n])                   # greatest first

    context = chain_premises(render, attribute, ordered, rng)

    # A non-adjacent pair, so the answer is never stated directly.
    i, j = rng.choice([(i, j) for i in range(n) for j in range(i + 2, n)])
    greater, lesser = ordered[i], ordered[j]
    pair = [greater, lesser]
    rng.shuffle(pair)

    ask_more = rng.random() < 0.5
    # The context mentions three or four entities, so unlike numeric_pair
    # the question has to name the two it is about: "Of X and Y, whose ...
    # is more?"
    question = render.question(
        "ask_pair_more" if ask_more else "ask_pair_less", attribute, rng,
        a=pair[0], b=pair[1],
    )

    answer = greater if ask_more else lesser

    return context, question, answer, pair, j - i, n


def task_chain_extreme(render, attribute, entities, rng):
    n = rng.choice([3, 4])
    ordered = list(entities[:n])

    context = chain_premises(render, attribute, ordered, rng)
    ask_most = rng.random() < 0.5
    question = render.question("ask_most" if ask_most else "ask_least", attribute, rng)

    answer = ordered[0] if ask_most else ordered[-1]

    # Reaching the top of a chain from its bottom takes n - 1 hops.
    return context, question, answer, list(ordered), n - 1, n


def task_chain_yesno(render, attribute, entities, rng):
    n = rng.choice([3, 4])
    ordered = list(entities[:n])

    context = chain_premises(render, attribute, ordered, rng)

    i, j = rng.choice([(i, j) for i in range(n) for j in range(i + 2, n)])
    greater, lesser = ordered[i], ordered[j]

    want_yes = rng.random() < 0.5
    ask_more = rng.random() < 0.5

    # "Is a more than b?" is yes when a is the greater; "is a less than b?"
    # is yes when a is the lesser. Choose a to hit the drawn answer.
    if ask_more:
        a, b = (greater, lesser) if want_yes else (lesser, greater)
    else:
        a, b = (lesser, greater) if want_yes else (greater, lesser)

    question = render.question(
        "ask_yes_more" if ask_more else "ask_yes_less", attribute, rng, a=a, b=b
    )

    labels = render.labels
    answer = labels["yes"] if want_yes else labels["no"]

    return context, question, answer, [labels["yes"], labels["no"]], j - i, n


GENERATORS = {
    "numeric_pair": task_numeric_pair,
    "numeric_relation": task_numeric_relation,
    "numeric_extreme": task_numeric_extreme,
    "chain_pair": task_chain_pair,
    "chain_extreme": task_chain_extreme,
    "chain_yesno": task_chain_yesno,
}


# =============================================================================
# DATASET
# =============================================================================


def generate_split(render, lexicon, pools, split, n_examples, seed):
    """
    n_examples for one split, balanced over tasks and attributes.

    Tasks are assigned round-robin rather than sampled, so every task has
    the same count to within one and per-task accuracy is measured on equal
    footing.
    """

    rng = random.Random(f"{seed}-{split}")
    attributes = lexicon["attributes"]

    examples = []
    seen = set()
    attempts = 0

    while len(examples) < n_examples:
        attempts += 1
        if attempts > n_examples * 50:
            raise RuntimeError(
                f"{split}: only {len(examples)} unique examples after "
                f"{attempts} attempts -- the entity pool is too small."
            )

        task = TASKS[len(examples) % len(TASKS)]
        attribute = rng.choice(attributes)
        pool = pools[attribute["kind"]][split]

        entities = rng.sample(pool, 4)
        context, question, answer, choices, hops, n_entities = GENERATORS[task](
            render, attribute, entities, rng
        )

        prompt = render.prompt(context, question)
        if prompt in seen:
            continue
        seen.add(prompt)

        # Some generators build choices in rank order (greatest first).
        # Nothing reads that order as a signal today, but a stored field
        # that encodes the answer is a leak waiting for a future caller.
        if task not in ("numeric_relation", "chain_yesno"):
            choices = list(choices)
            rng.shuffle(choices)

        examples.append({
            "id": f"{split}-{len(examples):06d}",
            "split": split,
            "task": task,
            "attribute": attribute["key"],
            "hops": hops,
            "n_entities": n_entities,
            "prompt": prompt,
            "answer": nfc(answer),
            "choices": [nfc(c) for c in choices],
        })

    return examples


def summarise(examples_by_split, pools, lexicon):
    """Sizes, task and answer balance, chance rates and pool sizes."""

    stats = {"splits": {}, "pools": {}, "templates": {}}

    for split, examples in examples_by_split.items():
        per_task = Counter(e["task"] for e in examples)
        per_attribute = Counter(e["attribute"] for e in examples)
        per_hops = Counter(e["hops"] for e in examples)

        answers = defaultdict(Counter)
        chance = defaultdict(list)
        for e in examples:
            answers[e["task"]][e["answer"]] += 1
            chance[e["task"]].append(1.0 / len(e["choices"]))

        stats["splits"][split] = {
            "examples": len(examples),
            "tasks": dict(per_task),
            "attributes": dict(per_attribute),
            "hops": {str(k): v for k, v in sorted(per_hops.items())},
            "chance_accuracy": {
                task: round(sum(v) / len(v), 4) for task, v in chance.items()
            },
            "label_balance": {
                task: dict(counter.most_common(5))
                for task, counter in answers.items()
                if task in ("numeric_relation", "chain_yesno")
            },
            "mean_prompt_characters": round(
                sum(len(e["prompt"]) for e in examples) / len(examples), 1
            ),
        }

    for kind, split_pools in pools.items():
        stats["pools"][kind] = {split: len(names) for split, names in split_pools.items()}
        stats["pools"][f"{kind}_test_names"] = split_pools["test"]

    templates = lexicon["templates"]
    stats["templates"] = {
        "comparison_variants": len(templates["compare_more"]) + len(templates["compare_less"]),
        "question_templates": sum(1 for k in templates if k.startswith("ask_")),
        "attributes": [a["key"] for a in lexicon["attributes"]],
        "fact_templates": len(lexicon["attributes"]),
    }

    return stats


def check_leakage(examples_by_split, pools):
    """Assert that no prompt, and no entity, is shared between splits."""

    prompts = {split: {e["prompt"] for e in ex} for split, ex in examples_by_split.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = prompts[a] & prompts[b]
                assert not shared, f"{len(shared)} prompts shared by {a} and {b}"

    for kind, split_pools in pools.items():
        for a in SPLITS:
            for b in SPLITS:
                if a < b:
                    shared = set(split_pools[a]) & set(split_pools[b])
                    assert not shared, f"{kind} {shared} in both {a} and {b}"


def write_samples(path, examples_by_split, title, per_task=3):
    """A readable Markdown sample: a few examples of every task from test."""

    lines = [
        f"# {title}: synthetic reasoning examples",
        "",
        "Drawn from the TEST split, whose names and items never occur in",
        "training. Generated by `common/reasoning_data.py` from",
        "`<language>/configs/reasoning.json`.",
        "",
    ]

    by_task = defaultdict(list)
    for e in examples_by_split["test"]:
        by_task[e["task"]].append(e)

    for task in TASKS:
        lines += [f"## {task}", ""]
        for e in by_task[task][:per_task]:
            lines += [
                f"- **prompt** ({e['attribute']}, {e['hops']} hop): {e['prompt']}",
                f"  **answer**: {e['answer']}  ·  choices: {' / '.join(e['choices'])}",
            ]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate the reasoning dataset.")
    add_language_argument(parser)
    parser.add_argument("--train", type=int, default=120000)
    parser.add_argument("--validation", type=int, default=1200)
    parser.add_argument("--test", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output directory for the .jsonl files.")
    args = parser.parse_args()

    language = get_language(args.language)
    workspace = Workspace(language)
    lexicon = load_lexicon(language)
    render = Renderer(lexicon)

    out_dir = Path(args.out) if args.out else workspace.root / "reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = REPO / "report" / "phase3" / workspace.slug
    report_dir.mkdir(parents=True, exist_ok=True)

    # Separate seeds per pool, derived from the language, so the Telugu and
    # Assamese partitions are independent draws.
    base = int(hashlib.sha256(f"{args.seed}-{workspace.slug}".encode()).hexdigest()[:8], 16)
    pools = {
        "people": partition_pool(lexicon["people"], base + 1),
        "items": partition_pool(lexicon["items"], base + 2),
    }

    sizes = {"train": args.train, "validation": args.validation, "test": args.test}
    examples_by_split = {
        split: generate_split(render, lexicon, pools, split, sizes[split], args.seed)
        for split in SPLITS
    }

    check_leakage(examples_by_split, pools)

    for split, examples in examples_by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as sink:
            for e in examples:
                sink.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats = summarise(examples_by_split, pools, lexicon)
    stats.update({"language": workspace.slug, "seed": args.seed, "out_dir": str(out_dir)})
    (report_dir / "reasoning_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_samples(report_dir / "reasoning_samples.md", examples_by_split, workspace.title)

    print("=" * 78)
    print(f"REASONING DATA - {workspace.title}")
    print("=" * 78)
    for split in SPLITS:
        s = stats["splits"][split]
        print(f"  {split:<11} {s['examples']:>6,} examples, "
              f"mean prompt {s['mean_prompt_characters']:.0f} chars")
    print(f"  people pools : {stats['pools']['people']}")
    print(f"  item pools   : {stats['pools']['items']}")
    print(f"  wrote {out_dir}")
    print(f"  wrote {report_dir}")


if __name__ == "__main__":
    main()
