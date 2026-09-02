"""
Where each language's files live.

Why this module exists
----------------------
The Telugu pipeline was written one stage at a time, and each stage hard
coded the paths it happened to need:

    plan_mix.py            C:\\lma_work\\corpus\\all
    make_splits.py         C:\\lma_work\\corpus\\splits
    run_vocab_sweep.py     C:\\lma_work\\corpus\\splits\\train.jsonl
    evaluate_tokenizer.py  C:\\lma_work\\corpus\\splits\\validation.jsonl
    corpus_stats.py        C:\\lma_work\\corpus\\splits

That is fine for one language and impossible for two: running the same
stage for Assamese would mean five near-identical copies of each script,
which then drift as bugs are fixed in one and not the other. Phase 3 has
to attribute the Model H versus Model L difference to the DATA, and that
claim only holds if both corpora were built by identical code.

So the stages take --language, ask this module where the files are, and
are otherwise unchanged.

Layout
------
Everything intermediate lives on LOCAL disk under one root per language:

    C:\\lma_work\\<language>\\
        raw\\                    collected source files, untouched
        processed\\              per-source cleaning output
        corpus\\
            all\\                one labelled .jsonl per source
            reports\\            pipeline summary and removal ledger
            mix_plan.json        how much of each source to keep
            splits\\             train / validation / test  (DELIVERABLE)
        tokenizer\\
            provisional\\        throwaway model used to plan the mix
            sweep\\              vocabulary-size candidates
            final\\              the chosen tokenizer  (DELIVERABLE)

Deliverables are written here first and copied to Google Drive afterwards
by publish_deliverables.py. That ordering is not cosmetic: a 5.78 GB
train.jsonl written straight into a Drive folder while syncing was paused
was evicted from Drive's local cache mid-run, and the tokenizer sweep
failed on its second candidate with "Missing corpus files". A sync-backed
folder does not make a file durable until it has uploaded; local disk has
no such ambiguity.

Report artifacts are per-language subdirectories of the repository, so
the two models' tables and figures cannot be confused with one another:

    report\\phase1\\<language>\\
        dataset_statistics.csv / .json
        tokenizer_evaluation.json
        figures\\*.png

Telugu note
-----------
Telugu originally wrote its corpus to C:\\lma_work\\corpus, with no
language segment, because it was the only language. That directory was
moved to C:\\lma_work\\telugu\\corpus when Assamese was added, so both
languages share one layout and this module needs no special case.
"""

from pathlib import Path

from language_config import LanguageConfig, get_language


# Local working root. Intermediates and deliverables are written here; see
# the module docstring for why not straight to Drive.
WORK = Path(r"C:\lma_work")

# Google Drive root, where deliverables are published for the README links.
DRIVE = Path(r"G:\My Drive\LMA_Individual_Project")

# The repository itself: <repo>/common/workspace.py -> <repo>
REPO = Path(__file__).resolve().parents[1]


class Workspace:
    """
    Resolves every path one language's pipeline stages need.

    Paths are properties rather than a dictionary so a typo raises
    AttributeError immediately instead of producing a silently wrong
    directory that a later stage then creates.
    """

    def __init__(self, language):
        """Accepts a LanguageConfig or a language name."""

        if not isinstance(language, LanguageConfig):
            language = get_language(str(language))

        self.language = language
        self.slug = language.name.lower()

    # ---- identity -----------------------------------------------------------

    @property
    def title(self):
        """e.g. "MODEL L (ASSAMESE)", for report and console headings."""

        return f"{self.language.model_tag} ({self.language.name})".upper()

    # ---- collection and cleaning -------------------------------------------

    @property
    def root(self):
        return WORK / self.slug

    @property
    def raw(self):
        return self.root / "raw"

    @property
    def processed(self):
        return self.root / "processed"

    # ---- corpus -------------------------------------------------------------

    @property
    def corpus(self):
        return self.root / "corpus"

    @property
    def corpus_all(self):
        """One labelled .jsonl per source, before mixing."""

        return self.corpus / "all"

    @property
    def corpus_reports(self):
        return self.corpus / "reports"

    @property
    def mix_plan(self):
        return self.corpus / "mix_plan.json"

    @property
    def splits(self):
        return self.corpus / "splits"

    def source_file(self, source_type):
        """Assembled corpus file for one source."""

        return self.corpus_all / f"corpus_{source_type}.jsonl"

    def split_file(self, split):
        """train.jsonl, validation.jsonl or test.jsonl."""

        return self.splits / f"{split}.jsonl"

    # ---- tokenizer ----------------------------------------------------------

    @property
    def tokenizer(self):
        return self.root / "tokenizer"

    @property
    def tokenizer_provisional(self):
        return self.tokenizer / "provisional"

    @property
    def tokenizer_sweep(self):
        return self.tokenizer / "sweep"

    @property
    def tokenizer_final(self):
        return self.tokenizer / "final"

    def final_model(self, vocab_size):
        """Path of a trained final tokenizer at one vocabulary size."""

        return self.tokenizer_final / f"{self.slug}_final_{vocab_size}.model"

    # ---- tokenized corpus (Phase 2) -----------------------------------------
    #
    # The splits are re-encoded once into flat uint16 arrays so the training
    # loop can np.memmap them instead of parsing JSON and running
    # SentencePiece inside its inner loop. See build_token_bins.py for why.
    #
    # These live beside the corpus rather than inside it because they are
    # derived from BOTH the splits and the tokenizer: regenerating either
    # one invalidates them, and the manifest beside each .bin records the
    # hashes that let the trainer detect that.

    @property
    def tokens(self):
        return self.root / "tokens"

    def token_file(self, split):
        """Flat uint16 token stream for one split."""

        return self.tokens / f"{split}.bin"

    def token_manifest(self, split):
        """Token count, tokenizer hash and source hash for one .bin."""

        return self.tokens / f"{split}.json"

    def token_index(self, split):
        """
        Document offsets within one held-out .bin.

        Written for validation and test only. Generation evaluation needs
        prefixes that start at real document starts; perplexity does not,
        and a million-entry index for train would serve nobody.
        """

        return self.tokens / f"{split}_documents.json"

    # ---- report -------------------------------------------------------------

    @property
    def report(self):
        return REPO / "report" / "phase1" / self.slug

    @property
    def figures(self):
        return self.report / "figures"

    # ---- Drive --------------------------------------------------------------

    @property
    def drive_splits(self):
        return DRIVE / "data" / "processed" / self.slug / "splits"

    @property
    def drive_tokenizer(self):
        return DRIVE / "tokenizer" / self.slug / "final"

    @property
    def drive_sweep(self):
        """
        Vocabulary-size candidates.

        Published because the sweep is the evidence behind the chosen
        vocabulary size, not scratch work: the report's selection argument
        cites its fertility and utilisation figures, and a reader who
        wants to check them needs the models those figures came from.
        """

        return DRIVE / "tokenizer" / self.slug / "sweep"

    @property
    def drive_raw(self):
        """
        Collected source files, as downloaded, before any cleaning.

        Kept beside the splits rather than inside them because it is
        evidence rather than a dataset: it is what lets a reader check
        that the corpus was collected as the report claims, and rerun
        the cleaning stages without re-downloading 2 GB from four
        different services.
        """

        return DRIVE / "data" / "raw" / self.slug

    # ---- convenience --------------------------------------------------------

    def require(self, path, hint):
        """
        Fail early, with the command that produces the missing file.

        Stages run hours apart and out of order often enough that "no such
        file" on its own is a poor error: it does not say which stage was
        skipped.
        """

        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {path}\n{hint}")

        return Path(path)

    def __repr__(self):
        return f"Workspace({self.slug})"


def add_language_argument(parser, help_text="Language to process."):
    """
    Add the --language flag every stage shares.

    Required rather than defaulting to Telugu: a default would make it
    possible to rebuild the Telugu corpus by forgetting a flag, and the
    stages overwrite their outputs.
    """

    parser.add_argument("--language", required=True, help=help_text)

    return parser
