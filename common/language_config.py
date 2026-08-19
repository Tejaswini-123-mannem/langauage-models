"""
Language definitions shared by the Model H and Model L pipelines.

Why this module exists
----------------------
Every collection, filtering and deduplication stage needs to answer the
same question: "is this text in the target language?" For Telugu that was
hardcoded as the Unicode range U+0C00-U+0C7F in eight separate scripts.

Adding a second language by copying those scripts would duplicate the
logic, and the copies would drift as bugs were fixed in one and not the
other. That matters beyond tidiness: Phase 3 has to attribute the
difference between Model H and Model L to the DATA. That claim only holds
if both corpora were built by identical code. A shared configuration is
what makes it identical by construction rather than by inspection.

So the pipeline is parameterised by a LanguageConfig, and a language is
added by describing it here.

The Assamese problem this module solves
---------------------------------------
Telugu has its own Unicode block, so "is this character Telugu?" is a
range check and nothing else can be confused with it.

Assamese does not. It is written in the Bengali-Assamese script, sharing
U+0980-U+09FF with Bengali, which has roughly fifteen times more text
online. Web crawls, Internet Archive metadata and Wikipedia mirrors
routinely mislabel one as the other. A range check alone would accept
Bengali text as Assamese without complaint, and the corpus would be
quietly, largely wrong. Perplexity would still look reasonable, so
nothing downstream would reveal the error.

Assamese is distinguished by two letters absent from standard Bengali
orthography:

    U+09F0  ৰ  ASSAMESE LETTER RA   (Bengali writes র, U+09B0)
    U+09F1  ৱ  ASSAMESE LETTER WA   (Bengali writes ব, U+09AC)

ৰ is the ordinary Assamese "ra", so it appears throughout any genuine
Assamese text at a frequency comparable to "r" in English. Bengali text
contains it rarely or never. Requiring a minimum density of these
characters is therefore a cheap and reliable language discriminator, and
it is expressed here as `required_char_ratio`.
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def use_utf8_output():
    """
    Force UTF-8 on stdout and stderr.

    Windows consoles default to a legacy code page (cp1252 here), which
    cannot represent Indic text. Any script that prints a language name,
    a sample of the corpus or a tokenized string will die with
    UnicodeEncodeError partway through a long run.

    Setting PYTHONIOENCODING externally also works, but relies on every
    future invocation remembering to do so. Doing it here means the
    scripts are safe to run directly from a terminal.
    """

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # Already wrapped, or not a real stream (e.g. redirected).
            pass


# =============================================================================
# CONFIGURATION TYPE
# =============================================================================

@dataclass(frozen=True)
class LanguageConfig:
    """Everything the pipeline needs to know about one target language."""

    # ---- identity -----------------------------------------------------------

    name: str                       # human-readable, e.g. "Telugu"
    endonym: str                    # name in its own script
    code_2: str                     # ISO 639-1, e.g. "te"
    code_3: str                     # ISO 639-3, e.g. "tel"

    # ---- script -------------------------------------------------------------

    script_name: str
    script_start: str               # first codepoint of the block
    script_end: str                 # last codepoint of the block

    # Characters that must be present for text to count as this language,
    # used where the script block is shared with another language. Empty
    # when the block is unambiguous.
    required_chars: Tuple[str, ...] = ()

    # Minimum share of in-script characters that must be `required_chars`.
    # Ignored when required_chars is empty.
    required_char_ratio: float = 0.0

    # ---- sources ------------------------------------------------------------

    wikipedia_prefix: str = ""      # e.g. "tewiki"
    wikisource_prefix: str = ""     # e.g. "tewikisource"
    sangraha_subset: str = ""       # e.g. "verified/tel"
    archive_queries: Tuple[str, ...] = ()

    # ---- MediaWiki cleanup patterns ----------------------------------------
    #
    # Wiki markup is localised, so the strings that identify reference
    # sections, file links and list articles differ per language. Holding
    # them here keeps one cleaner working for every language instead of a
    # copy per wiki.

    # Section headings after which the article is reference apparatus
    # rather than prose: citations, external links, "see also". Everything
    # from the first match onward is discarded.
    wiki_appendix_headings: Tuple[str, ...] = ()

    # Namespace prefixes used for embedded files and images.
    wiki_file_namespaces: Tuple[str, ...] = ("File", "Image")

    # Title fragments marking list, index and disambiguation pages. These
    # are mostly bullet lists of proper nouns, which teach a tokenizer
    # name fragments rather than language.
    wiki_list_title_patterns: Tuple[str, ...] = ()

    # Category namespace names, stripped when they survive parsing.
    wiki_category_names: Tuple[str, ...] = ("Category",)

    # -------------------------------------------------------------------------
    # WIKISOURCE Page: NAMESPACE (ns 104)
    # -------------------------------------------------------------------------
    #
    # Wikisource holds two kinds of page. Namespace 0 has assembled works;
    # namespace 104 ("Page:") has the transcription of a single scanned
    # page, which editors later stitch together into a namespace 0 work.
    #
    # Whether to keep namespace 104 depends entirely on the language, and
    # getting it wrong is costly in both directions:
    #
    #   Telugu    180,300 Page: entries against 21,769 assembled works.
    #             The transcriptions are largely duplicated by the works,
    #             so including them would double-count the corpus.
    #
    #   Assamese  82,410 Page: entries against 2,999 namespace 0 pages,
    #             of which only 368 survived cleaning. The community
    #             transcribes but rarely assembles, so the Page:
    #             namespace IS the corpus. Excluding it discarded
    #             85,079,684 characters, roughly 24M tokens, and left
    #             Wikisource contributing 0.4M tokens instead of ~24.7M.
    #
    # The Telugu rule was carried to Assamese unexamined and silently
    # removed almost the entire source. Hence a per-language flag rather
    # than a shared default.
    # -------------------------------------------------------------------------

    wikisource_include_page_namespace: bool = False

    # ---- filtering thresholds ----------------------------------------------

    min_characters: int = 100
    min_script_characters: int = 50
    min_script_ratio: float = 0.10

    # Tesseract language code, empty when OCR is not used.
    tesseract_code: str = ""

    # -------------------------------------------------------------------------
    # CORPUS MIXING POLICY
    # -------------------------------------------------------------------------
    #
    # Which sources exist for this language, and what happens when manual
    # collection over-delivers. Held here rather than in plan_mix.py so the
    # two languages run the same mixing code over different policy.
    #
    #   corpus_sources      every source, in assembly order, with the group
    #                       that decides whether it counts as manual
    #   manual_priority     order the manual token budget is spent in. The
    #                       budget is consumed front to back, so the scarce
    #                       and expensive sources are taken whole and the
    #                       abundant tail source absorbs any trimming.
    #   manual_source_caps  per-source ceilings in tokens. A source larger
    #                       than its ceiling leaves the surplus unused
    #                       rather than crowding out the sources behind it.
    # -------------------------------------------------------------------------

    corpus_sources: Tuple[Tuple[str, str], ...] = ()

    manual_priority: Tuple[str, ...] = ()

    manual_source_caps: Dict[str, int] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # PER-SOURCE HELD-OUT FRACTION
    # -------------------------------------------------------------------------
    #
    # make_splits assigns documents by content hash, and by default every
    # source gives the same share to validation and test. That is correct
    # when documents are comparable in size. They are not:
    #
    #     source       documents    mean document
    #     sangraha       315,259       ~2,800 chars
    #     wikisource      54,197       ~1,200 chars
    #     wikipedia       24,171       ~2,500 chars
    #     archive          2,416     ~318,000 chars      (whole books)
    #
    # Internet Archive documents are entire scanned books, so the source
    # is 50% of the Assamese corpus by characters but only 0.6% of it by
    # document count. At a 0.5% held-out fraction that leaves ELEVEN books
    # in validation, carrying over 40% of the split's text -- a single long
    # volume can move the number several percent.
    #
    # Per-source fertility on held-out text is what chooses the vocabulary
    # size, so measuring the largest source on eleven documents is the one
    # place this actually costs something.
    #
    # A larger held-out fraction for that source fixes it. Splitting books
    # into pages would give thousands of documents and is the wrong answer:
    # pages of one book would land in both train and test, turning sampling
    # noise into genuine leakage.
    #
    # Maps source_type -> (val_fraction, test_fraction). Sources absent
    # from the mapping use the command-line defaults.
    # -------------------------------------------------------------------------

    heldout_fractions: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # TOKENIZER POLICY
    # -------------------------------------------------------------------------
    #
    # Candidate vocabulary sizes for the sweep. The assignment asks for the
    # size to be chosen on measured fertility, so the range has to bracket
    # the answer from both sides: candidates that are visibly too small and
    # visibly too large are what make the chosen size an argument rather
    # than an assertion.
    #
    # WHY THE RANGE MOVED DOWN
    # ------------------------
    # The first sweeps ranged to 64k for Telugu and 32k for Assamese, and
    # chose on compression alone. That was the wrong objective, because
    # Phase 2 fixes the model at ~25M parameters and the embedding matrix
    # is part of that budget, not outside it.
    #
    # At d_model=384 with a tied output head, the embedding costs
    # vocab x 384 parameters and each Transformer block costs 12 x 384^2
    # = 1,769,472. So vocabulary is bought directly out of depth:
    #
    #     vocab    embedding    layers affordable within ~25M    embedding %
    #     32,000  12,288,000    7                                53%
    #     24,000   9,216,000    8                                46%
    #     16,000   6,144,000   10                                30%
    #      8,000   3,072,000   12                                13%
    #      4,000   1,536,000   13                                 6%
    #
    # A 32k vocabulary spends nearly half the model on a lookup table that
    # performs no computation, leaving half the depth for the comparative
    # reasoning Phase 3 is graded on. Dropping to 8k buys five more layers
    # at a measured cost of roughly 10% compression.
    #
    # So the bracket is 6k-12k for both languages, within the range the
    # parameter budget actually permits. 4k was dropped from the sweep:
    # the earlier Assamese run already measured it at 2.1147 fertility
    # against 8k's 1.8493 and 95.8% vocabulary utilisation, so it is
    # known to be too small rather than needing re-measuring. The
    # selection rule is unchanged: measured fertility on held-out text
    # decides.
    # -------------------------------------------------------------------------

    vocab_sweep_sizes: Tuple[int, ...] = (6000, 8000, 10000, 12000)

    # Sentences shown in the qualitative tokenization table. When empty,
    # evaluate_tokenizer.py draws examples from held-out text instead, one
    # per source, which keeps the table honest for a language whose
    # phrasing the author cannot vouch for.
    example_sentences: Tuple[str, ...] = ()

    # ---- report identity ----------------------------------------------------

    # "Model H" or "Model L", used in report titles and figure captions.
    model_tag: str = "Model H"

    # ---- notes for the report ----------------------------------------------

    notes: str = ""

    # -------------------------------------------------------------------------
    # SCRIPT TESTS
    # -------------------------------------------------------------------------

    def is_script_char(self, ch: str) -> bool:
        """True when a character belongs to this language's script block."""

        return self.script_start <= ch <= self.script_end

    def script_count(self, text: str) -> int:
        """Number of characters in this language's script block."""

        return sum(1 for ch in text if self.is_script_char(ch))

    def script_ratio(self, text: str) -> float:
        """
        Share of ALPHABETIC characters that belong to this script.

        Alphabetic characters are the denominator, not all characters, so
        that digits, punctuation and whitespace cannot drag the ratio down
        and cause legitimate text to be discarded.
        """

        alphabetic = [ch for ch in text if ch.isalpha()]

        if not alphabetic:
            return 0.0

        return sum(
            1 for ch in alphabetic if self.is_script_char(ch)
        ) / len(alphabetic)

    def required_char_density(self, text: str) -> float:
        """
        Share of in-script characters that are language-specific markers.

        Returns 1.0 when the language declares no markers, so callers can
        apply the same test regardless of language.
        """

        if not self.required_chars:
            return 1.0

        in_script = self.script_count(text)

        if in_script == 0:
            return 0.0

        markers = sum(1 for ch in text if ch in self.required_chars)

        return markers / in_script

    def is_target_language(self, text: str) -> bool:
        """
        Whether a document should be accepted as this language.

        Combines three tests: enough text at all, enough of it in the
        right script, and — where the script is shared — enough
        language-specific characters to rule out the neighbour language.
        """

        if not text or len(text.strip()) < self.min_characters:
            return False

        if self.script_count(text) < self.min_script_characters:
            return False

        if self.script_ratio(text) < self.min_script_ratio:
            return False

        if self.required_chars:
            if self.required_char_density(text) < self.required_char_ratio:
                return False

        return True

    # -------------------------------------------------------------------------
    # DERIVED NAMES
    # -------------------------------------------------------------------------

    def wikipedia_dump(self, date: str) -> str:
        return f"{self.wikipedia_prefix}-{date}-pages-articles.xml.bz2"

    def wikisource_dump(self, date: str) -> str:
        return f"{self.wikisource_prefix}-{date}-pages-articles.xml.bz2"


# =============================================================================
# TELUGU  (Model H)
# =============================================================================

TELUGU = LanguageConfig(
    name="Telugu",
    endonym="తెలుగు",
    code_2="te",
    code_3="tel",

    script_name="Telugu",
    script_start="ఀ",
    script_end="౿",

    # Telugu has a dedicated Unicode block shared with no other language,
    # so membership in the block is sufficient identification.
    required_chars=(),
    required_char_ratio=0.0,

    wikipedia_prefix="tewiki",
    wikisource_prefix="tewikisource",
    sangraha_subset="verified/tel",
    archive_queries=(
        'language:"tel" AND mediatype:texts',
        'subject:"Telugu" AND mediatype:texts',
        'title:"Telugu" AND mediatype:texts',
        'creator:"Telugu" AND mediatype:texts',
    ),

    wiki_appendix_headings=(
        r"ఇవి\s*కూడా\s*చూడండి",     # see also
        r"మూలాలు",                    # references
        r"బయటి\s*లింకులు",            # external links
        r"సూచికలు",                   # indexes
        r"గ్రంథ\s*సూచి",              # bibliography
        r"References",
        r"External\s+links",
        r"See\s+also",
        r"Further\s+reading",
    ),

    wiki_file_namespaces=("File", "Image", "దస్త్రం", "బొమ్మ"),

    wiki_list_title_patterns=(
        r"జాబితా",          # list
        r"సూచిక",           # index
        r"అయోమయ",          # disambiguation
        r"\blist\s+of\b",
        r"disambiguation",
    ),

    wiki_category_names=("Category", "వర్గం"),

    # Telugu Wikisource has 21,769 assembled works, and its 180,300 Page:
    # entries largely duplicate them. Including them would double-count.
    wikisource_include_page_namespace=False,

    min_characters=100,
    min_script_characters=50,
    min_script_ratio=0.10,

    tesseract_code="tel",

    corpus_sources=(
        ("textbook",   "manual"),
        ("wikisource", "manual"),
        ("wikipedia",  "manual"),
        ("archive",    "manual"),
        ("sangraha",   "sangraha"),
    ),

    # Archive is last because it is newspapers: the largest manual source
    # and the most repetitive, so trimming it costs the least information
    # per token removed. The three ahead of it are each the only source of
    # their register.
    manual_priority=("textbook", "wikisource", "wikipedia", "archive"),

    # Wikipedia yielded ~88M tokens against an expected 30-75M. Under a
    # strict priority order it would have consumed almost the whole manual
    # budget and reduced Internet Archive to ~1.6% of its collected size,
    # deleting the contemporary-newspaper register entirely.
    manual_source_caps={"wikipedia": 60_000_000},

    # Same bracket as Assamese. Telugu's larger corpus would support a
    # larger vocabulary on compression grounds alone -- the first sweep
    # chose 32k on exactly that reasoning -- but the ~25M parameter
    # budget is shared by both models, and 32k spends 53% of it on the
    # embedding. The corpus size argument does not survive the budget.
    vocab_sweep_sizes=(6000, 8000, 10000, 12000),

    example_sentences=(
        # Everyday prose.
        "తెలుగు భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో ఒకటి.",
        # Morphologically heavy, to expose fertility on long agglutinated forms.
        "విద్యార్థులందరూ పాఠశాలకు వెళ్ళి చదువుకుంటున్నారు.",
        # Classical / literary register, as found in the Wikisource poetry.
        "దేశభాషలందు తెలుగు లెస్స.",
        # Mixed script and numerals, typical of the OCR'd subject textbooks.
        "భౌతిక శాస్త్రంలో 25 డిగ్రీల ఉష్ణోగ్రత వద్ద density అనేది ముఖ్యం.",
    ),

    model_tag="Model H",

    notes=(
        "Higher-resource model. Chosen for abundant public text: Sangraha "
        "verified/tel spans 41 shards, and Internet Archive holds over "
        "667,000 Telugu items."
    ),
)


# =============================================================================
# ASSAMESE  (Model L)
# =============================================================================

ASSAMESE = LanguageConfig(
    name="Assamese",
    endonym="অসমীয়া",
    code_2="as",
    code_3="asm",

    script_name="Bengali-Assamese",
    script_start="ঀ",
    script_end="৿",

    # Shared script, so block membership alone cannot separate Assamese
    # from Bengali. See the module docstring.
    required_chars=("ৰ", "ৱ"),   # ৰ RA, ৱ WA

    # ৰ is ordinary Assamese "ra" and appears throughout natural text, so
    # a low bar separates Assamese from Bengali without rejecting short
    # Assamese passages. Deliberately conservative: the intent is to
    # exclude text with essentially no Assamese markers, not to demand a
    # particular style. Calibrate against measured density before
    # tightening.
    required_char_ratio=0.005,

    wikipedia_prefix="aswiki",
    wikisource_prefix="aswikisource",
    sangraha_subset="verified/asm",

    # -------------------------------------------------------------------------
    # INTERNET ARCHIVE QUERIES
    # -------------------------------------------------------------------------
    #
    # The first harvest ran only the spelled-out language name and recorded
    # that `language:"asm"` "returns nothing". That was a query-syntax
    # artefact, not a fact about the index. Measured against the search API:
    #
    #     language:(assamese) AND mediatype:(texts)     2,576 items
    #     language:(asm)      AND mediatype:(texts)    15,385 items
    #     the two OR'd together                        17,961 items
    #
    # The `asm` seam is almost entirely disjoint from the other, and it is
    # where the digitised books live: the Digital Library of India scans
    # (`in.ernet.dli.*`) tag language as the ISO 639-3 code, never the
    # English name. Those are the same volumes the Directorate of Library
    # Services catalogue lists, and they carry Archive's own `_djvu.txt`,
    # so they need no OCR pass of our own.
    #
    # Searching one spelling and not the other cost roughly seven eighths
    # of the available items, which is the single largest reason Model L's
    # corpus fell short of the token target.
    #
    # Both spellings are listed, plus the field-free forms, because the
    # collector deduplicates identifiers across queries anyway: an overlap
    # is free, a miss is not.
    # -------------------------------------------------------------------------

    archive_queries=(
        'language:(asm) AND mediatype:(texts)',
        'language:(assamese) AND mediatype:(texts)',
        'subject:(assamese) AND mediatype:(texts)',
        'title:(assamese) AND mediatype:(texts)',
    ),

    wiki_appendix_headings=(
        r"তথ্যসূত্ৰ",                 # references
        r"তথ্য\s*সংগ্ৰহ",             # sources
        r"বাহ্যিক\s*সংযোগ",           # external links
        r"লগতে\s*চাওক",               # see also
        r"গ্ৰন্থপঞ্জী",                # bibliography
        r"পাদটীকা",                   # footnotes
        r"References",
        r"External\s+links",
        r"See\s+also",
        r"Further\s+reading",
    ),

    wiki_file_namespaces=("File", "Image", "চিত্ৰ", "নথি"),

    wiki_list_title_patterns=(
        r"তালিকা",                    # list
        r"সূচী",                      # index
        r"দ্ব্যৰ্থতা\s*নিৰাকৰণ",       # disambiguation
        r"\blist\s+of\b",
        r"disambiguation",
    ),

    wiki_category_names=("Category", "শ্ৰেণী", "বিষয়শ্ৰেণী"),

    # Assamese Wikisource assembles almost nothing: 368 usable namespace 0
    # works against 82,410 Page: transcriptions, 94.7% of which hold real
    # text. Measured content: 85,079,684 characters, ~24M tokens, spanning
    # 44,723 validated and 12,731 proofread pages. Excluding it would
    # discard the source almost entirely.
    wikisource_include_page_namespace=True,

    min_characters=100,
    min_script_characters=50,
    min_script_ratio=0.10,

    # No Assamese pack installed, and OCR is not part of the Model L plan.
    tesseract_code="",

    # No textbook source: OCR was not run for Assamese.
    corpus_sources=(
        ("wikisource", "manual"),
        ("wikipedia",  "manual"),
        ("archive",    "manual"),
        ("sangraha",   "sangraha"),
    ),

    # Archive first: 455 documents harvested from an Internet Archive
    # holding of 2,576 Assamese items against Telugu's 667,239. It is the
    # scarcest source here and the one that cannot be regrown.
    #
    # The order barely binds in practice. Assamese manual collection does
    # not over-deliver against any plausible budget, so every manual source
    # is kept whole and nothing is trimmed. It is set correctly anyway, so
    # that raising --target-manual later cannot silently discard the
    # scarcest source.
    manual_priority=("archive", "wikisource", "wikipedia"),

    # Nothing is capped. Telugu capped Wikipedia because manual collection
    # overshot its budget; here every manual token available is needed to
    # get the corpus as close to the target as the language allows.
    manual_source_caps={},

    # Empty deliberately, after measuring the alternative.
    #
    # Archive contributes only ~12 documents to each held-out split,
    # because its documents are whole books. That looks alarming and is
    # not: those 12 books carry 3.8M characters, roughly 600,000 words,
    # which is an ample sample for fertility.
    #
    # More importantly, 0.5% is exactly the representative fraction here.
    # Archive is 43.1% of the corpus by characters, and at 0.5% it is
    # 43.1% of validation. Raising it to 2% was tried and pushed archive
    # to 73.7% of validation, which would have:
    #
    #   - measured Phase 2 perplexity on a held-out set dominated by the
    #     noisiest, OCR-degraded part of the corpus,
    #   - broken the Phase 3 comparison, since Telugu's validation
    #     mirrors its corpus and Assamese's would not, and
    #   - removed 21M characters from training for no gain.
    #
    # The residual limitation is book-level variance in per-source
    # figures, since OCR quality differs between volumes. That is
    # reported rather than engineered away.
    heldout_fractions={},

    # Bracketed an octave lower than Telugu's. The Assamese corpus is
    # roughly a tenth the size, and a vocabulary is supported by the data
    # that has to fill it: Telugu's 64k candidate already left 7.4% of its
    # pieces unused on held-out text, so sizes above ~32k here would be
    # measuring the same failure with less data behind it.
    # Bracketed for the parameter budget rather than the corpus size;
    # see the LanguageConfig field comment. The previous 4k-32k sweep
    # chose 16k on held-out fertility, which is 30% of the model.
    vocab_sweep_sizes=(6000, 8000, 10000, 12000),

    # Deliberately empty. The tokenization examples are drawn from held-out
    # Assamese text instead of being written by hand, because the author
    # does not read Assamese and invented sentences could carry errors that
    # the fertility table would then quietly attribute to the tokenizer.
    example_sentences=(),

    model_tag="Model L",

    notes=(
        "Lower-resource model, from the assignment's permitted list. "
        "Internet Archive holds 2,576 Assamese items against Telugu's "
        "667,239, so Wikipedia is the primary manual source rather than "
        "Archive. Shares its script with Bengali, which requires the "
        "marker-character test above."
    ),
)


# =============================================================================
# REGISTRY
# =============================================================================

LANGUAGES: Dict[str, LanguageConfig] = {
    "telugu": TELUGU,
    "assamese": ASSAMESE,
}


def get_language(name: str) -> LanguageConfig:
    """Look up a language configuration by name, case-insensitively."""

    key = name.strip().lower()

    if key not in LANGUAGES:
        raise KeyError(
            f"Unknown language {name!r}. "
            f"Available: {', '.join(sorted(LANGUAGES))}"
        )

    return LANGUAGES[key]


# =============================================================================
# SELF-DESCRIPTION
# =============================================================================

if __name__ == "__main__":

    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    for key, cfg in LANGUAGES.items():

        print()
        print("=" * 70)
        print(f"{cfg.name} ({cfg.endonym})   [{key}]")
        print("=" * 70)
        print(f"  codes            : {cfg.code_2} / {cfg.code_3}")
        print(
            f"  script           : {cfg.script_name} "
            f"U+{ord(cfg.script_start):04X}-U+{ord(cfg.script_end):04X}"
        )

        if cfg.required_chars:
            markers = " ".join(
                f"{c} (U+{ord(c):04X})" for c in cfg.required_chars
            )
            print(f"  marker chars     : {markers}")
            print(f"  marker min ratio : {cfg.required_char_ratio:.3%}")
        else:
            print("  marker chars     : none needed (unambiguous block)")

        print(f"  wikipedia        : {cfg.wikipedia_prefix}")
        print(f"  wikisource       : {cfg.wikisource_prefix}")
        print(f"  sangraha         : {cfg.sangraha_subset}")
        print(f"  tesseract        : {cfg.tesseract_code or '(not used)'}")
