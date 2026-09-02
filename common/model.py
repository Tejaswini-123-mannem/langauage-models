"""
Decoder-only Transformer language model, written from primitives.

Scope of what is allowed here
-----------------------------
The assignment permits nn.Linear, nn.Embedding, nn.LayerNorm and
nn.Dropout, and forbids nn.Transformer*, HuggingFace model classes, and
any pre-built attention block. It also forbids the shortcut that matters
most in practice: torch.nn.functional.scaled_dot_product_attention would
do the whole of CausalSelfAttention.forward in one call, and is not used.
The QK^T matmul, the 1/sqrt(d_k) scaling, the additive causal mask, the
softmax and the AV matmul are all written out, because the point of the
phase is being able to explain each one.

One model class, two models
---------------------------
Model H (Telugu) and Model L (Assamese) are the same code with different
configs and different weights. That is deliberate and is the load-bearing
assumption of Phase 3: if the two models were built by two copies of the
architecture, any H-versus-L difference could be an architecture bug in
one copy rather than a fact about the data. Identical code plus separate
configs, corpora, tokenizers and weights is what makes "the difference is
the data" a defensible claim.

The two configs differ only in the tokenizer they were built for. Both
vocabularies came out of Phase 1 at 8,000 pieces, so in this project the
parameter counts happen to be identical too -- but nothing here assumes
that, and vocab_size is read from each config.

Architecture summary
--------------------
    d_model 384, 12 layers, 6 heads (d_k = 64), d_ff 1536,
    context 512, learned absolute positions, pre-norm, GELU,
    tied output head, dropout 0.1

    token embedding    8000 x 384                =  3,072,000
    position embedding  512 x 384                =    196,608
    12 blocks         x 1,774,464                = 21,293,568
    final LayerNorm                              =        768
    lm_head           tied to token embedding    =          0
                                                   ----------
                                                   24,562,944

against the ~25M target. The tied head is what makes it fit: untied, the
same model is 27.6M.

Why this depth/width trade
--------------------------
The budget is ~25M parameters and the embedding is inside it, so
vocabulary, depth and width compete for the same pool. Phase 1 chose an
8,000-piece vocabulary precisely to leave room for depth: at d_model 384
one block costs 12 x 384^2 = 1,769,472, so every 1,769,472 parameters
spent elsewhere is a whole Transformer layer not built.

Given that, 12 layers x 384 is preferred over a shallower, wider model
(e.g. 6 x 544) because depth is what composes: attention can only route
information one hop per layer, and the comparative-reasoning task in
Phase 3 -- A > B, B > C, therefore A > C -- is explicitly multi-hop.
Width mostly buys per-token capacity, which matters less at this scale
than the number of routing steps available.

Positional encoding: learned absolute
-------------------------------------
Self-attention is permutation invariant. Permute the rows of X and the
attention output is the same set of vectors, permuted -- the mechanism
has no notion of "before" or "adjacent". Position has to be injected, and
here it is a learned nn.Embedding(512, 384) added to the token embedding.

This is GPT-2's scheme, chosen for being the simplest thing that is
clearly correct and the easiest to ablate for the optional bonus (set
use_positional_embedding false and the table is gone).

It constrains the maximum sequence length HARD. The table has exactly
block_size rows, so position 512 does not exist -- indexing it raises.
Even if the table were larger, rows never seen during training hold their
random initialisation, so the model cannot extrapolate to longer contexts
the way sinusoidal encodings or RoPE partially can. block_size is a
property of the trained weights, not just of the config.
"""

import json
import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class GPTConfig:
    """
    Hyperparameters of one model.

    Serialised into every checkpoint so that a checkpoint is self
    describing: resuming a run, or loading it months later for Phase 3,
    never depends on guessing which config file produced it.
    """

    # identity, carried into checkpoints so a stray file can be traced back
    name: str = "model"
    language: str = "unknown"

    # architecture
    vocab_size: int = 8000
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536

    # regularisation
    dropout: float = 0.1

    # nn.Linear bias terms. GPT-2 has them; later models drop them because
    # they cost parameters and buy little once LayerNorm is present. Kept
    # here so the model matches the classic formulation being taught.
    bias: bool = True

    # Tie lm_head.weight to the token embedding. Saves vocab_size x d_model
    # parameters -- 3,072,000 here, 12.5% of the budget -- and is well
    # motivated: the vector that represents a token on the way in and the
    # vector that scores it on the way out describe the same token.
    tie_weights: bool = True

    # Ablation switch for the optional bonus: with this false the model has
    # no position information at all and attention becomes a bag of tokens.
    use_positional_embedding: bool = True

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_head "
                f"{self.n_head}; heads must partition the model dimension."
            )

    @property
    def d_head(self):
        """Per-head subspace width, d_k in the assignment's notation."""

        return self.d_model // self.n_head

    @classmethod
    def from_json(cls, path):
        """Load a config file, ignoring keys that begin with an underscore."""

        raw = json.loads(open(path, encoding="utf-8").read())
        fields = {k: v for k, v in raw.items() if not k.startswith("_")}

        unknown = set(fields) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")

        return cls(**fields)

    def to_dict(self):
        return asdict(self)


# =============================================================================
# MULTI-HEAD CAUSAL SELF-ATTENTION
# =============================================================================


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention, computed explicitly.

    Shapes, for batch B and sequence length T, with h heads of width d_k:

        input                    (B, T, d)
        qkv projection           (B, T, 3d)      -> three (B, T, d)
        head split               (B, h, T, d_k)
        scores QK^T / sqrt(d_k)  (B, h, T, T)
        + causal mask, softmax   (B, h, T, T)
        att @ V                  (B, h, T, d_k)
        head merge               (B, T, d)
        output projection        (B, T, d)
    """

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head

        # Q, K and V are three independent linear maps, but they read the
        # same input, so they are fused into one (d -> 3d) matmul and split
        # afterwards. One large GEMM is meaningfully faster on a GPU than
        # three small ones, and it is mathematically identical.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout on attention PROBABILITIES: randomly severs some
        # query-key edges, so a head cannot depend on exactly one position.
        self.attn_dropout = nn.Dropout(config.dropout)
        # Dropout on the sublayer OUTPUT, before it re-enters the residual.
        self.resid_dropout = nn.Dropout(config.dropout)

        # The causal mask. Registered as a non-persistent buffer: it moves
        # to the GPU with .to(device) but is not written into checkpoints,
        # because it is a constant derived from block_size rather than
        # learned state.
        #
        # True on positions a query may attend to, i.e. the lower triangle
        # including the diagonal (position t may see itself and everything
        # before it).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x, return_attention=False):
        """
        Attend over x.

        Returns (output, attention) where attention is None unless
        return_attention is set. The Phase 2 attention analysis -- heatmaps,
        per-head entropy, mean attention distance -- needs the (B, h, T, T)
        probability tensor, and the only way to get it is to compute
        attention by hand, which is what this class does.
        """

        B, T, d = x.shape

        if T > self.causal_mask.shape[-1]:
            raise ValueError(
                f"Sequence length {T} exceeds block_size "
                f"{self.causal_mask.shape[-1]}: the learned position table "
                f"has no row for position {T - 1}."
            )

        # ---- project to queries, keys, values --------------------------------
        # (B, T, d) -> (B, T, 3d) -> three (B, T, d)
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # ---- split into heads -----------------------------------------------
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        #
        # The transpose is the step that makes the heads independent: after
        # it, the leading dimensions are (B, h) and every subsequent matmul
        # is batched over them, so head i never mixes with head j until the
        # output projection.
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # ---- scaled dot-product scores --------------------------------------
        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        #
        # scores[b, i, s, t] is how much query at position s wants the value
        # at position t, in head i.
        #
        # The 1/sqrt(d_k) is not cosmetic. q.k sums d_k products; for roughly
        # unit-variance independent entries that sum has variance d_k, so at
        # d_k = 64 the scores spread over roughly +/-8 before softmax.
        # Softmax on inputs that large saturates towards one-hot, and a
        # saturated softmax has near-zero gradient -- the attention layer
        # stops learning. Dividing by sqrt(d_k) restores unit variance and
        # keeps softmax in the range where it responds to its input.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # ---- causal mask -----------------------------------------------------
        # Applied BEFORE softmax, as an additive -inf on disallowed
        # positions: exp(-inf) = 0, so those keys receive exactly zero
        # probability and the remaining ones still sum to 1. Zeroing after
        # the softmax instead would leave the rows not summing to 1 and
        # would leak future information into the normaliser.
        #
        # Without this the model sees token t+1 while predicting token t+1.
        # Training loss collapses towards zero, and the model generates
        # nothing coherent, because at inference the future is not there.
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :T, :T],
            float("-inf"),
        )

        # ---- probabilities ---------------------------------------------------
        att = F.softmax(scores, dim=-1)          # (B, h, T, T), rows sum to 1
        att_weights = att if return_attention else None
        att = self.attn_dropout(att)

        # ---- weighted sum of values -----------------------------------------
        # (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        y = att @ v

        # ---- merge heads ----------------------------------------------------
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
        #
        # .contiguous() is required: transpose only rearranges strides, and
        # .view needs one contiguous buffer to reinterpret.
        y = y.transpose(1, 2).contiguous().view(B, T, d)

        # W_O mixes the head subspaces back together. Without it the heads'
        # outputs would stay in fixed disjoint slices of the residual
        # stream for the rest of the network.
        y = self.resid_dropout(self.proj(y))

        return y, att_weights


# =============================================================================
# FEED-FORWARD
# =============================================================================


class MLP(nn.Module):
    """
    Position-wise feed-forward network: d -> d_ff -> d, with GELU.

    Attention only MOVES information between positions -- every output is
    a convex combination of value vectors. It cannot apply a nonlinear
    transformation to a single token's representation. This sublayer is
    where that happens, independently and identically at every position,
    which is what "position-wise" means.

    The inner width is 4d (1536 for d=384), the standard ratio. Widening
    before the nonlinearity gives the layer room to represent many features
    and select among them; a d -> d -> d MLP is far weaker for the same
    reason a one-hidden-unit network is.

    GELU rather than ReLU: it is smooth, so it has non-zero gradient for
    small negative inputs, and units that go negative early in training can
    recover instead of dying.
    """

    def __init__(self, config):
        super().__init__()

        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)              # (B, T, d)    -> (B, T, d_ff)
        x = F.gelu(x)
        x = self.proj(x)            # (B, T, d_ff) -> (B, T, d)

        return self.dropout(x)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================


class Block(nn.Module):
    """
    One pre-norm Transformer block.

        x = x + Attn(LN(x))
        x = x + MLP(LN(x))

    Pre-norm rather than post-norm. In post-norm the normalisation sits ON
    the residual path -- x = LN(x + Attn(x)) -- so a gradient travelling
    from the loss to layer 0 is rescaled by a LayerNorm twenty-four times,
    and deep stacks either need careful warmup or diverge outright. In
    pre-norm the residual stream is an unbroken additive path from the
    embeddings to the final norm, and normalisation only touches the
    branch. Every GPT-2-and-later model is pre-norm for this reason.

    The residual connections themselves are what make twelve layers
    trainable: they give the gradient a route that skips each sublayer
    entirely, so the signal reaching early layers does not have to survive
    twelve matrix multiplications.
    """

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, return_attention=False):
        attn_out, att = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x, att


# =============================================================================
# THE MODEL
# =============================================================================


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Absent entirely in the ablation, rather than present and zeroed:
        # a table of zeros would still be a parameter tensor receiving
        # gradient, and the point of the ablation is a model with no
        # position information at all.
        self.position_embedding = (
            nn.Embedding(config.block_size, config.d_model)
            if config.use_positional_embedding
            else None
        )

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final normalisation before the output head. Without it the head
        # reads a residual stream whose scale has drifted over twelve
        # blocks, and the logits' scale drifts with it.
        self.ln_f = nn.LayerNorm(config.d_model)

        # No bias: with tied weights this is a similarity score between the
        # final hidden state and each token's embedding, and a per-token
        # bias would just relearn the unigram frequency the model already
        # has capacity for.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # GPT-2's scaled residual initialisation. The residual stream
        # accumulates 2 * n_layer additions, so at plain init its variance
        # grows with depth and the later blocks all write into an
        # already-large stream. Scaling down the tensors that WRITE into
        # the residual -- the two output projections per block -- by
        # 1/sqrt(2 * n_layer) keeps the stream's scale roughly constant
        # from layer 0 to layer 12 at initialisation.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        """N(0, 0.02) for weights, zeros for biases; LayerNorm left at 1/0."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- parameter accounting ----------------------------------------------

    def num_parameters(self, trainable_only=True):
        """
        Count parameters, counting a tied weight once.

        model.parameters() already deduplicates shared tensors by identity,
        so a tied lm_head is not double counted. Stated explicitly because
        the reported figure has to be defensible.
        """

        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)

        return sum(p.numel() for p in params)

    def parameter_breakdown(self):
        """Per-component parameter counts, for the report's budget table."""

        embedding = self.token_embedding.weight.numel()
        position = (
            self.position_embedding.weight.numel()
            if self.position_embedding is not None
            else 0
        )
        blocks = sum(p.numel() for p in self.blocks.parameters())
        final_norm = sum(p.numel() for p in self.ln_f.parameters())
        head = 0 if self.config.tie_weights else self.lm_head.weight.numel()

        return {
            "token_embedding": embedding,
            "position_embedding": position,
            "blocks": blocks,
            "block_each": blocks // self.config.n_layer,
            "final_layernorm": final_norm,
            "lm_head": head,
            "total": embedding + position + blocks + final_norm + head,
        }

    # ---- forward ------------------------------------------------------------

    def forward(self, idx, targets=None, return_attention=False):
        """
        Run the model.

        idx      (B, T) int64 token ids
        targets  (B, T) int64 next-token ids, or None at inference

        Returns (logits, loss, attentions):
            logits      (B, T, vocab_size)
            loss        scalar, or None when targets is None
            attentions  list of n_layer tensors (B, h, T, T), or None

        The targets are the inputs shifted one position left, and that
        shift is prepared by the data loader rather than here: a batch is
        tokens[i : i+T] with targets tokens[i+1 : i+T+1], so every one of
        the T positions supplies a training signal. Shifting inside the
        model instead would waste the last position of every window.
        """

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size "
                f"{self.config.block_size}."
            )

        # (B, T) -> (B, T, d)
        x = self.token_embedding(idx)

        if self.position_embedding is not None:
            # (T,) -> (T, d), broadcast over the batch. Positions are
            # absolute offsets within the window, so every window is
            # treated as if it began a document -- which is why a model
            # with learned absolute positions cannot be run on a longer
            # context than it was trained on.
            positions = torch.arange(T, device=idx.device)
            x = x + self.position_embedding(positions)

        x = self.embedding_dropout(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, att = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(att)

        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab) against (B*T,). Cross-entropy is
            # averaged over every position of every sequence, so the loss
            # is per token and comparable across batch sizes.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss, attentions

    # ---- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None):
        """
        Autoregressively extend idx by max_new_tokens.

        temperature divides the logits before softmax. Below 1 it sharpens
        the distribution towards the argmax; above 1 it flattens it towards
        uniform. temperature = 0 is treated as greedy decoding, which is
        the limit as temperature approaches 0 rather than a division that
        would be undefined.

        The context is cropped to the last block_size tokens because the
        position table has no row beyond it. This is the concrete cost of
        learned absolute positions: generation past 512 tokens works only
        because the window slides, and the model loses everything that
        falls off the left edge.
        """

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            window = idx[:, -self.config.block_size:]
            logits, _, _ = self(window)

            # Only the final position predicts the next token.
            logits = logits[:, -1, :]                  # (B, vocab_size)

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                    logits = logits.masked_fill(
                        logits < kth[:, [-1]], float("-inf")
                    )

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        if was_training:
            self.train()

        return idx


# =============================================================================
# CONSTRUCTION HELPERS
# =============================================================================


def build_model(config_path):
    """Build a model from a JSON config file. Returns (model, config)."""

    config = GPTConfig.from_json(config_path)

    return GPT(config), config
