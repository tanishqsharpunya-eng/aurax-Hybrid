"""
+==================================================================+
|                        AURAX-A1 Model                           |
|          GPT-style Decoder-only Transformer from Scratch        |
|                                                                  |
|  Architecture:                                                   |
|   * 24 Transformer Layers                                       |
|   * 16 Attention Heads                                          |
|   * 2048 Embedding Dimension                                    |
|   * 2048 Context Length                                         |
|   * 50272 Vocabulary Size                                       |
|   * RMSNorm (instead of LayerNorm)                              |
|   * RoPE Positional Embeddings                                  |
|   * Flash Attention (auto-detected)                             |
|   * GELU Activation                                             |
+==================================================================+

Usage:
    python aurax_a1.py

    Ya generate karne ke liye:
    from aurax_a1 import AURAXModel, AURAXConfig, generate_text
"""

import os
import sys
import math
import time
import json
import logging
import inspect
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Windows CP1252 fix: stdout ko UTF-8 force karo
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/aurax_train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AURAX-A1")


# ===============================================================
#  SECTION 1: CONFIGURATION
# ===============================================================

@dataclass
class AURAXConfig:
    """
    AURAX-A1 Model Configuration
    Yahan se saare hyperparameters control hote hain.
    """

    # -- Model Architecture --------------------------------------
    model_name: str = "AURAX-A1"
    vocab_size: int = 50272          # GPT-2 tokenizer vocab size
    context_length: int = 2048       # Maximum sequence length
    n_embd: int = 2048               # Embedding / hidden dimension
    n_layer: int = 24                # Number of transformer blocks
    n_head: int = 16                 # Number of attention heads
    n_kv_head: int = 16              # KV heads (same as n_head = MHA; less = GQA)
    head_dim: int = field(init=False)  # Auto-computed: n_embd // n_head
    ffn_mult: int = 4                # FFN hidden = ffn_mult * n_embd
    dropout: float = 0.0             # Dropout (0 during inference)
    bias: bool = False               # Bias in linear layers (GPT-3 style: False)
    norm_eps: float = 1e-5           # RMSNorm epsilon

    # -- Training Hyperparameters ---------------------------------
    batch_size: int = 4              # Batch size (adjust for VRAM)
    grad_accum_steps: int = 8        # Gradient accumulation steps
    learning_rate: float = 3e-4      # Peak learning rate
    min_lr: float = 3e-5             # Minimum LR (cosine decay end)
    weight_decay: float = 0.1        # AdamW weight decay
    beta1: float = 0.9               # AdamW beta1
    beta2: float = 0.95              # AdamW beta2
    grad_clip: float = 1.0           # Gradient clipping
    warmup_iters: int = 2000         # LR warmup steps
    max_iters: int = 100_000         # Total training iterations
    decay_lr: bool = True            # Cosine LR decay
    eval_interval: int = 500         # Steps between evaluations
    save_interval: int = 1000        # Steps between checkpoints
    eval_iters: int = 100            # Eval batches for loss estimate

    # -- Data -----------------------------------------------------
    data_path: str = "data/train.txt"
    val_split: float = 0.1           # 10% for validation

    # -- System ---------------------------------------------------
    device: str = "auto"             # "auto", "cuda", "cpu", "mps"
    dtype: str = "bfloat16"          # "float32", "float16", "bfloat16"
    compile_model: bool = False      # torch.compile (PyTorch 2.0+)
    seed: int = 42

    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

    def save(self, path: str):
        """Config ko JSON mein save karo"""
        d = {k: v for k, v in asdict(self).items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        log.info(f"Config saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "AURAXConfig":
        """JSON se config load karo"""
        with open(path) as f:
            d = json.load(f)
        # head_dim is auto-computed, remove if present
        d.pop("head_dim", None)
        return cls(**d)

    def __repr__(self):
        params_est = self._estimate_params()
        return (
            f"\n{'='*55}\n"
            f"  AURAX-A1 Configuration\n"
            f"{'-'*55}\n"
            f"  Layers       : {self.n_layer}\n"
            f"  Heads        : {self.n_head}\n"
            f"  Embedding    : {self.n_embd}\n"
            f"  Context      : {self.context_length}\n"
            f"  Vocab        : {self.vocab_size}\n"
            f"  Est. Params  : {params_est}\n"
            f"  Batch size   : {self.batch_size} x {self.grad_accum_steps} accum\n"
            f"  Max iters    : {self.max_iters:,}\n"
            f"{'='*55}"
        )

    def _estimate_params(self) -> str:
        """Rough parameter count estimate"""
        embd = self.vocab_size * self.n_embd
        attn = self.n_layer * (4 * self.n_embd * self.n_embd)
        ffn  = self.n_layer * (3 * self.n_embd * self.n_embd * self.ffn_mult)  # SwiGLU style
        total = embd + attn + ffn
        if total >= 1e9:
            return f"~{total/1e9:.1f}B"
        return f"~{total/1e6:.0f}M"


# ===============================================================
#  SECTION 2: BUILDING BLOCKS
# ===============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    LayerNorm se faster hai — no mean subtraction, no bias.
    LLaMA / Mistral / AURAX sab RMSNorm use karte hain.

    Formula: x / RMS(x) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))  ->  x / RMS
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 mein compute karo precision ke liye, phir original dtype pe wapas
        return (self._norm(x.float()) * self.weight.float()).to(x.dtype)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Rotary Position Embedding (RoPE) frequencies precompute karo.
    RoPE: position information directly attention mein inject hoti hai
    (not added, but ROTATED) -> length extrapolation better hoti hai.

    Returns complex tensor of shape (max_seq_len, head_dim//2)
    """
    # Theta = 1 / (base ^ (2i / d)) for i in [0, d/2)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Position indices
    seq = torch.arange(max_seq_len).float()
    # Outer product: (max_seq_len, head_dim//2)
    freqs = torch.outer(seq, theta)
    # Complex numbers mein convert karo (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    RoPE apply karo query / key tensors pe.

    x shape      : (B, T, n_head, head_dim)
    freqs_cis    : (T, head_dim // 2)
    Returns      : rotated x, same shape
    """
    # (B, T, n_head, head_dim) -> complex pairs: (B, T, n_head, head_dim//2)
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x_c = torch.view_as_complex(x_r)
    # Broadcast freqs: (1, T, 1, head_dim//2)
    freqs = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    # Rotate
    x_out = torch.view_as_real(x_c * freqs).flatten(3)
    return x_out.to(x.dtype)


# ===============================================================
#  SECTION 3: MULTI-HEAD ATTENTION (with Flash Attention)
# ===============================================================

class AURAXAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.

    Features:
    * Flash Attention (F.scaled_dot_product_attention) — automatic CUDA kernel
    * Standard fallback agar Flash Attention available nahi
    * Causal masking (decoder-only: future tokens nahi dekhta)
    * RoPE positional embeddings
    """

    def __init__(self, config: AURAXConfig):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.head_dim
        self.n_embd    = config.n_embd
        self.dropout   = config.dropout

        # Grouped Query Attention support
        self.n_rep = self.n_head // self.n_kv_head  # repetitions for GQA

        # Q, K, V projections
        self.q_proj = nn.Linear(config.n_embd, config.n_head    * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        # Output projection
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash Attention available hai?
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            log.warning("Flash Attention unavailable — using standard attention (slower)")
            # Causal mask register karo (upper triangle = -inf)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.context_length, config.context_length))
                .view(1, 1, config.context_length, config.context_length),
            )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # -- Project to Q, K, V ----------------------------------
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)

        # Reshape to (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head,    self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # -- Apply RoPE ------------------------------------------
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # -- GQA: K, V heads repeat karo -------------------------
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # -- Transpose: (B, n_head, T, head_dim) -----------------
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -- Attention --------------------------------------------
        if self.flash:
            # Flash Attention — fused CUDA kernel, memory efficient
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # Causal masking auto-applied
            )
        else:
            # Standard attention fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            # Causal mask apply karo
            scores = scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
            scores = self.attn_dropout(scores)
            y = torch.matmul(scores, v)

        # -- Merge heads and project ------------------------------
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.resid_dropout(self.o_proj(y))
        return y


# ===============================================================
#  SECTION 4: FEED-FORWARD NETWORK (SwiGLU / GELU)
# ===============================================================

class AURAXFFN(nn.Module):
    """
    Feed-Forward Network with GELU activation.

    Standard GPT-style: Linear -> GELU -> Linear
    FFN hidden size = 4 x n_embd (configurable via ffn_mult)

    Alternate option: SwiGLU (LLaMA style) — commented below.
    """

    def __init__(self, config: AURAXConfig):
        super().__init__()
        hidden_dim = config.ffn_mult * config.n_embd

        self.fc1  = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.fc2  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.act  = nn.GELU()  # GELU activation (GPT-2/3 style)
        self.drop = nn.Dropout(config.dropout)

        # -- SwiGLU alternate (uncomment agar use karna ho) ------
        # SwiGLU = gate mechanism -> better performance
        # self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        # forward mein: x = self.fc2(F.silu(self.fc1(x)) * self.gate(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)   # Project up
        x = self.act(x)   # GELU non-linearity
        x = self.drop(x)  # Dropout
        x = self.fc2(x)   # Project down
        return x


# ===============================================================
#  SECTION 5: TRANSFORMER BLOCK
# ===============================================================

class AURAXBlock(nn.Module):
    """
    Single Transformer Decoder Block.

    Structure (Pre-norm style — better training stability):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    RoPE frequencies bahar se pass hoti hain (buffer efficiency ke liye).
    """

    def __init__(self, config: AURAXConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn  = AURAXAttention(config)
        self.norm2 = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn   = AURAXFFN(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Residual + Attention (pre-norm)
        x = x + self.attn(self.norm1(x), freqs_cis)
        # Residual + FFN (pre-norm)
        x = x + self.ffn(self.norm2(x))
        return x


# ===============================================================
#  SECTION 6: AURAX-A1 FULL MODEL
# ===============================================================

class AURAXModel(nn.Module):
    """
    AURAX-A1: Complete GPT-style Decoder-only Transformer

    Components:
    1. Token Embedding (vocab_size -> n_embd)
    2. N x Transformer Blocks (Attention + FFN)
    3. Final RMSNorm
    4. Output Linear projection (n_embd -> vocab_size)

    Weight tying: Token embedding weights = Output projection weights
    (same as GPT-2, saves ~100M parameters)
    """

    def __init__(self, config: AURAXConfig):
        super().__init__()
        self.config = config

        # -- Token Embedding --------------------------------------
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # -- Transformer Blocks -----------------------------------
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([AURAXBlock(config) for _ in range(config.n_layer)])

        # -- Final Norm + Output ----------------------------------
        self.norm_f  = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # -- Weight Tying -----------------------------------------
        # Token embedding aur output projection ek hi weight share karte hain
        self.lm_head.weight = self.tok_emb.weight

        # -- RoPE Frequencies Buffer ------------------------------
        freqs = precompute_rope_freqs(config.head_dim, config.context_length)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # -- Weight Initialization --------------------------------
        self.apply(self._init_weights)
        # Special scaling for residual projections (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        total = sum(p.numel() for p in self.parameters())
        log.info(f"AURAX-A1 initialized | Parameters: {total/1e6:.1f}M")

    def _init_weights(self, module: nn.Module):
        """
        Weight initialization (mean=0, std=0.02) — GPT-2 style.
        Embeddings: normal(0, 0.02)
        Linear: normal(0, 0.02), bias -> zero
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token indices
            targets:   (B, T) token indices for loss computation (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, \
            f"Sequence length {T} > max context {self.config.context_length}"

        # -- Token Embedding --------------------------------------
        x = self.tok_emb(input_ids)   # (B, T, n_embd)
        x = self.drop(x)

        # -- RoPE freqs slice (current sequence length) -----------
        freqs = self.freqs_cis[:T]

        # -- Transformer Blocks -----------------------------------
        for block in self.blocks:
            x = block(x, freqs)

        # -- Final Norm -------------------------------------------
        x = self.norm_f(x)

        # -- Output Projection ------------------------------------
        if targets is not None:
            # Training: full sequence logits + loss
            logits = self.lm_head(x)                  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),      # (B*T, vocab_size)
                targets.view(-1),                      # (B*T,)
                ignore_index=-1,
            )
        else:
            # Inference: sirf last token ke logits (efficiency)
            logits = self.lm_head(x[:, [-1], :])      # (B, 1, vocab_size)
            loss = None

        return logits, loss

    # -------------------------------------------------------------
    # Checkpoint Save / Load
    # -------------------------------------------------------------

    def save_checkpoint(self, path: str, step: int, optimizer=None, loss: float = 0.0):
        """Model checkpoint save karo"""
        ckpt = {
            "step":       step,
            "model":      self.state_dict(),
            "loss":       loss,
            "config":     asdict(self.config),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        torch.save(ckpt, path)
        log.info(f"Checkpoint saved -> {path} (step {step}, loss {loss:.4f})")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> Tuple["AURAXModel", dict]:
        """Checkpoint se model load karo"""
        ckpt = torch.load(path, map_location=device)
        cfg_dict = ckpt["config"]
        cfg_dict.pop("head_dim", None)
        config = AURAXConfig(**cfg_dict)
        model = cls(config)
        model.load_state_dict(ckpt["model"])
        log.info(f"Checkpoint loaded ← {path} (step {ckpt['step']}, loss {ckpt['loss']:.4f})")
        return model, ckpt

    def param_groups(self, weight_decay: float):
        """
        Optimizer param groups:
        - Weight decay ONLY 2D params pe (weights)
        - No decay on 1D params (biases, norms)
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


# ===============================================================
#  SECTION 7: TEXT GENERATION
# ===============================================================

@torch.no_grad()
def generate_text(
    model: AURAXModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """
    Autoregressive text generation with:
    * Temperature scaling  — creativity control karo
    * Top-K sampling       — top K tokens mein se sample karo
    * Top-P (nucleus)      — probability mass ke hisaab se filter
    * Repetition penalty   — same tokens repeat hone se rokta hai

    Args:
        model          : AURAXModel (eval mode mein hona chahiye)
        input_ids      : (1, T) tensor of prompt token ids
        max_new_tokens : Kitne tokens generate karne hain
        temperature    : >1 = more random, <1 = more focused
        top_k          : 0 = disabled
        top_p          : 1.0 = disabled
        repetition_penalty: 1.0 = disabled, >1 = penalize repeats

    Returns:
        (1, T + max_new_tokens) generated token ids
    """
    model.eval()
    ctx = model.config.context_length

    for _ in range(max_new_tokens):
        # Context window ke andar rakho
        ids_cond = input_ids if input_ids.size(1) <= ctx else input_ids[:, -ctx:]

        # Forward pass
        logits, _ = model(ids_cond)     # (1, 1, vocab_size)
        logits = logits[:, -1, :]       # Last token: (1, vocab_size)

        # -- Repetition Penalty -----------------------------------
        if repetition_penalty != 1.0:
            for token_id in input_ids[0].tolist():
                logits[0, token_id] /= repetition_penalty

        # -- Temperature ------------------------------------------
        logits = logits / temperature

        # -- Top-K ------------------------------------------------
        if top_k > 0:
            topk_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
            logits[logits < topk_vals[:, [-1]]] = float("-inf")

        # -- Top-P (Nucleus) --------------------------------------
        if top_p < 1.0:
            probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
            cumulative = torch.cumsum(probs_sorted, dim=-1)
            # Top-p ke baad waale tokens remove karo
            remove_mask = cumulative - probs_sorted > top_p
            probs_sorted[remove_mask] = 0.0
            probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
            # Original order mein wapas
            probs = torch.zeros_like(logits).scatter_(1, sorted_idx, probs_sorted)
        else:
            probs = F.softmax(logits, dim=-1)

        # -- Sample -----------------------------------------------
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


# ===============================================================
#  SECTION 8: DATASET
# ===============================================================

class TextDataset(Dataset):
    """
    Simple character/token level dataset.
    text.txt ko tokenize karke training windows banata hai.
    """

    def __init__(self, tokens: np.ndarray, context_length: int):
        self.tokens = tokens
        self.ctx    = context_length

    def __len__(self):
        return max(0, len(self.tokens) - self.ctx)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.ctx + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def load_data(config: AURAXConfig):
    """
    Data file load karo aur train/val split banao.
    Tiktoken (GPT-2 tokenizer) use karta hai.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        log.error("tiktoken install karo: pip install tiktoken")
        sys.exit(1)

    if not os.path.exists(config.data_path):
        log.warning(f"Data file nahi mila: {config.data_path}")
        log.info("Sample data bana raha hun (demo ke liye)...")
        # Sample data - enough tokens for train + val split
        sentences = [
            "The quick brown fox jumps over the lazy dog. ",
            "Artificial intelligence is transforming the world rapidly. ",
            "Once upon a time in a land far away, there lived a wise AI. ",
            "The neural network learned to predict the next token in sequence. ",
            "Language models are trained on vast amounts of text data. ",
            "Deep learning has revolutionized natural language processing. ",
            "The transformer architecture uses self-attention mechanisms. ",
            "Training large models requires significant computational resources. ",
        ]
        sample = "".join(sentences * 3000)  # ~200K+ chars, enough for val split
        os.makedirs(os.path.dirname(config.data_path), exist_ok=True)
        with open(config.data_path, "w") as f:
            f.write(sample)

    log.info(f"Data load ho raha hai: {config.data_path}")
    with open(config.data_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    log.info(f"Text length: {len(text):,} characters")
    tokens = np.array(enc.encode(text), dtype=np.uint16)
    log.info(f"Tokens: {len(tokens):,}")

    # Train / Val split
    split = int(len(tokens) * (1 - config.val_split))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    train_ds = TextDataset(train_tokens, config.context_length)
    val_ds   = TextDataset(val_tokens,   config.context_length)

    # Agar val set bahut chhota hai, train set se use karo
    if len(val_ds) < config.batch_size:
        log.warning("Val set bahut chhota hai! Train set ko val ke liye bhi use kar raha hun.")
        log.warning("Achhe results ke liye data/train.txt mein zyada data daalo (min 1MB).")
        val_ds = TextDataset(train_tokens, config.context_length)

    log.info(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    return train_ds, val_ds, enc


# ===============================================================
#  SECTION 9: LEARNING RATE SCHEDULER
# ===============================================================

def get_lr(step: int, config: AURAXConfig) -> float:
    """
    Cosine learning rate decay with linear warmup.
    GPT-3 paper mein yahi use kiya gaya hai.

    Phase 1: Linear warmup (0 -> max_lr)
    Phase 2: Cosine decay (max_lr -> min_lr)
    """
    if not config.decay_lr:
        return config.learning_rate

    # Warmup phase
    if step < config.warmup_iters:
        return config.learning_rate * step / config.warmup_iters

    # Training complete hone ke baad
    if step > config.max_iters:
        return config.min_lr

    # Cosine decay
    progress = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ===============================================================
#  SECTION 10: HARDWARE DETECTION
# ===============================================================

def detect_hardware(config: AURAXConfig) -> Tuple[torch.device, torch.dtype]:
    """
    GPU / CPU auto-detect karo aur best settings choose karo.
    """
    # Device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # VRAM check (CUDA)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")

        if vram_gb < 6:
            log.warning("VRAM < 6GB — batch_size 2 ya gradient_accum badha lo")

    # dtype
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float32)

    # bfloat16 CPU support check
    if dtype == torch.bfloat16 and device.type == "cpu":
        log.warning("bfloat16 CPU pe slow hai — float32 pe switch kar raha hun")
        dtype = torch.float32

    log.info(f"Device: {device} | dtype: {dtype}")
    return device, dtype


# ===============================================================
#  SECTION 11: TRAINING LOOP
# ===============================================================

@torch.no_grad()
def estimate_loss(model, val_loader, config, device, ctx_manager):
    """Val set pe loss estimate karo (gradient nahi chahiye)"""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= config.eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with ctx_manager:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train(config: AURAXConfig):
    """
    ==========================================
      AURAX-A1 Main Training Loop
    ==========================================

    Steps:
    1. Hardware detect karo
    2. Data load karo
    3. Model banao (ya checkpoint se resume karo)
    4. Optimizer + scaler setup karo
    5. Training loop chalao
    6. Har eval_interval pe validation loss print karo
    7. Har save_interval pe checkpoint save karo
    """

    # -- Reproducibility ------------------------------------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    log.info(str(config))

    # -- Hardware -------------------------------------------------
    device, dtype = detect_hardware(config)

    # Mixed precision context manager
    use_amp = (dtype in [torch.float16, torch.bfloat16]) and device.type == "cuda"
    ctx_manager = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else torch.no_grad().__class__()
    if not use_amp:
        # Dummy context manager
        import contextlib
        ctx_manager = contextlib.nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # -- Data -----------------------------------------------------
    train_ds, val_ds, enc = load_data(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_iter = iter(train_loader)

    # -- Model -----------------------------------------------------
    start_step = 0
    best_val_loss = float("inf")

    # Checkpoint se resume karo?
    resume_path = "checkpoints/latest.pt"
    if os.path.exists(resume_path):
        log.info(f"Checkpoint mila! Resume karo ya fresh start? (r/f): ", end="")
        try:
            choice = input().strip().lower()
        except EOFError:
            choice = "r"

        if choice == "r":
            model, ckpt = AURAXModel.load_checkpoint(resume_path, device=str(device))
            model = model.to(device)
            start_step = ckpt["step"]
            best_val_loss = ckpt.get("loss", float("inf"))
        else:
            model = AURAXModel(config).to(device)
    else:
        model = AURAXModel(config).to(device)

    # torch.compile (PyTorch 2.0+)
    if config.compile_model:
        log.info("torch.compile chal raha hai (first run slow hoga)...")
        model = torch.compile(model)

    # -- Optimizer -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.param_groups(config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=(device.type == "cuda"),  # Fused AdamW (faster on CUDA)
    )

    # -- Training Loop ---------------------------------------------
    log.info("="*55)
    log.info("  AURAX-A1 Training Shuru! [LAUNCH]")
    log.info("="*55)

    t0 = time.time()
    step = start_step
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < config.max_iters:
        # -- Learning Rate Update -------------------------------
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # -- Gradient Accumulation ------------------------------
        for micro_step in range(config.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with ctx_manager:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()

        # -- Gradient Clip + Step -------------------------------
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # -- Logging -------------------------------------------
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (10 * config.batch_size * config.grad_accum_steps
                              * config.context_length) / dt
            log.info(
                f"Step {step:6d}/{config.max_iters} | "
                f"Loss: {running_loss/10:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )
            running_loss = 0.0

        # -- Evaluation ----------------------------------------
        if step % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config, device, ctx_manager)
            log.info(f"{'-'*45}")
            log.info(f"  EVAL | Step {step} | Val Loss: {val_loss:.4f}")
            log.info(f"{'-'*45}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(
                    "checkpoints/best_model.pt",
                    step=step,
                    optimizer=optimizer,
                    loss=val_loss,
                )

        # -- Checkpoint Save -----------------------------------
        if step % config.save_interval == 0:
            model.save_checkpoint(
                "checkpoints/latest.pt",
                step=step,
                optimizer=optimizer,
                loss=running_loss,
            )
            model.save_checkpoint(
                f"checkpoints/iter_{step:07d}.pt",
                step=step,
                loss=running_loss,
            )

    log.info("Training complete! [DONE]")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info("Best model: checkpoints/best_model.pt")


# ===============================================================
#  SECTION 12: QUICK DEMO / INFERENCE
# ===============================================================

def demo_inference():
    """
    Ek untrained model se text generate karo (random, sirf architecture test).
    Proper text ke liye pehle train karo.
    """
    log.info("Demo mode: untrained model se generate kar raha hun...")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        log.error("pip install tiktoken")
        return

    # Chhota config demo ke liye
    config = AURAXConfig(
        n_layer=4, n_head=4, n_embd=256, context_length=128,
        batch_size=1, max_iters=100
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AURAXModel(config).to(device)
    model.eval()

    prompt = "The future of artificial intelligence"
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    log.info(f"Prompt: '{prompt}'")
    out = generate_text(model, tokens, max_new_tokens=50, temperature=0.8)
    generated = enc.decode(out[0].tolist())
    log.info(f"Generated (untrained — random):\n{generated}")
    log.info("Train karo achha text ke liye!")


# ===============================================================
#  MAIN ENTRY POINT
# ===============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AURAX-A1 Training Script")
    parser.add_argument("--demo",         action="store_true", help="Demo inference (no training)")
    parser.add_argument("--n_layer",      type=int,   default=24)
    parser.add_argument("--n_head",       type=int,   default=16)
    parser.add_argument("--n_embd",       type=int,   default=2048)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--max_iters",    type=int,   default=100_000)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--data_path",    type=str,   default="data/train.txt")
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--dtype",        type=str,   default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile",      action="store_true", help="torch.compile use karo")
    args = parser.parse_args()

    if args.demo:
        demo_inference()
    else:
        config = AURAXConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            learning_rate=args.lr,
            data_path=args.data_path,
            device=args.device,
            dtype=args.dtype,
            compile_model=args.compile,
        )
        config.save("checkpoints/config.json")
        train(config)
