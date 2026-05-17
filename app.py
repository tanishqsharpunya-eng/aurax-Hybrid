"""
AURAX-A1 Trainer — HuggingFace Space
Data lo HF se, train karo, download karo!
"""

import os, sys, math, time, json, logging, threading
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import gradio as gr

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("AURAX")

# Global state
training_running = False
training_log     = []
training_thread  = None
stop_flag        = False


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class AURAXConfig:
    vocab_size:     int   = 50272
    context_length: int   = 256
    n_embd:         int   = 256
    n_layer:        int   = 6
    n_head:         int   = 8
    n_kv_head:      int   = 8
    head_dim:       int   = field(init=False)
    ffn_mult:       int   = 4
    dropout:        float = 0.1
    bias:           bool  = False
    norm_eps:       float = 1e-5
    batch_size:     int   = 8
    learning_rate:  float = 3e-4
    min_lr:         float = 3e-5
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    warmup_iters:   int   = 50
    max_iters:      int   = 500
    eval_interval:  int   = 100
    eval_iters:     int   = 10
    val_split:      float = 0.1
    seed:           int   = 42

    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_head

    def estimate_params(self):
        total = (self.vocab_size * self.n_embd
                 + self.n_layer * 4 * self.n_embd ** 2
                 + self.n_layer * 3 * self.n_embd ** 2 * self.ffn_mult)
        if total >= 1e9: return f"~{total/1e9:.1f}B"
        return f"~{total/1e6:.0f}M"


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight.float()).to(x.dtype)

def precompute_rope(head_dim, max_len, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), theta)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f  = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * f).flatten(3).to(x.dtype)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.hd = cfg.n_head, cfg.head_dim
        self.q    = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k    = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.v    = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.o    = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd,  bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs):
        B, T, _ = x.shape
        q = apply_rope(self.q(x).view(B, T, self.nh, self.hd), freqs).transpose(1,2)
        k = apply_rope(self.k(x).view(B, T, self.nh, self.hd), freqs).transpose(1,2)
        v = self.v(x).view(B, T, self.nh, self.hd).transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.o(y.transpose(1,2).contiguous().view(B, T, -1)))

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.ffn_mult * cfg.n_embd
        self.fc1  = nn.Linear(cfg.n_embd, h, bias=False)
        self.fc2  = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1   = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2   = RMSNorm(cfg.n_embd)
        self.ffn  = FFN(cfg)
    def forward(self, x, freqs):
        x = x + self.attn(self.n1(x), freqs)
        x = x + self.ffn(self.n2(x))
        return x

class AURAXModel(nn.Module):
    def __init__(self, cfg: AURAXConfig):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f  = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.register_buffer("freqs", precompute_rope(cfg.head_dim, cfg.context_length), persistent=False)
        self.apply(self._init)
        total = sum(p.numel() for p in self.parameters())
        log.info(f"AURAX-A1 ready | Params: {total/1e6:.1f}M")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.drop(self.tok_emb(ids))
        for blk in self.blocks:
            x = blk(x, self.freqs[:T])
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

    def save(self, path, step=0, loss=0.0):
        torch.save({"step": step, "loss": loss,
                    "config": asdict(self.cfg),
                    "model":  self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        d  = ck["config"]; d.pop("head_dim", None)
        m  = cls(AURAXConfig(**d))
        m.load_state_dict(ck["model"])
        return m, ck

    def param_groups(self, wd):
        dec, nodec = [], []
        for _, p in self.named_parameters():
            (dec if p.dim() >= 2 else nodec).append(p)
        return [{"params": dec, "weight_decay": wd},
                {"params": nodec, "weight_decay": 0.0}]


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class TokenDataset(Dataset):
    def __init__(self, tokens, ctx):
        self.tokens = tokens
        self.ctx    = ctx
    def __len__(self):
        return max(0, len(self.tokens) - self.ctx)
    def __getitem__(self, i):
        chunk = self.tokens[i: i + self.ctx + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int64)),
                torch.from_numpy(chunk[1:].astype(np.int64)))


# ══════════════════════════════════════════════════════════════════
# LOG HELPERS
# ══════════════════════════════════════════════════════════════════

def push_log(msg):
    global training_log
    training_log.append(msg)
    log.info(msg)
    if len(training_log) > 300:
        training_log = training_log[-300:]

def get_log():
    return "\n".join(training_log)


# ══════════════════════════════════════════════════════════════════
# LR SCHEDULER
# ══════════════════════════════════════════════════════════════════

def get_lr(step, cfg):
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    if step > cfg.max_iters:
        return cfg.min_lr
    prog  = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ══════════════════════════════════════════════════════════════════
# TRAINING THREAD
# ══════════════════════════════════════════════════════════════════

def run_training(dataset_name, subset, split, text_col, max_samples,
                 n_layer, n_head, n_embd, context_length,
                 batch_size, max_iters, learning_rate, dropout):
    global training_running, stop_flag, training_log

    training_running = True
    stop_flag        = False
    training_log     = []

    push_log("🚀 AURAX-A1 Training Shuru!")
    push_log("=" * 50)

    cfg = AURAXConfig(
        n_layer=int(n_layer),   n_head=int(n_head),
        n_embd=int(n_embd),     context_length=int(context_length),
        batch_size=int(batch_size), max_iters=int(max_iters),
        learning_rate=float(learning_rate), dropout=float(dropout),
        warmup_iters=max(10, int(max_iters) // 10),
        eval_interval=max(50, int(max_iters) // 5),
    )

    push_log(f"📐 Model size: {cfg.estimate_params()}")
    push_log(f"⚙️  Layers:{cfg.n_layer} | Heads:{cfg.n_head} | Embd:{cfg.n_embd} | Ctx:{cfg.context_length}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    push_log(f"💻 Device: {device}")

    # ── Load HF dataset ──────────────────────────────────────────
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        push_log("❌ tiktoken nahi hai! requirements.txt check karo.")
        training_running = False
        return

    push_log(f"📥 Dataset load ho raha hai: {dataset_name}")
    try:
        ds_args = {"path": dataset_name, "trust_remote_code": True}
        if subset.strip(): ds_args["name"]  = subset.strip()
        ds_args["split"] = split.strip() if split.strip() else "train"
        ds = load_dataset(**ds_args)

        n = int(max_samples)
        if n > 0: ds = ds.select(range(min(n, len(ds))))
        push_log(f"✅ Loaded: {len(ds)} samples")

        texts = [str(row.get(text_col, "")).strip() for row in ds]
        texts = [t for t in texts if t]
        if not texts:
            push_log(f"❌ '{text_col}' column mein kuch nahi mila!")
            training_running = False
            return

        full_text = "\n\n".join(texts)
        push_log(f"📊 Characters: {len(full_text):,}")
        tokens = np.array(enc.encode(full_text), dtype=np.uint16)
        push_log(f"🔢 Tokens: {len(tokens):,}")

        min_tokens = cfg.context_length * 20
        if len(tokens) < min_tokens:
            push_log(f"❌ Bahut kam data! {len(tokens)} tokens mila, {min_tokens} chahiye.")
            push_log("💡 max_samples badha lo ya doosra dataset use karo.")
            training_running = False
            return

        sp   = int(len(tokens) * (1 - cfg.val_split))
        tds  = TokenDataset(tokens[:sp],  cfg.context_length)
        vds  = TokenDataset(tokens[sp:],  cfg.context_length)
        push_log(f"✅ Train:{len(tds):,} | Val:{len(vds):,} samples")

    except Exception as e:
        push_log(f"❌ Dataset error: {e}")
        training_running = False
        return

    tl = DataLoader(tds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    vl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    ti = iter(tl)

    # ── Model + Optimizer ────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    model = AURAXModel(cfg).to(device)
    opt   = torch.optim.AdamW(
        model.param_groups(cfg.weight_decay),
        lr=cfg.learning_rate, betas=(0.9, 0.95)
    )

    push_log("=" * 50)
    push_log("🏃 Training loop chalu...")
    push_log("=" * 50)

    best_val = float("inf")
    t0 = time.time()
    cur_loss = 0.0

    for step in range(1, cfg.max_iters + 1):
        if stop_flag:
            push_log("⛔ Training rok di!")
            break

        lr = get_lr(step, cfg)
        for pg in opt.param_groups: pg["lr"] = lr

        try:
            x, y = next(ti)
        except StopIteration:
            ti = iter(tl)
            x, y = next(ti)

        x, y = x.to(device), y.to(device)
        model.train()
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        cur_loss = loss.item()

        if step % 10 == 0:
            dt = time.time() - t0
            tps = (10 * cfg.batch_size * cfg.context_length) / max(dt, 1e-6)
            push_log(f"Step {step:5d}/{cfg.max_iters} | Loss: {cur_loss:.4f} | LR: {lr:.2e} | {tps:.0f} tok/s")
            t0 = time.time()

        if step % cfg.eval_interval == 0:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(vl):
                    if i >= cfg.eval_iters: break
                    _, vl_ = model(vx.to(device), vy.to(device))
                    vlosses.append(vl_.item())
            vl_ = float(np.mean(vlosses)) if vlosses else 999
            push_log(f"{'─'*40}")
            push_log(f"📊 EVAL Step {step} | Val Loss: {vl_:.4f}")
            if vl_ < best_val:
                best_val = vl_
                model.save("checkpoints/best_model.pt", step=step, loss=vl_)
                push_log(f"💾 Best model saved! Val Loss: {vl_:.4f}")
            push_log(f"{'─'*40}")

    model.save("checkpoints/final_model.pt", step=step, loss=cur_loss)
    push_log("=" * 50)
    push_log("✅ Training Complete!")
    push_log(f"🏆 Best Val Loss: {best_val:.4f}")
    push_log("📥 Neeche se model download karo!")
    push_log("=" * 50)
    training_running = False


# ══════════════════════════════════════════════════════════════════
# GRADIO CALLBACKS
# ══════════════════════════════════════════════════════════════════

def start_training(*args):
    global training_thread, training_running
    if training_running:
        return "⚠️ Training pehle se chal rahi hai!"
    training_thread = threading.Thread(target=run_training, args=args, daemon=True)
    training_thread.start()
    return "🚀 Training shuru! Log refresh karo."

def stop_training():
    global stop_flag
    stop_flag = True
    return "⛔ Stop signal bheja..."

def refresh_log():
    return get_log()

def get_download_file():
    p = "checkpoints/best_model.pt"
    return p if os.path.exists(p) else None

@torch.no_grad()
def test_generate(prompt, max_tok, temperature, top_k):
    p = "checkpoints/best_model.pt"
    if not os.path.exists(p):
        return "❌ Pehle train karo!"
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        return "❌ tiktoken nahi hai"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = AURAXModel.load(p, str(device))
    model = model.to(device).eval()
    ctx = model.cfg.context_length

    ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(int(max_tok)):
        inp = ids if ids.size(1) <= ctx else ids[:, -ctx:]
        logits, _ = model(inp)
        logits = logits[:, -1, :] / max(float(temperature), 1e-6)
        if top_k > 0:
            tv = torch.topk(logits, min(int(top_k), logits.size(-1))).values
            logits[logits < tv[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        ids = torch.cat([ids, nxt], 1)
    return enc.decode(ids[0].tolist())


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════

css = """
.header{text-align:center;padding:24px;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:14px;margin-bottom:16px}
.header h1{color:#fff;font-size:2.4em;margin:0;letter-spacing:2px}
.header p{color:#9b9fc4;margin:6px 0 0}
.logbox textarea{font-family:monospace!important;font-size:12px!important;background:#0d1117!important;color:#58a6ff!important}
"""

with gr.Blocks(title="AURAX-A1 Trainer", theme=gr.themes.Soft(), css=css) as demo:

    gr.HTML("""<div class="header">
      <h1>⚡ AURAX-A1</h1>
      <p>Apna GPT-style AI model train karo — HuggingFace dataset se!</p>
    </div>""")

    with gr.Tabs():

        # ── TAB 1: TRAIN ──────────────────────────────────────────
        with gr.TabItem("🚀 Train"):

            gr.Markdown("### 📦 HuggingFace Dataset")
            gr.Markdown(
                "Popular: `wikitext` (subset: `wikitext-2-raw-v1`, col: `text`) | "
                "`ag_news` (col: `text`) | `daily_dialog` (col: `dialog`)"
            )

            with gr.Row():
                inp_ds      = gr.Textbox(label="Dataset Name",      value="wikitext")
                inp_subset  = gr.Textbox(label="Subset (optional)",  value="wikitext-2-raw-v1")
                inp_split   = gr.Textbox(label="Split",              value="train")
                inp_col     = gr.Textbox(label="Text Column",        value="text")
                inp_samples = gr.Number( label="Max Samples (0=all)", value=5000, precision=0)

            gr.Markdown("### 🧠 Model Size")
            with gr.Row():
                inp_layers = gr.Slider(2,  24,   value=6,   step=2,  label="Layers")
                inp_heads  = gr.Slider(2,  16,   value=8,   step=2,  label="Attention Heads")
                inp_embd   = gr.Slider(64, 1024, value=256, step=64, label="Embedding Dim")
                inp_ctx    = gr.Slider(64, 1024, value=256, step=64, label="Context Length")

            gr.Markdown("### ⚙️ Training")
            with gr.Row():
                inp_bs   = gr.Slider(1,  32,    value=8,    step=1,     label="Batch Size")
                inp_iter = gr.Slider(100,5000,  value=500,  step=100,   label="Max Iterations")
                inp_lr   = gr.Slider(1e-5,1e-3, value=3e-4, step=1e-5,  label="Learning Rate")
                inp_drop = gr.Slider(0.0, 0.5,  value=0.1,  step=0.05,  label="Dropout")

            with gr.Row():
                btn_start = gr.Button("🚀 Train Shuru Karo!", variant="primary", scale=3)
                btn_stop  = gr.Button("⛔ Rok Do",           variant="stop",    scale=1)

            status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### 📋 Live Log")
            logbox = gr.Textbox(label="", lines=18, interactive=False, elem_classes=["logbox"])

            with gr.Row():
                btn_refresh  = gr.Button("🔄 Refresh Log")
                btn_download = gr.Button("📥 Model Download")
            file_out = gr.File(label="Download Model")

            all_inputs = [inp_ds, inp_subset, inp_split, inp_col, inp_samples,
                          inp_layers, inp_heads, inp_embd, inp_ctx,
                          inp_bs, inp_iter, inp_lr, inp_drop]

            btn_start.click(start_training, inputs=all_inputs, outputs=status)
            btn_stop.click(stop_training, outputs=status)
            btn_refresh.click(refresh_log, outputs=logbox)
            btn_download.click(get_download_file, outputs=file_out)

        # ── TAB 2: TEST ──────────────────────────────────────────
        with gr.TabItem("💬 Test Karo"):
            gr.Markdown("### ✍️ Trained model se text generate karo")
            prompt_in = gr.Textbox(label="Prompt", value="Once upon a time", lines=3)
            with gr.Row():
                sl_tok  = gr.Slider(10,  500,  value=100, step=10,  label="Max Tokens")
                sl_temp = gr.Slider(0.1, 2.0,  value=0.8, step=0.1, label="Temperature")
                sl_topk = gr.Slider(0,   100,  value=40,  step=5,   label="Top-K")
            btn_gen  = gr.Button("⚡ Generate!", variant="primary")
            out_text = gr.Textbox(label="Generated Text", lines=10, interactive=False)
            btn_gen.click(test_generate, inputs=[prompt_in, sl_tok, sl_temp, sl_topk], outputs=out_text)

        # ── TAB 3: GUIDE ─────────────────────────────────────────
        with gr.TabItem("📖 Guide"):
            gr.Markdown("""
## Quick Start

### Step 1 — Dataset
| Dataset | Subset | Text Column |
|---------|--------|-------------|
| `wikitext` | `wikitext-2-raw-v1` | `text` |
| `ag_news` | _(khaali)_ | `text` |
| `daily_dialog` | _(khaali)_ | `dialog` |
| `openwebtext` | _(khaali)_ | `text` |

### Step 2 — Size Choose Karo
| Size | Layers | Heads | Embd | Params |
|------|--------|-------|------|--------|
| Tiny | 4 | 4 | 128 | ~10M |
| Small ✅ | 6 | 8 | 256 | ~40M |
| Medium | 12 | 8 | 512 | ~150M |

### Step 3 — Download → PC pe Chalao
```python
# PC pe (aurax_a1.py ke saath):
from aurax_a1 import AURAXModel, generate_text
import tiktoken, torch

model, _ = AURAXModel.load("best_model.pt", device="cpu")
enc = tiktoken.get_encoding("gpt2")
prompt = torch.tensor(enc.encode("Hello"), dtype=torch.long).unsqueeze(0)
out = generate_text(model, prompt, max_new_tokens=100)
print(enc.decode(out[0].tolist()))
```

### Tips
- Val loss **< 4** = model kuch seekh raha hai ✅
- Val loss **< 3** = achha model 🔥
- **Zyada data = better output** — max_samples badha lo
- HF Space pe free T4 GPU milta hai — fast training!
            """)

demo.launch()
