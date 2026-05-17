---
title: AURAX-A1 Trainer
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# ⚡ AURAX-A1 — Train Your Own GPT-style Model

Apna **AURAX-A1** model train karo HuggingFace datasets se — bilkul free!

## Features
- 🗄️ HuggingFace se koi bhi dataset directly load karo
- 🧠 Model size customize karo (layers, heads, embedding)
- 📊 Live training log dekho
- 💾 Best model auto-save hota hai
- 📥 Trained model download karo
- 💬 Directly Space mein test karo

## How to Use
1. **Data tab** mein dataset name daalo (e.g. `wikitext`)
2. Model size choose karo
3. **Train Shuru Karo!** dabao
4. Log refresh karo — loss dekhte raho
5. Training ke baad **Download** karo
6. Apne PC pe `aurax_a1.py` ke saath chalao!

## Architecture
- GPT-style Decoder-only Transformer
- RMSNorm (LayerNorm se fast)
- RoPE Positional Embeddings  
- Flash Attention
- GELU Activation
- Weight Tying
