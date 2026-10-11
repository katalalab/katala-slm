# Katala SLM — Architecture Specification

## Overview
Katala SLM is a medical-domain-specialized Small Language Model framework written in Rust.
It combines KS-series verification methodology with state-of-the-art SLM techniques.

## Core Design Principles (from KS Series)
1. **Evidence-Level Awareness**: Every output includes evidence classification (A/B/C/D)
2. **Self-Verification**: Model can assess its own confidence using KS40e-derived logic
3. **Minimal Footprint**: Target <500MB model, <2GB VRAM for inference
4. **Rust-First**: Core inference, training loop, and data pipeline in Rust

## Architecture

### 1. Model Core (`src/model/`)
- Transformer decoder-only architecture
- Configurable: 0.1B to 1.5B parameters
- RoPE positional encoding
- Grouped Query Attention (GQA) for memory efficiency
- SwiGLU activation
- RMSNorm

### 2. Inference Engine (`src/inference/`)
- Custom CUDA kernels via `cudarc` crate (optional, CPU fallback)
- Incremental per-request KV cache (unquantized, batch size 1); the INT8/INT4 quantizer is not wired into generation
- Speculative decoding support
- Continuous batching for serving

### 3. Training Pipeline (`src/training/`)
- QLoRA fine-tuning support
- Knowledge distillation (TAID-style temporal interpolation)
- RLHF/DPO for alignment
- Medical data preprocessing (PubMed parser, JMED format)

### 4. KS Verification Layer (`src/ks/`)
- Evidence level classifier (A/B/C/D)
- Confidence scorer (0.0-1.0)
- Source attribution tracker
- Contraindication detector
- Based on KS40e 18-axis verification logic

### 5. Data Pipeline (`src/data/`)
- PubMed XML parser
- Japanese clinical guideline parser
- MedQA dataset loader
- Tokenizer (BPE, SentencePiece compatible)
- Evidence-level annotated dataset format

### 6. Serving (`src/serve/`)
- REST API (axum)
- Structured output: `{answer, evidence_level, sources, confidence, contraindications}`
- LoRA adapter hot-swap
- Health check endpoint

## File Structure
```
katala-slm/
├── Cargo.toml
├── README.md
├── LICENSE (Apache-2.0)
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── config.rs        # Model configuration
│   │   ├── transformer.rs   # Core transformer blocks
│   │   ├── attention.rs     # GQA + RoPE
│   │   ├── mlp.rs          # SwiGLU MLP
│   │   ├── norm.rs         # RMSNorm
│   │   └── embedding.rs    # Token + position embeddings
│   ├── inference/
│   │   ├── mod.rs
│   │   ├── engine.rs       # Main inference engine
│   │   ├── kv_cache.rs     # KV cache management
│   │   ├── sampler.rs      # Temperature, top-p, etc.
│   │   └── speculative.rs  # Speculative decoding
│   ├── training/
│   │   ├── mod.rs
│   │   ├── qlora.rs        # QLoRA implementation
│   │   ├── distill.rs      # Knowledge distillation
│   │   ├── rlhf.rs         # RLHF/DPO
│   │   └── optimizer.rs    # AdamW, 8-bit Adam
│   ├── ks/
│   │   ├── mod.rs
│   │   ├── evidence.rs     # Evidence level classification
│   │   ├── confidence.rs   # Confidence scoring
│   │   ├── source.rs       # Source attribution
│   │   └── verify.rs       # Full verification pipeline
│   ├── data/
│   │   ├── mod.rs
│   │   ├── pubmed.rs       # PubMed parser
│   │   ├── jmed.rs         # Japanese medical data
│   │   ├── tokenizer.rs    # BPE tokenizer
│   │   └── dataset.rs      # Dataset abstractions
│   └── serve/
│       ├── mod.rs
│       └── api.rs          # REST API
├── tests/
│   ├── model_tests.rs
│   ├── inference_tests.rs
│   ├── ks_tests.rs
│   └── integration_tests.rs
├── benches/
│   └── inference_bench.rs
└── examples/
    ├── medical_qa.rs
    └── evidence_check.rs
```

## Dependencies
- `candle-core` / `candle-nn` — Rust ML framework (HuggingFace)
- `tokenizers` — HuggingFace tokenizers in Rust
- `cudarc` — CUDA bindings (optional)
- `axum` — HTTP server
- `serde` / `serde_json` — Serialization
- `rayon` — Parallel data processing
- `half` — f16/bf16 support

## Build & Run
```bash
# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda

# Run inference
cargo run --release -- --model path/to/model --prompt "インフルエンザの治療法は？"

# Start API server
cargo run --release -- serve --port 8080
```
