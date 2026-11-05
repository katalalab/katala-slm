# Katala SLM

![Rust MSRV](https://img.shields.io/badge/rust-MSRV_unverified-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Candle](https://img.shields.io/badge/ML-Candle-green)

katala-slmは、RustとCandleフレームワークをベースにした、医療領域向けの小規模言語モデルと出力検証機構の研究フレームワークです。モデルによる回答生成に加えて、参照元の根拠情報やスコアの算出を試み、あらかじめ定義された特定の禁忌キーワードの有無を検出してラベル付けを行います。

回答文に対して参照根拠のレベルやスコアを付与する検証処理を研究したい場面や、特定の禁忌キーワードを検知して警告ラベルを添えるパイプラインを試作したい場面に向いています。

禁忌チェック層は例示的な規則に基づくキーワード照合にとどまり、検知されなかったことが安全であることを意味しません。臨床的な判断を行うためのツールではなく、検出結果の提示と研究用途に限定されています。

## Scope and limits — read before using any output

**This is a research framework. It is not clinical decision support and must not be
used to make decisions about a patient.**

The KS verification layer *labels* output. It does not gate it:

- `verify()` returns the model's answer unchanged. A detected contraindication is
  attached as a field; it does not suppress, alter, or block the answer.
- Contraindication detection is **three illustrative keyword rules**
  (pregnancy/isotretinoin, anticoagulant/NSAID, renal/metformin). It is a
  demonstration of where such a check would sit, not a drug-interaction database.
  Any phrasing that avoids those literal words produces an empty contraindication
  list — that means "not checked", never "safe".
- `confidence` and `evidence_level` are scores over retrieved evidence. A low score
  reduces a number in the response; it does not stop the response.

If you are building on this, the gating decision is yours to add and yours to own.
The response carries `clinical_use: false` so that choice cannot be made by accident.

## Verified baseline — 2026-08-28

`cargo fmt --check`, workspace check, strict clippy, and all 56 tests pass. Continual-learning regressions without automatic rollback now return `NeedsReview`; they are no longer reported as success.

The useful KS lineage here is axis separation and explicit review state. Historical solver-majority or evidence-free promotion is intentionally not part of this repository.


## Features
- Decoder-only transformer core (GQA attention + RoPE + SwiGLU + RMSNorm)
- Candle-based forward pass and inference loop
- KS verification pipeline with:
  - Evidence-level classifier (`A/B/C/D`)
  - Confidence scoring (`0.0-1.0`)
  - Source attribution
  - Contraindication checks
- Axum REST API with structured medical output
- CLI modes for local inference and API serving

## Architecture Overview
- `src/model`: model configuration and transformer components
- `src/inference`: generation engine, KV cache, sampling
- `src/ks`: evidence classification, confidence, attribution, verification
- `src/data`: tokenizer wrapper and dataset abstractions
- `src/serve`: HTTP server and endpoints

## Build

### Rust toolchain and CPU verification

With the dependency versions in this lockfile, the declared compiler lower
bound is at least Rust 1.88: Candle 0.11 requires `zip` 8.6.0, which declares
Rust 1.88. This is a dependency requirement, not a verified minimum supported
Rust version (MSRV). Compatibility with Rust 1.88 has not been tested.
See the [Candle dependency declaration](https://github.com/huggingface/candle/blob/0.11.0/candle-core/Cargo.toml)
and [zip version metadata](https://crates.io/api/v1/crates/zip/8.6.0).

The [Linux CPU CI](https://github.com/katalalab/katala-slm/actions/runs/38053305587)
passed all 56 tests on main commit `1cc92d821f7daca2b4d9354eaef9ad330927a5a2`.
This verifies the locked CPU dependencies, including the project's tokenizers
0.23.2 and Candle's separate tokenizers 0.22.2 dependency. It does not establish
the project MSRV or a compiler version below the dependency requirement.

After the locked dependencies are available locally, run the CPU regression
suite without downloading models or starting the API server:

```bash
cargo test --offline --locked --no-default-features
```

CPU test results do not establish CUDA builds, GPU execution, tokenizer file
compatibility, or end-to-end model generation.
Grouped-query attention assigns consecutive query heads to each KV head; a
synthetic CPU regression checks this ordering. This does not establish
compatibility with any particular checkpoint.

```bash
# CPU
cargo build --release

# CUDA
cargo build --release --features cuda
```

## Usage

Model checkpoint loading is not implemented. As a temporary safeguard, supplying
both `infer` (with or without `--model`) and `serve` return an error before model
initialization, rather than silently ignoring a
requested checkpoint or reporting an answer from untrained weights as inference.
A future loader needs a defined checkpoint, model configuration, tokenizer and EOS
contract, plus an end-to-end test showing that loaded weights affect generation.
The lower-level `run_server` API can still be called explicitly with an untrained
engine. Its `/health` and `/v1/medical/generate` endpoints return HTTP 503 while
model loading is unavailable. The library server has no authentication; callers
choose the listening address.
The lower-level `InferenceEngine::generate` remains available for experiments with
untrained weights; `generate_verified` returns an error. None of these paths provides
pretrained medical inference or clinical use.

```bash
# Reports an error until checkpoint loading is implemented
cargo run --release -- infer --prompt "Influenza treatment options?"

# Reports an error; the CLI does not start a server
cargo run --release -- serve --port 8080
```

### Library server API request (returns HTTP 503)
```bash
curl -X POST http://localhost:8080/v1/medical/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is first-line treatment for influenza?"}'
```

The response reports `Generation unavailable`; it is not a medical answer or
evidence assessment.
