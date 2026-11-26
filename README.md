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
bound is at least Rust 1.88: Candle 0.11.1 resolves to `zip` 9.5.1, which declares
Rust 1.88. This is a dependency requirement, not a verified minimum supported
Rust version (MSRV). Compatibility with Rust 1.88 has not been tested.
See the [Candle dependency declaration](https://github.com/huggingface/candle/blob/0.11.1/candle-core/Cargo.toml)
and [zip version metadata](https://crates.io/api/v1/crates/zip/9.5.1).

The [Linux CPU CI](https://github.com/katalalab/katala-slm/actions/runs/38053305587)
passed all 56 tests on main commit `1cc92d821f7daca2b4d9354eaef9ad330927a5a2`.
That historical run used tokenizers 0.23.2 together with Candle's separate
0.22.2 dependency. The current lockfile resolves only tokenizers 0.23.2. Neither
that run nor the dependency requirements establish the project MSRV.

After the locked dependencies are available locally, run the CPU regression
suite without downloading models or starting the API server:

```bash
cargo test --offline --locked --no-default-features
```

CPU test results do not establish CUDA builds, GPU execution, tokenizer file
compatibility, or end-to-end model generation.

```bash
# CPU
cargo build --release

# CUDA
cargo build --release --features cuda
```

### Inference benchmark

```bash
KATALA_BENCH_DEVICE=cpu cargo bench --locked --bench inference_bench
KATALA_BENCH_DEVICE=cuda cargo bench --locked --features cuda --bench inference_bench
```

The default is `cpu`. CUDA mode requires both the Cargo feature and Candle
CUDA device 0. It exits with an error if either is unavailable. The benchmark
prints the actual Candle device, uses two warmups and five measured samples per
workload, and synchronizes the device before reading each elapsed time. Compare
runs with the same Cargo profile. The KS verification workload always runs on
CPU.

## Usage
```bash
# Inference CLI
cargo run --release -- --cpu infer --model ./checkpoint --prompt "The capital of France is" --max-new-tokens 4

# API server
cargo run --release -- --cpu serve --model ./checkpoint --bind 127.0.0.1 --port 8080
```

Supply a local Llama-format checkpoint directory containing its matching
`config.json`, `tokenizer.json`, and unsharded `model.safetensors`. The runtime
uses the checkpoint's tokenizer, EOS settings, context limit, and learned
weights. It returns only the newly generated continuation. Prompts use raw
completion text; supply the model's chat template yourself for chat inputs.
Missing or incompatible checkpoints fail at startup. No weights are bundled
or downloaded at runtime.

The CPU acceptance checkpoint is Hugging Face's general-language
[SmolLM2-135M-Instruct, revision 12fd25f](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac).
The weights file SHA-256 is
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
This model provides a reproducible generation check. Medical-domain validation
requires its own model, evidence, and evaluation.

Omitting `--model` on `serve` starts the citation endpoint alone.
`GET /health` reports `generation_available`, and generation without a loaded
model returns HTTP 503. Each server runs one generation at a time. A concurrent
generation returns HTTP 429 while health and citation checks remain available.
`infer` requires an explicit checkpoint.

### API Request
```bash
curl -X POST http://localhost:8080/v1/medical/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is first-line treatment for influenza?"}'
```

### API Response
```json
{
  "answer": "...",
  "evidence_level": "B",
  "sources": [
    {
      "source_id": "cdc-flu-antiviral",
      "title": "CDC Influenza Antiviral Guidance",
      "url": "https://www.cdc.gov/flu/professionals/antivirals/index.htm",
      "snippet": "Early antiviral treatment is recommended for high-risk patients."
    }
  ],
  "confidence": 0.72,
  "contraindications": []
}
```

### Check a quotation against supplied passages

`verify-citation` reads a claim and its source passages as JSON on stdin. The same
request is accepted by `POST /v1/medical/verify-citation`. The CLI runs without
initializing a model. Use one unique identifier per passage, such as an article
ID plus paragraph number.

```sh
printf '%s\n' '{"claim":"The line is blue.","cited_source_id":"example#2","passages":[{"source_id":"example#1","text":"The line is blue."}]}' | katala-slm verify-citation
```

This example returns `{"status":"rebound","source_id":"example#1"}`.
`exact_passage` retains an existing matching identifier. The check compares
the full claim to the full supplied passage after trimming outer whitespace.
Missing evidence, an unmatched quotation, ambiguous rebinding and invalid
passage records return `abstained` with a reason. Passage identity does not
establish source authenticity, semantic entailment or clinical suitability.

For generated answers, put `passages` and optional `cited_source_id` in the
request's `citation_evidence` object, or pass that object in a JSON file with
`infer --citation-evidence FILE`. The resulting `citation_review` describes the
whole answer. Prompt-selected source suggestions alone produce an abstention.
Research clients that require verbatim grounding should withhold responses
whose citation review abstains. Existing callers still receive the answer and
contraindication labels.
