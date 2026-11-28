//! Local, single-file Hugging Face Llama checkpoints. Loading never downloads files.
use std::{fs, path::Path};

use anyhow::{anyhow, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::{engine::GenerateOptions, sampler::Sampler};
use crate::ks::verify::{VerifiedAnswer, Verifier};

#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    rope_interleaved: bool,
    head_dim: Option<usize>,
    #[serde(flatten)]
    llama: LlamaConfig,
}

pub struct PretrainedLlama {
    model: Llama,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

impl PretrainedLlama {
    pub fn load(dir: &Path, force_cpu: bool) -> Result<Self> {
        let raw =
            fs::read(dir.join("config.json")).context("cannot read checkpoint config.json")?;
        let spec: CheckpointConfig =
            serde_json::from_slice(&raw).context("invalid Llama configuration")?;
        ensure!(
            spec.model_type == "llama"
                && spec.hidden_act == "silu"
                && !spec.attention_bias
                && !spec.mlp_bias
                && !spec.rope_interleaved,
            "unsupported model architecture"
        );
        let config = spec.llama.into_config(false);
        ensure!(
            config.num_attention_heads > 0
                && config.num_key_value_heads > 0
                && config.hidden_size > 0
                && config
                    .hidden_size
                    .is_multiple_of(config.num_attention_heads)
                && config
                    .num_attention_heads
                    .is_multiple_of(config.num_key_value_heads)
                && config.num_hidden_layers > 0
                && config.max_position_embeddings > 0,
            "invalid Llama dimensions"
        );
        let head_dim = config.hidden_size / config.num_attention_heads;
        ensure!(
            head_dim.is_multiple_of(2) && spec.head_dim.is_none_or(|d| d == head_dim),
            "unsupported Llama head dimension"
        );

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|err| anyhow!("cannot load tokenizer.json: {err}"))?;
        ensure!(
            tokenizer.get_vocab_size(true) > 0,
            "tokenizer has no vocabulary"
        );
        ensure!(
            tokenizer
                .get_vocab(true)
                .values()
                .all(|&id| (id as usize) < config.vocab_size),
            "tokenizer IDs exceed model vocabulary"
        );
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
        let weights = fs::read(dir.join("model.safetensors"))
            .context("cannot read checkpoint model.safetensors")?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        // Candle 0.11 unwraps errors when constructing individual Llama blocks.
        // Translate a malformed checkpoint into a startup error at this boundary.
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Llama::load(vb, &config)
        }))
        .map_err(|_| anyhow!("checkpoint contains a missing or invalid transformer block"))??;
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }

    pub fn generate_verified(
        &self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Result<VerifiedAnswer> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|err| anyhow!("tokenization failed: {err}"))?;
        let prompt_ids = encoding.get_ids();
        ensure!(
            !prompt.trim().is_empty() && !prompt_ids.is_empty(),
            "prompt must contain tokens"
        );
        ensure!(
            options.max_new_tokens
                <= self
                    .config
                    .max_position_embeddings
                    .saturating_sub(prompt_ids.len())
                && prompt_ids.len() <= self.config.max_position_embeddings,
            "request exceeds the model context window"
        );
        let sampler = Sampler::new(options.temperature, options.top_p);
        let mut generated = Vec::new();
        if options.max_new_tokens > 0 {
            let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
            let mut position = 0;
            let mut input = Tensor::new(prompt_ids, &self.device)?.unsqueeze(0)?;
            for _ in 0..options.max_new_tokens {
                let logits = self
                    .model
                    .forward(&input, position, &mut cache)?
                    .squeeze(0)?;
                let next = sampler.sample(&logits)?;
                let finished = match &self.config.eos_token_id {
                    Some(LlamaEosToks::Single(id)) => *id == next,
                    Some(LlamaEosToks::Multiple(ids)) => ids.contains(&next),
                    None => false,
                };
                if finished {
                    break;
                }
                generated.push(next);
                position += input.dim(1)?;
                input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            }
        }
        let answer = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|err| anyhow!("decoding failed: {err}"))?;
        Ok(Verifier::default().verify(prompt, &answer, &[]))
    }
}
