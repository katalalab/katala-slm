use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::Rng;

use crate::{
    data::tokenizer::TokenizerWrapper,
    ks::verify::{VerifiedAnswer, Verifier},
    model::{config::ModelConfig, transformer::TransformerModel},
};

use super::sampler::Sampler;

#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.8,
            top_p: 0.95,
        }
    }
}

pub struct InferenceEngine {
    pub model: Arc<TransformerModel>,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub verifier: Arc<Verifier>,
}

impl InferenceEngine {
    pub fn new(config: ModelConfig, force_cpu: bool) -> Result<Self> {
        let model = Arc::new(TransformerModel::new(config, force_cpu)?);
        let tokenizer = Arc::new(TokenizerWrapper::from_builtin()?);
        let verifier = Arc::new(Verifier::default());
        Ok(Self {
            model,
            tokenizer,
            verifier,
        })
    }

    pub fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<String> {
        self.generate_with_rng(prompt, options, &mut rand::rng())
    }

    /// Generate using caller-owned random state, which is advanced per token.
    pub fn generate_with_rng<R: Rng + ?Sized>(
        &self,
        prompt: &str,
        options: GenerateOptions,
        rng: &mut R,
    ) -> Result<String> {
        let sampler = Sampler::new(options.temperature, options.top_p);
        sampler.validate()?;
        let mut token_ids = self.tokenizer.encode(prompt)?;
        anyhow::ensure!(
            !token_ids.is_empty() && token_ids.len() <= self.model.context_limit(),
            "prompt token count exceeds model context or is empty"
        );

        for _ in 0..options.max_new_tokens {
            let input =
                Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &self.model.device)?
                    .to_dtype(DType::U32)?;
            let logits = self.model.forward(&input)?;
            let last = Sampler::last_token_logits(&logits)?;
            let next = sampler.sample_with_rng(&last, rng)?;
            token_ids.push(next);
            if next == self.tokenizer.eos_token_id() {
                break;
            }
        }

        self.tokenizer.decode(&token_ids)
    }

    pub fn generate_verified(
        &self,
        _prompt: &str,
        _options: GenerateOptions,
    ) -> Result<VerifiedAnswer> {
        anyhow::bail!("model loading is not supported")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_sampling_options_fail_before_generation_even_at_zero_tokens() -> Result<()> {
        let config = ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        };
        let engine = InferenceEngine::new(config, true)?;
        for max_new_tokens in [0, 1] {
            for (temperature, top_p, expected) in [(0., 1., "temperature"), (1., 0., "top_p")] {
                let options = GenerateOptions {
                    max_new_tokens,
                    temperature,
                    top_p,
                };
                let err = engine
                    .generate("test", options)
                    .expect_err("invalid sampling settings must fail");
                assert!(
                    err.to_string().contains(expected),
                    "unexpected error: {err}"
                );
            }
        }
        Ok(())
    }
}
