use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::{
    data::tokenizer::TokenizerWrapper,
    ks::verify::{VerifiedAnswer, Verifier},
    model::{config::ModelConfig, transformer::TransformerModel},
};

use super::{
    kv_cache::{KvCache, Quantization},
    sampler::Sampler,
};

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
        let model = Arc::new(TransformerModel::new(config.clone(), force_cpu)?);
        let tokenizer = Arc::new(TokenizerWrapper::from_builtin()?);
        let verifier = Arc::new(Verifier::default());
        Ok(Self {
            model,
            tokenizer,
            verifier,
        })
    }

    pub fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<String> {
        let mut token_ids = self.tokenizer.encode(prompt)?;
        let sampler = Sampler::new(options.temperature, options.top_p);
        if options.max_new_tokens == 0 {
            return self.tokenizer.decode(&token_ids);
        }

        let mut cache = KvCache::new(
            self.model.config.num_hidden_layers,
            self.model.config.max_position_embeddings,
            Quantization::None,
        );
        let input = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &self.model.device)?
            .to_dtype(DType::U32)?;
        let mut logits = self.model.forward_with_cache(&input, &mut cache)?;

        for generated in 0..options.max_new_tokens {
            let last = Sampler::last_token_logits(&logits)?;
            let next = sampler.sample(&last)?;
            token_ids.push(next);
            if next == self.tokenizer.eos_token_id() || generated + 1 == options.max_new_tokens {
                break;
            }
            let input =
                Tensor::from_vec(vec![next], (1, 1), &self.model.device)?.to_dtype(DType::U32)?;
            logits = self.model.forward_with_cache(&input, &mut cache)?;
        }

        self.tokenizer.decode(&token_ids)
    }

    pub fn generate_verified(
        &self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Result<VerifiedAnswer> {
        let answer = self.generate(prompt, options)?;
        Ok(self.verifier.verify(prompt, &answer, &[]))
    }
}
