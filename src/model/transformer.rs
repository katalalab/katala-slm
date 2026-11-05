use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder, VarMap};

use super::{
    attention::GqaAttention, config::ModelConfig, embedding::EmbeddingLayer, mlp::SwiGluMlp,
    norm::RMSNormLayer,
};

pub(crate) const MAX_ATTENTION_ELEMENTS: usize = 4_194_304;

pub(crate) fn attention_fits(batch: usize, heads: usize, seq: usize) -> bool {
    batch > 0
        && heads > 0
        && seq > 0
        && batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(seq))
            .and_then(|n| n.checked_mul(seq))
            .is_some_and(|n| n <= MAX_ATTENTION_ELEMENTS)
}

pub struct TransformerBlock {
    attn_norm: RMSNormLayer,
    attn: GqaAttention,
    mlp_norm: RMSNormLayer,
    mlp: SwiGluMlp,
}

impl TransformerBlock {
    fn new(config: &ModelConfig, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
        Ok(Self {
            attn_norm: RMSNormLayer::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("attn_norm"),
            )?,
            attn: GqaAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.rope_theta,
                vb.pp("attention"),
                device,
            )?,
            mlp_norm: RMSNormLayer::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("mlp_norm"),
            )?,
            mlp: SwiGluMlp::new(config.hidden_size, config.intermediate_size, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_in = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&attn_in, mask)?;
        let x = x.broadcast_add(&attn_out)?;
        let mlp_in = self.mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        x.broadcast_add(&mlp_out)
    }
}

pub struct TransformerModel {
    pub config: ModelConfig,
    runtime_config: ModelConfig,
    pub device: Device,
    _varmap: Option<VarMap>,
    embedding: EmbeddingLayer,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNormLayer,
    lm_head: candle_nn::Linear,
}

impl TransformerModel {
    pub(crate) fn context_limit(&self) -> usize {
        self.runtime_config.max_position_embeddings
    }

    pub fn new(config: ModelConfig, force_cpu: bool) -> Result<Self> {
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build(config, device.clone(), vb, Some(varmap.clone()))
    }

    pub(crate) fn from_tensors(
        config: ModelConfig,
        tensors: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        Self::build(config, device.clone(), vb, None)
    }

    fn build(
        config: ModelConfig,
        device: Device,
        vb: VarBuilder<'_>,
        varmap: Option<VarMap>,
    ) -> Result<Self> {
        let embedding = EmbeddingLayer::new(config.vocab_size, config.hidden_size, vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            blocks.push(TransformerBlock::new(
                &config,
                vb.pp(format!("layers.{idx}")),
                &device,
            )?);
        }
        let final_norm =
            RMSNormLayer::new(config.hidden_size, config.rms_norm_eps, vb.pp("final_norm"))?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            runtime_config: config.clone(),
            config,
            device,
            _varmap: varmap,
            embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), &self.device)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        if seq_len == 0 || seq_len > self.runtime_config.max_position_embeddings {
            candle_core::bail!("input length outside configured context limit");
        }
        if !attention_fits(batch, self.runtime_config.num_attention_heads, seq_len) {
            candle_core::bail!("attention size exceeds research budget");
        }
        let mut x = self.embedding.forward(input_ids)?;
        let mask = self.causal_mask(seq_len)?;
        for block in &self.blocks {
            x = block.forward(&x, Some(&mask))?;
        }
        let x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
