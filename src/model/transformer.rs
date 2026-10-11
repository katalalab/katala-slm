use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder, VarMap};

use crate::inference::kv_cache::{KvCache, LayerKvCache};

use super::{
    attention::GqaAttention, config::ModelConfig, embedding::EmbeddingLayer, mlp::SwiGluMlp,
    norm::RMSNormLayer,
};

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

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        causal_kv_offset: Option<usize>,
    ) -> Result<Tensor> {
        let attn_in = self.attn_norm.forward(x)?;
        let attn_out = match causal_kv_offset {
            Some(kv_offset) => self.attn.forward_causal(&attn_in, kv_offset)?,
            None => self.attn.forward(&attn_in, mask)?,
        };
        let x = x.broadcast_add(&attn_out)?;
        let mlp_in = self.mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        x.broadcast_add(&mlp_out)
    }

    fn forward_with_cache(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &LayerKvCache,
        position_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let attn_in = self.attn_norm.forward(x)?;
        let past_keys = cache.keys.iter().collect::<Vec<_>>();
        let past_values = cache.values.iter().collect::<Vec<_>>();
        let (attn_out, key, value) = self.attn.forward_with_cache_chunks(
            &attn_in,
            &past_keys,
            &past_values,
            position_offset,
            mask,
        )?;
        let x = x.broadcast_add(&attn_out)?;
        let mlp_in = self.mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok((x.broadcast_add(&mlp_out)?, key, value))
    }
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub device: Device,
    _varmap: VarMap,
    embedding: EmbeddingLayer,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNormLayer,
    lm_head: candle_nn::Linear,
}

impl TransformerModel {
    pub fn new(config: ModelConfig, force_cpu: bool) -> Result<Self> {
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
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

    fn causal_mask_with_past(&self, past_len: usize, seq_len: usize) -> Result<Tensor> {
        let key_len = past_len + seq_len;
        let mut mask = vec![0f32; seq_len * key_len];
        for query in 0..seq_len {
            for key in (past_len + query + 1)..key_len {
                mask[query * key_len + key] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask, (1, 1, seq_len, key_len), &self.device)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let use_causal_flash = batch_size > 1 && self.device.is_cpu();
        let mask = if use_causal_flash {
            None
        } else {
            Some(self.causal_mask(seq_len)?)
        };
        let mut x = self.embedding.forward(input_ids)?;
        for block in &self.blocks {
            x = block.forward(&x, mask.as_ref(), use_causal_flash.then_some(0))?;
        }
        let x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        if batch_size != 1 {
            candle_core::bail!("incremental KV cache currently supports batch size 1")
        }
        if cache.layers.len() != self.blocks.len() {
            candle_core::bail!("KV cache layer count does not match the model")
        }
        let past_len = cache.sequence_len()?;
        if past_len + seq_len > cache.max_seq_len {
            candle_core::bail!("KV cache sequence exceeds its configured capacity")
        }
        for layer in &cache.layers {
            if layer.sequence_len()? != past_len {
                candle_core::bail!("KV cache layers have inconsistent sequence lengths")
            }
        }

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask_with_past(past_len, seq_len)?)
        };
        let mut x = self.embedding.forward(input_ids)?;
        for (idx, block) in self.blocks.iter().enumerate() {
            let (next_x, key, value) =
                block.forward_with_cache(&x, mask.as_ref(), &cache.layers[idx], past_len)?;
            x = next_x;
            cache.append(idx, key, value)?;
        }
        let x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cross_device_tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn full_forward_matches_across_cpu_and_cuda_for_same_weights() -> Result<()> {
        let config = ModelConfig {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        };
        let cpu = TransformerModel::new(config.clone(), true)?;
        let temp = tempfile::tempdir()?;
        let checkpoint = temp.path().join("weights.safetensors");
        cpu._varmap.save(&checkpoint)?;

        let mut cuda = TransformerModel::new(config, false)?;
        assert!(
            cuda.device.is_cuda(),
            "CUDA parity test requires a visible CUDA device"
        );
        cuda._varmap.load(checkpoint)?;

        let tokens: Vec<u32> = (1..=12).collect();
        let cpu_input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &cpu.device)?
            .to_dtype(DType::U32)?;
        let cuda_input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &cuda.device)?
            .to_dtype(DType::U32)?;
        let cpu_logits = cpu.forward(&cpu_input)?.to_vec3::<f32>()?;
        let cuda_logits = cuda.forward(&cuda_input)?.to_vec3::<f32>()?;
        assert_eq!(cpu_logits.len(), cuda_logits.len());
        assert_eq!(cpu_logits[0].len(), cuda_logits[0].len());
        assert!(cpu_logits[0]
            .iter()
            .zip(&cuda_logits[0])
            .all(|(cpu, cuda)| cpu.len() == cuda.len()));
        let max_delta = cpu_logits[0]
            .iter()
            .flatten()
            .zip(cuda_logits[0].iter().flatten())
            .map(|(cpu, cuda)| (cpu - cuda).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_delta < 1e-4, "CPU/CUDA max logit delta: {max_delta}");
        Ok(())
    }
}
