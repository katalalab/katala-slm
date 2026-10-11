use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, ops, Linear, Module, VarBuilder};

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    device: Device,
}

impl GqaAttention {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f64,
        vb: VarBuilder<'_>,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_attention_heads;
        let kv_out = num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden_size, kv_out, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden_size, kv_out, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            device: device.clone(),
        })
    }

    fn apply_rope(&self, x: Tensor, seq_len: usize, position_offset: usize) -> Result<Tensor> {
        let (_b, _h, _s, d) = x.dims4()?;
        let half = d / 2;
        let positions = Tensor::arange(
            position_offset as u32,
            (position_offset + seq_len) as u32,
            &self.device,
        )?
        .to_dtype(DType::F32)?;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / self.rope_theta.powf((2 * i) as f64 / d as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, 1, half), &self.device)?;
        let angles = positions
            .reshape((1, 1, seq_len, 1))?
            .broadcast_mul(&inv_freq)?;
        let sin = angles.sin()?;
        let cos = angles.cos()?;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let y1 = x1
            .broadcast_mul(&cos)?
            .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let y2 = x1
            .broadcast_mul(&sin)?
            .broadcast_add(&x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&y1, &y2], 3)
    }

    fn causal_mask(&self, seq_len: usize, key_len: usize, kv_offset: usize) -> Result<Tensor> {
        let mut mask = vec![0f32; seq_len * key_len];
        for query in 0..seq_len {
            for key in (kv_offset + query + 1)..key_len {
                mask[query * key_len + key] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask, (1, 1, seq_len, key_len), &self.device)
    }

    #[cfg(test)]
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        if self.num_attention_heads == self.num_key_value_heads {
            return Ok(x);
        }
        let repeats = self.num_attention_heads / self.num_key_value_heads;
        let mut chunks = Vec::with_capacity(self.num_attention_heads);
        for head in 0..self.num_key_value_heads {
            let kv = x.narrow(1, head, 1)?;
            for _ in 0..repeats {
                chunks.push(kv.clone());
            }
        }
        Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 1)
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(x, None, None, 0, mask)
            .map(|(output, _, _)| output)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        past_key: Option<&Tensor>,
        past_value: Option<&Tensor>,
        position_offset: usize,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let past_keys = past_key.into_iter().collect::<Vec<_>>();
        let past_values = past_value.into_iter().collect::<Vec<_>>();
        self.forward_with_cache_chunks(x, &past_keys, &past_values, position_offset, mask)
    }

    pub(super) fn forward_with_cache_chunks(
        &self,
        x: &Tensor,
        past_keys: &[&Tensor],
        past_values: &[&Tensor],
        position_offset: usize,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_with_cache_chunks_impl(x, past_keys, past_values, position_offset, mask, None)
    }

    pub(super) fn forward_causal(&self, x: &Tensor, kv_offset: usize) -> Result<Tensor> {
        self.forward_with_cache_chunks_impl(x, &[], &[], kv_offset, None, Some(kv_offset))
            .map(|(output, _, _)| output)
    }

    fn forward_with_cache_chunks_impl(
        &self,
        x: &Tensor,
        past_keys: &[&Tensor],
        past_values: &[&Tensor],
        position_offset: usize,
        mask: Option<&Tensor>,
        causal_kv_offset: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rope(q, s, position_offset)?;
        let current_key = self.apply_rope(k, s, position_offset)?;
        let current_value = v;
        let all_key = if past_keys.is_empty() {
            current_key.clone()
        } else {
            let mut chunks = Vec::with_capacity(past_keys.len() + 1);
            chunks.extend_from_slice(past_keys);
            chunks.push(&current_key);
            Tensor::cat(&chunks, 2)?
        };
        let all_value = if past_values.is_empty() {
            current_value.clone()
        } else {
            let mut chunks = Vec::with_capacity(past_values.len() + 1);
            chunks.extend_from_slice(past_values);
            chunks.push(&current_value);
            Tensor::cat(&chunks, 2)?
        };
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let ctx = if self.device.is_cpu() && q.dtype() == DType::F32 && (b == 1 || mask.is_none()) {
            let attn_mask =
                match (mask, causal_kv_offset) {
                    (Some(mask), _) => candle_nn::attention::AttnMask::Mask(mask.clone()),
                    (None, Some(kv_offset)) if b > 1 => {
                        candle_nn::attention::AttnMask::causal_with_offset(kv_offset)
                    }
                    (None, Some(_)) if s == 1 => candle_nn::attention::AttnMask::None,
                    (None, Some(kv_offset)) => candle_nn::attention::AttnMask::Mask(
                        self.causal_mask(s, all_key.dims4()?.2, kv_offset)?,
                    ),
                    (None, None) => candle_nn::attention::AttnMask::None,
                };
            candle_nn::attention::flash_attn::<f32>(
                &q.transpose(1, 2)?,
                &all_key.transpose(1, 2)?,
                &all_value.transpose(1, 2)?,
                scale as f32,
                attn_mask,
                None,
                None,
            )?
            .transpose(1, 2)?
        } else {
            let query_groups = self.num_attention_heads / self.num_key_value_heads;
            let grouped_q =
                q.reshape((b, self.num_key_value_heads, query_groups, s, self.head_dim))?;
            let k = all_key.transpose(2, 3)?.contiguous()?;
            let v = all_value.contiguous()?;
            let generated_mask = match (mask, causal_kv_offset) {
                (None, Some(kv_offset)) => {
                    Some(self.causal_mask(s, all_key.dims4()?.2, kv_offset)?)
                }
                _ => None,
            };
            let mask = mask.or(generated_mask.as_ref());
            let mut contexts = Vec::with_capacity(query_groups);
            for group in 0..query_groups {
                let group_q = grouped_q.narrow(2, group, 1)?.reshape((
                    b,
                    self.num_key_value_heads,
                    s,
                    self.head_dim,
                ))?;
                let scores = group_q.matmul(&k)?.affine(scale, 0.0)?;
                let scores = match mask {
                    Some(m) => scores.broadcast_add(m)?,
                    None => scores,
                };
                let weights = ops::softmax_last_dim(&scores)?;
                contexts.push(weights.matmul(&v)?.unsqueeze(2)?);
            }
            Tensor::cat(&contexts.iter().collect::<Vec<_>>(), 2)?
                .reshape((b, self.num_attention_heads, s, self.head_dim))?
                .transpose(1, 2)?
        };
        let ctx = ctx.reshape((b, s, ()))?;
        Ok((self.o_proj.forward(&ctx)?, current_key, current_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn attention() -> Result<GqaAttention> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        GqaAttention::new(8, 4, 2, 10_000.0, vb, &device)
    }

    fn expand_kv_reference(x: Tensor, kv_heads: usize, query_heads: usize) -> Result<Tensor> {
        let repeats = query_heads / kv_heads;
        let mut chunks = Vec::with_capacity(query_heads);
        for head in 0..kv_heads {
            let kv = x.narrow(1, head, 1)?;
            for _ in 0..repeats {
                chunks.push(kv.clone());
            }
        }
        Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 1)
    }

    fn standard_gqa_reference(
        attention: &GqaAttention,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = attention
            .q_proj
            .forward(x)?
            .reshape((b, s, attention.num_attention_heads, attention.head_dim))?
            .transpose(1, 2)?;
        let k = attention
            .k_proj
            .forward(x)?
            .reshape((b, s, attention.num_key_value_heads, attention.head_dim))?
            .transpose(1, 2)?;
        let v = attention
            .v_proj
            .forward(x)?
            .reshape((b, s, attention.num_key_value_heads, attention.head_dim))?
            .transpose(1, 2)?;
        let q = attention.apply_rope(q, s, 0)?;
        let k = attention.apply_rope(k, s, 0)?;
        let k = expand_kv_reference(
            k,
            attention.num_key_value_heads,
            attention.num_attention_heads,
        )?;
        let v = expand_kv_reference(
            v,
            attention.num_key_value_heads,
            attention.num_attention_heads,
        )?;
        let k = k.transpose(2, 3)?.contiguous()?;
        let v = v.contiguous()?;
        let scores = q
            .matmul(&k)?
            .affine(1.0 / (attention.head_dim as f64).sqrt(), 0.0)?;
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let weights = ops::softmax_last_dim(&scores)?;
        let context = weights.matmul(&v)?.transpose(1, 2)?.reshape((b, s, ()))?;
        attention.o_proj.forward(&context)
    }

    #[test]
    fn gqa_repeats_each_kv_head_over_its_query_group() -> Result<()> {
        let attention = attention()?;
        let input = Tensor::from_vec(vec![10.0f32, 20.0], (1, 2, 1, 1), &Device::Cpu)?;
        let actual = attention
            .repeat_kv(input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, [10.0, 10.0, 20.0, 20.0]);
        Ok(())
    }

    #[test]
    fn gqa_forward_matches_standard_grouped_attention() -> Result<()> {
        let attention = attention()?;
        let x = Tensor::from_vec(
            (0..24).map(|x| x as f32 / 10.0).collect::<Vec<_>>(),
            (1, 3, 8),
            &Device::Cpu,
        )?;
        let actual = attention
            .forward(&x, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = standard_gqa_reference(&attention, &x, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn gqa_causal_forward_matches_standard_grouped_attention() -> Result<()> {
        let attention = attention()?;
        let x = Tensor::from_vec(
            (0..24).map(|x| x as f32 / 10.0).collect::<Vec<_>>(),
            (1, 3, 8),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (1, 1, 3, 3),
            &Device::Cpu,
        )?;
        let actual = attention
            .forward(&x, Some(&mask))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = standard_gqa_reference(&attention, &x, Some(&mask))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn batched_gqa_causal_forward_matches_standard_grouped_attention() -> Result<()> {
        let attention = attention()?;
        let x = Tensor::from_vec(
            (0..48).map(|x| x as f32 / 10.0).collect::<Vec<_>>(),
            (2, 3, 8),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (1, 1, 3, 3),
            &Device::Cpu,
        )?;
        let actual = attention
            .forward(&x, Some(&mask))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = standard_gqa_reference(&attention, &x, Some(&mask))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn batched_gqa_causal_flash_forward_matches_standard_grouped_attention() -> Result<()> {
        let attention = attention()?;
        let x = Tensor::from_vec(
            (0..48).map(|x| x as f32 / 10.0).collect::<Vec<_>>(),
            (2, 3, 8),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(
            vec![
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (1, 1, 3, 3),
            &Device::Cpu,
        )?;
        let actual = attention
            .forward_causal(&x, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = standard_gqa_reference(&attention, &x, Some(&mask))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn cached_gqa_forward_accepts_chunked_past_keys_and_values() -> Result<()> {
        let attention = attention()?;
        let x = Tensor::from_vec(
            (0..24).map(|x| x as f32 / 10.0).collect::<Vec<_>>(),
            (1, 3, 8),
            &Device::Cpu,
        )?;
        let prefix = x.narrow(1, 0, 2)?;
        let current = x.narrow(1, 2, 1)?;
        let (_, key, value) = attention.forward_with_cache(&prefix, None, None, 0, None)?;
        let key_chunks = [key.narrow(2, 0, 1)?, key.narrow(2, 1, 1)?];
        let value_chunks = [value.narrow(2, 0, 1)?, value.narrow(2, 1, 1)?];
        let key_refs = key_chunks.iter().collect::<Vec<_>>();
        let value_refs = value_chunks.iter().collect::<Vec<_>>();

        let (cached, _, _) =
            attention.forward_with_cache_chunks(&current, &key_refs, &value_refs, 2, None)?;
        let expected = attention.forward(&x, None)?.narrow(1, 2, 1)?;
        let cached = cached.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert!(cached
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-5));
        Ok(())
    }
}
