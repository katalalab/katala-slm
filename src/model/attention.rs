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

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        if self.num_attention_heads == self.num_key_value_heads {
            return Ok(x);
        }
        let repeats = self.num_attention_heads / self.num_key_value_heads;
        let mut chunks = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            chunks.push(x.clone());
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
        let all_key = match past_key {
            Some(past) => Tensor::cat(&[past, &current_key], 2)?,
            None => current_key.clone(),
        };
        let all_value = match past_value {
            Some(past) => Tensor::cat(&[past, &current_value], 2)?,
            None => current_value.clone(),
        };
        let k = self.repeat_kv(all_key)?;
        let v = self.repeat_kv(all_value)?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let attn_scores = match mask {
            Some(m) => attn_scores.broadcast_add(m)?,
            None => attn_scores,
        };
        let attn = ops::softmax_last_dim(&attn_scores)?;
        let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, ()))?;
        Ok((self.o_proj.forward(&ctx)?, current_key, current_value))
    }
}
