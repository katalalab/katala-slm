use std::collections::VecDeque;

use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    Int8,
    Int4,
}

#[derive(Default)]
pub struct LayerKvCache {
    pub keys: VecDeque<Tensor>,
    pub values: VecDeque<Tensor>,
}

impl LayerKvCache {
    pub fn sequence_len(&self) -> Result<usize> {
        self.keys.iter().try_fold(0, |total, tensor| {
            let (_, _, seq_len, _) = tensor.dims4()?;
            Ok(total + seq_len)
        })
    }

    pub fn joined_keys(&self) -> Result<Option<Tensor>> {
        Self::join(&self.keys)
    }

    pub fn joined_values(&self) -> Result<Option<Tensor>> {
        Self::join(&self.values)
    }

    fn join(chunks: &VecDeque<Tensor>) -> Result<Option<Tensor>> {
        match chunks.len() {
            0 => Ok(None),
            1 => Ok(chunks.front().cloned()),
            _ => Ok(Some(Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?)),
        }
    }

    fn append(&mut self, key: Tensor, value: Tensor, max_seq_len: usize) -> Result<()> {
        let (_, _, key_len, _) = key.dims4()?;
        let (_, _, value_len, _) = value.dims4()?;
        if key_len != value_len || self.sequence_len()? + key_len > max_seq_len {
            candle_core::bail!("KV cache sequence exceeds its configured capacity")
        }
        self.keys.push_back(key);
        self.values.push_back(value);
        Ok(())
    }
}

pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub quantization: Quantization,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, max_seq_len: usize, quantization: Quantization) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::default()).collect(),
            quantization,
            max_seq_len,
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    pub fn sequence_len(&self) -> Result<usize> {
        self.layers
            .first()
            .map_or(Ok(0), LayerKvCache::sequence_len)
    }

    pub fn append(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| candle_core::Error::msg("KV cache layer index is out of range"))?;
        layer.append(key, value, self.max_seq_len)
    }

    pub fn push(&mut self, layer_idx: usize, key: Tensor, value: Tensor) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            if layer.keys.len() >= self.max_seq_len {
                layer.keys.pop_front();
                layer.values.pop_front();
            }
            layer.keys.push_back(key);
            layer.values.push_back(value);
        }
    }
}
