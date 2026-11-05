use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, fs, path::Path};

pub fn fixture(dir: &Path, altered: bool, fault: &str) -> Result<()> {
    let config = serde_json::json!({"format":"katala-slm-research-v1", "eos_token_id":2,
        "model":{"vocab_size":4,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,
        "rope_theta":10000.0,"rms_norm_eps":0.000001}});
    fs::write(dir.join("config.json"), serde_json::to_vec(&config)?)?;
    let tokenizer = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2,"world":3},"unk_token":"<unk>"}});
    fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tokenizer)?)?;
    let mut t = HashMap::new();
    for (name, dims, value) in [
        ("embed.token_embedding.weight", vec![4, 4], 1f32),
        ("layers.0.attn_norm.rms_norm.weight", vec![4], 1.),
        ("layers.0.attention.q_proj.weight", vec![4, 4], 0.),
        ("layers.0.attention.k_proj.weight", vec![2, 4], 0.),
        ("layers.0.attention.v_proj.weight", vec![2, 4], 0.),
        ("layers.0.attention.o_proj.weight", vec![4, 4], 0.),
        ("layers.0.mlp_norm.rms_norm.weight", vec![4], 1.),
        ("layers.0.mlp.gate_proj.weight", vec![8, 4], 0.),
        ("layers.0.mlp.up_proj.weight", vec![8, 4], 0.),
        ("layers.0.mlp.down_proj.weight", vec![4, 8], 0.),
        ("final_norm.rms_norm.weight", vec![4], 1.),
        ("lm_head.weight", vec![4, 4], if altered { 2. } else { 1. }),
    ] {
        if fault == "missing" && name == "lm_head.weight" {
            continue;
        }
        let shape = if fault == "shape" && name == "lm_head.weight" {
            vec![2, 4]
        } else {
            dims
        };
        let dtype = if fault == "dtype" && name == "lm_head.weight" {
            DType::F64
        } else {
            DType::F32
        };
        let value = if fault == "nonfinite" && name == "lm_head.weight" {
            f32::NAN
        } else {
            value
        };
        t.insert(
            name.to_string(),
            Tensor::full(value, shape.as_slice(), &Device::Cpu)?.to_dtype(dtype)?,
        );
    }
    if fault == "extra" {
        t.insert(
            "unexpected".into(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu)?,
        );
    }
    candle_core::safetensors::save(&t, dir.join("weights.safetensors"))?;
    Ok(())
}

pub fn set_winner(dir: &Path, winner: usize) -> Result<()> {
    let path = dir.join("weights.safetensors");
    let mut tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
    let mut values = vec![0f32; 16];
    for index in 0..4 {
        values[winner * 4 + index] = 10.0;
    }
    tensors.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(values, (4, 4), &Device::Cpu)?,
    );
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}

pub fn set_two_candidates(dir: &Path) -> Result<()> {
    let path = dir.join("weights.safetensors");
    let mut tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
    let mut values = vec![-4.0f32; 16];
    for winner in [1, 3] {
        for index in 0..4 {
            values[winner * 4 + index] = 0.0;
        }
    }
    tensors.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(values, (4, 4), &Device::Cpu)?,
    );
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}
