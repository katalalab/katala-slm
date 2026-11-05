use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use katala_slm::model::bundle::ResearchBundle;
use std::{collections::HashMap, fs, path::Path};

fn fixture(dir: &Path, altered: bool, fault: &str) -> Result<()> {
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

#[test]
fn loaded_synthetic_weights_change_logits() -> Result<()> {
    let a = tempfile::tempdir()?;
    let b = tempfile::tempdir()?;
    fixture(a.path(), false, "")?;
    fixture(b.path(), true, "")?;
    let a = ResearchBundle::load(a.path())?;
    let b = ResearchBundle::load(b.path())?;
    let input = Tensor::new(&[[1u32]], &Device::Cpu)?;
    let x = a.model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    let y = b.model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    assert_ne!(x, y);
    assert_eq!(a.tokenizer.encode("hello")?, vec![1]);
    assert_eq!(a.tokenizer.eos_token_id(), 2);
    Ok(())
}

#[test]
fn malformed_checkpoints_are_rejected() -> Result<()> {
    for fault in ["missing", "extra", "shape", "dtype", "nonfinite"] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, fault)?;
        assert!(ResearchBundle::load(dir.path()).is_err(), "{fault}");
    }
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    fs::write(dir.path().join("weights.safetensors"), b"broken")?;
    assert!(ResearchBundle::load(dir.path()).is_err());
    Ok(())
}

#[test]
fn invalid_config_tokenizer_and_eos_are_rejected() -> Result<()> {
    for fault in [
        "version",
        "extra_config",
        "zero_heads",
        "bad_grouping",
        "odd_head",
        "context",
        "eos",
        "tokenizer",
        "budget",
        "rope_underflow",
        "rope_overflow",
        "epsilon_underflow",
    ] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let config_path = dir.path().join("config.json");
        let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        match fault {
            "version" => config["format"] = "foreign".into(),
            "extra_config" => config["model"]["unexpected"] = 1.into(),
            "zero_heads" => config["model"]["num_attention_heads"] = 0.into(),
            "bad_grouping" => config["model"]["num_key_value_heads"] = 3.into(),
            "odd_head" => {
                config["model"]["hidden_size"] = 6.into();
                config["model"]["num_attention_heads"] = 2.into();
            }
            "context" => config["model"]["max_position_embeddings"] = 0.into(),
            "eos" => config["eos_token_id"] = 3.into(),
            "tokenizer" => fs::write(dir.path().join("tokenizer.json"), b"bad")?,
            "budget" => {
                config["model"]["vocab_size"] = 131072.into();
                config["model"]["hidden_size"] = 4096.into();
            }
            "rope_underflow" => {
                config["model"]["hidden_size"] = 8.into();
                config["model"]["rope_theta"] = serde_json::json!(1e-300);
            }
            "rope_overflow" => {
                config["model"]["hidden_size"] = 8.into();
                config["model"]["rope_theta"] = serde_json::json!(1e300);
            }
            "epsilon_underflow" => config["model"]["rms_norm_eps"] = serde_json::json!(1e-300),
            _ => unreachable!(),
        }
        fs::write(config_path, serde_json::to_vec(&config)?)?;
        assert!(ResearchBundle::load(dir.path()).is_err(), "{fault}");
    }
    Ok(())
}

#[test]
fn oversized_metadata_is_rejected() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    fs::write(dir.path().join("config.json"), vec![b' '; 1_048_577])?;
    assert!(ResearchBundle::load(dir.path()).is_err());
    Ok(())
}

#[test]
fn attention_budget_config_boundary_is_enforced() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    config["model"]["max_position_embeddings"] = 1448.into();
    fs::write(&path, serde_json::to_vec(&config)?)?;
    let mut bundle = ResearchBundle::load(dir.path())?;
    bundle.model.config.num_attention_heads = 1;
    let too_many = Tensor::zeros((3, 1024), DType::U32, &Device::Cpu)?;
    assert!(bundle.model.forward(&too_many).is_err());
    config["model"]["max_position_embeddings"] = 1449.into();
    fs::write(&path, serde_json::to_vec(&config)?)?;
    assert!(ResearchBundle::load(dir.path()).is_err());
    Ok(())
}

#[test]
fn loaded_model_rejects_empty_and_overlength_inputs() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let mut bundle = ResearchBundle::load(dir.path())?;
    bundle.model.config.max_position_embeddings = 999;
    let empty = Tensor::zeros((1, 0), DType::U32, &Device::Cpu)?;
    let long = Tensor::zeros((1, 17), DType::U32, &Device::Cpu)?;
    assert!(bundle.model.forward(&empty).is_err());
    assert!(bundle.model.forward(&long).is_err());
    Ok(())
}
