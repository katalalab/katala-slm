use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use katala_slm::ks::verify::Verifier;
use katala_slm::model::bundle::ResearchBundle;
use katala_slm::{GenerateOptions, InferenceEngine};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::sync::Arc;
use std::{fs, path::Path};

mod support;
use support::{fixture, set_two_candidates, set_winner};

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

fn engine_from(dir: &Path) -> Result<InferenceEngine> {
    let bundle = ResearchBundle::load(dir)?;
    Ok(InferenceEngine {
        model: Arc::new(bundle.model),
        tokenizer: Arc::new(bundle.tokenizer),
        verifier: Arc::new(Verifier::default()),
    })
}

#[derive(Default)]
struct CountingRng {
    draws: usize,
    value: u64,
}

impl RngCore for CountingRng {
    fn next_u32(&mut self) -> u32 {
        self.draws += 1;
        (self.value >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.draws += 1;
        self.value
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.draws += 1;
        for (i, byte) in dest.iter_mut().enumerate() {
            *byte = self.value.to_le_bytes()[i % 8];
        }
    }
}

#[test]
fn loaded_bundle_sampling_uses_caller_rng_across_two_candidates() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    set_two_candidates(dir.path())?;
    let engine = engine_from(dir.path())?;
    let options = GenerateOptions {
        max_new_tokens: 1,
        temperature: 1.0,
        top_p: 0.9,
    };
    let mut low = CountingRng {
        draws: 0,
        value: u64::MAX / 4,
    };
    let mut high = CountingRng {
        draws: 0,
        value: (u64::MAX / 4) * 3,
    };
    let first = engine.generate_with_rng("hello", options, &mut low)?;
    let second = engine.generate_with_rng("hello", options, &mut high)?;
    assert_eq!(first, "hello hello");
    assert_eq!(second, "hello world");
    assert_eq!((low.draws, high.draws), (1, 1));
    Ok(())
}

#[test]
fn loaded_bundles_generate_with_seed_and_honor_eos_and_limit() -> Result<()> {
    let eos_dir = tempfile::tempdir()?;
    let word_dir = tempfile::tempdir()?;
    fixture(eos_dir.path(), false, "")?;
    fixture(word_dir.path(), false, "")?;
    set_winner(eos_dir.path(), 2)?;
    set_winner(word_dir.path(), 3)?;
    let eos = engine_from(eos_dir.path())?;
    let word = engine_from(word_dir.path())?;
    let options = GenerateOptions {
        max_new_tokens: 2,
        temperature: 1.0,
        top_p: 0.5,
    };
    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(42);
    let mut rng3 = StdRng::seed_from_u64(42);
    let eos_output = eos.generate_with_rng("hello", options, &mut rng1)?;
    let word_output = word.generate_with_rng("hello", options, &mut rng2)?;
    assert_eq!(
        word_output,
        word.generate_with_rng("hello", options, &mut rng3)?
    );
    assert_eq!(eos_output, "hello");
    assert_eq!(word_output, "hello world world");
    let one = word.generate_with_rng(
        "hello",
        GenerateOptions {
            max_new_tokens: 1,
            ..options
        },
        &mut StdRng::seed_from_u64(42),
    )?;
    assert_eq!(one, "hello world");
    let mut eos_count = CountingRng::default();
    eos.generate_with_rng("hello", options, &mut eos_count)?;
    assert_eq!(eos_count.draws, 1);
    let mut word_count = CountingRng::default();
    word.generate_with_rng("hello", options, &mut word_count)?;
    assert_eq!(word_count.draws, 2);
    let mut zero_count = CountingRng::default();
    let zero = word.generate_with_rng(
        "hello",
        GenerateOptions {
            max_new_tokens: 0,
            ..options
        },
        &mut zero_count,
    )?;
    assert_eq!(zero, "hello");
    assert_eq!(zero_count.draws, 0);
    assert!(word.generate_verified("hello", options).is_err());
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

#[test]
fn generation_checks_prompt_context_even_when_zero_tokens_requested() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    tokenizer["pre_tokenizer"] = serde_json::json!({"type":"Whitespace"});
    fs::write(&path, serde_json::to_vec(&tokenizer)?)?;
    let engine = engine_from(dir.path())?;
    let prompt = "hello ".repeat(17);
    let options = GenerateOptions {
        max_new_tokens: 0,
        temperature: 1.0,
        top_p: 1.0,
    };
    assert!(engine
        .generate_with_rng(&prompt, options, &mut StdRng::seed_from_u64(1))
        .is_err());
    Ok(())
}
