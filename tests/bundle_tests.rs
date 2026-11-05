use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use katala_slm::ks::verify::Verifier;
use katala_slm::model::bundle::ResearchBundle;
use katala_slm::{GenerateOptions, InferenceEngine};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::sync::Arc;
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

fn set_winner(dir: &Path, winner: usize) -> Result<()> {
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

fn engine_from(dir: &Path) -> Result<InferenceEngine> {
    let bundle = ResearchBundle::load(dir)?;
    Ok(InferenceEngine {
        model: Arc::new(bundle.model),
        tokenizer: Arc::new(bundle.tokenizer),
        verifier: Arc::new(Verifier::default()),
    })
}

#[test]
fn research_bundle_rejects_tokenizer_truncation_and_padding() -> Result<()> {
    use tokenizers::{
        pre_tokenizers::whitespace::Whitespace, PaddingParams, PaddingStrategy, Tokenizer,
        TruncationParams,
    };
    for padding in [false, true] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let path = dir.path().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        if padding {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(3),
                ..Default::default()
            }));
        } else {
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 1,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let ids = tokenizer
            .encode("hello world", true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        assert_eq!(ids, if padding { vec![1, 3, 0] } else { vec![1] });
        tokenizer
            .save(&path, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(
            ResearchBundle::load(dir.path()).is_err(),
            "padding={padding}"
        );
    }
    Ok(())
}

#[test]
fn bos_postprocessor_is_preserved_without_eos() -> Result<()> {
    use tokenizers::{processors::template::TemplateProcessing, AddedToken, Tokenizer};
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer
        .add_special_tokens([AddedToken::from("world", true)])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let processor = TemplateProcessing::builder()
        .try_single("world $A")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .special_tokens(vec![("world", 3)])
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_post_processor(Some(processor));
    tokenizer
        .save(&path, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let engine = engine_from(dir.path())?;
    assert_eq!(engine.tokenizer.encode("hello")?, vec![3, 1]);
    assert_eq!(
        engine.generate(
            "hello",
            GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            }
        )?,
        "hello"
    );
    Ok(())
}

#[test]
fn postprocessor_out_of_vocabulary_ids_are_rejected_even_without_generation() -> Result<()> {
    use tokenizers::{processors::template::TemplateProcessing, Tokenizer};
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let processor = TemplateProcessing::builder()
        .try_single("world $A")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .special_tokens(vec![("world", 99)])
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_post_processor(Some(processor));
    tokenizer
        .save(&path, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut bundle = ResearchBundle::load(dir.path())?;
    bundle.model.config.vocab_size = 100;
    let engine = InferenceEngine {
        model: Arc::new(bundle.model),
        tokenizer: Arc::new(bundle.tokenizer),
        verifier: Arc::new(Verifier::default()),
    };
    for max_new_tokens in [0, 1] {
        assert!(engine
            .generate(
                "hello",
                GenerateOptions {
                    max_new_tokens,
                    ..Default::default()
                }
            )
            .is_err());
    }
    Ok(())
}

#[test]
fn generation_checks_prompt_length_before_zero_or_one_token() -> Result<()> {
    use tokenizers::{pre_tokenizers::whitespace::Whitespace, Tokenizer};
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(&path, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut bundle = ResearchBundle::load(dir.path())?;
    bundle.model.config.max_position_embeddings = 999;
    let engine = InferenceEngine {
        model: Arc::new(bundle.model),
        tokenizer: Arc::new(bundle.tokenizer),
        verifier: Arc::new(Verifier::default()),
    };
    let prompt = vec!["hello"; 17].join(" ");
    let valid_prompt = vec!["hello"; 16].join(" ");
    assert!(engine
        .generate(
            &valid_prompt,
            GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            }
        )
        .is_ok());
    for max_new_tokens in [0, 1] {
        assert!(engine
            .generate(
                &prompt,
                GenerateOptions {
                    max_new_tokens,
                    ..Default::default()
                }
            )
            .is_err());
    }
    Ok(())
}

#[test]
fn generation_rejects_eos_inserted_by_postprocessor_and_empty_prompt() -> Result<()> {
    use tokenizers::{processors::template::TemplateProcessing, Tokenizer};
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let processor = TemplateProcessing::builder()
        .try_single("$A <eos>")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .special_tokens(vec![("<eos>", 2)])
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_post_processor(Some(processor));
    tokenizer
        .save(&path, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let engine = engine_from(dir.path())?;
    for max_new_tokens in [0, 1] {
        let options = GenerateOptions {
            max_new_tokens,
            ..Default::default()
        };
        assert!(engine.generate("hello", options).is_err());
    }
    let empty_dir = tempfile::tempdir()?;
    fixture(empty_dir.path(), false, "")?;
    let empty_engine = engine_from(empty_dir.path())?;
    assert!(empty_engine
        .generate(
            "",
            GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            }
        )
        .is_err());
    assert!(empty_engine
        .generate(
            "<eos>",
            GenerateOptions {
                max_new_tokens: 0,
                ..Default::default()
            }
        )
        .is_err());
    Ok(())
}

#[test]
fn research_bpe_rejects_stochastic_dropout_but_accepts_zero() -> Result<()> {
    for (dropout, reject) in [(0.5, true), (0.0, false)] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let path = dir.path().join("tokenizer.json");
        let mut tokenizer: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        tokenizer["model"] = serde_json::json!({"type":"BPE", "dropout":dropout,
            "unk_token":"<unk>", "continuing_subword_prefix":null, "end_of_word_suffix":null,
            "fuse_unk":false, "byte_fallback":false, "ignore_merges":false,
            "vocab":{"<unk>":0,"a":1,"<eos>":2,"aa":3}, "merges":[["a","a"]]});
        fs::write(&path, serde_json::to_vec(&tokenizer)?)?;
        tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(
            ResearchBundle::load(dir.path()).is_err(),
            reject,
            "dropout={dropout}"
        );
    }
    Ok(())
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
    let path = dir.path().join("weights.safetensors");
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
