use anyhow::Result;
use candle_core::{DType, Tensor};
use katala_slm::model::{config::ModelConfig, transformer::TransformerModel};

#[test]
fn model_generates_logits() -> Result<()> {
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
    let model = TransformerModel::new(config.clone(), true)?;
    let input =
        Tensor::from_vec(vec![1u32, 5, 10, 20], (1, 4), &model.device)?.to_dtype(DType::U32)?;
    let logits = model.forward(&input)?;
    let (b, s, v) = logits.dims3()?;
    assert_eq!(b, 1);
    assert_eq!(s, 4);
    assert_eq!(v, config.vocab_size);
    Ok(())
}

#[test]
fn research_engine_reports_empty_encoding_before_forward() -> Result<()> {
    use katala_slm::inference::engine::{GenerateOptions, InferenceEngine};
    let engine = InferenceEngine::new(
        ModelConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            ..Default::default()
        },
        true,
    )?;
    let err = engine
        .generate("test", GenerateOptions::default())
        .unwrap_err();
    assert!(err.to_string().contains("no tokens"));
    Ok(())
}

#[test]
fn cli_requires_checkpoint_and_rejects_an_incomplete_directory() {
    use std::process::Command;
    let dir = tempfile::tempdir().unwrap();
    let infer = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_katala-slm"));
        cmd.args(["--cpu", "infer", "--prompt", "test"]);
        cmd
    };
    let missing = infer().output().unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("--model"));
    let incomplete = infer().arg("--model").arg(dir.path()).output().unwrap();
    assert!(!incomplete.status.success());
    assert!(String::from_utf8_lossy(&incomplete.stderr).contains("config.json"));
}
