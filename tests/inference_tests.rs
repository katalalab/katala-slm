use anyhow::Result;
use candle_core::{DType, Tensor};
use katala_slm::inference::kv_cache::{KvCache, Quantization};
use katala_slm::model::{config::ModelConfig, transformer::TransformerModel};

fn model_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    }
}

fn assert_logits_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "logit {index}: cached={actual}, full={expected}"
        );
    }
}

#[test]
fn model_generates_logits() -> Result<()> {
    let config = model_config();
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

fn assert_cached_forward_matches_full(force_cpu: bool, require_cuda: bool) -> Result<()> {
    let config = model_config();
    let model = TransformerModel::new(config.clone(), force_cpu)?;
    if require_cuda {
        assert!(
            model.device.is_cuda(),
            "CUDA parity test requires a visible CUDA device"
        );
    }
    let token_ids: Vec<u32> = (1..=12).collect();
    let input = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &model.device)?
        .to_dtype(DType::U32)?;
    let expected = model.forward(&input)?.to_vec3::<f32>()?;
    let mut cache = KvCache::new(
        config.num_hidden_layers,
        config.max_position_embeddings,
        Quantization::None,
    );

    let prefix_len = 4;
    let prefix = Tensor::from_vec(
        token_ids[..prefix_len].to_vec(),
        (1, prefix_len),
        &model.device,
    )?
    .to_dtype(DType::U32)?;
    let actual_prefix = model
        .forward_with_cache(&prefix, &mut cache)?
        .to_vec3::<f32>()?;
    for (actual, expected) in actual_prefix[0].iter().zip(&expected[0][..prefix_len]) {
        assert_logits_close(actual, expected);
    }

    for (position, token) in token_ids.iter().enumerate().skip(prefix_len) {
        let step = Tensor::from_vec(vec![*token], (1, 1), &model.device)?.to_dtype(DType::U32)?;
        let actual = model
            .forward_with_cache(&step, &mut cache)?
            .to_vec3::<f32>()?;
        assert_logits_close(&actual[0][0], &expected[0][position]);
    }
    assert_eq!(cache.sequence_len()?, token_ids.len());
    Ok(())
}

#[test]
fn cached_forward_matches_full_forward_at_each_position() -> Result<()> {
    assert_cached_forward_matches_full(true, false)
}

#[cfg(feature = "cuda")]
#[test]
fn cached_forward_matches_full_forward_on_cuda() -> Result<()> {
    assert_cached_forward_matches_full(false, true)
}

#[test]
fn cached_forward_rejects_context_overflow_without_mutating_cache() -> Result<()> {
    let config = model_config();
    let model = TransformerModel::new(config.clone(), true)?;
    let mut cache = KvCache::new(config.num_hidden_layers, 4, Quantization::None);
    let prefix = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &model.device)?;
    model.forward_with_cache(&prefix, &mut cache)?;

    let extra_token = Tensor::from_vec(vec![5u32], (1, 1), &model.device)?;
    assert!(model.forward_with_cache(&extra_token, &mut cache).is_err());
    assert_eq!(cache.sequence_len()?, 4);
    Ok(())
}
