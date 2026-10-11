use anyhow::Result;
use candle_core::{DType, Tensor};
use katala_slm::model::{config::ModelConfig, transformer::TransformerModel};

#[test]
fn transformer_forward_has_expected_shape() -> Result<()> {
    let config = ModelConfig {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };
    let model = TransformerModel::new(config.clone(), true)?;

    let batched_input = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], (2, 4), &model.device)?
        .to_dtype(DType::U32)?;
    let batched_logits = model.forward(&batched_input)?.to_vec3::<f32>()?;
    assert_eq!(batched_logits.len(), 2);
    assert_eq!(batched_logits[0].len(), 4);
    assert_eq!(batched_logits[0][0].len(), config.vocab_size);

    for (idx, tokens) in [[1u32, 2, 3, 4], [5u32, 6, 7, 8]].iter().enumerate() {
        let input =
            Tensor::from_vec(tokens.to_vec(), (1, 4), &model.device)?.to_dtype(DType::U32)?;
        let logits = model.forward(&input)?.to_vec3::<f32>()?;
        let max_delta = batched_logits[idx]
            .iter()
            .flatten()
            .zip(logits[0].iter().flatten())
            .map(|(batched, single)| (batched - single).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta < 1e-4, "batch={idx} max_delta={max_delta}");
    }
    Ok(())
}
