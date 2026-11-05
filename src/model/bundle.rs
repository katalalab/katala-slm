//! Strict local, CPU-only research bundle. Not a clinical model or a format importer.
use super::{
    config::ModelConfig,
    transformer::{attention_fits, TransformerModel},
};
use crate::data::tokenizer::TokenizerWrapper;
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::Read,
    path::Path,
};

const MAX_JSON: u64 = 1_048_576;
const MAX_WEIGHTS: u64 = 268_435_456;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    format: String,
    eos_token_id: u32,
    model: ModelConfig,
}

pub struct ResearchBundle {
    pub model: TransformerModel,
    pub tokenizer: TokenizerWrapper,
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= limit,
        "bundle file exceeds size limit"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "bundle file exceeds size limit"
    );
    Ok(bytes)
}

fn expected(c: &ModelConfig) -> Result<HashMap<String, Vec<usize>>> {
    let h = c.hidden_size;
    let i = c.intermediate_size;
    let v = c.vocab_size;
    let n = c.num_attention_heads;
    let k = c.num_key_value_heads;
    ensure!(
        v > 0 && v <= 131_072 && h > 0 && h <= 4096 && i > 0 && i <= 16_384,
        "invalid dimensions"
    );
    ensure!(
        c.num_hidden_layers > 0 && c.num_hidden_layers <= 64 && n > 0 && k > 0,
        "invalid layer or head count"
    );
    ensure!(
        h.is_multiple_of(n) && n.is_multiple_of(k) && (h / n).is_multiple_of(2),
        "invalid GQA dimensions"
    );
    ensure!(
        c.max_position_embeddings > 0
            && c.max_position_embeddings <= 2048
            && attention_fits(1, n, c.max_position_embeddings),
        "invalid context limit"
    );
    ensure!(
        c.rope_theta.is_finite()
            && c.rope_theta > 0.0
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0,
        "invalid numeric config"
    );
    let eps = c.rms_norm_eps as f32;
    ensure!(
        eps.is_finite() && eps > 0.0,
        "epsilon is not positive finite f32"
    );
    let d = h / n;
    for index in 0..d / 2 {
        let denom = c.rope_theta.powf((2 * index) as f64 / d as f64) as f32;
        let inverse = 1.0_f32 / denom;
        let max_angle = inverse * (c.max_position_embeddings - 1) as f32;
        ensure!(
            denom.is_finite() && denom > 0.0 && inverse.is_finite() && max_angle.is_finite(),
            "invalid f32 rotary frequency"
        );
    }
    let kv = k * (h / n);
    let mut shapes = HashMap::new();
    shapes.insert("embed.token_embedding.weight".into(), vec![v, h]);
    shapes.insert("final_norm.rms_norm.weight".into(), vec![h]);
    shapes.insert("lm_head.weight".into(), vec![v, h]);
    for layer in 0..c.num_hidden_layers {
        for (name, shape) in [
            ("attn_norm.rms_norm.weight", vec![h]),
            ("attention.q_proj.weight", vec![h, h]),
            ("attention.k_proj.weight", vec![kv, h]),
            ("attention.v_proj.weight", vec![kv, h]),
            ("attention.o_proj.weight", vec![h, h]),
            ("mlp_norm.rms_norm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![i, h]),
            ("mlp.up_proj.weight", vec![i, h]),
            ("mlp.down_proj.weight", vec![h, i]),
        ] {
            shapes.insert(format!("layers.{layer}.{name}"), shape);
        }
    }
    let elements = shapes
        .values()
        .try_fold(0usize, |sum, shape| {
            shape
                .iter()
                .try_fold(1usize, |n, x| n.checked_mul(*x))
                .and_then(|n| sum.checked_add(n))
        })
        .context("tensor size overflow")?;
    ensure!(
        elements <= (MAX_WEIGHTS as usize / 4),
        "model exceeds weight budget"
    );
    Ok(shapes)
}

impl ResearchBundle {
    /// Loads only the exact v1 local format on CPU; no fallback to random weights.
    pub fn load(root: &Path) -> Result<Self> {
        let config_bytes = read_bounded(&root.join("config.json"), MAX_JSON)?;
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let model = raw
            .get("model")
            .and_then(|v| v.as_object())
            .context("missing model config")?;
        let names: BTreeSet<&str> = model.keys().map(String::as_str).collect();
        let expected_names: BTreeSet<&str> = [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
        ]
        .into_iter()
        .collect();
        ensure!(
            names == expected_names,
            "unexpected or missing model config fields"
        );
        let manifest: BundleManifest = serde_json::from_slice(&config_bytes)?;
        ensure!(
            manifest.format == "katala-slm-research-v1",
            "unsupported bundle format"
        );
        let shapes = expected(&manifest.model)?;
        let tokenizer_bytes = read_bounded(&root.join("tokenizer.json"), MAX_JSON)?;
        let tokenizer = TokenizerWrapper::from_research_bytes(
            &tokenizer_bytes,
            manifest.eos_token_id,
            manifest.model.vocab_size,
        )?;
        let bytes = read_bounded(&root.join("weights.safetensors"), MAX_WEIGHTS)?;
        let checkpoint = candle_core::safetensors::SliceSafetensors::new(&bytes)?;
        let views = checkpoint.tensors();
        ensure!(
            views.len() == shapes.len(),
            "checkpoint tensor count mismatch"
        );
        for (name, view) in &views {
            let shape = shapes.get(name).context("unexpected checkpoint tensor")?;
            ensure!(
                DType::try_from(view.dtype())? == DType::F32 && view.shape() == shape.as_slice(),
                "checkpoint tensor dtype or shape mismatch"
            );
        }
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, _) in views {
            let tensor = checkpoint.load(&name, &Device::Cpu)?;
            ensure!(
                tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()),
                "non-finite checkpoint weight"
            );
            tensors.insert(name, tensor);
        }
        let model = TransformerModel::from_tensors(manifest.model, tensors)?;
        Ok(Self { model, tokenizer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_read_rejects_oversized_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("file");
        std::fs::write(&path, b"123456789")?;
        assert!(read_bounded(&path, 8).is_err());
        assert_eq!(read_bounded(&path, 9)?, b"123456789");
        Ok(())
    }

    #[test]
    fn f32_rotary_and_epsilon_validation_rejects_underflow_and_overflow() {
        let mut c = ModelConfig {
            vocab_size: 4,
            hidden_size: 8,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        };
        assert!(expected(&c).is_ok());
        c.rope_theta = 1e-300;
        assert!(expected(&c).is_err());
        c.rope_theta = 1e300;
        assert!(expected(&c).is_err());
        c.rope_theta = 10_000.0;
        c.rms_norm_eps = 1e-300;
        assert!(expected(&c).is_err());
    }
}
