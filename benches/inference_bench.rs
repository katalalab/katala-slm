use std::{env, hint::black_box, time::Instant};

use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};
use katala_slm::{
    ks::verify::Verifier,
    model::{config::ModelConfig, transformer::TransformerModel},
};

const WARMUP: usize = 2;
const SAMPLES: usize = 5;

fn report(label: &str, device: &Device, mut work: impl FnMut() -> Result<()>) -> Result<()> {
    for _ in 0..WARMUP {
        work()?;
    }
    device.synchronize()?;

    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        work()?;
        device.synchronize()?;
        *sample = start.elapsed().as_secs_f64() * 1_000.0;
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{label:<34} {:>12.2} {:>12.2}",
        samples[SAMPLES / 2],
        samples[0]
    );
    Ok(())
}

fn main() -> Result<()> {
    let requested = match env::var("KATALA_BENCH_DEVICE") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => "cpu".into(),
        Err(error) => return Err(error.into()),
    };
    let force_cpu = match requested.as_str() {
        "cpu" => true,
        "cuda" => {
            ensure!(
                cfg!(feature = "cuda"),
                "CUDA requested; build the benchmark with --features cuda"
            );
            false
        }
        _ => bail!("KATALA_BENCH_DEVICE must be cpu or cuda"),
    };

    let config = ModelConfig {
        vocab_size: 512,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 512,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };

    let model = TransformerModel::new(config.clone(), force_cpu)?;
    ensure!(
        force_cpu || model.device.is_cuda(),
        "CUDA requested; Candle could not initialize CUDA device 0"
    );
    let verifier = Verifier::default();
    println!(
        "Requested: {requested}; Candle device: {:?}; warmup: {WARMUP}; samples: {SAMPLES}",
        model.device.location()
    );
    println!(
        "{:<34} {:>12} {:>12}",
        "Benchmark", "Median (ms)", "Min (ms)"
    );

    for seq_len in [32usize, 128, 512] {
        let input_tokens: Vec<u32> = (0..seq_len as u32)
            .map(|v| v % config.vocab_size as u32)
            .collect();
        let input =
            Tensor::from_vec(input_tokens, (1, seq_len), &model.device)?.to_dtype(DType::U32)?;
        report(
            &format!("Forward pass (seq={seq_len})"),
            &model.device,
            || {
                black_box(model.forward(&input)?);
                Ok(())
            },
        )?;
    }

    let ks_prompt = "Patient on warfarin asks about pain control";
    let ks_answer = "Randomized controlled trial and guideline evidence suggest avoiding NSAID.";
    let ks_sources = vec!["WHO guideline".to_string(), "Meta-analysis".to_string()];
    report("KS verification x1000 (CPU)", &Device::Cpu, || {
        for _ in 0..1_000 {
            black_box(verifier.verify(ks_prompt, ks_answer, &ks_sources));
        }
        Ok(())
    })?;

    for gen_tokens in [10usize, 50, 100] {
        report(
            &format!("Token loop ({gen_tokens} tokens)"),
            &model.device,
            || {
                let mut token_ids: Vec<u32> = vec![1, 2, 3, 4];
                for _ in 0..gen_tokens {
                    let input =
                        Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &model.device)?
                            .to_dtype(DType::U32)?;
                    black_box(model.forward(&input)?);
                    let next =
                        (token_ids.last().copied().unwrap_or(0) + 1) % config.vocab_size as u32;
                    token_ids.push(next);
                }
                Ok(())
            },
        )?;
    }

    Ok(())
}
