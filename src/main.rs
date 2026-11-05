use std::{
    io::{self, Read, Write},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{bail, ensure, Result};
use clap::{Parser, Subcommand};
use katala_slm::{
    inference::{
        engine::{GenerateOptions, InferenceEngine},
        sampler::Sampler,
    },
    ks::verify::Verifier,
    model::bundle::ResearchBundle,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use tracing_subscriber::EnvFilter;

const MAX_PROMPT_BYTES: u64 = 65_536;
const MAX_NEW_TOKENS: usize = 128;

#[derive(Debug, Parser)]
#[command(author, version, about = "Katala SLM CLI")]
struct Cli {
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Infer {
        #[arg(long)]
        research: bool,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, conflicts_with = "prompt_stdin")]
        prompt: Option<String>,
        #[arg(long)]
        prompt_stdin: bool,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        #[arg(long)]
        seed: Option<u64>,
    },
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

#[derive(Serialize)]
struct ResearchOutput {
    mode: &'static str,
    clinical_use: bool,
    includes_prompt: bool,
    text: String,
}

fn ensure_model_available(command: &Commands) -> Result<()> {
    match command {
        Commands::Infer {
            research: true,
            model: Some(_),
            ..
        } => Ok(()),
        _ => bail!("model loading is not supported without an explicit research bundle"),
    }
}

fn read_prompt(prompt: Option<String>, prompt_stdin: bool) -> Result<String> {
    let text = match (prompt, prompt_stdin) {
        (Some(text), false) => text,
        (None, true) => {
            let mut bytes = Vec::new();
            io::stdin()
                .lock()
                .take(MAX_PROMPT_BYTES + 1)
                .read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() as u64 <= MAX_PROMPT_BYTES,
                "prompt exceeds size limit"
            );
            String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("prompt must be UTF-8"))?
        }
        _ => bail!("specify exactly one prompt input"),
    };
    ensure!(
        text.len() as u64 <= MAX_PROMPT_BYTES,
        "prompt exceeds size limit"
    );
    ensure!(!text.trim().is_empty(), "prompt must not be empty");
    Ok(text)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    ensure_model_available(&cli.command)?;
    match cli.command {
        Commands::Infer {
            model: Some(model),
            prompt,
            prompt_stdin,
            max_new_tokens,
            temperature,
            top_p,
            seed,
            ..
        } => {
            ensure!(
                max_new_tokens <= MAX_NEW_TOKENS,
                "max-new-tokens exceeds limit"
            );
            Sampler::new(temperature, top_p).validate()?;
            let prompt = read_prompt(prompt, prompt_stdin)?;
            // Research bundles always load on CPU, regardless of the optional --cpu flag.
            let bundle = ResearchBundle::load(&model)
                .map_err(|_| anyhow::anyhow!("failed to load research bundle"))?;
            let ids = bundle
                .tokenizer
                .encode(&prompt)
                .map_err(|_| anyhow::anyhow!("prompt tokenization failed"))?;
            ensure!(
                !ids.is_empty() && ids.len() <= bundle.model.config.max_position_embeddings,
                "prompt token count exceeds model context or is empty"
            );
            let engine = InferenceEngine {
                model: Arc::new(bundle.model),
                tokenizer: Arc::new(bundle.tokenizer),
                verifier: Arc::new(Verifier::default()),
            };
            let options = GenerateOptions {
                max_new_tokens,
                temperature,
                top_p,
            };
            let text = match seed {
                Some(seed) => {
                    engine.generate_with_rng(&prompt, options, &mut StdRng::seed_from_u64(seed))
                }
                None => engine.generate(&prompt, options),
            }
            .map_err(|_| anyhow::anyhow!("research generation failed"))?;
            let output = ResearchOutput {
                mode: "research_unverified",
                clinical_use: false,
                includes_prompt: true,
                text,
            };
            let json = serde_json::to_string(&output)?;
            writeln!(io::stdout().lock(), "{json}")?;
            Ok(())
        }
        _ => bail!("model loading is not supported"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_reject_before_model_initialization() {
        for args in [
            vec!["katala-slm", "infer", "--prompt", "test"],
            vec![
                "katala-slm",
                "infer",
                "--model",
                "synthetic",
                "--prompt",
                "test",
            ],
            vec!["katala-slm", "serve", "--port", "0"],
        ] {
            let cli = Cli::try_parse_from(args).expect("synthetic CLI arguments");
            assert!(ensure_model_available(&cli.command).is_err());
        }
    }
}
