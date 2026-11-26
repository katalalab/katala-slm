use std::{net::IpAddr, path::PathBuf, sync::Arc};

use anyhow::Result;
use clap::{Parser, Subcommand};
use katala_slm::{
    inference::engine::{GenerateOptions, InferenceEngine},
    ks::citation::{CitationEvidence, CitationRequest},
    model::config::ModelConfig,
    serve::api::run_server,
};
use tracing_subscriber::EnvFilter;

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
    VerifyCitation,
    Infer {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        citation_evidence: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
    },
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        bind: IpAddr,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let engine = || InferenceEngine::new(ModelConfig::default(), cli.cpu);

    match cli.command {
        Commands::VerifyCitation => {
            let req: CitationRequest = serde_json::from_reader(std::io::stdin().lock())?;
            println!(
                "{}",
                serde_json::to_string(&req.evidence.review(&req.claim))?
            );
        }
        Commands::Infer {
            model: _,
            prompt,
            citation_evidence,
            max_new_tokens,
            temperature,
            top_p,
        } => {
            let options = GenerateOptions {
                max_new_tokens,
                temperature,
                top_p,
            };
            let evidence: Option<CitationEvidence> = citation_evidence
                .map(std::fs::File::open)
                .transpose()?
                .map(serde_json::from_reader)
                .transpose()?;
            let mut answer = engine()?.generate_verified(&prompt, options)?;
            if let Some(evidence) = evidence {
                answer.citation_review = evidence.review(&answer.answer);
            }
            println!("{}", serde_json::to_string_pretty(&answer)?);
        }
        Commands::Serve { bind, port } => {
            run_server((bind, port).into(), Arc::new(engine()?)).await?;
        }
    }

    Ok(())
}
