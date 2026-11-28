use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    inference::engine::{GenerateOptions, InferenceEngine},
    ks::citation::{CitationAbstention, CitationEvidence, CitationRequest, CitationReview},
    ks::evidence::AxisScores,
    ks::verify::VerifiedAnswer,
};

type Generate = dyn Fn(&str, GenerateOptions) -> Result<VerifiedAnswer> + Send + Sync;

#[derive(Clone)]
struct ApiState {
    generate: Option<Arc<Generate>>,
    generation_slot: Arc<tokio::sync::Semaphore>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub citation_evidence: CitationEvidence,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub generation_available: bool,
}

pub async fn run_server(addr: SocketAddr, engine: Arc<InferenceEngine>) -> Result<()> {
    serve(
        addr,
        Some(Arc::new(move |prompt, options| {
            engine.generate_verified(prompt, options)
        })),
    )
    .await
}

pub async fn serve_checkpoint(
    addr: SocketAddr,
    model: Option<crate::inference::pretrained::PretrainedLlama>,
) -> Result<()> {
    let generate = model.map(|m| -> Arc<Generate> {
        Arc::new(move |prompt, options| m.generate_verified(prompt, options))
    });
    serve(addr, generate).await
}

async fn serve(addr: SocketAddr, generate: Option<Arc<Generate>>) -> Result<()> {
    let state = ApiState {
        generate,
        generation_slot: Arc::new(tokio::sync::Semaphore::new(1)),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/medical/verify-citation", post(verify_citation))
        .route("/v1/medical/generate", post(generate_medical))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(state): State<ApiState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        generation_available: state.generate.is_some(),
    })
}

async fn verify_citation(Json(req): Json<CitationRequest>) -> impl IntoResponse {
    Json(req.evidence.review(&req.claim))
}

async fn generate_medical(
    State(state): State<ApiState>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let Some(generate) = state.generate else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error":"model_not_configured"})),
        )
            .into_response();
    };
    // Each CPU inference owns a KV cache. Keep health and citation service
    // responsive without queuing an unbounded number of heavyweight jobs.
    let Ok(slot) = state.generation_slot.try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({"error":"model_busy"})),
        )
            .into_response();
    };
    let defaults = GenerateOptions::default();
    let options = GenerateOptions {
        max_new_tokens: req.max_new_tokens.unwrap_or(defaults.max_new_tokens),
        temperature: req.temperature.unwrap_or(defaults.temperature),
        top_p: req.top_p.unwrap_or(defaults.top_p),
    };

    let generated = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        generate(&req.prompt, options)
    })
    .await;
    let generated = generated.map_err(anyhow::Error::from).and_then(|r| r);
    let status = if generated.is_ok() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    let response: VerifiedAnswer = generated
        .map(|mut response| {
            response.citation_review = req.citation_evidence.review(&response.answer);
            response
        })
        .unwrap_or_else(|err| VerifiedAnswer {
            clinical_use: false,
            citation_review: CitationReview::Abstained {
                reason: CitationAbstention::GenerationFailed,
            },
            answer: format!("Generation failed: {err}"),
            evidence_level: crate::ks::evidence::EvidenceLevel::D,
            axis_scores: AxisScores::new(0.0),
            composite_score: 0.0,
            sources: vec![],
            confidence: 0.0,
            contraindications: vec!["internal_error".to_string()],
        });

    (status, Json(response)).into_response()
}
