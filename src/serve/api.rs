use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    inference::engine::{GenerateOptions, InferenceEngine},
    ks::evidence::AxisScores,
    ks::verify::VerifiedAnswer,
};

#[derive(Clone)]
struct ApiState {
    engine: Arc<InferenceEngine>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

pub async fn run_server(addr: SocketAddr, engine: Arc<InferenceEngine>) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

fn router(engine: Arc<InferenceEngine>) -> Router {
    let state = ApiState { engine };
    Router::new()
        .route("/health", get(health))
        .route("/v1/medical/generate", post(generate_medical))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(HealthResponse {
            status: "unavailable",
        }),
    )
}

async fn generate_medical(
    State(state): State<ApiState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let options = GenerateOptions {
        max_new_tokens: req.max_new_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(0.8),
        top_p: req.top_p.unwrap_or(0.95),
    };

    let result = state.engine.generate_verified(&req.prompt, options);
    match result {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(VerifiedAnswer {
                clinical_use: false,
                answer: "Generation unavailable".to_string(),
                evidence_level: crate::ks::evidence::EvidenceLevel::D,
                axis_scores: AxisScores::new(0.0),
                composite_score: 0.0,
                sources: vec![],
                confidence: 0.0,
                contraindications: vec!["internal_error".to_string()],
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn unavailable_untrained_model_reports_503() {
        let config = ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        };
        let engine = Arc::new(InferenceEngine::new(config, true).expect("synthetic engine"));
        let request = GenerateRequest {
            prompt: "test".into(),
            max_new_tokens: Some(0),
            temperature: None,
            top_p: None,
        };
        assert_eq!(
            generate_medical(State(ApiState { engine }), Json(request))
                .await
                .into_response()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn health_reports_unavailable_without_model() {
        assert_eq!(
            health().await.into_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn verified_generation_rejects_untrained_engine() {
        let config = ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        };
        let engine = InferenceEngine::new(config, true).expect("synthetic engine");
        assert!(engine
            .generate_verified(
                "test",
                GenerateOptions {
                    max_new_tokens: 0,
                    ..Default::default()
                }
            )
            .is_err());
    }

    #[tokio::test]
    async fn loopback_routes_report_unavailable() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let config = ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
        };
        let engine = Arc::new(InferenceEngine::new(config, true).expect("synthetic engine"));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router(engine)).await });
        async fn request(addr: std::net::SocketAddr, request: &str) -> String {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut bytes = Vec::new();
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                stream.read_to_end(&mut bytes),
            )
            .await
            .unwrap()
            .unwrap();
            String::from_utf8(bytes).unwrap()
        }
        let health = request(
            addr,
            "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        let body = r#"{"prompt":"test","max_new_tokens":0}"#;
        let post = format!("POST /v1/medical/generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
        let generation = request(addr, &post).await;
        server.abort();
        assert!(health.starts_with("HTTP/1.1 503"));
        assert!(health.contains("\"status\":\"unavailable\""));
        assert!(generation.starts_with("HTTP/1.1 503"));
        assert!(generation.contains("Generation unavailable"));
        assert!(generation.contains("\"clinical_use\":false"));
    }
}
