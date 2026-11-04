use std::process::Command;

#[test]
fn unsupported_model_path_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_katala-slm"))
        .args([
            "--cpu",
            "infer",
            "--model",
            "synthetic-model.safetensors",
            "--prompt",
            "test",
            "--max-new-tokens",
            "0",
        ])
        .output()
        .expect("run CLI");
    assert!(
        !output.status.success(),
        "unsupported model must not be silently ignored"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("model loading is not supported"),
        "expected an explicit model-loading error, got: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn inference_without_a_loaded_model_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_katala-slm"))
        .args([
            "--cpu",
            "infer",
            "--prompt",
            "test",
            "--max-new-tokens",
            "0",
        ])
        .output()
        .expect("run CLI");
    assert!(
        !output.status.success(),
        "untrained model must not report successful inference"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("model loading is not supported"));
}
