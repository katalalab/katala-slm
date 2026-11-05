use anyhow::Result;
use serde_json::Value;
use std::{
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};
mod support;
use rand::{rngs::StdRng, Rng, SeedableRng};
use support::{fixture, set_two_candidates, set_winner};

fn run(bundle: &Path, extra: &[&str], input: &[u8]) -> Result<Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_katala-slm"))
        .args(["--cpu", "infer", "--research", "--model"])
        .arg(bundle)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let written = child.stdin.take().unwrap().write_all(input);
    let output = child.wait_with_output()?;
    if let Err(error) = written {
        if error.kind() != std::io::ErrorKind::BrokenPipe {
            return Err(error.into());
        }
    }
    Ok(output)
}

#[test]
fn synthetic_bundle_cli_generates_seeded_json_and_eos() -> Result<()> {
    let word = tempfile::tempdir()?;
    let eos = tempfile::tempdir()?;
    fixture(word.path(), false, "")?;
    fixture(eos.path(), false, "")?;
    set_winner(word.path(), 3)?;
    set_winner(eos.path(), 2)?;
    let args = [
        "--prompt-stdin",
        "--max-new-tokens",
        "2",
        "--seed",
        "42",
        "--temperature",
        "1",
        "--top-p",
        "0.5",
    ];
    let first = run(word.path(), &args, b"hello")?;
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let out: Value = serde_json::from_slice(&first.stdout)?;
    assert_eq!(out["mode"], "research_unverified");
    assert_eq!(out["clinical_use"], false);
    assert_eq!(out["includes_prompt"], true);
    assert_eq!(out["text"], "hello world world");
    let repeat = run(word.path(), &args, b"hello")?;
    assert_eq!(first.stdout, repeat.stdout);
    let early = run(eos.path(), &args, b"hello")?;
    assert!(early.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&early.stdout)?["text"],
        "hello"
    );
    Ok(())
}

#[test]
fn cli_seed_selects_from_two_retained_candidates() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    set_two_candidates(dir.path())?;
    let low = (0..100u64)
        .find(|seed| StdRng::seed_from_u64(*seed).random::<f64>() < 0.25)
        .unwrap();
    let high = (0..100u64)
        .find(|seed| StdRng::seed_from_u64(*seed).random::<f64>() > 0.75)
        .unwrap();
    for (seed, expected) in [(low, "hello hello"), (high, "hello world")] {
        let seed = seed.to_string();
        let out = run(
            dir.path(),
            &[
                "--prompt-stdin",
                "--max-new-tokens",
                "1",
                "--seed",
                &seed,
                "--temperature",
                "1",
                "--top-p",
                "0.9",
            ],
            b"hello",
        )?;
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout)?["text"],
            expected
        );
    }
    Ok(())
}

#[test]
fn prompt_argument_and_zero_tokens_return_explicit_prompt_inclusion() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    let out = run(
        dir.path(),
        &["--prompt", "hello", "--max-new-tokens", "0"],
        b"",
    )?;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let output: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(output["text"], "hello");
    assert_eq!(output["includes_prompt"], true);
    Ok(())
}

#[test]
fn failures_never_emit_success_json() -> Result<()> {
    let good = tempfile::tempdir()?;
    fixture(good.path(), false, "")?;
    let bad = tempfile::tempdir()?;
    fixture(bad.path(), false, "shape")?;
    let malformed = tempfile::tempdir()?;
    fixture(malformed.path(), false, "")?;
    std::fs::write(malformed.path().join("tokenizer.json"), b"invalid")?;
    for (bundle, args, input) in [
        (
            good.path(),
            vec!["--prompt-stdin", "--max-new-tokens", "129"],
            b"hello".as_slice(),
        ),
        (
            good.path(),
            vec!["--prompt-stdin", "--temperature", "0"],
            b"hello".as_slice(),
        ),
        (
            good.path(),
            vec!["--prompt-stdin", "--top-p", "1.1"],
            b"hello".as_slice(),
        ),
        (
            good.path(),
            vec!["--prompt-stdin", "--prompt", "hello"],
            b"hello".as_slice(),
        ),
        (good.path(), vec!["--prompt-stdin"], b"".as_slice()),
        (bad.path(), vec!["--prompt-stdin"], b"hello".as_slice()),
        (
            malformed.path(),
            vec!["--prompt-stdin"],
            b"hello".as_slice(),
        ),
        (good.path(), vec!["--prompt-stdin"], b"\xff".as_slice()),
        (good.path(), vec![], b"".as_slice()),
    ] {
        let out = run(bundle, &args, input)?;
        assert!(!out.status.success(), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
    }
    let too_large = vec![b'x'; 65_537];
    let out = run(good.path(), &["--prompt-stdin"], &too_large)?;
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    let marker = b"SYNTHETIC_PRIVATE_PROMPT";
    let out = run(
        good.path(),
        &["--prompt-stdin", "--temperature", "0"],
        marker,
    )?;
    assert!(!out.status.success());
    assert!(!out
        .stdout
        .windows(marker.len())
        .any(|window| window == marker));
    assert!(!out
        .stderr
        .windows(marker.len())
        .any(|window| window == marker));
    Ok(())
}

#[test]
fn research_requires_explicit_bundle_and_normal_mode_stays_disabled() -> Result<()> {
    let normal = Command::new(env!("CARGO_BIN_EXE_katala-slm"))
        .args(["--cpu", "infer", "--prompt", "hello"])
        .output()?;
    assert!(!normal.status.success());
    assert!(normal.stdout.is_empty());
    let missing = Command::new(env!("CARGO_BIN_EXE_katala-slm"))
        .args(["--cpu", "infer", "--research", "--prompt", "hello"])
        .output()?;
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
    Ok(())
}

#[test]
fn cli_rejects_tokenizer_truncation_padding_and_dropout() -> Result<()> {
    use tokenizers::{
        pre_tokenizers::whitespace::Whitespace, PaddingParams, PaddingStrategy, Tokenizer,
        TruncationParams,
    };
    for padding in [false, true] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let path = dir.path().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        if padding {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(3),
                ..Default::default()
            }));
        } else {
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 1,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let ids = tokenizer
            .encode("hello world", true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(
            ids.get_ids(),
            if padding { &[1, 3, 0][..] } else { &[1][..] }
        );
        tokenizer
            .save(&path, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = run(
            dir.path(),
            &["--prompt-stdin", "--max-new-tokens", "0"],
            b"hello world",
        )?;
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("failed to load research bundle"));
    }
    for (dropout, reject) in [(0.5, true), (0.0, false)] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let path = dir.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        tokenizer["model"] = serde_json::json!({"type":"BPE", "dropout":dropout,
            "unk_token":"<unk>", "continuing_subword_prefix":null, "end_of_word_suffix":null,
            "fuse_unk":false, "byte_fallback":false, "ignore_merges":false,
            "vocab":{"<unk>":0,"a":1,"<eos>":2,"aa":3}, "merges":[["a","a"]]});
        std::fs::write(&path, serde_json::to_vec(&tokenizer)?)?;
        Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = run(
            dir.path(),
            &["--prompt-stdin", "--max-new-tokens", "0"],
            b"a",
        )?;
        assert_eq!(out.status.success(), !reject);
        if reject {
            assert!(out.stdout.is_empty());
        } else {
            assert_eq!(serde_json::from_slice::<Value>(&out.stdout)?["text"], "a");
        }
    }
    Ok(())
}

#[test]
fn cli_rejects_eos_out_of_vocab_and_over_context_prompt() -> Result<()> {
    use tokenizers::{
        pre_tokenizers::whitespace::Whitespace, processors::template::TemplateProcessing, Tokenizer,
    };
    for mode in ["eos_post", "eos_input", "out_of_vocab", "over_context"] {
        let dir = tempfile::tempdir()?;
        fixture(dir.path(), false, "")?;
        let path = dir.path().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
        if mode == "over_context" {
            tokenizer.with_pre_tokenizer(Some(Whitespace));
        }
        if mode == "eos_post" || mode == "out_of_vocab" {
            let (template, token, id) = if mode == "eos_post" {
                ("$A <eos>", "<eos>", 2)
            } else {
                ("world $A", "world", 99)
            };
            let processor = TemplateProcessing::builder()
                .try_single(template)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .special_tokens(vec![(token, id)])
                .build()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            tokenizer.with_post_processor(Some(processor));
        }
        tokenizer
            .save(&path, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompt = match mode {
            "eos_input" => "<eos>".to_string(),
            "over_context" => vec!["hello"; 17].join(" "),
            _ => "hello".to_string(),
        };
        for limit in ["0", "1"] {
            let out = run(
                dir.path(),
                &["--prompt-stdin", "--max-new-tokens", limit],
                prompt.as_bytes(),
            )?;
            assert!(!out.status.success(), "{mode}, max={limit}");
            assert!(out.stdout.is_empty(), "{mode}, max={limit}");
            assert!(
                String::from_utf8_lossy(&out.stderr).contains("research generation failed"),
                "{mode}, max={limit}"
            );
        }
    }
    Ok(())
}

#[test]
fn cli_accepts_bos_and_still_generates_seeded_output() -> Result<()> {
    use tokenizers::{processors::template::TemplateProcessing, AddedToken, Tokenizer};
    let dir = tempfile::tempdir()?;
    fixture(dir.path(), false, "")?;
    set_winner(dir.path(), 1)?;
    let path = dir.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer
        .add_special_tokens([AddedToken::from("world", true)])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let processor = TemplateProcessing::builder()
        .try_single("world $A")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .special_tokens(vec![("world", 3)])
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_post_processor(Some(processor));
    tokenizer
        .save(&path, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    assert_eq!(
        tokenizer
            .encode("hello", true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids(),
        &[3, 1]
    );
    let out = run(
        dir.path(),
        &[
            "--prompt-stdin",
            "--max-new-tokens",
            "1",
            "--seed",
            "42",
            "--temperature",
            "1",
            "--top-p",
            "0.5",
        ],
        b"hello",
    )?;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let output: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(output["text"], "hello hello");
    assert_eq!(output["clinical_use"], false);
    Ok(())
}
