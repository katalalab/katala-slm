use katala_slm::ks::citation::{CitationAbstention::*, CitationRequest, CitationReview::*};
use serde_json::{json, Value};

fn check(
    claim: &str,
    cited: Option<&str>,
    passages: Value,
) -> katala_slm::ks::citation::CitationReview {
    let request: CitationRequest = serde_json::from_value(json!({
        "claim": claim, "cited_source_id": cited, "passages": passages,
    }))
    .unwrap();
    request.evidence.review(&request.claim)
}

#[test]
fn binds_a_complete_quote_or_repairs_a_unique_binding() {
    let passage = json!([{"source_id":"doc-a#p1","text":"  The line is blue.  "}]);
    let id = "doc-a#p1".to_string();
    assert_eq!(
        check("The line is blue.", Some(&id), passage.clone()),
        ExactPassage {
            source_id: id.clone()
        }
    );
    assert_eq!(
        check("The line is blue.", Some("wrong"), passage.clone()),
        Rebound {
            source_id: id.clone()
        }
    );
    assert_eq!(
        check("The line is blue.", None, passage),
        Rebound { source_id: id }
    );
}

#[test]
fn abstains_on_missing_ambiguous_and_partial_evidence() {
    let passages = json!([{"source_id":"a","text":"The line is blue."},{"source_id":"b","text":"The line is blue."}]);
    assert_eq!(
        check("The line is blue.", None, passages.clone()),
        Abstained {
            reason: AmbiguousSource
        }
    );
    assert_eq!(
        check("The line is blue.", Some("b"), passages.clone()),
        ExactPassage {
            source_id: "b".into()
        }
    );
    assert_eq!(
        check("blue", Some("a"), passages.clone()),
        Abstained {
            reason: NoExactPassage
        }
    );
    assert_eq!(check("", None, passages), Abstained { reason: EmptyClaim });
    assert_eq!(
        check("blue", None, json!([])),
        Abstained {
            reason: MissingPassages
        }
    );
    assert_eq!(
        check(
            "blue",
            Some("a"),
            json!([{"source_id":"a","text":"blue"},{"source_id":"a","text":"green"}])
        ),
        Abstained {
            reason: InvalidPassage
        }
    );
}

#[test]
fn legacy_results_and_prompt_derived_sources_do_not_imply_a_verified_quote() {
    let verified = katala_slm::Verifier::default().verify("influenza", "test", &[]);
    assert!(!verified.sources.is_empty());
    assert_eq!(
        verified.citation_review,
        Abstained {
            reason: MissingPassages
        }
    );
    let mut legacy = serde_json::to_value(verified).unwrap();
    legacy.as_object_mut().unwrap().remove("citation_review");
    let restored: katala_slm::VerifiedAnswer = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        restored.citation_review,
        Abstained {
            reason: MissingPassages
        }
    );
}
