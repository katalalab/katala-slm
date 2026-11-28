use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Passages supplied by the caller. Candidate sources found by a keyword search
/// do not establish that any claim occurs in those sources.
#[derive(Debug, Default, Deserialize)]
pub struct CitationEvidence {
    pub cited_source_id: Option<String>,
    pub passages: Vec<CitationPassage>,
}

#[derive(Debug, Deserialize)]
pub struct CitationPassage {
    pub source_id: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct CitationRequest {
    pub claim: String,
    #[serde(flatten)]
    pub evidence: CitationEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CitationReview {
    ExactPassage { source_id: String },
    Rebound { source_id: String },
    Abstained { reason: CitationAbstention },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationAbstention {
    MissingPassages,
    GenerationFailed,
    EmptyClaim,
    InvalidPassage,
    NoExactPassage,
    AmbiguousSource,
}

impl Default for CitationReview {
    fn default() -> Self {
        Self::Abstained {
            reason: CitationAbstention::MissingPassages,
        }
    }
}

impl CitationEvidence {
    /// Check full passage identity, ignoring only outer whitespace. Partial
    /// quotations can drop a qualification or negation, so they do not bind.
    pub fn review(&self, claim: &str) -> CitationReview {
        use CitationAbstention::*;
        use CitationReview::*;

        if self.passages.is_empty() {
            return CitationReview::default();
        }
        let claim = claim.trim();
        if claim.is_empty() {
            return Abstained { reason: EmptyClaim };
        }

        let mut ids = HashSet::new();
        let mut matching_ids = Vec::new();
        for passage in &self.passages {
            let id = passage.source_id.as_str();
            if id.is_empty() || id.trim() != id || passage.text.trim().is_empty() || !ids.insert(id)
            {
                return Abstained {
                    reason: InvalidPassage,
                };
            }
            if passage.text.trim() == claim {
                matching_ids.push(id);
            }
        }
        if let Some(cited) = self.cited_source_id.as_deref() {
            if matching_ids.contains(&cited) {
                return ExactPassage {
                    source_id: cited.to_owned(),
                };
            }
        }
        match matching_ids.as_slice() {
            [] => Abstained {
                reason: NoExactPassage,
            },
            [id] => Rebound {
                source_id: (*id).to_owned(),
            },
            _ => Abstained {
                reason: AmbiguousSource,
            },
        }
    }
}
