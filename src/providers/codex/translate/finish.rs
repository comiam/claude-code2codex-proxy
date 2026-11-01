//! Shared finish metadata and bounded replay/snapshot accounting.
use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::reasoning_signature::ReasoningReplay;
use super::reducer::CodexUsage;
use super::request::ResponsesInputItem;

#[derive(Debug, Clone)]
pub struct FinishMetadata {
    /// Trustworthy terminal and fully resolved, supported output, independently
    /// of whether this request needed optional replay storage.
    pub stream_success: bool,
    pub continuation_eligible: bool,
    pub replay_available: bool,
    pub response_id: Option<String>,
    pub terminal_type: String,
    pub usage: Option<CodexUsage>,
    pub unresolved_items: usize,
    pub output_items: Vec<ResponsesInputItem>,
}

pub(super) fn reasoning_input_item(replay: ReasoningReplay) -> ResponsesInputItem {
    ResponsesInputItem::Reasoning {
        id: replay.id,
        summary: Vec::new(),
        encrypted_content: replay.encrypted_content,
    }
}

pub(super) fn parse_codex_usage(response: &serde_json::Value) -> CodexUsage {
    let usage = &response["usage"];
    CodexUsage {
        input_tokens: usage["input_tokens"].as_u64(),
        output_tokens: usage["output_tokens"].as_u64(),
        input_tokens_details_cached: usage["input_tokens_details"]["cached_tokens"].as_u64(),
        output_tokens_details_reasoning: usage["output_tokens_details"]["reasoning_tokens"]
            .as_u64(),
        rate_limits: None,
    }
}

/// Only unresolved item strings and the optional ordered replay need storage.
/// Count string bytes plus a conservative per-item metadata allowance. This is
/// a retained payload budget, not allocator/RSS accounting. Overflow drops the
/// whole optional ledger, never a prefix that might be mistaken for full replay.
pub(super) struct ReplayLedger {
    pub items: BTreeMap<usize, ResponsesInputItem>,
    used: usize,
    limit: usize,
}

impl ReplayLedger {
    pub fn new(limit: usize) -> Self {
        Self {
            items: BTreeMap::new(),
            used: 0,
            limit,
        }
    }

    pub fn charge(&mut self, bytes: usize) -> bool {
        if bytes > self.limit.saturating_sub(self.used) {
            return false;
        }
        self.used += bytes;
        true
    }
}

/// Reconcile delta/done snapshots without retaining already-forwarded bytes.
/// A conflicting snapshot is a protocol failure, not reusable canonical output.
#[derive(Default)]
pub(super) struct SnapshotProgress {
    bytes: usize,
    digest: Sha256,
}

impl SnapshotProgress {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn append(&mut self, text: &str) {
        self.bytes += text.len();
        self.digest.update(text.as_bytes());
    }

    pub fn missing_suffix<'a>(&self, snapshot: &'a str) -> Result<&'a str, String> {
        let Some(prefix) = snapshot.get(..self.bytes) else {
            return Err("Conflicting Codex output snapshot".to_string());
        };
        if Sha256::digest(prefix.as_bytes()) != self.digest.clone().finalize() {
            return Err("Conflicting Codex output snapshot".to_string());
        }
        Ok(&snapshot[self.bytes..])
    }
}
