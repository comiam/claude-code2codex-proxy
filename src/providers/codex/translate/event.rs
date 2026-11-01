//! Decisions emitted by the shared Responses lifecycle, independent of transport.
use super::reducer::{CodexUsage, StopReason};

#[derive(Debug, Clone)]
pub enum TranslationEvent {
    ThinkingStart {
        index: usize,
    },
    ThinkingDelta {
        index: usize,
        text: String,
    },
    ThinkingSignature {
        index: usize,
        signature: String,
    },
    ThinkingStop {
        index: usize,
    },
    TextStart {
        index: usize,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    TextStop {
        index: usize,
    },
    ToolStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolDelta {
        index: usize,
        partial_json: String,
    },
    ToolStop {
        index: usize,
    },
    SearchComplete {
        index: usize,
        card: super::web_search_compat::SearchCard,
    },
    Progress {
        output_index: Option<usize>,
        item_id: Option<String>,
    },
    Skipped(SkippedEvent),
    Finish {
        stop_reason: StopReason,
        usage: Option<CodexUsage>,
        web_search_requests: usize,
    },
}

/// Payload-free compatibility decision. Identity strings are bounded at capture.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SkippedEvent {
    pub event_type: String,
    pub reason: &'static str,
    pub sequence_number: Option<u64>,
    pub output_index: Option<u64>,
    pub content_index: Option<u64>,
    pub item_id: Option<String>,
}
impl SkippedEvent {
    pub(crate) fn new(payload: &serde_json::Value, reason: &'static str) -> Self {
        Self {
            event_type: payload["type"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(128)
                .collect(),
            reason,
            sequence_number: payload["sequence_number"].as_u64(),
            output_index: payload["output_index"].as_u64(),
            content_index: payload["content_index"].as_u64(),
            item_id: payload
                .get("item_id")
                .or_else(|| payload.pointer("/item/id"))
                .and_then(serde_json::Value::as_str)
                .map(|s| s.chars().take(256).collect()),
        }
    }
}

/// Diagnostics are bounded and payload-free, without per-token log writes.
#[derive(Default)]
pub(crate) struct SkipDiagnostics {
    req_id: Option<String>,
    counts: std::collections::BTreeMap<String, u64>,
    suppressed: u64,
    pub(crate) no_representable_output: bool,
}
impl SkipDiagnostics {
    pub(crate) fn set_request_id(&mut self, req_id: &str) {
        self.req_id = Some(req_id.to_owned());
    }
    pub(crate) fn record(&mut self, event: &SkippedEvent) {
        if let Some(count) = self.counts.get_mut(&event.event_type) {
            *count = count.saturating_add(1);
            return;
        }
        if self.counts.len() >= 16 {
            self.suppressed = self.suppressed.saturating_add(1);
            return;
        }
        self.counts.insert(event.event_type.clone(), 1);
        if let Some(req_id) = &self.req_id {
            let mut fields = serde_json::to_value(event)
                .unwrap()
                .as_object()
                .unwrap()
                .clone();
            fields.insert("reqId".into(), serde_json::json!(req_id));
            crate::logging::create_logger("codex").info("codex_event_skipped", Some(fields));
        }
    }
}
impl Drop for SkipDiagnostics {
    fn drop(&mut self) {
        if let Some(req_id) = &self.req_id
            && !self.counts.is_empty()
        {
            crate::logging::create_logger("codex").info(
                "codex_event_skipped_summary",
                serde_json::json!({
                    "reqId":req_id, "counts":self.counts, "suppressed_count":self.suppressed,
                    "no_representable_output":self.no_representable_output,
                })
                .as_object()
                .cloned(),
            );
        }
    }
}
