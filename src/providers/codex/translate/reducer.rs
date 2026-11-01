use super::IncompleteResponsePolicy;
use super::event::TranslationEvent;
pub use super::finish::FinishMetadata;
use super::request::ResponsesInputItem;
use super::state::{ResponseState, TranslationError};
use crate::anthropic::sse::parse_sse_events;
use crate::providers::codex::events::{CodexFailureKind, classify_event_failure};

#[derive(Debug, Clone)]
pub struct UpstreamStreamError {
    pub kind: UpstreamErrorKind,
    pub message: String,
    pub retry_after_seconds: Option<u64>,
    pub diagnostics: Option<UpstreamStreamDiagnostics>,
}

impl std::fmt::Display for UpstreamStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "upstream error: {} ({:?})", self.message, self.kind)
    }
}

impl std::error::Error for UpstreamStreamError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpstreamErrorKind {
    RateLimit,
    Overloaded,
    Transient,
    Failed,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct UpstreamStreamDiagnostics {
    pub event_count: usize,
    pub last_event_type: Option<String>,
    pub saw_terminal_event: bool,
    pub open_blocks: Vec<OpenBlockDiagnostic>,
    pub source: Option<&'static str>,
    pub phase: Option<&'static str>,
    pub upstream_code: Option<String>,
    pub upstream_type: Option<String>,
    pub upstream_param: Option<String>,
    pub upstream_status: Option<u16>,
    pub client_status: Option<u16>,
    pub request_id: Option<String>,
    pub response_id: Option<String>,
    pub semantic_committed: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OpenBlockDiagnostic {
    pub output_index: usize,
    pub anthropic_index: usize,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argument_bytes: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct CodexUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub input_tokens_details_cached: Option<u64>,
    pub output_tokens_details_reasoning: Option<u64>,
    /// Latest `codex.rate_limits` snapshot seen on the stream that produced
    /// this usage. Codex sends it as its own event before any usage exists,
    /// so the translator carries it here for the mapper.
    pub rate_limits: Option<Box<CodexRateLimits>>,
}

/// Subscription meter summary from a `codex.rate_limits` stream event.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CodexRateLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_reached: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<CodexRateLimitWindow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary: Option<CodexRateLimitWindow>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CodexRateLimitWindow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_minutes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_at: Option<u64>,
}

/// Parse a `codex.rate_limits` event payload. Returns `None` when the event
/// carries no `rate_limits` object or no recognized, well-typed fields.
pub fn parse_codex_rate_limits(payload: &serde_json::Value) -> Option<CodexRateLimits> {
    let limits = payload.get("rate_limits")?.as_object()?;
    let window = |name: &str| -> Option<CodexRateLimitWindow> {
        let w = limits.get(name)?.as_object()?;
        let window = CodexRateLimitWindow {
            used_percent: w.get("used_percent").and_then(|v| v.as_f64()),
            window_minutes: w.get("window_minutes").and_then(|v| v.as_u64()),
            reset_at: w.get("reset_at").and_then(|v| v.as_u64()),
        };
        (window != CodexRateLimitWindow::default()).then_some(window)
    };
    let snapshot = CodexRateLimits {
        plan_type: payload
            .get("plan_type")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        limit_reached: limits.get("limit_reached").and_then(|v| v.as_bool()),
        primary: window("primary"),
        secondary: window("secondary"),
    };
    (snapshot != CodexRateLimits::default()).then_some(snapshot)
}

pub type StopReason = &'static str;
pub const STOP_END_TURN: &str = "end_turn";
pub const STOP_TOOL_USE: &str = "tool_use";
pub const STOP_MAX_TOKENS: &str = "max_tokens";

pub type TerminalType = &'static str;
pub const TERM_COMPLETED: &str = "response.completed";
pub const TERM_INCOMPLETE: &str = "response.incomplete";
pub const TERM_DONE: &str = "response.done";

#[derive(Debug, Clone)]
pub enum ReducerEvent {
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
    ToolProgress {
        index: usize,
    },
    Progress,
    WebSearch {
        index: usize,
        result_index: usize,
        id: String,
        query: String,
    },
    Finish {
        stop_reason: StopReason,
        terminal_type: String,
        continuation_eligible: bool,
        usage: Option<CodexUsage>,
        web_search_requests: usize,
        response_id: Option<String>,
        output_items: Vec<ResponsesInputItem>,
    },
}

// Protocol errors expose structured diagnostics to callers.
#[allow(clippy::result_large_err)]
pub fn finish_metadata_from_upstream(
    input: &[u8],
) -> Result<Option<FinishMetadata>, UpstreamStreamError> {
    translate_upstream_bytes_with_policy(input, IncompleteResponsePolicy::Error)
        .map(|(_, finish)| finish)
}

#[allow(clippy::result_large_err)]
pub fn reduce_upstream_bytes(input: &[u8]) -> Result<Vec<ReducerEvent>, UpstreamStreamError> {
    reduce_upstream_bytes_with_policy(input, IncompleteResponsePolicy::Error)
}

#[allow(clippy::result_large_err)]
pub(crate) fn reduce_upstream_bytes_with_policy(
    input: &[u8],
    policy: IncompleteResponsePolicy,
) -> Result<Vec<ReducerEvent>, UpstreamStreamError> {
    let (events, metadata) = translate_upstream_bytes_with_policy(input, policy)?;
    Ok(events
        .into_iter()
        .filter_map(|event| project_legacy(event, metadata.as_ref()))
        .collect())
}

#[allow(clippy::result_large_err)]
pub(crate) fn translate_upstream_bytes_with_policy(
    input: &[u8],
    policy: IncompleteResponsePolicy,
) -> Result<(Vec<TranslationEvent>, Option<FinishMetadata>), UpstreamStreamError> {
    translate_upstream_bytes_with_request_id(input, policy, None)
}

#[allow(clippy::result_large_err)]
pub(crate) fn translate_upstream_bytes_with_request_id(
    input: &[u8],
    policy: IncompleteResponsePolicy,
    req_id: Option<&str>,
) -> Result<(Vec<TranslationEvent>, Option<FinishMetadata>), UpstreamStreamError> {
    // The buffered facade already owns the complete response. Canonical replay
    // still has the same bounded optional ledger as the live path.
    let mut state = ResponseState::new()
        .with_incomplete_response_policy(policy)
        .with_replay_budget(Some(8 * 1024 * 1024));
    if let Some(req_id) = req_id {
        state.set_request_id(req_id);
    }
    let mut out = Vec::new();
    let mut diagnostics = UpstreamStreamDiagnostics {
        source: Some("buffered_reducer"),
        ..Default::default()
    };
    for frame in parse_sse_events(input) {
        let data = frame.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let payload: serde_json::Value =
            serde_json::from_str(data).map_err(|_| UpstreamStreamError {
                kind: UpstreamErrorKind::Failed,
                message: "malformed Codex event JSON".into(),
                retry_after_seconds: None,
                diagnostics: Some(UpstreamStreamDiagnostics {
                    phase: Some("decode"),
                    ..diagnostics.clone()
                }),
            })?;
        diagnostics.event_count += 1;
        diagnostics.last_event_type = payload["type"].as_str().map(str::to_owned);
        diagnostics.saw_terminal_event |=
            crate::providers::codex::events::event_is_terminal(&payload);
        crate::providers::codex::rate_limits::record_event(&payload);
        match state.accept(&payload) {
            Ok(events) => out.extend(events),
            Err(error) => {
                diagnostics.open_blocks = state.open_blocks();
                diagnostics.semantic_committed = state.has_semantic_output();
                diagnostics.phase = Some("terminal_validation");
                let mut kind = if error.as_str().contains("unsupported Codex sequence") {
                    UpstreamErrorKind::Failed
                } else {
                    UpstreamErrorKind::Transient
                };
                let mut retry_after_seconds = None;
                if let TranslationError::Upstream(_) = error {
                    if let Some(failure) = classify_event_failure(&payload) {
                        kind = match failure.kind {
                            CodexFailureKind::RateLimit => UpstreamErrorKind::RateLimit,
                            CodexFailureKind::Overloaded => UpstreamErrorKind::Overloaded,
                            CodexFailureKind::Transient => UpstreamErrorKind::Transient,
                            _ => UpstreamErrorKind::Failed,
                        };
                        retry_after_seconds = failure
                            .retry_after
                            .as_deref()
                            .and_then(|v| v.parse::<f64>().ok())
                            .filter(|v| v.is_finite() && *v >= 0.0)
                            .map(|v| v.ceil() as u64);
                        diagnostics.phase = Some("event_failure");
                        diagnostics.upstream_status = failure.explicit_status;
                        diagnostics.client_status = Some(failure.status);
                        diagnostics.upstream_code = failure.code;
                        diagnostics.upstream_type = failure.error_type;
                        diagnostics.upstream_param = failure.param;
                        diagnostics.request_id = failure.request_id;
                        diagnostics.response_id = failure.response_id;
                    }
                } else if error.as_str().contains("reasoning replay unavailable") {
                    kind = UpstreamErrorKind::Failed;
                    diagnostics.phase = Some("completed_reasoning");
                }
                return Err(UpstreamStreamError {
                    kind,
                    message: error.as_str().into(),
                    retry_after_seconds,
                    diagnostics: Some(diagnostics),
                });
            }
        }
        diagnostics.semantic_committed = state.has_semantic_output();
    }
    state.eof().map_err(|error| UpstreamStreamError {
        kind: UpstreamErrorKind::Transient,
        message: error.as_str().into(),
        retry_after_seconds: None,
        diagnostics: Some(UpstreamStreamDiagnostics {
            open_blocks: state.open_blocks(),
            phase: Some("terminal_validation"),
            ..diagnostics
        }),
    })?;
    Ok((out, state.finish_metadata().cloned()))
}

fn project_legacy(
    event: TranslationEvent,
    finish: Option<&FinishMetadata>,
) -> Option<ReducerEvent> {
    Some(match event {
        TranslationEvent::ThinkingStart { index } => ReducerEvent::ThinkingStart { index },
        TranslationEvent::ThinkingDelta { index, text } => {
            ReducerEvent::ThinkingDelta { index, text }
        }
        TranslationEvent::ThinkingSignature { index, signature } => {
            ReducerEvent::ThinkingSignature { index, signature }
        }
        TranslationEvent::ThinkingStop { index } => ReducerEvent::ThinkingStop { index },
        TranslationEvent::TextStart { index } => ReducerEvent::TextStart { index },
        TranslationEvent::TextDelta { index, text } => ReducerEvent::TextDelta { index, text },
        TranslationEvent::TextStop { index } => ReducerEvent::TextStop { index },
        TranslationEvent::ToolStart { index, id, name } => {
            ReducerEvent::ToolStart { index, id, name }
        }
        TranslationEvent::ToolDelta {
            index,
            partial_json,
        } => ReducerEvent::ToolDelta {
            index,
            partial_json,
        },
        TranslationEvent::ToolStop { index } => ReducerEvent::ToolStop { index },
        TranslationEvent::Skipped(_) | TranslationEvent::SearchComplete { .. } => return None,
        TranslationEvent::Progress {
            output_index,
            item_id,
        } => {
            let _ = (output_index, item_id); // Internal identity is intentionally omitted by the legacy facade.
            ReducerEvent::Progress
        }
        TranslationEvent::Finish {
            stop_reason,
            usage,
            web_search_requests,
        } => {
            let finish = finish.expect("shared state emitted finish metadata");
            ReducerEvent::Finish {
                stop_reason,
                usage,
                web_search_requests,
                terminal_type: finish.terminal_type.clone(),
                continuation_eligible: finish.continuation_eligible,
                response_id: finish.response_id.clone(),
                output_items: finish.output_items.clone(),
            }
        }
    })
}

pub fn map_codex_usage_to_anthropic(
    u: &Option<CodexUsage>,
    web_search_requests: Option<usize>,
) -> AnthropicUsage {
    let usage = match u {
        Some(u) => u,
        None => return AnthropicUsage::default(),
    };
    let cached = usage.input_tokens_details_cached.unwrap_or(0);
    let total_input = usage.input_tokens.unwrap_or(0);
    let input_tokens = total_input.saturating_sub(cached);

    let mut result = AnthropicUsage {
        input_tokens,
        output_tokens: usage.output_tokens.unwrap_or(0),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
        server_tool_use: None,
        output_tokens_details: usage
            .output_tokens_details_reasoning
            .map(|reasoning_tokens| OutputTokensDetails {
                reasoning_tokens,
                thinking_tokens: reasoning_tokens,
            }),
        codex_rate_limits: usage.rate_limits.as_deref().cloned(),
    };

    if let Some(requests) = web_search_requests
        && requests > 0
    {
        result.server_tool_use = Some(WebSearchUsage {
            web_search_requests: requests,
        });
    }

    result
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<WebSearchUsage>,
    /// Reasoning tokens as reported by Codex. `thinking_tokens` mirrors the
    /// field name Claude Code records for Anthropic models so per-turn
    /// accounting in transcripts stays comparable across providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Subscription meter as of this response, from the `codex.rate_limits`
    /// event. This optional proxy extension is not part of Anthropic's schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_rate_limits: Option<CodexRateLimits>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
    pub thinking_tokens: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebSearchUsage {
    pub web_search_requests: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::read_rewrite::completed_read_args;
    use super::*;
    use serde_json::json;

    #[test]
    fn read_repair_does_not_drop_active_or_future_tool() {
        let prefix = format!(
            "{{\"file_path\":\"/tmp/a\",\"pages\":\"\"{}",
            " ".repeat(2048)
        );
        let args = format!("{prefix}}}");
        let events = vec![
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs","encrypted_content":"opaque-completed"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","name":"Bash","call_id":"active"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"command\":"}),
            json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","name":"Read","call_id":"read"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":prefix}),
            json!({"type":"response.function_call_arguments.done","output_index":2,"arguments":args}),
            json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","arguments":args}}),
            json!({"type":"response.function_call_arguments.done","output_index":1,"arguments":"{\"command\":\"pwd\"}"}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","arguments":"{\"command\":\"pwd\"}"}}),
            json!({"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","name":"Bash","call_id":"future"}}),
            json!({"type":"response.output_item.done","output_index":3,"item":{"type":"function_call","arguments":"{\"command\":\"ls\"}"}}),
            json!({"type":"response.completed","response":{"id":"resp","status":"completed"}}),
        ];
        let mut live = super::super::live_stream::LiveStreamTranslator::new("msg", "model")
            .with_replay_budget(Some(64 * 1024));
        let mut bytes = Vec::new();
        for (i, event) in events.iter().enumerate() {
            bytes.extend(live.accept(event, None).unwrap());
            assert_eq!(
                live.is_finished(),
                i == events.len() - 1,
                "premature finish at {i}"
            );
        }
        let finish = live.finish_metadata().unwrap();
        assert!(finish.continuation_eligible);
        assert_eq!(finish.output_items.len(), 4);
        let rendered = String::from_utf8(bytes).unwrap();
        assert_eq!(rendered.matches("event: message_stop").count(), 1);
        assert!(rendered.contains("future"));
        assert!(rendered.contains("signature_delta"));

        let upstream: String = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect();
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let Some(ReducerEvent::Finish {
            output_items,
            continuation_eligible,
            ..
        }) = out.last()
        else {
            panic!("missing finish")
        };
        assert!(*continuation_eligible);
        assert_eq!(output_items.len(), 4);
        let arguments: Vec<_> = output_items
            .iter()
            .filter_map(|item| match item {
                ResponsesInputItem::FunctionCall {
                    call_id, arguments, ..
                } => Some((
                    call_id.as_str(),
                    serde_json::from_str::<serde_json::Value>(arguments).unwrap(),
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            arguments,
            vec![
                ("active", json!({"command":"pwd"})),
                ("read", json!({"file_path":"/tmp/a"})),
                ("future", json!({"command":"ls"}))
            ]
        );
    }

    #[test]
    fn tool_done_snapshots_recover_suffix_once() {
        for name in ["Read", "Bash"] {
            let events = [
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":name,"call_id":"c"}}),
                json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"file_path\":"}),
                json!({"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"file_path\":\"/tmp/a\"}"}),
                json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","arguments":"{\"file_path\":\"/tmp/a\"}"}}),
                json!({"type":"response.completed","response":{"status":"completed"}}),
            ];
            let upstream: String = events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect();
            let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
            let args: String = out
                .iter()
                .filter_map(|event| match event {
                    ReducerEvent::ToolDelta { partial_json, .. } => Some(partial_json.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(args, "{\"file_path\":\"/tmp/a\"}", "{name}");
        }
    }

    #[test]
    fn completed_read_arguments_reject_truncation_and_oversize_snapshots() {
        for args in [
            "{\"file_path\":".to_string(),
            format!("{{}}{}", " ".repeat(5_000_000)),
        ] {
            let events = [
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"Read","call_id":"c"}}),
                json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","arguments":args}}),
                json!({"type":"response.completed","response":{"status":"completed"}}),
            ];
            let upstream: String = events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect();
            assert!(
                reduce_upstream_bytes(upstream.as_bytes()).is_err(),
                "invalid final Read must fail"
            );
            let mut live = super::super::live_stream::LiveStreamTranslator::new("msg", "model");
            live.accept(&events[0], None).unwrap();
            assert!(live.accept(&events[1], None).is_err());
            assert!(!live.is_finished());
        }
    }

    fn sse(type_name: &str, payload: serde_json::Value) -> String {
        let mut obj = payload.as_object().cloned().unwrap_or_default();
        obj.insert("type".into(), json!(type_name));
        format!("data: {}\n\n", serde_json::to_string(&obj).unwrap())
    }

    #[test]
    fn read_whitespace_does_not_bypass_missing_completed_reasoning() {
        let prefix = format!("{{\"file_path\":\"/tmp/a\"{}", " ".repeat(1024));
        let events = [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs","encrypted_content":"early"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c","name":"Read"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":prefix}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","arguments":format!("{prefix}}}")}}),
            json!({"type":"response.completed","response":{"status":"completed"}}),
        ];
        let upstream: String = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect();
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert_eq!(error.kind, UpstreamErrorKind::Failed);
        assert!(
            error
                .message
                .contains("completed reasoning replay unavailable")
        );
    }

    #[test]
    fn malformed_event_cannot_be_hidden_by_completed_terminal() {
        let upstream = format!(
            "data: {{malformed\n\n{}",
            sse(
                "response.completed",
                json!({"response":{"status":"completed"}})
            )
        );
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert_eq!(error.kind, UpstreamErrorKind::Failed);
        assert!(error.message.contains("malformed Codex event"));
    }

    #[test]
    fn completed_reasoning_after_summary_closes_inherits_id_and_preserves_blob() {
        let upstream = format!(
            "{}{}{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"early"}})
            ),
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"delta":"plan"})
            ),
            sse(
                "response.output_item.added",
                json!({"output_index":1,"item":{"type":"message","id":"msg_up"}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","encrypted_content":"final:opaque\nexact"}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":1,"item":{"type":"message","id":"msg_up"}})
            ),
            sse(
                "response.done",
                json!({"response":{"id":"resp_up","status":"completed"}})
            )
        );
        let metadata = finish_metadata_from_upstream(upstream.as_bytes())
            .unwrap()
            .unwrap();
        assert!(metadata.continuation_eligible);
        assert!(
            matches!(metadata.output_items.first(),Some(ResponsesInputItem::Reasoning {id,encrypted_content,..})
            if id == "rs_1" && encrypted_content == "final:opaque\nexact")
        );
    }

    #[test]
    fn event_failure_diagnostics_preserve_provenance_without_payload() {
        let upstream = format!(
            "{}{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"message","id":"msg_up"}})
            ),
            sse(
                "response.output_text.delta",
                json!({"output_index":0,"delta":"private prompt text"})
            ),
            sse(
                "error",
                json!({"status":422,"code":"invalid_request_error","param":"input","request_id":"req_up","response":{"id":"resp_up"},"message":"synthetic rejection"})
            )
        );
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        let value = serde_json::to_value(error.diagnostics.expect("failure diagnostics")).unwrap();
        assert_eq!(value["source"], "buffered_reducer");
        assert_eq!(value["phase"], "event_failure");
        assert_eq!(value["upstream_code"], "invalid_request_error");
        assert_eq!(value["upstream_status"], 422);
        assert_eq!(value["upstream_param"], "input");
        assert_eq!(value["request_id"], "req_up");
        assert_eq!(value["response_id"], "resp_up");
        assert_eq!(value["last_event_type"], "error");
        assert_eq!(value["semantic_committed"], true);
        assert!(!value.to_string().contains("private prompt text"));
    }

    #[test]
    fn orphan_function_done_is_explicit_failure() {
        let upstream = format!(
            "{}{}",
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{}"}})
            ),
            sse(
                "response.completed",
                json!({"response":{"status":"completed"}})
            )
        );
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert!(error.message.contains("unsupported Codex sequence"));
        assert_eq!(error.kind, UpstreamErrorKind::Failed);
    }

    #[test]
    fn missing_completed_reasoning_is_continuation_ineligible() {
        let upstream = format!(
            "{}{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"early"}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","id":"rs_1"}})
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed"}})
            )
        );
        let events = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ReducerEvent::ThinkingSignature { .. }))
        );
        let ReducerEvent::Finish {
            continuation_eligible,
            output_items,
            ..
        } = events.last().unwrap()
        else {
            panic!("finish");
        };
        assert!(!continuation_eligible);
        assert!(output_items.is_empty());
    }

    #[test]
    fn tool_round_trip_without_completed_reasoning_fails_explicitly() {
        let upstream = format!(
            "{}{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"early"}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","id":"rs_1"}})
            ),
            sse(
                "response.output_item.added",
                json!({"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":""}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{}"}})
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed"}})
            )
        );
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert_eq!(error.kind, UpstreamErrorKind::Failed);
        assert!(
            error
                .message
                .contains("completed reasoning replay unavailable")
        );
        assert!(!error.message.contains("early"));
    }

    #[test]
    fn reduce_text_response() {
        let upstream = format!(
            "{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index": 0,
                    "item": {"type":"message","id":"msg_up"}
                })
            ),
            sse(
                "response.output_text.delta",
                json!({
                    "output_index":0,"delta":"hello"
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,"item":{"type":"message"}
                })
            ),
            sse(
                "response.completed",
                json!({
                    "response":{"id":"resp_1","status":"completed","incomplete_details":null,"usage":{"input_tokens":5,"output_tokens":1}}
                })
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let last = out.last().unwrap();
        if let ReducerEvent::Finish {
            stop_reason,
            response_id,
            usage,
            ..
        } = last
        {
            assert_eq!(*stop_reason, "end_turn");
            assert_eq!(response_id.as_deref(), Some("resp_1"));
            assert_eq!(usage.as_ref().unwrap().input_tokens, Some(5));
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn reduce_tool_use_response() {
        let upstream = format!(
            "{}{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read"}
                })
            ),
            sse(
                "response.function_call_arguments.delta",
                json!({
                    "output_index":0,"delta":"{\"file_path\":"
                })
            ),
            sse(
                "response.function_call_arguments.delta",
                json!({
                    "output_index":0,"delta":"\"/tmp/a\"}"
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{\"file_path\":\"/tmp/a\"}"}
                })
            ),
            sse(
                "response.completed",
                json!({
                    "response":{"id":"resp_1","usage":{}}
                })
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let last = out.last().unwrap();
        if let ReducerEvent::Finish { stop_reason, .. } = last {
            assert_eq!(*stop_reason, "tool_use");
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn reduce_rate_limit_snapshot_is_progress_telemetry() {
        let upstream = format!(
            "{}{}",
            sse(
                "codex.rate_limits",
                json!({"plan_type":"prolite","rate_limits":{"limit_reached":true,"primary":{"reset_after_seconds":30,"used_percent":42}}}),
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed","usage":{}}}),
            )
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        assert!(matches!(out.first(), Some(ReducerEvent::Progress)));
        let Some(ReducerEvent::Finish { usage, .. }) = out.last() else {
            panic!("expected Finish");
        };
        let limits = usage
            .as_ref()
            .and_then(|usage| usage.rate_limits.as_ref())
            .expect("rate limits carried into finish usage");
        assert_eq!(limits.plan_type.as_deref(), Some("prolite"));
        assert_eq!(limits.limit_reached, Some(true));
        assert_eq!(
            limits.primary.as_ref().and_then(|w| w.used_percent),
            Some(42.0)
        );
    }

    #[test]
    fn read_whitespace_without_item_completion_is_not_success() {
        let upstream = format!(
            "{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"function_call","call_id":"c","name":"Read"}})
            ),
            sse(
                "response.function_call_arguments.delta",
                json!({"output_index":0,"delta":format!("{{\"file_path\":\"/tmp/a\"{}", " ".repeat(1024))})
            ),
        );
        let error = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert!(error.message.contains("before terminal"));
        assert_eq!(error.diagnostics.unwrap().open_blocks.len(), 1);
    }

    #[test]
    fn reduce_upstream_error_event() {
        let upstream = sse("error", json!({"error":{"message":"upstream failure"}}));
        let result = reduce_upstream_bytes(upstream.as_bytes());
        assert!(result.is_err());
        match result.unwrap_err().kind {
            UpstreamErrorKind::Failed => {}
            _ => panic!("expected Failed"),
        }
    }

    #[test]
    fn reduce_web_search_output() {
        let upstream = format!(
            "{}{}{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index":0,
                    "item":{"type":"web_search_call","id":"ws_1"}
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,
                    "item":{"type":"web_search_call","id":"ws_1","action":{"query":"test query"}}
                })
            ),
            sse(
                "response.output_item.added",
                json!({
                    "output_index":1,
                    "item":{"type":"message","id":"msg_up"}
                })
            ),
            sse(
                "response.output_text.delta",
                json!({
                    "output_index":1,"delta":"result"
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":1,"item":{"type":"message"}
                })
            ),
            sse(
                "response.completed",
                json!({
                    "response":{"id":"resp_1","usage":{"input_tokens":3}}
                })
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let has_web_search = out
            .iter()
            .any(|e| matches!(e, ReducerEvent::WebSearch { .. }));
        assert!(
            !has_web_search,
            "strict projection omits native search cards"
        );
        let last = out.last().unwrap();
        if let ReducerEvent::Finish {
            web_search_requests,
            ..
        } = last
        {
            assert_eq!(*web_search_requests, 1);
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn reduce_missing_terminal_is_error() {
        let upstream = format!(
            "{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index": 0,
                    "item": {"type":"message","id":"msg_up"}
                })
            ),
            sse(
                "response.output_text.delta",
                json!({
                    "output_index":0,"delta":"partial"
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,"item":{"type":"message"}
                })
            ),
        );
        let err = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert_eq!(err.kind, UpstreamErrorKind::Transient);
        assert!(err.message.contains("terminal"));
    }

    #[test]
    fn reduce_incomplete_is_upstream_error() {
        let upstream = sse(
            "response.incomplete",
            json!({"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{}}}),
        );
        let err = reduce_upstream_bytes(upstream.as_bytes()).unwrap_err();
        assert_eq!(err.kind, UpstreamErrorKind::Transient);
        assert!(err.message.contains("max_output_tokens"));
    }

    #[test]
    fn standard_responses_policy_maps_max_output_tokens_to_finish() {
        let upstream = sse(
            "response.incomplete",
            json!({"response":{"id":"resp_1","status":"incomplete","error":null,"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":2,"output_tokens":8}}}),
        );
        let out = reduce_upstream_bytes_with_policy(
            upstream.as_bytes(),
            IncompleteResponsePolicy::AllowMaxOutputTokens,
        )
        .unwrap();
        let Some(ReducerEvent::Finish {
            stop_reason,
            terminal_type,
            continuation_eligible,
            usage,
            ..
        }) = out.last()
        else {
            panic!("expected Finish");
        };
        assert_eq!(*stop_reason, STOP_MAX_TOKENS);
        assert_eq!(terminal_type, TERM_INCOMPLETE);
        assert!(!continuation_eligible);
        assert_eq!(
            usage.as_ref().and_then(|usage| usage.output_tokens),
            Some(8)
        );
    }

    #[test]
    fn standard_responses_policy_rejects_a_second_terminal_event() {
        let upstream = format!(
            "{}{}",
            sse(
                "response.incomplete",
                json!({"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{}}}),
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed","usage":{}}}),
            ),
        );
        let err = reduce_upstream_bytes_with_policy(
            upstream.as_bytes(),
            IncompleteResponsePolicy::AllowMaxOutputTokens,
        )
        .unwrap_err();
        assert_eq!(err.kind, UpstreamErrorKind::Transient);
        assert!(err.message.contains("multiple terminal"));
    }

    #[test]
    fn standard_responses_policy_rejects_content_after_terminal() {
        let trailing_events = [
            (
                "response.output_text.delta",
                json!({"output_index":0,"delta":"late"}),
            ),
            (
                "response.output_item.added",
                json!({
                    "output_index":0,
                    "item":{"type":"message","id":"msg_late"}
                }),
            ),
        ];

        for (event_type, payload) in trailing_events {
            let upstream = format!(
                "{}{}",
                sse(
                    "response.incomplete",
                    json!({"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{}}}),
                ),
                sse(event_type, payload),
            );
            let err = reduce_upstream_bytes_with_policy(
                upstream.as_bytes(),
                IncompleteResponsePolicy::AllowMaxOutputTokens,
            )
            .unwrap_err();

            assert_eq!(err.kind, UpstreamErrorKind::Transient);
            assert!(err.message.contains("after the terminal"));
            let diagnostics = err.diagnostics.expect("expected diagnostics");
            assert_eq!(diagnostics.event_count, 2);
            assert_eq!(diagnostics.last_event_type.as_deref(), Some(event_type));
            assert!(diagnostics.saw_terminal_event);
        }
    }

    #[test]
    fn completed_terminal_followed_by_done_marker_remains_valid() {
        let upstream = format!(
            "{}data: [DONE]\n\n",
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed","usage":{}}}),
            ),
        );

        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let Some(ReducerEvent::Finish { stop_reason, .. }) = out.last() else {
            panic!("expected Finish");
        };
        assert_eq!(*stop_reason, STOP_END_TURN);
    }

    #[test]
    fn completed_terminal_followed_by_ping_remains_valid() {
        let upstream = format!(
            "{}{}",
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","status":"completed","usage":{}}}),
            ),
            sse("ping", json!({"cost":"0"})),
        );

        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let Some(ReducerEvent::Finish { stop_reason, .. }) = out.last() else {
            panic!("expected Finish");
        };
        assert_eq!(*stop_reason, STOP_END_TURN);
    }

    #[test]
    fn standard_responses_max_tokens_takes_priority_over_tool_use() {
        let upstream = format!(
            "{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read"}
                }),
            ),
            sse(
                "response.function_call_arguments.delta",
                json!({"output_index":0,"delta":"{}"}),
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{}"}
                }),
            ),
            sse(
                "response.incomplete",
                json!({"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{}}}),
            ),
        );
        let out = reduce_upstream_bytes_with_policy(
            upstream.as_bytes(),
            IncompleteResponsePolicy::AllowMaxOutputTokens,
        )
        .unwrap();
        let Some(ReducerEvent::Finish { stop_reason, .. }) = out.last() else {
            panic!("expected Finish");
        };
        assert_eq!(*stop_reason, STOP_MAX_TOKENS);
    }

    #[test]
    fn reduce_completed_with_null_incomplete_details_is_end_turn() {
        let upstream = sse(
            "response.completed",
            json!({"response":{"id":"resp_1","status":"completed","incomplete_details":null,"usage":{}}}),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let last = out.last().unwrap();
        if let ReducerEvent::Finish {
            stop_reason,
            continuation_eligible,
            ..
        } = last
        {
            assert_eq!(*stop_reason, "end_turn");
            assert!(continuation_eligible);
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn reduce_completed_is_continuation_eligible() {
        let upstream = sse(
            "response.completed",
            json!({"response":{"id":"resp_1","usage":{}}}),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let last = out.last().unwrap();
        if let ReducerEvent::Finish {
            continuation_eligible,
            ..
        } = last
        {
            assert!(continuation_eligible);
        } else {
            panic!("expected Finish");
        }
    }

    #[test]
    fn finish_metadata_extracts_continuation_state() {
        let upstream = format!(
            "{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index": 0,
                    "item": {"type":"message","id":"msg_up"}
                })
            ),
            sse(
                "response.output_text.delta",
                json!({
                    "output_index":0,"delta":"hello"
                })
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,"item":{"type":"message"}
                })
            ),
            sse(
                "response.completed",
                json!({
                    "response":{"id":"resp_1","usage":{}}
                })
            ),
        );
        let metadata = finish_metadata_from_upstream(upstream.as_bytes())
            .unwrap()
            .unwrap();
        assert!(metadata.continuation_eligible);
        assert_eq!(metadata.response_id.as_deref(), Some("resp_1"));
        assert_eq!(metadata.output_items.len(), 1);
    }

    #[test]
    fn sanitize_tool_args_removes_empty_pages() {
        let args = r#"{"file_path":"/tmp/a","pages":""}"#;
        let sanitized = completed_read_args(args, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert!(parsed.get("pages").is_none());
        assert_eq!(
            parsed.get("file_path").and_then(|v| v.as_str()),
            Some("/tmp/a")
        );
    }

    #[test]
    fn map_usage_reports_cached_prompt_tokens() {
        let usage = CodexUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            input_tokens_details_cached: Some(20),
            output_tokens_details_reasoning: None,
            rate_limits: None,
        };
        let mapped = map_codex_usage_to_anthropic(&Some(usage), None);
        assert_eq!(mapped.input_tokens, 80);
        assert_eq!(mapped.output_tokens, 50);
        assert_eq!(mapped.cache_read_input_tokens, 20);
        assert!(mapped.output_tokens_details.is_none());
        assert!(mapped.codex_rate_limits.is_none());
    }

    #[test]
    #[allow(clippy::format_in_format_args)]
    fn reduce_reasoning_summary_before_text() {
        let upstream = format!(
            "{}{}{}{}{}{}",
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":0,"delta":"Plan"})
            ),
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":0,"delta":"ning"})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.output_item.added",
                json!({"output_index":1,"item":{"type":"message","id":"msg_up"}})
            ),
            sse(
                "response.output_text.delta",
                json!({"output_index":1,"delta":"answer"})
            ),
            format!(
                "{}{}",
                sse(
                    "response.output_item.done",
                    json!({"output_index":1,"item":{"type":"message"}})
                ),
                sse(
                    "response.completed",
                    json!({"response":{"id":"resp_1","usage":{}}})
                )
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        assert!(matches!(
            out.iter()
                .find(|event| matches!(event, ReducerEvent::ThinkingStart { .. })),
            Some(ReducerEvent::ThinkingStart { index: 0 })
        ));
        let thinking_text: String = out
            .iter()
            .filter_map(|event| match event {
                ReducerEvent::ThinkingDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking_text, "Planning");
        let thinking_stop = out
            .iter()
            .position(|event| matches!(event, ReducerEvent::ThinkingStop { .. }))
            .unwrap();
        let text_start = out
            .iter()
            .position(|event| matches!(event, ReducerEvent::TextStart { .. }))
            .unwrap();
        assert!(thinking_stop < text_start);
        assert!(matches!(
            out.iter()
                .find(|event| matches!(event, ReducerEvent::TextStart { .. })),
            Some(ReducerEvent::TextStart { index: 1 })
        ));
    }

    #[test]
    #[allow(clippy::format_in_format_args)]
    fn reduce_empty_reasoning_summary_emits_no_thinking() {
        let upstream = format!(
            "{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({"output_index":0,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.output_item.added",
                json!({"output_index":1,"item":{"type":"message","id":"msg_up"}})
            ),
            format!(
                "{}{}{}",
                sse(
                    "response.output_text.delta",
                    json!({"output_index":1,"delta":"answer"})
                ),
                sse(
                    "response.output_item.done",
                    json!({"output_index":1,"item":{"type":"message"}})
                ),
                sse(
                    "response.completed",
                    json!({"response":{"id":"resp_1","usage":{}}})
                )
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        assert!(!out.iter().any(|event| matches!(
            event,
            ReducerEvent::ThinkingStart { .. }
                | ReducerEvent::ThinkingDelta { .. }
                | ReducerEvent::ThinkingStop { .. }
        )));
    }

    #[test]
    fn reduce_multiple_reasoning_summary_parts() {
        let upstream = format!(
            "{}{}{}{}{}",
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":0,"delta":"part one"})
            ),
            sse(
                "response.reasoning_summary_part.added",
                json!({"output_index":0,"summary_index":1,"part":{"type":"summary_text","text":""}})
            ),
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":1,"delta":"part two"})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","usage":{}}})
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let deltas: Vec<&str> = out
            .iter()
            .filter_map(|event| match event {
                ReducerEvent::ThinkingDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["part one", "\n\n", "part two"]);
        assert_eq!(
            out.iter()
                .filter(|event| matches!(event, ReducerEvent::ThinkingStop { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn reduce_two_reasoning_items() {
        let upstream = format!(
            "{}{}{}{}{}",
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":0,"delta":"first"})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":0,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":1,"summary_index":0,"delta":"second"})
            ),
            sse(
                "response.output_item.done",
                json!({"output_index":1,"item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}})
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","usage":{}}})
            ),
        );
        let out = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        assert_eq!(
            out.iter()
                .filter(|event| matches!(event, ReducerEvent::ThinkingStart { .. }))
                .count(),
            2
        );
        assert_eq!(
            out.iter()
                .filter(|event| matches!(event, ReducerEvent::ThinkingStop { .. }))
                .count(),
            2
        );
        let deltas: Vec<&str> = out
            .iter()
            .filter_map(|event| match event {
                ReducerEvent::ThinkingDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["first", "second"]);
    }

    #[test]
    fn reasoning_signature_precedes_stop_and_enters_continuation_transcript() {
        let upstream = format!(
            "{}{}{}{}",
            sse(
                "response.output_item.added",
                json!({
                    "output_index":0,
                    "item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"}
                })
            ),
            sse(
                "response.reasoning_summary_text.delta",
                json!({"output_index":0,"summary_index":0,"delta":"plan"})
            ),
            sse(
                "response.output_item.done",
                json!({
                    "output_index":0,
                    "item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"}
                })
            ),
            sse(
                "response.completed",
                json!({"response":{"id":"resp_1","usage":{}}})
            ),
        );
        let events = reduce_upstream_bytes(upstream.as_bytes()).unwrap();
        let signature_index = events
            .iter()
            .position(|event| matches!(event, ReducerEvent::ThinkingSignature { .. }))
            .unwrap();
        let stop_index = events
            .iter()
            .position(|event| matches!(event, ReducerEvent::ThinkingStop { .. }))
            .unwrap();
        assert!(signature_index < stop_index);

        let ReducerEvent::ThinkingSignature { signature, .. } = &events[signature_index] else {
            unreachable!();
        };
        let replay =
            super::super::reasoning_signature::decode_reasoning_signature(signature).unwrap();
        assert_eq!(replay.id, "rs_1");
        assert_eq!(replay.encrypted_content, "opaque");

        let ReducerEvent::Finish { output_items, .. } = events.last().unwrap() else {
            panic!("expected Finish");
        };
        assert!(matches!(
            output_items.as_slice(),
            [ResponsesInputItem::Reasoning {
                id,
                encrypted_content,
                ..
            }] if id == "rs_1" && encrypted_content == "opaque"
        ));
    }
    #[test]
    fn map_usage_reports_reasoning_and_rate_limits() {
        let usage = CodexUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            input_tokens_details_cached: None,
            output_tokens_details_reasoning: Some(30),
            rate_limits: Some(Box::new(CodexRateLimits {
                plan_type: Some("prolite".into()),
                limit_reached: Some(false),
                primary: Some(CodexRateLimitWindow {
                    used_percent: Some(8.0),
                    window_minutes: Some(10080),
                    reset_at: Some(1_789_765_755),
                }),
                secondary: None,
            })),
        };
        let mapped = map_codex_usage_to_anthropic(&Some(usage), None);
        let details = mapped
            .output_tokens_details
            .as_ref()
            .expect("reasoning details");
        assert_eq!(details.reasoning_tokens, 30);
        assert_eq!(details.thinking_tokens, 30);
        let json = serde_json::to_value(&mapped).unwrap();
        assert_eq!(json["codex_rate_limits"]["primary"]["used_percent"], 8.0);
        assert_eq!(json["codex_rate_limits"]["plan_type"], "prolite");
        assert!(json["codex_rate_limits"].get("secondary").is_none());
    }
}
