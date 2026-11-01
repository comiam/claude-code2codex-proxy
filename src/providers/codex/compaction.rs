use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::anthropic::sse::parse_sse_events;
use crate::provider::RequestContext;
use crate::providers::codex::client::{CodexError, CodexHttpClient};

use super::translate::request::{
    ResponsesContentPart, ResponsesInputItem, ResponsesRequest, is_compact_message_text,
};

const RETAINED_MESSAGE_TOKEN_BUDGET: u64 = 20_000;
const STATE_TTL_MS: u64 = 30 * 60 * 1_000;
const MAX_STATES: usize = 1_000;
const MAX_STATE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOTAL_STATE_BYTES: usize = 20_000_000;
const MIN_PORTABLE_SUMMARY_BYTES: usize = 32;

#[derive(Debug)]
pub enum CompactionError {
    Upstream(CodexError),
    InvalidResponse(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(error) => write!(f, "{error}"),
            Self::InvalidResponse(message) => f.write_str(message),
        }
    }
}

enum CompactionPhase {
    Preparing,
    Unconfirmed,
    Anchored { portable_summary: String },
}

struct CompactionState {
    attempt: CompactionAttempt,
    model: String,
    native_history: Vec<ResponsesInputItem>,
    phase: CompactionPhase,
    updated_at: u64,
    retained_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionAttempt(u64);

pub struct CompactionReplay {
    pub request: ResponsesRequest,
    pub attempt: CompactionAttempt,
}

#[derive(Default)]
struct CompactionRegistry {
    states: HashMap<String, CompactionState>,
    total_bytes: usize,
}

impl CompactionRegistry {
    fn remove(&mut self, session_id: &str) {
        if let Some(state) = self.states.remove(session_id) {
            self.total_bytes -= state.retained_bytes;
        }
    }

    fn insert(&mut self, session_id: String, state: CompactionState) {
        self.remove(&session_id);
        self.total_bytes += state.retained_bytes;
        self.states.insert(session_id, state);
    }
}

static REGISTRY: Mutex<Option<CompactionRegistry>> = Mutex::new(None);
static NEXT_ATTEMPT_ID: AtomicU64 = AtomicU64::new(1);

pub async fn request_compaction(
    client: &CodexHttpClient,
    request: &ResponsesRequest,
    ctx: &RequestContext,
) -> Result<Vec<ResponsesInputItem>, CompactionError> {
    let (envelope, conversation) = split_input_envelope(&request.input);
    let conversation = without_compaction_instruction(conversation);
    let mut compaction_request = request.clone();
    compaction_request.instructions = None;
    compaction_request.input = envelope
        .iter()
        .filter(|item| matches!(item, ResponsesInputItem::AdditionalTools { .. }))
        .cloned()
        .chain(conversation.iter().cloned())
        .chain(std::iter::once(ResponsesInputItem::CompactionTrigger))
        .collect();
    compaction_request.include = Some(vec!["reasoning.encrypted_content".to_string()]);

    let response = client
        .post_codex_for_owner(&compaction_request, ctx, None)
        .await
        .map_err(CompactionError::Upstream)?;
    let compaction = parse_compaction_response(&response.body)?;
    Ok(build_compacted_history(&conversation, compaction))
}

pub fn begin_compaction(session_id: &str, model: &str) -> CompactionAttempt {
    let attempt = CompactionAttempt(NEXT_ATTEMPT_ID.fetch_add(1, Ordering::Relaxed));
    let state = CompactionState {
        attempt,
        model: model.to_string(),
        native_history: Vec::new(),
        phase: CompactionPhase::Preparing,
        updated_at: now_ms(),
        // Empty JSON input is two bytes. Metadata and summary use UTF-8 lengths.
        retained_bytes: session_id.len() + model.len() + 2,
    };
    let now = state.updated_at;
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    evict_states(registry, now);
    registry.insert(session_id.to_string(), state);
    evict_states(registry, now);
    attempt
}

pub fn store_compaction(
    session_id: &str,
    attempt: CompactionAttempt,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    // The potentially large history is measured before acquiring the registry.
    // Publication below rechecks the attempt after this work.
    let history_bytes = serialized_size(&native_history);
    #[cfg(test)]
    if let Some(hook) = AFTER_HISTORY_MEASUREMENT.with(|hook| hook.borrow_mut().take()) {
        hook();
    }
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    evict_states(registry, now);
    let Some(state) = registry.states.get_mut(session_id) else {
        return false;
    };
    if state.attempt != attempt || !matches!(state.phase, CompactionPhase::Preparing) {
        return false;
    }
    let retained_bytes = session_id
        .len()
        .saturating_add(state.model.len())
        .saturating_add(history_bytes);
    if retained_bytes > MAX_STATE_BYTES {
        registry.remove(session_id);
        return false;
    }
    registry.total_bytes = registry.total_bytes - state.retained_bytes + retained_bytes;
    state.retained_bytes = retained_bytes;
    state.native_history = native_history;
    state.phase = CompactionPhase::Unconfirmed;
    state.updated_at = now;
    evict_states(registry, now);
    registry
        .states
        .get(session_id)
        .is_some_and(|state| state.attempt == attempt)
}

pub fn activate_compaction(
    session_id: Option<&str>,
    attempt: Option<CompactionAttempt>,
    model: &str,
    output: &[ResponsesInputItem],
) -> bool {
    let (Some(session_id), Some(attempt)) = (session_id, attempt) else {
        return false;
    };
    let Some(portable_summary) = portable_summary_text(output) else {
        abort_compaction_attempt(Some(session_id), Some(attempt));
        return false;
    };

    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    evict_states(registry, now);
    let Some(state) = registry.states.get_mut(session_id) else {
        return false;
    };
    if state.attempt != attempt {
        return false;
    }
    if state.model != model || !matches!(state.phase, CompactionPhase::Unconfirmed) {
        registry.remove(session_id);
        return false;
    }
    let retained_bytes = state.retained_bytes.saturating_add(portable_summary.len());
    if retained_bytes > MAX_STATE_BYTES {
        registry.remove(session_id);
        return false;
    }
    registry.total_bytes += retained_bytes - state.retained_bytes;
    state.retained_bytes = retained_bytes;
    state.phase = CompactionPhase::Anchored { portable_summary };
    state.updated_at = now;
    evict_states(registry, now);
    registry.states.contains_key(session_id)
}

pub fn apply_compaction_replay(
    session_id: Option<&str>,
    request: &ResponsesRequest,
) -> Option<CompactionReplay> {
    let session_id = session_id?;
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.as_mut()?;
    evict_states(registry, now);
    let state = registry.states.get_mut(session_id)?;
    if !matches!(state.phase, CompactionPhase::Anchored { .. }) {
        return None;
    }
    if state.model != request.model {
        registry.remove(session_id);
        return None;
    }
    let CompactionPhase::Anchored { portable_summary } = &state.phase else {
        return None;
    };

    let (envelope, conversation) = split_input_envelope(&request.input);
    let summary_item = conversation.first()?;
    let Some(text) = message_text(summary_item) else {
        registry.remove(session_id);
        return None;
    };
    if text.match_indices(portable_summary).count() != 1 {
        registry.remove(session_id);
        return None;
    }
    if conversation.len() == 1 {
        return None;
    }

    let mut replay = request.clone();
    replay.input = envelope
        .iter()
        .cloned()
        .chain(state.native_history.iter().cloned())
        .chain(conversation[1..].iter().cloned())
        .collect();
    if serialized_size(&replay.input) > MAX_STATE_BYTES {
        registry.remove(session_id);
        return None;
    }
    state.updated_at = now;
    Some(CompactionReplay {
        request: replay,
        attempt: state.attempt,
    })
}

pub fn abort_compaction_attempt(session_id: Option<&str>, attempt: Option<CompactionAttempt>) {
    let (Some(session_id), Some(attempt)) = (session_id, attempt) else {
        return;
    };
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    if registry
        .states
        .get(session_id)
        .is_some_and(|state| state.attempt == attempt)
    {
        registry.remove(session_id);
    }
}

pub fn clear_compaction(session_id: &str) {
    let mut guard = REGISTRY.lock().unwrap();
    if let Some(registry) = guard.as_mut() {
        registry.remove(session_id);
    }
}

fn split_input_envelope(
    input: &[ResponsesInputItem],
) -> (&[ResponsesInputItem], &[ResponsesInputItem]) {
    let prefix_len = input
        .iter()
        .take_while(|item| is_envelope_item(item))
        .count();
    input.split_at(prefix_len)
}

fn without_compaction_instruction(input: &[ResponsesInputItem]) -> Vec<ResponsesInputItem> {
    let mut input = input.to_vec();
    let remove_empty_message = if let Some(ResponsesInputItem::Message { role, content }) =
        input.last_mut()
        && role == "user"
    {
        content.retain(|part| {
            !matches!(
                part,
                ResponsesContentPart::InputText { text }
                    if is_compact_message_text(text)
            )
        });
        content.is_empty()
    } else {
        false
    };
    if remove_empty_message {
        input.pop();
    }
    input
}

fn is_envelope_item(item: &ResponsesInputItem) -> bool {
    match item {
        ResponsesInputItem::AdditionalTools { .. } => true,
        ResponsesInputItem::Message { role, .. } => role == "developer",
        _ => false,
    }
}

fn portable_summary_text(output: &[ResponsesInputItem]) -> Option<String> {
    let text = output
        .iter()
        .filter_map(|item| match item {
            ResponsesInputItem::Message { role, content } if role == "assistant" => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter_map(|part| match part {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text } => Some(text.as_str()),
            ResponsesContentPart::InputImage { .. } => None,
        })
        .collect::<String>();
    let trimmed = text.trim();
    let summary = trimmed
        .split_once("<summary>")
        .and_then(|(_, rest)| rest.split_once("</summary>"))
        .map(|(summary, _)| summary.trim())
        .filter(|summary| !summary.is_empty())
        .unwrap_or(trimmed);
    (summary.len() >= MIN_PORTABLE_SUMMARY_BYTES).then(|| summary.to_string())
}

fn message_text(item: &ResponsesInputItem) -> Option<String> {
    let ResponsesInputItem::Message { content, .. } = item else {
        return None;
    };
    Some(
        content
            .iter()
            .filter_map(|part| match part {
                ResponsesContentPart::InputText { text }
                | ResponsesContentPart::OutputText { text } => Some(text.as_str()),
                ResponsesContentPart::InputImage { .. } => None,
            })
            .collect(),
    )
}

fn parse_compaction_response(body: &[u8]) -> Result<ResponsesInputItem, CompactionError> {
    let mut completed = false;
    let mut compacted = Vec::new();

    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.data) else {
            continue;
        };
        match payload.get("type").and_then(serde_json::Value::as_str) {
            Some("error" | "response.error" | "response.failed") => {
                let message = payload
                    .pointer("/response/error/message")
                    .or_else(|| payload.pointer("/error/message"))
                    .or_else(|| payload.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("remote compaction failed");
                return Err(CompactionError::InvalidResponse(message.to_string()));
            }
            Some("response.output_item.done") => {
                let Some(item) = payload.get("item") else {
                    continue;
                };
                if item.get("type").and_then(serde_json::Value::as_str) == Some("compaction")
                    && let Some(encrypted_content) = item
                        .get("encrypted_content")
                        .and_then(serde_json::Value::as_str)
                {
                    compacted.push(ResponsesInputItem::Compaction {
                        encrypted_content: encrypted_content.to_string(),
                    });
                }
            }
            Some("response.completed") => completed = true,
            _ => {}
        }
    }

    if !completed {
        return Err(CompactionError::InvalidResponse(
            "remote compaction stream ended before response.completed".to_string(),
        ));
    }
    if compacted.len() != 1 {
        return Err(CompactionError::InvalidResponse(format!(
            "remote compaction expected exactly one compaction item, got {}",
            compacted.len()
        )));
    }
    Ok(compacted.pop().expect("validated one compaction item"))
}

fn build_compacted_history(
    input: &[ResponsesInputItem],
    compaction: ResponsesInputItem,
) -> Vec<ResponsesInputItem> {
    let retained = input
        .iter()
        .filter(|item| {
            matches!(
                item,
                ResponsesInputItem::Message { role, .. }
                    if matches!(role.as_str(), "user" | "developer" | "system")
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut retained = truncate_retained_messages(retained, RETAINED_MESSAGE_TOKEN_BUDGET);
    retained.push(compaction);
    retained
}

fn truncate_retained_messages(
    items: Vec<ResponsesInputItem>,
    max_tokens: u64,
) -> Vec<ResponsesInputItem> {
    let mut remaining = max_tokens;
    let mut retained = Vec::new();
    for item in items.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let tokens = message_tokens(&item).max(1);
        if tokens <= remaining {
            retained.push(item);
            remaining -= tokens;
        } else if let Some(item) = truncate_message(item, remaining) {
            retained.push(item);
            remaining = 0;
        }
    }
    retained.reverse();
    retained
}

fn message_tokens(item: &ResponsesInputItem) -> u64 {
    let ResponsesInputItem::Message { content, .. } = item else {
        return 0;
    };
    content
        .iter()
        .map(|part| match part {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text } => text.len().div_ceil(4) as u64,
            ResponsesContentPart::InputImage { .. } => 2_000,
        })
        .sum()
}

fn truncate_message(item: ResponsesInputItem, max_tokens: u64) -> Option<ResponsesInputItem> {
    let ResponsesInputItem::Message { role, content } = item else {
        return Some(item);
    };
    let mut remaining_chars = max_tokens.saturating_mul(4) as usize;
    let mut truncated = Vec::new();
    for part in content {
        match part {
            ResponsesContentPart::InputImage { .. } => truncated.push(part),
            ResponsesContentPart::InputText { text } => {
                let text = truncate_text(text, &mut remaining_chars);
                if !text.is_empty() {
                    truncated.push(ResponsesContentPart::InputText { text });
                }
            }
            ResponsesContentPart::OutputText { text } => {
                let text = truncate_text(text, &mut remaining_chars);
                if !text.is_empty() {
                    truncated.push(ResponsesContentPart::OutputText { text });
                }
            }
        }
    }
    (!truncated.is_empty()).then_some(ResponsesInputItem::Message {
        role,
        content: truncated,
    })
}

fn truncate_text(mut text: String, remaining_chars: &mut usize) -> String {
    if *remaining_chars == 0 {
        return String::new();
    }
    if text.len() > *remaining_chars {
        let mut boundary = *remaining_chars;
        while !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    *remaining_chars -= text.len();
    text
}

fn serialized_size(items: &[ResponsesInputItem]) -> usize {
    #[cfg(test)]
    {
        HISTORY_MEASUREMENTS.with(|count| count.set(count.get() + 1));
        if REGISTRY.try_lock().is_err() {
            LOCKED_HISTORY_MEASUREMENTS.with(|count| count.set(count.get() + 1));
        }
    }
    super::translate::request::bounded_json_size(items, MAX_STATE_BYTES)
        .ok()
        .flatten()
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
thread_local! {
    static HISTORY_MEASUREMENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LOCKED_HISTORY_MEASUREMENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static AFTER_HISTORY_MEASUREMENT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn evict_states(registry: &mut CompactionRegistry, now: u64) {
    let total_bytes = &mut registry.total_bytes;
    registry.states.retain(|_, state| {
        let keep = now.saturating_sub(state.updated_at) <= STATE_TTL_MS;
        if !keep {
            *total_bytes -= state.retained_bytes;
        }
        keep
    });
    while registry.states.len() > MAX_STATES || registry.total_bytes > MAX_TOTAL_STATE_BYTES {
        let oldest = registry
            .states
            .iter()
            .min_by_key(|(_, state)| state.updated_at)
            .map(|(session_id, _)| session_id.clone());
        let Some(oldest) = oldest else {
            break;
        };
        registry.remove(&oldest);
    }
}

pub fn clear_all_compactions_for_tests() {
    *REGISTRY.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SUMMARY: &str =
        "portable summary with enough detail to identify this compacted conversation";
    const STALE_SUMMARY: &str =
        "stale portable summary from an older overlapping compaction attempt";
    static TEST_REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn maintenance_never_remeasures_retained_histories() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let first = stored_compaction("first", output("first retained history"));
        stored_compaction("second", output("second retained history"));
        HISTORY_MEASUREMENTS.with(|count| count.set(0));
        clear_compaction("absent");
        abort_compaction_attempt(Some("first"), Some(first));
        let mut guard = REGISTRY.lock().unwrap();
        let registry = guard.as_mut().unwrap();
        evict_states(registry, u64::MAX);
        assert_eq!(registry.total_bytes, 0);
        assert_eq!(HISTORY_MEASUREMENTS.with(|count| count.get()), 0);
    }

    #[test]
    fn storing_history_measures_once_and_activation_only_adds_summary_bytes() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let attempt = begin_compaction("owner", "gpt-5.6-sol");
        HISTORY_MEASUREMENTS.with(|count| count.set(0));
        LOCKED_HISTORY_MEASUREMENTS.with(|count| count.set(0));
        assert!(store_compaction("owner", attempt, output("retained")));
        assert_eq!(HISTORY_MEASUREMENTS.with(|count| count.get()), 1);
        assert_eq!(LOCKED_HISTORY_MEASUREMENTS.with(|count| count.get()), 0);
        let before = REGISTRY.lock().unwrap().as_ref().unwrap().total_bytes;
        assert!(activate_compaction(
            Some("owner"),
            Some(attempt),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        assert_eq!(
            REGISTRY.lock().unwrap().as_ref().unwrap().total_bytes,
            before + SUMMARY.len()
        );
        assert_eq!(HISTORY_MEASUREMENTS.with(|count| count.get()), 1);
        begin_compaction("owner", "gpt-5.6-sol");
        assert_eq!(
            REGISTRY.lock().unwrap().as_ref().unwrap().total_bytes,
            "owner".len() + "gpt-5.6-sol".len() + 2
        );
    }

    #[test]
    fn store_rechecks_attempt_after_outside_lock_history_measurement() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let old = begin_compaction("racing-owner", "gpt-5.6-sol");
        AFTER_HISTORY_MEASUREMENT.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(|| {
                begin_compaction("racing-owner", "gpt-5.6-sol");
            }))
        });
        assert!(!store_compaction(
            "racing-owner",
            old,
            output("old large history")
        ));
        let guard = REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        let current = registry.states.get("racing-owner").unwrap();
        assert_ne!(current.attempt, old);
        assert!(current.native_history.is_empty());
        assert_eq!(
            registry.total_bytes,
            "racing-owner".len() + "gpt-5.6-sol".len() + 2
        );
    }

    #[test]
    fn total_eviction_uses_cached_sizes_and_activation_enforces_owner_budget() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let history = output(&"x".repeat(3_900_000));
        HISTORY_MEASUREMENTS.with(|count| count.set(0));
        for index in 0..6 {
            // Order the previously published sessions before the new insertion.
            // Activation updates timestamps, so order them at this boundary.
            let now = now_ms();
            if let Some(registry) = REGISTRY.lock().unwrap().as_mut() {
                for prior in 0..index {
                    if let Some(state) = registry.states.get_mut(&format!("large-{prior}")) {
                        state.updated_at = now - (index - prior) as u64;
                    }
                }
            }
            let owner = format!("large-{index}");
            let attempt = stored_compaction(&owner, history.clone());
            assert!(activate_compaction(
                Some(&owner),
                Some(attempt),
                "gpt-5.6-sol",
                &output(SUMMARY)
            ));
        }
        let mut guard = REGISTRY.lock().unwrap();
        let registry = guard.as_mut().unwrap();
        assert!(registry.total_bytes <= 20_000_000);
        assert!(!registry.states.contains_key("large-0"));
        assert_eq!(
            HISTORY_MEASUREMENTS.with(|count| count.get()),
            6,
            "one new measurement per session, including eviction/activation"
        );
        HISTORY_MEASUREMENTS.with(|count| count.set(0));
        evict_states(registry, u64::MAX);
        assert_eq!(registry.total_bytes, 0);
        assert_eq!(HISTORY_MEASUREMENTS.with(|count| count.get()), 0);
        drop(guard);
        let attempt = stored_compaction("owner", output("small history"));
        assert!(!activate_compaction(
            Some("owner"),
            Some(attempt),
            "gpt-5.6-sol",
            &output(&"x".repeat(MAX_STATE_BYTES))
        ));
        assert_eq!(REGISTRY.lock().unwrap().as_ref().unwrap().total_bytes, 0);
    }

    fn request(input: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "input": input,
            "store": false,
            "stream": true,
            "parallel_tool_calls": false,
            "client_metadata": {"lite":"true"},
            "text": {"verbosity":"low"}
        }))
        .unwrap()
    }

    fn output(text: &str) -> Vec<ResponsesInputItem> {
        serde_json::from_value(json!([{
            "type":"message",
            "role":"assistant",
            "content":[{"type":"output_text","text":text}]
        }]))
        .unwrap()
    }

    fn stored_compaction(
        session_id: &str,
        native_history: Vec<ResponsesInputItem>,
    ) -> CompactionAttempt {
        let attempt = begin_compaction(session_id, "gpt-5.6-sol");
        assert!(store_compaction(session_id, attempt, native_history));
        attempt
    }

    #[test]
    fn parses_exactly_one_completed_compaction_item() {
        let body = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert!(matches!(
            parse_compaction_response(body).unwrap(),
            ResponsesInputItem::Compaction { encrypted_content } if encrypted_content == "opaque"
        ));
    }

    #[test]
    fn rejects_incomplete_or_ambiguous_compaction_streams() {
        let incomplete = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}}\n\n";
        assert!(
            parse_compaction_response(incomplete)
                .unwrap_err()
                .to_string()
                .contains("before response.completed")
        );
        let missing = b"data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert!(
            parse_compaction_response(missing)
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
    }

    #[test]
    fn replay_requires_activation_and_wrapped_summary_anchor() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let attempt = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        let next = request(json!([
            {"type":"additional_tools","role":"developer","tools":[]},
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"instructions"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":format!("<summary>{SUMMARY}</summary>")}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay(Some("session"), &next).is_none());
        assert!(activate_compaction(
            Some("session"),
            Some(attempt),
            "gpt-5.6-sol",
            &output(&format!(
                "<analysis>summary preparation</analysis>\n<summary>\n{SUMMARY}\n</summary>"
            ))
        ));

        let replay = apply_compaction_replay(Some("session"), &next)
            .unwrap()
            .request;
        assert!(matches!(
            replay.input[0],
            ResponsesInputItem::AdditionalTools { .. }
        ));
        assert!(
            matches!(replay.input[1], ResponsesInputItem::Message { ref role, .. } if role == "developer")
        );
        assert!(matches!(
            replay.input[2],
            ResponsesInputItem::Compaction { .. }
        ));
        assert_eq!(replay.client_metadata, next.client_metadata);
    }

    #[test]
    fn stale_activation_cannot_anchor_newer_native_history() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "older-native-history".to_string(),
            }],
        );
        let newer = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        );

        assert!(!activate_compaction(
            Some("session"),
            Some(older),
            "gpt-5.6-sol",
            &output(STALE_SUMMARY),
        ));
        assert!(activate_compaction(
            Some("session"),
            Some(newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn stale_store_cannot_replace_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction("session", "gpt-5.6-sol");
        let newer = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        );

        assert!(!store_compaction(
            "session",
            older,
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "late-older-native-history".to_string(),
            }],
        ));
        assert!(activate_compaction(
            Some("session"),
            Some(newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("session"), &next)
            .unwrap()
            .request;
        assert!(replay.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content }
                if encrypted_content == "newer-native-history"
        )));
    }

    #[test]
    fn preparing_compaction_survives_model_mismatched_replay_check() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let attempt = begin_compaction("session", "gpt-5.6-sol");
        let mut mismatched = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        mismatched.model = "gpt-5.6-terra".to_string();

        assert!(apply_compaction_replay(Some("session"), &mismatched).is_none());
        assert!(store_compaction(
            "session",
            attempt,
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "native-history".to_string(),
            }],
        ));
    }

    #[test]
    fn stale_abort_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction("session", "gpt-5.6-sol");
        let newer = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        );

        abort_compaction_attempt(Some("session"), Some(older));

        assert!(activate_compaction(
            Some("session"),
            Some(newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn stale_replay_abort_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "older-native-history".to_string(),
            }],
        );
        assert!(activate_compaction(
            Some("session"),
            Some(older),
            "gpt-5.6-sol",
            &output(STALE_SUMMARY),
        ));
        let older_next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":STALE_SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue old"}]}
        ]));
        let older_replay = apply_compaction_replay(Some("session"), &older_next).unwrap();

        let newer = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        );
        assert!(activate_compaction(
            Some("session"),
            Some(newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));

        abort_compaction_attempt(Some("session"), Some(older_replay.attempt));

        let newer_next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue new"}]}
        ]));
        let replay = apply_compaction_replay(Some("session"), &newer_next)
            .unwrap()
            .request;
        assert!(replay.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content }
                if encrypted_content == "newer-native-history"
        )));
    }

    #[test]
    fn invalid_stale_summary_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction("session", "gpt-5.6-sol");
        let newer = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        );

        assert!(!activate_compaction(
            Some("session"),
            Some(older),
            "gpt-5.6-sol",
            &output("too short"),
        ));
        assert!(activate_compaction(
            Some("session"),
            Some(newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn replay_clears_on_missing_or_duplicate_anchor() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        for text in [
            "different conversation without the expected summary".to_string(),
            format!("{SUMMARY} and {SUMMARY}"),
        ] {
            clear_all_compactions_for_tests();
            let attempt = stored_compaction(
                "session",
                vec![ResponsesInputItem::Compaction {
                    encrypted_content: "opaque".to_string(),
                }],
            );
            activate_compaction(
                Some("session"),
                Some(attempt),
                "gpt-5.6-sol",
                &output(SUMMARY),
            );
            let changed = request(json!([
                {"type":"message","role":"user","content":[{"type":"input_text","text":text}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
            ]));
            assert!(apply_compaction_replay(Some("session"), &changed).is_none());
            assert!(apply_compaction_replay(Some("session"), &changed).is_none());
        }
    }

    #[test]
    fn compacted_history_excludes_lite_envelope() {
        let input: Vec<ResponsesInputItem> = serde_json::from_value(json!([
            {"type":"additional_tools","role":"developer","tools":[]},
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"summarize"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"remember me"}]}
        ])).unwrap();
        let (_, conversation) = split_input_envelope(&input);
        let history = build_compacted_history(
            conversation,
            ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            },
        );
        assert_eq!(history.len(), 2);
        assert!(
            matches!(history[0], ResponsesInputItem::Message { ref role, .. } if role == "user")
        );
    }

    #[test]
    fn failed_replay_clears_anchored_state() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let attempt = stored_compaction(
            "failed-replay",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        activate_compaction(
            Some("failed-replay"),
            Some(attempt),
            "gpt-5.6-sol",
            &output(SUMMARY),
        );
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("failed-replay"), &next).unwrap();

        abort_compaction_attempt(Some("failed-replay"), Some(replay.attempt));

        assert!(apply_compaction_replay(Some("failed-replay"), &next).is_none());
    }

    #[test]
    fn replay_clears_on_model_change() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let attempt = stored_compaction(
            "session",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        activate_compaction(
            Some("session"),
            Some(attempt),
            "gpt-5.6-sol",
            &output(SUMMARY),
        );
        let mut changed = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        changed.model = "gpt-5.4".to_string();
        assert!(apply_compaction_replay(Some("session"), &changed).is_none());
    }

    #[test]
    fn retained_history_obeys_token_budget_at_utf8_boundary() {
        let text = "é".repeat((RETAINED_MESSAGE_TOKEN_BUDGET as usize + 10) * 4);
        let input: Vec<ResponsesInputItem> = serde_json::from_value(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":text}]}
        ]))
        .unwrap();
        let history = build_compacted_history(
            &input,
            ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            },
        );
        let ResponsesInputItem::Message { content, .. } = &history[0] else {
            panic!("expected retained message");
        };
        let ResponsesContentPart::InputText { text } = &content[0] else {
            panic!("expected retained text");
        };
        assert!(text.len() <= RETAINED_MESSAGE_TOKEN_BUDGET as usize * 4);
        assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    }
}
