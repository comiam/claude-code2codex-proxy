//! One bounded Responses lifecycle for live rendering and buffered collection.
use std::collections::HashMap;

use crate::monitor::{TokenObservation, UsageObservation};
use crate::providers::codex::events::{
    classify_event_failure, is_standard_max_output_tokens_incomplete,
};

use super::IncompleteResponsePolicy;
use super::event::{SkipDiagnostics, SkippedEvent, TranslationEvent};
use super::finish::{
    FinishMetadata, ReplayLedger, SnapshotProgress, parse_codex_usage, reasoning_input_item,
};
use super::read_rewrite::{BUFFERED_TOOL_MAX_ARGS_BYTES, completed_read_args};
use super::reasoning_signature::{PendingReasoning, encode_reasoning_signature};
use super::reducer::{
    CodexRateLimits, CodexUsage, STOP_END_TURN, STOP_MAX_TOKENS, STOP_TOOL_USE,
    parse_codex_rate_limits,
};
use super::request::ResponsesInputItem;
use super::web_search_compat::{
    SearchCard, WebSearchResult, server_tool_use_id_from_codex_web_search_id,
};

pub(super) enum LiveBlock {
    Text {
        index: usize,
        text: String,
        progress: SnapshotProgress,
        closed: bool,
    },
    Tool {
        index: usize,
        call_id: String,
        name: String,
        args_accum: String,
        progress: SnapshotProgress,
        buffer_until_done: bool,
    },
}

#[derive(Default)]
struct PartLinks {
    links: Vec<(String, String, bool)>,
}

#[derive(Clone, Copy)]
struct LiveThinking {
    output_index: usize,
    anthropic_index: usize,
}

#[derive(Debug)]
pub enum TranslationError {
    Upstream(String),
    LocalProtocol(String),
}

impl TranslationError {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Upstream(message) | Self::LocalProtocol(message) => message,
        }
    }
}

pub(crate) struct ResponseState {
    pub(super) blocks_by_output_index: HashMap<(usize, usize), LiveBlock>,
    item_id_to_output_index: HashMap<String, usize>,
    anthropic_index: usize,
    thinking: Option<LiveThinking>,
    reasoning_by_output_index: HashMap<usize, PendingReasoning>,
    summary_parts: HashMap<(usize, usize), SnapshotProgress>,
    saw_tool_use: bool,
    saw_refusal: bool,
    reasoning_unavailable: std::collections::BTreeSet<usize>,
    unresolved_output_items: HashMap<usize, String>,
    web_search_requests: usize,
    completed_searches: std::collections::BTreeMap<usize, SearchCard>,
    search_annotations: Vec<(usize, WebSearchResult)>,
    part_links: HashMap<(usize, usize), PartLinks>,
    had_compatibility_skips: bool,
    skip_diagnostics: SkipDiagnostics,
    semantic_output_started: bool,
    answer_output_started: bool,
    completed_tool_use: bool,
    early_reasoning: bool,
    early_reasoning_started: bool,
    incomplete_response_policy: IncompleteResponsePolicy,
    finished: bool,
    response_id: Option<String>,
    replay: Option<ReplayLedger>,
    finish_metadata: Option<FinishMetadata>,
    search_buffer_bytes: usize,
    closed_before: usize,
    closed_indices: std::collections::BTreeSet<usize>,
    pub(crate) usage: UsageObservation,
    rate_limits: Option<CodexRateLimits>,
}

impl ResponseState {
    pub(crate) fn with_early_reasoning(mut self) -> Self {
        self.early_reasoning = true;
        self
    }

    pub(crate) fn with_replay_budget(mut self, budget: Option<usize>) -> Self {
        self.replay = budget.map(ReplayLedger::new);
        self
    }

    pub fn finish_metadata(&self) -> Option<&FinishMetadata> {
        self.finish_metadata.as_ref()
    }

    fn charge_replay(&mut self, bytes: usize) {
        if self
            .replay
            .as_mut()
            .is_some_and(|ledger| !ledger.charge(bytes))
        {
            self.discard_replay();
        }
    }

    fn discard_replay(&mut self) {
        self.replay = None;
        for block in self.blocks_by_output_index.values_mut() {
            match block {
                LiveBlock::Text { text, .. } => *text = String::new(),
                LiveBlock::Tool {
                    args_accum,
                    buffer_until_done: false,
                    ..
                } => *args_accum = String::new(),
                _ => {}
            }
        }
    }

    fn check_active_limit(&self) -> Result<(), String> {
        if self.unresolved_output_items.len()
            + self.closed_indices.len()
            + self.reasoning_unavailable.len()
            + self.blocks_by_output_index.len()
            + self.summary_parts.len()
            + self.part_links.len()
            + self.completed_searches.len()
            >= 1024
        {
            return Err("Codex unresolved output item limit exceeded".to_string());
        }
        Ok(())
    }

    fn charge_search(&mut self, bytes: usize) -> Result<(), String> {
        // Bound retained URL/title metadata, never the forwarded answer.
        if bytes > (8 * 1024 * 1024usize).saturating_sub(self.search_buffer_bytes) {
            return Err("Codex hosted search buffer limit exceeded".to_string());
        }
        self.search_buffer_bytes += bytes;
        Ok(())
    }

    fn is_closed(&self, index: usize) -> bool {
        index < self.closed_before || self.closed_indices.contains(&index)
    }

    fn mark_closed(&mut self, index: usize) -> Result<(), String> {
        self.closed_indices.insert(index);
        while self.closed_indices.remove(&self.closed_before) {
            self.closed_before += 1;
        }
        self.check_active_limit()
    }

    fn text_snapshot(
        &mut self,
        index: usize,
        content_index: usize,
        text: &str,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        if let Some(links) = self.part_links.get_mut(&(index, content_index)) {
            for (_, url, represented) in &mut links.links {
                *represented |= super::web_search_compat::source_url_is_represented(text, url);
            }
        }
        let suffix = match self.blocks_by_output_index.get(&(index, content_index)) {
            Some(LiveBlock::Text { progress, .. }) => progress.missing_suffix(text)?,
            _ if self.is_closed(index) => return Ok(()),
            _ => text,
        };
        if !suffix.is_empty() {
            self.text_delta(
                &serde_json::json!({"output_index": index, "content_index": content_index, "delta": suffix}),
                out,
            )?;
        }
        Ok(())
    }

    fn tool_snapshot(
        &mut self,
        index: usize,
        args: &str,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let suffix = match self.blocks_by_output_index.get(&(index, 0)) {
            Some(LiveBlock::Tool { progress, .. }) => progress.missing_suffix(args)?,
            _ => return Ok(()),
        };
        if matches!(
            self.blocks_by_output_index.get(&(index, 0)),
            Some(LiveBlock::Tool {
                buffer_until_done: true,
                ..
            })
        ) {
            // Reconcile the snapshot while keeping Read buffered until item.done.
            self.charge_replay(suffix.len());
            if let Some(LiveBlock::Tool {
                args_accum,
                progress,
                ..
            }) = self.blocks_by_output_index.get_mut(&(index, 0))
            {
                if suffix.len() > BUFFERED_TOOL_MAX_ARGS_BYTES.saturating_sub(args_accum.len()) {
                    return Err("Buffered Read tool arguments exceeded safe limits".to_string());
                }
                args_accum.push_str(suffix);
                progress.append(suffix);
            }
            return Ok(());
        }
        if !suffix.is_empty() {
            self.tool_delta(
                &serde_json::json!({"output_index": index, "delta": suffix}),
                out,
            )?;
        }
        Ok(())
    }

    pub fn new() -> Self {
        Self {
            blocks_by_output_index: HashMap::new(),
            item_id_to_output_index: HashMap::new(),
            anthropic_index: 0,
            thinking: None,
            reasoning_by_output_index: HashMap::new(),
            summary_parts: HashMap::new(),
            saw_tool_use: false,
            saw_refusal: false,
            reasoning_unavailable: std::collections::BTreeSet::new(),
            unresolved_output_items: HashMap::new(),
            web_search_requests: 0,
            completed_searches: std::collections::BTreeMap::new(),
            search_annotations: Vec::new(),
            part_links: HashMap::new(),
            had_compatibility_skips: false,
            skip_diagnostics: SkipDiagnostics::default(),
            semantic_output_started: false,
            answer_output_started: false,
            completed_tool_use: false,
            early_reasoning: false,
            early_reasoning_started: false,
            incomplete_response_policy: IncompleteResponsePolicy::Error,
            finished: false,
            response_id: None,
            replay: None,
            finish_metadata: None,
            search_buffer_bytes: 0,
            closed_before: 0,
            closed_indices: std::collections::BTreeSet::new(),
            usage: UsageObservation::default(),
            rate_limits: None,
        }
    }

    pub(crate) fn with_incomplete_response_policy(
        mut self,
        policy: IncompleteResponsePolicy,
    ) -> Self {
        self.incomplete_response_policy = policy;
        self
    }

    pub fn accept(
        &mut self,
        payload: &serde_json::Value,
    ) -> Result<Vec<TranslationEvent>, TranslationError> {
        if self.finished {
            if payload["type"] == "ping" {
                return Ok(Vec::new());
            }
            let message = if crate::providers::codex::events::event_is_terminal(payload) {
                "upstream stream contained multiple terminal Codex response events"
            } else {
                "upstream stream contained a JSON event after the terminal Codex response event"
            };
            return Err(TranslationError::LocalProtocol(message.into()));
        }
        // Trustworthy terminal compute usage survives local output rejection.
        // This says nothing about successful translation or body delivery.
        if crate::providers::codex::events::event_is_terminal(payload)
            && let Some(response) = payload.get("response")
        {
            let usage = parse_codex_usage(response);
            self.usage = UsageObservation {
                input_tokens: usage.input_tokens.map(TokenObservation::provider),
                cached_input_tokens: usage.input_tokens_details_cached,
                output_tokens: usage.output_tokens.map(TokenObservation::provider),
                reasoning_tokens: usage.output_tokens_details_reasoning,
            };
        }
        let allowed_incomplete = self.incomplete_response_policy
            == IncompleteResponsePolicy::AllowMaxOutputTokens
            && is_standard_max_output_tokens_incomplete(payload);
        if !allowed_incomplete && let Some(failure) = classify_event_failure(payload) {
            return Err(TranslationError::Upstream(failure.message));
        }
        self.accept_inner(payload)
            .map_err(TranslationError::LocalProtocol)
    }

    fn accept_inner(
        &mut self,
        payload: &serde_json::Value,
    ) -> Result<Vec<TranslationEvent>, String> {
        let kind = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if let Some(id) = payload
            .pointer("/response/id")
            .and_then(serde_json::Value::as_str)
        {
            if id.len() > 4096 {
                return Err("Codex response identity limit exceeded".into());
            }
            // A precommit transport retry may supply a new created envelope.
            self.response_id = Some(id.to_owned());
        }
        let mut out = Vec::new();

        match kind {
            "codex.rate_limits" => {
                if let Some(snapshot) = parse_codex_rate_limits(payload) {
                    self.rate_limits = Some(snapshot);
                }
                self.emit_ping(&mut out);
            }
            "keepalive" | "response.created" | "response.in_progress" => {
                self.emit_ping(&mut out);
            }
            "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed" => {
                self.skip(
                    payload,
                    "anthropic_search_opaque_replay_unavailable",
                    &mut out,
                );
                out.push(TranslationEvent::Progress {
                    output_index: payload["output_index"].as_u64().map(|v| v as usize),
                    item_id: payload["item_id"]
                        .as_str()
                        .map(|v| v.chars().take(256).collect()),
                });
            }
            "response.output_item.added" => {
                self.output_item_added(payload, &mut out)?;
            }
            "response.reasoning_summary_part.added" | "response.reasoning_summary_text.delta" => {
                self.reasoning_delta(payload, &mut out)?;
            }
            "response.reasoning_summary_text.done" | "response.reasoning_summary_part.done" => {
                let text = payload
                    .get("text")
                    .or_else(|| payload.pointer("/part/text"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                self.reasoning_snapshot(
                    self.output_index(payload)?,
                    summary_index(payload),
                    text,
                    &mut out,
                )?;
            }
            "response.content_part.added" => {
                if let Some(text) = self.part_text(&payload["part"], payload, &mut out) {
                    self.text_delta(payload, &mut out)?;
                    let index = self.output_index(payload)?;
                    let part = content_index(payload);
                    self.text_snapshot(index, part, text, &mut out)?;
                    self.capture_part_annotations(
                        index,
                        part,
                        &payload["part"],
                        payload,
                        &mut out,
                    )?;
                }
            }
            "response.content_part.done" => {
                if let Some(text) = self.part_text(&payload["part"], payload, &mut out) {
                    let index = self.output_index(payload)?;
                    let part = content_index(payload);
                    self.text_snapshot(index, part, text, &mut out)?;
                    if let Some(LiveBlock::Text { index, closed, .. }) =
                        self.blocks_by_output_index.get_mut(&(index, part))
                    {
                        if !*closed {
                            out.push(TranslationEvent::TextStop { index: *index });
                        }
                        *closed = true;
                    }
                    self.capture_part_annotations(
                        index,
                        part,
                        &payload["part"],
                        payload,
                        &mut out,
                    )?;
                    self.emit_part_links(index, part, Some(text), &mut out);
                }
            }
            "response.refusal.delta" => {
                self.saw_refusal = true;
                self.text_delta(payload, &mut out)?;
            }
            "response.refusal.done" => {
                self.saw_refusal = true;
                if let Some(text) = payload.get("refusal").and_then(serde_json::Value::as_str) {
                    self.text_snapshot(
                        self.output_index(payload)?,
                        content_index(payload),
                        text,
                        &mut out,
                    )?;
                }
            }
            "response.output_text.delta" => {
                self.text_delta(payload, &mut out)?;
            }
            "response.output_text.done" => {
                if let Some(text) = payload.get("text").and_then(serde_json::Value::as_str) {
                    self.text_snapshot(
                        self.output_index(payload)?,
                        content_index(payload),
                        text,
                        &mut out,
                    )?;
                }
            }
            "response.output_text.annotation.added" => {
                self.web_search_annotation(payload, &mut out)?;
            }
            "response.function_call_arguments.delta" => {
                self.tool_delta(payload, &mut out)?;
            }
            "response.function_call_arguments.done" => {
                self.tool_arguments_done(payload, &mut out)?;
            }
            "response.output_item.done" => {
                if self.is_closed(self.output_index(payload)?) {
                    self.skip(payload, "redundant_snapshot", &mut out);
                    return Ok(out);
                }
                if payload
                    .pointer("/item/type")
                    .and_then(serde_json::Value::as_str)
                    == Some("function_call")
                    && !self
                        .blocks_by_output_index
                        .contains_key(&(self.output_index(payload)?, 0))
                {
                    return Err(
                        "unsupported Codex sequence: function_call done without added item"
                            .to_string(),
                    );
                }
                self.output_item_done(payload, &mut out)?;
            }
            "response.completed" | "response.incomplete" | "response.done" => {
                if self.saw_tool_use && !self.reasoning_unavailable.is_empty() {
                    return Err(
                        "completed reasoning replay unavailable for tool round-trip".to_string()
                    );
                }
                self.finish(payload, &mut out)?;
            }
            _ => self.skip(payload, skip_reason(kind), &mut out),
        }

        Ok(out)
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn has_semantic_output(&self) -> bool {
        // An empty published thinking block commits retries, but cannot by
        // itself satisfy the successful-completion content check.
        self.semantic_output_started || self.early_reasoning_started
    }

    pub(crate) fn abort(&mut self) -> Vec<TranslationEvent> {
        let mut out = Vec::new();
        if !self.finished {
            self.close_open_blocks(&mut out);
            self.finished = true;
        }
        out
    }

    pub(crate) fn eof(&self) -> Result<(), TranslationError> {
        if self.finished {
            Ok(())
        } else {
            Err(TranslationError::LocalProtocol(
                "upstream stream ended before terminal Codex response event".to_string(),
            ))
        }
    }

    fn part_text<'a>(
        &mut self,
        part: &'a serde_json::Value,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Option<&'a str> {
        match part["type"].as_str() {
            Some("output_text") => Some(part["text"].as_str().unwrap_or("")),
            Some("refusal") => {
                self.saw_refusal = true;
                Some(part["refusal"].as_str().unwrap_or(""))
            }
            _ => {
                self.skip(payload, "unsupported_part", out);
                self.discard_replay();
                None
            }
        }
    }

    fn output_index(&self, payload: &serde_json::Value) -> Result<usize, String> {
        let explicit = payload
            .get("output_index")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);
        let id = payload
            .get("item_id")
            .or_else(|| payload.pointer("/item/id"))
            .and_then(serde_json::Value::as_str);
        let known = id
            .and_then(|id| self.item_id_to_output_index.get(id))
            .copied();
        if explicit.zip(known).is_some_and(|(a, b)| a != b)
            || explicit.zip(id).is_some_and(|(index, id)| {
                self.item_id_to_output_index
                    .iter()
                    .any(|(registered_id, registered_index)| {
                        *registered_index == index && registered_id != id
                    })
            })
        {
            return Err("Conflicting Codex item identity".into());
        }
        explicit
            .or(known)
            .ok_or_else(|| "Codex output event has no known item identity".into())
    }

    pub(crate) fn open_blocks(&self) -> Vec<super::reducer::OpenBlockDiagnostic> {
        self.blocks_by_output_index
            .iter()
            .map(|((output_index, _), block)| {
                let (index, kind, name, call_id, text_bytes, argument_bytes) = match block {
                    LiveBlock::Text {
                        index, progress, ..
                    } => (*index, "text", None, None, Some(progress.bytes()), None),
                    LiveBlock::Tool {
                        index,
                        name,
                        call_id,
                        progress,
                        ..
                    } => (
                        *index,
                        "tool",
                        Some(name.clone()),
                        Some(call_id.clone()),
                        None,
                        Some(progress.bytes()),
                    ),
                };
                super::reducer::OpenBlockDiagnostic {
                    output_index: *output_index,
                    anthropic_index: index,
                    kind: kind.into(),
                    name,
                    call_id,
                    text_bytes,
                    argument_bytes,
                }
            })
            .collect()
    }

    fn emit_ping(&mut self, out: &mut Vec<TranslationEvent>) {
        out.push(TranslationEvent::Progress {
            output_index: None,
            item_id: None,
        });
    }

    fn output_item_added(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let Some(item) = payload.get("item") else {
            return Err("Codex output item start is missing its item".to_string());
        };
        // Item identity is retained until done; bound it independently of the
        // frame containing it, consistent with the reasoning signature ID cap.
        if ["id", "call_id", "name", "type"].iter().any(|key| {
            item.get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value.len() > 4096)
        }) {
            return Err("Codex output item identity limit exceeded".to_string());
        }
        let output_index = self.output_index(payload)?;
        if self.is_closed(output_index) || self.unresolved_output_items.contains_key(&output_index)
        {
            return Err("Duplicate Codex output item start".to_string());
        }
        self.check_active_limit()?;
        self.charge_replay(
            256 + item
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map_or(0, str::len)
                + item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .map_or(0, str::len)
                + item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map_or(0, str::len),
        );
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        self.unresolved_output_items
            .insert(output_index, item_type.to_string());

        if let Some(id) = item.get("id").and_then(serde_json::Value::as_str) {
            if self
                .item_id_to_output_index
                .get(id)
                .is_some_and(|known| *known != output_index)
            {
                return Err("Conflicting Codex item identity".to_string());
            }
            self.item_id_to_output_index
                .insert(id.to_owned(), output_index);
        }
        match item_type {
            "reasoning" => {
                self.reasoning_by_output_index
                    .entry(output_index)
                    .or_default()
                    .capture(item);
                self.reasoning_unavailable.insert(output_index);
                if self.early_reasoning {
                    self.early_reasoning_started = true;
                    self.start_thinking(output_index, out);
                }
            }
            "message" => {}
            "function_call" => {
                self.close_thinking(out);
                self.saw_tool_use = true;
                self.semantic_output_started = true;
                let index = self.anthropic_index;
                self.anthropic_index += 1;
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.blocks_by_output_index.insert(
                    (output_index, 0),
                    LiveBlock::Tool {
                        index,
                        call_id: call_id.clone(),
                        name: name.clone(),
                        args_accum: String::new(),
                        progress: SnapshotProgress::default(),
                        buffer_until_done: name == "Read",
                    },
                );
                out.push(TranslationEvent::ToolStart {
                    index,
                    id: call_id.clone(),
                    name: name.clone(),
                });
            }
            "web_search_call" => {
                self.discard_replay();
                self.skip(payload, "anthropic_search_opaque_replay_unavailable", out);
            }
            _ => {
                self.discard_replay();
                self.unresolved_output_items.remove(&output_index);
                self.item_id_to_output_index
                    .retain(|_, i| *i != output_index);
                self.mark_closed(output_index)?;
                self.skip(payload, "unsupported_output_item", out);
            }
        }
        Ok(())
    }

    fn reasoning_snapshot(
        &mut self,
        output_index: usize,
        summary_index: usize,
        text: &str,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let suffix = match self.summary_parts.get(&(output_index, summary_index)) {
            Some(progress) => progress.missing_suffix(text)?,
            None => text,
        };
        if !suffix.is_empty() {
            self.reasoning_delta(&serde_json::json!({"output_index":output_index,"summary_index":summary_index,"delta":suffix}), out)?;
        }
        Ok(())
    }

    fn start_thinking(&mut self, output_index: usize, out: &mut Vec<TranslationEvent>) -> usize {
        if self.thinking.map(|thinking| thinking.output_index) != Some(output_index) {
            self.close_thinking(out);
            let index = self.anthropic_index;
            self.anthropic_index += 1;
            self.thinking = Some(LiveThinking {
                output_index,
                anthropic_index: index,
            });
            out.push(TranslationEvent::ThinkingStart { index });
        }
        self.thinking
            .expect("thinking block was started")
            .anthropic_index
    }

    fn reasoning_delta(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let output_index = self.output_index(payload)?;
        let delta = payload.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        let key = (output_index, summary_index(payload));
        if self.is_closed(output_index) {
            return Err("Reasoning delta after completed item".into());
        }
        let new_part = !self.summary_parts.contains_key(&key);
        let separate = new_part
            && self
                .summary_parts
                .keys()
                .any(|(item, _)| *item == output_index);
        if new_part {
            self.check_active_limit()?;
            self.summary_parts.insert(key, SnapshotProgress::default());
        }
        self.summary_parts.get_mut(&key).unwrap().append(delta);
        self.unresolved_output_items
            .entry(output_index)
            .or_insert_with(|| "reasoning".into());
        self.reasoning_unavailable.insert(output_index);
        self.semantic_output_started |= !delta.is_empty();
        let index = self.start_thinking(output_index, out);
        if separate {
            out.push(TranslationEvent::ThinkingDelta {
                index,
                text: "\n\n".into(),
            });
        }
        if delta.is_empty() {
            return Ok(());
        }
        out.push(TranslationEvent::ThinkingDelta {
            index,
            text: (delta).to_string(),
        });
        Ok(())
    }

    fn text_delta(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        self.close_thinking(out);
        let delta = payload.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() && payload["type"] != "response.content_part.added" {
            return Ok(());
        }
        self.semantic_output_started |= !delta.is_empty();

        let output_index = self.output_index(payload)?;
        let content_index = content_index(payload);

        if self.is_closed(output_index) {
            return Err("Text delta after completed item".to_string());
        }
        self.charge_replay(delta.len());
        if !self
            .blocks_by_output_index
            .contains_key(&(output_index, content_index))
        {
            self.check_active_limit()?;
            self.charge_replay(256);
            self.unresolved_output_items
                .insert(output_index, "message".to_string());
            let index = self.anthropic_index;
            self.anthropic_index += 1;
            self.blocks_by_output_index.insert(
                (output_index, content_index),
                LiveBlock::Text {
                    index,
                    text: String::new(),
                    progress: SnapshotProgress::default(),
                    closed: false,
                },
            );
            out.push(TranslationEvent::TextStart { index });
        }
        let retain = self.replay.is_some();
        let Some(LiveBlock::Text {
            index,
            text,
            progress,
            closed,
        }) = self
            .blocks_by_output_index
            .get_mut(&(output_index, content_index))
        else {
            return Err("unsupported Codex sequence: text delta for non-message item".to_string());
        };
        if *closed {
            return Err("Text delta after completed part".to_string());
        }
        progress.append(delta);
        self.answer_output_started |= !delta.is_empty();
        if retain {
            text.push_str(delta);
        }
        let index = *index;
        out.push(TranslationEvent::TextDelta {
            index,
            text: (delta).to_string(),
        });
        Ok(())
    }

    fn tool_delta(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let output_index = self.output_index(payload)?;
        let delta = payload.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() {
            return Ok(());
        }
        self.semantic_output_started = true;
        self.charge_replay(delta.len());
        let retain = self.replay.is_some();
        let Some(LiveBlock::Tool {
            index,
            name,
            args_accum,
            progress,
            buffer_until_done,
            ..
        }) = self.blocks_by_output_index.get_mut(&(output_index, 0))
        else {
            return Err(
                "unsupported Codex sequence: tool arguments without added item".to_string(),
            );
        };
        if *buffer_until_done
            && delta.len() > BUFFERED_TOOL_MAX_ARGS_BYTES.saturating_sub(args_accum.len())
        {
            return Err(format!(
                "Buffered {name} tool arguments exceeded safe limits"
            ));
        }
        progress.append(delta);
        if retain || *buffer_until_done {
            args_accum.push_str(delta);
        }
        if !*buffer_until_done {
            let index = *index;
            self.semantic_output_started = true;
            out.push(TranslationEvent::ToolDelta {
                index,
                partial_json: (delta).to_string(),
            });
        }
        Ok(())
    }

    fn tool_arguments_done(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let index = self.output_index(payload)?;
        if let Some(args) = payload.get("arguments").and_then(serde_json::Value::as_str) {
            self.tool_snapshot(index, args, out)?;
        }
        Ok(())
    }

    fn output_item_done(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        let output_index = self.output_index(payload)?;
        if payload.get("item").is_none() {
            return Err("Codex output item completion is missing its item".to_string());
        }
        if let Some(item) = payload.get("item") {
            if let Some(known) = self.unresolved_output_items.get(&output_index)
                && item
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|kind| kind != known)
            {
                return Err("Conflicting Codex item type".into());
            }
            if let Some(id) = item.get("id").and_then(serde_json::Value::as_str)
                && self
                    .item_id_to_output_index
                    .iter()
                    .any(|(known, index)| *index == output_index && known != id)
            {
                return Err("Conflicting Codex item identity".into());
            }
            if let Some(LiveBlock::Tool { call_id, name, .. }) =
                self.blocks_by_output_index.get(&(output_index, 0))
                && (item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|id| id != call_id)
                    || item
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| value != name))
            {
                return Err("Conflicting Codex function call identity".into());
            }
        }
        if let Some(item) = payload.get("item") {
            if !matches!(
                item.get("type").and_then(serde_json::Value::as_str),
                Some("message" | "reasoning" | "function_call" | "web_search_call")
            ) {
                self.discard_replay();
                self.unresolved_output_items.remove(&output_index);
                self.mark_closed(output_index)?;
                self.skip(payload, "unsupported_output_item", out);
            }
            match item.get("type").and_then(serde_json::Value::as_str) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(serde_json::Value::as_array) {
                        for (content_index, part) in parts.iter().enumerate() {
                            if let Some(text) = self.part_text(part, payload, out) {
                                self.text_snapshot(output_index, content_index, text, out)?;
                                self.capture_part_annotations(
                                    output_index,
                                    content_index,
                                    part,
                                    payload,
                                    out,
                                )?;
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(parts) = item.get("summary").and_then(serde_json::Value::as_array) {
                        for (summary_index, part) in parts.iter().enumerate() {
                            if let Some(text) = part.get("text").and_then(serde_json::Value::as_str)
                            {
                                self.reasoning_snapshot(output_index, summary_index, text, out)?;
                            }
                        }
                    }
                }
                Some("function_call") => {
                    if let Some(args) = item.get("arguments").and_then(serde_json::Value::as_str) {
                        self.tool_snapshot(output_index, args, out)?;
                    }
                }
                _ => {}
            }
        }
        let completed_search_id = payload["item"]["id"]
            .as_str()
            .map(str::to_owned)
            .or_else(|| {
                self.item_id_to_output_index
                    .iter()
                    .find(|(_, index)| **index == output_index)
                    .map(|(id, _)| id.clone())
            });
        self.item_id_to_output_index
            .retain(|_, index| *index != output_index);

        if let Some(item_type) = payload
            .pointer("/item/type")
            .and_then(serde_json::Value::as_str)
            && matches!(
                item_type,
                "reasoning" | "message" | "function_call" | "web_search_call"
            )
            && self
                .unresolved_output_items
                .get(&output_index)
                .is_some_and(|kind| kind == item_type)
        {
            self.unresolved_output_items.remove(&output_index);
        }

        if let Some(item) = payload
            .get("item")
            .and_then(|item| item.get("type"))
            .and_then(|v| v.as_str())
            .filter(|item_type| *item_type == "reasoning")
            .and_then(|_| payload.get("item"))
        {
            self.reasoning_by_output_index
                .entry(output_index)
                .or_default()
                .capture_done(item);
            if self
                .reasoning_by_output_index
                .get(&output_index)
                .is_some_and(PendingReasoning::has_encodable_replay)
            {
                self.reasoning_unavailable.remove(&output_index);
            } else {
                self.reasoning_unavailable.insert(output_index);
                self.discard_replay();
                crate::logging::create_logger("codex").warn("reasoning_replay_unavailable", serde_json::json!({
                    "source":"live_translator", "phase":"completed_reasoning", "output_index":output_index,
                    "semantic_committed":self.has_semantic_output(), "continuation_eligible":false
                }).as_object().cloned());
            }
            if let Some(replay) = self
                .reasoning_by_output_index
                .get(&output_index)
                .and_then(PendingReasoning::replay)
            {
                self.charge_replay(replay.id.len() + replay.encrypted_content.len() + 256);
                if let Some(ledger) = self.replay.as_mut() {
                    ledger
                        .items
                        .insert(output_index, reasoning_input_item(replay));
                }
            }
            self.mark_closed(output_index)?;
            let had_active_summary = self
                .thinking
                .is_some_and(|thinking| thinking.output_index == output_index);
            self.close_thinking(out);
            if !had_active_summary {
                self.emit_signature_only_reasoning(output_index, out);
            }
            self.reasoning_by_output_index.remove(&output_index);
            self.summary_parts
                .retain(|(item, _), _| *item != output_index);
            return Ok(());
        }

        if payload
            .get("item")
            .and_then(|item| item.get("type"))
            .and_then(|v| v.as_str())
            == Some("web_search_call")
        {
            self.close_thinking(out);
            self.discard_replay();
            self.mark_closed(output_index)?;
            self.skip(payload, "anthropic_search_opaque_replay_unavailable", out);
            for path in ["/action/sources", "/results"] {
                if let Some(links) = payload["item"]
                    .pointer(path)
                    .and_then(serde_json::Value::as_array)
                {
                    for link in links {
                        self.capture_link(output_index, 0, link)?;
                    }
                }
            }
            self.emit_part_links(output_index, 0, None, out);
            let links = self
                .part_links
                .remove(&(output_index, 0))
                .unwrap_or_default();
            self.web_search_requests += 1;
            // Keep completed calls until terminal: only then can unscoped answer
            // citations be attributed safely to a response with one search.
            // The Messages ID is local transport identity, never an upstream
            // replay credential. An omitted item.id still has a validated index.
            let id = completed_search_id
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| {
                    format!(
                        "ccp_local_{}_{}",
                        self.response_id.as_deref().unwrap_or("response"),
                        output_index
                    )
                });
            let input = payload["item"]
                .get("action")
                .filter(|action| action.is_object())
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            self.charge_search(id.len() + input.to_string().len() + 128)?;
            self.completed_searches.insert(
                output_index,
                SearchCard {
                    id: server_tool_use_id_from_codex_web_search_id(&id),
                    input,
                    results: links
                        .links
                        .into_iter()
                        .map(|(title, url, _)| WebSearchResult { title, url })
                        .collect(),
                },
            );
            self.check_active_limit()?;
            return Ok(());
        }

        if payload
            .pointer("/item/type")
            .and_then(serde_json::Value::as_str)
            == Some("message")
        {
            let mut keys: Vec<_> = self
                .blocks_by_output_index
                .keys()
                .filter(|(item, _)| *item == output_index)
                .copied()
                .collect();
            keys.sort_unstable();
            let mut replay_parts = Vec::new();
            for key in keys {
                if let Some(LiveBlock::Text {
                    index,
                    text,
                    closed,
                    ..
                }) = self.blocks_by_output_index.remove(&key)
                {
                    if self.replay.is_some() {
                        replay_parts.push(super::request::ResponsesContentPart::OutputText {
                            text: text.clone(),
                        });
                    }
                    if !closed {
                        out.push(TranslationEvent::TextStop { index });
                    }
                }
            }
            if let Some(ledger) = self.replay.as_mut() {
                ledger.items.insert(
                    output_index,
                    ResponsesInputItem::Message {
                        role: "assistant".into(),
                        content: replay_parts,
                    },
                );
            }
            if let Some(parts) = payload
                .pointer("/item/content")
                .and_then(serde_json::Value::as_array)
            {
                for (part_index, part) in parts.iter().enumerate() {
                    self.emit_part_links(output_index, part_index, part["text"].as_str(), out);
                }
            } else {
                let parts: Vec<_> = self
                    .part_links
                    .keys()
                    .filter(|(item, _)| *item == output_index)
                    .map(|(_, part)| *part)
                    .collect();
                for part in parts {
                    self.emit_part_links(output_index, part, None, out);
                }
            }
            self.part_links.retain(|(item, _), _| *item != output_index);
            self.mark_closed(output_index)?;
            return Ok(());
        }

        let Some(mut state) = self.blocks_by_output_index.remove(&(output_index, 0)) else {
            return Ok(());
        };

        match &mut state {
            LiveBlock::Text { .. } => unreachable!("message parts completed above"),
            LiveBlock::Tool {
                index,
                name,
                call_id,
                args_accum,
                buffer_until_done,
                ..
            } => {
                self.semantic_output_started = true;
                if *buffer_until_done {
                    *args_accum = completed_read_args(args_accum, Some(call_id.as_str()))?;
                    out.push(TranslationEvent::ToolDelta {
                        index: *index,
                        partial_json: (args_accum).to_string(),
                    });
                }
                if let Some(ledger) = self.replay.as_mut() {
                    ledger.items.insert(
                        output_index,
                        ResponsesInputItem::FunctionCall {
                            call_id: std::mem::take(call_id),
                            name: std::mem::take(name),
                            arguments: std::mem::take(args_accum),
                        },
                    );
                }
                self.completed_tool_use = true;
                out.push(TranslationEvent::ToolStop { index: *index });
            }
        }
        self.mark_closed(output_index)?;
        Ok(())
    }

    pub(crate) fn set_request_id(&mut self, req_id: &str) {
        self.skip_diagnostics.set_request_id(req_id);
    }
    pub(crate) fn had_compatibility_skips(&self) -> bool {
        self.had_compatibility_skips
    }
    fn skip(
        &mut self,
        payload: &serde_json::Value,
        reason: &'static str,
        out: &mut Vec<TranslationEvent>,
    ) {
        if reason != "redundant_snapshot" {
            self.had_compatibility_skips = true;
        }
        let event = SkippedEvent::new(payload, reason);
        self.skip_diagnostics.record(&event);
        out.push(TranslationEvent::Skipped(event));
    }
    fn capture_link(
        &mut self,
        item: usize,
        part: usize,
        link: &serde_json::Value,
    ) -> Result<(), String> {
        let Some(url) = link["url"]
            .as_str()
            .filter(|v| v.starts_with("https://") || v.starts_with("http://"))
        else {
            return Ok(());
        };
        if self
            .part_links
            .get(&(item, part))
            .is_some_and(|p| p.links.iter().any(|(_, u, _)| u == url))
        {
            return Ok(());
        }
        self.check_active_limit()?;
        let title = link["title"]
            .as_str()
            .filter(|v| !v.is_empty())
            .unwrap_or(url);
        self.charge_search(title.len() + url.len() + 128)?;
        self.part_links
            .entry((item, part))
            .or_default()
            .links
            .push((title.to_owned(), url.to_owned(), false));
        Ok(())
    }
    fn capture_search_annotation(
        &mut self,
        item: usize,
        annotation: &serde_json::Value,
    ) -> Result<(), String> {
        let Some(result) = WebSearchResult::from_value(annotation) else {
            return Ok(());
        };
        if self
            .search_annotations
            .iter()
            .any(|(known_item, known)| *known_item == item && known.url == result.url)
        {
            return Ok(());
        }
        self.charge_search(result.title.len() + result.url.len() + 128)?;
        self.search_annotations.push((item, result));
        Ok(())
    }
    fn capture_part_annotations(
        &mut self,
        item: usize,
        part: usize,
        snapshot: &serde_json::Value,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        if let Some(annotations) = snapshot["annotations"].as_array() {
            for a in annotations {
                if a["type"] == "url_citation" {
                    self.capture_link(item, part, a)?;
                    self.capture_search_annotation(item, a)?;
                } else {
                    self.skip(payload, "unsupported_annotation", out);
                    self.discard_replay();
                }
            }
        }
        Ok(())
    }
    fn emit_part_links(
        &mut self,
        item: usize,
        part: usize,
        snapshot: Option<&str>,
        out: &mut Vec<TranslationEvent>,
    ) {
        let Some(links) = self.part_links.get_mut(&(item, part)) else {
            return;
        };
        let mut text = String::new();
        for (title, url, emitted) in &mut links.links {
            if *emitted {
                continue;
            }
            *emitted = true;
            if snapshot
                .is_some_and(|text| super::web_search_compat::source_url_is_represented(text, url))
            {
                continue;
            }
            text.push_str(&super::web_search_compat::markdown_link(title, url));
        }
        if text.is_empty() {
            return;
        }
        let index = self.anthropic_index;
        self.anthropic_index += 1;
        self.semantic_output_started = true;
        self.answer_output_started = true;
        out.extend([
            TranslationEvent::TextStart { index },
            TranslationEvent::TextDelta { index, text },
            TranslationEvent::TextStop { index },
        ]);
    }
    fn web_search_annotation(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        if payload["annotation"]["type"] != "url_citation" {
            self.skip(payload, "unsupported_annotation", out);
            self.discard_replay();
            return Ok(());
        }
        let item = self.output_index(payload)?;
        if self.is_closed(item) {
            self.skip(payload, "redundant_snapshot", out);
            return Ok(());
        }
        self.capture_link(item, content_index(payload), &payload["annotation"])?;
        self.capture_search_annotation(item, &payload["annotation"])
    }

    fn finish(
        &mut self,
        payload: &serde_json::Value,
        out: &mut Vec<TranslationEvent>,
    ) -> Result<(), String> {
        if let Some(items) = payload
            .pointer("/response/output")
            .and_then(serde_json::Value::as_array)
        {
            for (index, item) in items.iter().enumerate() {
                if !matches!(
                    item["type"].as_str(),
                    Some("message" | "reasoning" | "function_call" | "web_search_call")
                ) && !self.is_closed(index)
                {
                    self.discard_replay();
                    self.skip(&serde_json::json!({"type":payload["type"],"output_index":index,"item":{"id":item["id"]}}), "unsupported_output_item", out);
                    self.mark_closed(index)?;
                }
            }
        }
        let unresolved_items =
            self.unresolved_output_items.len() + self.blocks_by_output_index.len();
        let terminal_output_resolved = payload
            .pointer("/response/output")
            .and_then(serde_json::Value::as_array)
            .is_none_or(|items| {
                items.iter().enumerate().all(|(index, item)| {
                    let _ = item;
                    self.is_closed(index)
                })
            });
        let stream_success = crate::providers::codex::events::event_is_success_terminal(payload)
            && unresolved_items == 0
            && terminal_output_resolved;
        let allowed_incomplete = self.incomplete_response_policy
            == IncompleteResponsePolicy::AllowMaxOutputTokens
            && is_standard_max_output_tokens_incomplete(payload);
        if !stream_success && !allowed_incomplete {
            return Err("Codex completion contains unresolved or unsupported output".to_string());
        }
        // Thinking commits the stream but is not an answer. Skipped items and
        // reasoning signatures cannot turn a thinking-only response into success.
        if self.early_reasoning_started
            && !self.answer_output_started
            && !self.completed_tool_use
            && self.completed_searches.is_empty()
        {
            return Err("Codex completed without producing output".into());
        }
        let replay_available = self.replay.is_some();
        let mut usage = payload.get("response").map(parse_codex_usage);
        if let Some(snapshot) = &self.rate_limits {
            usage.get_or_insert_with(CodexUsage::default).rate_limits =
                Some(Box::new(snapshot.clone()));
        }
        self.finish_metadata = Some(FinishMetadata {
            stream_success,
            continuation_eligible: stream_success && replay_available,
            replay_available,
            response_id: self.response_id.clone(),
            terminal_type: payload["type"].as_str().unwrap_or("").to_string(),
            usage: usage.clone(),
            unresolved_items,
            output_items: self
                .replay
                .take()
                .map(|ledger| ledger.items.into_values().collect())
                .unwrap_or_default(),
        });
        self.close_thinking(out);
        self.close_open_blocks(out);
        // Late atomic pairs preserve incremental answer text and never expose an
        // orphan use block on failure. Ambiguous multi-search citations stay text.
        if self.web_search_requests == 1 && self.completed_searches.len() == 1 {
            let (search_index, card) = self.completed_searches.iter_mut().next().unwrap();
            for (message_index, result) in std::mem::take(&mut self.search_annotations) {
                // A citation in a preceding message cannot belong to this new
                // search. Its ordinary text link is already preserved separately.
                if message_index > *search_index
                    && !card.results.iter().any(|known| known.url == result.url)
                {
                    card.results.push(result);
                }
            }
        }
        for (_, card) in std::mem::take(&mut self.completed_searches) {
            let index = self.anthropic_index;
            self.anthropic_index += 2;
            self.semantic_output_started = true;
            out.push(TranslationEvent::SearchComplete { index, card });
        }
        self.search_annotations.clear();
        self.skip_diagnostics.no_representable_output = !self.semantic_output_started;
        let stop_reason = if allowed_incomplete {
            STOP_MAX_TOKENS
        } else if self.saw_tool_use {
            STOP_TOOL_USE
        } else if self.saw_refusal {
            "refusal"
        } else {
            STOP_END_TURN
        };
        self.emit_finish(stop_reason, usage, out);
        Ok(())
    }

    fn emit_finish(
        &mut self,
        stop_reason: super::reducer::StopReason,
        usage: Option<CodexUsage>,
        out: &mut Vec<TranslationEvent>,
    ) {
        out.push(TranslationEvent::Finish {
            stop_reason,
            usage,
            web_search_requests: self.web_search_requests,
        });
        self.finished = true;
    }

    fn close_open_blocks(&mut self, out: &mut Vec<TranslationEvent>) {
        self.close_thinking(out);
        let open: Vec<_> = self.blocks_by_output_index.keys().copied().collect();
        for output_index in open {
            let Some(state) = self.blocks_by_output_index.remove(&output_index) else {
                continue;
            };
            let index = match state {
                LiveBlock::Text { closed: true, .. } => continue,
                LiveBlock::Text { index, .. } => {
                    out.push(TranslationEvent::TextStop { index });
                    continue;
                }
                LiveBlock::Tool { index, .. } => index,
            };
            out.push(TranslationEvent::ToolStop { index });
        }
    }

    fn emit_signature_only_reasoning(
        &mut self,
        output_index: usize,
        out: &mut Vec<TranslationEvent>,
    ) {
        let Some(replay) = self
            .reasoning_by_output_index
            .get(&output_index)
            .and_then(PendingReasoning::replay)
        else {
            return;
        };
        let Some(signature) = encode_reasoning_signature(&replay) else {
            return;
        };
        self.reasoning_by_output_index.remove(&output_index);
        self.semantic_output_started = true;
        let index = self.anthropic_index;
        self.anthropic_index += 1;
        out.push(TranslationEvent::ThinkingStart { index });
        out.push(TranslationEvent::ThinkingSignature {
            index,
            signature: (signature).to_string(),
        });
        out.push(TranslationEvent::ThinkingStop { index });
    }

    fn close_thinking(&mut self, out: &mut Vec<TranslationEvent>) {
        let Some(thinking) = self.thinking.take() else {
            return;
        };
        if let Some(signature) = self
            .reasoning_by_output_index
            .get(&thinking.output_index)
            .and_then(PendingReasoning::replay)
            .and_then(|replay| encode_reasoning_signature(&replay))
        {
            self.reasoning_by_output_index
                .remove(&thinking.output_index);
            // A real replay carrier supplies content to an early empty block.
            self.semantic_output_started |= self.early_reasoning_started;
            out.push(TranslationEvent::ThinkingSignature {
                index: thinking.anthropic_index,
                signature: (signature).to_string(),
            });
        }
        out.push(TranslationEvent::ThinkingStop {
            index: thinking.anthropic_index,
        });
    }
}

fn content_index(payload: &serde_json::Value) -> usize {
    payload
        .get("content_index")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

fn summary_index(payload: &serde_json::Value) -> usize {
    payload
        .get("summary_index")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

fn skip_reason(kind: &str) -> &'static str {
    match kind {
        "response.file_search_call.in_progress"
        | "response.file_search_call.searching"
        | "response.file_search_call.completed"
        | "response.mcp_call.in_progress"
        | "response.mcp_call.completed"
        | "response.mcp_call.failed"
        | "response.mcp_call_arguments.delta"
        | "response.mcp_call_arguments.done"
        | "response.mcp_list_tools.in_progress"
        | "response.mcp_list_tools.completed"
        | "response.mcp_list_tools.failed"
        | "response.code_interpreter_call.in_progress"
        | "response.code_interpreter_call.interpreting"
        | "response.code_interpreter_call.completed"
        | "response.code_interpreter_call_code.delta"
        | "response.code_interpreter_call_code.done"
        | "response.shell_call_command.added"
        | "response.shell_call_command.delta"
        | "response.shell_call_command.done"
        | "response.shell_call_output_content.delta"
        | "response.shell_call_output_content.done" => "unsupported_hosted_tool",
        "response.image_generation_call.in_progress"
        | "response.image_generation_call.generating"
        | "response.image_generation_call.partial_image"
        | "response.image_generation_call.completed"
        | "response.audio.delta"
        | "response.audio.done"
        | "response.audio.transcript.delta"
        | "response.audio.transcript.done" => "unsupported_modality",
        "response.custom_tool_call_input.delta" | "response.custom_tool_call_input.done" => {
            "unsupported_tool_input"
        }
        "response.compaction.compacting" => "no_standard_event",
        "response.reasoning_text.delta" | "response.reasoning_text.done" => {
            "unsupported_raw_reasoning"
        }
        _ => "unknown_event",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn early_reasoning(summary: &str, encrypted: bool) -> ResponseState {
        let mut state = ResponseState::new().with_early_reasoning();
        let mut item = json!({"type":"reasoning","id":"r"});
        if encrypted {
            item["encrypted_content"] = json!("encrypted");
        }
        let events = state
            .accept(&json!({"type":"response.output_item.added","output_index":0,"item":item}))
            .unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, TranslationEvent::ThinkingStart { .. }))
        );
        assert!(state.has_semantic_output());
        if !summary.is_empty() {
            state.accept(&json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":summary})).unwrap();
        }
        let events = state
            .accept(&json!({"type":"response.output_item.done","output_index":0,"item":item}))
            .unwrap();
        if encrypted {
            assert!(
                events
                    .iter()
                    .any(|event| matches!(event, TranslationEvent::ThinkingSignature { .. }))
            );
        }
        state
    }

    #[test]
    fn early_thinking_alone_never_finishes_successfully() {
        for (summary, encrypted, skipped) in [
            ("", false, false),
            ("thought", false, false),
            ("", true, false),
            ("thought", true, true),
            ("", false, true),
        ] {
            let mut state = early_reasoning(summary, encrypted);
            if skipped {
                let events = state.accept(&json!({"type":"response.unknown"})).unwrap();
                assert!(
                    events
                        .iter()
                        .any(|event| matches!(event, TranslationEvent::Skipped(_)))
                );
            }
            let result = state.accept(&json!({"type":"response.completed","response":{"status":"completed","usage":{"output_tokens":5}}}));
            assert!(
                result.is_err(),
                "summary={summary:?}, encrypted={encrypted}, skipped={skipped}"
            );
            assert!(state.has_semantic_output());
            assert!(!state.is_finished());
            assert!(state.finish_metadata().is_none());
            assert!(
                state
                    .abort()
                    .iter()
                    .all(|event| !matches!(event, TranslationEvent::Finish { .. }))
            );
        }
    }

    #[test]
    fn early_thinking_accepts_real_answer_or_completed_tools() {
        for kind in ["text", "refusal", "tool", "search"] {
            let mut state = early_reasoning("thought", true);
            let events = match kind {
                "text" | "refusal" => {
                    let part = if kind == "text" {
                        json!({"type":"output_text","text":"answer"})
                    } else {
                        json!({"type":"refusal","refusal":"no"})
                    };
                    vec![
                        json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m"}}),
                        json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"m","content":[part]}}),
                    ]
                }
                "tool" => vec![
                    json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"f","call_id":"call","name":"Bash"}}),
                    json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"f","call_id":"call","name":"Bash","arguments":"{}"}}),
                ],
                _ => vec![
                    json!({"type":"response.output_item.added","output_index":1,"item":{"type":"web_search_call","id":"s"}}),
                    json!({"type":"response.output_item.done","output_index":1,"item":{"type":"web_search_call","id":"s","action":{"query":"test"}}}),
                ],
            };
            for event in events {
                state.accept(&event).unwrap();
            }
            let terminal = state
                .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
                .unwrap();
            assert!(
                terminal
                    .iter()
                    .any(|event| matches!(event, TranslationEvent::Finish { .. })),
                "{kind}"
            );
            if kind == "search" {
                assert!(
                    terminal
                        .iter()
                        .any(|event| matches!(event, TranslationEvent::SearchComplete { .. }))
                );
            }
        }
    }

    #[test]
    fn early_thinking_rejects_unfinished_tool() {
        let mut state = early_reasoning("thought", true);
        state.accept(&json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"f","call_id":"call","name":"Bash"}})).unwrap();
        assert!(
            state
                .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
                .is_err()
        );
        assert!(!state.is_finished());
        assert!(state.finish_metadata().is_none());
    }

    #[test]
    fn quota_latest_valid_snapshot_replaces_and_is_response_local() {
        let mut state = ResponseState::new();
        let progress = state.accept(&json!({"type":"codex.rate_limits","plan_type":"pro","rate_limits":{"primary":{"used_percent":10},"secondary":{"used_percent":20}}})).unwrap();
        assert!(
            progress
                .iter()
                .all(|event| !matches!(event, TranslationEvent::Finish { .. }))
        );
        state.accept(&json!({"type":"codex.rate_limits","rate_limits":{"primary":{"used_percent":"invalid"}}})).unwrap();
        assert_eq!(
            state
                .rate_limits
                .as_ref()
                .unwrap()
                .primary
                .as_ref()
                .unwrap()
                .used_percent,
            Some(10.0)
        );
        state.accept(&json!({"type":"codex.rate_limits","plan_type":"new","rate_limits":{"secondary":{"used_percent":30}}})).unwrap();
        let events = state
            .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
            .unwrap();
        let TranslationEvent::Finish {
            usage: Some(usage), ..
        } = events.last().unwrap()
        else {
            panic!("missing finish usage")
        };
        let limits = usage.rate_limits.as_ref().unwrap();
        assert_eq!(limits.plan_type.as_deref(), Some("new"));
        assert!(limits.primary.is_none());
        assert_eq!(limits.secondary.as_ref().unwrap().used_percent, Some(30.0));
        assert_eq!(
            state
                .finish_metadata()
                .unwrap()
                .usage
                .as_ref()
                .unwrap()
                .rate_limits
                .as_deref(),
            Some(limits.as_ref())
        );
        assert!(usage.input_tokens.is_none() && usage.output_tokens.is_none());
        assert!(state.usage.input_tokens.is_none() && state.usage.output_tokens.is_none());
        let mut other = ResponseState::new();
        let events = other
            .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
            .unwrap();
        let TranslationEvent::Finish {
            usage: Some(usage), ..
        } = events.last().unwrap()
        else {
            panic!("missing finish usage")
        };
        assert!(usage.rate_limits.is_none());
        assert!(usage.input_tokens.is_none() && usage.output_tokens.is_none());
    }

    #[test]
    fn review_round1_rejects_unknown_item_id_for_registered_index_before_appending() {
        let mut state = ResponseState::new().with_replay_budget(Some(4096));
        state.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call","name":"Bash"}})).unwrap();
        let error = state.accept(&json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"wrong","delta":"BAD"})).unwrap_err();
        assert!(error.as_str().contains("Conflicting Codex item identity"));

        // The failed identity must not corrupt the call. Both supported legacy
        // omissions still resolve the same item: known ID only, or index only.
        state.accept(&json!({"type":"response.function_call_arguments.delta","item_id":"f0","delta":"{\"command\":"})).unwrap();
        state.accept(&json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"pwd\"}"})).unwrap();
        state.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call","name":"Bash","arguments":"{\"command\":\"pwd\"}"}})).unwrap();
        state
            .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
            .unwrap();
        let [
            ResponsesInputItem::FunctionCall {
                call_id, arguments, ..
            },
        ] = state.finish_metadata().unwrap().output_items.as_slice()
        else {
            panic!("canonical call missing")
        };
        assert_eq!(call_id, "call");
        assert_eq!(arguments, "{\"command\":\"pwd\"}");
    }

    #[test]
    fn review_round1_empty_final_arguments_are_reconciled_as_a_snapshot() {
        for name in ["Bash", "Read"] {
            let mut state = ResponseState::new();
            state.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call","name":name}})).unwrap();
            state.accept(&json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"file_path\":\"/tmp/fixture\"}"})).unwrap();
            let error = state.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call","name":name,"arguments":""}})).unwrap_err();
            assert!(
                error.as_str().contains("Conflicting Codex output snapshot"),
                "{name}: {}",
                error.as_str()
            );
            assert!(!state.is_finished());
            assert!(state.finish_metadata().is_none());
            assert!(state.eof().is_err());
        }

        // Empty is valid when no argument prefix was emitted.
        let mut state = ResponseState::new();
        state.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call","name":"Bash"}})).unwrap();
        let done = state.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","arguments":""}})).unwrap();
        assert!(
            done.iter()
                .any(|event| matches!(event, TranslationEvent::ToolStop { .. }))
        );
        state
            .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
            .unwrap();
    }

    #[test]
    fn created_identity_is_preserved_when_terminal_omits_it() {
        let mut state = ResponseState::new().with_replay_budget(Some(4096));
        state
            .accept(&json!({"type":"response.created","response":{"id":"upstream_response"}}))
            .unwrap();
        state
            .accept(&json!({"type":"response.completed","response":{"status":"completed"}}))
            .unwrap();
        assert_eq!(
            state.finish_metadata().unwrap().response_id.as_deref(),
            Some("upstream_response")
        );
    }

    #[test]
    fn reasoning_parts_reconcile_each_snapshot_before_completed_signature() {
        let mut state = ResponseState::new();
        let mut output = Vec::new();
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r","encrypted_content":"early"}}),
            json!({"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}),
            json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"First"}),
            json!({"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":0,"text":"First full"}),
            json!({"type":"response.reasoning_summary_part.done","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"First full"}}),
            json!({"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":1,"part":{"type":"summary_text","text":""}}),
            json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":1,"delta":"Second"}),
            json!({"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":1,"text":"Second full"}),
        ] {
            let translated = state.accept(&event).unwrap();
            assert!(!translated.iter().any(|e| matches!(
                e,
                TranslationEvent::ThinkingSignature { .. }
                    | TranslationEvent::ThinkingStop { .. }
                    | TranslationEvent::Finish { .. }
            )));
            output.extend(translated);
        }
        let text: String = output
            .iter()
            .filter_map(|e| {
                if let TranslationEvent::ThinkingDelta { text, .. } = e {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "First full\n\nSecond full");
        let done = state.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"r","encrypted_content":"complete"}})).unwrap();
        assert_eq!(
            done.iter()
                .filter(|e| matches!(e, TranslationEvent::ThinkingSignature { .. }))
                .count(),
            1
        );
        assert_eq!(
            done.iter()
                .filter(|e| matches!(e, TranslationEvent::ThinkingStop { .. }))
                .count(),
            1
        );
        assert!(state.eof().is_err());
    }

    #[test]
    fn immutable_tool_identity_conflicts_are_rejected() {
        for changed in [
            json!({"type":"function_call","id":"f","call_id":"other","name":"Bash","arguments":"{}"}),
            json!({"type":"function_call","id":"other","call_id":"call","name":"Bash","arguments":"{}"}),
            json!({"type":"function_call","id":"f","call_id":"call","name":"Read","arguments":"{}"}),
        ] {
            let mut state = ResponseState::new();
            state.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"call","name":"Bash"}})).unwrap();
            assert!(
                state
                    .accept(
                        &json!({"type":"response.output_item.done","output_index":0,"item":changed})
                    )
                    .is_err()
            );
        }
    }
}
