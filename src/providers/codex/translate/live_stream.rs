use super::IncompleteResponsePolicy;
use super::event::TranslationEvent;
use super::finish::FinishMetadata;
use super::reducer::map_codex_usage_to_anthropic;
#[cfg(test)]
use super::state::LiveBlock;
use super::state::ResponseState;
pub use super::state::TranslationError as LiveStreamError;
use crate::anthropic::sse::encode_sse_event;
use crate::monitor::{TokenObservation, UsageObservation};
use crate::traffic::TrafficCapture;

pub struct LiveStreamTranslator {
    state: ResponseState,
    message_id: String,
    model: String,
    estimated_input_tokens: u64,
    message_started: bool,
    progress: EmissionProgress,
    capture_staged: bool,
}

impl LiveStreamTranslator {
    pub fn new(message_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_estimated_input_tokens(message_id, model, 0)
    }
    pub fn with_estimated_input_tokens(
        message_id: impl Into<String>,
        model: impl Into<String>,
        estimated_input_tokens: u64,
    ) -> Self {
        Self {
            state: ResponseState::new(),
            message_id: message_id.into(),
            model: model.into(),
            estimated_input_tokens,
            message_started: false,
            progress: EmissionProgress::default(),
            capture_staged: false,
        }
    }
    pub(crate) fn with_early_reasoning(mut self) -> Self {
        self.state = self.state.with_early_reasoning();
        self
    }
    pub(crate) fn with_replay_budget(mut self, budget: Option<usize>) -> Self {
        self.state = self.state.with_replay_budget(budget);
        self
    }
    pub(crate) fn with_incomplete_response_policy(
        mut self,
        policy: IncompleteResponsePolicy,
    ) -> Self {
        self.state = self.state.with_incomplete_response_policy(policy);
        self
    }
    pub fn stage_capture(&mut self, enabled: bool) {
        self.capture_staged = enabled;
    }
    pub fn take_progress(&mut self) -> EmissionProgress {
        std::mem::take(&mut self.progress)
    }
    pub(crate) fn set_request_id(&mut self, req_id: &str) {
        self.state.set_request_id(req_id);
    }
    pub(crate) fn had_compatibility_skips(&self) -> bool {
        self.state.had_compatibility_skips()
    }
    pub fn finish_metadata(&self) -> Option<&FinishMetadata> {
        self.state.finish_metadata()
    }
    pub fn is_finished(&self) -> bool {
        self.state.is_finished()
    }
    pub fn has_semantic_output(&self) -> bool {
        self.state.has_semantic_output()
    }
    pub fn accept(
        &mut self,
        payload: &serde_json::Value,
        traffic: Option<&TrafficCapture>,
    ) -> Result<Vec<u8>, LiveStreamError> {
        let result = self.state.accept(payload);
        self.progress.append(EmissionProgress {
            usage: std::mem::take(&mut self.state.usage),
            ..Default::default()
        });
        match result {
            Ok(events) => Ok(self.render_events(&events, traffic)),
            Err(error) => {
                self.progress.failed = Some(error.as_str().to_string());
                Err(error)
            }
        }
    }
    pub fn ping_chunk(&mut self, traffic: Option<&TrafficCapture>) -> Vec<u8> {
        if self.is_finished() {
            return Vec::new();
        }
        self.render_events(
            &[TranslationEvent::Progress {
                output_index: None,
                item_id: None,
            }],
            traffic,
        )
    }
    pub fn error_chunk(
        &mut self,
        message: &str,
        error_type: &str,
        traffic: Option<&TrafficCapture>,
    ) -> Vec<u8> {
        if self.is_finished() {
            return Vec::new();
        }
        let events = self.state.abort();
        let mut out = self.render_events(&events, traffic);
        self.ensure_message_start(traffic, &mut out);
        self.emit(
            traffic,
            &mut out,
            "error",
            serde_json::json!({"type":"error","error":{"type":error_type,"message":message}}),
        );
        self.progress.failed = Some(message.to_string());
        out
    }
    fn ensure_message_start(&mut self, traffic: Option<&TrafficCapture>, out: &mut Vec<u8>) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        if self.progress.usage.input_tokens.is_none() {
            self.progress.usage.input_tokens =
                Some(TokenObservation::estimated(self.estimated_input_tokens));
        }
        self.emit(traffic, out, "message_start", serde_json::json!({"type":"message_start","message":{"id":self.message_id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":self.estimated_input_tokens,"output_tokens":0}}}));
    }
    fn emit(
        &mut self,
        traffic: Option<&TrafficCapture>,
        out: &mut Vec<u8>,
        event: &str,
        data: serde_json::Value,
    ) {
        self.progress.events += 1;
        self.progress.thinking_started |= event == "content_block_start"
            && data
                .pointer("/content_block/type")
                .and_then(serde_json::Value::as_str)
                == Some("thinking");
        if self.capture_staged && traffic.is_none() {
            self.progress
                .captured_events
                .push(serde_json::json!({"event":event,"data":data}));
        }
        if let Some(traffic) = traffic {
            traffic.write_json_event(
                "050-downstream-event",
                &serde_json::json!({"event":event,"data":data}),
            );
        }
        out.extend(encode_sse_event(Some(event), &data.to_string()));
    }
    pub(crate) fn render_events(
        &mut self,
        events: &[TranslationEvent],
        traffic: Option<&TrafficCapture>,
    ) -> Vec<u8> {
        use TranslationEvent::*;
        let mut out = Vec::new();
        for event in events {
            if matches!(event, TranslationEvent::Skipped(_)) {
                continue;
            }
            self.ensure_message_start(traffic, &mut out);
            match event {
                TextStart { index } => self.emit(traffic, &mut out, "content_block_start", serde_json::json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}})),
                ThinkingStart { index } => self.emit(traffic, &mut out, "content_block_start", serde_json::json!({"type":"content_block_start","index":index,"content_block":{"type":"thinking","thinking":"","signature":""}})),
                ToolStart { index, id, name } => self.emit(traffic, &mut out, "content_block_start", serde_json::json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}})),
                TextDelta { index, text } => self.emit(traffic, &mut out, "content_block_delta", serde_json::json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}})),
                ThinkingDelta { index, text } => self.emit(traffic, &mut out, "content_block_delta", serde_json::json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":text}})),
                ThinkingSignature { index, signature } => self.emit(traffic, &mut out, "content_block_delta", serde_json::json!({"type":"content_block_delta","index":index,"delta":{"type":"signature_delta","signature":signature}})),
                ToolDelta { index, partial_json } => self.emit(traffic, &mut out, "content_block_delta", serde_json::json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":partial_json}})),
                TextStop { index } | ThinkingStop { index } | ToolStop { index } => self.emit(traffic, &mut out, "content_block_stop", serde_json::json!({"type":"content_block_stop","index":index})),
                SearchComplete { index, card } => {
                    for (offset, block) in card.blocks().into_iter().enumerate() {
                        let index = index + offset;
                        self.emit(traffic, &mut out, "content_block_start", serde_json::json!({"type":"content_block_start","index":index,"content_block":block}));
                        self.emit(traffic, &mut out, "content_block_stop", serde_json::json!({"type":"content_block_stop","index":index}));
                    }
                }
                Progress { .. } => self.emit(traffic, &mut out, "ping", serde_json::json!({"type":"ping"})),
                Skipped(_) => {},
                Finish { stop_reason, usage, web_search_requests, .. } => {
                    self.emit(traffic, &mut out, "message_delta", serde_json::json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_sequence":null},"usage":map_codex_usage_to_anthropic(usage, Some(*web_search_requests))}));
                    self.emit(traffic, &mut out, "message_stop", serde_json::json!({"type":"message_stop"}));
                }
            }
        }
        out
    }
}

/// Metadata for exactly the bytes returned since the last drain. Structural
/// staging carries this alongside its bytes; discarded retry output is uncounted.
#[derive(Default)]
pub struct EmissionProgress {
    pub events: u64,
    pub usage: UsageObservation,
    pub failed: Option<String>,
    pub captured_events: Vec<serde_json::Value>,
    /// A real thinking block start is contained in this emission's bytes.
    pub thinking_started: bool,
}

impl EmissionProgress {
    pub fn append(&mut self, next: Self) {
        self.events += next.events;
        self.thinking_started |= next.thinking_started;
        if next.usage.input_tokens.is_some() {
            self.usage.input_tokens = next.usage.input_tokens;
        }
        if next.usage.cached_input_tokens.is_some() {
            self.usage.cached_input_tokens = next.usage.cached_input_tokens;
        }
        if next.usage.output_tokens.is_some() {
            self.usage.output_tokens = next.usage.output_tokens;
        }
        if next.usage.reasoning_tokens.is_some() {
            self.usage.reasoning_tokens = next.usage.reasoning_tokens;
        }
        self.failed = next.failed.or(self.failed.take());
        self.captured_events.extend(next.captured_events);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn task3_search_answer_streams_before_terminal_and_keeps_real_links() {
        use serde_json::json;
        let mut t = LiveStreamTranslator::new("m", "model");
        let mut events = vec![
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"s"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"s","action":{"query":"q","sources":[{"title":"Hosted source","url":"https://example.test/hosted"}]}}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m"}}),
        ];
        let mut all = Vec::new();
        for event in &events {
            all.extend(t.accept(event, None).unwrap());
        }
        let answer = format!(
            "{}[Existing](https://example.test/existing) [Longer source](https://example.test/source-long) [Article](https://en.wikipedia.org/wiki/Article_%28topic%29)",
            "Long answer ".repeat(1024)
        );
        let early = t
            .accept(
                &json!({"type":"response.output_text.delta","output_index":1,"delta":answer}),
                None,
            )
            .unwrap();
        assert!(
            String::from_utf8_lossy(&early).contains("Long answer"),
            "answer must cross the consumer barrier before terminal"
        );
        all.extend(early);
        events.push(json!({"type":"response.output_text.delta","output_index":1,"delta":answer}));
        let tail = [
            json!({"type":"response.output_text.annotation.added","output_index":1,"content_index":0,"annotation":{"type":"url_citation","title":"Source","url":"https://example.test/source"}}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"m","content":[{"type":"output_text","text":answer,"annotations":[{"type":"url_citation","title":"Source","url":"https://example.test/source"},{"type":"url_citation","title":"Existing","url":"https://example.test/existing"},{"type":"url_citation","title":"Snapshot source","url":"https://example.test/snapshot"},{"type":"url_citation","title":"Article","url":"https://en.wikipedia.org/wiki/Article_(topic)"}]}]}}),
            json!({"type":"response.completed","response":{"status":"completed"}}),
        ];
        for event in &tail {
            all.extend(t.accept(event, None).unwrap());
        }
        events.extend(tail);
        let raw = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        let buffered =
            super::super::accumulate::accumulate_response(raw.as_bytes(), "m", "model").unwrap();
        let content = buffered["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|block| block["text"].as_str())
            .collect::<String>();
        for url in [
            "https://example.test/source",
            "https://example.test/existing",
            "https://example.test/snapshot",
            "https://example.test/hosted",
        ] {
            assert_eq!(
                content.matches(&format!("({url})")).count(),
                1,
                "genuine sources must appear once: {url}"
            );
        }
        assert_eq!(
            content
                .matches("[Article](https://en.wikipedia.org/wiki/Article_%28topic%29)")
                .count(),
            1,
            "encoded source must appear once"
        );
        let all = String::from_utf8(all).unwrap();
        assert!(all.contains("[Source](https://example.test/source)"));
        assert!(all.contains("web_search_tool_result"));
        assert!(all.contains(r#""type":"server_tool_use""#));
        assert!(!t.finish_metadata().unwrap().continuation_eligible);
    }

    #[test]
    fn task3_unsupported_item_does_not_hide_unfinished_supported_call() {
        use serde_json::json;
        let mut t = LiveStreamTranslator::new("m", "model");
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"c","name":"Bash"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{}"}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"f","call_id":"c","name":"Bash","arguments":"{}"}}),
            json!({"type":"response.future.progress","sequence_number":9}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"mcp_call","id":"skip"}}),
            json!({"type":"response.mcp_call.failed","output_index":1,"item_id":"skip","error":{"message":"tool-local error"}}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"mcp_call","id":"skip"}}),
            json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"f2","call_id":"c2","name":"Bash"}}),
            json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"f2","call_id":"c2","name":"Bash","arguments":"{}"}}),
            json!({"type":"response.completed","response":{"status":"completed"}}),
        ] {
            t.accept(&event, None).unwrap();
        }
        assert!(t.is_finished());
        assert!(!t.finish_metadata().unwrap().continuation_eligible);
        let mut unfinished = LiveStreamTranslator::new("m", "model");
        unfinished.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"c","name":"Bash"}}),None).unwrap();
        unfinished
            .accept(&json!({"type":"response.future.progress"}), None)
            .unwrap();
        assert!(
            unfinished
                .accept(
                    &json!({"type":"response.completed","response":{"status":"completed"}}),
                    None
                )
                .is_err()
        );
    }

    #[test]
    fn interleaved_calls_and_message_parts_share_live_buffered_decisions() {
        use serde_json::json;
        let events = vec![
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call_a","name":"Bash"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"f1","call_id":"call_b","name":"mcp__fixture__lookup"}}),
            json!({"type":"response.function_call_arguments.delta","item_id":"f1","delta":"{\"q\":"}),
            json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"cmd\":"}),
            json!({"type":"response.function_call_arguments.done","output_index":1,"arguments":"{\"q\":\"x\"}"}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"f1","call_id":"call_b","name":"mcp__fixture__lookup","arguments":"{\"q\":\"x\"}"}}),
            json!({"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"cmd\":\"ls\"}"}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call_a","name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}),
            json!({"type":"response.output_item.added","output_index":2,"item":{"type":"message","id":"m"}}),
            json!({"type":"response.content_part.added","item_id":"m","content_index":0,"part":{"type":"output_text","text":""}}),
            json!({"type":"response.output_text.delta","item_id":"m","content_index":0,"delta":"First"}),
            json!({"type":"response.content_part.done","item_id":"m","content_index":0,"part":{"type":"output_text","text":"First part"}}),
            json!({"type":"response.content_part.added","item_id":"m","content_index":1,"part":{"type":"refusal","refusal":""}}),
            json!({"type":"response.refusal.delta","item_id":"m","content_index":1,"delta":"Cannot"}),
            json!({"type":"response.refusal.done","item_id":"m","content_index":1,"refusal":"Cannot comply"}),
            json!({"type":"response.output_item.done","output_index":2,"item":{"type":"message","id":"m","content":[{"type":"output_text","text":"First part"},{"type":"refusal","refusal":"Cannot comply"}]}}),
            json!({"type":"response.completed","response":{"id":"r","status":"completed"}}),
        ];
        let mut translator = LiveStreamTranslator::new("msg", "model");
        let mut live = Vec::new();
        for event in &events {
            let bytes = translator.accept(event, None).unwrap();
            if event["type"] != "response.completed" {
                assert!(!String::from_utf8_lossy(&bytes).contains("message_stop"));
            }
            live.extend(bytes);
        }
        let raw = events
            .iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect::<String>();
        let buffered =
            super::super::accumulate::accumulate_response(raw.as_bytes(), "msg", "model").unwrap();
        assert_eq!(buffered["content"][0]["id"], "call_a");
        assert_eq!(buffered["content"][0]["input"], json!({"cmd":"ls"}));
        assert_eq!(buffered["content"][1]["id"], "call_b");
        assert_eq!(buffered["content"][1]["input"], json!({"q":"x"}));
        assert_eq!(buffered["content"][2]["text"], "First part");
        assert_eq!(buffered["content"][3]["text"], "Cannot comply");
        assert_eq!(buffered["stop_reason"], "tool_use");
        let frames: Vec<serde_json::Value> = crate::anthropic::sse::parse_sse_events(&live)
            .into_iter()
            .map(|e| serde_json::from_str(&e.data).unwrap())
            .collect();
        for (index, expected) in [(0, "{\"cmd\":\"ls\"}"), (1, "{\"q\":\"x\"}")] {
            let args: String = frames
                .iter()
                .filter(|e| e["index"] == index)
                .filter_map(|e| {
                    e.pointer("/delta/partial_json")
                        .and_then(serde_json::Value::as_str)
                })
                .collect();
            assert_eq!(args, expected);
        }
    }

    #[test]
    fn cached_input_is_reported_by_terminal_progress_and_preserved_when_staged() {
        use super::*;
        let mut translator = LiveStreamTranslator::new("msg", "model");
        translator
            .accept(
                &serde_json::json!({
                    "type": "response.completed", "response": {"status": "completed", "usage": {
                        "input_tokens": 790845, "input_tokens_details": {"cached_tokens": 782848}
                    }}
                }),
                None,
            )
            .unwrap();
        let mut staged = EmissionProgress::default();
        staged.append(translator.take_progress());
        staged.append(EmissionProgress::default());
        let value = serde_json::to_value(&staged.usage).unwrap();
        assert_eq!(value["cached_input_tokens"], 782848);
        let zero = serde_json::from_value(serde_json::json!({"cached_input_tokens": 0})).unwrap();
        staged.append(EmissionProgress {
            usage: zero,
            ..Default::default()
        });
        staged.append(EmissionProgress::default());
        assert_eq!(
            serde_json::to_value(&staged.usage).unwrap()["cached_input_tokens"],
            0
        );
    }

    #[test]
    fn emission_progress_counts_generated_events_and_retains_failed_terminal_usage() {
        use super::*;
        let mut translator = LiveStreamTranslator::with_estimated_input_tokens("msg", "model", 321);
        translator.stage_capture(true);
        assert!(!translator.ping_chunk(None).is_empty());
        let initial = translator.take_progress();
        assert_eq!(initial.events, 2);
        assert_eq!(initial.captured_events.len(), 2);
        assert_eq!(
            initial.usage.input_tokens,
            Some(TokenObservation::estimated(321))
        );
        assert_eq!(initial.usage.output_tokens, None);
        assert!(!translator.ping_chunk(None).is_empty());
        let heartbeat = translator.take_progress();
        assert_eq!(heartbeat.events, 1);
        assert_eq!(heartbeat.captured_events.len(), 1);
        assert_eq!(heartbeat.usage, UsageObservation::default());
        translator.accept(&serde_json::json!({"type":"response.output_item.added", "output_index":0,"item":{"type":"message","id":"i"}}), None).unwrap();
        translator.take_progress();
        let result = translator.accept(&serde_json::json!({"type":"response.completed","response":{"status":"completed","usage":{"output_tokens":300,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":20}}}}), None);
        assert!(result.is_err());
        let error = translator.error_chunk("local rejection", "api_error", None);
        assert!(!error.is_empty());
        let progress = translator.take_progress();
        assert_eq!(progress.usage.input_tokens, None);
        assert_eq!(progress.usage.cached_input_tokens, Some(0));
        assert_eq!(
            progress.usage.output_tokens,
            Some(TokenObservation::provider(300))
        );
        assert_eq!(progress.usage.reasoning_tokens, Some(20));
        assert!(progress.failed.is_some());
        assert_eq!(progress.events, progress.captured_events.len() as u64);
        assert_eq!(progress.captured_events.last().unwrap()["event"], "error");
        assert!(
            translator
                .error_chunk("duplicate", "api_error", None)
                .is_empty()
        );
        assert_eq!(translator.take_progress().events, 0);
    }

    #[test]
    fn terminal_emission_preserves_provider_total_including_reasoning_on_wire() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        let bytes = translator.accept(&serde_json::json!({"type":"response.completed", "response":{"status":"completed","usage":{"output_tokens":300,"output_tokens_details":{"reasoning_tokens":20}}}}), None).unwrap();
        let progress = translator.take_progress();
        assert_eq!(
            progress.usage.output_tokens,
            Some(TokenObservation::provider(300))
        );
        assert_eq!(progress.usage.reasoning_tokens, Some(20));
        assert_eq!(progress.events, 3);
        let delta = parse_sse_events(&bytes)
            .into_iter()
            .find(|event| event.event.as_deref() == Some("message_delta"))
            .unwrap();
        let data: serde_json::Value = serde_json::from_str(&delta.data).unwrap();
        assert_eq!(data["usage"]["output_tokens"], 300);
        assert!(translator.ping_chunk(None).is_empty());
        assert_eq!(translator.take_progress().events, 0);
    }
    use super::*;
    use crate::anthropic::sse::parse_sse_events;
    use serde_json::json;

    #[test]
    fn ordinary_live_deltas_do_not_retain_forwarded_answer() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        let delta = json!({"type":"response.output_text.delta","output_index":0,"delta":"abcdefghijklmnop"});
        for count in [100_000, 200_000] {
            for _ in 0..100_000 {
                let bytes = translator.accept(&delta, None).unwrap();
                assert!(!bytes.is_empty());
            }
            let retained: usize = translator
                .state
                .blocks_by_output_index
                .values()
                .map(|block| match block {
                    LiveBlock::Text { text, .. } => text.capacity(),
                    LiveBlock::Tool { args_accum, .. } => args_accum.capacity(),
                })
                .sum();
            assert!(
                retained < 1024,
                "retained {retained} bytes after {count} deltas"
            );
        }
    }

    #[test]
    fn canonical_live_replay_is_ordered_and_snapshots_do_not_duplicate() {
        let mut translator =
            LiveStreamTranslator::new("msg", "model").with_replay_budget(Some(4096));
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m"}}),
            json!({"type":"response.output_text.delta","output_index":1,"delta":"hello"}),
            json!({"type":"response.output_text.done","output_index":1,"text":"hello world"}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"hello world"}]}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs","encrypted_content":"complete:opaque"}}),
            json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","name":"Bash","call_id":"call"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"cmd\":"}),
            json!({"type":"response.function_call_arguments.done","output_index":2,"arguments":"{\"cmd\":\"ls\"}"}),
            json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","arguments":"{\"cmd\":\"ls\"}"}}),
            json!({"type":"response.completed","response":{"id":"resp","usage":{"input_tokens":12,"output_tokens":3}}}),
        ] {
            translator.accept(&event, None).unwrap();
        }
        let finish = translator.finish_metadata().unwrap();
        assert!(finish.stream_success && finish.continuation_eligible);
        assert_eq!(finish.response_id.as_deref(), Some("resp"));
        assert_eq!(finish.usage.as_ref().unwrap().output_tokens, Some(3));
        assert_eq!(
            serde_json::to_value(&finish.output_items).unwrap(),
            json!([
                {"type":"reasoning","id":"rs","summary":[],"encrypted_content":"complete:opaque"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello world"}]},
                {"type":"function_call","call_id":"call","name":"Bash","arguments":"{\"cmd\":\"ls\"}"}
            ])
        );
    }

    #[test]
    fn replay_overflow_drops_optional_answer_but_preserves_stream_success() {
        let mut translator =
            LiveStreamTranslator::new("msg", "model").with_replay_budget(Some(300));
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}),
            json!({"type":"response.output_text.delta","output_index":0,"delta":"a".repeat(1024)}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}),
            json!({"type":"response.completed","response":{"id":"r"}}),
        ] {
            translator.accept(&event, None).unwrap();
        }
        let finish = translator.finish_metadata().unwrap();
        assert!(finish.stream_success);
        assert!(!finish.replay_available && !finish.continuation_eligible);
        assert!(finish.output_items.is_empty());
    }

    #[test]
    fn unresolved_or_unsupported_items_never_publish_replay() {
        for item in [json!({"type":"message"}), json!({"type":"unknown"})] {
            let mut translator =
                LiveStreamTranslator::new("msg", "model").with_replay_budget(Some(4096));
            translator
                .accept(
                    &json!({"type":"response.output_item.added","output_index":0,"item":item}),
                    None,
                )
                .unwrap();
            let terminal = translator.accept(
                &json!({"type":"response.completed","response":{"id":"r"}}),
                None,
            );
            if item["type"] == "message" {
                assert!(terminal.is_err());
                assert!(translator.finish_metadata().is_none());
                assert!(!translator.is_finished());
            } else {
                terminal.unwrap();
                let finish = translator.finish_metadata().unwrap();
                assert!(finish.stream_success);
                assert!(!finish.continuation_eligible && !finish.replay_available);
            }
        }
    }

    #[test]
    fn unsupported_message_content_preserves_success_without_native_replay() {
        let mut translator =
            LiveStreamTranslator::new("msg", "model").with_replay_budget(Some(4096));
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}),
            json!({"type":"response.output_text.delta","output_index":0,"delta":"visible"}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_audio","audio":"unsupported"}]}}),
        ] {
            translator.accept(&event, None).unwrap();
        }
        translator
            .accept(
                &json!({"type":"response.completed","response":{"id":"r"}}),
                None,
            )
            .unwrap();
        let finish = translator.finish_metadata().unwrap();
        assert!(finish.stream_success);
        assert!(!finish.continuation_eligible && !finish.replay_available);
    }

    #[test]
    fn conflicting_snapshots_fail_without_replay() {
        let mut translator =
            LiveStreamTranslator::new("msg", "model").with_replay_budget(Some(4096));
        translator
            .accept(
                &json!({"type":"response.output_text.delta","output_index":0,"delta":"hello"}),
                None,
            )
            .unwrap();
        assert!(
            translator
                .accept(
                    &json!({"type":"response.output_text.done","output_index":0,"text":"world"}),
                    None
                )
                .is_err()
        );
        assert!(translator.finish_metadata().is_none());
    }

    #[test]
    fn read_done_snapshot_preserves_buffer_until_item_done() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        translator.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"Read","call_id":"c"}}), None).unwrap();
        let bytes = translator.accept(&json!({"type":"response.function_call_arguments.done","output_index":0,"arguments":format!("{{\"file_path\":\"/tmp/a\"}}{}", " ".repeat(1024))}), None).unwrap();
        assert!(
            !translator.is_finished(),
            "a done snapshot must not take the delta-only repair exit"
        );
        assert!(
            bytes.is_empty(),
            "Read arguments remain buffered until output_item.done"
        );
    }

    #[test]
    fn ordinary_non_read_arguments_are_not_retained() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        translator.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"Bash","call_id":"c"}}), None).unwrap();
        let delta = json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"x".repeat(1024)});
        for _ in 0..10_000 {
            translator.accept(&delta, None).unwrap();
        }
        let LiveBlock::Tool { args_accum, .. } = &translator.state.blocks_by_output_index[&(0, 0)]
        else {
            panic!()
        };
        assert_eq!(args_accum.capacity(), 0);
    }

    #[test]
    fn live_done_snapshot_emits_only_missing_suffix() {
        let rendered = render(vec![
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}),
            json!({"type":"response.output_text.delta","output_index":0,"delta":"hello"}),
            json!({"type":"response.output_text.done","output_index":0,"text":"hello world"}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_text","text":"hello world"}]}}),
            json!({"type":"response.completed","response":{"id":"r","status":"completed"}}),
        ]);
        let text: String = parse_sse_events(rendered.as_bytes())
            .into_iter()
            .filter_map(|event| {
                let value: serde_json::Value = serde_json::from_str(&event.data).ok()?;
                value
                    .pointer("/delta/text")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(text, "hello world");
    }

    fn render(events: Vec<serde_json::Value>) -> String {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let mut out = Vec::new();
        for event in events {
            out.extend(translator.accept(&event, None).unwrap());
        }
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn read_whitespace_does_not_bypass_missing_completed_reasoning() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        let prefix = format!("{{\"file_path\":\"/tmp/a\"{}", " ".repeat(1024));
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs","encrypted_content":"early"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c","name":"Read"}}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":prefix}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","arguments":format!("{prefix}}}")}}),
        ] {
            translator.accept(&event, None).unwrap();
            assert!(!translator.is_finished());
        }
        let error = translator
            .accept(
                &json!({"type":"response.completed","response":{"status":"completed"}}),
                None,
            )
            .unwrap_err();
        assert!(
            error
                .as_str()
                .contains("completed reasoning replay unavailable")
        );
    }

    #[test]
    fn orphan_function_done_is_explicit_failure() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-6.1-sol");
        let error = translator.accept(&json!({"type":"response.output_item.done","output_index":1,
            "item":{"type":"function_call","call_id":"call_orphan","name":"Read","arguments":"{}"}}),None).unwrap_err();
        assert!(error.as_str().contains("unsupported Codex sequence"));
        assert!(!translator.has_semantic_output());
    }

    #[test]
    fn missing_completed_reasoning_prevents_tool_success() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-6.1-sol");
        for event in [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"early"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":""}}),
            json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{}"}}),
        ] {
            let chunk = String::from_utf8(translator.accept(&event, None).unwrap()).unwrap();
            assert!(!chunk.contains("signature_delta"));
        }
        let error = translator
            .accept(
                &json!({"type":"response.completed","response":{"status":"completed"}}),
                None,
            )
            .unwrap_err();
        assert!(
            error
                .as_str()
                .contains("completed reasoning replay unavailable")
        );
    }

    #[test]
    fn emits_text_delta_before_terminal_event() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let out = translator
            .accept(
                &json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "hello"
                }),
                None,
            )
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("message_start"));
        assert!(out.contains("content_block_start"));
        assert!(out.contains("text_delta"));
        assert!(out.contains("hello"));
        assert!(!out.contains("message_stop"));
        assert!(translator.has_semantic_output());
    }

    #[test]
    fn estimated_input_is_visible_at_start_and_provider_usage_is_exact_at_finish() {
        let mut translator =
            LiveStreamTranslator::with_estimated_input_tokens("msg_1", "gpt-5.5", 321);

        let started = translator
            .accept(
                &json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "abcdefgh"
                }),
                None,
            )
            .unwrap();
        let started = parse_sse_events(&started)
            .into_iter()
            .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.data).ok())
            .find(|value| {
                value.get("type").and_then(serde_json::Value::as_str) == Some("message_start")
            })
            .unwrap();
        assert_eq!(
            started.pointer("/message/usage/input_tokens"),
            Some(&json!(321))
        );

        translator.accept(&json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "content": [{"type": "output_text", "text": "abcdefgh"}]}
        }), None).unwrap();

        let finished = translator
            .accept(
                &json!({
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "usage": {"input_tokens": 300, "output_tokens": 9}
                    }
                }),
                None,
            )
            .unwrap();
        let finished = parse_sse_events(&finished)
            .into_iter()
            .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.data).ok())
            .find(|value| {
                value.get("type").and_then(serde_json::Value::as_str) == Some("message_delta")
            })
            .unwrap();
        assert_eq!(finished.pointer("/usage/input_tokens"), Some(&json!(300)));
        assert_eq!(finished.pointer("/usage/output_tokens"), Some(&json!(9)));
    }

    #[test]
    fn finishes_text_stream() {
        let out = render(vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_up"}
            }),
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "hello"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "incomplete_details": null, "usage": {"input_tokens": 2, "output_tokens": 1}}
            }),
        ]);
        assert!(out.contains("content_block_stop"));
        assert!(out.contains("message_delta"));
        assert!(out.contains(r#""stop_reason":"end_turn""#));
        assert!(out.contains("message_stop"));
    }

    #[test]
    fn terminal_only_completion_remains_non_semantic() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let out = translator
            .accept(
                &json!({
                    "type": "response.completed",
                    "response": {"id": "resp_1", "status": "completed", "incomplete_details": null, "usage": {}}
                }),
                None,
            )
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains(r#""stop_reason":"end_turn""#));
        assert!(!out.contains(r#""stop_reason":"max_tokens""#));
        assert!(!translator.has_semantic_output());
    }

    #[test]
    fn strict_live_translator_rejects_incomplete_response() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let error = translator
            .accept(
                &json!({
                    "type":"response.incomplete",
                    "response": {
                        "status":"incomplete",
                        "incomplete_details":{"reason":"max_output_tokens"}
                    }
                }),
                None,
            )
            .unwrap_err();
        assert!(error.as_str().contains("max_output_tokens"));
    }

    #[test]
    fn standard_responses_policy_maps_max_output_tokens_to_message_stop() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.6-luna")
            .with_incomplete_response_policy(IncompleteResponsePolicy::AllowMaxOutputTokens)
            .with_replay_budget(Some(4096));
        let out = translator
            .accept(
                &json!({
                    "type":"response.incomplete",
                    "response": {
                        "status":"incomplete",
                        "error":null,
                        "incomplete_details":{"reason":"max_output_tokens"},
                        "usage":{"input_tokens":2,"output_tokens":8}
                    }
                }),
                None,
            )
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains(r#""stop_reason":"max_tokens""#));
        assert!(out.contains(r#""output_tokens":8"#));
        assert!(out.contains("event: message_stop"));
        assert!(translator.is_finished());
        let metadata = translator.finish_metadata().unwrap();
        assert!(!metadata.stream_success && !metadata.continuation_eligible);
    }

    #[test]
    fn standard_responses_policy_rejects_other_incomplete_reasons() {
        for reason in ["content_filter", "unknown"] {
            let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.6-luna")
                .with_incomplete_response_policy(IncompleteResponsePolicy::AllowMaxOutputTokens);
            assert!(
                translator
                    .accept(
                        &json!({
                            "type":"response.incomplete",
                            "response": {
                                "status":"incomplete",
                                "incomplete_details":{"reason":reason}
                            }
                        }),
                        None,
                    )
                    .is_err(),
                "{reason}"
            );
        }
    }

    #[test]
    fn visible_tool_start_commits_before_arguments_arrive() {
        let mut tool = LiveStreamTranslator::new("msg_tool", "gpt-5.5");
        tool.accept(
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read"}
            }),
            None,
        )
        .unwrap();
        assert!(tool.has_semantic_output());
        tool.accept(
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{}"
            }),
            None,
        )
        .unwrap();
        assert!(tool.has_semantic_output());

        let mut thinking = LiveStreamTranslator::new("msg_thinking", "gpt-5.5");
        thinking
            .accept(
                &json!({
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "delta": "plan"
                }),
                None,
            )
            .unwrap();
        assert!(thinking.has_semantic_output());

        let mut web_search = LiveStreamTranslator::new("msg_search", "gpt-5.5");
        web_search
            .accept(
                &json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "web_search_call", "id": "ws_1"}
                }),
                None,
            )
            .unwrap();
        assert!(!web_search.has_semantic_output());
        web_search
            .accept(
                &json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "action": {"query": "claude-code-proxy"}
                    }
                }),
                None,
            )
            .unwrap();
        assert!(!web_search.has_semantic_output());
        assert!(web_search.had_compatibility_skips());
    }

    #[test]
    fn buffers_read_tool_args_until_done() {
        let out = render(vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read"}
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{\"file_path\":\"/tmp/a\",\"pages\":\"\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"/tmp/a\",\"pages\":\"\"}"
                }
            }),
            json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(out.contains("tool_use"));
        assert!(out.contains("input_json_delta"));
        assert!(out.contains("/tmp/a"));
        assert!(!out.contains("pages"));
    }

    #[test]
    fn read_whitespace_waits_for_authoritative_completion() {
        let mut translator = LiveStreamTranslator::new("msg", "model");
        translator.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"Read"}}), None).unwrap();
        let prefix = format!(
            "{{\"file_path\":\"/tmp/a\",\"pages\":\"\"{}",
            " ".repeat(1024)
        );
        let bytes = translator.accept(&json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":prefix}), None).unwrap();
        assert!(bytes.is_empty(), "unfinished Read must remain buffered");
        assert!(!translator.is_finished());
        let bytes = translator.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","arguments":format!("{prefix}}}")}}), None).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        assert!(rendered.contains(r#""partial_json":"{\"file_path\":\"/tmp/a\"}""#));
        assert!(!rendered.contains("message_stop"));
        assert!(!translator.is_finished());
        let ending = translator.error_chunk("Missing response terminal", "api_error", None);
        assert!(String::from_utf8(ending).unwrap().contains("event: error"));
        assert!(
            translator.finish_metadata().is_none(),
            "EOF compatibility cannot publish continuation"
        );
    }

    #[test]
    fn closed_tool_requires_response_terminal() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let mut out = Vec::new();
        for event in [
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type":"function_call","call_id":"call_1","name":"WebSearch"}
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{\"query\":\"claude-code-proxy github\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type":"function_call",
                    "call_id":"call_1",
                    "name":"WebSearch",
                    "arguments":"{\"query\":\"claude-code-proxy github\"}"
                }
            }),
        ] {
            out.extend(translator.accept(&event, None).unwrap());
        }
        out.extend(translator.error_chunk("Missing response terminal", "api_error", None));
        let rendered = String::from_utf8(out).unwrap();
        assert!(rendered.contains("content_block_start"));
        assert!(rendered.contains("input_json_delta"));
        assert!(!rendered.contains("message_stop"));
        assert!(rendered.contains("event: error"));
    }

    #[test]
    fn emits_real_citation_links_and_completed_claudecode_search_pair() {
        let out = render(vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "web_search_call", "id": "ws_1"}
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "action": {"query": "grok reasoning effort"}
                }
            }),
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "message", "id": "msg_up"}
            }),
            json!({
                "type": "response.output_text.delta",
                "output_index": 1,
                "delta": "See the official docs."
            }),
            json!({
                "type": "response.output_text.annotation.added",
                "output_index": 1,
                "annotation": {
                    "type": "url_citation",
                    "title": "Reasoning",
                    "url": "https://docs.x.ai/docs/guides/reasoning"
                }
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {"type": "message"}
            }),
            json!({
                "type": "response.completed",
                "response": {"status": "completed", "usage": {}}
            }),
        ]);

        assert!(out.contains(r#""type":"server_tool_use""#));
        assert!(out.contains("web_search_tool_result"));
        let text = out.find("See the official docs.").unwrap();
        let link = out
            .find("[Reasoning](https://docs.x.ai/docs/guides/reasoning)")
            .unwrap();
        assert!(text < link);
        assert!(out.contains(r#""web_search_requests":1"#));
    }

    #[test]
    fn rate_limit_event_is_progress_telemetry() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let out = translator
            .accept(
                &json!({
                    "type": "codex.rate_limits",
                    "rate_limits": {"limit_reached": true}
                }),
                None,
            )
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("event: ping"));
        assert!(!out.contains("event: error"));

        let finish = translator
            .accept(
                &json!({
                    "type": "response.completed",
                    "response": {"id": "resp_1", "status": "completed", "usage": {"input_tokens": 5, "output_tokens": 2}}
                }),
                None,
            )
            .unwrap();
        let finish = String::from_utf8(finish).unwrap();
        assert!(
            finish.contains(r#""codex_rate_limits":{"limit_reached":true}"#),
            "{finish}"
        );
    }

    #[test]
    fn progress_events_start_message_and_emit_pings() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let first = String::from_utf8(
            translator
                .accept(&json!({"type": "response.created"}), None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(first.matches("event: message_start").count(), 1);
        assert_eq!(first.matches("event: ping").count(), 1);

        let second = String::from_utf8(translator.ping_chunk(None)).unwrap();
        assert!(!second.contains("event: message_start"));
        assert_eq!(second.matches("event: ping").count(), 1);
    }

    #[test]
    fn live_stream_emits_signature_delta_before_thinking_stop() {
        let out = render(vec![
            json!({
                "type":"response.output_item.added",
                "output_index":0,
                "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
            }),
            json!({
                "type":"response.reasoning_summary_text.delta",
                "output_index":0,
                "delta":"plan"
            }),
            json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
            }),
            json!({
                "type":"response.completed",
                "response":{"id":"resp_1","usage":{}}
            }),
        ]);
        let thinking_delta = out.find(r#""type":"thinking_delta""#).unwrap();
        let signature_delta = out.find(r#""type":"signature_delta""#).unwrap();
        let thinking_stop = out[signature_delta..]
            .find("event: content_block_stop")
            .map(|offset| signature_delta + offset)
            .unwrap();
        assert!(thinking_delta < signature_delta);
        assert!(signature_delta < thinking_stop);
        assert!(out.contains("ccp:codex:v1:"));
    }

    #[test]
    fn early_reasoning_reuses_one_block_before_rejecting_thinking_only() {
        use crate::anthropic::sse::parse_sse_events;
        for summary in [false, true] {
            let added = json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r"}});
            let mut default = LiveStreamTranslator::new("m", "model");
            assert!(default.accept(&added, None).unwrap().is_empty());
            assert!(!default.has_semantic_output());
            let mut early = LiveStreamTranslator::new("m", "model").with_early_reasoning();
            let mut wire = early.accept(&added, None).unwrap();
            assert!(
                early.has_semantic_output(),
                "published reasoning start must commit output"
            );
            let initial = parse_sse_events(&wire);
            assert_eq!(
                initial
                    .iter()
                    .filter(|e| e.event.as_deref() == Some("content_block_start"))
                    .count(),
                1
            );
            assert!(!String::from_utf8_lossy(&wire).contains("thinking_delta"));
            if summary {
                wire.extend(early.accept(&json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"Real summary"}),None).unwrap());
            }
            wire.extend(early.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"r","encrypted_content":"opaque"}}),None).unwrap());
            let error = early.accept(&json!({"type":"response.completed","response":{"id":"resp","status":"completed"}}),None).expect_err("thinking alone cannot finish successfully");
            assert!(early.finish_metadata().is_none());
            wire.extend(early.error_chunk(error.as_str(), "api_error", None));
            let events = parse_sse_events(&wire);
            for (name, count) in [
                ("message_start", 1),
                ("content_block_start", 1),
                ("content_block_stop", 1),
                ("message_stop", 0),
                ("error", 1),
            ] {
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e.event.as_deref() == Some(name))
                        .count(),
                    count,
                    "{name}"
                );
            }
            let deltas: Vec<serde_json::Value> = events
                .iter()
                .filter(|e| e.event.as_deref() == Some("content_block_delta"))
                .map(|e| serde_json::from_str(&e.data).unwrap())
                .collect();
            assert!(deltas.iter().all(|e| e["index"] == 0));
            assert_eq!(
                deltas
                    .iter()
                    .filter(|e| e["delta"]["type"] == "signature_delta")
                    .count(),
                1
            );
            assert!(String::from_utf8_lossy(&wire).contains("ccp:codex:v1:"));
        }
    }

    #[test]
    fn early_reasoning_summary_without_answer_cannot_finish_successfully() {
        for summary in [false, true] {
            let mut t = LiveStreamTranslator::new("m", "model").with_early_reasoning();
            let mut wire = t.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r"}}), None).unwrap();
            if summary {
                wire.extend(t.accept(&json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Real summary"}), None).unwrap());
            }
            wire.extend(t.accept(&json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"r","summary":[]}}), None).unwrap());
            let terminal = t.accept(
                &json!({"type":"response.completed","response":{"id":"resp","status":"completed"}}),
                None,
            );
            let error = terminal.expect_err("thinking alone is not an answer");
            assert!(
                t.finish_metadata().is_none(),
                "thinking cannot authorize success"
            );
            assert!(
                t.has_semantic_output(),
                "failed completion must not reopen retry"
            );
            wire.extend(t.error_chunk(error.as_str(), "api_error", None));
            let wire = String::from_utf8(wire).unwrap();
            assert_eq!(wire.matches("event: content_block_start").count(), 1);
            assert_eq!(wire.matches("event: content_block_stop").count(), 1);
            assert!(!wire.contains("signature_delta"));
            assert!(wire.contains("event: error"));
            assert!(!wire.contains("message_stop"));
        }
    }

    #[test]
    fn early_reasoning_abort_closes_empty_block_without_success_or_fake_signature() {
        let mut t = LiveStreamTranslator::new("m", "model").with_early_reasoning();
        let mut wire=t.accept(&json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r"}}),None).unwrap();
        wire.extend(t.error_chunk("upstream failed", "api_error", None));
        let wire = String::from_utf8(wire).unwrap();
        assert_eq!(wire.matches("event: content_block_stop").count(), 1);
        assert!(wire.contains("event: error"));
        assert!(!wire.contains("signature_delta"));
        assert!(!wire.contains("message_stop"));
    }

    #[test]
    fn signature_only_reasoning_is_semantic_output() {
        let mut translator = LiveStreamTranslator::new("msg_1", "gpt-5.5");
        let mut out = Vec::new();
        for event in [
            json!({
                "type":"response.output_item.added",
                "output_index":0,
                "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
            }),
            json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
            }),
        ] {
            out.extend(translator.accept(&event, None).unwrap());
        }

        let out = String::from_utf8(out).unwrap();
        assert!(out.contains(r#""type":"signature_delta""#));
        assert!(translator.has_semantic_output());
    }
}
