use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::Ordering};

use crate::config::AliasProvider;
use crate::logging::REDACT_KEYS;
use crate::logging::writer::{self, CaptureProgress, Record};
use crate::paths;
use crate::providers::codex::translate::reasoning_signature::is_proxy_reasoning_signature;

#[derive(Debug)]
pub struct TrafficCapture {
    root: PathBuf,
    sequence: Mutex<u64>,
    progress: Arc<CaptureProgress>,
}

pub const MAX_SSE_CAPTURE_BYTES: usize = 1024 * 1024;
pub const MAX_STREAM_CAPTURE_EVENT_BYTES: usize = 1024 * 1024;
pub const MAX_STREAM_CAPTURE_EVENTS: usize = 1_024;
pub const MAX_STREAM_CAPTURE_FRAME_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub struct TrafficCaptureOptions {
    pub req_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub provider: Option<String>,
    pub state_dir_override: Option<PathBuf>,
}

pub fn traffic_capture_enabled() -> bool {
    if let Some(enabled) = crate::config::captured_traffic_log() {
        return enabled;
    }
    traffic_capture_enabled_for_env(&std::env::vars().collect())
}

pub fn traffic_capture_enabled_for_env(env: &std::collections::HashMap<String, String>) -> bool {
    match env.get("CCP_TRAFFIC_LOG").map(String::as_str) {
        Some(v) => matches!(v, "1" | "true" | "yes"),
        None => false,
    }
}

pub fn create_traffic_capture(opts: TrafficCaptureOptions) -> Option<TrafficCapture> {
    if !traffic_capture_enabled() {
        return None;
    }
    let state_root = opts
        .state_dir_override
        .unwrap_or_else(paths::state_dir)
        .join("traffic")
        .join(sanitize_path_part(
            opts.session_id.as_deref().unwrap_or("no-session"),
        ))
        .join(format!(
            "{:06}-{}-{}",
            opts.session_seq.unwrap_or(0),
            sanitize_path_part(opts.provider.as_deref().unwrap_or("unknown-provider")),
            sanitize_path_part(&opts.req_id),
        ));

    Some(TrafficCapture {
        root: state_root,
        sequence: Mutex::new(0),
        progress: Arc::new(CaptureProgress::default()),
    })
}

impl TrafficCapture {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn write_json(&self, name: &str, value: &Value) {
        self.json_record(name, value, false);
    }

    pub fn write_text(&self, name: &str, text: &str) {
        let name = if name.ends_with(".txt") {
            name.to_owned()
        } else {
            format!("{name}.txt")
        };
        self.write_bytes(&name, text.as_bytes());
    }

    pub fn write_bytes(&self, name: &str, value: &[u8]) {
        if value.len() > writer::MAX_RECORD_BYTES - 4096 {
            self.omit();
            return;
        }
        // Raw capture is never a trusted escape hatch. JSON and SSE data must
        // cross the same recursive redaction boundary as structured captures.
        if let Ok(value) = serde_json::from_slice::<Value>(value) {
            self.json_record(name, &value, false);
            return;
        }
        if let Some(payload) = redacted_sse(value) {
            self.enqueue(name, payload, false, false);
        } else {
            self.omit();
        }
    }

    pub fn write_json_event(&self, name: &str, value: &Value) {
        self.json_record(name, value, true);
    }

    pub fn stream_capture(&self) -> StreamTrafficCapture {
        StreamTrafficCapture::default()
    }

    pub(crate) fn omit(&self) {
        writer::global().omit(Some(&self.progress));
    }

    fn json_record(&self, name: &str, value: &Value, event: bool) {
        if crate::bounded_json_size(value, writer::MAX_RECORD_BYTES - 4096)
            .ok()
            .flatten()
            .is_none()
        {
            self.omit();
            return;
        }
        let Some(payload) = writer::bounded_json(&redact_traffic(value), true) else {
            self.omit();
            return;
        };
        let file = if name.ends_with(".json") {
            name.to_owned()
        } else {
            format!("{name}.json")
        };
        self.enqueue(&file, payload, event, false);
    }

    fn enqueue(&self, name: &str, payload: Vec<u8>, event: bool, completion: bool) {
        // This lock covers ordinal assignment + try-enqueue only. It never
        // waits for the writer; concurrent producers drop on contention.
        let Ok(mut ordinal) = self.sequence.try_lock() else {
            self.omit();
            return;
        };
        *ordinal += 1;
        let dir = if event {
            self.root.join("events")
        } else {
            self.root.clone()
        };
        let path = dir.join(format!("{:06}-{}", *ordinal, sanitize_path_part(name)));
        writer::global().enqueue(Record::capture(
            path,
            payload,
            self.progress.clone(),
            completion,
        ));
    }
}

impl Drop for TrafficCapture {
    fn drop(&mut self) {
        let marker = serde_json::json!({
            "kind":"diagnostic_capture_completion", "complete":false,
            "accepted_records":self.progress.accepted.load(Ordering::Relaxed),
            "dropped":self.progress.dropped.load(Ordering::Relaxed),
            "failed":self.progress.failed.load(Ordering::Relaxed),
        });
        self.enqueue(
            "capture-completion.json",
            writer::bounded_json(&marker, false).unwrap(),
            false,
            true,
        );
    }
}

#[cfg(test)]
pub(crate) fn test_capture(root: PathBuf) -> TrafficCapture {
    TrafficCapture {
        root,
        sequence: Mutex::new(0),
        progress: Arc::new(CaptureProgress::default()),
    }
}

fn redacted_sse(raw: &[u8]) -> Option<Vec<u8>> {
    std::str::from_utf8(raw).ok()?;
    let events = crate::anthropic::sse::parse_sse_events(raw);
    if events.is_empty() {
        return None;
    }
    let mut output = Vec::new();
    for event in events {
        let data = if event.data == "[DONE]" {
            event.data
        } else {
            let value: Value = serde_json::from_str(&event.data).ok()?;
            String::from_utf8(writer::bounded_json(&redact_traffic(&value), false)?).ok()?
        };
        output.extend_from_slice(&crate::anthropic::sse::encode_sse_event(
            event.event.as_deref(),
            &data,
        ));
        if output.len() > writer::MAX_RECORD_BYTES - 4096 {
            return None;
        }
    }
    Some(output)
}

pub struct StreamTrafficCapture {
    upstream_sse: Vec<u8>,
    upstream_events: Vec<Value>,
    downstream_events: Vec<Value>,
    malformed: Vec<Value>,
    upstream_event_bytes: usize,
    downstream_event_bytes: usize,
    upstream_sse_truncated: u64,
    upstream_events_truncated: u64,
    downstream_events_truncated: u64,
    malformed_truncated: u64,
    upstream_frames_truncated: u64,
}

impl Default for StreamTrafficCapture {
    fn default() -> Self {
        Self {
            upstream_sse: Vec::with_capacity(MAX_SSE_CAPTURE_BYTES.min(64 * 1024)),
            upstream_events: Vec::new(),
            downstream_events: Vec::new(),
            malformed: Vec::new(),
            upstream_event_bytes: 0,
            downstream_event_bytes: 0,
            upstream_sse_truncated: 0,
            upstream_events_truncated: 0,
            downstream_events_truncated: 0,
            malformed_truncated: 0,
            upstream_frames_truncated: 0,
        }
    }
}

impl StreamTrafficCapture {
    pub fn upstream_event(&mut self, event: Option<&str>, value: &Value) {
        if crate::bounded_json_size(value, MAX_STREAM_CAPTURE_FRAME_BYTES)
            .ok()
            .flatten()
            .is_none()
        {
            self.upstream_frames_truncated = self.upstream_frames_truncated.saturating_add(1);
            return;
        }
        let value = redact_traffic(value);
        let frame = serde_json::to_vec(&value).unwrap_or_default();
        let event = event.unwrap_or("message");
        let frame_len = event.len().saturating_add(frame.len()).saturating_add(16);
        if frame_len > MAX_STREAM_CAPTURE_FRAME_BYTES {
            self.upstream_frames_truncated = self.upstream_frames_truncated.saturating_add(1);
        } else if self.upstream_sse.len().saturating_add(frame_len) <= MAX_SSE_CAPTURE_BYTES {
            self.upstream_sse.extend_from_slice(b"event: ");
            self.upstream_sse.extend_from_slice(event.as_bytes());
            self.upstream_sse.extend_from_slice(b"\ndata: ");
            self.upstream_sse.extend_from_slice(&frame);
            self.upstream_sse.extend_from_slice(b"\n\n");
        } else {
            self.upstream_sse_truncated = self.upstream_sse_truncated.saturating_add(1);
        }
        self.push_event(true, serde_json::json!({"event":event,"data":value}));
    }

    pub fn malformed(&mut self, stage: &str, kind: &str) {
        if self.malformed.len() < MAX_STREAM_CAPTURE_EVENTS {
            self.malformed
                .push(serde_json::json!({"stage":stage,"kind":kind}));
        } else {
            self.malformed_truncated = self.malformed_truncated.saturating_add(1);
        }
    }

    pub fn downstream_event(&mut self, event: &str, data: Value) {
        if crate::bounded_json_size(&data, MAX_STREAM_CAPTURE_EVENT_BYTES)
            .ok()
            .flatten()
            .is_none()
        {
            self.downstream_events_truncated = self.downstream_events_truncated.saturating_add(1);
            return;
        }
        self.push_event(
            false,
            serde_json::json!({"event":event,"data":redact_traffic(&data)}),
        );
    }

    fn push_event(&mut self, upstream: bool, value: Value) {
        let bytes = serde_json::to_vec(&value).map_or(0, |value| value.len());
        let (events, total, truncated) = if upstream {
            (
                &mut self.upstream_events,
                &mut self.upstream_event_bytes,
                &mut self.upstream_events_truncated,
            )
        } else {
            (
                &mut self.downstream_events,
                &mut self.downstream_event_bytes,
                &mut self.downstream_events_truncated,
            )
        };
        if events.len() < MAX_STREAM_CAPTURE_EVENTS
            && total.saturating_add(bytes) <= MAX_STREAM_CAPTURE_EVENT_BYTES
        {
            *total += bytes;
            events.push(value);
        } else {
            *truncated = truncated.saturating_add(1);
        }
    }

    pub fn finish(self, traffic: &TrafficCapture, completion: Value) {
        self.finish_named(traffic, completion, "061-grok-stream-summary");
    }

    pub fn finish_named(self, traffic: &TrafficCapture, completion: Value, summary_name: &str) {
        if self.upstream_sse_truncated > 0
            || self.upstream_events_truncated > 0
            || self.downstream_events_truncated > 0
            || self.malformed_truncated > 0
            || self.upstream_frames_truncated > 0
        {
            // The detailed summary carries exact truncation counts. One omitted
            // aggregate also makes the final diagnostic marker incomplete.
            traffic.omit();
        }
        let upstream_event_count = self.upstream_events.len();
        let downstream_event_count = self.downstream_events.len();
        let upstream_sse_bytes = self.upstream_sse.len();
        if upstream_sse_bytes > 0 {
            // This private builder already redacted every event. Do not parse
            // and redact its bounded safe mirror a second time.
            traffic.enqueue(
                "032-upstream-response-body.sse",
                self.upstream_sse,
                false,
                false,
            );
        }
        traffic.write_json(
            "033-upstream-response-capture",
            &serde_json::json!({
                "truncated": self.upstream_sse_truncated > 0 || self.upstream_frames_truncated > 0 || self.upstream_events_truncated > 0,
                "captured_bytes": upstream_sse_bytes,
                "truncated_frames": self.upstream_sse_truncated,
                "oversized_frames": self.upstream_frames_truncated,
                "captured_events": self.upstream_events.len(),
                "captured_event_bytes": self.upstream_event_bytes,
                "truncated_events": self.upstream_events_truncated,
                "malformed": self.malformed,
                "truncated_malformed": self.malformed_truncated,
            }),
        );
        for value in self.upstream_events {
            traffic.write_json_event("040-upstream-event", &value);
        }
        for value in self.downstream_events {
            traffic.write_json_event("050-downstream-event", &value);
        }
        traffic.write_json(
            summary_name,
            &serde_json::json!({
                "completion": completion,
                "upstream_sse": {
                    "captured_bytes": upstream_sse_bytes,
                    "truncated_frames": self.upstream_sse_truncated,
                    "oversized_frames": self.upstream_frames_truncated,
                },
                "upstream_events": {
                    "captured": upstream_event_count,
                    "captured_bytes": self.upstream_event_bytes,
                    "truncated": self.upstream_events_truncated,
                },
                "downstream_events": {
                    "captured": downstream_event_count,
                    "captured_bytes": self.downstream_event_bytes,
                    "truncated": self.downstream_events_truncated,
                },
            }),
        );
    }
}

pub fn sanitize_path_part(input: &str) -> String {
    let cleaned: String = input
        .chars()
        .take(160)
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    let truncated = if cleaned.len() > 160 {
        &cleaned[..160]
    } else {
        &cleaned
    };
    if truncated.is_empty() {
        "unknown".to_string()
    } else {
        truncated.to_string()
    }
}

pub fn redact_traffic(value: &Value) -> Value {
    redact_traffic_with_depth(value, 0)
}

fn redact_traffic_with_depth(value: &Value, depth: u16) -> Value {
    if depth > 100 {
        return Value::String("[depth-limit]".to_string());
    }

    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                let normalized = key.to_lowercase();
                if normalized == "image_url" {
                    // Grok/OpenAI `input_image` parts carry data URLs; never
                    // persist the payload in traffic captures.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "data" && looks_like_image_source(map) {
                    // Anthropic image blocks carry raw base64 under
                    // `source.data`; redact it for the same reason.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "encrypted_content" {
                    // Codex `compaction` and `reasoning` items carry an opaque
                    // blob that replays a whole conversation's context; it is
                    // bulk payload, not debug signal, so only its size stays.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "signature"
                    && value.as_str().is_some_and(is_proxy_reasoning_signature)
                {
                    // The Codex translator replays reasoning as an Anthropic
                    // thinking signature of the form
                    // `ccp:codex:v1:<encoded-id>:<encrypted_content>`, which
                    // carries the same opaque conversation handle as the
                    // `encrypted_content` key. Only proxy-owned signatures are
                    // touched; foreign ones (for example, real Anthropic
                    // signatures) are preserved.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if REDACT_KEYS.contains(&normalized.as_str())
                    || matches!(
                        normalized.as_str(),
                        "token"
                            | "bearer_token"
                            | "oauth_token"
                            | "oauth_access_token"
                            | "oauth_refresh_token"
                            | "client_secret"
                            | "secret"
                            | "password"
                            | "email"
                            | "user_id"
                            | "account_id"
                            | "identity"
                            | "identity_id"
                            | "subject"
                            | "sub"
                    )
                {
                    // Credential keys hide the entire value, including a
                    // structured value whose leaves have no sensitive key.
                    let redacted = match value {
                        Value::String(_) => redact_traffic_value(value),
                        _ => Value::String("[redacted]".to_string()),
                    };
                    out.insert(key.clone(), redacted);
                } else {
                    out.insert(key.clone(), redact_traffic_with_depth(value, depth + 1));
                }
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| redact_traffic_with_depth(value, depth + 1))
                .collect(),
        ),
        Value::String(text) if is_data_url(text) => {
            Value::String(format!("[redacted data-url len={}]", text.len()))
        }
        _ => value.clone(),
    }
}

fn is_data_url(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.starts_with("data:image/") && lower.contains(";base64,")
}

/// An object shaped like an Anthropic image source: `type: "base64"` and
/// either an image `media_type` or none at all. Used to redact only image
/// payloads, not arbitrary `data` keys. Missing `media_type` is treated as
/// an image: the capture is written before translation, so a malformed block
/// that the translator would reject must still not persist its payload.
fn looks_like_image_source(map: &Map<String, Value>) -> bool {
    if !map
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|source_type| source_type.eq_ignore_ascii_case("base64"))
    {
        return false;
    }
    match map.get("media_type").and_then(Value::as_str) {
        Some(media_type) => media_type.to_ascii_lowercase().starts_with("image/"),
        None => true,
    }
}

fn redact_traffic_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(format!("[redacted len={}]", s.len())),
        // Structured values (e.g. Kimi's `{url: ...}` image_url) keep their
        // shape so captures stay parseable; only the payload leaves.
        Value::Object(_) | Value::Array(_) => redact_traffic(value),
        _ => Value::String("[redacted]".to_string()),
    }
}

#[allow(dead_code)]
fn _provider_alias(_provider: &str) -> Option<AliasProvider> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_completion_marks_omitted_raw_and_oversized_records_incomplete() {
        let writer = writer::pause_for_fixture();
        let temp = tempfile::tempdir().unwrap();
        let capture = test_capture(temp.path().to_path_buf());
        capture.write_text("malformed", "not-json access_token=do-not-copy");
        capture.write_json(
            "oversized",
            &serde_json::json!({"text":"x".repeat(writer::MAX_RECORD_BYTES)}),
        );
        assert_eq!(capture.progress.dropped.load(Ordering::Relaxed), 2);
        drop(capture);
        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let files: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1);
        let marker: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(marker["kind"], "diagnostic_capture_completion");
        assert_eq!(marker["complete"], false);
        assert_eq!(marker["dropped"], 2);
    }

    #[test]
    fn accepted_capture_records_keep_ordinals_and_completion_order() {
        let writer = writer::pause_for_fixture();
        let temp = tempfile::tempdir().unwrap();
        let capture = test_capture(temp.path().to_path_buf());
        capture.write_json("request", &serde_json::json!({"safe":1}));
        capture.write_json_event("event", &serde_json::json!({"safe":2}));
        assert_eq!(capture.progress.accepted.load(Ordering::Relaxed), 2);
        drop(capture);
        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        assert!(temp.path().join("000001-request.json").exists());
        assert!(temp.path().join("events/000002-event.json").exists());
        let marker: Value = serde_json::from_slice(
            &std::fs::read(temp.path().join("000003-capture-completion.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(marker["complete"], true);
        assert_eq!(marker["accepted_records"], 2);
    }

    #[test]
    fn raw_capture_cannot_bypass_structured_redaction() {
        let writer = writer::pause_for_fixture();
        let temp = tempfile::tempdir().unwrap();
        let capture = test_capture(temp.path().to_path_buf());
        capture.write_bytes("response.sse", b"data: {\"access_token\":\"private-fixture\",\"encrypted_content\":\"opaque-fixture\"}\n\n");
        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let path = std::fs::read_dir(temp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(path).unwrap();
        assert!(
            !text.contains("private-fixture"),
            "raw SSE bypassed redaction"
        );
        assert!(!text.contains("opaque-fixture"));
        assert!(text.contains("redacted"));
    }

    #[test]
    fn redact_traffic_strips_input_image_data_urls() {
        let value = serde_json::json!({
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "what color?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "base64 payload leaked: {rendered}"
        );
        assert!(
            !rendered.contains("data:image/png;base64,iVBOR"),
            "data URL payload leaked: {rendered}"
        );
        // The image part is still structurally visible (redacted marker).
        assert!(rendered.contains("input_image"));
    }

    #[test]
    fn redact_traffic_strips_inline_base64_image_values() {
        let value = serde_json::json!({
            "output": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(!rendered.contains("iVBORw0KGgo"));
    }

    #[test]
    fn redact_traffic_strips_anthropic_image_source_data() {
        // The incoming Anthropic request body carries raw base64 under
        // `source.data` (no data: URL wrapper) — it must not persist either.
        let value = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
                    }}
                ]
            }]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "source.data base64 leaked: {rendered}"
        );
        assert!(rendered.contains("redacted"));
        // Structure is preserved.
        assert!(rendered.contains("image/png"));
    }

    #[test]
    fn redact_traffic_strips_compaction_encrypted_content() {
        // Server-side compaction replays the transcript as an opaque blob in
        // the upstream request; it must not persist, but its size should.
        let value = serde_json::json!({
            "model": "gpt-5",
            "input": [
                {"type": "compaction", "encrypted_content": "gAAAAABopaque-compacted-transcript"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hello"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("gAAAAA"),
            "encrypted_content leaked: {rendered}"
        );
        assert_eq!(
            redacted["input"][0]["encrypted_content"],
            "[redacted len=34]"
        );
        assert_eq!(redacted["input"][0]["type"], "compaction");
        assert_eq!(redacted["input"][1]["content"][0]["text"], "hello");
        assert_eq!(redacted["model"], "gpt-5");
    }

    #[test]
    fn redact_traffic_strips_reasoning_encrypted_content_in_events() {
        // Upstream response items carry the same blob under `reasoning`.
        let value = serde_json::json!({
            "type": "response.output_item.done",
            "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAAAopaque"}
        });
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["item"]["encrypted_content"], "[redacted len=12]");
        assert_eq!(redacted["item"]["id"], "rs_1");
    }

    #[test]
    fn redact_traffic_strips_proxy_signature_delta_in_downstream_event() {
        // The Codex translator emits a 050 downstream `signature_delta` whose
        // signature is `ccp:codex:v1:<encoded-id>:<encrypted_content>`.
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let value = serde_json::json!({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature}
            }
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("ccp:codex:v1:"),
            "proxy signature leaked: {rendered}"
        );
        assert!(
            !rendered.contains("gAAAAABopaque"),
            "encrypted payload leaked: {rendered}"
        );
        assert_eq!(
            redacted["data"]["delta"]["signature"],
            format!("[redacted len={}]", signature.len())
        );
        // Neighboring structure survives.
        assert_eq!(redacted["data"]["delta"]["type"], "signature_delta");
        assert_eq!(redacted["data"]["index"], 0);
    }

    #[test]
    fn redact_traffic_strips_proxy_signature_in_thinking_blocks() {
        // Buffered downstream content and the replayed incoming Anthropic
        // request both carry the signature on a `thinking` block.
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let value = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "reasoned", "signature": signature},
                {"type": "text", "text": "answer"}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("ccp:codex:v1:"),
            "proxy signature leaked: {rendered}"
        );
        assert_eq!(
            redacted["content"][0]["signature"],
            format!("[redacted len={}]", signature.len())
        );
        assert_eq!(redacted["content"][0]["thinking"], "reasoned");
        assert_eq!(redacted["content"][1]["text"], "answer");
    }

    #[test]
    fn redact_traffic_keeps_foreign_signatures() {
        // Real Anthropic signatures and other opaque values are not
        // proxy-owned and must survive capture redaction unchanged.
        let value = serde_json::json!({
            "delta": {"type": "signature_delta", "signature": "ErUBCkYIBRgCIkA-real"},
            "content": [{"type": "thinking", "signature": "another-opaque-signature"}]
        });
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["delta"]["signature"], "ErUBCkYIBRgCIkA-real");
        assert_eq!(
            redacted["content"][0]["signature"],
            "another-opaque-signature"
        );
    }

    #[test]
    fn traffic_capture_redacts_proxy_signatures_at_json_boundary() {
        let writer = writer::pause_for_fixture();
        // Exercise write_json_event (050 downstream) and write_json (010
        // incoming request) end to end, where redaction is actually applied.
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let foreign = "ErUBCkYIBRgCIkA-real-anthropic-signature";

        capture.write_json_event(
            "050-downstream-event",
            &serde_json::json!({
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": signature}
                }
            }),
        );
        capture.write_json(
            "010-anthropic-request",
            &serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "kept", "signature": signature},
                        {"type": "thinking", "thinking": "kept", "signature": foreign}
                    ]
                }]
            }),
        );

        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let mut captured = String::new();
        for dir in [capture.root().to_path_buf(), capture.root().join("events")] {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    captured.push_str(&std::fs::read_to_string(path).unwrap());
                }
            }
        }

        assert!(
            !captured.contains("ccp:codex:v1:"),
            "proxy signature leaked to capture: {captured}"
        );
        assert!(
            !captured.contains("gAAAAABopaque"),
            "encrypted payload leaked to capture: {captured}"
        );
        assert!(
            captured.contains(foreign),
            "foreign signature lost: {captured}"
        );
        assert!(captured.contains("kept"));
    }

    #[test]
    fn redact_traffic_keeps_unrelated_data_keys() {
        let value = serde_json::json!({"data": "some-non-image-payload", "count": 3});
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["data"], "some-non-image-payload");
    }

    #[test]
    fn redact_traffic_strips_data_urls_in_array_shaped_tool_output() {
        // L2b inline mode: function_call_output.output is an array of content
        // parts — the image_url inside must be redacted just like message parts.
        let value = serde_json::json!({
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "screenshot"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "base64 payload leaked: {rendered}"
        );
        assert!(
            !rendered.contains("data:image/png;base64,iVBOR"),
            "data URL payload leaked: {rendered}"
        );
        // Structure and text survive.
        assert!(rendered.contains("input_image"));
        assert!(rendered.contains("screenshot"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn redact_traffic_strips_uppercase_media_type_image_source() {
        // Captures are written before translation validates anything, so a
        // non-standard casing must still not persist its payload.
        let value = serde_json::json!({
            "source": {"type": "base64", "media_type": "IMAGE/PNG", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
        assert!(
            redacted["source"]["data"]
                .as_str()
                .unwrap()
                .contains("redacted")
        );
    }

    #[test]
    fn redact_traffic_strips_uppercase_image_source_type() {
        let value = serde_json::json!({
            "source": {"type": "BASE64", "media_type": "image/png", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_strips_image_source_without_media_type() {
        // A malformed block the translator would reject must still be redacted
        // in the pre-translation capture.
        let value = serde_json::json!({
            "source": {"type": "base64", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_strips_uppercase_data_url_scheme() {
        let value = serde_json::json!({
            "image_url": "DATA:IMAGE/PNG;base64,iVBORw0KGgoAAAANSUhEUg"
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "data URL payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_keeps_structured_image_url_shape() {
        // Kimi-style object image_url: redact the payload but keep the object
        // structure so captures stay parseable.
        let value = serde_json::json!({
            "image_url": {"url": "DATA:IMAGE/PNG;BASE64,iVBORw0KGgoAAAANSUhEUg"}
        });
        let redacted = redact_traffic(&value);
        assert!(redacted["image_url"].is_object(), "shape lost: {redacted}");
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "nested payload leaked: {rendered}"
        );
    }

    #[test]
    fn traffic_redacts_proxy_authorization() {
        let redacted = redact_traffic(&serde_json::json!({
            "headers": {
                "proxy-authorization": "Basic dXNlcjpwYXNz",
                "authorization": ["Bearer structured-secret"],
                "access": {"value": "nested-secret"},
                "x-safe": "kept"
            }
        }));

        assert_eq!(redacted["headers"]["x-safe"], "kept");
        assert_eq!(redacted["headers"]["authorization"], "[redacted]");
        assert_eq!(redacted["headers"]["access"], "[redacted]");
        assert_eq!(
            redacted["headers"]["proxy-authorization"],
            "[redacted len=18]"
        );
    }
}
