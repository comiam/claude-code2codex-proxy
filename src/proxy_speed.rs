//! Versioned, explicitly user-invoked proxy-speed control messages.

use crate::anthropic::schema::{Message, MessagesRequest};
use crate::request_identity::{CLAUDE_SESSION_HEADER, ConversationIdentity};
use http::{HeaderMap, HeaderValue};
use serde_json::Value;
use std::borrow::Cow;

const MAX_ENVELOPE_BYTES: usize = 4 * 1024;
const COMMAND_MESSAGE_TAG: &str = "<command-message>proxy-speed</command-message>";
const COMMAND_PREFIX: &str =
    "<command-message>proxy-speed</command-message>\n<command-name>/proxy-speed</command-name>\n";
const BASE_DIRECTORY_PREFIX: &str = "Base directory for this skill: ";
const OMITTED_CONTROL: &str = "[Proxy speed command omitted.]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxySpeedMode {
    Off,
    Fast,
    Ultrafast,
}

impl ProxySpeedMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fast => "fast",
            Self::Ultrafast => "ultrafast",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxySpeedCommand {
    pub session_id: String,
    pub mode: ProxySpeedMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ControlError {
    #[error("Usage: /proxy-speed off|fast|ultrafast")]
    InvalidArguments,
    #[error("Unsupported proxy-speed protocol version")]
    UnsupportedVersion,
    #[error("Invalid proxy-speed control envelope")]
    InvalidEnvelope,
    #[error("Proxy-speed control envelope exceeds 4 KiB")]
    EnvelopeTooLarge,
    #[error("Proxy-speed requires a valid main session identity")]
    InvalidIdentity,
    #[error("Proxy-speed cannot be changed by an agent")]
    AgentNotAllowed,
    #[error("Proxy-speed session does not match the request identity")]
    SessionMismatch,
}

/// Only the latest non-system message can execute control. Never scans nested data.
pub fn parse_current_command(
    request: &MessagesRequest,
    headers: &HeaderMap,
) -> Result<Option<ProxySpeedCommand>, ControlError> {
    let Some(message) = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role != "system")
    else {
        return Ok(None);
    };
    let Some(command) = parse_message(message)? else {
        return Ok(None);
    };
    match ConversationIdentity::from_headers(headers) {
        Some(ConversationIdentity::Main(session)) if session == command.session_id => {
            Ok(Some(command))
        }
        Some(ConversationIdentity::Main(_)) => Err(ControlError::SessionMismatch),
        Some(ConversationIdentity::Agent(_, _)) => Err(ControlError::AgentNotAllowed),
        None => Err(ControlError::InvalidIdentity),
    }
}

/// Returns a separate provider/counting copy. Valid controls are removed, never applied.
pub fn sanitized_request(request: &MessagesRequest) -> MessagesRequest {
    let mut sanitized = request.clone();
    for message in &mut sanitized.messages {
        if !matches!(parse_message(message), Ok(Some(_))) {
            continue;
        }
        match &mut message.content {
            Value::String(text) => *text = OMITTED_CONTROL.to_string(),
            Value::Array(blocks) => {
                let frame_start = command_frame_start(blocks)
                    .expect("parsed control has an explicit command frame");
                for block in blocks.iter_mut().skip(frame_start) {
                    // Preserve cache metadata without emitting empty cached text.
                    block["text"] = Value::String(OMITTED_CONTROL.to_string());
                }
            }
            _ => unreachable!("parse_message accepts only text"),
        }
    }
    sanitized
}

/// Local protocol acknowledgment; zero usage is not an upstream observation.
pub fn acknowledgment_response(
    command: &ProxySpeedCommand,
    model: &str,
    stream: bool,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use serde_json::json;
    let mode = command.mode.as_str();
    let policy = if command.mode == ProxySpeedMode::Off {
        "Service tier is omitted for this session, including global and model-suffix settings."
    } else {
        "The literal tier is sent on the next ordinary Codex generation; support is determined by the API, not checked locally."
    };
    let text = format!(
        "Proxy speed: {mode}, saved for this session only. {policy} Model and reasoning effort are unchanged. Compact and standalone search keep their existing behavior; acceleration is not promised."
    );
    let mut message = json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message", "role": "assistant", "model": model,
        "content": [{"type":"text", "text":text}],
        "stop_reason":"end_turn", "stop_sequence":null,
        "usage":{"input_tokens":0, "output_tokens":0}
    });
    if !stream {
        return axum::Json(message).into_response();
    }
    message["content"] = json!([]);
    message["stop_reason"] = Value::Null;
    let events = [
        (
            "message_start",
            json!({"type":"message_start", "message":message}),
        ),
        (
            "content_block_start",
            json!({"type":"content_block_start", "index":0, "content_block":{"type":"text", "text":""}}),
        ),
        (
            "content_block_delta",
            json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":text}}),
        ),
        (
            "content_block_stop",
            json!({"type":"content_block_stop", "index":0}),
        ),
        (
            "message_delta",
            json!({"type":"message_delta", "delta":{"stop_reason":"end_turn", "stop_sequence":null}, "usage":{"output_tokens":0}}),
        ),
        ("message_stop", json!({"type":"message_stop"})),
    ];
    let bytes: Vec<u8> = events
        .iter()
        .flat_map(|(event, data)| {
            crate::anthropic::encode_sse_event(Some(event), &data.to_string())
        })
        .collect();
    (
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
        ],
        bytes,
    )
        .into_response()
}

fn parse_message(message: &Message) -> Result<Option<ProxySpeedCommand>, ControlError> {
    if message.role != "user" {
        return Ok(None);
    }
    let Some(text) = text_content(&message.content) else {
        return Ok(None);
    };
    let Some(metadata) = text.strip_prefix(COMMAND_PREFIX) else {
        return Ok(None);
    };
    // Claude Code omits command-args entirely when the slash command has no arguments.
    let (arguments, body) = if let Some(arguments) = metadata.strip_prefix("<command-args>") {
        arguments
            .split_once("</command-args>\n")
            .ok_or(ControlError::InvalidEnvelope)?
    } else {
        ("", metadata)
    };
    let Some(body) = body.strip_prefix(BASE_DIRECTORY_PREFIX) else {
        return Ok(None);
    };
    let Some((directory, envelope)) = body.split_once("\n\n") else {
        return Ok(None);
    };
    if !directory.starts_with('/')
        || !directory.ends_with("/skills/proxy-speed")
        || directory.contains('\n')
    {
        return Ok(None);
    }
    if !envelope.starts_with("<ccp-proxy-speed") {
        return Ok(None);
    }
    // Later input after a closed frame is not part of that control packet.
    // Do this before applying the envelope budget to the remaining message.
    if let Some((_, trailing)) = envelope.split_once(r#""/>"#)
        && !trailing.trim().is_empty()
    {
        return if trailing.trim_start().starts_with("<ccp-proxy-speed") {
            Err(ControlError::InvalidEnvelope)
        } else {
            Ok(None)
        };
    }
    if envelope.len() > MAX_ENVELOPE_BYTES {
        return Err(ControlError::EnvelopeTooLarge);
    }
    let attributes = envelope
        .trim_end()
        .strip_prefix("<ccp-proxy-speed version=\"")
        .ok_or(ControlError::InvalidEnvelope)?;
    let (version, attributes) = attributes
        .split_once("\" session=\"")
        .ok_or(ControlError::InvalidEnvelope)?;
    let (session, attributes) = attributes
        .split_once("\" mode=\"")
        .ok_or(ControlError::InvalidEnvelope)?;
    let mode = attributes
        .strip_suffix("\"/>")
        .ok_or(ControlError::InvalidEnvelope)?;
    if [version, session, mode]
        .iter()
        .any(|value| value.contains(['"', '<', '>']))
    {
        return Err(ControlError::InvalidEnvelope);
    }
    if version != "1" {
        return Err(ControlError::UnsupportedVersion);
    }
    let parsed_mode = match mode {
        "off" => ProxySpeedMode::Off,
        "fast" => ProxySpeedMode::Fast,
        "ultrafast" => ProxySpeedMode::Ultrafast,
        _ => return Err(ControlError::InvalidArguments),
    };
    if mode != arguments {
        return Err(ControlError::InvalidEnvelope);
    }
    // Reuse the ingress identity policy, including its 512-byte limit. Do not let
    // header whitespace normalization change the literal session inside the skill.
    let value = HeaderValue::from_str(session).map_err(|_| ControlError::InvalidIdentity)?;
    let mut headers = HeaderMap::new();
    headers.insert(CLAUDE_SESSION_HEADER, value);
    if ConversationIdentity::from_headers(&headers)
        != Some(ConversationIdentity::Main(session.to_string()))
    {
        return Err(ControlError::InvalidIdentity);
    }
    Ok(Some(ProxySpeedCommand {
        session_id: session.to_string(),
        mode: parsed_mode,
    }))
}

fn text_block(block: &Value) -> Option<&str> {
    (block["type"] == "text")
        .then(|| block["text"].as_str())
        .flatten()
}

// Consecutive CLI inputs can share one API user message after an error. Only a
// new reserved frame at a top-level block boundary is control; prior input and
// local output remain context, without scanning inside their text or tool data.
fn command_frame_start(blocks: &[Value]) -> Option<usize> {
    (0..blocks.len()).rev().find(|&start| {
        blocks[start..]
            .iter()
            .map_while(text_block)
            .flat_map(str::bytes)
            .take(COMMAND_MESSAGE_TAG.len())
            .eq(COMMAND_MESSAGE_TAG.bytes())
    })
}

fn text_content(content: &Value) -> Option<Cow<'_, str>> {
    match content {
        Value::String(text) => Some(Cow::Borrowed(text.as_str())),
        Value::Array(blocks) => {
            let frame_start = command_frame_start(blocks)?;
            let parts = blocks[frame_start..]
                .iter()
                .map(text_block)
                .collect::<Option<Vec<_>>>()?;
            // Ingress bounds the complete request. Check the logical prefix before
            // joining; only a recognized envelope is subject to the 4 KiB limit.
            if !parts
                .iter()
                .flat_map(|part| part.bytes())
                .take(COMMAND_PREFIX.len())
                .eq(COMMAND_PREFIX.bytes())
            {
                return None;
            }
            Some(Cow::Owned(parts.concat()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::schema::Message;
    use crate::request_identity::{CLAUDE_AGENT_HEADER, CLAUDE_SESSION_HEADER};
    use http::HeaderValue;
    use serde_json::{Value, json};

    const SESSION: &str = "11111111-1111-4111-8111-111111111111";

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/proxy-speed-command.json")).unwrap()
    }

    fn request() -> MessagesRequest {
        serde_json::from_value(fixture()["cases"][1]["request"].clone()).unwrap()
    }

    fn headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CLAUDE_SESSION_HEADER, HeaderValue::from_static(SESSION));
        headers
    }

    fn expanded_text(request: &mut MessagesRequest) -> &mut String {
        // Use Value::String directly so mutations retain the real content-block shape.
        match &mut request.messages[0].content[1]["text"] {
            Value::String(text) => text,
            _ => unreachable!(),
        }
    }

    fn assert_ordinary(request: &MessagesRequest) {
        assert_eq!(parse_current_command(request, &headers()), Ok(None));
        assert_eq!(
            serde_json::to_value(sanitized_request(request)).unwrap(),
            serde_json::to_value(request).unwrap(),
            "ordinary data must not be rewritten"
        );
    }

    #[test]
    fn real_client_fixture_preserves_three_distinct_modes_and_identity() {
        for (case, mode) in fixture()["cases"].as_array().unwrap().iter().zip([
            ProxySpeedMode::Off,
            ProxySpeedMode::Fast,
            ProxySpeedMode::Ultrafast,
        ]) {
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            assert_eq!(
                parse_current_command(&request, &headers()),
                Ok(Some(ProxySpeedCommand {
                    session_id: SESSION.to_string(),
                    mode
                }))
            );
        }
    }

    #[test]
    fn normal_client_context_is_preserved_while_the_command_is_recognized() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/proxy-speed-context.json"))
                .unwrap();
        let mut cases: Vec<_> = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .cloned()
            .map(|case| (case, ProxySpeedMode::Ultrafast))
            .collect();
        let compact: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/proxy-speed-after-compact.json"
        ))
        .unwrap();
        cases.push((compact, ProxySpeedMode::Fast));
        let failed_input: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/proxy-speed-after-api-error.json"
        ))
        .unwrap();
        cases.push((failed_input, ProxySpeedMode::Ultrafast));
        for (case, mode) in cases {
            let request: MessagesRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let context_blocks = case["context_blocks"].as_u64().unwrap() as usize;
            assert_eq!(
                parse_current_command(&request, &headers()),
                Ok(Some(ProxySpeedCommand {
                    session_id: SESSION.to_string(),
                    mode,
                })),
                "{}",
                case["case"]
            );
            let cleaned = sanitized_request(&request);
            let original = request.messages[0].content.as_array().unwrap();
            let sanitized = cleaned.messages[0].content.as_array().unwrap();
            assert_eq!(
                &sanitized[..context_blocks],
                &original[..context_blocks],
                "client instructions and cache metadata must remain intact"
            );
            assert_eq!(sanitized.len(), original.len());
            for block in &sanitized[context_blocks..] {
                assert_eq!(block["text"], OMITTED_CONTROL);
                assert!(!block["text"].as_str().unwrap().is_empty());
            }
            assert_eq!(parse_current_command(&cleaned, &headers()), Ok(None));
            assert_eq!(
                serde_json::to_value(sanitized_request(&cleaned)).unwrap(),
                serde_json::to_value(&cleaned).unwrap()
            );
            assert!(
                original.last().unwrap()["text"]
                    .as_str()
                    .unwrap()
                    .contains("<ccp-proxy-speed")
            );
            let mut historical = request;
            historical.messages.push(Message {
                role: "assistant".into(),
                content: json!("acknowledged"),
            });
            assert_eq!(parse_current_command(&historical, &headers()), Ok(None));
            assert_eq!(
                sanitized_request(&historical).messages[0].content,
                cleaned.messages[0].content
            );
        }
    }

    #[test]
    fn unknown_tagged_context_is_ignored_for_control_but_preserved_in_history() {
        let old = format!("<ccp-proxy-speed version=\"1\" session=\"{SESSION}\" mode=\"off\"/>");
        for text in [
            format!("<future-client-context>{old}</future-client-context>"),
            "<client-state detail=\"a > b\"/>".to_string(),
            "<first>one</first>\n<second>two</second>".to_string(),
            "<opaque-context>Markdown with <unclosed-example> text.</opaque-context>".to_string(),
        ] {
            let mut req = request();
            let context = json!({"type":"text","text":text,"cache_control":{"type":"ephemeral"}});
            req.messages[0]
                .content
                .as_array_mut()
                .unwrap()
                .insert(0, context.clone());
            assert_eq!(
                parse_current_command(&req, &headers()),
                Ok(Some(ProxySpeedCommand {
                    session_id: SESSION.to_string(),
                    mode: ProxySpeedMode::Fast,
                }))
            );
            assert_eq!(sanitized_request(&req).messages[0].content[0], context);
            req.messages[0].content = json!([context]);
            assert_ordinary(&req);
        }
    }

    #[test]
    fn a_closed_control_before_later_input_is_not_the_current_command() {
        for later in [
            "A new ordinary user question.".to_string(),
            "Новый ввод для суммаризации. ".repeat(400),
        ] {
            let mut req = request();
            req.messages[0]
                .content
                .as_array_mut()
                .unwrap()
                .push(json!({"type":"text","text":later}));
            assert_ordinary(&req);
            let combined = req.messages[0]
                .content
                .as_array()
                .unwrap()
                .iter()
                .map(|block| block["text"].as_str().unwrap())
                .collect::<String>();
            req.messages[0].content = json!(combined);
            assert_ordinary(&req);
        }
    }

    #[test]
    fn restored_skill_and_local_output_do_not_replay_a_historical_speed() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/proxy-speed-after-compact.json"
        ))
        .unwrap();
        let mut req: MessagesRequest = serde_json::from_value(fixture["request"].clone()).unwrap();
        let blocks = req.messages[0].content.as_array_mut().unwrap();
        blocks.truncate(fixture["context_blocks"].as_u64().unwrap() as usize);
        blocks.push(
            json!({"type":"text","text":"Continue normally, without changing proxy settings."}),
        );
        assert_ordinary(&req);
    }

    #[test]
    fn quoted_or_nested_command_data_is_not_an_input_frame() {
        let full = request().messages[0]
            .content
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["text"].as_str().unwrap())
            .collect::<String>();
        // Pasted examples remain within their data block. A separate final SDK
        // metadata block instead represents a new input after a pending question.
        for data in [
            json!({"type":"text", "text":format!("Explain this example:\n{full}")}),
            json!({"type":"text", "text":format!("<system-reminder>{full}</system-reminder>")}),
            json!({"type":"text", "text":format!("<future-context>{full}</wrong-tag>")}),
            json!({"type":"text", "text":format!("<unclosed>{full}")}),
            json!({"type":"text", "text":format!("```xml\n{full}\n```")}),
            json!({"type":"text", "text":format!("<command-message>proxy-speed</command-message>\n{full}")}),
            json!({"type":"tool_result", "tool_use_id":"tool-1", "content":[{"type":"text","text":full}]}),
        ] {
            let mut req = request();
            req.messages[0].content = json!([
                {"type":"text", "text":"<system-reminder>Client context.</system-reminder>"},
                data
            ]);
            assert_ordinary(&req);
        }
    }

    #[test]
    fn string_content_and_trailing_system_messages_are_current() {
        let mut request = request();
        request.messages[0].content = Value::String(
            request.messages[0]
                .content
                .as_array()
                .unwrap()
                .iter()
                .map(|block| block["text"].as_str().unwrap())
                .collect::<String>(),
        );
        request.messages.push(Message {
            role: "system".into(),
            content: json!("trailing notice"),
        });
        assert_eq!(
            parse_current_command(&request, &headers())
                .unwrap()
                .unwrap()
                .mode,
            ProxySpeedMode::Fast
        );
    }

    #[test]
    fn historical_command_is_not_executed_and_cleanup_is_idempotent() {
        let mut request = request();
        request.messages.extend([
            Message {
                role: "assistant".into(),
                content: json!("acknowledged"),
            },
            Message {
                role: "user".into(),
                content: json!("continue normally"),
            },
        ]);
        assert_eq!(parse_current_command(&request, &headers()), Ok(None));
        let cleaned = sanitized_request(&request);
        assert!(
            !serde_json::to_string(&cleaned.messages[0])
                .unwrap()
                .contains("ccp-proxy-speed")
        );
        assert_eq!(cleaned.messages.len(), 3);
        assert_eq!(cleaned.messages[0].role, "user");
        assert_eq!(cleaned.messages[1].content, json!("acknowledged"));
        assert_eq!(cleaned.messages[2].content, json!("continue normally"));
        assert_eq!(
            serde_json::to_value(sanitized_request(&cleaned)).unwrap(),
            serde_json::to_value(&cleaned).unwrap()
        );
        assert!(
            serde_json::to_string(&request.messages[0])
                .unwrap()
                .contains("ccp-proxy-speed"),
            "source request is untouched"
        );
    }

    #[test]
    fn current_command_can_be_cleaned_without_executing_it_for_counting() {
        let cleaned = sanitized_request(&request());
        assert!(
            !serde_json::to_string(&cleaned)
                .unwrap()
                .contains("ccp-proxy-speed")
        );
        assert_eq!(parse_current_command(&cleaned, &headers()), Ok(None));
    }

    #[test]
    fn ordinary_slash_mentions_quotes_and_unwrapped_markers_are_data() {
        let full = request().messages[0]
            .content
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["text"].as_str().unwrap())
            .collect::<String>();
        for text in [
            "/proxy-speed fast".to_string(),
            "Please discuss /proxy-speed fast".to_string(),
            format!("```xml\n{full}\n```"),
            format!("> {full}"),
            format!("Explain this example:\n{full}"),
            format!("<ccp-proxy-speed version=\"1\" session=\"{SESSION}\" mode=\"fast\"/>"),
        ] {
            let mut request = request();
            request.messages[0].content = json!(text);
            assert_ordinary(&request);
        }
    }

    #[test]
    fn model_invoked_skill_without_command_metadata_is_data() {
        let mut request = request();
        request.messages[0]
            .content
            .as_array_mut()
            .unwrap()
            .remove(0);
        assert_ordinary(&request);
    }

    #[test]
    fn system_assistant_and_tool_results_are_not_controls() {
        for role in ["system", "assistant", "tool"] {
            let mut request = request();
            request.messages[0].role = role.into();
            assert_ordinary(&request);
        }
        let mut request = request();
        let content = request.messages[0].content.clone();
        request.messages[0].content =
            json!([{"type": "tool_result", "tool_use_id": "tool-1", "content": content}]);
        assert_ordinary(&request);
    }

    #[test]
    fn empty_arguments_without_client_args_tag_return_usage() {
        let mut request = request();
        request.messages[0].content[0]["text"] = json!(
            "<command-message>proxy-speed</command-message>\n<command-name>/proxy-speed</command-name>\n"
        );
        let text = expanded_text(&mut request);
        *text = text.replace("mode=\"fast\"", "mode=\"\"");
        assert_eq!(
            parse_current_command(&request, &headers()),
            Err(ControlError::InvalidArguments)
        );
    }

    #[test]
    fn malformed_or_multiple_arguments_are_validation_errors() {
        for mode in ["", "turbo", "fast ultrafast", " fast", "FAST"] {
            let mut request = request();
            request.messages[0].content[0]["text"] = json!(format!(
                "<command-message>proxy-speed</command-message>\n<command-name>/proxy-speed</command-name>\n<command-args>{mode}</command-args>\n"
            ));
            *expanded_text(&mut request) = format!(
                "Base directory for this skill: /tmp/project/.claude/skills/proxy-speed\n\n<ccp-proxy-speed version=\"1\" session=\"{SESSION}\" mode=\"{mode}\"/>\n"
            );
            assert_eq!(
                parse_current_command(&request, &headers()),
                Err(ControlError::InvalidArguments),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn unknown_version_and_multiple_envelopes_are_validation_errors() {
        let mut unknown = request();
        let text = expanded_text(&mut unknown);
        *text = text.replace("version=\"1\"", "version=\"2\"");
        assert_eq!(
            parse_current_command(&unknown, &headers()),
            Err(ControlError::UnsupportedVersion)
        );
        let mut multiple = request();
        expanded_text(&mut multiple).push_str(&format!(
            "<ccp-proxy-speed version=\"1\" session=\"{SESSION}\" mode=\"off\"/>"
        ));
        assert_eq!(
            parse_current_command(&multiple, &headers()),
            Err(ControlError::InvalidEnvelope)
        );
    }

    #[test]
    fn oversized_envelopes_are_rejected_independently_of_text_block_boundaries() {
        let mut request = request();
        let mut full = request.messages[0]
            .content
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["text"].as_str().unwrap())
            .collect::<String>();
        full.push_str(&" ".repeat(17_000));
        for content in [
            json!(full),
            json!([{"type": "text", "text": full}]),
            json!([
                {"type": "text", "text": &full[..1]},
                {"type": "text", "text": &full[1..]}
            ]),
        ] {
            request.messages[0].content = content;
            assert_eq!(
                parse_current_command(&request, &headers()),
                Err(ControlError::EnvelopeTooLarge)
            );
        }
    }

    #[test]
    fn long_metadata_prefix_without_control_body_remains_ordinary() {
        let mut request = request();
        let full = format!(
            "{COMMAND_PREFIX}{}",
            "explain this ordinary text\n".repeat(700)
        );
        for content in [
            json!(full),
            json!([{"type": "text", "text": full}]),
            json!([
                {"type": "text", "text": &full[..1]},
                {"type": "text", "text": &full[1..]}
            ]),
        ] {
            request.messages[0].content = content;
            assert_ordinary(&request);
        }
    }

    #[test]
    fn envelope_limit_is_enforced_without_scrubbing_invalid_controls() {
        let mut request = request();
        expanded_text(&mut request).push_str(&" ".repeat(4097));
        assert_eq!(
            parse_current_command(&request, &headers()),
            Err(ControlError::EnvelopeTooLarge)
        );
        assert_eq!(
            serde_json::to_value(sanitized_request(&request)).unwrap(),
            serde_json::to_value(&request).unwrap()
        );
    }

    #[test]
    fn metadata_must_agree_with_the_envelope_and_no_extra_blocks_execute() {
        let mut mismatch = request();
        let text = expanded_text(&mut mismatch);
        *text = text.replace("mode=\"fast\"", "mode=\"off\"");
        assert_eq!(
            parse_current_command(&mismatch, &headers()),
            Err(ControlError::InvalidEnvelope)
        );
        let mut mixed = request();
        mixed.messages[0].content.as_array_mut().unwrap().push(json!({"type": "image", "source": {"type": "base64", "data": "AA==", "media_type": "image/png"}}));
        assert_ordinary(&mixed);
    }

    #[test]
    fn invalid_missing_duplicate_agent_and_mismatched_identities_are_rejected() {
        assert_eq!(
            parse_current_command(&request(), &HeaderMap::new()),
            Err(ControlError::InvalidIdentity)
        );
        let mut duplicate = headers();
        duplicate.append(CLAUDE_SESSION_HEADER, HeaderValue::from_static(SESSION));
        assert_eq!(
            parse_current_command(&request(), &duplicate),
            Err(ControlError::InvalidIdentity)
        );
        let mut agent = headers();
        agent.insert(CLAUDE_AGENT_HEADER, HeaderValue::from_static("agent-1"));
        assert_eq!(
            parse_current_command(&request(), &agent),
            Err(ControlError::AgentNotAllowed)
        );
        let mut mismatch = headers();
        mismatch.insert(
            CLAUDE_SESSION_HEADER,
            HeaderValue::from_static("another-session"),
        );
        assert_eq!(
            parse_current_command(&request(), &mismatch),
            Err(ControlError::SessionMismatch)
        );
        for session in [
            "".to_string(),
            "two values".to_string(),
            "a,b".to_string(),
            "x".repeat(513),
        ] {
            let mut request = request();
            let text = expanded_text(&mut request);
            *text = text.replace(SESSION, &session);
            assert_eq!(
                parse_current_command(&request, &headers()),
                Err(ControlError::InvalidIdentity)
            );
        }
    }
}
