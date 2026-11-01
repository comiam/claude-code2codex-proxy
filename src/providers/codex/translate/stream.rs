use super::live_stream::LiveStreamTranslator;
use super::reducer::{UpstreamStreamError, translate_upstream_bytes_with_policy};
use crate::traffic::TrafficCapture;

pub fn translate_stream_bytes(
    upstream: &[u8],
    message_id: &str,
    model: &str,
) -> Result<Vec<u8>, anyhow::Error> {
    translate_stream_bytes_with_traffic(upstream, message_id, model, 0, None)
}
pub fn translate_stream_bytes_with_traffic(
    upstream: &[u8],
    message_id: &str,
    model: &str,
    estimated_input_tokens: u64,
    traffic: Option<&TrafficCapture>,
) -> Result<Vec<u8>, anyhow::Error> {
    let (events, _) =
        translate_upstream_bytes_with_policy(upstream, super::IncompleteResponsePolicy::Error)
            .inspect_err(|error| {
                write_reducer_error_capture(traffic, error);
            })?;
    Ok(
        LiveStreamTranslator::with_estimated_input_tokens(
            message_id,
            model,
            estimated_input_tokens,
        )
        .render_events(&events, traffic),
    )
}

fn write_reducer_error_capture(traffic: Option<&TrafficCapture>, err: &UpstreamStreamError) {
    let Some(traffic) = traffic else {
        return;
    };
    traffic.write_json(
        "060-codex-stream-reducer-error",
        &serde_json::json!({
            "kind": format!("{:?}", err.kind),
            "retryAfterSeconds": err.retry_after_seconds,
            "diagnostics": err.diagnostics,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse_event(type_name: &str, payload: serde_json::Value) -> String {
        let mut obj = if let serde_json::Value::Object(m) = payload {
            m
        } else {
            return String::new();
        };
        obj.insert("type".into(), serde_json::json!(type_name));
        format!(
            "data: {}\n\n",
            serde_json::to_string(&serde_json::Value::Object(obj)).unwrap()
        )
    }

    #[test]
    fn quota_usage_matches_live_and_buffered_streams() {
        let quota = serde_json::json!({"type":"codex.rate_limits","plan_type":"pro","rate_limits":{"primary":{"used_percent":42},"secondary":{"window_minutes":60}}});
        let terminal = serde_json::json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":12,"output_tokens":9,"input_tokens_details":{"cached_tokens":4},"output_tokens_details":{"reasoning_tokens":3}}}});
        let upstream = format!("data: {quota}\n\ndata: {terminal}\n\n");
        let buffered = translate_stream_bytes(upstream.as_bytes(), "msg", "model").unwrap();
        let mut live = LiveStreamTranslator::new("msg", "model");
        let mut streamed = live.accept(&quota, None).unwrap();
        streamed.extend(live.accept(&terminal, None).unwrap());
        assert_eq!(streamed, buffered);
        let text = String::from_utf8(streamed).unwrap();
        assert!(text.contains("codex_rate_limits"));
        assert!(text.contains("reasoning_tokens"));
    }

    #[test]
    fn stream_translates_text_response() {
        let upstream = format!(
            "{}{}{}{}",
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index": 0,
                    "item": {"type":"message","id":"item_1"}
                })
            ),
            sse_event(
                "response.output_text.delta",
                serde_json::json!({
                    "output_index":0,"delta":"hello"
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":0,"item":{"type":"message"}
                })
            ),
            sse_event(
                "response.completed",
                serde_json::json!({
                    "response":{"id":"resp_1","usage":{"input_tokens":5,"output_tokens":1}}
                })
            ),
        );
        let out = String::from_utf8(
            translate_stream_bytes_with_traffic(upstream.as_bytes(), "msg_1", "gpt-5.5", 321, None)
                .unwrap(),
        )
        .unwrap();
        assert!(out.contains("message_start"));
        assert!(out.contains(r#""input_tokens":321"#));
        assert!(out.contains("text_delta"));
        assert!(out.contains(r#""input_tokens":5"#));
        assert!(out.contains(r#""output_tokens":1"#));
        assert!(out.contains("message_stop"));
    }

    #[test]
    fn stream_translates_web_search_response() {
        let upstream = format!(
            "{}{}{}{}{}{}{}{}",
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"web_search_call","id":"ws_1"}
                })
            ),
            sse_event(
                "response.web_search_call.in_progress",
                serde_json::json!({
                    "output_index":0,"item_id":"ws_1"
                })
            ),
            sse_event(
                "response.web_search_call.completed",
                serde_json::json!({
                    "output_index":0,"item_id":"ws_1"
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"web_search_call","id":"ws_1","action":{"query":"test query"}}
                })
            ),
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":1,
                    "item":{"type":"message","id":"msg_up"}
                })
            ),
            sse_event(
                "response.output_text.delta",
                serde_json::json!({
                    "output_index":1,"delta":"See [Result](https://result.com)"
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":1,"item":{"type":"message"}
                })
            ),
            sse_event(
                "response.completed",
                serde_json::json!({
                    "response":{"id":"resp_1","usage":{"input_tokens":3,"output_tokens":1}}
                })
            ),
        );
        let result = translate_stream_bytes(upstream.as_bytes(), "msg_1", "gpt-5.5").unwrap();
        let out = String::from_utf8(result).unwrap();
        assert!(out.contains("result"));
        assert!(out.contains(r#""type":"server_tool_use""#));
        assert!(out.contains("web_search_tool_result"));
    }

    #[test]
    fn stream_translates_reasoning_summary_to_thinking() {
        let upstream = format!(
            "{}{}{}{}{}{}",
            sse_event(
                "response.reasoning_summary_text.delta",
                serde_json::json!({
                    "output_index":0,"summary_index":0,"delta":"Working it out"
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}
                })
            ),
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":1,
                    "item":{"type":"message","id":"msg_up"}
                })
            ),
            sse_event(
                "response.output_text.delta",
                serde_json::json!({
                    "output_index":1,"delta":"answer"
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":1,"item":{"type":"message"}
                })
            ),
            sse_event(
                "response.completed",
                serde_json::json!({
                    "response":{"id":"resp_1","usage":{}}
                })
            ),
        );
        let out = String::from_utf8(
            translate_stream_bytes(upstream.as_bytes(), "msg_1", "gpt-5.5").unwrap(),
        )
        .unwrap();
        assert!(out.contains("\"type\":\"thinking\""));
        assert!(out.contains("\"signature\":\"\""));
        assert!(out.contains("\"type\":\"thinking_delta\""));
        assert!(out.contains("\"thinking\":\"Working it out\""));
        assert!(out.contains("event: message_stop"));
        assert!(
            out.find("\"type\":\"thinking_delta\"").unwrap()
                < out.find("\"type\":\"text_delta\"").unwrap()
        );
    }

    #[test]
    #[allow(clippy::format_in_format_args)]
    fn stream_omits_empty_reasoning_summary() {
        let upstream = format!(
            "{}{}{}{}{}",
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}
                })
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"reasoning","summary":[],"encrypted_content":"enc"}
                })
            ),
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":1,
                    "item":{"type":"message","id":"msg_up"}
                })
            ),
            sse_event(
                "response.output_text.delta",
                serde_json::json!({
                    "output_index":1,"delta":"answer"
                })
            ),
            format!(
                "{}{}",
                sse_event(
                    "response.output_item.done",
                    serde_json::json!({
                        "output_index":1,"item":{"type":"message"}
                    })
                ),
                sse_event(
                    "response.completed",
                    serde_json::json!({
                        "response":{"id":"resp_1","usage":{}}
                    })
                )
            ),
        );
        let out = String::from_utf8(
            translate_stream_bytes(upstream.as_bytes(), "msg_1", "gpt-5.5").unwrap(),
        )
        .unwrap();
        assert!(!out.contains("\"type\":\"thinking\""));
        assert!(!out.contains("\"type\":\"thinking_delta\""));
        assert!(out.contains("event: message_stop"));
    }

    #[test]
    fn stream_emits_signature_delta_before_thinking_stop() {
        let upstream = format!(
            "{}{}{}{}",
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
                })
            ),
            sse_event(
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index":0,"delta":"plan"})
            ),
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index":0,
                    "item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}
                })
            ),
            sse_event(
                "response.completed",
                serde_json::json!({"response":{"id":"resp_1","usage":{}}})
            ),
        );
        let out = String::from_utf8(
            translate_stream_bytes(upstream.as_bytes(), "msg_1", "gpt-5.5").unwrap(),
        )
        .unwrap();
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
}
