use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::anthropic::schema::MessagesRequest;
use crate::anthropic::sse::encode_sse_event;
use crate::traffic::TrafficCapture;

use super::count_tokens::{approx_token_count, truncate_to_token_budget};
use super::translate::event::{SkipDiagnostics, SkippedEvent};
use super::translate::web_search_compat::{
    SearchCard, WebSearchResult, markdown_link, server_tool_use_id_from_codex_web_search_id,
    source_url_is_represented,
};

const SEARCH_OUTPUT_TOKEN_BUDGET: u64 = 2_500;
const SEARCH_ASSISTANT_CONTEXT_TOKEN_BUDGET: u64 = 1_000;
const SEARCH_USER_CONTEXT_MESSAGES: usize = 2;
const CLAUDE_SEARCH_PROMPT_PREFIX: &str = "Perform a web search for the query:";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchRequest {
    pub id: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    pub commands: SearchCommands,
    pub settings: SearchSettings,
    pub max_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchCommands {
    pub search_query: Vec<SearchQuery>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchQuery {
    pub q: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<SearchFilters>,
    pub allowed_callers: Vec<&'static str>,
    pub external_web_access: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SearchResponse {
    pub encrypted_output: Option<String>,
    pub output: String,
    #[serde(default)]
    pub results: Option<Vec<Value>>,
}

pub fn is_standalone_search_request(req: &MessagesRequest) -> bool {
    let Some(choice) = req.extra.get("tool_choice").and_then(Value::as_object) else {
        return false;
    };
    if choice.get("type").and_then(Value::as_str) != Some("tool") {
        return false;
    }
    let Some(selected_name) = choice.get("name").and_then(Value::as_str) else {
        return false;
    };
    req.extra
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("web_search_20250305")
                    && tool.get("name").and_then(Value::as_str) == Some(selected_name)
            })
        })
}

pub fn build_search_request(
    req: &MessagesRequest,
    model: &str,
    session_id: Option<&str>,
) -> Result<(SearchRequest, String), anyhow::Error> {
    let query = extract_search_query(req)
        .ok_or_else(|| anyhow::anyhow!("web_search request does not contain a text query"))?;
    let input = search_input(req);
    let filters = search_filters(req);
    let id = session_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("search-{}", uuid::Uuid::new_v4()));

    Ok((
        SearchRequest {
            id,
            model: model.to_string(),
            reasoning: None,
            input,
            commands: SearchCommands {
                search_query: vec![SearchQuery { q: query.clone() }],
            },
            settings: SearchSettings {
                filters,
                allowed_callers: vec!["direct"],
                external_web_access: true,
            },
            max_output_tokens: SEARCH_OUTPUT_TOKEN_BUDGET,
        },
        query,
    ))
}

pub fn search_request_input_tokens(request: &SearchRequest) -> u64 {
    let mut tokens = approx_token_count(&request.model);
    tokens += request
        .commands
        .search_query
        .iter()
        .map(|query| approx_token_count(&query.q))
        .sum::<u64>();
    tokens += request.input.as_ref().map(value_text_tokens).unwrap_or(0);
    if let Some(filters) = &request.settings.filters {
        tokens += filters
            .allowed_domains
            .iter()
            .flatten()
            .chain(filters.blocked_domains.iter().flatten())
            .map(|domain| approx_token_count(domain))
            .sum::<u64>();
    }
    tokens.max(1)
}

pub fn search_response_output_tokens(response: &SearchResponse) -> u64 {
    (approx_token_count(&response.output)
        + response
            .results
            .as_ref()
            .map(|results| results.iter().map(value_text_tokens).sum())
            .unwrap_or(0))
    .max(1)
}

fn value_text_tokens(value: &Value) -> u64 {
    match value {
        Value::String(text) => approx_token_count(text),
        Value::Array(values) => values.iter().map(value_text_tokens).sum(),
        Value::Object(values) => values.values().map(value_text_tokens).sum(),
        _ => 0,
    }
}

pub fn anthropic_search_response(
    response: &SearchResponse,
    query: &str,
    message_id: &str,
    model: &str,
    stream: bool,
    input_tokens: u64,
    traffic: Option<&TrafficCapture>,
) -> axum::response::Response {
    anthropic_search_response_with_request_id(
        response,
        query,
        message_id,
        model,
        stream,
        input_tokens,
        traffic,
        None,
    )
}

// Search response fields share the request-scoped diagnostic context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anthropic_search_response_with_request_id(
    response: &SearchResponse,
    query: &str,
    message_id: &str,
    model: &str,
    stream: bool,
    input_tokens: u64,
    traffic: Option<&TrafficCapture>,
    request_id: Option<&str>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let content = response_content(response, query, message_id);
    let mut diagnostics = SkipDiagnostics::default();
    if let Some(request_id) = request_id {
        diagnostics.set_request_id(request_id);
    }
    diagnostics.no_representable_output = content.is_empty();
    diagnostics.record(&SkippedEvent::new(
        &json!({"type":"standalone_search.native_pair"}),
        "anthropic_search_opaque_replay_unavailable",
    ));
    let output_tokens = search_response_output_tokens(response);
    let usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_tool_use": {"web_search_requests": 1}
    });

    if !stream {
        return (
            http::StatusCode::OK,
            axum::Json(json!({
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": usage,
            })),
        )
            .into_response();
    }

    let mut body = Vec::new();
    emit(
        &mut body,
        traffic,
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        }),
    );
    for (index, block) in content.iter().enumerate() {
        emit(
            &mut body,
            traffic,
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": if block["type"] == "text" { json!({"type": "text", "text": ""}) } else { block.clone() }
            }),
        );
        if block["type"] == "text" {
            emit(
                &mut body,
                traffic,
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]}
                }),
            );
        }
        emit_block_stop(&mut body, traffic, index);
    }
    emit(
        &mut body,
        traffic,
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": usage
        }),
    );
    emit(
        &mut body,
        traffic,
        "message_stop",
        &json!({"type": "message_stop"}),
    );

    let headers = [
        (http::header::CONTENT_TYPE, "text/event-stream"),
        (http::header::CACHE_CONTROL, "no-cache"),
        (http::header::CONNECTION, "keep-alive"),
    ];
    (headers, body).into_response()
}

fn extract_search_query(req: &MessagesRequest) -> Option<String> {
    req.messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .find_map(|message| {
            let text = content_text(&message.content);
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            Some(
                text.strip_prefix(CLAUDE_SEARCH_PROMPT_PREFIX)
                    .map(str::trim)
                    .filter(|query| !query.is_empty())
                    .unwrap_or(text)
                    .to_string(),
            )
        })
}

fn search_input(req: &MessagesRequest) -> Option<Value> {
    let mut messages: Vec<(&str, String)> = req
        .messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .filter_map(|message| {
            let text = content_text(&message.content);
            (!text.is_empty()).then_some((message.role.as_str(), text))
        })
        .collect();
    let latest_user = messages.iter().rposition(|(role, _)| *role == "user")?;
    messages.truncate(latest_user + 1);
    let first_user = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, (role, _))| *role == "user")
        .take(SEARCH_USER_CONTEXT_MESSAGES)
        .last()
        .map(|(index, _)| index)
        .unwrap_or(latest_user);
    messages.drain(..first_user);

    let mut assistant_budget = SEARCH_ASSISTANT_CONTEXT_TOKEN_BUDGET;
    let items: Vec<Value> = messages
        .into_iter()
        .filter_map(|(role, text)| {
            let (content_type, text) = if role == "assistant" {
                if assistant_budget == 0 {
                    return None;
                }
                let text = truncate_to_token_budget(&text, assistant_budget);
                assistant_budget = assistant_budget.saturating_sub(approx_token_count(&text));
                ("output_text", text)
            } else {
                ("input_text", text)
            };
            (!text.is_empty()).then(|| {
                json!({
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": text}]
                })
            })
        })
        .collect();
    (!items.is_empty()).then_some(Value::Array(items))
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn search_filters(req: &MessagesRequest) -> Option<SearchFilters> {
    let tool = req
        .extra
        .get("tools")?
        .as_array()?
        .iter()
        .find(|tool| tool.get("type").and_then(Value::as_str) == Some("web_search_20250305"))?;
    let allowed_domains = string_array(tool.get("allowed_domains"));
    let blocked_domains = string_array(tool.get("blocked_domains"));
    (allowed_domains.is_some() || blocked_domains.is_some()).then_some(SearchFilters {
        allowed_domains,
        blocked_domains,
    })
}

fn string_array(value: Option<&Value>) -> Option<Vec<String>> {
    let values = value?.as_array()?;
    let values: Vec<String> = values
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    (!values.is_empty()).then_some(values)
}

fn search_results(response: &SearchResponse) -> Vec<WebSearchResult> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for result in response.results.iter().flatten() {
        let Some(result) = WebSearchResult::from_value(result) else {
            continue;
        };
        if seen.insert(result.url.clone()) {
            results.push(result);
        }
    }
    results
}

fn response_content(response: &SearchResponse, query: &str, message_id: &str) -> Vec<Value> {
    let mut text = response.output.clone();
    for result in search_results(response) {
        if !source_url_is_represented(&text, &result.url) {
            text.push_str(&markdown_link(&result.title, &result.url));
        }
    }
    let card = SearchCard {
        id: server_tool_use_id_from_codex_web_search_id(message_id),
        input: json!({"query": query}),
        results: search_results(response),
    };
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(json!({"type":"text","text":text}));
    }
    content.extend(card.blocks());
    content
}

fn emit_block_stop(out: &mut Vec<u8>, traffic: Option<&TrafficCapture>, index: usize) {
    emit(
        out,
        traffic,
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    );
}

fn emit(out: &mut Vec<u8>, traffic: Option<&TrafficCapture>, event: &str, data: &Value) {
    if let Some(traffic) = traffic {
        traffic.write_json_event(
            "050-downstream-event",
            &json!({"event": event, "data": data}),
        );
    }
    out.extend_from_slice(&encode_sse_event(Some(event), &data.to_string()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::sse::parse_sse_events;

    fn request() -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 32000,
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Perform a web search for the query: Codex standalone search"
                }]
            }],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["openai.com"]
            }],
            "tool_choice": {"type": "tool", "name": "web_search"}
        }))
        .unwrap()
    }

    #[test]
    fn session_speed_does_not_add_service_tier_to_standalone_search() {
        use crate::proxy_speed::ProxySpeedMode;
        for mode in [
            ProxySpeedMode::Off,
            ProxySpeedMode::Fast,
            ProxySpeedMode::Ultrafast,
        ] {
            let mut req = request();
            req.proxy_speed_snapshot = Some(mode);
            let (search, query) =
                build_search_request(&req, "gpt-5.6-luna", Some("search-session")).unwrap();
            let wire = serde_json::to_value(search).unwrap();
            assert!(wire.get("service_tier").is_none());
            assert_eq!(wire["model"], "gpt-5.6-luna");
            assert_eq!(query, "Codex standalone search");
        }
    }

    #[test]
    fn request_preserves_luna_and_omits_reasoning() {
        let (request, query) =
            build_search_request(&request(), "gpt-5.6-luna", Some("session-1")).unwrap();
        assert_eq!(request.model, "gpt-5.6-luna");
        assert_eq!(request.reasoning, None);
        assert_eq!(request.id, "session-1");
        assert_eq!(request.max_output_tokens, 2_500);
        assert_eq!(query, "Codex standalone search");
        assert_eq!(request.commands.search_query[0].q, query);
        assert_eq!(
            request.settings.filters.unwrap().allowed_domains,
            Some(vec!["openai.com".to_string()])
        );
    }

    #[test]
    fn only_forced_claude_search_uses_standalone_endpoint() {
        let forced = request();
        assert!(is_standalone_search_request(&forced));

        let mut automatic = forced.clone();
        automatic
            .extra
            .insert("tool_choice".to_string(), json!({"type": "auto"}));
        assert!(!is_standalone_search_request(&automatic));
    }

    #[test]
    fn search_input_uses_role_specific_content_and_recent_context() {
        let mut req = request();
        req.messages = serde_json::from_value(json!([
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "previous user"},
            {"role": "assistant", "content": "previous assistant"},
            {
                "role": "user",
                "content": "Perform a web search for the query: current query"
            },
            {"role": "assistant", "content": "content after latest user"}
        ]))
        .unwrap();

        let (search, _) = build_search_request(&req, "gpt-5.6-luna", None).unwrap();
        let input = search.input.unwrap();
        let items = input.as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["content"][0]["text"], "previous user");
        assert_eq!(items[0]["content"][0]["type"], "input_text");
        assert_eq!(items[1]["content"][0]["text"], "previous assistant");
        assert_eq!(items[1]["content"][0]["type"], "output_text");
        assert_eq!(items[2]["content"][0]["type"], "input_text");
    }

    #[test]
    fn search_input_bounds_assistant_context() {
        let mut req = request();
        let long_assistant = (0..2_000)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        req.messages = serde_json::from_value(json!([
            {"role": "user", "content": "previous user"},
            {"role": "assistant", "content": long_assistant},
            {
                "role": "user",
                "content": "Perform a web search for the query: current query"
            }
        ]))
        .unwrap();

        let (search, _) = build_search_request(&req, "gpt-5.6-luna", None).unwrap();
        let assistant = search.input.unwrap()[1]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(approx_token_count(&assistant) <= SEARCH_ASSISTANT_CONTEXT_TOKEN_BUDGET);
        assert!(!assistant.contains("word1999"));
    }

    #[test]
    fn missing_structured_results_does_not_infer_urls_from_output() {
        let response = SearchResponse {
            encrypted_output: None,
            output: "Result from https://github.com with an embedded https://example.com link"
                .to_string(),
            results: None,
        };

        assert!(search_results(&response).is_empty());
    }

    #[test]
    fn standalone_usage_estimates_are_nonzero() {
        let (request, _) = build_search_request(&request(), "gpt-5.6-luna", None).unwrap();
        let response = SearchResponse {
            encrypted_output: None,
            output: "search output".to_string(),
            results: None,
        };

        assert!(search_request_input_tokens(&request) > 0);
        assert!(search_response_output_tokens(&response) > 0);
    }

    #[tokio::test]
    async fn standalone_response_projects_real_text_links_with_json_sse_parity() {
        for (response, expected) in [
            (
                SearchResponse {
                    encrypted_output: Some("opaque-must-not-escape".into()),
                    output: "See [OpenAI](https://openai.com/article_%28topic%29) and https://example.com/longer.".into(),
                    results: Some(vec![
                        json!({"url":"https://openai.com/article_(topic)","title":"OpenAI"}),
                        json!({"url":"https://example.com/long","title":"Distinct source"}),
                        json!({"url":"https://example.com/long","title":"Duplicate"}),
                    ]),
                },
                "See [OpenAI](https://openai.com/article_%28topic%29) and https://example.com/longer.\n[Distinct source](https://example.com/long)\n",
            ),
            (SearchResponse { encrypted_output: None, output: "Actual plain https://example.test/text".into(), results: None }, "Actual plain https://example.test/text"),
            (SearchResponse { encrypted_output: None, output: String::new(), results: Some(vec![json!({"url":"https://example.test/only","title":"Only source"})]) }, "\n[Only source](https://example.test/only)\n"),
            (SearchResponse { encrypted_output: Some("opaque-must-not-escape".into()), output: String::new(), results: None }, ""),
        ] {
            let buffered = anthropic_search_response(&response, "query-must-not-be-invented-output", "msg_test", "model", false, 12, None);
            let body = axum::body::to_bytes(buffered.into_body(), usize::MAX).await.unwrap();
            let buffered: Value = serde_json::from_slice(&body).unwrap();
            let expected_content = if expected.is_empty() { json!([]) } else { json!([{"type":"text","text":expected}]) };
            let text_content = buffered["content"].as_array().unwrap().iter().filter(|block| block["type"] == "text").cloned().collect::<Vec<_>>();
            assert_eq!(serde_json::json!(text_content), expected_content);
            assert_eq!(buffered["content"].as_array().unwrap().iter().filter(|block| block["type"] == "server_tool_use").count(), 1);
            assert_eq!(buffered["content"].as_array().unwrap().iter().filter(|block| block["type"] == "web_search_tool_result").count(), 1);
            let streamed = anthropic_search_response(&response, "query-must-not-be-invented-output", "msg_test", "model", true, 12, None);
            let body = axum::body::to_bytes(streamed.into_body(), usize::MAX).await.unwrap();
            let payloads: Vec<Value> = parse_sse_events(&body).iter().map(|event| serde_json::from_str(&event.data).unwrap()).collect();
            let mut text = String::new();
            for payload in &payloads {
                if payload["type"] == "content_block_start" && payload["content_block"]["type"] == "text" {
                    assert_eq!(payload["index"], 0);
                    assert_eq!(payload["content_block"], json!({"type":"text","text":""}));
                }
                if payload["type"] == "content_block_delta" {
                    assert_eq!(payload["delta"]["type"], "text_delta");
                    text.push_str(payload["delta"]["text"].as_str().unwrap());
                }
            }
            assert_eq!(text, expected);
            assert_eq!(payloads.iter().filter(|p| p["type"] == "content_block_start").count(), usize::from(!expected.is_empty()) + 2);
            assert_eq!(payloads.iter().filter(|p| p["type"] == "message_start").count(), 1);
            assert_eq!(payloads.iter().filter(|p| p["type"] == "message_stop").count(), 1);
            let final_delta = payloads.iter().find(|p| p["type"] == "message_delta").unwrap();
            assert_eq!(final_delta["usage"], buffered["usage"]);
            assert_eq!(buffered["usage"]["server_tool_use"]["web_search_requests"], 1);
            assert_eq!(buffered["usage"]["output_tokens"], search_response_output_tokens(&response));
            assert!(!String::from_utf8_lossy(&body).contains("opaque-must-not-escape"));
        }
    }
}
