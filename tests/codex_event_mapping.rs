//! Regressions exercise the real transport normalizer and incremental translator.
use std::{ffi::OsString, sync::Arc, time::Duration};

use claude_code_proxy::{
    provider::RequestContext,
    providers::codex::{
        auth::{manager::CodexAuthManager, token_store::file_store},
        client::CodexHttpClient,
        translate::{live_stream::LiveStreamTranslator, request::ResponsesRequest},
    },
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

// This binary has one test: no concurrent environment readers or real auth/config.
struct IsolatedEnvironment(Vec<(OsString, Option<OsString>)>);
impl IsolatedEnvironment {
    fn new(config: &std::path::Path) -> Self {
        let keys = std::env::vars_os()
            .map(|(k, _)| k)
            .filter(|k| {
                let k = k.to_string_lossy();
                k.starts_with("CCP_")
                    || k.starts_with("ANTHROPIC_")
                    || k.starts_with("OPENAI_")
                    || k.starts_with("CLAUDE_")
                    || matches!(
                        k.as_ref(),
                        "HTTP_PROXY"
                            | "HTTPS_PROXY"
                            | "ALL_PROXY"
                            | "http_proxy"
                            | "https_proxy"
                            | "all_proxy"
                            | "REQUEST_METHOD"
                    )
            })
            .chain(["CCP_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_STATE_HOME"].map(OsString::from));
        let mut previous = std::collections::BTreeMap::new();
        for key in keys {
            previous
                .entry(key.clone())
                .or_insert_with(|| std::env::var_os(&key));
        }
        unsafe {
            for key in previous.keys() {
                std::env::remove_var(key);
            }
            std::env::set_var("CCP_CONFIG_DIR", config);
            std::env::set_var("XDG_CONFIG_HOME", config.join("xdg"));
            std::env::set_var("XDG_STATE_HOME", config.join("state"));
        }
        Self(previous.into_iter().collect())
    }
}
impl Drop for IsolatedEnvironment {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn search_events(prefix: bool) -> Vec<Value> {
    let mut events = vec![json!({"type":"response.created","response":{"id":"resp_fixture"}})];
    if prefix {
        events.extend([
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_prefix"}}),
            json!({"type":"response.output_text.delta","output_index":0,"delta":"Working"}),
            json!({"type":"response.output_text.annotation.added","output_index":0,"annotation":{"type":"url_citation","title":"Prior source","url":"https://example.test/prior-source"}}),
            json!({"type":"response.output_text.annotation.added","output_index":0,"annotation":{"type":"url_citation","title":"Earlier mention","url":"https://example.test/real-source"}}),
            json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_prefix","content":[{"type":"output_text","text":"Working"}]}}),
        ]);
    }
    let i = usize::from(prefix);
    events.extend([
        json!({"type":"response.output_item.added","output_index":i,"item":{"type":"web_search_call","id":"ws_fixture","status":"in_progress"}}),
        json!({"type":"response.web_search_call.in_progress","output_index":i,"item_id":"ws_fixture"}),
        json!({"type":"response.web_search_call.searching","output_index":i,"item_id":"ws_fixture"}),
        json!({"type":"response.web_search_call.completed","output_index":i,"item_id":"ws_fixture"}),
        json!({"type":"response.output_item.done","output_index":i,"item":{"type":"web_search_call","id":"ws_fixture","status":"completed","action":{"type":"search","query":"synthetic"}}}),
        json!({"type":"response.output_item.added","output_index":i+1,"item":{"type":"message","id":"msg_answer"}}),
        json!({"type":"response.output_text.delta","output_index":i+1,"delta":"Answer"}),
        json!({"type":"response.output_text.annotation.added","output_index":i+1,"annotation":{"type":"url_citation","title":"Fixture source","url":"https://example.test/real-source"}}),
        json!({"type":"response.output_item.done","output_index":i+1,"item":{"type":"message","id":"msg_answer","content":[{"type":"output_text","text":"Answer"}]}}),
        json!({"type":"response.completed","response":{"id":"resp_fixture","status":"completed","usage":{"input_tokens":10,"output_tokens":2}}}),
    ]);
    for (n, event) in events.iter_mut().enumerate() {
        event["sequence_number"] = json!(n + 1);
    }
    events
}

// Optional artifacts are complete responses from production renderers for the
// isolated real-ClaudeCode probe; the test never manufactures downstream cards.
async fn search_ui_fixtures() {
    use claude_code_proxy::anthropic::sse::parse_sse_events;
    use claude_code_proxy::providers::codex::{
        search::{SearchResponse, anthropic_search_response},
        translate::accumulate::accumulate_response,
    };
    let events = search_events(false);
    let raw = events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect::<String>();
    let automatic = accumulate_response(raw.as_bytes(), "msg_fixture", "gpt-6-astra").unwrap();
    let mut translator = LiveStreamTranslator::new("msg_fixture", "gpt-6-astra");
    let mut automatic_sse = Vec::new();
    for event in &events {
        automatic_sse.extend(translator.accept(event, None).unwrap());
    }
    let response = SearchResponse {
        encrypted_output: Some("opaque-not-replay".into()),
        output: "Answer".into(),
        results: Some(vec![
            json!({"title":"Fixture source","url":"https://example.test/real-source"}),
        ]),
    };
    let standalone = anthropic_search_response(
        &response,
        "synthetic",
        "msg_fixture",
        "gpt-6-astra",
        false,
        10,
        None,
    );
    let standalone_json = axum::body::to_bytes(standalone.into_body(), usize::MAX)
        .await
        .unwrap();
    let standalone = serde_json::from_slice::<Value>(&standalone_json).unwrap();
    let standalone_sse = axum::body::to_bytes(
        anthropic_search_response(
            &response,
            "synthetic",
            "msg_fixture",
            "gpt-6-astra",
            true,
            10,
            None,
        )
        .into_body(),
        usize::MAX,
    )
    .await
    .unwrap();
    let mut two_events = search_events(false);
    let search_done = two_events
        .iter_mut()
        .find(|event| {
            event["type"] == "response.output_item.done"
                && event["item"]["type"] == "web_search_call"
        })
        .unwrap();
    search_done["item"]["action"]["sources"] =
        json!([{"title":"Fixture source","url":"https://example.test/real-source"}]);
    let terminal = two_events.pop().unwrap();
    let duplicate = two_events
        .iter()
        .find(|event| {
            event["type"] == "response.output_item.done"
                && event["item"]["type"] == "web_search_call"
        })
        .unwrap()
        .clone();
    two_events.push(duplicate);
    two_events.extend([
        json!({"type":"response.output_item.added","output_index":2,"item":{"type":"web_search_call","id":"ws_second","status":"in_progress"}}),
        json!({"type":"response.output_item.done","output_index":2,"item":{"type":"web_search_call","id":"ws_second","status":"completed","action":{"type":"search","query":"second query","sources":[{"title":"Second source","url":"https://example.test/second-source"}]}}}),
        terminal,
    ]);
    for (n, event) in two_events.iter_mut().enumerate() {
        event["sequence_number"] = json!(n + 1);
    }
    let raw = two_events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect::<String>();
    let two = accumulate_response(raw.as_bytes(), "msg_fixture", "gpt-6-astra").unwrap();
    let mut translator = LiveStreamTranslator::new("msg_fixture", "gpt-6-astra");
    let mut two_sse = Vec::new();
    for event in &two_events {
        two_sse.extend(translator.accept(event, None).unwrap());
    }
    let prefix_events = search_events(true);
    let raw = prefix_events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    let prefix = accumulate_response(raw.as_bytes(), "msg_fixture", "gpt-6-astra").unwrap();
    let mut translator = LiveStreamTranslator::new("msg_fixture", "gpt-6-astra");
    let mut prefix_sse = Vec::new();
    for event in &prefix_events {
        prefix_sse.extend(translator.accept(event, None).unwrap());
    }
    let cases = [
        ("automatic", automatic, automatic_sse),
        ("automatic-prefix", prefix, prefix_sse),
        ("standalone", standalone, standalone_sse.to_vec()),
        ("two-search", two, two_sse),
    ];
    if let Some(dir) = std::env::var_os("WEBSEARCH_UI_FIXTURE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, json, sse) in &cases {
            std::fs::write(
                dir.join(format!("{name}.json")),
                serde_json::to_vec(json).unwrap(),
            )
            .unwrap();
            std::fs::write(dir.join(format!("{name}.sse")), sse).unwrap();
        }
    }
    for (name, response, sse) in cases {
        let content = response["content"].as_array().unwrap();
        let uses = content
            .iter()
            .filter(|b| b["type"] == "server_tool_use")
            .collect::<Vec<_>>();
        let results = content
            .iter()
            .filter(|b| b["type"] == "web_search_tool_result")
            .collect::<Vec<_>>();
        if name == "automatic-prefix" {
            let text = content
                .iter()
                .filter_map(|block| block["text"].as_str())
                .collect::<String>();
            assert!(
                text.contains("[Prior source](https://example.test/prior-source)"),
                "the earlier citation must remain in ordinary text"
            );
            assert!(
                !results[0]
                    .to_string()
                    .contains("https://example.test/prior-source"),
                "an earlier message source must not be attributed to the new search"
            );
        }
        let count = if name == "two-search" { 2 } else { 1 };
        assert_eq!(
            uses.len(),
            count,
            "{name}: ClaudeCode must count each actual completed search"
        );
        assert_eq!(
            results.len(),
            count,
            "{name}: each completed search must have one paired result"
        );
        assert_eq!(
            response["usage"]["server_tool_use"]["web_search_requests"],
            count
        );
        assert_eq!(uses[0]["input"]["query"], "synthetic");
        assert_eq!(uses[0]["id"], results[0]["tool_use_id"]);
        assert_eq!(
            results[0]["content"],
            json!([{"type":"web_search_result","title":"Fixture source","url":"https://example.test/real-source"}])
        );
        let payloads = parse_sse_events(&sse)
            .iter()
            .map(|e| serde_json::from_str::<Value>(&e.data).unwrap())
            .collect::<Vec<_>>();
        let starts = payloads
            .iter()
            .filter(|p| p["type"] == "content_block_start")
            .collect::<Vec<_>>();
        let native = starts
            .iter()
            .map(|p| &p["content_block"])
            .filter(|b| b["type"] != "text")
            .cloned()
            .collect::<Vec<_>>();
        let expected_native = uses
            .iter()
            .zip(&results)
            .flat_map(|(call, result)| [(*call).clone(), (*result).clone()])
            .collect::<Vec<_>>();
        assert_eq!(
            native, expected_native,
            "{name}: JSON/SSE search cards differ"
        );
        if count == 2 {
            assert_ne!(uses[0]["id"], uses[1]["id"]);
            assert_eq!(uses[1]["input"]["query"], "second query");
            assert_eq!(uses[1]["id"], results[1]["tool_use_id"]);
            assert_eq!(
                results[1]["content"],
                json!([{"type":"web_search_result","title":"Second source","url":"https://example.test/second-source"}])
            );
        }
        let mut last_index = None;
        for start in starts {
            let index = start["index"].as_u64().unwrap();
            assert!(
                last_index.is_none_or(|last| index > last),
                "starts must be monotonic"
            );
            last_index = Some(index);
            assert_eq!(
                payloads
                    .iter()
                    .filter(|p| p["type"] == "content_block_stop" && p["index"] == index)
                    .count(),
                1
            );
        }
        assert!(!String::from_utf8_lossy(&sse).contains("opaque-not-replay"));
    }
}

async fn run_search(websocket: bool, prefix: bool, tool_failure: bool) -> Result<(), String> {
    let events = if tool_failure {
        vec![
            json!({"type":"response.created","response":{"id":"r"}}),
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f","call_id":"call_visible","name":"mcp__fixture__lookup"}}),
            json!({"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"fixture failure","retry_after":"0"}}}),
        ]
    } else {
        search_events(prefix)
    };
    let expected_order = events
        .iter()
        .map(|e| e["type"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/responses", listener.local_addr().unwrap());
    let peer = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        if websocket {
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            let request = socket.next().await.unwrap().unwrap().into_text().unwrap();
            let request: Value = serde_json::from_str(&request).unwrap();
            assert_eq!(request["type"], "response.create");
            for event in events {
                socket
                    .send(tokio_tungstenite::tungstenite::Message::Text(
                        event.to_string(),
                    ))
                    .await
                    .unwrap();
            }
        } else {
            let mut request = Vec::new();
            let mut buf = [0; 4096];
            loop {
                let n = stream.read(&mut buf).await.unwrap();
                assert_ne!(n, 0, "HTTP peer closed before sending request body");
                request.extend_from_slice(&buf[..n]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]).to_lowercase();
                    let length: usize = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap();
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let content = events
                .iter()
                .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
                .collect::<String>();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{content}",
                content.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        }
    });
    let client = Arc::new(CodexHttpClient::new_with_client(
        reqwest::Client::builder().no_proxy().build().unwrap(),
        CodexAuthManager::new(file_store()),
        url,
    ));
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model":"gpt-6-astra","input":[],"tools":[{"type":"web_search","external_web_access":true,"search_content_types":["text"]}],
        "stream":true,"store":false,"parallel_tool_calls":false,"text":{}
    })).unwrap();
    let ctx = RequestContext {
        req_id: "event-mapping-fixture".into(),
        session_id: None,
        session_seq: None,
        provider: "codex".into(),
        traffic: None,
        monitor: None,
    };
    let mut receiver = if websocket {
        client
            .stream_codex_websocket_events(&request, &ctx, None)
            .await
            .unwrap()
    } else {
        client
            .stream_codex_http_events(&request, &ctx)
            .await
            .unwrap()
    };
    let mut translator = LiveStreamTranslator::new("msg_fixture", "gpt-6-astra");
    let mut order = Vec::new();
    let mut output = Vec::new();
    let mut error = None;
    while let Some(event) = receiver.recv().await {
        let event = event.unwrap();
        if event["type"] == "keepalive" {
            continue;
        }
        order.push(event["type"].as_str().unwrap().to_owned());
        if error.is_none() {
            match translator.accept(&event, None) {
                Ok(bytes) => {
                    if event["type"] == "response.output_text.delta" && event["delta"] == "Answer" {
                        assert!(
                            String::from_utf8_lossy(&bytes).contains("Answer"),
                            "answer must cross barrier before terminal"
                        );
                    }
                    output.extend(bytes)
                }
                Err(reason) => error = Some(reason.as_str().to_owned()),
            }
        }
    }
    peer.await.unwrap();
    let output = String::from_utf8(output).unwrap();
    if tool_failure {
        if order == expected_order
            && output.contains("call_visible")
            && error.as_deref() == Some("fixture failure")
            && !output.contains("message_stop")
        {
            return Ok(());
        }
        return Err(format!(
            "tool start commit failed: websocket={websocket}, order={order:?}, error={error:?}, output={output}"
        ));
    }
    if prefix {
        let search_results = claude_code_proxy::anthropic::sse::parse_sse_events(output.as_bytes())
            .iter()
            .map(|event| serde_json::from_str::<Value>(&event.data).unwrap())
            .filter(|event| {
                event["type"] == "content_block_start"
                    && event["content_block"]["type"] == "web_search_tool_result"
            })
            .map(|event| event["content_block"]["content"].clone())
            .collect::<Vec<_>>();
        if !output.contains("[Prior source](https://example.test/prior-source)")
            || search_results
                != vec![
                    json!([{"type":"web_search_result","title":"Fixture source","url":"https://example.test/real-source"}]),
                ]
        {
            return Err(format!(
                "text-before-search source ownership: websocket={websocket}, results={search_results:?}, output={output}"
            ));
        }
    }
    if error.is_some()
        || order != expected_order
        || !output.contains("Answer")
        || !output.contains("[Fixture source](https://example.test/real-source)")
        || !output.contains("web_search_tool_result")
        || !output.contains("message_stop")
    {
        return Err(format!(
            "transport={} text_before_search={prefix}: translator_error={error:?}; expected_order={expected_order:?}; received_order={order:?}; answer={}; message_stop={}",
            if websocket { "websocket" } else { "http" },
            output.contains("Answer"),
            output.contains("message_stop")
        ));
    }
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn search_progress_preserves_order() {
    let config = tempfile::tempdir().unwrap();
    let _environment = IsolatedEnvironment::new(config.path());
    std::fs::create_dir(config.path().join("codex")).unwrap();
    std::fs::write(config.path().join("codex/auth.json"), json!({
        "access":"test-access","refresh":"test-refresh","expires":4_102_444_800_000_i64,"account_id":"acct_fixture"
    }).to_string()).unwrap();
    search_ui_fixtures().await;
    function_roundtrip_keeps_ids_outputs_and_control_wire_in_both_lanes();
    let mut failures = Vec::new();
    for websocket in [false, true] {
        for prefix in [false, true] {
            if let Err(reason) = tokio::time::timeout(
                Duration::from_secs(10),
                run_search(websocket, prefix, false),
            )
            .await
            .expect("loopback scenario timed out")
            {
                failures.push(reason);
            }
        }
    }
    for websocket in [false, true] {
        if let Err(reason) =
            tokio::time::timeout(Duration::from_secs(10), run_search(websocket, false, true))
                .await
                .expect("tool commit fixture timed out")
        {
            failures.push(reason);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn function_roundtrip_keeps_ids_outputs_and_control_wire_in_both_lanes() {
    use claude_code_proxy::{
        anthropic::schema::MessagesRequest,
        providers::codex::{
            client::build_websocket_request,
            translate::{
                accumulate::accumulate_response,
                request::{ServiceTier, TranslateOptions, translate_request},
            },
        },
    };
    let events = [
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call_a","name":"mcp__fixture__lookup"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"f1","call_id":"call_b","name":"SearchNotes"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"x\"}"}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"f0","call_id":"call_a","name":"mcp__fixture__lookup","arguments":"{\"q\":\"x\"}"}}),
        json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"f1","call_id":"call_b","name":"SearchNotes","arguments":"{}"}}),
        json!({"type":"response.completed","response":{"id":"response_fixture","status":"completed"}}),
    ];
    let raw = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    let completion = accumulate_response(raw.as_bytes(), "msg_fixture", "gpt-6-sol").unwrap();
    let request: MessagesRequest = serde_json::from_value(json!({
        "model":"gpt-6-sol","max_tokens":77,"stream":false,"system":"Fixture system",
        "tools":[
            {"name":"mcp__fixture__lookup","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}},
            {"name":"SearchNotes","input_schema":{"type":"object"}}
        ],
        "tool_choice":{"type":"any","disable_parallel_tool_use":false},
        "output_config":{"effort":"high"},
        "messages":[{"role":"user","content":"Start"},{"role":"assistant","content":completion["content"]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_b","content":"notes result"},{"type":"tool_result","tool_use_id":"call_a","content":"lookup result"},{"type":"text","text":"Continue"}]}]
    })).unwrap();
    let tools = json!([
        {"type":"function","name":"mcp__fixture__lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}}},"strict":false},
        {"type":"function","name":"SearchNotes","parameters":{"type":"object"},"strict":false}
    ]);
    let history = json!([
        {"type":"message","role":"user","content":[{"type":"input_text","text":"Start"}]},
        {"type":"function_call","call_id":"call_a","name":"mcp__fixture__lookup","arguments":"{\"q\":\"x\"}"},
        {"type":"function_call","call_id":"call_b","name":"SearchNotes","arguments":"{}"},
        {"type":"function_call_output","call_id":"call_b","output":"notes result"},
        {"type":"function_call_output","call_id":"call_a","output":"lookup result"},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"Continue"}]}
    ]);
    for lite in [false, true] {
        let translated = translate_request(
            &request,
            TranslateOptions {
                model: "gpt-6-sol".into(),
                service_tier: Some(ServiceTier::Flex),
                session_id: Some("fixture-session".into()),
                use_responses_lite: lite,
            },
        )
        .unwrap();
        let mut expected = json!({
            "model":"gpt-6-sol","instructions":"Fixture system","input":history,"tools":tools,
            "tool_choice":"required","store":false,"stream":true,"parallel_tool_calls":true,
            "include":["reasoning.encrypted_content"],"service_tier":"flex","prompt_cache_key":"fixture-session",
            "text":{"verbosity":"low"},"reasoning":{"effort":"high","summary":"auto"}
        });
        if lite {
            expected.as_object_mut().unwrap().remove("instructions");
            expected.as_object_mut().unwrap().remove("tools");
            let mut input = vec![
                json!({"type":"additional_tools","role":"developer","tools":tools}),
                json!({"type":"message","role":"developer","content":[{"type":"input_text","text":"Fixture system"}]}),
            ];
            input.extend(history.as_array().unwrap().clone());
            expected["input"] = json!(input);
            expected["parallel_tool_calls"] = json!(false);
            expected["client_metadata"] =
                json!({"ws_request_header_x_openai_internal_codex_responses_lite":"true"});
            expected["reasoning"]["context"] = json!("all_turns");
        }
        assert_eq!(serde_json::to_value(&translated).unwrap(), expected);
        expected.as_object_mut().unwrap().remove("stream");
        expected["type"] = json!("response.create");
        assert_eq!(build_websocket_request(&translated, None), expected);
    }
}
