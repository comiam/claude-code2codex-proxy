//! Real HTTP ingress -> Codex client -> loopback upstream bodies. No process-global config mutation.
#[path = "support/proxy_speed.rs"]
mod support;
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use support::Proxy;

async fn post(proxy: &Proxy, session: &str, body: Value) -> reqwest::Response {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap()
        .post(format!("{}/v1/messages", proxy.base))
        .header("x-claude-code-session-id", session)
        .json(&body)
        .send()
        .await
        .unwrap()
}

fn command(session: &str, mode: &str) -> Value {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/proxy-speed-command.json")).unwrap();
    let mut body = fixture["cases"][1]["request"].clone();
    body["model"] = json!("gpt-5.5-fast");
    body["stream"] = json!(false);
    for block in body["messages"][0]["content"].as_array_mut().unwrap() {
        let text = block["text"]
            .as_str()
            .unwrap()
            .replace("11111111-1111-4111-8111-111111111111", session)
            .replace("fast", mode);
        block["text"] = json!(text);
    }
    body
}

fn generation(text: &str, stream: bool) -> Value {
    json!({"model":"gpt-5.5-fast","stream":stream,"max_tokens":64,
        "output_config":{"effort":"high"},"messages":[{"role":"user","content":text}]})
}

type Captures = Arc<Mutex<Vec<Value>>>;

async fn stub(
    State(captures): State<Captures>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    captures.lock().unwrap().push(body.clone());
    let text = body["input"].to_string();
    if text.contains("reject-tier") {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"API rejected literal tier","type":"invalid_request_error"}}))).into_response();
    }
    (
        [("content-type", "text/event-stream")],
        support::completion("resp_local", "Local generation."),
    )
        .into_response()
}

async fn upstream() -> (String, Captures, tokio::task::JoinHandle<()>) {
    let captures = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/responses", axum::routing::post(stub))
        .fallback(|| async { StatusCode::BAD_GATEWAY })
        .with_state(captures.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, captures, server)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn command_generation_cycle_sends_literal_tiers_and_cleans_history() {
    let (base, captures, server) = upstream().await;
    let proxy = Proxy::start(&base, Some("ultrafast"), "http", true).await;
    let session = uuid::Uuid::new_v4().to_string();
    // No auth file exists: even these successful commands cannot enter provider/auth work.
    for mode in ["off", "fast", "ultrafast", "off"] {
        let before = captures.lock().unwrap().len();
        let response = post(&proxy, &session, command(&session, mode)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let ack: Value = response.json().await.unwrap();
        assert!(ack["content"][0]["text"].as_str().unwrap().contains(mode));
        assert_eq!(
            captures.lock().unwrap().len(),
            before,
            "command hit upstream"
        );
    }
    proxy.write_synthetic_auth();
    for (mode, tier) in [
        ("off", None),
        ("fast", Some("fast")),
        ("ultrafast", Some("ultrafast")),
        ("off", None),
    ] {
        let response = post(&proxy, &session, command(&session, mode)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut request = generation("continue", false);
        request["messages"].as_array_mut().unwrap().splice(
            0..0,
            [
                command(&session, "ultrafast")["messages"][0].clone(),
                json!({"role":"assistant","content":"Old local acknowledgment."}),
            ],
        );
        let response = post(&proxy, &session, request).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "{}",
            response.text().await.unwrap()
        );
        let result: Value = response.json().await.unwrap();
        assert_eq!(result["content"][0]["text"], "Local generation.");
        let wire = captures.lock().unwrap().last().unwrap().clone();
        assert_eq!(wire.get("service_tier").and_then(Value::as_str), tier);
        assert_eq!(wire["model"], "gpt-5.5");
        assert_eq!(wire["reasoning"]["effort"], "high");
        assert!(!wire.to_string().contains("ccp-proxy-speed"));
        assert!(!wire.to_string().contains("command-name"));
        assert!(!wire["input"].to_string().contains("\"text\":\"\""));
    }
    // Another session still uses legacy global ultrafast -> priority narrowing.
    let response = post(&proxy, "other-session", generation("isolated", false)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let _: Value = response.json().await.unwrap();
    assert_eq!(
        captures.lock().unwrap().last().unwrap()["service_tier"],
        "priority"
    );
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn api_rejection_never_downgrades_literal_fast_or_ultrafast() {
    let (base, captures, server) = upstream().await;
    let proxy = Proxy::start(&base, None, "http", true).await;
    proxy.write_synthetic_auth();
    let session = uuid::Uuid::new_v4().to_string();
    for (mode, stream) in [
        ("fast", false),
        ("ultrafast", false),
        ("fast", true),
        ("ultrafast", true),
    ] {
        assert_eq!(
            post(&proxy, &session, command(&session, mode))
                .await
                .status(),
            StatusCode::OK
        );
        let before = captures.lock().unwrap().len();
        let response = post(&proxy, &session, generation("reject-tier", stream)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error: Value = response.json().await.unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("API rejected literal tier")
        );
        let captured = captures.lock().unwrap();
        assert_eq!(captured.len(), before + 1, "no retry at another tier");
        assert_eq!(captured.last().unwrap()["service_tier"], mode);
    }
    server.abort();
}

struct Pending {
    body: Value,
    reply: tokio::sync::oneshot::Sender<axum::response::Response>,
}

async fn controlled_upstream() -> (
    String,
    tokio::sync::mpsc::Receiver<Pending>,
    tokio::task::JoinHandle<()>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<Pending>(8);
    let handshake = tx.clone();
    let app = Router::new().route(
        "/responses",
        axum::routing::post(move |Json(body): Json<Value>| {
            let tx = tx.clone();
            async move {
                let (reply, response) = tokio::sync::oneshot::channel();
                tx.send(Pending { body, reply }).await.unwrap();
                response.await.unwrap()
            }
        })
        .get(move || {
            let handshake = handshake.clone();
            async move {
                let (reply, response) = tokio::sync::oneshot::channel();
                handshake
                    .send(Pending {
                        body: Value::Null,
                        reply,
                    })
                    .await
                    .unwrap();
                response.await.unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, rx, server)
}

async fn next(rx: &mut tokio::sync::mpsc::Receiver<Pending>) -> Pending {
    tokio::time::timeout(std::time::Duration::from_secs(15), rx.recv())
        .await
        .unwrap()
        .unwrap()
}

fn success() -> axum::response::Response {
    (
        [("content-type", "text/event-stream")],
        support::completion("resp_barrier", "Completed after barrier."),
    )
        .into_response()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_flight_generation_retry_keeps_snapshot_after_a_later_command() {
    let (base, mut pending, server) = controlled_upstream().await;
    let proxy = Arc::new(Proxy::start(&base, None, "http", true).await);
    proxy.write_synthetic_auth();
    let session = uuid::Uuid::new_v4().to_string();
    assert_eq!(
        post(&proxy, &session, command(&session, "fast"))
            .await
            .status(),
        StatusCode::OK
    );
    let task = {
        let proxy = proxy.clone();
        let session = session.clone();
        tokio::spawn(async move {
            let response = post(
                &proxy,
                &session,
                generation("barrier-controlled generation", true),
            )
            .await;
            let status = response.status();
            let body = response.text().await.unwrap();
            assert_eq!(status, StatusCode::OK, "{body}");
            assert!(body.contains("Completed after barrier."));
        })
    };
    let first = next(&mut pending).await;
    assert_eq!(first.body["service_tier"], "fast");
    // The first upstream request proves the immutable request snapshot exists.
    assert_eq!(
        post(&proxy, &session, command(&session, "off"))
            .await
            .status(),
        StatusCode::OK
    );
    first
        .reply
        .send(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [("retry-after", "0")],
                Json(json!({"error":{"message":"retry before output"}})),
            )
                .into_response(),
        )
        .unwrap();
    let retried = next(&mut pending).await;
    assert_eq!(retried.body["service_tier"], "fast");
    assert_eq!(retried.body["model"], first.body["model"]);
    retried.reply.send(success()).unwrap();
    task.await.unwrap();
    let task = {
        let proxy = proxy.clone();
        let session = session.clone();
        tokio::spawn(async move {
            post(&proxy, &session, generation("next generation", false))
                .await
                .text()
                .await
                .unwrap()
        })
    };
    let after = next(&mut pending).await;
    assert!(after.body.get("service_tier").is_none());
    after.reply.send(success()).unwrap();
    task.await.unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_transport_fallback_keeps_snapshot_while_command_changes_store() {
    for stream in [false, true] {
        let (base, mut pending, server) = controlled_upstream().await;
        let proxy = Arc::new(Proxy::start(&base, None, "auto", true).await);
        proxy.write_synthetic_auth();
        let session = uuid::Uuid::new_v4().to_string();
        assert_eq!(
            post(&proxy, &session, command(&session, "ultrafast"))
                .await
                .status(),
            StatusCode::OK
        );
        let task = {
            let proxy = proxy.clone();
            let session = session.clone();
            tokio::spawn(async move {
                post(&proxy, &session, generation("transport fallback", stream))
                    .await
                    .text()
                    .await
                    .unwrap()
            })
        };
        let handshake = next(&mut pending).await;
        assert!(handshake.body.is_null(), "must first attempt WebSocket");
        assert_eq!(
            post(&proxy, &session, command(&session, "off"))
                .await
                .status(),
            StatusCode::OK
        );
        handshake
            .reply
            .send(StatusCode::UPGRADE_REQUIRED.into_response())
            .unwrap();
        let fallback = next(&mut pending).await;
        assert_eq!(fallback.body["service_tier"], "ultrafast");
        fallback.reply.send(success()).unwrap();
        assert!(task.await.unwrap().contains("Completed after barrier."));
        server.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_commands_preserve_same_tier_continuation_and_change_tier_requires_full_context() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    for enabled in [false, true] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/responses", listener.local_addr().unwrap());
        let (tx, mut requests) =
            tokio::sync::mpsc::channel::<(Value, tokio::sync::oneshot::Sender<String>)>(8);
        let server = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (socket, _) = accepted.unwrap();
                        let tx = tx.clone();
                        connections.spawn(async move {
                            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
                            while let Some(Ok(message)) = socket.next().await {
                                let text = match message {
                                    Message::Text(text) => text,
                                    Message::Ping(bytes) => {socket.send(Message::Pong(bytes)).await.unwrap(); continue;},
                                    Message::Close(_) => break,
                                    other => panic!("unexpected client frame: {other:?}")
                                };
                                let body: Value = serde_json::from_str(&text).unwrap();
                                let (reply, response) = tokio::sync::oneshot::channel();
                                tx.send((body, reply)).await.unwrap();
                                let frames = response.await.unwrap();
                                for frame in claude_code_proxy::anthropic::parse_sse_events(frames.as_bytes()) {
                                    socket.send(Message::Text(frame.data)).await.unwrap();
                                }
                            }
                        });
                    }
                    Some(result) = connections.join_next(), if !connections.is_empty() => { result.unwrap(); }
                }
            }
        });
        let proxy = Arc::new(Proxy::start(&base, None, "websocket", enabled).await);
        proxy.write_synthetic_auth();
        let session = uuid::Uuid::new_v4().to_string();
        let mut history = Vec::new();
        for (turn, mode, expected_previous) in [
            (1, "fast", None),
            (2, "fast", enabled.then_some("resp_turn_1")),
            (3, "ultrafast", None),
        ] {
            let mut control = command(&session, mode);
            control["messages"] = json!(history);
            control["messages"]
                .as_array_mut()
                .unwrap()
                .push(command(&session, mode)["messages"][0].clone());
            let control_messages = control["messages"].as_array().unwrap().clone();
            let response = post(&proxy, &session, control).await;
            assert_eq!(response.status(), StatusCode::OK);
            let ack: Value = response.json().await.unwrap();
            assert!(
                requests.try_recv().is_err(),
                "control must not reserve/send a continuation turn"
            );
            // First generation has no control history; subsequent turns include
            // real local command/acknowledgment transcript, cleaned by ingress.
            if turn > 1 {
                history = control_messages;
                history.push(json!({"role":"assistant","content":ack["content"]}));
            }
            history.push(json!({"role":"user","content":format!("user turn {turn}")}));
            let mut body = generation("unused", false);
            body["messages"] = json!(history);
            let task = {
                let proxy = proxy.clone();
                let session = session.clone();
                tokio::spawn(async move {
                    let response = post(&proxy, &session, body).await;
                    let status = response.status();
                    let text = response.text().await.unwrap();
                    assert_eq!(status, StatusCode::OK, "{text}");
                    serde_json::from_str::<Value>(&text).unwrap()
                })
            };
            let (wire, reply) =
                tokio::time::timeout(std::time::Duration::from_secs(15), requests.recv())
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(wire["service_tier"], mode);
            assert_eq!(
                wire.get("previous_response_id").and_then(Value::as_str),
                expected_previous,
                "enabled={enabled}, turn={turn}: {wire}"
            );
            assert!(!wire.to_string().contains("ccp-proxy-speed"));
            if turn == 3 {
                assert!(
                    wire["input"].to_string().contains("user turn 1"),
                    "tier change keeps full context"
                );
            }
            let answer = format!("assistant turn {turn}");
            reply
                .send(support::completion(&format!("resp_turn_{turn}"), &answer))
                .unwrap();
            let response = task.await.unwrap();
            history.push(json!({"role":"assistant","content":response["content"]}));
        }
        server.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_resolved_model_is_rejected_before_command_mutation() {
    let (base, captures, server) = upstream().await;
    let proxy = Proxy::start(&base, None, "http", true).await;
    let config = proxy.root.path().join("config.json");
    std::fs::write(&config, r#"{"codex":{"model":"not-registered"}}"#).unwrap();
    let session = uuid::Uuid::new_v4().to_string();
    let response = post(&proxy, &session, command(&session, "ultrafast")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(captures.lock().unwrap().is_empty());
    std::fs::write(config, "{}").unwrap();
    proxy.write_synthetic_auth();
    let response = post(&proxy, &session, generation("state was not changed", false)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let _: Value = response.json().await.unwrap();
    // Session is still unset, so the pre-existing -fast suffix means priority.
    assert_eq!(captures.lock().unwrap()[0]["service_tier"], "priority");
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepted_mode_and_effective_generation_tier_are_observable_by_request() {
    let (base, _, server) = upstream().await;
    let proxy = Proxy::start(&base, None, "http", true).await;
    proxy.write_synthetic_auth();
    let session = uuid::Uuid::new_v4().to_string();
    for mode in ["fast", "ultrafast"] {
        let response = post(&proxy, &session, command(&session, mode)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _: Value = response.json().await.unwrap();
        let response = post(&proxy, &session, generation("observable generation", false)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let _: Value = response.json().await.unwrap();
    }
    let logs = std::fs::read_to_string(proxy.root.path().join("proxy.log")).unwrap();
    let records = logs
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect::<Vec<_>>();
    let changes = records
        .iter()
        .filter(|record| record["msg"] == "proxy_speed_changed")
        .collect::<Vec<_>>();
    assert_eq!(
        changes.len(),
        2,
        "accepted changes need correlated diagnostic events"
    );
    for (change, mode) in changes.iter().zip(["fast", "ultrafast"]) {
        assert_eq!(change["fields"]["mode"], mode);
        assert_eq!(change["fields"]["sessionId"], session);
        assert!(change["fields"]["sessionSeq"].as_u64().unwrap() > 0);
        assert!(uuid::Uuid::parse_str(change["fields"]["reqId"].as_str().unwrap()).is_ok());
    }
    let starts = records
        .iter()
        .filter(|record| record["msg"] == "codex_upstream_request_started")
        .collect::<Vec<_>>();
    assert_eq!(
        starts.len(),
        2,
        "acknowledgments must not appear as generation"
    );
    for (start, mode) in starts.iter().zip(["fast", "ultrafast"]) {
        assert_eq!(start["fields"]["requestedProxySpeed"], mode);
        assert_eq!(start["fields"]["serviceTier"], mode);
    }
    server.abort();
}
