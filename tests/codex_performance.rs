//! Synthetic peers exercise CodexHttpClient; these are local forwarding measurements,
//! not inference, wire delivery, or UI latency measurements.
use axum::{Router, body::Body, response::Response, routing::post};
use claude_code_proxy::{MessagesRequest, Provider, providers::codex::CodexProvider};
use claude_code_proxy::{
    RequestContext,
    auth::{KeychainFileAuthStore, SystemKeychain},
    providers::codex::{
        auth::{manager::CodexAuthManager, token_store::CodexTokenStore},
        client::CodexHttpClient,
        performance,
        translate::request::ResponsesRequest,
    },
};
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::json;
use std::ffi::{OsStr, OsString};
use std::{sync::Arc, time::Duration};
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
// Queue setup includes the process-wide WebSocket startup gate.
const QUEUE_FILL_WATCHDOG: Duration = Duration::from_secs(120);

// A blocked native writer or rejected destination must not hold a Tokio runtime
// or the whole process open. Each case owns its process because the real global
// stderr sink stays blocked.
#[cfg(unix)]
fn diagnostic_subprocess(test: &str, mode: &str) {
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("CCP_DIAGNOSTIC_CHILD", mode)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped()) // intentionally never drain while running
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{mode}: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            return;
        }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "{mode}: stream/shutdown exceeded subprocess deadline: {}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
fn diagnostic_child(mode: &str, capture: bool) {
    // The retention guard rejects nonregular disk destinations before opening
    // them. Only the undrained stderr pipe models a genuinely blocked writer.
    let rejected_destination = matches!(mode, "disk" | "failing");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let fixture = fixture(true).await;
        let _capture = EnvGuard::set("CCP_TRAFFIC_LOG", if capture { "1" } else { "0" });
        let log_path = claude_code_proxy::logging::log_file();
        std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
        if mode == "failing" {
            std::fs::create_dir(&log_path).unwrap();
        } else if mode == "disk" {
            assert!(
                std::process::Command::new("mkfifo")
                    .arg(&log_path)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let logger = claude_code_proxy::logging::create_logger("synthetic-slow-sink");
        let (_, first_policy) = performance::observe(async {
            if mode == "stderr" {
                logger.warn(&"s".repeat(1024 * 1024), None);
            } else {
                logger.info("block the diagnostic file", None);
            }
        })
        .await;
        assert_eq!(
            first_policy.config_reads, 1,
            "the first library policy read must be visible"
        );
        let (_, reused_policy) = performance::observe(async {
            logger.info("reuse startup policy", None);
        })
        .await;
        assert_eq!(
            reused_policy.config_reads, 0,
            "ordinary logs must not reread configuration per record"
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let stats = claude_code_proxy::logging::diagnostics_snapshot();
                if (rejected_destination && stats.failed > 0)
                    || (!rejected_destination && stats.inflight_bytes > 0)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if capture {
            let traffic = claude_code_proxy::traffic::create_traffic_capture(
                claude_code_proxy::traffic::TrafficCaptureOptions {
                    req_id: "saturated".into(),
                    session_id: None,
                    session_seq: Some(1),
                    provider: Some("codex".into()),
                    state_dir_override: Some(fixture._auth.path().into()),
                },
            )
            .unwrap();
            for _ in 0..16 {
                traffic.write_json_event("fixture", &json!({"text":"x".repeat(1024 * 1024)}));
            }
            drop(traffic); // completion marker is best effort too
            let stats = claude_code_proxy::logging::diagnostics_snapshot();
            assert!(stats.dropped > 0);
            assert!(stats.queued_bytes <= 8 * 1024 * 1024);
            assert!(stats.queued_records <= 2048);
            assert!(stats.inflight_bytes <= 2 * 1024 * 1024);
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(claude_code_proxy::server::serve_listener(
            listener,
            Some(claude_code_proxy::monitor::MonitorHandle::default()),
            async {
                let _ = shutdown_rx.await;
            },
        ));
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            let stream = client
                .post(format!("http://{address}/v1/messages"))
                .json(&messages(128))
                .send();
            let health = client.get(format!("http://{address}/healthz")).send();
            let (response, health) = tokio::join!(stream, health);
            assert_eq!(health.unwrap().status(), http::StatusCode::OK);
            let response = response.unwrap();
            assert_eq!(response.status(), http::StatusCode::OK);
            let text = response.text().await.unwrap();
            let events = claude_code_proxy::anthropic::sse::parse_sse_events(text.as_bytes());
            let mut answer = String::new();
            for event in events {
                let data: serde_json::Value = serde_json::from_str(&event.data).unwrap();
                if data["delta"]["type"] == "text_delta" {
                    answer.push_str(data["delta"]["text"].as_str().unwrap());
                }
            }
            assert_eq!(answer, "answer");
            assert!(text.contains("message_stop"));
            assert!(!text.contains("event: error"));
            let monitor: serde_json::Value = client
                .get(format!("http://{address}/monitor"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if capture {
                assert!(
                    monitor["snapshot"]["diagnostics"]["dropped"]
                        .as_u64()
                        .unwrap()
                        > 0
                );
            }
        })
        .await
        .expect("ordinary logs/capture must not delay streaming or health");
        let _ = shutdown_tx.send(());
        server.await.unwrap().unwrap();
    });
    let started = std::time::Instant::now();
    let drained = claude_code_proxy::logging::shutdown_diagnostics();
    assert_eq!(drained, rejected_destination);
    drop(runtime); // Includes the complete runtime teardown, with writer still blocked.
    assert!(started.elapsed() < Duration::from_millis(2500));
}

#[cfg(unix)]
#[test]
fn ordinary_logs_do_not_block_stream_without_capture() {
    if let Ok(mode) = std::env::var("CCP_DIAGNOSTIC_CHILD") {
        diagnostic_child(&mode, false);
        return;
    }
    for mode in ["disk", "stderr", "failing"] {
        diagnostic_subprocess("ordinary_logs_do_not_block_stream_without_capture", mode);
    }
}

#[cfg(unix)]
#[test]
fn capture_saturation_preserves_api_output() {
    if let Ok(mode) = std::env::var("CCP_DIAGNOSTIC_CHILD") {
        diagnostic_child(&mode, true);
        return;
    }
    diagnostic_subprocess("capture_saturation_preserves_api_output", "stderr");
}
struct EnvGuard {
    key: &'static str,
    previous: Option<OsString>,
}
impl EnvGuard {
    fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
    fn unset(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn request(bytes: usize) -> ResponsesRequest {
    serde_json::from_value(json!({"model":"gpt-6-sol","instructions":"synthetic instructions","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"h".repeat(bytes)}]}],"tools":[{"type":"function","name":"fixture_tool","description":"fixture","parameters":{"type":"object","properties":{}}}],"store":false,"stream":true,"parallel_tool_calls":false,"prompt_cache_key":"synthetic-cache","text":{"verbosity":"low"},"reasoning":{"effort":"high"},"service_tier":"priority"})).unwrap()
}

struct Fixture {
    client: Arc<CodexHttpClient>,
    provider: Arc<CodexProvider>,
    _env: Vec<EnvGuard>,
    _auth: tempfile::TempDir,
    server: tokio::task::JoinHandle<()>,
    upstream_closed: Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}
async fn fixture(delayed: bool) -> Fixture {
    fixture_mode(delayed, false, false).await
}

struct PeerBodyDrop(Arc<std::sync::atomic::AtomicBool>);
impl Drop for PeerBodyDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

async fn fixture_mode(delayed: bool, flood: bool, websocket: bool) -> Fixture {
    // Never fall back to the operator's auth. Configuration/log paths are isolated by
    // the test command's synthetic CCP_CONFIG_DIR and XDG_STATE_HOME environment.
    let auth = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(auth.path().join("codex")).unwrap();
    std::fs::write(
        auth.path().join("config.json"),
        r#"{"log":{"verbose":false,"stderr":false}}"#,
    )
    .unwrap();
    let auth_file = auth.path().join("codex/auth.json");
    std::fs::write(&auth_file, r#"{"access":"synthetic-token","refresh":"","expires":4102444800000,"accountId":"synthetic-account"}"#).unwrap();
    let store = KeychainFileAuthStore::new(
        auth_file.to_string_lossy().into_owned(),
        auth.path()
            .join("missing.json")
            .to_string_lossy()
            .into_owned(),
        "fixture",
        "fixture",
        false,
        SystemKeychain,
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut guards = [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "REQUEST_METHOD",
        "CCP_AUTO_REVIEW_MODEL",
        "CCP_CODEX_MODEL",
        "CCP_CODEX_EFFORT",
        "CCP_CODEX_SERVICE_TIER",
        "CCP_CODEX_ORIGINATOR",
        "CCP_CODEX_USER_AGENT",
        "CCP_USER_AGENT",
    ]
    .into_iter()
    .map(EnvGuard::unset)
    .collect::<Vec<_>>();
    guards.extend([
        EnvGuard::set("CCP_CONFIG_DIR", auth.path()),
        EnvGuard::set("XDG_CONFIG_HOME", auth.path()),
        EnvGuard::set("XDG_STATE_HOME", auth.path()),
        EnvGuard::set("NO_PROXY", "127.0.0.1,localhost"),
        EnvGuard::set("no_proxy", "127.0.0.1,localhost"),
        EnvGuard::set(
            "CCP_CODEX_TRANSPORT",
            if websocket { "websocket" } else { "http" },
        ),
        EnvGuard::set("CCP_CODEX_BASE_URL", format!("http://{address}/responses")),
        EnvGuard::set("CCP_CODEX_PREVIOUS_RESPONSE_ID", "0"),
        EnvGuard::set("CCP_CODEX_SERVER_COMPACTION", "0"),
    ]);
    let upstream_closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let body_closed = upstream_closed.clone();
    let app = Router::new().route("/responses", post(move |headers: http::HeaderMap, axum::Json(payload): axum::Json<serde_json::Value>| { let body_closed = body_closed.clone(); async move {
        assert_eq!(headers["authorization"], "Bearer synthetic-token");
        assert_eq!(payload["model"], "gpt-6-sol");
        if payload["client_metadata"].is_object() {
            assert!(payload.get("instructions").is_none());
            assert_eq!(payload["input"][0]["tools"][0]["name"], "fixture_tool");
            assert_eq!(payload["input"][1]["role"], "developer");
            assert_eq!(payload["input"][1]["content"][0]["text"], "synthetic instructions");
        } else {
            assert_eq!(payload["instructions"], "synthetic instructions");
            assert_eq!(payload["tools"][0]["name"], "fixture_tool");
        }
        assert_eq!(payload["reasoning"]["effort"], "high");
        assert_eq!(payload["service_tier"], "priority");
        assert_eq!(payload["parallel_tool_calls"], false);
        assert!(payload.get("previous_response_id").is_none());
        let frames = [r#"{"type":"response.created","response":{"id":"synthetic-response"}}"#, r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"synthetic-message"}}"#, r#"{"type":"response.output_text.delta","output_index":0,"delta":"answer"}"#, r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"synthetic-message","content":[{"type":"output_text","text":"answer"}]}}"#, r#"{"type":"response.completed","response":{"id":"synthetic-response","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}"#];
        if flood {
            let body = stream::unfold((0usize, PeerBodyDrop(body_closed)), |(index, guard)| async move {
                if index == 256 { std::future::pending::<()>().await; }
                let event = if index == 0 { json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"flood-message"}}) } else { json!({"type":"response.output_text.delta","output_index":0,"delta":"x".repeat(1024)}) };
                Some((Ok::<_, std::io::Error>(format!("data: {event}\n\n")), (index + 1, guard)))
            });
            return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(body)).unwrap();
        }
        let body = stream::unfold((0, frames), move |(index, frames)| async move {
            if index == frames.len() { return None; }
            if delayed { tokio::time::sleep(Duration::from_millis(15)).await; }
            Some((Ok::<_, std::io::Error>(format!("data: {}\n\n", frames[index])), (index + 1, frames)))
        });
        Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(body)).unwrap()
    }}));
    let socket_closed = upstream_closed.clone();
    let server = tokio::spawn(async move {
        if websocket {
            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;
            let (io, _) = listener.accept().await.unwrap();
            let mut peer = tokio_tungstenite::accept_async(io).await.unwrap();
            assert!(matches!(peer.next().await, Some(Ok(Message::Text(_)))));
            peer.send(Message::Text(json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"flood-message"}}).to_string())).await.unwrap();
            for _ in 0..256 {
                if peer
                    .send(Message::Text(
                        json!({"type":"response.output_text.delta","output_index":0,"delta":"x".repeat(1024)})
                            .to_string(),
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            while let Some(Ok(frame)) = peer.next().await {
                match frame {
                    Message::Ping(data) => {
                        if peer.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            socket_closed.store(true, std::sync::atomic::Ordering::Release);
        } else {
            axum::serve(listener, app).await.unwrap();
        }
    });
    let client = Arc::new(CodexHttpClient::new_with_client(
        reqwest::Client::builder().no_proxy().build().unwrap(),
        CodexAuthManager::new(CodexTokenStore::new(store)),
        format!("http://{address}/responses"),
    ));
    Fixture {
        client,
        provider: Arc::new(CodexProvider::new()),
        _env: guards,
        _auth: auth,
        server,
        upstream_closed,
    }
}
fn context(owner: usize) -> RequestContext {
    RequestContext {
        req_id: format!("fixture-{owner}"),
        session_id: Some(format!("synthetic-owner-{owner}")),
        session_seq: None,
        provider: "codex".into(),
        traffic: None,
        monitor: None,
    }
}

// Successful telemetry requires trustworthy terminal usage and completed item lifecycles.
async fn telemetry_fixture(usage: serde_json::Value, resolved: bool) -> Fixture {
    let mut fixture = fixture(false).await;
    fixture.server.abort();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    fixture._env.push(EnvGuard::set(
        "CCP_CODEX_BASE_URL",
        format!("http://{address}/responses"),
    ));
    fixture.provider = Arc::new(CodexProvider::new());
    let app = Router::new().route("/responses", post(move || {
        let usage = usage.clone();
        async move {
            let mut events = vec![json!({"type":"response.created","response":{"id":"telemetry-response"}})];
            if resolved { events.push(json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"telemetry-item"}})); }
            events.push(json!({"type":"response.output_text.delta","output_index":0,"delta":"answer"}));
            if resolved { events.push(json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"telemetry-item","content":[{"type":"output_text","text":"answer"}]}})); }
            events.push(json!({"type":"response.completed","response":{"id":"telemetry-response","status":"completed","usage":usage}}));
            let body = events.iter().map(|event| format!("data: {event}\n\n")).collect::<String>();
            Response::builder().header("content-type", "text/event-stream").body(Body::from(body)).unwrap()
        }
    }));
    fixture.server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    fixture
}
async fn turn(client: &Arc<CodexHttpClient>, request: &ResponsesRequest, owner: usize) {
    let mut receiver = client
        .stream_codex_http_events(request, &context(owner))
        .await
        .unwrap();
    let mut text = String::new();
    let mut terminal = false;
    while let Some(event) = receiver.recv().await {
        let event = event.unwrap();
        if event["type"] == "response.output_text.delta" {
            text.push_str(event["delta"].as_str().unwrap());
        }
        terminal |= event["type"] == "response.completed";
    }
    assert_eq!(text, "answer");
    assert!(terminal);
}

// A missing or misplaced event hook would collapse event/text/terminal boundaries;
// a synthetic keepalive must never count as the first upstream event.
#[tokio::test]
async fn upstream_timing_distinguishes_control_text_and_completion() {
    let _lock = ENV_LOCK.lock().await;
    let fixture = fixture(true).await;
    let (_, measured) = performance::observe(turn(&fixture.client, &request(64 * 1024), 0)).await;
    let event = measured.first_upstream_event.unwrap();
    let text = measured.first_upstream_text.unwrap();
    let semantic = measured.first_semantic.unwrap();
    let complete = measured.body_complete.unwrap();
    assert!(text > event + Duration::from_millis(5));
    assert_eq!(text, semantic);
    assert!(complete > text + Duration::from_millis(5));
    assert!(measured.finalization.unwrap() >= complete);
    assert!(measured.preparation_start.unwrap() <= measured.preparation_end.unwrap());
    assert!(measured.preparation_end.unwrap() <= measured.response_headers.unwrap());
    assert!(measured.serialized_bytes > 64 * 1024);
    assert_eq!(measured.full_request_clone_bytes, measured.serialized_bytes);
    assert_eq!(measured.attempts, 1);
    assert!(measured.gate_start.is_none());
    assert!(measured.handshake_end.is_none());
    assert!(
        measured.first_downstream_yield.is_none(),
        "receiver consumption is not downstream delivery"
    );
    assert!(measured.body_cancel.is_none());
}

fn messages(bytes: usize) -> MessagesRequest {
    let sentence = "Synthetic history contains words for repeatable token preparation. ";
    let text = sentence.repeat(bytes / sentence.len() + 1)[..bytes].to_string();
    serde_json::from_value(json!({"model":"gpt-6-sol-fast","stream":true,"system":"synthetic instructions","messages":[{"role":"user","content":text}],"tools":[{"name":"fixture_tool","description":"fixture","input_schema":{"type":"object","properties":{}}}],"output_config":{"effort":"high"}})).unwrap()
}
async fn provider_turn(provider: &CodexProvider, body: MessagesRequest, owner: usize) {
    let response = provider.handle_messages(body, context(owner)).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("answer"), "{text}");
    assert!(text.contains("message_stop"), "{text}");
    assert!(!text.contains("event: error"), "{text}");
}

// Include provider translation and token estimation in the long-history check.
#[tokio::test]
async fn provider_long_history_forwards_scripted_output() {
    let _lock = ENV_LOCK.lock().await;
    let fixture = fixture(false).await;
    let (_, measured) =
        performance::observe(provider_turn(&fixture.provider, messages(64 * 1024), 0)).await;
    assert_eq!(measured.attempts, 1);
    assert!(measured.config_parses > 0);
    assert!(measured.config_reads >= measured.config_parses);
}

// Dropping a consumer before any scripted event must not fabricate completion.
#[tokio::test]
async fn production_listener_yields_text_before_terminal_and_observes_cancel() {
    use claude_code_proxy::monitor::{MonitorHandle, RequestStatus};
    use futures_util::StreamExt;
    let _lock = ENV_LOCK.lock().await;
    let fixture = fixture_mode(false, true, false).await;
    let monitor = MonitorHandle::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(claude_code_proxy::server::serve_listener(
        listener,
        Some(monitor.clone()),
        async {
            let _ = shutdown_rx.await;
        },
    ));
    let response = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("http://{address}/v1/messages"))
        .json(&messages(128))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), http::StatusCode::OK);
    let mut stream = response.bytes_stream();
    let mut received = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !String::from_utf8_lossy(&received).contains("text_delta") {
            received.extend_from_slice(&stream.next().await.unwrap().unwrap());
        }
    })
    .await
    .expect("actual downstream TCP must receive text before upstream completion");
    let state = monitor.snapshot();
    assert_eq!(state.active.len(), 1);
    assert!(state.recent.is_empty());
    assert!(state.active[0].delivery.response_ready_at.is_some());
    assert!(state.active[0].delivery.first_body_yield_at.is_some());
    assert!(state.active[0].delivery.body_finished_at.is_none());
    assert!(state.active[0].usage.output_tokens.is_none());
    assert!(!String::from_utf8_lossy(&received).contains("message_stop"));
    drop(stream);
    tokio::time::timeout(Duration::from_secs(2), async {
        while monitor.snapshot().recent.is_empty()
            || !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("downstream cancellation must release the body and upstream");
    let state = monitor.snapshot();
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert!(state.recent[0].delivery.body_cancelled_at.is_some());
    assert!(state.recent[0].delivery.body_finished_at.is_none());
    let _ = shutdown_tx.send(());
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn provider_usage_total_preserves_wire_tokens_and_unknown_breakdowns() {
    use claude_code_proxy::monitor::{EndpointKind, MonitorHandle, UsageProvenance};
    let _lock = ENV_LOCK.lock().await;
    let fixture = telemetry_fixture(json!({"input_tokens":1,"output_tokens":1}), true).await;
    let monitor = MonitorHandle::default();
    let mut ctx = context(77);
    monitor.request_started(
        &ctx.req_id,
        ctx.session_id.clone(),
        None,
        EndpointKind::Messages,
    );
    ctx.monitor = Some(monitor.clone());
    let response = fixture.provider.handle_messages(messages(128), ctx).await;
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("message_stop"));
    let state = monitor.snapshot();
    assert_eq!(state.active[0].input_tokens, Some(1));
    assert_eq!(state.active[0].output_tokens, Some(1));
    assert_eq!(
        state.active[0].usage.output_tokens.unwrap().provenance,
        UsageProvenance::Provider
    );
    assert_eq!(state.active[0].usage.reasoning_tokens, None);
    assert_eq!(state.active[0].streamed_bytes, bytes.len() as u64);
    assert_eq!(
        state.active[0].stream_chunks,
        text.matches("event: ").count() as u64
    );
    assert_eq!(state.sessions[0].input_tokens, 1);
    assert_eq!(state.sessions[0].output_tokens, 1);
    let snapshot: claude_code_proxy::monitor::snapshot::MonitorSnapshot = state.into();
    assert_eq!(
        snapshot.active[0].output_token_basis.as_deref(),
        Some("provider-total")
    );
}

#[tokio::test]
async fn production_listener_rejected_output_retains_provider_usage_without_delivery_success() {
    use claude_code_proxy::monitor::{MonitorHandle, RequestStatus, UsageProvenance};
    let _lock = ENV_LOCK.lock().await;
    let _fixture = telemetry_fixture(
        json!({"output_tokens":300,"output_tokens_details":{"reasoning_tokens":20}}),
        false,
    )
    .await;
    let monitor = MonitorHandle::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(claude_code_proxy::server::serve_listener(
        listener,
        Some(monitor.clone()),
        async {
            let _ = shutdown_rx.await;
        },
    ));
    let response = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("http://{address}/v1/messages"))
        .json(&messages(128))
        .send()
        .await
        .unwrap();
    let text = response.text().await.unwrap();
    assert!(text.contains("event: error"));
    assert!(!text.contains("message_stop"));
    let state = monitor.snapshot();
    let done = &state.recent[0];
    assert_eq!(done.status, RequestStatus::Failed);
    assert_eq!(done.output_tokens, Some(300));
    assert_eq!(
        done.usage.output_tokens.unwrap().provenance,
        UsageProvenance::Provider
    );
    assert_eq!(done.usage.reasoning_tokens, Some(20));
    assert!(done.usage.input_tokens.unwrap().provenance == UsageProvenance::Estimated);
    assert_eq!(state.sessions[0].input_tokens, 0);
    assert_eq!(state.sessions[0].output_tokens, 300);
    assert!(done.delivery.body_finished_at.is_some());
    assert!(done.delivery.body_cancelled_at.is_none());
    let _ = shutdown_tx.send(());
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn cancelled_reader_records_cancel_without_terminal_or_text() {
    let _lock = ENV_LOCK.lock().await;
    let fixture = fixture(true).await;
    let (_, measured) = performance::observe(async {
        let receiver = fixture
            .client
            .stream_codex_http_events(&request(128), &context(0))
            .await
            .unwrap();
        drop(receiver);
        tokio::time::sleep(Duration::from_millis(30)).await;
    })
    .await;
    assert!(measured.body_cancel.is_some());
    assert!(measured.body_complete.is_none());
    assert!(measured.first_upstream_text.is_none());
    assert!(measured.finalization.is_some());
}

// Real HTTP/WS peers drive the production reader until its bounded queue is full.
// The fake clock advances only after that observable barrier.
#[tokio::test]
async fn stalled_consumer_closes_http_and_websocket_upstream_without_terminal() {
    let _lock = ENV_LOCK.lock().await;
    for websocket in [false, true] {
        let fixture = fixture_mode(false, true, websocket).await;
        let mut receiver = if websocket {
            fixture
                .client
                .stream_codex_websocket_events(&request(128), &context(0), None)
                .await
                .unwrap()
        } else {
            fixture
                .client
                .stream_codex_http_events(&request(128), &context(0))
                .await
                .unwrap()
        };
        tokio::time::timeout(QUEUE_FILL_WATCHDOG, async {
            while receiver.len() != 64 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("peer must fill the production queue");
        // Let the ready sender poll its next reservation after filling slot 64.
        tokio::task::yield_now().await;
        tokio::time::pause();
        // Step the clock so both relay tasks and the local socket reactor run
        // between ticks; a single jump can starve delivery of frame 65.
        for _ in 0..299 {
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }
        assert!(
            !receiver.is_closed(),
            "stall budget must remain 300 seconds"
        );
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }
        assert!(
            receiver.is_closed(),
            "stalled consumer must release the reader (websocket={websocket})"
        );
        assert_eq!(
            receiver.len(),
            64,
            "no fake terminal/error pushed into the full queue"
        );
        let mut queued_text_bytes = 0;
        while let Some(event) = receiver.recv().await {
            let event = event.unwrap();
            assert_ne!(event["type"], "response.completed");
            queued_text_bytes += event["delta"].as_str().map_or(0, str::len);
        }
        assert_eq!(
            queued_text_bytes,
            if websocket { 63 * 1024 } else { 62 * 1024 }
        );
        tokio::time::resume();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stall must close the live upstream body/socket");
    }
}

#[tokio::test]
async fn cancelled_full_consumer_closes_http_and_websocket_upstream_promptly() {
    let _lock = ENV_LOCK.lock().await;
    for websocket in [false, true] {
        let fixture = fixture_mode(false, true, websocket).await;
        let receiver = if websocket {
            fixture
                .client
                .stream_codex_websocket_events(&request(128), &context(0), None)
                .await
                .unwrap()
        } else {
            fixture
                .client
                .stream_codex_http_events(&request(128), &context(0))
                .await
                .unwrap()
        };
        tokio::time::timeout(QUEUE_FILL_WATCHDOG, async {
            while receiver.len() != 64 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(2), async {
            while !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("closed downstream must cancel a blocked send promptly");
    }
}

#[tokio::test]
async fn cancelled_response_body_closes_http_and_websocket_upstream() {
    let _lock = ENV_LOCK.lock().await;
    for websocket in [false, true] {
        let fixture = fixture_mode(false, true, websocket).await;
        let response = fixture
            .provider
            .handle_messages(messages(128), context(0))
            .await;
        assert_eq!(
            response.status(),
            http::StatusCode::OK,
            "websocket={websocket}"
        );
        let mut body = response.into_body();
        let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
        assert!(!first.is_empty());
        drop(body);
        tokio::time::timeout(Duration::from_secs(2), async {
            while !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("body cancellation must close upstream while provider delivery is active");
    }
}

#[tokio::test]
async fn stalled_response_body_closes_http_and_websocket_without_success_or_error() {
    let _lock = ENV_LOCK.lock().await;
    for websocket in [false, true] {
        let fixture = fixture_mode(false, true, websocket).await;
        let response = fixture
            .provider
            .handle_messages(messages(128), context(0))
            .await;
        assert_eq!(
            response.status(),
            http::StatusCode::OK,
            "websocket={websocket}"
        );
        let mut body = response.into_body();
        body.frame().await.unwrap().unwrap();
        tokio::time::pause();
        for _ in 0..299 {
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }
        assert!(
            !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
        );
        for _ in 0..200 {
            if fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                break;
            }
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }
        tokio::time::resume();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !fixture
                .upstream_closed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("stalled body must release upstream (websocket={websocket})"));
        let remaining = body.collect().await.unwrap().to_bytes();
        let remaining = String::from_utf8_lossy(&remaining);
        assert!(!remaining.contains("message_stop"));
        assert!(!remaining.contains("event: error"));
    }
}
