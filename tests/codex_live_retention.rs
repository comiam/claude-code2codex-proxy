//! Steady retained allocations, not cumulative allocations or process RSS.
//! This allocator belongs only to this test binary; frozen benchmarks are unchanged.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};

use claude_code_proxy::auth::{KeychainFileAuthStore, SystemKeychain};
use claude_code_proxy::providers::codex::{
    CodexProvider,
    auth::{manager::CodexAuthManager, token_store::CodexTokenStore},
    client::CodexHttpClient,
    translate::request::ResponsesRequest,
};
use claude_code_proxy::{MessagesRequest, Provider, RequestContext};
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::Message;

struct CountingAllocator;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
fn allocated(bytes: isize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size() as isize);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size() as isize);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(pointer, layout, size) };
        if !new.is_null() {
            allocated(size as isize - layout.size() as isize);
        }
        new
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const DELTA: &str = "abcdefghijklmnop";
const CHECKPOINT: usize = 100_000;
// Accommodates the fixed 64-slot channels, HTTP/Tungstenite buffers and one
// current frame on either side. Old full-answer journals grow by many MiB.
const TRANSIENT_MARGIN: isize = 512 * 1024;

async fn checkpoint(count: usize, first: &mut Option<isize>, ack: &tokio::sync::mpsc::Sender<()>) {
    if count != CHECKPOINT && count != 2 * CHECKPOINT {
        return;
    }
    tokio::task::yield_now().await;
    let live = LIVE.load(Ordering::Relaxed);
    if let Some(initial) = *first {
        let peak = PEAK.load(Ordering::Relaxed);
        eprintln!("steady allocations: 100k={initial}, 200k={live}, interval_peak={peak}");
        assert!(
            live - initial < TRANSIENT_MARGIN,
            "retained output grew by {} bytes",
            live - initial
        );
        assert!(
            peak - initial < TRANSIENT_MARGIN,
            "retained interval peak grew by {} bytes",
            peak - initial
        );
    } else {
        *first = Some(live);
        PEAK.store(live, Ordering::Relaxed);
    }
    ack.send(()).await.unwrap();
}

#[tokio::test]
async fn live_reader_and_provider_retain_no_answer_journals() {
    // One current-thread test exclusively owns this binary's environment and
    // allocator samples. Auth, config, capture and monitor are all synthetic/off.
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("codex")).unwrap();
    std::fs::write(
        directory.path().join("config.json"),
        r#"{"log":{"verbose":false,"stderr":false}}"#,
    )
    .unwrap();
    let auth_file = directory.path().join("codex/auth.json");
    std::fs::write(
        &auth_file,
        r#"{"access":"fixture","refresh":"","expires":4102444800000,"accountId":"fixture"}"#,
    )
    .unwrap();
    for key in [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "CCP_CODEX_MODEL",
        "CCP_AUTO_REVIEW_MODEL",
    ] {
        unsafe {
            std::env::remove_var(key);
        }
    }
    for key in ["CCP_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_STATE_HOME"] {
        unsafe {
            std::env::set_var(key, directory.path());
        }
    }
    for key in [
        "CCP_CODEX_PREVIOUS_RESPONSE_ID",
        "CCP_CODEX_SERVER_COMPACTION",
        "CCP_CODEX_WEBSOCKET_REUSE",
    ] {
        unsafe {
            std::env::set_var(key, "0");
        }
    }
    for websocket in [false, true] {
        for provider in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            unsafe {
                std::env::set_var(
                    "CCP_CODEX_TRANSPORT",
                    if websocket { "websocket" } else { "http" },
                );
                std::env::set_var("CCP_CODEX_BASE_URL", format!("http://{address}/responses"));
            }
            let (ack, mut release) = tokio::sync::mpsc::channel(1);
            let server = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let mut http_socket = Some(socket);
                let mut ws = if websocket {
                    let mut ws = tokio_tungstenite::accept_async(http_socket.take().unwrap())
                        .await
                        .unwrap();
                    assert!(matches!(ws.next().await, Some(Ok(Message::Text(_)))));
                    Some(ws)
                } else {
                    let socket = http_socket.as_mut().unwrap();
                    let mut request = vec![0_u8; 8192];
                    assert!(socket.read(&mut request).await.unwrap() > 0);
                    drop(request);
                    socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await.unwrap();
                    None
                };
                // Keep one fixed event, never a complete response String/Vec.
                let delta =
                    json!({"type":"response.output_text.delta","output_index":0,"delta":DELTA})
                        .to_string();
                for index in 0..=2 * CHECKPOINT + 2 {
                    let event = match index {
                        0 => json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}).to_string(),
                        n if n <= 2 * CHECKPOINT => delta.clone(),
                        n if n == 2 * CHECKPOINT + 1 => json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}).to_string(),
                        _ => json!({"type":"response.completed","response":{"id":"r","status":"completed"}}).to_string(),
                    };
                    if let Some(ws) = ws.as_mut() {
                        ws.send(Message::Text(event)).await.unwrap();
                    } else {
                        let frame = format!("data: {event}\n\n");
                        let socket = http_socket.as_mut().unwrap();
                        socket
                            .write_all(format!("{:x}\r\n", frame.len()).as_bytes())
                            .await
                            .unwrap();
                        socket.write_all(frame.as_bytes()).await.unwrap();
                        socket.write_all(b"\r\n").await.unwrap();
                    }
                    if index == CHECKPOINT || index == 2 * CHECKPOINT {
                        release.recv().await.unwrap();
                    }
                }
                if let Some(socket) = http_socket.as_mut() {
                    socket.write_all(b"0\r\n\r\n").await.unwrap();
                }
            });
            let context = RequestContext {
                req_id: "retention".into(),
                session_id: None,
                session_seq: None,
                provider: "codex".into(),
                traffic: None,
                monitor: None,
            };
            let mut count = 0;
            let mut first = None;
            eprintln!("retention path websocket={websocket}, provider={provider}");
            if provider {
                let messages: MessagesRequest = serde_json::from_value(json!({"model":"gpt-6-sol","stream":true,"messages":[{"role":"user","content":"hello"}]})).unwrap();
                let response = CodexProvider::new()
                    .handle_messages(messages, context)
                    .await;
                assert_eq!(response.status(), http::StatusCode::OK);
                let mut body = response.into_body();
                let mut terminal = false;
                while let Some(frame) = body.frame().await {
                    let data = frame.unwrap().into_data().unwrap();
                    for event in claude_code_proxy::anthropic::sse::parse_sse_events(&data) {
                        let value: serde_json::Value = serde_json::from_str(&event.data).unwrap();
                        if value
                            .pointer("/delta/type")
                            .and_then(|value| value.as_str())
                            == Some("text_delta")
                        {
                            assert_eq!(value["delta"]["text"], DELTA);
                            count += 1;
                            checkpoint(count, &mut first, &ack).await;
                        }
                        terminal |= value["type"] == "message_stop";
                        assert_ne!(value["type"], "error");
                    }
                }
                assert!(terminal);
            } else {
                let store = KeychainFileAuthStore::new(
                    auth_file.to_string_lossy().into_owned(),
                    directory
                        .path()
                        .join("absent")
                        .to_string_lossy()
                        .into_owned(),
                    "fixture",
                    "fixture",
                    false,
                    SystemKeychain,
                );
                let client = Arc::new(CodexHttpClient::new_with_client(
                    reqwest::Client::builder().no_proxy().build().unwrap(),
                    CodexAuthManager::new(CodexTokenStore::new(store)),
                    format!("http://{address}/responses"),
                ));
                let request: ResponsesRequest = serde_json::from_value(json!({"model":"gpt-6-sol","input":[],"store":false,"stream":true,"parallel_tool_calls":false,"text":{}})).unwrap();
                let mut events = if websocket {
                    client
                        .stream_codex_websocket_events(&request, &context, None)
                        .await
                        .unwrap()
                } else {
                    client
                        .stream_codex_http_events(&request, &context)
                        .await
                        .unwrap()
                };
                let mut terminal = false;
                while let Some(event) = events.recv().await {
                    let value = event.unwrap();
                    if value["type"] == "response.output_text.delta" {
                        assert_eq!(value["delta"], DELTA);
                        count += 1;
                        checkpoint(count, &mut first, &ack).await;
                    }
                    terminal |= value["type"] == "response.completed";
                }
                assert!(terminal);
            }
            assert_eq!(count, 2 * CHECKPOINT);
            server.await.unwrap();
        }
    }
}
