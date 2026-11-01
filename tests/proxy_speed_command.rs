//! Opt-in characterization of the installed client's actual skill expansion.
//! Both expansion characterization and real proxy -> loopback Codex integration.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use claude_code_proxy::proxy_speed::{
    ControlError, ProxySpeedCommand, ProxySpeedMode, parse_current_command,
};
use claude_code_proxy::request_identity::{CLAUDE_SESSION_HEADER, ConversationIdentity};
use serde_json::{Value, json};

#[derive(Clone)]
struct Capture {
    headers: HeaderMap,
    body: Value,
}

async fn capture(
    State(captures): State<Arc<Mutex<Vec<Capture>>>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    captures.lock().unwrap().push(Capture {
        headers,
        body: body.clone(),
    });
    let message = json!({
        "id": "msg_local_capture", "type": "message", "role": "assistant",
        "model": body["model"], "content": [{"type": "text", "text": "Local capture complete."}],
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    });
    if body["stream"] != true {
        return Json(message).into_response();
    }
    let mut start = message;
    start["content"] = json!([]);
    start["stop_reason"] = Value::Null;
    let events = [
        (
            "message_start",
            json!({"type": "message_start", "message": start}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Local capture complete."}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 0}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ];
    let sse: String = events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
        .collect();
    ([("content-type", "text/event-stream")], sse).into_response()
}

fn normalize(value: &mut Value, root: &str, session: &str) {
    match value {
        Value::String(text) => {
            *text = text
                .replace(root, "/tmp/ccp-proxy-speed-fixture")
                .replace(session, "11111111-1111-4111-8111-111111111111");
        }
        Value::Array(values) => {
            for value in values {
                normalize(value, root, session);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                normalize(value, root, session);
            }
        }
        _ => {}
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires installed Claude Code; loopback only, isolated temporary profile"]
async fn installed_client_expands_all_proxy_speed_modes() {
    let root = tempfile::tempdir().unwrap();
    let project = root.path().join("project");
    let profile = root.path().join("profile");
    let home = root.path().join("home");
    let skill = project.join(".claude/skills/proxy-speed");
    std::fs::create_dir_all(&skill).unwrap();
    std::fs::create_dir_all(&profile).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/integrations/claude-code/skills/proxy-speed/SKILL.md"
        ),
        skill.join("SKILL.md"),
    )
    .unwrap();

    let captures = Arc::new(Mutex::new(Vec::<Capture>::new()));
    let app = Router::new()
        .route("/v1/messages", post(capture))
        // Also serves as a dead-end proxy for any unexpected non-loopback traffic.
        .fallback(|| async { (StatusCode::BAD_GATEWAY, "local-only test") })
        .with_state(captures.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let binary = std::env::var_os("CCP_CLAUDE_BIN").unwrap_or_else(|| "claude".into());
    let executable_path = std::env::var_os("PATH").unwrap_or_default();
    let mut version = assert_cmd::Command::new(&binary);
    version
        .env_clear()
        .env("PATH", &executable_path)
        .env("HOME", &home)
        .env("CLAUDE_CONFIG_DIR", &profile)
        .current_dir(&project)
        .arg("--version")
        .timeout(Duration::from_secs(10));
    let version_output =
        tokio::task::spawn_blocking(move || version.assert().success().get_output().clone())
            .await
            .unwrap();
    let client_version = String::from_utf8(version_output.stdout)
        .unwrap()
        .trim()
        .to_string();
    eprintln!("client: {client_version}");
    let mut fixtures = Vec::new();
    let mut empty_arguments = None;
    let session = uuid::Uuid::new_v4().to_string();

    for (mode, expected_mode) in [
        ("off", Some(ProxySpeedMode::Off)),
        ("fast", Some(ProxySpeedMode::Fast)),
        ("ultrafast", Some(ProxySpeedMode::Ultrafast)),
        ("", None),
    ] {
        captures.lock().unwrap().clear();
        let prompt = if mode.is_empty() {
            "/proxy-speed".to_string()
        } else {
            format!("/proxy-speed {mode}")
        };
        let mut command = assert_cmd::Command::new(&binary);
        command
            .env_clear()
            .env("PATH", &executable_path)
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", &home)
            .env("CLAUDE_CONFIG_DIR", &profile)
            .env("ANTHROPIC_API_KEY", "dummy-local-capture-key")
            .env("ANTHROPIC_BASE_URL", &base)
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            .env("DISABLE_TELEMETRY", "1")
            .env("DISABLE_ERROR_REPORTING", "1")
            .env("DISABLE_AUTOUPDATER", "1")
            .env("HTTP_PROXY", &base)
            .env("HTTPS_PROXY", &base)
            .env("ALL_PROXY", &base)
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("API_TIMEOUT_MS", "10000")
            .current_dir(&project)
            .args(["--bare", "--add-dir"])
            .arg(&project)
            .args(["--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}"])
            .args(["--tools", "", "--model", "claude-sonnet-4-6"])
            .args([
                "--max-turns",
                "1",
                "--no-session-persistence",
                "--output-format",
                "json",
            ])
            .args(["--session-id", &session, "-p", &prompt])
            .timeout(Duration::from_secs(45));
        let output =
            tokio::task::spawn_blocking(move || command.assert().success().get_output().clone())
                .await
                .unwrap();
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["result"], "Local capture complete.");
        assert_ne!(result["is_error"], true);

        let captured = captures.lock().unwrap();
        assert_eq!(captured.len(), 1, "one Messages request for {mode}");
        let request = &captured[0];
        assert_eq!(
            ConversationIdentity::from_headers(&request.headers),
            Some(ConversationIdentity::Main(session.clone())),
            "canonical main identity for {mode}"
        );
        let parsed_request = serde_json::from_value(request.body.clone()).unwrap();
        let expected_command = match expected_mode {
            Some(mode) => Ok(Some(ProxySpeedCommand {
                session_id: session.clone(),
                mode,
            })),
            None => Err(ControlError::InvalidArguments),
        };
        assert_eq!(
            parse_current_command(&parsed_request, &request.headers),
            expected_command
        );
        let messages = request.body["messages"].as_array().unwrap();
        let texts: Vec<_> = messages
            .iter()
            .filter(|message| message["role"] == "user")
            .flat_map(|message| message["content"].as_array().unwrap())
            .filter_map(|block| block["text"].as_str())
            .collect();
        let expected =
            format!("<ccp-proxy-speed version=\"1\" session=\"{session}\" mode=\"{mode}\"/>");
        let expanded: Vec<_> = texts
            .iter()
            .filter(|text| text.contains("<ccp-proxy-speed"))
            .collect();
        assert_eq!(expanded.len(), 1, "exactly one expanded skill for {mode}");
        assert!(expanded[0].starts_with("Base directory for this skill: "));
        assert!(expanded[0].trim_end().ends_with(&expected), "{expanded:?}");
        assert!(
            !texts
                .iter()
                .any(|text| text.contains("command not available"))
        );

        let mut fixture = json!({
            "mode": mode,
            "headers": {CLAUDE_SESSION_HEADER: session},
            "request": {
                "model": request.body["model"], "max_tokens": request.body["max_tokens"],
                "stream": request.body["stream"], "messages": request.body["messages"]
            }
        });
        normalize(&mut fixture, root.path().to_str().unwrap(), &session);
        if mode.is_empty() {
            empty_arguments = Some(fixture);
            eprintln!("empty argument: expanded once; local usage validation error confirmed");
        } else {
            fixtures.push(fixture);
            eprintln!("{mode}: expanded once; header/session match; local response accepted");
        }
    }
    server.abort();
    if let Some(path) = std::env::var_os("CCP_PROXY_SPEED_FIXTURE") {
        let fixture = json!({"client_version": client_version, "cases": fixtures, "empty_arguments": empty_arguments});
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&fixture).unwrap()),
        )
        .unwrap();
    }
}

#[path = "support/proxy_speed.rs"]
mod runtime_support;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires installed Claude Code; real isolated proxy with loopback Codex upstream"]
async fn installed_client_runs_local_commands_then_generates_literal_tiers() {
    run_installed_client_modes(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires installed Claude Code; normal client context with isolated local proxy"]
async fn normal_client_runs_local_commands_then_generates_literal_tiers() {
    run_installed_client_modes(false).await;
}

async fn run_installed_client_modes(bare: bool) {
    let captures = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = captures.clone();
    let stub = Router::new()
        .route(
            "/responses",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    captured.lock().unwrap().push(body);
                    (
                        [("content-type", "text/event-stream")],
                        runtime_support::completion("resp_installed", "Local upstream generation."),
                    )
                }
            }),
        )
        .fallback(|| async { StatusCode::BAD_GATEWAY });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, stub).await.unwrap() });
    let proxy = runtime_support::Proxy::start(&upstream, Some("ultrafast"), "http", true).await;
    let root = tempfile::tempdir().unwrap();
    let project = root.path().join("project");
    let profile = root.path().join("profile");
    let home = root.path().join("home");
    let skill = project.join(".claude/skills/proxy-speed");
    for path in [&skill, &profile, &home] {
        std::fs::create_dir_all(path).unwrap();
    }
    if !bare {
        std::fs::create_dir_all(profile.join("rules")).unwrap();
        std::fs::write(
            profile.join("rules/local-test.md"),
            "Keep this isolated compatibility test local. Do not invoke tools.\n",
        )
        .unwrap();
        std::fs::write(
            project.join("CLAUDE.md"),
            "This project verifies local proxy speed commands.\n",
        )
        .unwrap();
    }
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/integrations/claude-code/skills/proxy-speed/SKILL.md"
        ),
        skill.join("SKILL.md"),
    )
    .unwrap();
    let session = uuid::Uuid::new_v4().to_string();
    let binary = std::env::var_os("CCP_CLAUDE_BIN").unwrap_or_else(|| "claude".into());
    let executable_path = std::env::var_os("PATH").unwrap_or_default();
    let model = if bare { "gpt-5.5" } else { "gpt-6.1-sol" };
    for (mode, tier) in [
        ("off", None),
        ("fast", Some("fast")),
        ("ultrafast", Some("ultrafast")),
    ] {
        for is_command in [true, false] {
            let prompt = if is_command {
                format!("/proxy-speed {mode}")
            } else {
                "Reply with the local test response.".to_string()
            };
            let mut command = assert_cmd::Command::new(&binary);
            if bare {
                command.arg("--bare");
            }
            command
                .env_clear()
                .env("PATH", &executable_path)
                .env("HOME", &home)
                .env("XDG_CONFIG_HOME", &home)
                .env("CLAUDE_CONFIG_DIR", &profile)
                .env("ANTHROPIC_API_KEY", "dummy-local-only-key")
                .env("ANTHROPIC_BASE_URL", &proxy.base)
                .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
                .env("DISABLE_TELEMETRY", "1")
                .env("DISABLE_ERROR_REPORTING", "1")
                .env("DISABLE_AUTOUPDATER", "1")
                .env("HTTP_PROXY", &proxy.base)
                .env("HTTPS_PROXY", &proxy.base)
                .env("ALL_PROXY", &proxy.base)
                .env("NO_PROXY", "127.0.0.1,localhost")
                .env("API_TIMEOUT_MS", "10000")
                .current_dir(&project)
                .arg("--add-dir")
                .arg(&project)
                .args([
                    "--strict-mcp-config",
                    "--mcp-config",
                    "{\"mcpServers\":{}}",
                    "--tools",
                    "",
                    "--model",
                    model,
                ])
                .args([
                    "--max-turns",
                    "1",
                    "--no-session-persistence",
                    "--output-format",
                    "json",
                    "--session-id",
                    &session,
                    "-p",
                    &prompt,
                ])
                .timeout(Duration::from_secs(45));
            let before = captures.lock().unwrap().len();
            let output = tokio::task::spawn_blocking(move || {
                command.assert().success().get_output().clone()
            })
            .await
            .unwrap();
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_ne!(result["is_error"], true, "{result}");
            if is_command {
                assert!(
                    result["result"]
                        .as_str()
                        .unwrap()
                        .contains(&format!("Proxy speed: {mode}")),
                    "{result}"
                );
                assert_eq!(
                    captures.lock().unwrap().len(),
                    before,
                    "installed command must stay local"
                );
                // First acknowledgment succeeded even without any proxy auth file.
                if mode == "off" {
                    proxy.write_synthetic_auth();
                }
            } else {
                assert_eq!(result["result"], "Local upstream generation.");
                let captures = captures.lock().unwrap();
                assert_eq!(captures.len(), before + 1);
                let body = captures.last().unwrap();
                assert_eq!(body.get("service_tier").and_then(Value::as_str), tier);
                assert_eq!(body["model"], model);
                assert!(!body.to_string().contains("ccp-proxy-speed"));
            }
        }
    }
    eprintln!(
        "installed client ({}; {model}): 3 local acknowledgments, 3 Codex generations: omitted -> fast -> ultrafast",
        if bare { "bare" } else { "normal" }
    );
    server.abort();
}

struct PersistentTestClient {
    child: std::process::Child,
    events: std::sync::mpsc::Receiver<Value>,
    compactions: usize,
}

impl PersistentTestClient {
    fn start(mut command: std::process::Command) -> Self {
        use std::io::BufRead;
        use std::process::Stdio;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, events) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(event) = serde_json::from_str::<Value>(&line)
                    && sender.send(event).is_err()
                {
                    break;
                }
            }
        });
        Self {
            child,
            events,
            compactions: 0,
        }
    }

    fn prompt(&mut self, text: &str) -> Value {
        let result = self.prompt_result(text);
        assert_ne!(result["is_error"], true, "{result}");
        result
    }

    fn prompt_result(&mut self, text: &str) -> Value {
        use std::io::Write;
        writeln!(
            self.child.stdin.as_mut().unwrap(),
            "{}",
            json!({
                "type":"user", "message":{"role":"user","content":text},
                "parent_tool_use_id":null, "priority":"later"
            })
        )
        .unwrap();
        self.child.stdin.as_mut().unwrap().flush().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let event = self
                .events
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .expect("persistent local client did not finish its turn");
            if event["type"] == "system" && event["subtype"] == "compact_boundary" {
                self.compactions += 1;
            }
            if event["type"] == "result" {
                return event;
            }
        }
    }
}

impl Drop for PersistentTestClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn persistent_client_command(
    project: &std::path::Path,
    profile: &std::path::Path,
    home: &std::path::Path,
    base: &str,
    session: &str,
) -> std::process::Command {
    let binary = std::env::var_os("CCP_CLAUDE_BIN").unwrap_or_else(|| "claude".into());
    let mut command = std::process::Command::new(binary);
    command
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home)
        .env("CLAUDE_CONFIG_DIR", profile)
        .env("ANTHROPIC_API_KEY", "dummy-local-only-key")
        .env("ANTHROPIC_BASE_URL", base)
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_ERROR_REPORTING", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env("HTTP_PROXY", base)
        .env("HTTPS_PROXY", base)
        .env("ALL_PROXY", base)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("API_TIMEOUT_MS", "10000")
        .current_dir(project)
        .args([
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ])
        .arg("--add-dir")
        .arg(project)
        .args([
            "--strict-mcp-config",
            "--mcp-config",
            r#"{"mcpServers":{}}"#,
            "--tools",
            "",
            "--model",
            "gpt-6.1-sol",
            "--no-session-persistence",
            "--session-id",
            session,
        ]);
    command
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires installed Claude Code; persistent session with real compact and local-only upstream"]
async fn first_speed_command_after_compact_stays_local_and_session_scoped() {
    let captures = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = captures.clone();
    let app = Router::new()
        .route(
            "/responses",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    let index = {
                        let mut rows = captured.lock().unwrap();
                        rows.push(body);
                        rows.len()
                    };
                    let text = if index == 1 {
                        "Recorded local conversation context.\n".repeat(200)
                    } else if index == 2 {
                        "Summary: continue testing local proxy-speed commands in this conversation."
                            .into()
                    } else {
                        "Local post-compact generation.".into()
                    };
                    (
                        [("content-type", "text/event-stream")],
                        runtime_support::completion(&format!("resp_persistent_{index}"), &text),
                    )
                }
            }),
        )
        .fallback(|| async { StatusCode::BAD_GATEWAY });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let proxy = runtime_support::Proxy::start(&upstream, None, "http", false).await;
    tokio::task::spawn_blocking(move || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let profile = root.path().join("profile");
        let home = root.path().join("home");
        let skill = project.join(".claude/skills/proxy-speed");
        for path in [&skill, &profile.join("rules"), &home] {
            std::fs::create_dir_all(path).unwrap();
        }
        std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"),
            "/integrations/claude-code/skills/proxy-speed/SKILL.md"), skill.join("SKILL.md")).unwrap();
        std::fs::write(profile.join("rules/local-test.md"),
            "Keep this compaction compatibility test local. Do not invoke tools.\n").unwrap();
        let session = uuid::Uuid::new_v4().to_string();
        let command = persistent_client_command(&project, &profile, &home, &proxy.base, &session);
        let mut client = PersistentTestClient::start(command);
        let initial = client.prompt("/proxy-speed fast");
        assert_eq!(initial["session_id"], session);
        assert!(initial["result"].as_str().unwrap().contains("Proxy speed: fast"));
        assert!(captures.lock().unwrap().is_empty());
        proxy.write_synthetic_auth();
        client.prompt("Prepare local history for compaction.");
        assert_eq!(captures.lock().unwrap().last().unwrap()["service_tier"], "fast");
        let before_refresh = captures.lock().unwrap().len();
        let refreshed = client.prompt("/proxy-speed fast");
        assert!(refreshed["result"].as_str().unwrap().contains("Proxy speed: fast"));
        assert_eq!(captures.lock().unwrap().len(), before_refresh);
        let compact = client.prompt_result("/compact Запомни результаты 2го этапа");
        assert_ne!(compact["is_error"], true,
            "compact immediately after a local control must not parse its old envelope: {compact}");
        assert_eq!(compact["session_id"], session);
        assert_eq!(client.compactions, 1,
            "a real focused compact must complete after a local control: {compact}");

        assert!(captures.lock().unwrap().last().unwrap()["input"].to_string()
            .contains("Запомни результаты 2го этапа"),
            "the compact focus must reach the upstream unchanged");

        // These HTTP probes do not add a message to the persistent CLI. Its first
        // prompt after /compact remains the speed command below.
        let http = reqwest::blocking::Client::builder().no_proxy().build().unwrap();
        for (id, expected) in [(session.clone(), Some("fast")),
            (uuid::Uuid::new_v4().to_string(), None)] {
            let response = http.post(format!("{}/v1/messages", proxy.base))
                .header(CLAUDE_SESSION_HEADER, &id)
                .header("x-claude-code-agent-id", "compact-state-probe")
                .json(&json!({"model":"gpt-6.1-sol","stream":false,"max_tokens":64,
                    "messages":[{"role":"user","content":"Probe the stored session preference."}]}))
                .send().unwrap();
            assert!(response.status().is_success());
            let _: Value = response.json().unwrap();
            assert_eq!(captures.lock().unwrap().last().unwrap()
                .get("service_tier").and_then(Value::as_str), expected);
        }
        let before = captures.lock().unwrap().len();
        let switched = client.prompt("/proxy-speed ultrafast");
        assert_eq!(switched["session_id"], session);
        assert!(switched["result"].as_str().unwrap().contains("Proxy speed: ultrafast"));
        assert_eq!(captures.lock().unwrap().len(), before,
            "the first command after compact must not enter generation");
        for prompt in ["Continue after compact.", "Continue once more."] {
            let result = client.prompt(prompt);
            assert_eq!(result["session_id"], session);
            let rows = captures.lock().unwrap();
            let wire = rows.last().unwrap();
            assert_eq!(wire["model"], "gpt-6.1-sol");
            assert_eq!(wire["service_tier"], "ultrafast");
            assert_eq!(wire["prompt_cache_key"], session);
        }
        let before = captures.lock().unwrap().len();
        let off = client.prompt("/proxy-speed off");
        assert!(off["result"].as_str().unwrap().contains("Proxy speed: off"));
        assert_eq!(captures.lock().unwrap().len(), before);
        client.prompt("Continue at standard speed.");
        assert!(captures.lock().unwrap().last().unwrap().get("service_tier").is_none());
        eprintln!("persistent client: focused compact directly after a local control completed; same session kept fast; first post-compact ultrafast stayed local; other session unchanged; ultrafast -> off wire verified");
    }).await.unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires installed Claude Code; persistent local API-error recovery"]
async fn speed_command_recovers_after_api_error_with_pending_user_input() {
    let captures = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured = captures.clone();
    let app = Router::new()
        .route(
            "/responses",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    let rejected = body.get("service_tier").and_then(Value::as_str) == Some("fast");
                    captured.lock().unwrap().push(body);
                    if rejected {
                        return (StatusCode::BAD_REQUEST, Json(json!({"error":{
                    "type":"invalid_request_error", "message":"Unsupported service_tier: fast"
                }}))).into_response();
                    }
                    (
                        [("content-type", "text/event-stream")],
                        runtime_support::completion(
                            "resp_recovered",
                            "Local recovered generation.",
                        ),
                    )
                        .into_response()
                }
            }),
        )
        .fallback(|| async { StatusCode::BAD_GATEWAY });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let proxy = runtime_support::Proxy::start(&upstream, None, "http", false).await;
    tokio::task::spawn_blocking(move || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let profile = root.path().join("profile");
        let home = root.path().join("home");
        let skill = project.join(".claude/skills/proxy-speed");
        for path in [&skill, &profile.join("rules"), &home] {
            std::fs::create_dir_all(path).unwrap();
        }
        std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"),
            "/integrations/claude-code/skills/proxy-speed/SKILL.md"), skill.join("SKILL.md")).unwrap();
        std::fs::write(profile.join("rules/local-test.md"),
            "Keep this API-error recovery test local. Do not invoke tools.\n").unwrap();
        let session = uuid::Uuid::new_v4().to_string();
        let command = persistent_client_command(&project, &profile, &home, &proxy.base, &session);
        let mut client = PersistentTestClient::start(command);
        for (mode, expected, change_model) in [
            ("ultrafast", Some("ultrafast"), true),
            ("off", None, false),
        ] {
            let before = captures.lock().unwrap().len();
            let fast = client.prompt("/proxy-speed fast");
            assert!(fast["result"].as_str().unwrap().contains("Proxy speed: fast"));
            assert_eq!(captures.lock().unwrap().len(), before);
            proxy.write_synthetic_auth();
            let failed = client.prompt_result("Cause the local fast error");
            assert_eq!(failed["is_error"], true, "the test must really fail the generation");
            assert!(failed["result"].as_str().unwrap().contains("Unsupported service_tier: fast"));
            assert_eq!(captures.lock().unwrap().last().unwrap()["service_tier"], "fast");
            if change_model {
                let _ = client.prompt_result("/model gpt-6.1-sol");
            }
            let before = captures.lock().unwrap().len();
            let switched = client.prompt_result(&format!("/proxy-speed {mode}"));
            assert_ne!(switched["is_error"], true,
                "a control must not retry the failed generation: {switched}");
            assert_eq!(switched["session_id"], session);
            assert!(switched["result"].as_str().unwrap().contains(&format!("Proxy speed: {mode}")));
            assert_eq!(captures.lock().unwrap().len(), before,
                "recovery control must remain local despite pending user text");
            let response = client.prompt("Continue after changing the proxy speed.");
            assert_eq!(response["result"], "Local recovered generation.");
            let rows = captures.lock().unwrap();
            let wire = rows.last().unwrap();
            assert_eq!(wire.get("service_tier").and_then(Value::as_str), expected);
            assert_eq!(wire["model"], "gpt-6.1-sol");
        }
        eprintln!("persistent client: rejected fast -> model switch -> local ultrafast; rejected fast -> local off; subsequent wire values verified");
    }).await.unwrap();
    server.abort();
}
