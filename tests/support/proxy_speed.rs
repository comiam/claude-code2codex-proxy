//! Isolated real proxy subprocess for loopback-only integration tests.
use serde_json::json;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

pub struct Proxy {
    pub root: tempfile::TempDir,
    pub base: String,
    child: Child,
}

impl Proxy {
    pub async fn start(
        upstream: &str,
        tier: Option<&str>,
        transport: &str,
        continuation: bool,
    ) -> Self {
        let root = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut command = Command::new(assert_cmd::cargo::cargo_bin!("claude-code-proxy"));
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", root.path())
            .env("XDG_CONFIG_HOME", root.path())
            .env("XDG_STATE_HOME", root.path())
            .env("CCP_CONFIG_DIR", root.path())
            .env("CCP_ALIAS_PROVIDER", "codex")
            .env("CCP_BIND_ADDRESS", "127.0.0.1")
            .env("CCP_LOG_STDERR", "1")
            .env("CCP_CODEX_BASE_URL", upstream)
            .env("CCP_CODEX_TRANSPORT", transport)
            .env("CCP_CODEX_SERVER_COMPACTION", "0")
            .env(
                "CCP_CODEX_PREVIOUS_RESPONSE_ID",
                if continuation { "1" } else { "0" },
            )
            // Unexpected external traffic has only a local dead-end proxy.
            .env("HTTP_PROXY", upstream)
            .env("HTTPS_PROXY", upstream)
            .env("ALL_PROXY", upstream)
            .env("NO_PROXY", "127.0.0.1,localhost")
            .args(["serve", "--no-monitor", "--port", &port.to_string()])
            .current_dir(root.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(root.path().join("proxy.log")).unwrap());
        if let Some(tier) = tier {
            command.env("CCP_CODEX_SERVICE_TIER", tier);
        }
        let child = command.spawn().unwrap();
        let mut proxy = Self {
            root,
            base: format!("http://127.0.0.1:{port}"),
            child,
        };
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                assert!(
                    proxy.child.try_wait().unwrap().is_none(),
                    "proxy exited: {}",
                    std::fs::read_to_string(proxy.root.path().join("proxy.log")).unwrap()
                );
                if let Ok(response) = client.get(format!("{}/healthz", proxy.base)).send().await
                    && response.status().is_success()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("local proxy startup timeout");
        proxy
    }

    pub fn write_synthetic_auth(&self) {
        let directory = self.root.path().join("codex");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("auth.json"), json!({
            "access":"local-test-only", "refresh":"", "expires":4102444800000u64, "accountId":"test-account"
        }).to_string()).unwrap();
    }
}

impl Drop for Proxy {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn completion(id: &str, text: &str) -> String {
    [
        json!({"type":"response.created","response":{"id":id,"model":"gpt-5.5"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item-1","role":"assistant","content":[]}}),
        json!({"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}),
        json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":text}),
        json!({"type":"response.output_text.done","output_index":0,"content_index":0,"text":text}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item-1","role":"assistant","content":[{"type":"output_text","text":text}]}}),
        json!({"type":"response.completed","response":{"id":id,"status":"completed","usage":{"input_tokens":4,"output_tokens":2}}}),
    ].iter().map(|event|format!("data: {event}\n\n")).collect()
}
