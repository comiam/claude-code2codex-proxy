//! Full-translation regression coverage for the compaction effort cap.
//!
//! These tests exercise the serialized Codex request body, not just the
//! helper value table, because the cap interacts with effort overrides and
//! with which reasoning artifacts get requested. Every case mutates process
//! environment variables, so it holds the shared `ENV_LOCK` and restores the
//! previous environment through `EnvGuard`. The test binary runs in its own
//! process, keeping this environment manipulation away from the unit tests.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use claude_code_proxy::MessagesRequest;
use claude_code_proxy::providers::codex::translate::request::{
    TranslateOptions, translate_openai_compatible_request, translate_request,
};
use serde_json::{Value, json};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

/// Clears the environment knobs that feed effort resolution and reasoning
/// summary selection, then points the config dir at an empty directory so a
/// real user config cannot leak into the test.
fn isolated_environment(config_dir: &Path) -> Vec<EnvGuard> {
    let mut guards = vec![
        EnvGuard::unset("CCP_CODEX_EFFORT"),
        EnvGuard::unset("CCP_CODEX_REASONING_SUMMARY"),
        EnvGuard::unset("CCP_COMPACT_EFFORT"),
    ];
    guards.push(EnvGuard::set("CCP_CONFIG_DIR", config_dir));
    guards
}

fn compact_request() -> MessagesRequest {
    serde_json::from_value(json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "summarize"}],
        "system": "You are a helpful AI assistant tasked with summarizing conversations."
    }))
    .unwrap()
}

fn translate_compact() -> Value {
    let opts = TranslateOptions {
        session_id: None,
        service_tier: None,
        model: "gpt-5.5".to_string(),
        use_responses_lite: false,
    };
    let out = translate_request(&compact_request(), opts).unwrap();
    serde_json::to_value(out).unwrap()
}

#[test]
fn omitted_effort_compact_uses_default_low_cap() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let _env = isolated_environment(config.path());

    let wire = translate_compact();

    assert_eq!(
        wire["reasoning"],
        json!({"effort": "low", "summary": "auto"})
    );
    assert_eq!(wire["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn compact_cap_none_overrides_default_without_reasoning_artifacts() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_COMPACT_EFFORT", "none"));

    let wire = translate_compact();

    // `none` must reach the wire to displace the upstream default, but it
    // must not ask for a summary or encrypted continuation content.
    assert_eq!(wire["reasoning"], json!({"effort": "none"}));
    assert!(wire.get("include").is_none(), "unexpected include: {wire}");
}

#[test]
fn compact_cap_off_leaves_omitted_effort_unset() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_COMPACT_EFFORT", "off"));

    let wire = translate_compact();

    assert!(
        wire.get("reasoning").is_none(),
        "unexpected reasoning: {wire}"
    );
    assert!(wire.get("include").is_none(), "unexpected include: {wire}");
}

#[test]
fn global_high_is_lowered_by_default_compact_cap() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_CODEX_EFFORT", "high"));

    let wire = translate_compact();

    assert_eq!(
        wire["reasoning"],
        json!({"effort": "low", "summary": "auto"})
    );
    assert_eq!(wire["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn global_none_survives_compact_cap_without_reasoning_artifacts() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_CODEX_EFFORT", "none"));

    let wire = translate_compact();

    // `none` is at or below the cap, so it is preserved rather than raised.
    assert_eq!(wire["reasoning"], json!({"effort": "none"}));
    assert!(wire.get("include").is_none(), "unexpected include: {wire}");
}

#[test]
fn global_low_is_preserved_under_higher_compact_cap() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_CODEX_EFFORT", "low"));
    env.push(EnvGuard::set("CCP_COMPACT_EFFORT", "medium"));

    let wire = translate_compact();

    assert_eq!(
        wire["reasoning"],
        json!({"effort": "low", "summary": "auto"})
    );
    assert_eq!(wire["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn compact_text_preserves_history_focus_model_tier_and_effort_in_both_lanes() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_CODEX_EFFORT", "high"));
    for lite in [false, true] {
        for compact in [false, true] {
            let req: MessagesRequest = serde_json::from_value(json!({
                "model": "gpt-5.6-sol",
                "system": if compact { "You are a helpful AI assistant tasked with summarizing conversations. Focus on database migration." } else { "You are Claude Code. Focus on database migration." },
                "tools": [{"name":"Read","input_schema":{"type":"object"}}],
                "tool_choice": {"type":"tool","name":"Read"},
                "messages": [
                    {"role":"assistant","content":[{"type":"tool_use","id":"call-old","name":"Read","input":{"path":"old.rs"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call-old","content":"historical result"}]}
                ]
            })).unwrap();
            // The OpenAI-compatible translator retains ordinary tool registration,
            // even when prompt text happens to match the Claude compact detector.
            let openai_wire = serde_json::to_value(
                translate_openai_compatible_request(&req, "gpt-5.6-sol".into(), None).unwrap(),
            )
            .unwrap();
            assert!(openai_wire["tools"].is_array());
            assert_eq!(openai_wire["tool_choice"]["name"], "Read");
            let wire = serde_json::to_value(translate_request(&req, TranslateOptions {
                session_id: None, service_tier: Some(claude_code_proxy::providers::codex::translate::request::ServiceTier::Priority),
                model: "gpt-5.6-sol".into(), use_responses_lite: lite,
            }).unwrap()).unwrap();
            assert_eq!(wire["model"], "gpt-5.6-sol");
            assert_eq!(wire["service_tier"], "priority");
            assert_eq!(
                wire["reasoning"]["effort"],
                if compact { "low" } else { "high" }
            );
            let text = wire.to_string();
            assert!(text.contains("historical result"));
            assert!(text.contains("call-old"));
            assert!(text.contains("Focus on database migration."));
            if compact {
                assert!(wire.get("tools").is_none());
                assert!(!text.contains("additional_tools"));
                assert!(wire.get("tool_choice").is_none());
            } else {
                assert_eq!(wire["tool_choice"]["name"], "Read");
                assert!(if lite {
                    text.contains("additional_tools")
                } else {
                    wire["tools"].is_array()
                });
            }
        }
    }
}

#[test]
fn quoted_compact_prompts_preserve_ordinary_tools_choice_and_effort() {
    let _guard = env_lock();
    let config = tempfile::TempDir::new().unwrap();
    let mut env = isolated_environment(config.path());
    env.push(EnvGuard::set("CCP_CODEX_EFFORT", "high"));
    let prompt = concat!(
        "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n",
        "Your task is to create a detailed summary of the conversation so far."
    );
    let cases = [
        (
            json!("You are Claude Code."),
            json!(format!(
                "Explain this compact prompt: \"{prompt}\" Then inspect src/main.rs with Read."
            )),
        ),
        (
            json!("You are Claude Code."),
            json!(format!(
                "```text\n{prompt}\n```\nThen inspect src/main.rs with Read."
            )),
        ),
        (
            json!("You are Claude Code."),
            json!([
                {"type":"text","text":"Explain this quoted prompt, then inspect src/main.rs with Read:"},
                {"type":"text","text":prompt}
            ]),
        ),
        (
            json!(
                "You are Claude Code. Example prompt to discuss, not execute: \"You are a helpful AI assistant tasked with summarizing conversations.\""
            ),
            json!("Inspect src/main.rs with Read."),
        ),
        (
            json!([
                {"type":"text","text":"You are Claude Code. The following is an example to discuss:"},
                {"type":"text","text":"You are a helpful AI assistant tasked with summarizing conversations."}
            ]),
            json!("Inspect src/main.rs with Read."),
        ),
    ];
    for (system, content) in cases {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model":"gpt-5.6-sol", "system":system,
            "messages":[{"role":"user", "content":content}],
            "tools":[{"name":"Read","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"tool","name":"Read"}
        }))
        .unwrap();
        for lite in [false, true] {
            let wire = serde_json::to_value(
                translate_request(
                    &request,
                    TranslateOptions {
                        model: "gpt-5.6-sol".into(),
                        session_id: None,
                        service_tier: None,
                        use_responses_lite: lite,
                    },
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(wire["tool_choice"]["name"], "Read", "{wire}");
            assert_eq!(wire["reasoning"]["effort"], "high", "{wire}");
            if lite {
                assert!(
                    wire["input"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|item| item["type"] == "additional_tools"
                            && item["tools"][0]["name"] == "Read"),
                    "{wire}"
                );
            } else {
                assert_eq!(wire["tools"][0]["name"], "Read", "{wire}");
            }
        }
    }
}
