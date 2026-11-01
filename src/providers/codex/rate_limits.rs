//! Latest Codex account quota snapshot captured from `codex.rate_limits`.
//! The upstream payload is intentionally kept verbatim.

use serde_json::{Value, json};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

static LATEST: RwLock<Option<Value>> = RwLock::new(None);

#[cfg(test)]
thread_local! {
    pub(super) static TEST_RECORD_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub fn record_event(payload: &Value) -> bool {
    if payload.get("type").and_then(Value::as_str) != Some("codex.rate_limits") {
        return false;
    }

    let Some(rate_limits) = payload.get("rate_limits") else {
        return false;
    };

    #[cfg(test)]
    TEST_RECORD_CALLS.with(|calls| calls.set(calls.get() + 1));

    let captured_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs());

    let snapshot = json!({
        "rate_limits": rate_limits.clone(),
        "credits": payload.get("credits").cloned().unwrap_or(Value::Null),
        "captured_at_unix": captured_at_unix,
    });

    match LATEST.write() {
        Ok(mut guard) => {
            *guard = Some(snapshot);
            true
        }
        Err(_) => false,
    }
}

pub fn latest() -> Option<Value> {
    LATEST.read().ok().and_then(|guard| guard.clone())
}

/// Project the latest Codex quota snapshot into the response headers that
/// Claude Code normally receives from Anthropic subscription responses.
///
/// `used_percent` from Codex is 0..100; Anthropic utilization is 0..1.
pub fn apply_anthropic_headers(headers: &mut http::HeaderMap) {
    let Some(snapshot) = latest() else {
        return;
    };
    let Some(rate_limits) = snapshot.get("rate_limits") else {
        return;
    };

    fn set(headers: &mut http::HeaderMap, name: &'static str, value: String) {
        if let Ok(value) = http::HeaderValue::from_str(&value) {
            headers
                .entry(http::header::HeaderName::from_static(name))
                .or_insert(value);
        }
    }

    fn project_window(headers: &mut http::HeaderMap, window: Option<&Value>, prefix: &'static str) {
        let Some(window) = window else {
            return;
        };

        if let Some(used) = window.get("used_percent").and_then(Value::as_f64) {
            let utilization = (used / 100.0).clamp(0.0, 1.0);
            set(
                headers,
                match prefix {
                    "5h" => "anthropic-ratelimit-unified-5h-utilization",
                    "7d" => "anthropic-ratelimit-unified-7d-utilization",
                    _ => return,
                },
                utilization.to_string(),
            );
        }

        if let Some(reset_at) = window.get("reset_at").and_then(Value::as_u64) {
            set(
                headers,
                match prefix {
                    "5h" => "anthropic-ratelimit-unified-5h-reset",
                    "7d" => "anthropic-ratelimit-unified-7d-reset",
                    _ => return,
                },
                reset_at.to_string(),
            );
        }

        let allowed = window
            .get("allowed")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        set(
            headers,
            match prefix {
                "5h" => "anthropic-ratelimit-unified-5h-status",
                "7d" => "anthropic-ratelimit-unified-7d-status",
                _ => return,
            },
            if allowed { "allowed" } else { "rejected" }.to_string(),
        );
    }

    project_window(headers, rate_limits.get("primary"), "5h");
    project_window(headers, rate_limits.get("secondary"), "7d");

    let allowed = rate_limits
        .get("allowed")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    set(
        headers,
        "anthropic-ratelimit-unified-status",
        if allowed { "allowed" } else { "rejected" }.to_string(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_snapshot_preserves_authoritative_usage_rejection() {
        record_event(&json!({"type":"codex.rate_limits","rate_limits":{
            "allowed":true,"primary":{"allowed":true,"used_percent":25,"reset_at":100}
        }}));
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "anthropic-ratelimit-unified-status",
            "rejected".parse().unwrap(),
        );
        headers.insert("anthropic-ratelimit-unified-reset", "900".parse().unwrap());
        headers.insert(
            "anthropic-ratelimit-unified-5h-reset",
            "900".parse().unwrap(),
        );
        headers.insert("x-should-retry", "false".parse().unwrap());
        apply_anthropic_headers(&mut headers);
        assert_eq!(headers["anthropic-ratelimit-unified-status"], "rejected");
        assert_eq!(headers["anthropic-ratelimit-unified-reset"], "900");
        assert_eq!(headers["anthropic-ratelimit-unified-5h-reset"], "900");
        assert_eq!(headers["x-should-retry"], "false");
        assert!(headers.contains_key("anthropic-ratelimit-unified-5h-utilization"));
    }
}
