use crate::config;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) mod writer;
pub use writer::{
    DiagnosticsSnapshot, diagnostics_snapshot, flush_diagnostics, shutdown_diagnostics,
};

struct LogPolicy {
    verbose: bool,
    stderr: bool,
    path: std::path::PathBuf,
}
static FALLBACK_POLICY: std::sync::OnceLock<LogPolicy> = std::sync::OnceLock::new();

fn fallback_policy() -> &'static LogPolicy {
    FALLBACK_POLICY.get_or_init(|| {
        let policy = config::load_config();
        LogPolicy {
            verbose: policy.log_verbose,
            stderr: policy.log_stderr,
            path: log_file(),
        }
    })
}

/// Initialize the small unscoped logging policy before serving. Request records
/// continue to use their own immutable settings snapshot.
pub fn start_diagnostics() -> DiagnosticsGuard {
    let _ = fallback_policy();
    let _ = writer::global();
    DiagnosticsGuard
}
pub struct DiagnosticsGuard;
impl Drop for DiagnosticsGuard {
    fn drop(&mut self) {
        shutdown_diagnostics();
    }
}

pub const MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;

static STDERR_SUPPRESSION_DEPTH: AtomicUsize = AtomicUsize::new(0);

pub const REDACT_KEYS: [&str; 15] = [
    "authorization",
    "proxy-authorization",
    "access",
    "access_token",
    "refresh",
    "refresh_token",
    "id_token",
    "code",
    "code_verifier",
    "chatgpt-account-id",
    "cookie",
    "set-cookie",
    "x-api-key",
    "apikey",
    "api_key",
];

pub fn log_file() -> std::path::PathBuf {
    config::request_log_file()
}

#[must_use]
pub struct StderrSuppressionGuard;

impl Drop for StderrSuppressionGuard {
    fn drop(&mut self) {
        STDERR_SUPPRESSION_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn suppress_stderr() -> StderrSuppressionGuard {
    STDERR_SUPPRESSION_DEPTH.fetch_add(1, Ordering::Relaxed);
    StderrSuppressionGuard
}

fn stderr_suppressed() -> bool {
    STDERR_SUPPRESSION_DEPTH.load(Ordering::Relaxed) > 0
}

#[cfg(test)]
fn should_mirror_to_stderr(level: &str) -> bool {
    !stderr_suppressed() && (matches!(level, "warn" | "error") || config::log_stderr())
}

#[derive(Clone)]
pub struct Logger {
    service: String,
    base: serde_json::Map<String, Value>,
}

impl Logger {
    pub fn child(&self, bindings: serde_json::Map<String, Value>) -> Logger {
        let mut merged = self.base.clone();
        merged.extend(bindings);
        Logger {
            service: self.service.clone(),
            base: merged,
        }
    }

    pub fn debug(&self, msg: &str, fields: Option<serde_json::Map<String, Value>>) {
        self.emit("debug", msg, fields)
    }

    pub fn info(&self, msg: &str, fields: Option<serde_json::Map<String, Value>>) {
        self.emit("info", msg, fields)
    }

    pub fn warn(&self, msg: &str, fields: Option<serde_json::Map<String, Value>>) {
        self.emit("warn", msg, fields)
    }

    pub fn error(&self, msg: &str, fields: Option<serde_json::Map<String, Value>>) {
        self.emit("error", msg, fields)
    }

    fn emit(&self, level: &str, msg: &str, fields: Option<serde_json::Map<String, Value>>) {
        let limit = writer::MAX_RECORD_BYTES - 4096;
        let fits = |value: &serde_json::Map<String, Value>| {
            crate::bounded_json_size(value, limit)
                .ok()
                .flatten()
                .is_some()
        };
        if msg.len() > limit
            || !fits(&self.base)
            || fields.as_ref().is_some_and(|value| !fits(value))
        {
            writer::global().omit(None);
            return;
        }
        let (verbose, stderr, path) = if config::captured_request_settings().is_some() {
            let policy = config::load_config();
            (policy.log_verbose, policy.log_stderr, log_file())
        } else {
            let policy = fallback_policy();
            (policy.verbose, policy.stderr, policy.path.clone())
        };
        let mut body = serde_json::Map::new();
        body.insert("t".into(), Value::String(now_iso8601()));
        body.insert("level".into(), Value::String(level.to_string()));
        body.insert("service".into(), Value::String(self.service.clone()));
        body.insert("msg".into(), Value::String(msg.to_string()));

        let mut merged = self.base.clone();
        if let Some(fields) = fields {
            merged.extend(fields);
        }
        if !merged.is_empty() {
            body.insert(
                "fields".into(),
                redact_with_depth(Value::Object(merged), 0, verbose),
            );
        }

        let Some(mut line) = writer::bounded_json(&Value::Object(body), false) else {
            writer::global().omit(None);
            return;
        };
        line.push(b'\n');
        let mirror_to_stderr =
            !stderr_suppressed() && (matches!(level, "warn" | "error") || stderr);
        writer::global().enqueue(writer::Record::log(path, line, mirror_to_stderr));
    }
}

pub fn create_logger(service: &str) -> Logger {
    Logger {
        service: service.to_string(),
        base: serde_json::Map::new(),
    }
}

fn rotate_file(path: &Path) -> io::Result<()> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let rotated = path.with_extension(format!("{ts}"));
    fs::rename(path, rotated)?;
    Ok(())
}

fn create_dir(path: &Path, mode: u32) -> io::Result<()> {
    fs::create_dir_all(path)?;
    set_mode(path, mode);
    Ok(())
}

fn set_mode(path: &Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perm = meta.permissions();
            perm.set_mode(mode);
            let _ = fs::set_permissions(path, perm);
        }
    }
}

fn now_iso8601() -> String {
    let now = time::OffsetDateTime::now_utc();
    let format = time::format_description::parse_borrowed::<3>(
        "[year]-[month]-[day]T[hour]:[minute]:[second]Z",
    )
    .unwrap();
    now.format(&format).unwrap_or_else(|_| String::new())
}

pub fn redact_value(value: Value) -> Value {
    let verbose = if config::captured_request_settings().is_some() {
        config::log_verbose()
    } else {
        fallback_policy().verbose
    };
    redact_with_depth(value, 0, verbose)
}

fn redact_with_depth(value: Value, depth: u8, verbose: bool) -> Value {
    if depth > 6 {
        return Value::String("[depth-limit]".into());
    }

    match value {
        Value::String(s) => {
            if verbose {
                Value::String(s)
            } else if s.len() > 4000 {
                let mut end = 4000;
                while !s.is_char_boundary(end) {
                    end -= 1;
                }
                Value::String(format!("{}…[{} more]", &s[..end], s.len() - end))
            } else {
                Value::String(s)
            }
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|v| redact_with_depth(v, depth + 1, verbose))
                .collect(),
        ),
        Value::Object(fields) => {
            let mut out = serde_json::Map::new();
            for (key, value) in fields {
                if REDACT_KEYS.contains(&key.to_lowercase().as_str())
                    || key.eq_ignore_ascii_case("encrypted_content")
                    || (key.eq_ignore_ascii_case("signature") && value.as_str().is_some_and(crate::providers::codex::translate::reasoning_signature::is_proxy_reasoning_signature)) {
                    out.insert(key, redact_key_redaction(value));
                } else {
                    out.insert(key, redact_with_depth(value, depth + 1, verbose));
                }
            }
            Value::Object(out)
        }
        value => value,
    }
}

fn redact_key_redaction(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(format!("[redacted len={}]", s.len())),
        _ => Value::String("[redacted]".to_string()),
    }
}

pub fn redacted_keys() -> HashSet<&'static str> {
    REDACT_KEYS.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static STDERR_TEST_LOCK: Mutex<()> = Mutex::new(());

    // A FIFO with no reader models a permanently stalled filesystem syscall.
    #[cfg(unix)]
    #[test]
    fn ordinary_logger_returns_while_disk_is_stalled() {
        let temp = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
            ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
        ]));
        let path = settings.clone().sync_scope(log_file);
        // Retention admits only regular files. Pause the real disk worker rather
        // than making the managed log destination an unaccountable FIFO.
        let pause = writer::pause_for_fixture();
        let (sent, received) = std::sync::mpsc::channel();
        let emitter = std::thread::spawn(move || {
            settings.sync_scope(|| {
                create_logger("stalled-disk").info("still delivering", None);
                sent.send(()).unwrap();
            })
        });
        let returned = received
            .recv_timeout(std::time::Duration::from_millis(300))
            .is_ok();
        drop(pause);
        emitter.join().unwrap();
        assert!(flush_diagnostics(std::time::Duration::from_secs(2)));
        let line = std::fs::read_to_string(path).unwrap();
        assert!(returned, "ordinary logging waited for the stalled disk");
        assert!(line.contains("still delivering"));
    }

    #[test]
    fn redaction_truncates_unicode_at_a_character_boundary() {
        let text = "€".repeat(1400);
        let value = redact_with_depth(
            serde_json::json!({"nested": [text], "access_token": "secret"}),
            0,
            false,
        );
        let truncated = value["nested"][0].as_str().unwrap();
        assert!(truncated.starts_with(&"€".repeat(1333)));
        assert!(truncated.ends_with("…[201 more]"));
        assert_eq!(value["access_token"], "[redacted len=6]");
    }

    #[test]
    fn stderr_suppression_disables_level_mirroring() {
        let _lock = STDERR_TEST_LOCK.lock().unwrap();
        assert!(should_mirror_to_stderr("warn"));

        {
            let _guard = suppress_stderr();
            assert!(!should_mirror_to_stderr("warn"));
            assert!(!should_mirror_to_stderr("error"));
        }

        assert!(should_mirror_to_stderr("warn"));
    }

    #[test]
    fn stderr_suppression_supports_nested_guards() {
        let _lock = STDERR_TEST_LOCK.lock().unwrap();
        let outer = suppress_stderr();
        let inner = suppress_stderr();
        assert!(!should_mirror_to_stderr("warn"));

        drop(inner);
        assert!(!should_mirror_to_stderr("warn"));

        drop(outer);
        assert!(should_mirror_to_stderr("warn"));
    }

    #[test]
    fn redacts_proxy_authorization_case_insensitively() {
        let redacted = redact_value(serde_json::json!({
            "Proxy-Authorization": "Basic dXNlcjpwYXNz",
            "safe": "kept"
        }));

        assert_eq!(redacted["safe"], "kept");
        assert_eq!(redacted["Proxy-Authorization"], "[redacted len=18]");
    }
}
