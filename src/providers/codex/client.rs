use super::StreamSenderExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::anthropic::sse::parse_sse_events;
use crate::config;
use crate::logging::create_logger;
use crate::provider::RequestContext;
use crate::request_identity::ConversationIdentity;
use crate::retry::{compute_backoff_delay, should_retry_status, sleep};
use crate::traffic::TrafficCapture;

use super::auth::constants::{CODEX_API_ENDPOINT, ORIGINATOR, RESPONSES_LITE_ORIGINATOR};
use super::auth::manager::CodexAuthManager;
use super::auth::token_store::{DefaultCodexAuthStore, StoredAuth, file_store};
pub use super::events::{CodexLimitWindow, CodexUsageLimit};
use super::performance;
use super::search::{SearchRequest, SearchResponse};
use super::translate::request::ResponsesRequest;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CodexError {
    pub status: u16,
    pub message: String,
    pub detail: Option<String>,
    pub retry_after: Option<String>,
    pub usage_limit: Option<Box<CodexUsageLimit>>,
    pub origin: CodexErrorOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexErrorOrigin {
    Http,
    WebSocket,
    WebSocketHandshake,
    Auth,
    BufferedHttp,
    BufferedWebSocket,
    /// A local guard rejected upstream structure; 502 without transparent retry.
    HttpProtocol,
    WebSocketProtocol,
}

impl CodexErrorOrigin {
    pub(crate) fn is_protocol_rejection(self) -> bool {
        matches!(self, Self::HttpProtocol | Self::WebSocketProtocol)
    }
}

impl CodexError {
    pub fn new(status: u16, message: String) -> Self {
        Self {
            status,
            message,
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        }
    }
}

impl std::fmt::Display for CodexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codex error {}: {}", self.status, self.message)
    }
}

#[derive(Debug)]
pub struct CodexHeaderTimeoutError {
    pub timeout_ms: u64,
}

impl std::fmt::Display for CodexHeaderTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Timed out waiting {}ms for Codex response headers",
            self.timeout_ms
        )
    }
}

#[derive(Debug)]
pub struct CodexTransportError {
    pub message: String,
}

impl std::fmt::Display for CodexTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codex transport error: {}", self.message)
    }
}

// ---------------------------------------------------------------------------
// Header builder
// ---------------------------------------------------------------------------

fn default_user_agent(use_responses_lite: bool) -> String {
    if use_responses_lite {
        RESPONSES_LITE_ORIGINATOR.to_string()
    } else {
        format!("claude-code-proxy/{}", env!("CARGO_PKG_VERSION"))
    }
}

pub fn build_codex_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
    use_responses_lite: bool,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        header_value("content-type", "application/json")?,
    );
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "text/event-stream")?,
    );
    let bearer = format!("Bearer {}", auth.access);
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &bearer)?,
    );
    let originator = if use_responses_lite {
        RESPONSES_LITE_ORIGINATOR.to_string()
    } else {
        config::codex_originator(ORIGINATOR)
    };
    headers.insert("originator", header_value("originator", &originator)?);
    headers.insert(
        "openai-beta",
        header_value("openai-beta", "responses=experimental")?,
    );
    headers.insert(
        "x-codex-beta-features",
        header_value("x-codex-beta-features", "remote_compaction_v2")?,
    );
    if use_responses_lite {
        headers.insert(
            "x-openai-internal-codex-responses-lite",
            header_value("x-openai-internal-codex-responses-lite", "true")?,
        );
    }
    if let Some(ref account_id) = auth.account_id {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if let Some(ref session_id) = ctx.session_id {
        // Match the official Codex Responses session header.
        headers.insert("session-id", header_value("session-id", session_id)?);
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", session_id)?,
        );
        let window_id = format!("{session_id}:0");
        headers.insert(
            "x-codex-window-id",
            header_value("x-codex-window-id", &window_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(use_responses_lite));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_native_codex_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
    use_responses_lite: bool,
    stream: bool,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = build_codex_headers(auth, ctx, use_responses_lite)?;
    headers.insert(
        http::header::ACCEPT,
        header_value(
            "accept",
            if stream {
                "text/event-stream"
            } else {
                "application/json"
            },
        )?,
    );
    Ok(headers)
}

pub fn build_codex_search_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = build_codex_headers(auth, ctx, false)?;
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    let originator = config::codex_originator(RESPONSES_LITE_ORIGINATOR);
    headers.insert("originator", header_value("originator", &originator)?);
    let user_agent = config::codex_user_agent(RESPONSES_LITE_ORIGINATOR);
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_codex_image_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        header_value("content-type", "application/json")?,
    );
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &format!("Bearer {}", auth.access))?,
    );
    headers.insert(
        "originator",
        header_value("originator", &config::codex_originator(ORIGINATOR))?,
    );
    if let Some(account_id) = auth.account_id.as_deref() {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if let Some(session_id) = ctx.session_id.as_deref() {
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", session_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(false));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_codex_transcription_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &format!("Bearer {}", auth.access))?,
    );
    headers.insert("originator", header_value("originator", "Codex Desktop")?);
    if let Some(account_id) = auth.account_id.as_deref() {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if let Some(session_id) = ctx.session_id.as_deref() {
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", session_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(false));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

fn header_value(name: &str, value: &str) -> Result<http::HeaderValue, CodexError> {
    http::HeaderValue::from_str(value).map_err(|e| CodexError {
        status: 500,
        message: format!("Failed to parse {name} header"),
        detail: Some(e.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::Http,
    })
}

fn search_endpoint(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    match base_url.strip_suffix("/responses") {
        Some(api_root) => format!("{api_root}/alpha/search"),
        None => format!("{base_url}/alpha/search"),
    }
}

// ---------------------------------------------------------------------------
// WebSocket request shaping
// ---------------------------------------------------------------------------

pub fn build_websocket_request(
    body: &ResponsesRequest,
    continuation: Option<&super::continuation::ContinuationCandidate>,
) -> serde_json::Value {
    let input = continuation
        .and_then(|candidate| candidate.input_delta.as_deref())
        .unwrap_or(&body.input);
    let previous_response_id =
        continuation.and_then(|candidate| candidate.previous_response_id.as_deref());
    serde_json::to_value(body.wire(input, false, previous_response_id))
        .expect("wire request serializes")
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualTransport {
    Http,
    WebSocket,
}

pub struct CodexResponse {
    pub body: Vec<u8>,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub transport: ActualTransport,
}

pub type CodexHttpEventReceiver =
    tokio::sync::mpsc::Receiver<Result<serde_json::Value, CodexError>>;

const MAX_HTTP_SSE_FRAME_BYTES: usize = 8 * 1024 * 1024;

#[derive(Default)]
struct HttpSseDecoder {
    frame: Vec<u8>,
    line_start: usize,
    skip_lf: bool,
    pending_error: Option<CodexError>,
}

struct DecodedHttpSseEvent {
    event: Option<String>,
    payload: Option<serde_json::Value>,
}

struct HttpEventStreamState {
    resp: reqwest::Response,
    response_headers: Vec<(String, String)>,
    started_at: Instant,
    body_json: String,
    auth: StoredAuth,
    auth_refresh_attempted: bool,
    use_responses_lite: bool,
    retries: u32,
}

impl HttpSseDecoder {
    fn push(&mut self, input: &[u8]) -> Result<Vec<DecodedHttpSseEvent>, CodexError> {
        let mut events = Vec::new();
        for &byte in input {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            let result = match byte {
                b'\n' => self.end_line(&mut events),
                b'\r' => {
                    let result = self.end_line(&mut events);
                    self.skip_lf = true;
                    result
                }
                _ => self.push_byte(byte),
            };
            if let Err(error) = result {
                if events.is_empty() {
                    return Err(error);
                }
                // Preserve event order across a network chunk: deliver the
                // decoded semantic prefix before its subsequent decode error.
                self.pending_error = Some(error);
                break;
            }
        }
        Ok(events)
    }

    fn finish(&self) -> Result<(), CodexError> {
        if self.frame.is_empty() {
            Ok(())
        } else {
            Err(http_sse_error(
                "Codex SSE stream ended with an incomplete frame",
            ))
        }
    }

    fn push_byte(&mut self, byte: u8) -> Result<(), CodexError> {
        if self.frame.len() >= MAX_HTTP_SSE_FRAME_BYTES {
            return Err(http_sse_error("Codex SSE frame exceeds the size limit"));
        }
        self.frame.push(byte);
        Ok(())
    }

    fn end_line(&mut self, events: &mut Vec<DecodedHttpSseEvent>) -> Result<(), CodexError> {
        if self.frame.len() == self.line_start {
            if !self.frame.is_empty()
                && let Some(event) = decode_http_sse_frame(&self.frame)?
            {
                events.push(event);
            }
            self.frame.clear();
            self.line_start = 0;
            return Ok(());
        }
        self.push_byte(b'\n')?;
        self.line_start = self.frame.len();
        Ok(())
    }
}

fn decode_http_sse_frame(frame: &[u8]) -> Result<Option<DecodedHttpSseEvent>, CodexError> {
    let frame = std::str::from_utf8(frame)
        .map_err(|_| http_sse_error("Codex SSE frame contains invalid UTF-8"))?;
    let mut event = None;
    let mut data = Vec::new();
    for line in frame.lines() {
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_owned()),
            "data" => data.push(value),
            _ => {}
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let data = data.join("\n");
    if data == "[DONE]" {
        return Ok(Some(DecodedHttpSseEvent {
            event,
            payload: None,
        }));
    }
    let payload = serde_json::from_str(&data)
        .map_err(|_| http_sse_error("Codex SSE frame contains invalid JSON"))?;
    Ok(Some(DecodedHttpSseEvent {
        event,
        payload: Some(payload),
    }))
}

fn http_sse_error(message: &str) -> CodexError {
    CodexError {
        status: 0,
        message: message.to_string(),
        detail: Some("http_response_sse".to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::Http,
    }
}

pub(crate) struct OwnerAwareCodexResponse {
    response: CodexResponse,
    pub(crate) socket_id: Option<u64>,
    pub(crate) had_intervening_failure: bool,
}

impl OwnerAwareCodexResponse {
    pub(crate) fn new(response: CodexResponse, socket_id: Option<u64>) -> Self {
        Self {
            response,
            socket_id,
            had_intervening_failure: false,
        }
    }

    pub(crate) fn with_intervening_failure(mut self, failure: bool) -> Self {
        self.had_intervening_failure |= failure;
        self
    }

    fn into_response(self) -> CodexResponse {
        self.response
    }
}

impl std::ops::Deref for OwnerAwareCodexResponse {
    type Target = CodexResponse;

    fn deref(&self) -> &Self::Target {
        &self.response
    }
}

/// Internal classification survives retry decisions without changing the
/// public CodexError construction surface.
struct CodexRequestFailure {
    error: CodexError,
    retryable: bool,
}

impl From<CodexError> for CodexRequestFailure {
    fn from(error: CodexError) -> Self {
        let retryable = retryable_http_stream_error(&error);
        Self { error, retryable }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

const MAX_BUFFERED_TRANSPORT_RETRIES: u32 = 3;
const MAX_BUFFERED_TRANSPORT_ATTEMPTS: u32 = MAX_BUFFERED_TRANSPORT_RETRIES + 1;
const HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS: u64 = 300_000;
const IMAGE_HEADER_TIMEOUT_MS: u64 = 300_000;
/// Marks the error raised when the wait for the Codex response headers runs
/// out, so the retry classifiers can tell it apart from a transport failure.
const HTTP_RESPONSE_HEADERS_DETAIL: &str = "http_response_headers";

#[derive(Clone)]
struct ProxyEnvironment {
    http_proxy: Option<String>,
    https_proxy: Option<String>,
    all_proxy: Option<String>,
    no_proxy: Option<reqwest::NoProxy>,
    no_proxy_value: Option<String>,
}

impl ProxyEnvironment {
    fn from_env() -> Self {
        if std::env::var_os("REQUEST_METHOD").is_some() {
            return Self {
                http_proxy: None,
                https_proxy: None,
                all_proxy: None,
                no_proxy: None,
                no_proxy_value: None,
            };
        }

        let no_proxy_value = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .ok();
        Self {
            http_proxy: proxy_env_value("HTTP_PROXY", "http_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            https_proxy: proxy_env_value("HTTPS_PROXY", "https_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            all_proxy: proxy_env_value("ALL_PROXY", "all_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            no_proxy: no_proxy_value
                .as_deref()
                .and_then(reqwest::NoProxy::from_string),
            no_proxy_value,
        }
    }

    fn websocket_proxy_config(&self) -> super::websocket::WebSocketProxyConfig {
        super::websocket::WebSocketProxyConfig::new(
            self.http_proxy.as_deref(),
            self.https_proxy.as_deref(),
            self.all_proxy.as_deref(),
            self.no_proxy_value.as_deref(),
        )
    }

    fn apply(&self, mut builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        builder = builder.no_proxy();
        if let Some(proxy) = self.http_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::http(proxy)
                    .expect("validated HTTP_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        if let Some(proxy) = self.https_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::https(proxy)
                    .expect("validated HTTPS_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        if let Some(proxy) = self.all_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy)
                    .expect("validated ALL_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        builder
    }
}

fn native_http_client(proxy_environment: &ProxyEnvironment) -> reqwest::Client {
    proxy_environment
        .apply(
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .redirect(reqwest::redirect::Policy::none()),
        )
        .build()
        .expect("failed to create native Responses HTTP client")
}

fn proxy_env_value(
    uppercase: &'static str,
    lowercase: &'static str,
) -> Result<Option<String>, &'static str> {
    let Some(raw) = std::env::var_os(uppercase).or_else(|| std::env::var_os(lowercase)) else {
        return Ok(None);
    };
    let raw = raw.into_string().map_err(|_| uppercase)?;
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    normalize_proxy_url(raw).map(Some).ok_or(uppercase)
}

fn normalize_proxy_url(raw: &str) -> Option<String> {
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    let parsed = url::Url::parse(&candidate).ok()?;
    if !matches!(
        parsed.scheme(),
        "http" | "https" | "socks4" | "socks4a" | "socks5" | "socks5h"
    ) || parsed.host_str().is_none()
        || matches!(parsed.scheme(), "socks4" | "socks4a")
            && (!parsed.username().is_empty() || parsed.password().is_some())
    {
        return None;
    }
    Some(parsed.to_string())
}

fn websocket_http_client(proxy_environment: &ProxyEnvironment) -> reqwest::Client {
    let tls_config = super::websocket::websocket_tls_config();
    proxy_environment
        .apply(
            reqwest::Client::builder()
                .http1_only()
                .redirect(reqwest::redirect::Policy::none())
                .use_preconfigured_tls((*tls_config).clone()),
        )
        .build()
        .expect("failed to create Codex WebSocket HTTP client")
}

fn custom_client_auto_http_fallback_enabled(
    base_url: &str,
    proxy_config: &super::websocket::WebSocketProxyConfig,
) -> bool {
    let Ok(websocket_url) = super::websocket::to_websocket_url(base_url) else {
        return false;
    };
    !proxy_config.uses_proxy_for(&websocket_url)
}

#[cfg(test)]
fn test_native_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .expect("failed to create test native Responses HTTP client")
}

#[cfg(test)]
fn test_websocket_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .http1_only()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .expect("failed to create test WebSocket HTTP client")
}

pub struct CodexHttpClient {
    client: reqwest::Client,
    native_client: reqwest::Client,
    websocket_client: reqwest::Client,
    websocket_proxy_config: super::websocket::WebSocketProxyConfig,
    auto_http_fallback_enabled: bool,
    websocket_reuse: bool,
    early_reasoning: bool,
    auth_manager: CodexAuthManager<DefaultCodexAuthStore>,
    base_url: String,
    header_timeout_ms: u64,
    body_idle_timeout_ms: u64,
}

impl Default for CodexHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexHttpClient {
    /// Messages can publish a thinking block before its summary; that start
    /// closes the same retry window as a visible tool start. Raw APIs opt out.
    pub(crate) fn with_early_reasoning(mut self) -> Self {
        self.early_reasoning = true;
        self
    }

    pub fn new() -> Self {
        let timeout_ms = config::codex_header_timeout_ms();
        let proxy_environment = ProxyEnvironment::from_env();
        Self {
            client: proxy_environment
                .apply(reqwest::Client::builder().connect_timeout(Duration::from_secs(15)))
                .build()
                .expect("failed to create HTTP client"),
            native_client: native_http_client(&proxy_environment),
            websocket_client: websocket_http_client(&proxy_environment),
            websocket_proxy_config: proxy_environment.websocket_proxy_config(),
            auto_http_fallback_enabled: true,
            websocket_reuse: config::codex_websocket_reuse(),
            early_reasoning: false,
            auth_manager: CodexAuthManager::new(file_store()),
            base_url: config::codex_base_url(CODEX_API_ENDPOINT),
            header_timeout_ms: timeout_ms,
            body_idle_timeout_ms: HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS,
        }
    }

    pub fn new_with_client(
        client: reqwest::Client,
        auth_manager: CodexAuthManager<DefaultCodexAuthStore>,
        base_url: String,
    ) -> Self {
        let proxy_environment = ProxyEnvironment::from_env();
        let websocket_proxy_config = proxy_environment.websocket_proxy_config();
        let auto_http_fallback_enabled =
            custom_client_auto_http_fallback_enabled(&base_url, &websocket_proxy_config);
        Self {
            native_client: native_http_client(&proxy_environment),
            websocket_client: websocket_http_client(&proxy_environment),
            websocket_proxy_config,
            auto_http_fallback_enabled,
            websocket_reuse: config::codex_websocket_reuse(),
            early_reasoning: false,
            client,
            auth_manager,
            base_url,
            header_timeout_ms: config::codex_header_timeout_ms(),
            body_idle_timeout_ms: HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS,
        }
    }

    #[cfg(test)]
    pub fn new_for_test(
        client: reqwest::Client,
        base_url: String,
        header_timeout_ms: u64,
        body_idle_timeout_ms: u64,
    ) -> Self {
        Self {
            native_client: test_native_http_client(),
            websocket_client: test_websocket_http_client(),
            websocket_proxy_config: super::websocket::WebSocketProxyConfig::direct(),
            auto_http_fallback_enabled: true,
            websocket_reuse: config::codex_websocket_reuse(),
            early_reasoning: false,
            client,
            auth_manager: CodexAuthManager::new(file_store()),
            base_url,
            header_timeout_ms,
            body_idle_timeout_ms,
        }
    }

    pub(crate) fn websocket_reuse(&self) -> bool {
        self.websocket_reuse
    }

    pub fn auth_manager(&self) -> &CodexAuthManager<DefaultCodexAuthStore> {
        &self.auth_manager
    }

    pub fn body_idle_timeout_ms(&self) -> u64 {
        self.body_idle_timeout_ms
    }

    pub(crate) async fn post_transcription(
        &self,
        base_url: &str,
        input: &super::transcription::PreparedTranscription,
        ctx: &RequestContext,
    ) -> Result<reqwest::Response, CodexError> {
        let url = format!("{}/transcribe", base_url.trim_end_matches('/'));
        let mut auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|error| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let mut refresh_attempted = false;

        loop {
            let headers = build_codex_transcription_headers(&auth, ctx)?;
            let response = self.attempt_transcription(&url, &headers, input).await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !refresh_attempted {
                refresh_attempted = true;
                drop(response);
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }
            return Ok(response);
        }
    }

    async fn attempt_transcription(
        &self,
        url: &str,
        headers: &http::HeaderMap,
        input: &super::transcription::PreparedTranscription,
    ) -> Result<reqwest::Response, CodexError> {
        let part = reqwest::multipart::Part::bytes(input.audio.to_vec())
            .file_name(input.filename.clone())
            .mime_str(&input.content_type)
            .map_err(|error| CodexError {
                status: 400,
                message: "Invalid audio content type".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?;
        let mut form = reqwest::multipart::Form::new().part("file", part);
        if let Some(language) = input.language.as_deref() {
            form = form.text("language", language.to_string());
        }
        let mut request = self.native_client.post(url).multipart(form);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }
        tokio::time::timeout(
            Duration::from_millis(IMAGE_HEADER_TIMEOUT_MS),
            request.send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex transcription response headers",
                IMAGE_HEADER_TIMEOUT_MS
            ),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|error| CodexError {
            status: 0,
            message: format!("Codex transcription transport error: {error}"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub(crate) async fn post_image_json(
        &self,
        base_url: &str,
        operation: super::images::ImageOperation,
        body: &serde_json::Value,
        ctx: &RequestContext,
    ) -> Result<reqwest::Response, CodexError> {
        let body_json = serde_json::to_vec(body).map_err(|error| CodexError {
            status: 500,
            message: "Failed to serialize image request".to_string(),
            detail: Some(error.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let url = format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            operation.upstream_path()
        );
        let mut auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|error| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let mut refresh_attempted = false;

        loop {
            let headers = build_codex_image_headers(&auth, ctx)?;
            let response = self
                .attempt_image_json(&url, &headers, body_json.clone())
                .await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !refresh_attempted {
                refresh_attempted = true;
                drop(response);
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }
            return Ok(response);
        }
    }

    async fn attempt_image_json(
        &self,
        url: &str,
        headers: &http::HeaderMap,
        body_json: Vec<u8>,
    ) -> Result<reqwest::Response, CodexError> {
        let mut request = self.native_client.post(url);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }
        tokio::time::timeout(
            Duration::from_millis(IMAGE_HEADER_TIMEOUT_MS),
            request.body(body_json).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex image response headers",
                IMAGE_HEADER_TIMEOUT_MS
            ),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|error| CodexError {
            status: 0,
            message: format!("Codex image transport error: {error}"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub async fn post_native_responses(
        &self,
        body: &serde_json::Value,
        ctx: &RequestContext,
        use_responses_lite: bool,
        stream: bool,
    ) -> Result<reqwest::Response, CodexError> {
        let body_json = serde_json::to_string(body).map_err(|err| CodexError {
            status: 500,
            message: "Failed to serialize native Responses request".to_string(),
            detail: Some(err.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let mut auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|err| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(err.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let mut refresh_attempted = false;

        loop {
            let started_at = Instant::now();
            let headers = build_native_codex_headers(&auth, ctx, use_responses_lite, stream)?;
            if let Some(traffic) = ctx.traffic.as_deref() {
                write_codex_http_request_capture(traffic, &self.base_url, &headers, &body_json);
            }

            let response = self
                .attempt_native_responses(&headers, body_json.clone())
                .await?;
            let status = response.status().as_u16();
            if status == 401 && !refresh_attempted {
                refresh_attempted = true;
                drop(response);
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }

            if let Some(traffic) = ctx.traffic.as_deref() {
                write_live_upstream_response_headers(traffic, &response, started_at.elapsed());
            }
            return Ok(response);
        }
    }

    async fn attempt_native_responses(
        &self,
        headers: &http::HeaderMap,
        body_json: String,
    ) -> Result<reqwest::Response, CodexError> {
        let mut request = self.native_client.post(&self.base_url);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }

        tokio::time::timeout(
            Duration::from_millis(self.header_timeout_ms),
            request.body(body_json).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex response headers",
                self.header_timeout_ms
            ),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|err| CodexError {
            status: 0,
            message: format!("Native Responses transport error: {err}"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub async fn post_codex(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<CodexResponse, CodexError> {
        let reservation =
            continuation.map(super::continuation::ContinuationReservation::from_public_candidate);
        self.post_codex_with_transport(
            body,
            ctx,
            reservation.as_ref(),
            crate::config::codex_transport(),
        )
        .await
        .map(OwnerAwareCodexResponse::into_response)
    }

    pub(crate) async fn post_codex_for_owner(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
    ) -> Result<OwnerAwareCodexResponse, CodexError> {
        self.post_codex_with_transport(body, ctx, continuation, crate::config::codex_transport())
            .await
    }

    pub async fn post_search(
        &self,
        body: &SearchRequest,
        ctx: &RequestContext,
    ) -> Result<SearchResponse, CodexError> {
        let mut auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Auth,
        })?;
        let body_json = serde_json::to_string(body).map_err(|e| CodexError {
            status: 500,
            message: "Failed to serialize search request".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let mut auth_refresh_attempted = false;
        let mut retries = 0_u32;

        loop {
            let response = self.attempt_post_search(&auth, &body_json, ctx).await?;
            if response.status == 401 && !auth_refresh_attempted {
                auth_refresh_attempted = true;
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }
            if normalized_response_failure(&response)
                .as_ref()
                .is_some_and(super::events::CodexEventFailure::retryable)
                && super::events::usage_limit_from_response(&response.body, &response.headers)
                    .is_none()
                && retries < MAX_BUFFERED_TRANSPORT_RETRIES
            {
                let retry_after = response
                    .headers
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case("retry-after"))
                    .map(|(_, value)| value.as_str());
                let delay = compute_backoff_delay(retries, retry_after);
                if delay.exceeds_budget {
                    return Err(codex_status_error(response));
                }
                retries += 1;
                sleep(delay.wait_ms).await;
                continue;
            }
            if !(200..300).contains(&response.status) {
                return Err(codex_status_error(response));
            }
            return serde_json::from_slice(&response.body).map_err(|e| CodexError {
                status: 502,
                message: "Failed to decode Codex search response".to_string(),
                detail: Some(e.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            });
        }
    }

    pub async fn stream_codex_http_events(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
    ) -> Result<CodexHttpEventReceiver, CodexError> {
        self.stream_codex_http_events_for_owner(body, ctx)
            .await
            .map(super::websocket::CodexWebSocketEventStream::into_receiver)
    }

    pub(crate) async fn stream_codex_http_events_for_owner(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
    ) -> Result<super::websocket::CodexWebSocketEventStream, CodexError> {
        performance::record(|m, elapsed| {
            m.preparation_start.get_or_insert(elapsed);
        });
        let mut auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Auth,
        })?;
        let body_json = serde_json::to_string(body).map_err(|e| CodexError {
            status: 500,
            message: "Failed to serialize request".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?;
        performance::record(|m, elapsed| {
            m.preparation_end = Some(elapsed);
            m.serialized_bytes += body_json.len() as u64;
        });
        let mut auth_refresh_attempted = false;
        let use_responses_lite = body.client_metadata.is_some();
        let mut retries = 0_u32;
        let (resp, response_headers, started_at) = loop {
            match self
                .start_http_event_attempt(
                    &mut auth,
                    &body_json,
                    ctx,
                    use_responses_lite,
                    &mut auth_refresh_attempted,
                )
                .await
            {
                Ok(attempt) => break attempt,
                Err(failure) if failure.retryable => {
                    let error = failure.error;
                    if retries >= MAX_BUFFERED_TRANSPORT_RETRIES {
                        return Err(error);
                    }
                    let delay = compute_backoff_delay(retries, error.retry_after.as_deref());
                    if delay.exceeds_budget {
                        return Err(error);
                    }
                    retries += 1;
                    sleep(delay.wait_ms).await;
                }
                Err(failure) => return Err(failure.error),
            }
        };

        Ok(self.spawn_http_event_stream(
            HttpEventStreamState {
                resp,
                response_headers,
                started_at,
                body_json,
                auth,
                auth_refresh_attempted,
                use_responses_lite,
                retries,
            },
            ctx.clone(),
        ))
    }

    async fn start_http_event_attempt(
        &self,
        auth: &mut StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
        use_responses_lite: bool,
        auth_refresh_attempted: &mut bool,
    ) -> Result<(reqwest::Response, Vec<(String, String)>, Instant), CodexRequestFailure> {
        loop {
            let (resp, started_at) = self
                .start_post_http(auth, body_json, ctx, use_responses_lite)
                .await?;

            if resp.status().as_u16() == 401 && !*auth_refresh_attempted {
                *auth_refresh_attempted = true;
                *auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }

            if !resp.status().is_success() {
                let response = self.collect_http_response(resp, started_at, ctx).await?;
                if let Some(traffic) = ctx.traffic.as_deref() {
                    write_buffered_failure_diagnostics(traffic, &ctx.req_id, &response);
                }
                let mut failure = codex_status_failure(response);
                failure.error.origin = CodexErrorOrigin::Http;
                return Err(failure);
            }

            let status = resp.status().as_u16();
            let headers = response_headers(&resp);
            if let Some(traffic) = ctx.traffic.as_deref() {
                write_upstream_response_headers_capture(
                    traffic,
                    status,
                    started_at.elapsed(),
                    &headers,
                );
            }
            return Ok((resp, headers, started_at));
        }
    }

    pub async fn stream_codex_auto_events(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<super::websocket::CodexWebSocketEventReceiver, CodexError> {
        let reservation =
            continuation.map(super::continuation::ContinuationReservation::from_public_candidate);
        self.stream_codex_auto_events_for_owner(body, ctx, reservation.as_ref())
            .await
            .map(super::websocket::CodexWebSocketEventStream::into_receiver)
    }

    pub(crate) async fn stream_codex_auto_events_for_owner(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
    ) -> Result<super::websocket::CodexWebSocketEventStream, CodexError> {
        self.stream_codex_auto_events_shared_for_owner(Arc::new(body.clone()), ctx, continuation)
            .await
    }

    pub(crate) async fn stream_codex_auto_events_shared_for_owner(
        self: &Arc<Self>,
        body: Arc<ResponsesRequest>,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
    ) -> Result<super::websocket::CodexWebSocketEventStream, CodexError> {
        let mut websocket = self
            .stream_codex_websocket_events_shared_for_owner(Arc::clone(&body), ctx, continuation)
            .await?;
        match websocket.recv().await {
            Some(Err(err)) if should_fallback_to_http(&err) => {
                let http = self.stream_codex_http_events_for_owner(&body, ctx).await?;
                if websocket.had_intervening_failure() {
                    http.mark_intervening_failure();
                }
                Ok(http)
            }
            Some(item) => {
                let (tx, rx) = tokio::sync::mpsc::channel(64);
                let receiver = websocket.replace_receiver(rx);
                let delivery_aborted = websocket.abort_signal();
                tokio::spawn(crate::config::inherit_request_scope(async move {
                    if tx.send_bounded(item).await.is_err()
                        || !forward_codex_events(receiver, tx).await
                    {
                        delivery_aborted.store(true, std::sync::atomic::Ordering::Release);
                    }
                }));
                Ok(websocket)
            }
            None => Err(CodexError {
                status: 0,
                message: "WebSocket connection closed before the first Codex event".to_string(),
                detail: Some(super::websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::WebSocket,
            }),
        }
    }

    fn spawn_http_event_stream(
        self: &Arc<Self>,
        state: HttpEventStreamState,
        ctx: RequestContext,
    ) -> super::websocket::CodexWebSocketEventStream {
        let HttpEventStreamState {
            mut resp,
            mut response_headers,
            mut started_at,
            body_json,
            mut auth,
            mut auth_refresh_attempted,
            use_responses_lite,
            mut retries,
        } = state;
        let client = self.clone();
        let body_idle_timeout_ms = self.body_idle_timeout_ms;
        let req_id = ctx.req_id.clone();
        let traffic = ctx.traffic.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let (rx, publisher) = super::websocket::CodexWebSocketEventStream::pending(rx);
        if retries > 0 || auth_refresh_attempted {
            publisher.mark_intervening_failure();
        }
        let tx = publisher.sender(tx);

        let observation = performance::current();
        let reader = performance::in_scope(observation, async move {
            let _finalization = performance::ReaderFinalization;
            let log = create_logger("codex");
            let mut semantic_output_forwarded = false;

            if tx
                .send_bounded(Ok(serde_json::json!({
                    "type": "keepalive",
                    "_ccp_synthetic": true
                })))
                .await
                .is_err()
            {
                performance::record(|m, elapsed| m.body_cancel = Some(elapsed));
                return;
            }

            'attempts: loop {
                let mut decoder = HttpSseDecoder::default();
                let mut body_bytes = 0_u64;
                let mut body_chunks = 0_u64;
                let mut event_count = 0_u64;
                let mut pending_terminal: Option<serde_json::Value> = None;
                let mut pending_events = super::events::PendingStructuralEvents::default();
                if client.early_reasoning {
                    pending_events = pending_events.with_early_reasoning();
                }

                let mut retry_error = 'read_attempt: loop {
                    let chunk = tokio::select! {
                        _ = tx.closed() => {
                            performance::record(|m, elapsed| m.body_cancel = Some(elapsed));
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_dropped",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                None,
                                None,
                            );
                            return;
                        }
                        chunk = tokio::time::timeout(
                            Duration::from_millis(body_idle_timeout_ms),
                            resp.chunk(),
                        ) => chunk
                    };

                    let chunk = match chunk {
                        Ok(Ok(Some(chunk))) => chunk,
                        Ok(Ok(None)) => {
                            let finished = decoder.finish();
                            if finished.is_ok()
                                && let Some(payload) = pending_terminal.take()
                            {
                                let outcome = super::events::terminal_outcome(&payload);
                                let ready = match pending_events
                                    .stage(payload, &mut semantic_output_forwarded)
                                {
                                    Ok(ready) => ready,
                                    Err(()) => {
                                        let _ = tx
                                            .send_bounded(Err(presemantic_limit_error(
                                                CodexErrorOrigin::Http,
                                            )))
                                            .await;
                                        return;
                                    }
                                };
                                for event in ready {
                                    if tx.send_bounded(Ok(event)).await.is_err() {
                                        return;
                                    }
                                }
                                performance::record(|m, elapsed| m.body_complete = Some(elapsed));
                                log_http_stream_end(
                                    &log,
                                    "codex_http_stream_completed",
                                    &req_id,
                                    started_at,
                                    body_bytes,
                                    body_chunks,
                                    event_count,
                                    None,
                                    outcome,
                                );
                                return;
                            }
                            let error = match finished {
                                Ok(()) => http_sse_error(
                                    "Codex SSE stream ended before a terminal response event",
                                ),
                                Err(error) => error,
                            };
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_failed",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                Some(&error.message),
                                None,
                            );
                            if !semantic_output_forwarded && retryable_http_stream_error(&error) {
                                break 'read_attempt error;
                            }
                            let _ = tx.send_bounded(Err(error)).await;
                            return;
                        }
                        Ok(Err(err)) => {
                            let error = CodexError {
                                status: 0,
                                message: format!(
                                    "Transport error reading Codex response body: {err}"
                                ),
                                detail: Some("http_response_body".to_string()),
                                retry_after: None,
                                usage_limit: None,
                                origin: CodexErrorOrigin::Http,
                            };
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_failed",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                Some(&error.message),
                                None,
                            );
                            if !semantic_output_forwarded {
                                break 'read_attempt error;
                            }
                            let _ = tx.send_bounded(Err(error)).await;
                            return;
                        }
                        Err(_) => {
                            let error = CodexError {
                                status: 0,
                                message: format!(
                                    "Timed out waiting {body_idle_timeout_ms}ms for the next Codex response body chunk"
                                ),
                                detail: Some("http_response_body".to_string()),
                                retry_after: None,
                                usage_limit: None,
                                origin: CodexErrorOrigin::Http,
                            };
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_failed",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                Some(&error.message),
                                None,
                            );
                            if !semantic_output_forwarded {
                                break 'read_attempt error;
                            }
                            let _ = tx.send_bounded(Err(error)).await;
                            return;
                        }
                    };

                    body_bytes = body_bytes.saturating_add(chunk.len() as u64);
                    body_chunks = body_chunks.saturating_add(1);
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(error) => {
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_failed",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                Some(&error.message),
                                None,
                            );
                            if !semantic_output_forwarded && retryable_http_stream_error(&error) {
                                break 'read_attempt error;
                            }
                            let _ = tx.send_bounded(Err(error)).await;
                            return;
                        }
                    };

                    for event in events {
                        let Some(payload) = event.payload else {
                            continue;
                        };
                        performance::record(|m, elapsed| {
                            m.first_upstream_event.get_or_insert(elapsed);
                            if payload.get("type").and_then(serde_json::Value::as_str)
                                == Some("response.output_text.delta")
                                && payload
                                    .get("delta")
                                    .and_then(serde_json::Value::as_str)
                                    .is_some_and(|text| !text.is_empty())
                            {
                                m.first_upstream_text.get_or_insert(elapsed);
                            }
                            if super::events::classify_stream_event(&payload)
                                == super::events::CodexStreamEventKind::Semantic
                            {
                                m.first_semantic.get_or_insert(elapsed);
                            }
                        });
                        event_count = event_count.saturating_add(1);
                        if let Some(traffic) = traffic.as_deref() {
                            write_codex_http_sse_event_capture(
                                traffic,
                                event.event.as_deref(),
                                &payload,
                            );
                        }

                        if pending_terminal.is_some() {
                            if payload["type"] == "ping" {
                                continue;
                            }
                            let message = if super::events::event_is_terminal(&payload) {
                                "Codex SSE stream contained multiple terminal response events"
                            } else {
                                "Codex SSE stream contained a JSON event after its terminal response event"
                            };
                            let error = http_sse_error(message);
                            log_http_stream_end(
                                &log,
                                "codex_http_stream_failed",
                                &req_id,
                                started_at,
                                body_bytes,
                                body_chunks,
                                event_count,
                                Some(&error.message),
                                None,
                            );
                            let _ = tx.send_bounded(Err(error)).await;
                            return;
                        }

                        let failure = super::events::classify_event_failure(&payload);
                        if let (Some(failure), Some(traffic)) =
                            (failure.as_ref(), traffic.as_deref())
                        {
                            traffic.write_json(
                                "060-codex-event-error",
                                &failure.diagnostics(
                                    "http_ingress",
                                    "event_failure",
                                    Some(&req_id),
                                    payload.get("type").and_then(serde_json::Value::as_str),
                                    semantic_output_forwarded,
                                ),
                            );
                        }
                        if !semantic_output_forwarded {
                            if let Some(limit) = super::events::usage_limit_from_event_with_headers(
                                &payload,
                                &response_headers,
                            ) {
                                pending_events.clear();
                                let _ = tx
                                    .send_bounded(Err(codex_usage_limit_error(
                                        limit,
                                        CodexErrorOrigin::Http,
                                    )))
                                    .await;
                                return;
                            }
                            if let Some(failure) = failure.as_ref()
                                && failure.retryable()
                            {
                                pending_events.clear();
                                break 'read_attempt codex_event_failure_error(failure.clone());
                            }
                        }

                        if super::events::terminal_outcome(&payload).is_some() {
                            pending_terminal = Some(payload);
                            continue;
                        }
                        let ready =
                            match pending_events.stage(payload, &mut semantic_output_forwarded) {
                                Ok(ready) => ready,
                                Err(()) => {
                                    let _ = tx
                                        .send_bounded(Err(presemantic_limit_error(
                                            CodexErrorOrigin::Http,
                                        )))
                                        .await;
                                    return;
                                }
                            };
                        for event in ready {
                            if tx.send_bounded(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                    if let Some(error) = decoder.pending_error.take() {
                        log_http_stream_end(
                            &log,
                            "codex_http_stream_failed",
                            &req_id,
                            started_at,
                            body_bytes,
                            body_chunks,
                            event_count,
                            Some(&error.message),
                            None,
                        );
                        if !semantic_output_forwarded && retryable_http_stream_error(&error) {
                            break 'read_attempt error;
                        }
                        let _ = tx.send_bounded(Err(error)).await;
                        return;
                    }
                };

                publisher.mark_intervening_failure();
                loop {
                    if retries >= MAX_BUFFERED_TRANSPORT_RETRIES {
                        let _ = tx.send_bounded(Err(retry_error)).await;
                        return;
                    }
                    let delay = compute_backoff_delay(retries, retry_error.retry_after.as_deref());
                    if delay.exceeds_budget {
                        let _ = tx.send_bounded(Err(retry_error)).await;
                        return;
                    }
                    retries += 1;
                    tokio::select! {
                        _ = tx.closed() => return,
                        _ = sleep(delay.wait_ms) => {}
                    }

                    let next_attempt = tokio::select! {
                        _ = tx.closed() => return,
                        result = client.start_http_event_attempt(
                            &mut auth,
                            &body_json,
                            &ctx,
                            use_responses_lite,
                            &mut auth_refresh_attempted,
                        ) => result
                    };
                    match next_attempt {
                        Ok((next_resp, next_response_headers, next_started_at)) => {
                            resp = next_resp;
                            response_headers = next_response_headers;
                            started_at = next_started_at;
                            continue 'attempts;
                        }
                        Err(failure) if failure.retryable => {
                            retry_error = failure.error;
                        }
                        Err(failure) => {
                            let _ = tx.send_bounded(Err(failure.error)).await;
                            return;
                        }
                    }
                }
            }
        });
        tokio::spawn(crate::config::inherit_request_scope(reader));

        rx
    }

    async fn post_codex_with_transport(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
        transport: crate::config::CodexTransport,
    ) -> Result<OwnerAwareCodexResponse, CodexError> {
        use crate::config::CodexTransport;

        let mut auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Auth,
        })?;

        let mut active_continuation = continuation.cloned();
        let mut auth_refresh_attempted = false;
        let mut transport_failures = 0u32;
        let mut intervening_failure = false;
        loop {
            let result = match transport {
                CodexTransport::Http => {
                    let body_json = serde_json::to_string(body).map_err(|e| CodexError {
                        status: 500,
                        message: "Failed to serialize request".to_string(),
                        detail: Some(e.to_string()),
                        retry_after: None,
                        usage_limit: None,
                        origin: CodexErrorOrigin::Http,
                    })?;
                    self.attempt_post_http(&auth, &body_json, ctx, body.client_metadata.is_some())
                        .await
                        .map(|response| OwnerAwareCodexResponse::new(response, None))
                }
                CodexTransport::WebSocket => {
                    let ws_headers =
                        build_codex_headers(&auth, ctx, body.client_metadata.is_some())?;
                    let ws_headers = super::websocket::codex_websocket_headers(&ws_headers);
                    match super::websocket::codex_websocket_responses_stream(
                        &self.websocket_client,
                        &self.websocket_proxy_config,
                        &self.base_url,
                        &ws_headers,
                        body,
                        ctx,
                        active_continuation.as_ref(),
                    )
                    .await
                    {
                        Ok(events) => {
                            super::websocket::collect_codex_websocket_response(
                                events,
                                ctx.traffic.as_deref(),
                                std::time::Instant::now(),
                            )
                            .await
                        }
                        Err(error) => Err(error),
                    }
                }
                CodexTransport::Auto => {
                    let ws_headers =
                        build_codex_headers(&auth, ctx, body.client_metadata.is_some())?;
                    let ws_headers = super::websocket::codex_websocket_headers(&ws_headers);
                    let ws_result = match super::websocket::codex_websocket_responses_stream(
                        &self.websocket_client,
                        &self.websocket_proxy_config,
                        &self.base_url,
                        &ws_headers,
                        body,
                        ctx,
                        active_continuation.as_ref(),
                    )
                    .await
                    {
                        Ok(events) => {
                            super::websocket::collect_codex_websocket_response(
                                events,
                                ctx.traffic.as_deref(),
                                std::time::Instant::now(),
                            )
                            .await
                        }
                        Err(error) => Err(error),
                    };
                    match ws_result {
                        Ok(response) => Ok(response),
                        Err(err)
                            if should_retry_without_continuation(
                                &err,
                                active_continuation.as_ref(),
                            ) =>
                        {
                            // Drop stale continuation state before considering a
                            // replacement transport or connection.
                            Err(err)
                        }
                        Err(err)
                            if self.auto_http_fallback_enabled && should_fallback_to_http(&err) =>
                        {
                            // Fall back to HTTP only if WebSocket failed before sending
                            let body_json =
                                serde_json::to_string(body).map_err(|e| CodexError {
                                    status: 500,
                                    message: "Failed to serialize request".to_string(),
                                    detail: Some(e.to_string()),
                                    retry_after: None,
                                    usage_limit: None,
                                    origin: CodexErrorOrigin::Http,
                                })?;
                            self.attempt_post_http(
                                &auth,
                                &body_json,
                                ctx,
                                body.client_metadata.is_some(),
                            )
                            .await
                            .map(|response| OwnerAwareCodexResponse::new(response, None))
                        }
                        Err(err) => Err(err),
                    }
                }
            };

            if should_refresh_after_unauthorized(&result, auth_refresh_attempted, transport) {
                auth_refresh_attempted = true;
                match self.auth_manager.force_refresh(&auth.access).await {
                    Ok(new_auth) => {
                        auth = new_auth;
                        invalidate_live_continuation_pool(active_continuation.as_ref());
                        active_continuation =
                            full_context_continuation(active_continuation.as_ref());
                        continue;
                    }
                    Err(e) => {
                        return Err(CodexError {
                            status: 401,
                            message: "Unauthorized".to_string(),
                            detail: Some(e.to_string()),
                            retry_after: None,
                            usage_limit: None,
                            origin: CodexErrorOrigin::Http,
                        });
                    }
                }
            }

            if let (Ok(response), Some(traffic)) = (&result, ctx.traffic.as_deref()) {
                write_buffered_failure_diagnostics(traffic, &ctx.req_id, response);
            }

            if let Ok(response) = &result
                && let Some(limit) =
                    super::events::usage_limit_from_response(&response.body, &response.headers)
            {
                let origin = match response.transport {
                    ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
                    ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
                };
                return Err(codex_usage_limit_error(limit, origin));
            }

            if let Ok(response) = &result
                && let Some(failure) = normalized_response_failure(response)
            {
                let retryable = failure.retryable();
                let mut error = codex_event_failure_error(failure);
                error.origin = match response.transport {
                    ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
                    ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
                };
                if retryable && transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                    let delay =
                        compute_backoff_delay(transport_failures, error.retry_after.as_deref());
                    if delay.exceeds_budget {
                        return Err(error);
                    }
                    log_buffered_retry(
                        ctx,
                        transport,
                        transport_failures + 1,
                        delay.wait_ms,
                        error.status,
                        "upstream_event",
                        &error.message,
                    );
                    intervening_failure = true;
                    transport_failures += 1;
                    active_continuation = full_context_continuation(active_continuation.as_ref());
                    sleep(delay.wait_ms).await;
                    continue;
                }
                if retryable {
                    log_buffered_retry_exhausted(
                        ctx,
                        transport,
                        error.status,
                        "upstream_event",
                        &error.message,
                    );
                }
                return Err(error);
            }

            match result {
                Ok(response) => {
                    return Ok(response
                        .with_intervening_failure(intervening_failure || auth_refresh_attempted));
                }
                Err(err)
                    if should_retry_without_continuation(&err, active_continuation.as_ref()) =>
                {
                    intervening_failure = true;
                    active_continuation = full_context_continuation(active_continuation.as_ref());
                    continue;
                }
                Err(err) => {
                    let retryable = is_retryable_transport_error(&err);
                    if retryable && transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                        let delay =
                            compute_backoff_delay(transport_failures, err.retry_after.as_deref());
                        if delay.exceeds_budget {
                            return Err(err);
                        }
                        log_buffered_retry(
                            ctx,
                            transport,
                            transport_failures + 1,
                            delay.wait_ms,
                            err.status,
                            codex_error_origin_name(err.origin),
                            &err.message,
                        );
                        intervening_failure = true;
                        transport_failures += 1;
                        sleep(delay.wait_ms).await;
                        continue;
                    }
                    if retryable {
                        log_buffered_retry_exhausted(
                            ctx,
                            transport,
                            err.status,
                            codex_error_origin_name(err.origin),
                            &err.message,
                        );
                    }
                    return Err(err);
                }
            }
        }
    }

    pub async fn stream_codex_websocket_events(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<super::websocket::CodexWebSocketEventReceiver, CodexError> {
        let reservation =
            continuation.map(super::continuation::ContinuationReservation::from_public_candidate);
        self.stream_codex_websocket_events_for_owner(body, ctx, reservation.as_ref())
            .await
            .map(super::websocket::CodexWebSocketEventStream::into_receiver)
    }

    pub(crate) async fn stream_codex_websocket_events_for_owner(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
    ) -> Result<super::websocket::CodexWebSocketEventStream, CodexError> {
        self.stream_codex_websocket_events_shared_for_owner(
            Arc::new(body.clone()),
            ctx,
            continuation,
        )
        .await
    }

    pub(crate) async fn stream_codex_websocket_events_shared_for_owner(
        self: &Arc<Self>,
        body: Arc<ResponsesRequest>,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationReservation>,
    ) -> Result<super::websocket::CodexWebSocketEventStream, CodexError> {
        let auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Auth,
        })?;

        let client = self.clone();
        let ctx = ctx.clone();
        let continuation = continuation.cloned();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let (rx, socket_id_publisher) = super::websocket::CodexWebSocketEventStream::pending(rx);
        let rx = rx.track_attempt(continuation.as_ref(), true);
        let tx = socket_id_publisher.sender(tx);
        tokio::spawn(crate::config::inherit_request_scope(async move {
            client
                .coordinate_live_websocket_events(
                    body,
                    ctx,
                    continuation,
                    auth,
                    tx,
                    socket_id_publisher,
                )
                .await;
        }));

        Ok(rx)
    }

    #[allow(clippy::too_many_arguments)]
    async fn coordinate_live_websocket_events(
        &self,
        body: Arc<ResponsesRequest>,
        ctx: RequestContext,
        mut continuation: Option<super::continuation::ContinuationReservation>,
        mut auth: StoredAuth,
        tx: super::websocket::EventSender,
        socket_id_publisher: super::websocket::CodexWebSocketSocketIdPublisher,
    ) {
        let mut auth_refresh_attempted = false;
        let mut continuation_retry_available = continuation
            .as_ref()
            .and_then(|reservation| reservation.candidate().previous_response_id.as_deref())
            .is_some();
        let mut forwarded_any = false;

        'attempt: loop {
            socket_id_publisher.publish(None);
            socket_id_publisher.publish_diagnostic_attempt(None);
            let ws_headers = match build_codex_headers(&auth, &ctx, body.client_metadata.is_some())
            {
                Ok(headers) => super::websocket::codex_websocket_headers(&headers),
                Err(err) => {
                    if tx.send_bounded(Err(err)).await.is_err() {
                        abort_abandoned_live_continuation(
                            continuation.as_ref(),
                            &socket_id_publisher,
                        );
                    }
                    return;
                }
            };
            let start = super::websocket::codex_websocket_responses_stream(
                &self.websocket_client,
                &self.websocket_proxy_config,
                &self.base_url,
                &ws_headers,
                &body,
                &ctx,
                continuation.as_ref(),
            );
            let mut stream = tokio::select! {
                biased;
                _ = tx.closed() => {
                    abort_abandoned_live_continuation(
                        continuation.as_ref(),
                        &socket_id_publisher,
                    );
                    return;
                }
                result = start => match result {
                    Ok(stream) => stream,
                    Err(err) if err.status == 401 && !auth_refresh_attempted && !forwarded_any => {
                        auth_refresh_attempted = true;
                        socket_id_publisher.mark_intervening_failure();
                        invalidate_live_continuation_pool(continuation.as_ref());
                        let refresh = self.auth_manager.force_refresh(&auth.access);
                        auth = match tokio::select! { biased; _ = tx.closed() => { abort_abandoned_live_continuation(continuation.as_ref(), &socket_id_publisher); return; }, result = refresh => result } {
                            Ok(auth) => {
                                if tx.is_closed() {
                                    abort_abandoned_live_continuation(
                                        continuation.as_ref(),
                                        &socket_id_publisher,
                                    );
                                    return;
                                }
                                auth
                            },
                            Err(refresh_err) => {
                                if tx.send_bounded(Err(auth_refresh_error(refresh_err))).await.is_err() {
                                    abort_abandoned_live_continuation(
                                        continuation.as_ref(),
                                        &socket_id_publisher,
                                    );
                                }
                                return;
                            }
                        };
                        if continuation_retry_available {
                            socket_id_publisher.mark_full_context_retry();
                        }
                        continuation = full_context_continuation(continuation.as_ref());
                        continuation_retry_available = false;
                        continue 'attempt;
                    }
                    Err(err) if continuation_retry_available && is_continuation_retry_error(&err) => {
                        socket_id_publisher.mark_full_context_retry();
                        continuation = full_context_continuation(continuation.as_ref());
                        continuation_retry_available = false;
                        continue 'attempt;
                    }
                    Err(err) => {
                        if tx.send_bounded(Err(err)).await.is_err() {
                            abort_abandoned_live_continuation(
                                continuation.as_ref(),
                                &socket_id_publisher,
                            );
                        }
                        return;
                    }
                }
            };

            let mut pending_events = super::events::PendingStructuralEvents::default();
            if self.early_reasoning {
                pending_events = pending_events.with_early_reasoning();
            }
            loop {
                let item = tokio::select! {
                    biased;
                    _ = tx.closed() => {
                        if let Some(reservation) = continuation.as_ref() {
                            super::websocket::invalidate_codex_websocket_pool_socket(
                                reservation,
                                stream.socket_id(),
                            );
                        }
                        abort_abandoned_live_continuation(
                            continuation.as_ref(),
                            &socket_id_publisher,
                        );
                        return;
                    }
                    item = stream.recv() => item,
                };
                if tx.is_closed() {
                    if let Some(reservation) = continuation.as_ref() {
                        super::websocket::invalidate_codex_websocket_pool_socket(
                            reservation,
                            stream.socket_id(),
                        );
                    }
                    abort_abandoned_live_continuation(continuation.as_ref(), &socket_id_publisher);
                    return;
                }
                let Some(item) = item else {
                    if stream.delivery_aborted() {
                        socket_id_publisher.abort_delivery();
                    }
                    return;
                };
                if stream.had_intervening_failure() {
                    socket_id_publisher.mark_intervening_failure();
                }
                socket_id_publisher.publish(stream.socket_id());
                socket_id_publisher.publish_diagnostic_attempt(stream.diagnostic_attempt_id());

                let unauthorized = match &item {
                    Err(err) => err.status == 401,
                    Ok(payload) => super::websocket::event_error_status(payload) == Some(401),
                };
                if unauthorized && !auth_refresh_attempted && !forwarded_any {
                    auth_refresh_attempted = true;
                    socket_id_publisher.mark_intervening_failure();
                    invalidate_live_continuation_pool(continuation.as_ref());
                    let refresh = self.auth_manager.force_refresh(&auth.access);
                    auth = match tokio::select! { biased; _ = tx.closed() => { abort_abandoned_live_continuation(continuation.as_ref(), &socket_id_publisher); return; }, result = refresh => result }
                    {
                        Ok(auth) => {
                            if tx.is_closed() {
                                if let Some(reservation) = continuation.as_ref() {
                                    super::websocket::invalidate_codex_websocket_pool_socket(
                                        reservation,
                                        stream.socket_id(),
                                    );
                                }
                                abort_abandoned_live_continuation(
                                    continuation.as_ref(),
                                    &socket_id_publisher,
                                );
                                return;
                            }
                            auth
                        }
                        Err(refresh_err) => {
                            if tx
                                .send_bounded(Err(auth_refresh_error(refresh_err)))
                                .await
                                .is_err()
                            {
                                if let Some(reservation) = continuation.as_ref() {
                                    super::websocket::invalidate_codex_websocket_pool_socket(
                                        reservation,
                                        stream.socket_id(),
                                    );
                                }
                                abort_abandoned_live_continuation(
                                    continuation.as_ref(),
                                    &socket_id_publisher,
                                );
                            }
                            return;
                        }
                    };
                    if continuation_retry_available {
                        socket_id_publisher.mark_full_context_retry();
                    }
                    continuation = full_context_continuation(continuation.as_ref());
                    continuation_retry_available = false;
                    continue 'attempt;
                }

                if let Err(err) = &item
                    && continuation_retry_available
                    && is_continuation_retry_error(err)
                    && !forwarded_any
                {
                    socket_id_publisher.mark_full_context_retry();
                    continuation = full_context_continuation(continuation.as_ref());
                    continuation_retry_available = false;
                    continue 'attempt;
                }

                let terminal = item.as_ref().is_err()
                    || item.as_ref().is_ok_and(super::websocket::is_terminal_event);
                match item {
                    Ok(payload) => {
                        let ready = match pending_events.stage(payload, &mut forwarded_any) {
                            Ok(ready) => ready,
                            Err(()) => {
                                let _ = tx
                                    .send_bounded(Err(presemantic_limit_error(
                                        CodexErrorOrigin::WebSocket,
                                    )))
                                    .await;
                                abort_abandoned_live_continuation(
                                    continuation.as_ref(),
                                    &socket_id_publisher,
                                );
                                return;
                            }
                        };
                        for event in ready {
                            if tx.send_bounded(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(err) => {
                        if tx.send_bounded(Err(err)).await.is_err() {
                            if let Some(reservation) = continuation.as_ref() {
                                super::websocket::invalidate_codex_websocket_pool_socket(
                                    reservation,
                                    stream.socket_id(),
                                );
                            }
                            abort_abandoned_live_continuation(
                                continuation.as_ref(),
                                &socket_id_publisher,
                            );
                            return;
                        }
                    }
                }
                if terminal {
                    stream.accept_terminal();
                    return;
                }
            }
        }
    }

    async fn attempt_post_http(
        &self,
        auth: &StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
        use_responses_lite: bool,
    ) -> Result<CodexResponse, CodexError> {
        let (resp, started_at) = self
            .start_post_http(auth, body_json, ctx, use_responses_lite)
            .await?;
        self.collect_http_response(resp, started_at, ctx).await
    }

    async fn start_post_http(
        &self,
        auth: &StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
        use_responses_lite: bool,
    ) -> Result<(reqwest::Response, Instant), CodexError> {
        let url = &self.base_url;
        let headers = build_codex_headers(auth, ctx, use_responses_lite)?;

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_codex_http_request_capture(traffic, url, &headers, body_json);
        }

        let mut req_builder = self.client.post(url);
        for (key, value) in headers.iter() {
            req_builder = req_builder.header(key.as_str(), value.as_bytes());
        }

        let started_at = Instant::now();
        performance::record(|m, _| {
            m.attempts += 1;
            m.full_request_clone_bytes += body_json.len() as u64;
        });
        let send_fut = req_builder.body(body_json.to_string()).send();
        let header_timeout_dur = Duration::from_millis(self.header_timeout_ms);

        let resp = tokio::time::timeout(header_timeout_dur, send_fut)
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for Codex response headers",
                    self.header_timeout_ms
                ),
                detail: Some(HTTP_RESPONSE_HEADERS_DETAIL.to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| {
                if is_retryable_reqwest_error(&e) {
                    CodexError {
                        status: 0,
                        message: format!("Transport error: {e}"),
                        detail: None,
                        retry_after: None,
                        usage_limit: None,
                        origin: CodexErrorOrigin::Http,
                    }
                } else {
                    CodexError {
                        status: 0,
                        message: format!("Network error: {e}"),
                        detail: None,
                        retry_after: None,
                        usage_limit: None,
                        origin: CodexErrorOrigin::Http,
                    }
                }
            })?;

        performance::record(|m, elapsed| {
            m.response_headers.get_or_insert(elapsed);
        });
        Ok((resp, started_at))
    }

    async fn collect_http_response(
        &self,
        mut resp: reqwest::Response,
        started_at: Instant,
        ctx: &RequestContext,
    ) -> Result<CodexResponse, CodexError> {
        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let mut body_bytes = Vec::new();
        let mut response_started = false;
        loop {
            let chunk = tokio::time::timeout(
                Duration::from_millis(self.body_idle_timeout_ms),
                resp.chunk(),
            )
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for the next Codex response body chunk",
                    self.body_idle_timeout_ms
                ),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| CodexError {
                status: 0,
                message: format!("Transport error reading Codex response body: {e}"),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?;

            let Some(chunk) = chunk else {
                break;
            };
            if !response_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.generation_started(&ctx.req_id);
                }
                response_started = true;
            }
            body_bytes.extend_from_slice(&chunk);
        }

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_upstream_response_capture(
                traffic,
                status,
                started_at.elapsed(),
                &headers,
                &body_bytes,
            );
        }

        Ok(CodexResponse {
            body: body_bytes,
            status,
            headers,
            transport: ActualTransport::Http,
        })
    }

    async fn attempt_post_search(
        &self,
        auth: &StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
    ) -> Result<CodexResponse, CodexError> {
        let url = search_endpoint(&self.base_url);
        let headers = build_codex_search_headers(auth, ctx)?;

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_codex_http_request_capture(traffic, &url, &headers, body_json);
        }

        let mut request = self.client.post(&url);
        for (key, value) in headers.iter() {
            request = request.header(key.as_str(), value.as_bytes());
        }
        let started_at = Instant::now();
        let mut response = tokio::time::timeout(
            Duration::from_millis(self.header_timeout_ms),
            request.body(body_json.to_string()).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex search response headers",
                self.header_timeout_ms
            ),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|e| CodexError {
            status: 0,
            message: format!("Codex search network error: {e}"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        })?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(key, value)| {
                (
                    key.to_string(),
                    value.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let mut body = Vec::new();
        let mut response_started = false;
        loop {
            let chunk = tokio::time::timeout(
                Duration::from_millis(self.body_idle_timeout_ms),
                response.chunk(),
            )
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for the next Codex search response body chunk",
                    self.body_idle_timeout_ms
                ),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| CodexError {
                status: 0,
                message: format!("Transport error reading Codex search response body: {e}"),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                usage_limit: None,
                origin: CodexErrorOrigin::Http,
            })?;
            let Some(chunk) = chunk else {
                break;
            };
            if !response_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.generation_started(&ctx.req_id);
                }
                response_started = true;
            }
            body.extend_from_slice(&chunk);
        }

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_upstream_response_capture(traffic, status, started_at.elapsed(), &headers, &body);
        }

        Ok(CodexResponse {
            body,
            status,
            headers,
            transport: ActualTransport::Http,
        })
    }
}

async fn forward_codex_events(
    mut source: tokio::sync::mpsc::Receiver<Result<serde_json::Value, CodexError>>,
    tx: tokio::sync::mpsc::Sender<Result<serde_json::Value, CodexError>>,
) -> bool {
    loop {
        let item = tokio::select! {
            biased;
            _ = tx.closed() => return false,
            item = source.recv() => item,
        };
        let Some(item) = item else {
            return true;
        };
        if tx.send_bounded(item).await.is_err() {
            return false;
        }
    }
}

fn response_headers(resp: &reqwest::Response) -> Vec<(String, String)> {
    resp.headers()
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_str().unwrap_or("").to_string()))
        .collect()
}

fn codex_event_failure_error(failure: super::events::CodexEventFailure) -> CodexError {
    CodexError {
        status: failure.status,
        message: failure.message.clone(),
        detail: Some(failure.message),
        retry_after: failure.retry_after,
        usage_limit: None,
        origin: CodexErrorOrigin::Http,
    }
}

fn codex_usage_limit_error(limit: CodexUsageLimit, origin: CodexErrorOrigin) -> CodexError {
    CodexError {
        status: 429,
        message: limit.message.clone(),
        detail: Some(limit.message.clone()),
        retry_after: None,
        usage_limit: Some(Box::new(limit)),
        origin,
    }
}

fn presemantic_limit_error(origin: CodexErrorOrigin) -> CodexError {
    CodexError {
        status: 502,
        message: "Codex pre-semantic structural event limit exceeded".to_string(),
        detail: Some("codex_presemantic_limit".to_string()),
        retry_after: None,
        usage_limit: None,
        origin: match origin {
            CodexErrorOrigin::Http => CodexErrorOrigin::HttpProtocol,
            _ => CodexErrorOrigin::WebSocketProtocol,
        },
    }
}

fn retryable_http_stream_error(error: &CodexError) -> bool {
    if error.origin.is_protocol_rejection() {
        return false;
    }
    if error.usage_limit.is_some() {
        return false;
    }
    if should_retry_codex_status(error.status) || is_retryable_transport_error(error) {
        return true;
    }
    if error.status != 0 {
        return false;
    }
    if error.detail.as_deref() == Some("http_response_sse") {
        return error.message != "Codex SSE frame exceeds the size limit";
    }
    let message = error.message.to_ascii_lowercase();
    message.contains("ended before a terminal response event")
        || message.contains("ended with an incomplete frame")
}

#[allow(clippy::too_many_arguments)]
fn log_http_stream_end(
    log: &crate::logging::Logger,
    message: &str,
    req_id: &str,
    started_at: Instant,
    body_bytes: u64,
    body_chunks: u64,
    event_count: u64,
    error: Option<&str>,
    terminal_outcome: Option<&str>,
) {
    let mut fields = serde_json::Map::from_iter([
        ("reqId".to_string(), serde_json::json!(req_id)),
        ("transport".to_string(), serde_json::json!("http")),
        (
            "phase".to_string(),
            serde_json::json!(if terminal_outcome.is_some() {
                "upstream_terminal"
            } else {
                "upstream_stream"
            }),
        ),
        (
            "outcome".to_string(),
            serde_json::json!(terminal_outcome.unwrap_or(
                if message == "codex_http_stream_dropped" {
                    "cancelled"
                } else {
                    "failed"
                }
            )),
        ),
        ("bodyBytes".to_string(), serde_json::json!(body_bytes)),
        ("bodyChunks".to_string(), serde_json::json!(body_chunks)),
        ("eventCount".to_string(), serde_json::json!(event_count)),
        (
            "ms".to_string(),
            serde_json::json!(started_at.elapsed().as_millis()),
        ),
    ]);
    if let Some(error) = error {
        fields.insert("error".to_string(), serde_json::json!(error));
    }
    match message {
        "codex_http_stream_completed" => log.info(message, Some(fields)),
        _ => log.warn(message, Some(fields)),
    }
}

fn write_codex_http_request_capture(
    traffic: &TrafficCapture,
    url: &str,
    headers: &http::HeaderMap,
    body_json: &str,
) {
    let body = serde_json::from_str(body_json).unwrap_or_else(|_| {
        serde_json::json!({
            "unparseable": true,
            "bytes": body_json.len(),
        })
    });
    traffic.write_json("020-upstream-request", &body);
    traffic.write_json(
        "021-upstream-request-metadata",
        &serde_json::json!({
            "provider": "codex",
            "transport": "http",
            "url": url,
            "method": "POST",
            "headers": headers_to_json(headers),
            "size": summarize_json_request_size(&body, body_json),
        }),
    );
}

fn write_live_upstream_response_headers(
    traffic: &TrafficCapture,
    response: &reqwest::Response,
    elapsed: Duration,
) {
    traffic.write_json(
        "030-upstream-response-headers",
        &serde_json::json!({
            "status": response.status().as_u16(),
            "elapsedMs": elapsed.as_millis(),
            "headers": headers_to_json(response.headers()),
        }),
    );
}

fn write_upstream_response_capture(
    traffic: &TrafficCapture,
    status: u16,
    elapsed: Duration,
    headers: &[(String, String)],
    body: &[u8],
) {
    write_upstream_response_headers_capture(traffic, status, elapsed, headers);
    if status >= 400 {
        traffic.write_text("031-upstream-error-body", &String::from_utf8_lossy(body));
    } else {
        traffic.write_bytes("032-upstream-response-body.sse", body);
        write_codex_sse_event_capture(traffic, body);
    }
}

fn write_upstream_response_headers_capture(
    traffic: &TrafficCapture,
    status: u16,
    elapsed: Duration,
    headers: &[(String, String)],
) {
    traffic.write_json(
        "030-upstream-response-headers",
        &serde_json::json!({
            "status": status,
            "elapsedMs": elapsed.as_millis(),
            "headers": headers_to_json_from_pairs(headers),
        }),
    );
}

fn write_codex_http_sse_event_capture(
    traffic: &TrafficCapture,
    event: Option<&str>,
    payload: &serde_json::Value,
) {
    let mut payload = payload.clone();
    if let Some(event) = event
        && let Some(object) = payload.as_object_mut()
    {
        object
            .entry("_sse_event")
            .or_insert_with(|| serde_json::json!(event));
    }
    traffic.write_json_event("040-upstream-event", &payload);
}

fn write_codex_sse_event_capture(traffic: &TrafficCapture, body: &[u8]) {
    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            traffic.write_json_event(
                "040-upstream-event",
                &serde_json::json!({
                    "event": event.event,
                    "data": "[DONE]",
                }),
            );
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(&event.data) {
            Ok(mut value) => {
                if let Some(name) = event.event
                    && let Some(obj) = value.as_object_mut()
                {
                    obj.entry("_sse_event").or_insert(serde_json::json!(name));
                }
                traffic.write_json_event("040-upstream-event", &value);
            }
            Err(_) => {
                traffic.write_json_event(
                    "040-upstream-event",
                    &serde_json::json!({
                        "event": event.event,
                        "unparseable": true,
                        "data": event.data,
                    }),
                );
            }
        }
    }
}

fn headers_to_json(headers: &http::HeaderMap) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in headers.iter() {
        out.insert(
            key.to_string(),
            serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
        );
    }
    serde_json::Value::Object(out)
}

fn headers_to_json_from_pairs(headers: &[(String, String)]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in headers {
        out.insert(key.clone(), serde_json::Value::String(value.clone()));
    }
    serde_json::Value::Object(out)
}

fn summarize_json_request_size(body: &serde_json::Value, body_json: &str) -> serde_json::Value {
    serde_json::json!({
        "bytes": body_json.len(),
        "inputCount": body
            .get("input")
            .and_then(|v| v.as_array())
            .map(|items| items.len()),
        "toolCount": body
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|items| items.len()),
    })
}

fn auth_refresh_error(err: anyhow::Error) -> CodexError {
    CodexError {
        status: 401,
        message: "Unauthorized".to_string(),
        detail: Some(err.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::Auth,
    }
}

fn normalized_response_failure(
    response: &CodexResponse,
) -> Option<super::events::CodexEventFailure> {
    // Preserve the typed event even when WebSocket has projected a non-2xx
    // status; that projection must never replace Unknown/Permanent with 500 retry.
    let mut failure =
        if let Ok(mut payload) = serde_json::from_slice::<serde_json::Value>(&response.body) {
            if !(200..300).contains(&response.status)
                && let Some(object) = payload.as_object_mut()
            {
                object.insert("type".to_string(), serde_json::json!("error"));
                object.insert("status".to_string(), serde_json::json!(response.status));
            }
            super::events::classify_event_failure(&payload)
        } else {
            super::events::first_event_failure(&response.body)
        };
    if failure.is_none() && !(200..300).contains(&response.status) {
        failure = super::events::classify_event_failure(&serde_json::json!({
            "type":"error", "status":response.status,
            "message":codex_status_error_message(&response.body).unwrap_or_else(|| format!("Upstream Codex request failed with status {}",response.status))
        }));
    }
    if let Some(failure) = failure.as_mut() {
        failure.retry_after = response
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("retry-after"))
            .map(|(_, value)| value.clone())
            .or_else(|| failure.retry_after.take());
        failure.request_id = failure.request_id.take().or_else(|| {
            response
                .headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case("x-request-id"))
                .map(|(_, value)| value.clone())
        });
    }
    failure
}

fn codex_status_failure(response: CodexResponse) -> CodexRequestFailure {
    let usage_limit =
        super::events::usage_limit_from_response(&response.body, &response.headers).map(Box::new);
    let classified = normalized_response_failure(&response);
    let retryable = classified
        .as_ref()
        .is_some_and(super::events::CodexEventFailure::retryable)
        && usage_limit.is_none();
    let origin = match response.transport {
        ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
        ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
    };
    let mut error = if let Some(failure) = classified {
        codex_event_failure_error(failure)
    } else {
        CodexError::new(
            response.status,
            codex_status_error_message(&response.body).unwrap_or_else(|| {
                format!(
                    "Upstream Codex request failed with status {}",
                    response.status
                )
            }),
        )
    };
    error.origin = origin;
    error.usage_limit = usage_limit;
    CodexRequestFailure { error, retryable }
}

fn codex_status_error(response: CodexResponse) -> CodexError {
    codex_status_failure(response).error
}

fn write_buffered_failure_diagnostics(
    traffic: &TrafficCapture,
    request_id: &str,
    response: &CodexResponse,
) {
    let source = match response.transport {
        ActualTransport::Http => "buffered_http",
        ActualTransport::WebSocket => "buffered_websocket",
    };
    let payloads =
        if let Ok(mut payload) = serde_json::from_slice::<serde_json::Value>(&response.body) {
            if response.status >= 400
                && let Some(object) = payload.as_object_mut()
            {
                object.insert("type".to_string(), serde_json::json!("error"));
                object.insert("status".to_string(), serde_json::json!(response.status));
                if let Some((_, value)) = response
                    .headers
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case("x-request-id"))
                {
                    object
                        .entry("request_id".to_string())
                        .or_insert_with(|| serde_json::json!(value));
                }
            }
            vec![payload]
        } else {
            parse_sse_events(&response.body)
                .into_iter()
                .filter_map(|event| serde_json::from_str(&event.data).ok())
                .collect()
        };
    let mut semantic_output_seen = false;
    for payload in payloads {
        if let Some(failure) = super::events::classify_event_failure(&payload) {
            traffic.write_json(
                "060-codex-event-error",
                &failure.diagnostics(
                    source,
                    "event_failure",
                    Some(request_id),
                    payload.get("type").and_then(serde_json::Value::as_str),
                    semantic_output_seen,
                ),
            );
        }
        semantic_output_seen |= super::events::classify_stream_event(&payload)
            == super::events::CodexStreamEventKind::Semantic;
    }
}

fn codex_status_error_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            [
                value.pointer("/error/detail"),
                value.get("detail"),
                value.pointer("/error/message"),
                value.get("message"),
            ]
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .find(|message| !message.trim().is_empty())
            .map(super::events::sanitize_error_message)
        })
        .or_else(|| {
            parse_sse_events(body).into_iter().find_map(|event| {
                let payload = serde_json::from_str::<serde_json::Value>(&event.data).ok()?;
                super::events::classify_event_failure(&payload).map(|failure| failure.message)
            })
        })
}

fn should_retry_codex_status(status: u16) -> bool {
    should_retry_status(status) || status == 529
}

fn codex_error_origin_name(origin: CodexErrorOrigin) -> &'static str {
    match origin {
        CodexErrorOrigin::Http => "http",
        CodexErrorOrigin::WebSocket => "websocket",
        CodexErrorOrigin::WebSocketHandshake => "websocket_handshake",
        CodexErrorOrigin::Auth => "auth",
        CodexErrorOrigin::BufferedHttp => "buffered_http",
        CodexErrorOrigin::BufferedWebSocket => "buffered_websocket",
        CodexErrorOrigin::HttpProtocol => "http_protocol",
        CodexErrorOrigin::WebSocketProtocol => "websocket_protocol",
    }
}

fn log_buffered_retry(
    ctx: &RequestContext,
    transport: crate::config::CodexTransport,
    failed_attempt: u32,
    delay_ms: u64,
    status: u16,
    origin: &str,
    reason: &str,
) {
    let mut fields = serde_json::Map::new();
    fields.insert("reqId".into(), serde_json::json!(ctx.req_id));
    fields.insert("transport".into(), serde_json::json!(transport.as_str()));
    fields.insert("failedAttempt".into(), serde_json::json!(failed_attempt));
    fields.insert("nextAttempt".into(), serde_json::json!(failed_attempt + 1));
    fields.insert(
        "maxAttempts".into(),
        serde_json::json!(MAX_BUFFERED_TRANSPORT_ATTEMPTS),
    );
    fields.insert("delayMs".into(), serde_json::json!(delay_ms));
    fields.insert("status".into(), serde_json::json!(status));
    fields.insert("origin".into(), serde_json::json!(origin));
    fields.insert("reason".into(), serde_json::json!(reason));
    create_logger("codex").warn("buffered_transport_retry", Some(fields));
}

fn log_buffered_retry_exhausted(
    ctx: &RequestContext,
    transport: crate::config::CodexTransport,
    status: u16,
    origin: &str,
    reason: &str,
) {
    let mut fields = serde_json::Map::new();
    fields.insert("reqId".into(), serde_json::json!(ctx.req_id));
    fields.insert("transport".into(), serde_json::json!(transport.as_str()));
    fields.insert(
        "attempts".into(),
        serde_json::json!(MAX_BUFFERED_TRANSPORT_ATTEMPTS),
    );
    fields.insert("status".into(), serde_json::json!(status));
    fields.insert("origin".into(), serde_json::json!(origin));
    fields.insert("reason".into(), serde_json::json!(reason));
    create_logger("codex").warn("buffered_transport_retry_exhausted", Some(fields));
}

fn is_retryable_transport_error(err: &CodexError) -> bool {
    if err.origin.is_protocol_rejection() {
        return false;
    }
    if err.detail.as_deref() == Some(super::websocket::WEBSOCKET_AMBIGUOUS_WRITE_DETAIL) {
        return false;
    }
    // Running out of patience for the response headers is not a transport
    // failure. The request reached Codex, which is holding the head while the
    // model reasons, so re-sending it starts that reasoning again from scratch
    // and the new attempt runs out of time exactly like the last one. Wait
    // longer with CCP_CODEX_HEADER_TIMEOUT_MS instead.
    if err.detail.as_deref() == Some(HTTP_RESPONSE_HEADERS_DETAIL) {
        return false;
    }
    if err.origin == CodexErrorOrigin::WebSocketHandshake {
        if err.detail.as_deref() == Some(super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL) {
            return false;
        }
        return err.status == 0 || should_retry_codex_status(err.status);
    }
    if err.detail.as_deref() == Some("websocket_pre_request") {
        return err.status == 0 || should_retry_codex_status(err.status);
    }
    if err.detail.as_deref() == Some(super::websocket::WEBSOCKET_KEEPALIVE_FAILURE_DETAIL) {
        return true;
    }
    if err.status != 0 {
        return false;
    }

    let message = err.message.to_ascii_lowercase();
    message.contains("timed out waiting")
        || message.contains("transport error")
        || message.contains("connection reset")
        || message.contains("connection closed")
        || message.contains("timed out")
        || message.contains("econnreset")
        || message.contains("etimedout")
        || message.contains("broken pipe")
        || message.contains("epipe")
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    let msg = err.to_string().to_lowercase();
    msg.contains("connection reset")
        || msg.contains("connection closed")
        || msg.contains("econnreset")
        || msg.contains("etimedout")
        || msg.contains("epipe")
}

fn should_refresh_after_unauthorized(
    result: &Result<OwnerAwareCodexResponse, CodexError>,
    auth_refresh_attempted: bool,
    transport: crate::config::CodexTransport,
) -> bool {
    if auth_refresh_attempted {
        return false;
    }
    match result {
        Ok(response) => response.status == 401,
        Err(err) => {
            err.status == 401
                && (err.origin != CodexErrorOrigin::WebSocketHandshake
                    || transport == crate::config::CodexTransport::WebSocket)
        }
    }
}

fn should_fallback_to_http(err: &CodexError) -> bool {
    err.origin == CodexErrorOrigin::WebSocketHandshake
        && err.status != http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16()
        && err.detail.as_deref() != Some(super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL)
}

fn should_retry_without_continuation(
    err: &CodexError,
    continuation: Option<&super::continuation::ContinuationReservation>,
) -> bool {
    if continuation
        .and_then(|reservation| reservation.candidate().previous_response_id.as_deref())
        .is_none()
    {
        return false;
    }

    is_continuation_retry_error(err)
}

fn full_context_continuation(
    continuation: Option<&super::continuation::ContinuationReservation>,
) -> Option<super::continuation::ContinuationReservation> {
    continuation.map(super::continuation::ContinuationReservation::full_context_retry)
}

fn abort_live_continuation(continuation: Option<&super::continuation::ContinuationReservation>) {
    if let Some(continuation) = continuation {
        super::continuation::abort_continuation_for_owner(continuation);
    }
}

fn abort_abandoned_live_continuation(
    continuation: Option<&super::continuation::ContinuationReservation>,
    socket_id_publisher: &super::websocket::CodexWebSocketSocketIdPublisher,
) {
    if !socket_id_publisher.is_provider_retry_handoff() {
        abort_live_continuation(continuation);
    }
}

fn invalidate_live_continuation_pool(
    continuation: Option<&super::continuation::ContinuationReservation>,
) {
    let Some(continuation) = continuation else {
        return;
    };
    let Some(owner) = websocket_pool_owner(Some(continuation)) else {
        return;
    };
    super::websocket::invalidate_codex_websocket_pool_turn_for_owner(owner, continuation.turn_id());
}

#[cfg(test)]
fn event_closes_live_retry_window(payload: &serde_json::Value) -> bool {
    super::events::event_commits_output(payload)
}

pub(super) fn is_continuation_retry_error(err: &CodexError) -> bool {
    matches!(
        err.detail.as_deref(),
        Some("previous_response_not_found")
            | Some(super::websocket::WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
            | Some(super::websocket::WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL)
            | Some(super::websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL)
            | Some(super::websocket::WEBSOCKET_KEEPALIVE_FAILURE_DETAIL)
    )
}

fn websocket_pool_owner(
    continuation: Option<&super::continuation::ContinuationReservation>,
) -> Option<&ConversationIdentity> {
    let continuation = continuation?;
    if !continuation.pool_enabled() {
        return None;
    }
    continuation.owner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn diagnostic_outcomes_distinguish_http_terminal_failure_from_completion() {
        for (terminal, outcome) in [
            (
                serde_json::json!({"type":"response.completed","response":{"status":"completed"}}),
                "completed",
            ),
            (
                serde_json::json!({"type":"response.failed","response":{"status":"failed","error":{"code":"server_error"}}}),
                "failed",
            ),
            (
                serde_json::json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}),
                "incomplete",
            ),
            (
                serde_json::json!({"type":"response.done","response":{"status":"failed"}}),
                "failed",
            ),
            (
                serde_json::json!({"type":"response.completed","response":{"status":"cancelled"}}),
                "cancelled",
            ),
            (
                serde_json::json!({"type":"error","error":{"code":"server_error"}}),
                "failed",
            ),
            (
                serde_json::json!({"type":"response.completed","response":{"status":"completed","error":{"code":"server_error"}}}),
                "failed",
            ),
            (serde_json::Value::Null, "failed"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let settings =
                crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                    ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
                    ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
                ]));
            let path = settings.clone().sync_scope(crate::logging::log_file);
            let is_terminal = !terminal.is_null();
            settings.scope(async {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let expected_terminal = terminal.clone();
                let server = tokio::spawn(async move {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    read_http_request(&mut stream).await;
                    // Commit output first so a terminal error cannot be transparently retried.
                    let mut body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"private-output\"}\n\n".to_string();
                    if !terminal.is_null() { body.push_str(&format!("data: {terminal}\n\n")); }
                    stream.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                });
                let client = Arc::new(authenticated_http_test_client(format!("http://{addr}/responses")));
                let mut events = client.stream_codex_http_events(&buffered_test_request(), &http_test_context()).await.unwrap();
                let mut received = Vec::new();
                let mut read_error = false;
                while let Some(event) = events.recv().await {
                    match event { Ok(event) => received.push(event), Err(_) => read_error = true }
                }
                if is_terminal { assert_eq!(received.last(), Some(&expected_terminal)); }
                assert_eq!(read_error, !is_terminal);
                server.await.unwrap();
            }).await;
            assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
            let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let record = records
                .iter()
                .find(|r| {
                    r["msg"]
                        == if is_terminal {
                            "codex_http_stream_completed"
                        } else {
                            "codex_http_stream_failed"
                        }
                })
                .expect("stream-end diagnostics missing");
            assert_eq!(record["fields"]["reqId"], "http-body-test");
            assert_eq!(
                record["fields"]["phase"],
                if is_terminal {
                    "upstream_terminal"
                } else {
                    "upstream_stream"
                }
            );
            assert_eq!(record["fields"]["outcome"], outcome);
            assert_eq!(record["fields"]["transport"], "http");
            assert_eq!(
                record["fields"]["eventCount"],
                if is_terminal { 2 } else { 1 }
            );
            assert!(!record.to_string().contains("private-output"));
        }
    }

    #[tokio::test]
    async fn dropping_consumer_during_reused_turn_closes_socket_promptly() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/responses", listener.local_addr().unwrap());
        let (request_sent_tx, request_sent_rx) = tokio::sync::oneshot::channel();
        let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
        let peer = tokio::spawn(async move {
            let (io, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(io).await.unwrap();
            next_websocket_json(&mut websocket).await;
            send_completed_websocket_response(&mut websocket, "first").await;
            next_websocket_json(&mut websocket).await;
            request_sent_tx.send(()).unwrap();
            // Withhold the next response on the reused socket.
            assert!(websocket.next().await.is_none_or(|frame| frame.is_err()));
            closed_tx.send(()).unwrap();
        });
        let client = Arc::new(authenticated_http_test_client(endpoint));
        let owner = ConversationIdentity::Main("warm-turn-cancel".into());
        let body = buffered_test_request();
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &body, false, true);
        let mut first = client
            .stream_codex_websocket_events_for_owner(
                &body,
                &http_test_context(),
                Some(&reservation),
            )
            .await
            .unwrap();
        while first.recv().await.is_some() {}
        first.accept_terminal();
        drop(first);
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &body, false, true);
        let second = client
            .stream_codex_websocket_events_for_owner(
                &body,
                &http_test_context(),
                Some(&reservation),
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), request_sent_rx)
            .await
            .unwrap()
            .unwrap();
        drop(second);
        tokio::time::timeout(Duration::from_secs(1), closed_rx)
            .await
            .expect("consumer cancellation must close the reused active socket promptly")
            .unwrap();
        assert!(!super::super::continuation::is_current_turn_for_owner(
            &reservation
        ));
        peer.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn relay_cancellation_closes_silent_upstream() {
        let (source_tx, source) = tokio::sync::mpsc::channel(64);
        let (tx, downstream) = tokio::sync::mpsc::channel(64);
        let relay = tokio::spawn(forward_codex_events(source, tx));
        drop(downstream);
        tokio::task::yield_now().await;
        assert!(
            relay.is_finished(),
            "relay must observe consumer close during upstream read"
        );
        assert!(source_tx.is_closed());
        relay.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn relay_continuous_full_queue_expires_without_fabricated_terminal() {
        let (source_tx, source) = tokio::sync::mpsc::channel(64);
        let (tx, mut downstream) = tokio::sync::mpsc::channel(64);
        for _ in 0..64 {
            tx.send(Ok(
                serde_json::json!({"type":"response.output_text.delta","delta":"x"}),
            ))
            .await
            .unwrap();
        }
        source_tx
            .send(Ok(serde_json::json!({"type":"response.completed"})))
            .await
            .unwrap();
        let relay = tokio::spawn(forward_codex_events(source, tx));
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(299)).await;
        assert!(!relay.is_finished());
        tokio::time::advance(Duration::from_secs(2)).await;
        tokio::task::yield_now().await;
        assert!(
            relay.is_finished(),
            "continuous downstream stall must dispose relay"
        );
        assert!(source_tx.is_closed());
        while let Some(event) = downstream.recv().await {
            assert_eq!(event.unwrap()["type"], "response.output_text.delta");
        }
        relay.await.unwrap();
    }

    fn test_continuation(
        owner: Option<ConversationIdentity>,
        turn_id: Option<u64>,
        previous_response_id: Option<&str>,
        origin_socket_id: Option<u64>,
        disabled_reason: Option<&str>,
    ) -> super::super::continuation::ContinuationReservation {
        super::super::continuation::ContinuationReservation::new(
            super::super::continuation::ContinuationCandidate {
                turn_id,
                previous_response_id: previous_response_id.map(str::to_string),
                input_delta: None,
                input_delta_count: 1,
                disabled_reason: disabled_reason.map(str::to_string),
            },
            owner,
            origin_socket_id,
        )
    }

    #[test]
    fn http_error_message_prefers_nonempty_detail_then_message() {
        assert_eq!(
            codex_status_error_message(
                br#"{"detail":"specific detail","message":"generic message"}"#
            )
            .as_deref(),
            Some("specific detail")
        );
        assert_eq!(
            codex_status_error_message(
                br#"{"error":{"message":"  "},"message":"synthetic message"}"#
            )
            .as_deref(),
            Some("synthetic message")
        );
    }

    #[test]
    fn normalizes_supported_proxy_urls() {
        assert_eq!(
            normalize_proxy_url("127.0.0.1:8080").as_deref(),
            Some("http://127.0.0.1:8080/")
        );
        assert_eq!(
            normalize_proxy_url("https://user:pass@proxy.example:8443").as_deref(),
            Some("https://user:pass@proxy.example:8443/")
        );
        for scheme in ["socks4", "socks4a"] {
            let proxy = format!("{scheme}://proxy.example:1080");
            assert_eq!(normalize_proxy_url(&proxy), Some(proxy));
        }
        for scheme in ["socks5", "socks5h"] {
            let proxy = format!("{scheme}://user:pass@proxy.example:1080");
            assert_eq!(normalize_proxy_url(&proxy), Some(proxy));
        }
    }

    #[test]
    fn rejects_malformed_or_unsupported_proxy_urls() {
        assert!(normalize_proxy_url("http://").is_none());
        assert!(normalize_proxy_url("ftp://proxy.example:21").is_none());
        assert!(normalize_proxy_url("socks4://user@proxy.example:1080").is_none());
        assert!(normalize_proxy_url("socks4a://user:pass@proxy.example:1080").is_none());
    }

    #[test]
    fn custom_client_auto_fallback_tracks_effective_proxy_route() {
        let proxy = "http://proxy.example:8080";
        let proxied =
            super::super::websocket::WebSocketProxyConfig::new(None, Some(proxy), None, None);
        assert!(!custom_client_auto_http_fallback_enabled(
            "https://codex.invalid/responses",
            &proxied
        ));

        let bypassed = super::super::websocket::WebSocketProxyConfig::new(
            None,
            Some(proxy),
            None,
            Some("codex.invalid"),
        );
        assert!(custom_client_auto_http_fallback_enabled(
            "https://codex.invalid/responses",
            &bypassed
        ));
    }

    fn http_test_auth() -> StoredAuth {
        StoredAuth {
            access: "test".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        }
    }

    fn http_test_context() -> RequestContext {
        RequestContext {
            req_id: "http-body-test".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        }
    }

    fn http_test_client(base_url: String, body_idle_timeout_ms: u64) -> CodexHttpClient {
        CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            base_url,
            100,
            body_idle_timeout_ms,
        )
    }

    fn buffered_test_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-5.6-sol".into(),
            instructions: None,
            input: vec![],
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: super::super::translate::request::ResponsesText {
                verbosity: None,
                format: None,
            },
            reasoning: None,
        }
    }

    fn buffered_request_with_texts(texts: &[&str]) -> ResponsesRequest {
        let mut request = buffered_test_request();
        request.input = texts
            .iter()
            .map(
                |text| super::super::translate::request::ResponsesInputItem::Message {
                    role: "user".to_string(),
                    content: vec![
                        super::super::translate::request::ResponsesContentPart::InputText {
                            text: (*text).to_string(),
                        },
                    ],
                },
            )
            .collect();
        request
    }

    #[tokio::test]
    async fn stream_handoffs_share_full_history_and_release_it_on_cancellation() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (payload_tx, payload_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            payload_tx
                .send(next_websocket_json(&mut websocket).await)
                .unwrap();
            release_rx.await.unwrap();
        });
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let request = Arc::new(buffered_request_with_texts(&["original", "last"]));
        let weak = Arc::downgrade(&request);
        let events = client
            .stream_codex_websocket_events_shared_for_owner(
                Arc::clone(&request),
                &http_test_context(),
                None,
            )
            .await
            .unwrap();
        let payload = payload_rx.await.unwrap();
        assert_eq!(payload["input"][0]["content"][0]["text"], "original");
        assert_eq!(payload["input"][1]["content"][0]["text"], "last");
        assert_eq!(
            Arc::strong_count(&request),
            2,
            "reader retains the same immutable history allocation"
        );
        drop(request);
        assert!(weak.upgrade().is_some());
        drop(events);
        tokio::time::timeout(Duration::from_secs(1), async {
            while weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled reader releases request history");
        release_tx.send(()).unwrap();
        server.await.unwrap();
    }

    async fn next_websocket_json(
        websocket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    ) -> serde_json::Value {
        loop {
            match websocket.next().await {
                Some(Ok(tokio_tungstenite::tungstenite::Message::Ping(payload))) => {
                    websocket
                        .send(tokio_tungstenite::tungstenite::Message::Pong(payload))
                        .await
                        .unwrap();
                }
                Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                    return serde_json::from_str(&text).unwrap();
                }
                other => panic!("unexpected WebSocket request frame: {other:?}"),
            }
        }
    }

    async fn send_completed_websocket_response(
        websocket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        response_id: &str,
    ) {
        websocket
            .send(tokio_tungstenite::tungstenite::Message::Text(
                serde_json::json!({
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "output": []
                    }
                })
                .to_string(),
            ))
            .await
            .unwrap();
    }

    async fn send_nested_previous_response_missing(
        websocket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    ) {
        websocket
            .send(tokio_tungstenite::tungstenite::Message::Text(
                serde_json::json!({
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "previous_response_not_found",
                            "message": "Previous response not found"
                        }
                    }
                })
                .to_string(),
            ))
            .await
            .unwrap();
    }

    fn authenticated_http_test_client(base_url: String) -> CodexHttpClient {
        let client = http_test_client(base_url, 100);
        client.auth_manager().set_test_auth(http_test_auth());
        client
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read > 0, "request ended before its body was complete");
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }

    #[tokio::test]
    async fn image_request_refreshes_once_after_unauthorized() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let server_client = client.clone();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut stream).await;
                let request = String::from_utf8_lossy(&request);
                if attempt == 0 {
                    assert!(request.contains("authorization: Bearer test"));
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "rotated".into(),
                        refresh: "rotated-refresh".into(),
                        account_id: Some("acct-rotated".into()),
                        expires: u64::MAX,
                    });
                    stream
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    assert!(request.contains("authorization: Bearer rotated"));
                    assert!(request.contains("chatgpt-account-id: acct-rotated"));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let response = client
            .post_image_json(
                &format!("http://{addr}"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn image_request_does_not_retry_server_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _request = read_http_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(50), listener.accept())
                    .await
                    .is_err()
            );
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let response = client
            .post_image_json(
                &format!("http://{addr}"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn image_request_uses_fixed_path_oauth_and_json_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let header_end = request
                .windows(4)
                .position(|part| part == b"\r\n\r\n")
                .unwrap();
            let headers = String::from_utf8_lossy(&request[..header_end]);
            assert!(headers.starts_with("POST /root/images/generations HTTP/1.1"));
            assert!(headers.contains("authorization: Bearer test"));
            assert!(headers.contains("chatgpt-account-id: acct"));
            let body: serde_json::Value =
                serde_json::from_slice(&request[header_end + 4..]).unwrap();
            assert_eq!(body["model"], "gpt-image-2");
            assert_eq!(body["prompt"], "draw a fox");
            let response = br#"{"created":1,"data":[{"b64_json":"aW1n"}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            stream.write_all(response).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let response = client
            .post_image_json(
                &format!("http://{addr}/root"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw a fox"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.unwrap();
    }

    async fn write_http_chunk(stream: &mut tokio::net::TcpStream, body: &[u8]) {
        stream
            .write_all(format!("{:x}\r\n", body.len()).as_bytes())
            .await
            .unwrap();
        stream.write_all(body).await.unwrap();
        stream.write_all(b"\r\n").await.unwrap();
        stream.flush().await.unwrap();
    }

    #[test]
    fn http_sse_decoder_handles_fragmented_crlf_and_done_marker() {
        let mut decoder = HttpSseDecoder::default();
        assert!(
            decoder
                .push(b"event: response.output_text.delta\r")
                .unwrap()
                .is_empty()
        );
        let events = decoder
            .push(b"\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\r\n\r\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].event.as_deref(),
            Some("response.output_text.delta")
        );
        assert_eq!(
            events[0]
                .payload
                .as_ref()
                .and_then(|payload| payload.get("delta"))
                .and_then(|value| value.as_str()),
            Some("ok")
        );

        let done = decoder.push(b"data: [DONE]\n\n").unwrap();
        assert_eq!(done.len(), 1);
        assert!(done[0].payload.is_none());
        decoder.finish().unwrap();
    }

    #[test]
    fn http_sse_size_limit_is_not_retryable() {
        assert!(!retryable_http_stream_error(&http_sse_error(
            "Codex SSE frame exceeds the size limit",
        )));
        assert!(retryable_http_stream_error(&http_sse_error(
            "Codex SSE frame contains invalid JSON",
        )));
        assert!(retryable_http_stream_error(&http_sse_error(
            "Codex SSE frame contains invalid UTF-8",
        )));
    }

    #[tokio::test]
    async fn http_structural_overflow_is_explicit_and_never_retried() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let mut body = String::new();
            for index in 0..1025 {
                body.push_str(&format!("data: {{\"type\":\"response.output_item.added\",\"output_index\":{index},\"item\":{{\"type\":\"message\"}}}}\n\n"));
            }
            body.push_str("data: {\"type\":\"response.completed\",\"response\":{}}\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(100), listener.accept())
                    .await
                    .is_err()
            );
        });
        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let mut events = Arc::new(client)
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
            .unwrap();
        let mut failure = None;
        while let Some(event) = events.recv().await {
            match event {
                Err(error) => {
                    failure = Some(error);
                    break;
                }
                Ok(value) => assert_ne!(
                    value["type"], "response.completed",
                    "overflow fabricated success"
                ),
            }
        }
        let error = failure.expect("bounded structural staging must fail explicitly");
        assert_eq!(error.detail.as_deref(), Some("codex_presemantic_limit"));
        assert!(!retryable_http_stream_error(&error));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn http_stream_forwards_event_before_terminal_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            write_http_chunk(
                &mut stream,
                b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"hello\"}\n\n",
            )
            .await;
            release_rx.await.unwrap();
            write_http_chunk(
                &mut stream,
                b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{}}}\n\n",
            )
            .await;
            stream.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 1_000));
        client.auth_manager().set_test_auth(http_test_auth());
        let mut events = client
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
            .unwrap();

        let synthetic = events.recv().await.unwrap().unwrap();
        assert_eq!(
            synthetic.get("type").and_then(|value| value.as_str()),
            Some("keepalive")
        );
        let first_upstream = tokio::time::timeout(Duration::from_millis(200), events.recv())
            .await
            .expect("first upstream event must arrive before the response completes")
            .unwrap()
            .unwrap();
        assert_eq!(
            first_upstream.get("type").and_then(|value| value.as_str()),
            Some("response.output_text.delta")
        );

        release_tx.send(()).unwrap();
        let terminal = events.recv().await.unwrap().unwrap();
        assert_eq!(
            terminal.get("type").and_then(|value| value.as_str()),
            Some("response.completed")
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn http_live_terminal_waits_for_eof_and_rejects_all_trailing_data() {
        for case in [
            "clean",
            "duplicate",
            "late_text",
            "malformed",
            "truncated",
            "coalesced_malformed",
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (sent_tx, sent_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_http_request(&mut socket).await;
                socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await.unwrap();
                let prefix = [
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","content":[]}}),
                    serde_json::json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"answer"}),
                    serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m","content":[{"type":"output_text","text":"answer"}]}}),
                ].iter().map(|event| format!("data: {event}\n\n")).collect::<String>();
                write_http_chunk(&mut socket, prefix.as_bytes()).await;
                let terminal = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\",\"usage\":{}}}\n\n";
                let terminal = if case == "coalesced_malformed" {
                    format!("{terminal}data: not-json\n\n")
                } else {
                    terminal.to_string()
                };
                write_http_chunk(&mut socket, terminal.as_bytes()).await;
                sent_tx.send(()).unwrap();
                release_rx.await.unwrap();
                let tail = match case {
                    "duplicate" => {
                        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                    }
                    "late_text" => {
                        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"late\"}\n\n"
                    }
                    "malformed" => "data: not-json\n\n",
                    _ => "",
                };
                if !tail.is_empty() {
                    write_http_chunk(&mut socket, tail.as_bytes()).await;
                }
                if case != "truncated" {
                    let _ = socket.write_all(b"0\r\n\r\n").await;
                }
            });
            let client = Arc::new(http_test_client(
                format!("http://{address}/responses"),
                5_000,
            ));
            client.auth_manager().set_test_auth(http_test_auth());
            let mut events = client
                .stream_codex_http_events_for_owner(&buffered_test_request(), &http_test_context())
                .await
                .unwrap();
            let mut translator =
                crate::providers::codex::translate::live_stream::LiveStreamTranslator::new(
                    "m", "model",
                );
            let mut wire = Vec::new();
            for _ in 0..4 {
                let payload = tokio::time::timeout(Duration::from_secs(5), events.recv())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(!super::super::events::event_is_terminal(&payload), "{case}");
                wire.extend(translator.accept(&payload, None).unwrap());
            }
            sent_rx.await.unwrap();
            if case != "coalesced_malformed" {
                assert!(
                    tokio::time::timeout(Duration::from_millis(30), events.recv())
                        .await
                        .is_err(),
                    "terminal escaped before body EOF: {case}"
                );
            }
            release_tx.send(()).unwrap();
            let mut failure = None;
            while let Some(event) = tokio::time::timeout(Duration::from_secs(5), events.recv())
                .await
                .unwrap()
            {
                match event {
                    Ok(payload) => wire.extend(translator.accept(&payload, None).unwrap()),
                    Err(error) => {
                        wire.extend(translator.error_chunk(&error.message, "api_error", None));
                        failure = Some(error);
                        break;
                    }
                }
            }
            let wire = String::from_utf8(wire).unwrap();
            assert!(wire.contains("answer"), "{case}");
            assert_eq!(wire.contains("message_stop"), case == "clean", "{case}");
            assert_eq!(failure.is_some(), case != "clean", "{case}");
            assert!(
                !events.had_intervening_failure(),
                "committed text must not be replayed: {case}"
            );
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn http_stream_header_only_usage_limit_uses_response_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await;
            let body = br#"data: {"type":"error","status_code":429,"error":{"type":"usage_limit_reached","message":"weekly \"limit\" reached","resets_in_seconds":90}}

"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nx-codex-secondary-reset-after-seconds: 90\r\nx-codex-secondary-reset-at: 1789466238\r\nx-codex-secondary-window-minutes: 10080\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
        });

        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 1_000));
        client.auth_manager().set_test_auth(http_test_auth());
        let mut events = client
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
            .unwrap();

        assert_eq!(
            events.recv().await.unwrap().unwrap().get("type"),
            Some(&serde_json::json!("keepalive"))
        );
        let error = events.recv().await.unwrap().unwrap_err();
        assert_eq!(error.status, 429);
        let limit = error.usage_limit.expect("usage limit");
        assert_eq!(limit.message, "weekly \"limit\" reached");
        assert_eq!(limit.resets_at, Some(1789466238));
        assert_eq!(limit.window, Some(CodexLimitWindow::SevenDay));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn http_stream_retries_with_current_attempt_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_http_request(&mut stream).await;
                let (body, headers) = if attempt == 0 {
                    (
                        br#"data: {"type":"response.failed","status_code":503,"error":{"message":"server error","retry_after":"0"}}

"#.as_slice(),
                        "",
                    )
                } else {
                    (
                        br#"data: {"type":"error","status_code":429,"error":{"type":"usage_limit_reached","message":"weekly limit reached","resets_in_seconds":90}}

"#.as_slice(),
                        "x-codex-secondary-reset-after-seconds: 90\r\nx-codex-secondary-reset-at: 1789466238\r\nx-codex-secondary-window-minutes: 10080\r\n",
                    )
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(head.as_bytes()).await.unwrap();
                stream.write_all(body).await.unwrap();
            }
        });

        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 1_000));
        client.auth_manager().set_test_auth(http_test_auth());
        let mut events = client
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
            .unwrap();
        let _ = events.recv().await.unwrap().unwrap();
        let error = events.recv().await.unwrap().unwrap_err();
        let limit = error.usage_limit.expect("usage limit");
        assert_eq!(limit.resets_at, Some(1789466238));
        assert_eq!(limit.window, Some(CodexLimitWindow::SevenDay));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn http_stream_does_not_resend_after_a_header_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let mut attempts = 0_u32;
            // Hold every accepted connection open. Closing one would look like
            // a transport failure and mask the header timeout under test.
            let mut held = Vec::new();
            while let Ok(Ok((mut stream, _))) =
                tokio::time::timeout(Duration::from_millis(400), listener.accept()).await
            {
                let mut request = [0_u8; 16 * 1024];
                assert!(stream.read(&mut request).await.unwrap() > 0);
                attempts += 1;
                held.push(stream);
            }
            attempts
        });

        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 1_000));
        client.auth_manager().set_test_auth(http_test_auth());
        let started_at = Instant::now();
        let error = match client
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
        {
            Ok(_) => panic!("a silent upstream must fail the request"),
            Err(error) => error,
        };
        let elapsed = started_at.elapsed();

        assert_eq!(error.status, 0);
        assert!(
            error
                .message
                .contains("Timed out waiting 100ms for Codex response headers"),
            "{}",
            error.message
        );
        assert_eq!(
            error.detail.as_deref(),
            Some(HTTP_RESPONSE_HEADERS_DETAIL),
            "the header timeout must stay distinguishable from a transport failure"
        );
        assert!(
            elapsed < Duration::from_millis(crate::retry::RETRY_INITIAL_DELAY_MS),
            "the header timeout must fail before the first retry backoff, took {elapsed:?}"
        );
        assert_eq!(
            server.await.unwrap(),
            1,
            "a header timeout must not re-send the request"
        );
    }

    #[tokio::test]
    async fn http_stream_bounds_initial_status_retries() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let mut attempts = 0_u32;
            while let Ok(Ok((mut stream, _))) =
                tokio::time::timeout(Duration::from_millis(100), listener.accept()).await
            {
                let mut request = [0_u8; 16 * 1024];
                assert!(stream.read(&mut request).await.unwrap() > 0);
                attempts += 1;
                stream
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nretry-after: 0\r\nconnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
            }
            attempts
        });

        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 1_000));
        client.auth_manager().set_test_auth(http_test_auth());
        let error = match client
            .stream_codex_http_events(&buffered_test_request(), &http_test_context())
            .await
        {
            Ok(_) => panic!("retryable status must exhaust with an error"),
            Err(error) => error,
        };

        assert_eq!(error.status, 503);
        assert_eq!(
            server.await.unwrap(),
            MAX_BUFFERED_TRANSPORT_ATTEMPTS,
            "initial status failures must share the HTTP stream retry budget"
        );
    }

    #[tokio::test]
    async fn native_responses_replaces_auth_and_preserves_json_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            let read = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]);
            assert!(request.contains("authorization: Bearer test"));
            assert!(request.contains("chatgpt-account-id: acct"));
            assert!(request.contains("accept: application/json"));
            assert!(request.contains(r#""extra":{"kept":true}"#));
            let body = br#"{"id":"resp_native","object":"response"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/v1/responses"));
        let response = client
            .post_native_responses(
                &serde_json::json!({
                    "model": "gpt-5.4",
                    "input": "hello",
                    "stream": false,
                    "extra": {"kept": true}
                }),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response.bytes().await.unwrap(),
            br#"{"id":"resp_native","object":"response"}"#.as_slice()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_responses_refreshes_once_before_returning_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/v1/responses"
        )));
        let server_client = client.clone();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 16 * 1024];
                let read = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                if attempt == 0 {
                    assert!(request.contains("authorization: Bearer test"));
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "rotated".into(),
                        refresh: "rotated-refresh".into(),
                        account_id: Some("acct-rotated".into()),
                        expires: u64::MAX,
                    });
                    stream
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    assert!(request.contains("authorization: Bearer rotated"));
                    assert!(request.contains("chatgpt-account-id: acct-rotated"));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let response = client
            .post_native_responses(
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.bytes().await.unwrap(), b"{}".as_slice());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_responses_does_not_follow_redirects() {
        let source = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let source_addr = source.local_addr().unwrap();
        let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target_addr = target.local_addr().unwrap();
        let source_server = tokio::spawn(async move {
            let (mut stream, _) = source.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: http://{target_addr}/stolen\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{source_addr}/v1/responses"));
        let response = client
            .post_native_responses(
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        source_server.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), target.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn buffered_missing_origin_retries_full_context_and_rebinds_exact_socket() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Agent(
            "buffered-recovery-session".to_string(),
            "buffered-recovery-agent".to_string(),
        );
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            request_tx
                .send(next_websocket_json(&mut first_websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut first_websocket, "resp_a").await;
            drop(first_websocket);

            let (second_socket, _) = listener.accept().await.unwrap();
            let mut second_websocket = tokio_tungstenite::accept_async(second_socket)
                .await
                .unwrap();
            request_tx
                .send(next_websocket_json(&mut second_websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut second_websocket, "resp_b").await;
            request_tx
                .send(next_websocket_json(&mut second_websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut second_websocket, "resp_c").await;
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let context = http_test_context();
        let first_request = buffered_request_with_texts(&["one"]);
        let first_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &first_request,
            true,
        );
        let first_response = client
            .post_codex_with_transport(
                &first_request,
                &context,
                Some(&first_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
            .unwrap();
        let first_socket_id = first_response
            .socket_id
            .expect("first socket must be reusable");
        super::super::update_continuation_from_upstream(
            &context,
            &first_candidate,
            &first_request,
            &first_response.body,
            first_response.socket_id,
        );

        let second_request = buffered_request_with_texts(&["one", "two"]);
        let second_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &second_request,
            true,
        );
        assert_eq!(second_candidate.origin_socket_id(), Some(first_socket_id));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let second_response = client
            .post_codex_with_transport(
                &second_request,
                &context,
                Some(&second_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
            .unwrap();
        let second_socket_id = second_response
            .socket_id
            .expect("full-context retry socket must be reusable");
        assert_ne!(second_socket_id, first_socket_id);
        super::super::update_continuation_from_upstream(
            &context,
            &second_candidate,
            &second_request,
            &second_response.body,
            second_response.socket_id,
        );

        let third_request = buffered_request_with_texts(&["one", "two", "three"]);
        let third_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &third_request,
            true,
        );
        assert_eq!(
            third_candidate.candidate().previous_response_id.as_deref(),
            Some("resp_b")
        );
        assert_eq!(third_candidate.origin_socket_id(), Some(second_socket_id));
        let third_response = client
            .post_codex_with_transport(
                &third_request,
                &context,
                Some(&third_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
            .unwrap();
        assert_eq!(third_response.socket_id, Some(second_socket_id));

        let first_payload = request_rx.recv().await.unwrap();
        let retry_payload = request_rx.recv().await.unwrap();
        let continued_payload = request_rx.recv().await.unwrap();
        assert!(first_payload.get("previous_response_id").is_none());
        assert_eq!(first_payload["input"].as_array().unwrap().len(), 1);
        assert!(retry_payload.get("previous_response_id").is_none());
        assert_eq!(retry_payload["input"].as_array().unwrap().len(), 2);
        assert_eq!(continued_payload["previous_response_id"], "resp_b");
        assert_eq!(continued_payload["input"].as_array().unwrap().len(), 1);
        server.await.unwrap();

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&third_candidate);
    }

    #[tokio::test]
    async fn buffered_nested_missing_response_retries_once_without_stale_previous_id() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("buffered-nested-recovery-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            request_tx
                .send(next_websocket_json(&mut first_websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut first_websocket, "resp_nested_origin").await;
            request_tx
                .send(next_websocket_json(&mut first_websocket).await)
                .unwrap();
            send_nested_previous_response_missing(&mut first_websocket).await;

            let (retry_socket, _) = listener.accept().await.unwrap();
            let mut retry_websocket = tokio_tungstenite::accept_async(retry_socket).await.unwrap();
            request_tx
                .send(next_websocket_json(&mut retry_websocket).await)
                .unwrap();
            send_nested_previous_response_missing(&mut retry_websocket).await;
            assert!(
                tokio::time::timeout(Duration::from_millis(1_200), listener.accept())
                    .await
                    .is_err(),
                "nested missing-response recovery must not open a third socket"
            );
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let context = http_test_context();
        let first_request = buffered_request_with_texts(&["one"]);
        let first_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &first_request,
            true,
        );
        let first_response = client
            .post_codex_with_transport(
                &first_request,
                &context,
                Some(&first_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
            .unwrap();
        super::super::update_continuation_from_upstream(
            &context,
            &first_candidate,
            &first_request,
            &first_response.body,
            first_response.socket_id,
        );

        let second_request = buffered_request_with_texts(&["one", "two"]);
        let second_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &second_request,
            true,
        );
        assert_eq!(
            second_candidate.candidate().previous_response_id.as_deref(),
            Some("resp_nested_origin")
        );
        let error = match client
            .post_codex_with_transport(
                &second_request,
                &context,
                Some(&second_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
        {
            Ok(_) => {
                panic!("the bounded full-context retry must surface a repeated nested failure")
            }
            Err(error) => error,
        };
        assert_eq!(error.detail.as_deref(), Some("previous_response_not_found"));

        let first_payload = request_rx.recv().await.unwrap();
        let continued_payload = request_rx.recv().await.unwrap();
        let retry_payload = request_rx.recv().await.unwrap();
        assert!(first_payload.get("previous_response_id").is_none());
        assert_eq!(
            continued_payload["previous_response_id"],
            "resp_nested_origin"
        );
        assert_eq!(continued_payload["input"].as_array().unwrap().len(), 1);
        assert!(retry_payload.get("previous_response_id").is_none());
        assert_eq!(retry_payload["input"].as_array().unwrap().len(), 2);
        assert!(request_rx.try_recv().is_err());
        server.await.unwrap();

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&second_candidate);
    }

    #[tokio::test]
    async fn live_nested_missing_response_retries_once_without_stale_previous_id() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Agent(
            "live-nested-recovery-session".to_string(),
            "live-nested-recovery-agent".to_string(),
        );
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            request_tx
                .send(next_websocket_json(&mut first_websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut first_websocket, "resp_live_nested_origin")
                .await;
            request_tx
                .send(next_websocket_json(&mut first_websocket).await)
                .unwrap();
            send_nested_previous_response_missing(&mut first_websocket).await;

            let (retry_socket, _) = listener.accept().await.unwrap();
            let mut retry_websocket = tokio_tungstenite::accept_async(retry_socket).await.unwrap();
            request_tx
                .send(next_websocket_json(&mut retry_websocket).await)
                .unwrap();
            send_nested_previous_response_missing(&mut retry_websocket).await;
            assert!(
                tokio::time::timeout(Duration::from_millis(1_200), listener.accept())
                    .await
                    .is_err(),
                "live nested missing-response recovery must not open a third socket"
            );
        });

        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let context = http_test_context();
        let first_request = buffered_request_with_texts(&["one"]);
        let first_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &first_request,
            true,
        );
        let first_response = client
            .post_codex_with_transport(
                &first_request,
                &context,
                Some(&first_candidate),
                crate::config::CodexTransport::WebSocket,
            )
            .await
            .unwrap();
        super::super::update_continuation_from_upstream(
            &context,
            &first_candidate,
            &first_request,
            &first_response.body,
            first_response.socket_id,
        );

        let second_request = buffered_request_with_texts(&["one", "two"]);
        let second_candidate = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &second_request,
            true,
        );
        assert_eq!(
            second_candidate.candidate().previous_response_id.as_deref(),
            Some("resp_live_nested_origin")
        );
        let mut events = client
            .stream_codex_websocket_events_for_owner(
                &second_request,
                &context,
                Some(&second_candidate),
            )
            .await
            .unwrap();
        let error = events.recv().await.unwrap().unwrap_err();
        assert_eq!(error.detail.as_deref(), Some("previous_response_not_found"));
        assert!(events.used_full_context_retry());
        assert_eq!(events.socket_id(), None);

        let first_payload = request_rx.recv().await.unwrap();
        let continued_payload = request_rx.recv().await.unwrap();
        let retry_payload = request_rx.recv().await.unwrap();
        assert!(first_payload.get("previous_response_id").is_none());
        assert_eq!(
            continued_payload["previous_response_id"],
            "resp_live_nested_origin"
        );
        assert_eq!(continued_payload["input"].as_array().unwrap().len(), 1);
        assert!(retry_payload.get("previous_response_id").is_none());
        assert_eq!(retry_payload["input"].as_array().unwrap().len(), 2);
        assert!(request_rx.try_recv().is_err());
        server.await.unwrap();

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&second_candidate);
    }

    #[tokio::test]
    async fn live_missing_origin_retries_once_with_full_context_and_actual_socket() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("live-recovery-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one", "two"]);
        let reserved = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let continuation = super::super::continuation::ContinuationReservation::new(
            super::super::continuation::ContinuationCandidate {
                turn_id: reserved.turn_id(),
                previous_response_id: Some("resp_missing".to_string()),
                input_delta: Some(vec![request.input.last().unwrap().clone()]),
                input_delta_count: 1,
                disabled_reason: None,
            },
            Some(owner.clone()),
            Some(u64::MAX),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (payload_tx, payload_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            payload_tx
                .send(next_websocket_json(&mut websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut websocket, "resp_live_retry").await;
            assert!(
                tokio::time::timeout(Duration::from_millis(100), listener.accept())
                    .await
                    .is_err()
            );
        });

        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let mut events = client
            .stream_codex_websocket_events_for_owner(
                &request,
                &http_test_context(),
                Some(&continuation),
            )
            .await
            .unwrap();
        let terminal = events.recv().await.unwrap().unwrap();
        assert_eq!(terminal["type"], "response.completed");
        assert!(events.used_full_context_retry());
        let socket_id = events
            .socket_id()
            .expect("successful internal retry must publish its socket");
        let payload = payload_rx.await.unwrap();
        assert!(payload.get("previous_response_id").is_none());
        assert_eq!(payload["input"].as_array().unwrap().len(), 2);
        server.await.unwrap();
        assert_eq!(
            super::super::websocket::pooled_socket_id_for_tests(&owner),
            Some(socket_id)
        );

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&reserved);
    }

    #[tokio::test]
    async fn public_ownerless_candidate_never_writes_previous_id_to_websocket() {
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        super::super::websocket::clear_codex_websocket_pool_for_tests();
        let request = buffered_request_with_texts(&["one", "two"]);
        let continuation = super::super::continuation::ContinuationCandidate {
            turn_id: Some(17),
            previous_response_id: Some("resp_unproven".to_string()),
            input_delta: Some(vec![request.input.last().unwrap().clone()]),
            input_delta_count: 1,
            disabled_reason: None,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (payload_tx, payload_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            payload_tx
                .send(next_websocket_json(&mut websocket).await)
                .unwrap();
            send_completed_websocket_response(&mut websocket, "resp_public_full_context").await;
            assert!(
                tokio::time::timeout(Duration::from_millis(100), listener.accept())
                    .await
                    .is_err(),
                "an ownerless stale candidate must use one full-context socket"
            );
        });

        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let mut context = http_test_context();
        context.session_id = Some("must-not-be-derived-as-owner".to_string());
        let mut events = client
            .stream_codex_websocket_events(&request, &context, Some(&continuation))
            .await
            .unwrap();
        let terminal = events.recv().await.unwrap().unwrap();
        assert_eq!(terminal["type"], "response.completed");

        let payload = payload_rx.await.unwrap();
        assert!(payload.get("previous_response_id").is_none());
        assert_eq!(payload["input"].as_array().unwrap().len(), 2);
        server.await.unwrap();
        super::super::websocket::clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn live_missing_origin_does_not_enter_second_full_context_loop() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("live-bounded-recovery-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one", "two"]);
        let reserved = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let continuation = super::super::continuation::ContinuationReservation::new(
            super::super::continuation::ContinuationCandidate {
                turn_id: reserved.turn_id(),
                previous_response_id: Some("resp_missing".to_string()),
                input_delta: Some(vec![request.input.last().unwrap().clone()]),
                input_delta_count: 1,
                disabled_reason: None,
            },
            Some(owner.clone()),
            Some(u64::MAX),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (payload_tx, payload_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            payload_tx
                .send(next_websocket_json(&mut websocket).await)
                .unwrap();
            websocket.close(None).await.unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(150), listener.accept())
                    .await
                    .is_err()
            );
        });

        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let mut events = client
            .stream_codex_websocket_events_for_owner(
                &request,
                &http_test_context(),
                Some(&continuation),
            )
            .await
            .unwrap();
        let error = events.recv().await.unwrap().unwrap_err();
        assert_eq!(
            error.detail.as_deref(),
            Some(super::super::websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL)
        );
        assert!(events.used_full_context_retry());
        assert_eq!(events.socket_id(), None);
        let payload = payload_rx.await.unwrap();
        assert!(payload.get("previous_response_id").is_none());
        assert_eq!(payload["input"].as_array().unwrap().len(), 2);
        server.await.unwrap();

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&reserved);
    }

    #[tokio::test]
    async fn dropping_live_receiver_clears_reserved_turn() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("live-drop-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one"]);
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_websocket_json(&mut websocket).await;
            request_seen_tx.send(()).unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = websocket.close(None).await;
        });

        let events = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )))
        .stream_codex_websocket_events_for_owner(
            &request,
            &http_test_context(),
            Some(&continuation),
        )
        .await
        .unwrap();
        request_seen_rx.await.unwrap();
        drop(events);

        tokio::time::timeout(Duration::from_secs(1), async {
            while super::super::continuation::is_current_turn_for_owner(&continuation) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping the live receiver must clear its reserved turn");
        server.await.unwrap();
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn dropping_retry_handoff_receiver_preserves_reserved_turn() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("live-retry-handoff-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one"]);
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_websocket_json(&mut websocket).await;
            request_seen_tx.send(()).unwrap();
            while websocket.next().await.is_some() {}
            socket_closed_tx.send(()).unwrap();
        });

        let events = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )))
        .stream_codex_websocket_events_for_owner(
            &request,
            &http_test_context(),
            Some(&continuation),
        )
        .await
        .unwrap();
        request_seen_rx.await.unwrap();
        events.mark_provider_retry_handoff();
        drop(events);

        tokio::time::timeout(Duration::from_secs(1), socket_closed_rx)
            .await
            .expect("marked receiver drop must close only the abandoned attempt socket")
            .expect("socket-close acknowledgement sender dropped");
        assert!(super::super::continuation::is_current_turn_for_owner(
            &continuation
        ));
        server.await.unwrap();
        super::super::continuation::abort_continuation_for_owner(&continuation);
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn delayed_retry_handoff_cleanup_preserves_replacement_state_and_socket() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Main("live-retry-cleanup-race-session".to_string());
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one"]);
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (release_replacement_tx, release_replacement_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            let _ = next_websocket_json(&mut first_websocket).await;
            send_completed_websocket_response(&mut first_websocket, "resp_attempt_a").await;
            drop(first_websocket);

            let (replacement_socket, _) = listener.accept().await.unwrap();
            let mut replacement_websocket = tokio_tungstenite::accept_async(replacement_socket)
                .await
                .unwrap();
            let _ = next_websocket_json(&mut replacement_websocket).await;
            send_completed_websocket_response(&mut replacement_websocket, "resp_attempt_b").await;
            let _ = release_replacement_rx.await;
        });
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));

        let mut attempt_a = client
            .stream_codex_websocket_events_for_owner(
                &request,
                &http_test_context(),
                Some(&continuation),
            )
            .await
            .unwrap();
        let terminal_a = attempt_a.recv().await.unwrap().unwrap();
        assert_eq!(terminal_a["response"]["id"], "resp_attempt_a");
        let socket_a = attempt_a.socket_id().expect("attempt A socket ID");
        super::super::websocket::invalidate_codex_websocket_pool_socket(
            &continuation,
            Some(socket_a),
        );

        let mut attempt_b = client
            .stream_codex_websocket_events_for_owner(
                &request,
                &http_test_context(),
                Some(&continuation),
            )
            .await
            .unwrap();
        let terminal_b = attempt_b.recv().await.unwrap().unwrap();
        assert_eq!(terminal_b["response"]["id"], "resp_attempt_b");
        let socket_b = attempt_b.socket_id().expect("attempt B socket ID");
        assert_ne!(socket_a, socket_b);
        super::super::continuation::record_continuation_for_owner(
            &continuation,
            &request,
            Some("resp_attempt_b"),
            Some(socket_b),
            &[],
        );

        let (_handoff_tx, handoff_rx) =
            tokio::sync::mpsc::channel::<Result<serde_json::Value, CodexError>>(1);
        let (handoff_stream, handoff_publisher) =
            super::super::websocket::CodexWebSocketEventStream::pending(handoff_rx);
        handoff_stream.mark_provider_retry_handoff();
        let cleanup_barrier = Arc::new(tokio::sync::Barrier::new(2));
        let cleanup_task_barrier = cleanup_barrier.clone();
        let cleanup_continuation = continuation.clone();
        let cleanup = tokio::spawn(async move {
            cleanup_task_barrier.wait().await;
            super::super::websocket::invalidate_codex_websocket_pool_socket(
                &cleanup_continuation,
                Some(socket_a),
            );
            abort_abandoned_live_continuation(Some(&cleanup_continuation), &handoff_publisher);
        });

        cleanup_barrier.wait().await;
        cleanup.await.unwrap();
        assert!(super::super::continuation::is_current_turn_for_owner(
            &continuation
        ));
        assert!(super::super::continuation::has_continuation_for_owner_for_tests(&owner));
        assert_eq!(
            super::super::websocket::pooled_socket_id_for_tests(&owner),
            Some(socket_b)
        );

        let _ = release_replacement_tx.send(());
        server.await.unwrap();
        super::super::continuation::abort_continuation_for_owner(&continuation);
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn auto_clears_missing_origin_before_ordinary_http_fallback() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = super::super::websocket::lock_codex_websocket_pool_for_tests().await;
        let owner = ConversationIdentity::Agent(
            "auto-recovery-session".to_string(),
            "auto-recovery-agent".to_string(),
        );
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = buffered_request_with_texts(&["one", "two"]);
        let reserved = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let continuation = super::super::continuation::ContinuationReservation::new(
            super::super::continuation::ContinuationCandidate {
                turn_id: reserved.turn_id(),
                previous_response_id: Some("resp_missing".to_string()),
                input_delta: Some(vec![request.input.last().unwrap().clone()]),
                input_delta_count: 1,
                disabled_reason: None,
            },
            Some(owner.clone()),
            Some(u64::MAX),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut websocket, _) = listener.accept().await.unwrap();
            let websocket_request = read_http_request(&mut websocket).await;
            assert!(String::from_utf8_lossy(&websocket_request).starts_with("GET "));
            websocket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            drop(websocket);

            let (mut http, _) = listener.accept().await.unwrap();
            let http_request = read_http_request(&mut http).await;
            let body_start = http_request
                .windows(4)
                .position(|part| part == b"\r\n\r\n")
                .unwrap()
                + 4;
            let body: serde_json::Value =
                serde_json::from_slice(&http_request[body_start..]).unwrap();
            assert!(body.get("previous_response_id").is_none());
            assert_eq!(body["input"].as_array().unwrap().len(), 2);
            let response_body =
                b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http\"}}\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response_body.len()
            );
            http.write_all(response.as_bytes()).await.unwrap();
            http.write_all(response_body).await.unwrap();
        });

        let response = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &request,
                &http_test_context(),
                Some(&continuation),
                crate::config::CodexTransport::Auto,
            )
            .await
            .unwrap();
        assert_eq!(response.transport, ActualTransport::Http);
        assert_eq!(response.socket_id, None);
        server.await.unwrap();

        super::super::websocket::invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&reserved);
    }

    #[tokio::test]
    async fn buffered_http_retries_retryable_status() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 16 * 1024];
                assert!(stream.read(&mut request).await.unwrap() > 0);
                let (status, body): (&str, &[u8]) = if attempt == 0 {
                    ("503 Service Unavailable", b"retry")
                } else {
                    ("200 OK", b"data: keep\n\n")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nretry-after: 0\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.write_all(body).await.unwrap();
            }
        });

        let response = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"data: keep\n\n");
    }

    #[tokio::test]
    async fn standalone_search_posts_json_to_alpha_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let header_end = request
                .windows(4)
                .position(|part| part == b"\r\n\r\n")
                .unwrap();
            let headers = String::from_utf8_lossy(&request[..header_end]);
            assert!(headers.starts_with("POST /alpha/search HTTP/1.1"));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("accept: application/json")
            );
            assert!(headers.contains("authorization: Bearer test"));
            let body: serde_json::Value =
                serde_json::from_slice(&request[header_end + 4..]).unwrap();
            assert_eq!(body["model"], "gpt-5.6-luna");
            assert!(body.get("reasoning").is_none());
            assert_eq!(body["commands"]["search_query"][0]["q"], "find Codex");

            let response = serde_json::to_vec(&serde_json::json!({
                "encrypted_output": "opaque",
                "output": "search output",
                "results": [{
                    "type": "text_result",
                    "ref_id": "turn0search0",
                    "url": "https://example.com",
                    "title": "Example"
                }]
            }))
            .unwrap();
            let response_headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(response_headers.as_bytes()).await.unwrap();
            stream.write_all(&response).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let request = super::super::search::SearchRequest {
            id: "session".to_string(),
            model: "gpt-5.6-luna".to_string(),
            reasoning: None,
            input: None,
            commands: super::super::search::SearchCommands {
                search_query: vec![super::super::search::SearchQuery {
                    q: "find Codex".to_string(),
                }],
            },
            settings: super::super::search::SearchSettings {
                filters: None,
                allowed_callers: vec!["direct"],
                external_web_access: true,
            },
            max_output_tokens: 2_500,
        };
        let response = client
            .post_search(&request, &http_test_context())
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(response.output, "search output");
        assert_eq!(response.results.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn auto_falls_back_to_http_after_statusful_websocket_handshake_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut websocket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            let read = websocket.read(&mut request).await.unwrap();
            assert!(read > 0);
            assert!(
                String::from_utf8_lossy(&request[..read])
                    .to_ascii_lowercase()
                    .contains("upgrade: websocket")
            );
            websocket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 13\r\nconnection: close\r\n\r\npolicy denied",
                )
                .await
                .unwrap();
            drop(websocket);

            let (mut http, _) = listener.accept().await.unwrap();
            let read = http.read(&mut request).await.unwrap();
            assert!(read > 0);
            assert!(String::from_utf8_lossy(&request[..read]).starts_with("POST "));
            let body = b"data: keep\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            http.write_all(response.as_bytes()).await.unwrap();
            http.write_all(body).await.unwrap();
        });

        let response = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Auto,
            )
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"data: keep\n\n");
    }

    #[tokio::test]
    async fn over_budget_retry_after_stops_without_replay() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 4\r\nretry-after: 120\r\nconnection: close\r\n\r\nstop",
                )
                .await
                .unwrap();
        });

        let error = match authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await
        {
            Ok(_) => panic!("over-budget Retry-After should propagate"),
            Err(error) => error,
        };
        server.await.unwrap();
        assert_eq!(error.status, 503);
        assert_eq!(error.retry_after.as_deref(), Some("120"));
    }

    #[tokio::test]
    async fn buffered_http_rejects_non_retryable_error_status_before_sse_parsing() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            let body = br#"{"error":{"code":"model_not_found","param":"model","message":"Model not found gpt-test"},"request_id":"req_up","response":{"id":"resp_up"}}"#;
            let response = format!(
                "HTTP/1.1 404 Not Found\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
        });

        let capture_root = tempfile::TempDir::new().unwrap();
        let writer = crate::logging::writer::pause_for_fixture();
        let mut context = http_test_context();
        context.traffic = Some(Arc::new(crate::traffic::test_capture(
            capture_root.path().to_path_buf(),
        )));
        let result = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &context,
                None,
                crate::config::CodexTransport::Http,
            )
            .await;
        server.await.unwrap();
        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let diagnostic_path = std::fs::read_dir(capture_root.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains("060-codex-event-error")
            })
            .expect("actual transport must retain provenance before public error mapping");
        let diagnostic: serde_json::Value =
            serde_json::from_slice(&std::fs::read(diagnostic_path).unwrap()).unwrap();
        assert_eq!(diagnostic["source"], "buffered_http");
        assert_eq!(diagnostic["upstream_code"], "model_not_found");
        assert_eq!(diagnostic["upstream_param"], "model");
        assert_eq!(diagnostic["upstream_status"], 404);
        assert_eq!(diagnostic["request_id"], "req_up");
        assert_eq!(diagnostic["response_id"], "resp_up");
        assert_eq!(diagnostic["local_request_id"], context.req_id);
        assert_eq!(diagnostic["semantic_committed"], false);
        assert!(diagnostic.get("message").is_none());

        let error = match result {
            Ok(_) => panic!("non-success HTTP status must not reach the SSE reducer"),
            Err(error) => error,
        };

        assert_eq!(error.status, 404);
        assert_eq!(error.detail.as_deref(), Some("Model not found gpt-test"));
        assert_eq!(error.origin, CodexErrorOrigin::BufferedHttp);
    }

    #[test]
    fn status_error_preserves_buffered_websocket_event_message() {
        let error = codex_status_error(CodexResponse {
            body: b"data: {\"type\":\"error\",\"error\":{\"status\":400,\"message\":\"bad request\"}}\n\n"
                .to_vec(),
            status: 400,
            headers: Vec::new(),
            transport: ActualTransport::WebSocket,
        });

        assert_eq!(error.status, 400);
        assert_eq!(error.detail.as_deref(), Some("bad request"));
        assert_eq!(error.origin, CodexErrorOrigin::BufferedWebSocket);
    }

    #[tokio::test]
    async fn active_http_body_can_exceed_header_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in [b"a".as_slice(), b"b", b"c"] {
                stream.write_all(b"1\r\n").await.unwrap();
                stream.write_all(chunk).await.unwrap();
                stream.write_all(b"\r\n").await.unwrap();
                tokio::time::sleep(Duration::from_millis(45)).await;
            }
            stream.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let response = http_test_client(format!("http://{addr}/responses"), 80)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await
            .expect("active body should not hit a whole-request timeout");
        server.await.unwrap();

        assert_eq!(response.body, b"abc");
    }

    #[tokio::test]
    async fn stalled_http_body_hits_idle_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 1\r\n\r\n")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        let result = http_test_client(format!("http://{addr}/responses"), 30)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await;
        server.await.unwrap();
        let error = result.err().expect("stalled body should time out");

        assert!(error.message.contains("next Codex response body chunk"));
        assert_eq!(error.detail.as_deref(), Some("http_response_body"));
    }

    #[tokio::test]
    async fn reset_http_body_returns_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 10\r\n\r\npartial")
                .await
                .unwrap();
        });

        let result = http_test_client(format!("http://{addr}/responses"), 100)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await;
        server.await.unwrap();
        let error = result.err().expect("truncated body should fail");

        assert!(
            error
                .message
                .contains("Transport error reading Codex response body")
        );
        assert_eq!(error.detail.as_deref(), Some("http_response_body"));
    }

    #[test]
    fn codex_error_display() {
        let err = CodexError {
            status: 429,
            message: "Rate limited".to_string(),
            detail: Some("body".to_string()),
            retry_after: Some("5".to_string()),
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        };
        let display = format!("{err}");
        assert!(display.contains("429"));
        assert!(display.contains("Rate limited"));
    }

    #[test]
    fn header_timeout_is_not_retried_by_either_classifier() {
        let err = CodexError {
            status: 0,
            message: "Timed out waiting 300000ms for Codex response headers".to_string(),
            detail: Some(HTTP_RESPONSE_HEADERS_DETAIL.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::Http,
        };

        assert!(!is_retryable_transport_error(&err));
        assert!(!retryable_http_stream_error(&err));
    }

    #[test]
    fn websocket_pre_request_502_is_retryable() {
        let err = CodexError {
            status: 502,
            message: "WebSocket connect error".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: Some("3".to_string()),
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn auto_fallback_requires_an_eligible_pre_send_failure() {
        for status in [0, 400, 401, 403, 407, 429, 502] {
            for origin in [
                CodexErrorOrigin::WebSocketHandshake,
                CodexErrorOrigin::WebSocket,
                CodexErrorOrigin::Auth,
                CodexErrorOrigin::Http,
            ] {
                let err = CodexError {
                    status,
                    message: "transport failure".to_string(),
                    detail: None,
                    retry_after: None,
                    usage_limit: None,
                    origin,
                };
                assert_eq!(
                    should_fallback_to_http(&err),
                    origin == CodexErrorOrigin::WebSocketHandshake && status != 407,
                    "status={status}, origin={origin:?}"
                );
            }
        }
    }

    #[test]
    fn proxy_tunnel_rejection_is_not_retried_or_used_for_http_fallback() {
        let err = CodexError {
            status: 0,
            message: "WebSocket proxy tunnel was rejected".to_string(),
            detail: Some(
                super::super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string(),
            ),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(!is_retryable_transport_error(&err));
        assert!(!should_fallback_to_http(&err));
    }

    #[test]
    fn websocket_pre_request_statusless_error_is_retryable() {
        let err = CodexError {
            status: 0,
            message: "WebSocket connect timeout after 15000ms".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn websocket_pre_request_400_is_not_retryable() {
        let err = CodexError {
            status: 400,
            message: "WebSocket connect error".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(!is_retryable_transport_error(&err));
    }

    #[test]
    fn statusless_transport_error_matching_is_case_insensitive() {
        let err = CodexError {
            status: 0,
            message: "WebSocket protocol error: Connection reset without closing handshake"
                .to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn keepalive_failure_is_retryable_with_full_context() {
        let err = CodexError {
            status: 0,
            message: "WebSocket keepalive error: test write failed".to_string(),
            detail: Some(super::super::websocket::WEBSOCKET_KEEPALIVE_FAILURE_DETAIL.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
        assert!(is_continuation_retry_error(&err));
    }

    #[test]
    fn statusless_broken_pipe_is_retryable() {
        let err = CodexError {
            status: 0,
            message: "WebSocket stream error: IO error: Broken pipe (os error 32)".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn image_headers_reuse_oauth_without_responses_beta_headers() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        };
        let headers = build_codex_image_headers(&auth, &http_test_context()).unwrap();

        assert_eq!(
            headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer tok"
        );
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acct");
        assert_eq!(
            headers.get(http::header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(
            headers.get(http::header::ACCEPT).unwrap(),
            "application/json"
        );
        assert!(headers.get("openai-beta").is_none());
        assert!(headers.get("x-codex-beta-features").is_none());
    }

    #[test]
    fn codex_headers_include_session_and_beta() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: Some("s".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, false).unwrap();
        assert_eq!(
            headers.get("openai-beta").unwrap(),
            "responses=experimental"
        );
        assert_eq!(headers.get("session-id").unwrap(), "s");
        assert!(headers.get("session_id").is_none());
        assert_eq!(
            headers.get("x-codex-beta-features").unwrap(),
            "remote_compaction_v2"
        );
    }

    #[test]
    fn codex_headers_include_responses_lite_when_requested() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, true).unwrap();
        assert_eq!(
            headers
                .get("x-openai-internal-codex-responses-lite")
                .unwrap(),
            "true"
        );
        assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
        assert_eq!(default_user_agent(true), "codex_cli_rs");
    }

    #[test]
    fn codex_headers_omit_session_when_missing() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, false).unwrap();
        assert!(headers.get("session-id").is_none());
        assert!(headers.get("x-client-request-id").is_none());
    }

    #[test]
    fn codex_headers_return_error_for_invalid_session_header() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: Some("bad\nsession".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let err = build_codex_headers(&auth, &ctx, false).unwrap_err();
        assert_eq!(err.status, 500);
        assert!(err.message.contains("session-id"));
    }

    #[test]
    fn build_websocket_request_removes_stream() {
        let input = vec![
            super::super::translate::request::ResponsesInputItem::Message {
                role: "user".to_string(),
                content: vec![
                    super::super::translate::request::ResponsesContentPart::InputText {
                        text: "hello".to_string(),
                    },
                ],
            },
        ];
        let req = ResponsesRequest {
            model: "gpt-5.5".to_string(),
            instructions: None,
            input,
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: super::super::translate::request::ResponsesText {
                verbosity: Some("low".to_string()),
                format: None,
            },
            reasoning: None,
        };
        let payload = build_websocket_request(&req, None);
        assert_eq!(
            payload.get("type").and_then(|v| v.as_str()),
            Some("response.create")
        );
        assert!(payload.get("stream").is_none());
        assert!(payload.get("previous_response_id").is_none());
    }

    #[test]
    fn websocket_pool_owner_tracks_typed_continuation_opt_in() {
        let owner = ConversationIdentity::Agent("session".into(), "agent".into());
        let disabled = test_continuation(Some(owner.clone()), None, None, None, Some("disabled"));
        let first_enabled = test_continuation(
            Some(owner.clone()),
            Some(1),
            None,
            None,
            Some("missing_state"),
        );
        let append = test_continuation(Some(owner.clone()), Some(2), Some("resp_1"), Some(1), None);
        let missing_identity = test_continuation(None, None, None, None, Some("missing_identity"));

        assert_eq!(websocket_pool_owner(Some(&disabled)), None);
        assert_eq!(websocket_pool_owner(Some(&first_enabled)), Some(&owner));
        assert_eq!(websocket_pool_owner(Some(&append)), Some(&owner));
        assert_eq!(websocket_pool_owner(Some(&missing_identity)), None);
    }

    #[test]
    fn build_codex_headers_error_on_empty_access() {
        let auth = StoredAuth {
            access: "".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let result = build_codex_headers(&auth, &ctx, false);
        assert!(
            result.is_ok(),
            "empty access should still produce valid Bearer header"
        );
    }

    #[test]
    fn codex_header_timeout_error_display() {
        let err = CodexHeaderTimeoutError { timeout_ms: 60000 };
        let display = format!("{err}");
        assert!(display.contains("60000"));
    }

    #[test]
    fn codex_transport_error_display() {
        let err = CodexTransportError {
            message: "connection reset".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("connection reset"));
    }

    #[test]
    fn unauthorized_retry_distinguishes_auto_and_strict_websocket_handshakes() {
        let http_unauthorized = Ok(OwnerAwareCodexResponse::new(
            CodexResponse {
                body: Vec::new(),
                status: 401,
                headers: Vec::new(),
                transport: ActualTransport::Http,
            },
            None,
        ));
        let websocket_unauthorized = Err(CodexError {
            status: 401,
            message: "WebSocket connect error".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        });
        let forbidden = Err(CodexError {
            status: 403,
            message: "Forbidden".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        });
        let rejected_handshake = Err(CodexError {
            status: 401,
            message: "WebSocket connect error".to_string(),
            detail: Some("policy denied".to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        });
        let rejected_handshake_err = match &rejected_handshake {
            Err(error) => error,
            Ok(_) => panic!("expected rejected handshake"),
        };

        assert!(should_refresh_after_unauthorized(
            &http_unauthorized,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_refresh_after_unauthorized(
            &websocket_unauthorized,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(!should_refresh_after_unauthorized(
            &forbidden,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(!should_refresh_after_unauthorized(
            &rejected_handshake,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_refresh_after_unauthorized(
            &rejected_handshake,
            false,
            crate::config::CodexTransport::WebSocket
        ));
        assert!(!should_refresh_after_unauthorized(
            &http_unauthorized,
            true,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_fallback_to_http(rejected_handshake_err));
    }

    #[test]
    fn informational_events_keep_live_continuation_retry_available() {
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": false}
        })));
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "keepalive"
        })));
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "response.created"
        })));
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1"}
        })));
        assert!(event_closes_live_retry_window(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "Read"}
        })));
        assert!(event_closes_live_retry_window(&serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "hello"
        })));
        assert!(event_closes_live_retry_window(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{}"
        })));
    }

    #[test]
    fn continuation_retry_requires_previous_response_id() {
        let owner = ConversationIdentity::Main("session".into());
        let append =
            test_continuation(Some(owner.clone()), Some(17), Some("resp_1"), Some(1), None);
        let initial =
            test_continuation(Some(owner.clone()), None, None, None, Some("missing_state"));
        let timeout = CodexError {
            status: 0,
            message: "WebSocket response start timeout after 60000ms".to_string(),
            detail: Some(
                super::super::websocket::WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL.to_string(),
            ),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };
        let missing = CodexError {
            status: 0,
            message: "Previous response not found".to_string(),
            detail: Some("previous_response_not_found".to_string()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };
        let idle = CodexError {
            status: 0,
            message: "WebSocket idle timeout after 60000ms".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        let full_context = full_context_continuation(Some(&append)).unwrap();
        assert_eq!(full_context.owner(), Some(&owner));
        assert_eq!(full_context.turn_id(), Some(17));
        assert_eq!(full_context.candidate().previous_response_id, None);
        assert_eq!(full_context.origin_socket_id(), None);
        assert_eq!(
            full_context.candidate().disabled_reason.as_deref(),
            Some("full_context_retry")
        );
        assert!(should_retry_without_continuation(&timeout, Some(&append)));
        assert!(should_retry_without_continuation(&missing, Some(&append)));
        assert!(!should_retry_without_continuation(&idle, Some(&append)));
        assert!(!should_retry_without_continuation(&timeout, Some(&initial)));
        assert!(!should_retry_without_continuation(&timeout, None));
    }
}
