use super::StreamSenderExt;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use http::HeaderMap;
use hyper_util::client::proxy::matcher::Matcher as ProxyMatcher;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc;

mod connection;
use connection::ConnectionHandle;
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Message,
        client::IntoClientRequest,
        handshake::{client::generate_key, derive_accept_key},
        protocol::Role,
    },
};

use crate::logging::{Logger, create_logger};
use crate::provider::RequestContext;
use crate::request_identity::ConversationIdentity;
use crate::retry::sleep as retry_sleep;
use crate::traffic::TrafficCapture;

use super::client::{
    ActualTransport, CodexError, CodexErrorOrigin, CodexResponse, OwnerAwareCodexResponse,
};
use super::continuation::ContinuationReservation;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const WEBSOCKET_PROTOCOL_HEADER: &str = "responses_websockets=2026-02-06";
pub const WEBSOCKET_CONNECT_TIMEOUT_MS: u64 = 15_000;
pub(super) const WEBSOCKET_AMBIGUOUS_WRITE_DETAIL: &str = "websocket_ambiguous_write";
pub const WEBSOCKET_IDLE_TIMEOUT_MS: u64 = 300_000;
pub const WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL: &str = "websocket_response_start_timeout";
pub const WEBSOCKET_MISSING_TERMINAL_DETAIL: &str = "websocket_missing_terminal";
pub const WEBSOCKET_KEEPALIVE_FAILURE_DETAIL: &str = "websocket_keepalive_failure";
pub const WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL: &str =
    "websocket_continuation_socket_missing";
pub(super) const WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL: &str = "websocket_proxy_tunnel_rejected";

const MAX_CONNECT_RESPONSE_HEADER_BYTES: usize = 8 * 1024;
const WEBSOCKET_CONNECT_START_SPACING: Duration = Duration::from_secs(1);
const WEBSOCKET_CONNECT_FORBIDDEN_COOLDOWN: Duration = Duration::from_secs(3);
const WEBSOCKET_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
const WEBSOCKET_KEEPALIVE_SEND_TIMEOUT: Duration = Duration::from_secs(10);

pub type CodexWebSocketEventReceiver =
    tokio::sync::mpsc::Receiver<Result<serde_json::Value, CodexError>>;

pub(crate) struct CodexWebSocketEventStream {
    receiver: CodexWebSocketEventReceiver,
    cleanup: Option<StreamCleanup>,
    delivery_aborted: Arc<AtomicBool>,
    socket_id: Arc<AtomicU64>,
    full_context_retry: Arc<AtomicBool>,
    intervening_failure: Arc<AtomicBool>,
    provider_retry_handoff: Arc<AtomicBool>,
    diagnostic_attempt_id: Arc<AtomicU64>,
}

#[derive(Clone)]
pub(crate) struct CodexWebSocketSocketIdPublisher {
    delivery_aborted: Arc<AtomicBool>,
    socket_id: Arc<AtomicU64>,
    full_context_retry: Arc<AtomicBool>,
    intervening_failure: Arc<AtomicBool>,
    provider_retry_handoff: Arc<AtomicBool>,
    diagnostic_attempt_id: Arc<AtomicU64>,
}

pub(super) struct EventSender {
    sender: mpsc::Sender<Result<serde_json::Value, CodexError>>,
    aborted: Arc<AtomicBool>,
}

impl std::ops::Deref for EventSender {
    type Target = mpsc::Sender<Result<serde_json::Value, CodexError>>;
    fn deref(&self) -> &Self::Target {
        &self.sender
    }
}

impl EventSender {
    pub(super) async fn send_bounded(
        &self,
        item: Result<serde_json::Value, CodexError>,
    ) -> Result<(), ()> {
        let result = self.sender.send_bounded(item).await;
        if result.is_err() {
            self.aborted.store(true, Ordering::Release);
        }
        result
    }
}

/// Owns only this attempt's publication; stale drop cannot invalidate a later turn.
#[derive(Clone)]
pub(super) struct StreamCleanup {
    reservation: ContinuationReservation,
    socket_id: Arc<AtomicU64>,
    handoff: Arc<AtomicBool>,
    delivery_aborted: Arc<AtomicBool>,
    abort_turn: bool,
    accepted: bool,
}

impl StreamCleanup {
    pub(super) fn accept(&mut self) {
        self.accepted = true;
    }
}

impl Drop for StreamCleanup {
    fn drop(&mut self) {
        if self.accepted {
            return;
        }
        // Paired with publisher.store(ID) -> load(aborted): SeqCst prevents
        // both sides observing initial values on separate atomics.
        self.delivery_aborted.store(true, Ordering::SeqCst);
        let id = self.socket_id.load(Ordering::SeqCst);
        invalidate_codex_websocket_pool_socket(&self.reservation, (id != 0).then_some(id));
        if self.abort_turn && !self.handoff.load(Ordering::Acquire) {
            super::continuation::abort_continuation_for_owner(&self.reservation);
        }
    }
}

impl CodexWebSocketEventStream {
    pub(crate) fn pending(
        receiver: CodexWebSocketEventReceiver,
    ) -> (Self, CodexWebSocketSocketIdPublisher) {
        let delivery_aborted = Arc::new(AtomicBool::new(false));
        let socket_id = Arc::new(AtomicU64::new(0));
        let full_context_retry = Arc::new(AtomicBool::new(false));
        let intervening_failure = Arc::new(AtomicBool::new(false));
        let provider_retry_handoff = Arc::new(AtomicBool::new(false));
        let diagnostic_attempt_id = Arc::new(AtomicU64::new(0));
        (
            Self {
                receiver,
                cleanup: None,
                delivery_aborted: delivery_aborted.clone(),
                socket_id: socket_id.clone(),
                full_context_retry: full_context_retry.clone(),
                intervening_failure: intervening_failure.clone(),
                provider_retry_handoff: provider_retry_handoff.clone(),
                diagnostic_attempt_id: diagnostic_attempt_id.clone(),
            },
            CodexWebSocketSocketIdPublisher {
                delivery_aborted,
                socket_id,
                full_context_retry,
                intervening_failure,
                provider_retry_handoff,
                diagnostic_attempt_id,
            },
        )
    }

    pub(super) fn track_attempt(
        mut self,
        reservation: Option<&ContinuationReservation>,
        abort_turn: bool,
    ) -> Self {
        self.cleanup = reservation.cloned().map(|reservation| StreamCleanup {
            reservation,
            socket_id: self.socket_id.clone(),
            handoff: self.provider_retry_handoff.clone(),
            delivery_aborted: self.delivery_aborted.clone(),
            abort_turn,
            accepted: false,
        });
        self
    }

    pub(super) fn delivery_guard(&self) -> Option<StreamCleanup> {
        self.cleanup.clone()
    }

    pub(super) fn accept_terminal(&mut self) {
        if let Some(cleanup) = self.cleanup.as_mut() {
            cleanup.accept();
        }
    }

    pub(super) fn delivery_aborted(&self) -> bool {
        self.delivery_aborted.load(Ordering::Acquire)
    }

    pub(super) fn abort_signal(&self) -> Arc<AtomicBool> {
        self.delivery_aborted.clone()
    }

    pub(crate) async fn recv(&mut self) -> Option<Result<serde_json::Value, CodexError>> {
        if self.delivery_aborted() {
            return None;
        }
        let item = self.receiver.recv().await;
        if self.delivery_aborted() { None } else { item }
    }

    pub(crate) fn socket_id(&self) -> Option<u64> {
        match self.socket_id.load(Ordering::Acquire) {
            0 => None,
            socket_id => Some(socket_id),
        }
    }

    pub(super) fn diagnostic_attempt_id(&self) -> Option<u64> {
        match self.diagnostic_attempt_id.load(Ordering::Acquire) {
            0 => None,
            id => Some(id),
        }
    }

    pub(super) fn diagnostic_attempt_handle(&self) -> Arc<AtomicU64> {
        self.diagnostic_attempt_id.clone()
    }

    pub(crate) fn used_full_context_retry(&self) -> bool {
        self.full_context_retry.load(Ordering::Acquire)
    }

    pub(crate) fn had_intervening_failure(&self) -> bool {
        self.intervening_failure.load(Ordering::Acquire)
    }

    pub(crate) fn mark_intervening_failure(&self) {
        self.intervening_failure.store(true, Ordering::Release);
    }

    pub(crate) fn mark_provider_retry_handoff(&self) {
        self.provider_retry_handoff.store(true, Ordering::Release);
    }

    pub(crate) fn into_receiver(mut self) -> CodexWebSocketEventReceiver {
        self.accept_terminal();
        self.receiver
    }

    pub(crate) fn replace_receiver(
        &mut self,
        receiver: CodexWebSocketEventReceiver,
    ) -> CodexWebSocketEventReceiver {
        std::mem::replace(&mut self.receiver, receiver)
    }
}

impl CodexWebSocketSocketIdPublisher {
    pub(super) fn publish_diagnostic_attempt(&self, id: Option<u64>) {
        self.diagnostic_attempt_id
            .store(id.unwrap_or(0), Ordering::Release);
    }
    pub(super) fn sender(
        &self,
        sender: mpsc::Sender<Result<serde_json::Value, CodexError>>,
    ) -> EventSender {
        EventSender {
            sender,
            aborted: self.delivery_aborted.clone(),
        }
    }
    pub(super) fn abort_delivery(&self) {
        self.delivery_aborted.store(true, Ordering::Release);
    }

    pub(super) fn publish(&self, socket_id: Option<u64>) {
        self.socket_id
            .store(socket_id.unwrap_or(0), Ordering::SeqCst);
    }

    pub(super) fn mark_full_context_retry(&self) {
        self.full_context_retry.store(true, Ordering::Release);
        self.mark_intervening_failure();
    }

    pub(super) fn mark_intervening_failure(&self) {
        self.intervening_failure.store(true, Ordering::Release);
    }

    pub(super) fn is_provider_retry_handoff(&self) -> bool {
        self.provider_retry_handoff.load(Ordering::Acquire)
    }
}

trait WebSocketIo: AsyncRead + AsyncWrite + Unpin + Send {}

impl<T> WebSocketIo for T where T: AsyncRead + AsyncWrite + Unpin + Send {}

type BoxedWebSocketIo = Box<dyn WebSocketIo>;
type CodexWebSocketStream = WebSocketStream<BoxedWebSocketIo>;

pub(super) struct WebSocketProxyConfig {
    scope: u64,
    matcher: ProxyMatcher,
    tls_config: Arc<rustls::ClientConfig>,
    cache_policy: crate::config::CodexCachePolicy,
}

#[derive(Clone)]
struct WebSocketProxyRoute {
    uri: http::Uri,
    basic_auth: Option<http::HeaderValue>,
}

impl WebSocketProxyConfig {
    pub(super) fn new(
        http_proxy: Option<&str>,
        https_proxy: Option<&str>,
        all_proxy: Option<&str>,
        no_proxy: Option<&str>,
    ) -> Self {
        let mut builder = ProxyMatcher::builder();
        if let Some(proxy) = all_proxy {
            builder = builder.all(proxy.to_string());
        }
        if let Some(proxy) = http_proxy {
            builder = builder.http(proxy.to_string());
        }
        if let Some(proxy) = https_proxy {
            builder = builder.https(proxy.to_string());
        }
        if let Some(no_proxy) = no_proxy {
            builder = builder.no(no_proxy.to_string());
        }
        Self {
            scope: next_monotonic_nonzero(&NEXT_CLIENT_SCOPE, "WebSocket client scope"),
            matcher: builder.build(),
            tls_config: websocket_tls_config(),
            cache_policy: crate::config::codex_cache_policy(),
        }
    }

    #[cfg(test)]
    pub(super) fn direct() -> Self {
        let mut config = Self::new(None, None, None, None);
        config.scope = 0;
        config
    }

    pub(super) fn uses_proxy_for(&self, websocket_url: &str) -> bool {
        let Ok(http_url) = to_http_upgrade_url(websocket_url) else {
            return true;
        };
        let Ok(destination) = http_url.parse::<http::Uri>() else {
            return true;
        };
        self.matcher.intercept(&destination).is_some()
    }

    fn http_connect_route(
        &self,
        websocket_url: &str,
    ) -> Result<Option<WebSocketProxyRoute>, CodexError> {
        let http_url = to_http_upgrade_url(websocket_url).map_err(|error| CodexError {
            status: 0,
            message: error.message,
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        })?;
        if !http_url.starts_with("https://") {
            return Ok(None);
        }
        let destination = http_url.parse::<http::Uri>().map_err(|_| {
            websocket_protocol_error("WebSocket destination URL could not be routed")
        })?;
        let Some(proxy) = self.matcher.intercept(&destination) else {
            return Ok(None);
        };
        if !matches!(proxy.uri().scheme_str(), Some("http" | "https")) {
            return Ok(None);
        }
        Ok(Some(WebSocketProxyRoute {
            uri: proxy.uri().clone(),
            basic_auth: proxy.basic_auth().cloned(),
        }))
    }
}

static WEBSOCKET_TLS_CONFIG: once_cell::sync::Lazy<Arc<rustls::ClientConfig>> =
    once_cell::sync::Lazy::new(|| {
        let mut roots = rustls::RootCertStore::empty();
        let native = rustls_native_certs::load_native_certs();
        let load_error_count = native.errors.len();
        let (_, parse_error_count) = roots.add_parsable_certificates(native.certs);
        if load_error_count > 0 || parse_error_count > 0 {
            let mut fields = serde_json::Map::new();
            fields.insert("loadErrorCount".into(), serde_json::json!(load_error_count));
            fields.insert(
                "parseErrorCount".into(),
                serde_json::json!(parse_error_count),
            );
            create_logger("codex").warn("native_certificate_load_errors", Some(fields));
        }
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    });

pub(super) fn websocket_tls_config() -> Arc<rustls::ClientConfig> {
    WEBSOCKET_TLS_CONFIG.clone()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CodexWebSocketError {
    pub message: String,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub retry_after: Option<String>,
    pub request_sent: bool,
}

impl CodexWebSocketError {
    pub fn new(message: String) -> Self {
        Self {
            message,
            status: None,
            code: None,
            retry_after: None,
            request_sent: false,
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_code(mut self, code: String) -> Self {
        self.code = Some(code);
        self
    }
}

impl std::fmt::Display for CodexWebSocketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codex WebSocket error: {}", self.message)
    }
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

static NEXT_CLIENT_SCOPE: AtomicU64 = AtomicU64::new(0);

// Deliberately neither Debug nor Serialize: credentials are represented only by
// an opaque digest and never exposed through diagnostics or pool metadata.
#[derive(Clone, PartialEq, Eq)]
struct PoolCompatibility {
    client_scope: u64,
    endpoint: String,
    auth_and_handshake_revision: [u8; 32],
    model: Option<String>,
}

impl PoolCompatibility {
    fn capture(
        config: &WebSocketProxyConfig,
        endpoint: &str,
        headers: &HeaderMap,
        reservation: Option<&ContinuationReservation>,
    ) -> Self {
        use sha2::{Digest, Sha256};
        let mut revision = Sha256::new();
        for name in [
            "authorization",
            "chatgpt-account-id",
            "openai-beta",
            "originator",
            "user-agent",
            "session-id",
            "x-codex-beta-features",
            "x-openai-internal-codex-responses-lite",
        ] {
            revision.update(name.as_bytes());
            let value = headers.get(name).map_or(&[][..], |value| value.as_bytes());
            revision.update((value.len() as u64).to_be_bytes());
            revision.update(value);
        }
        Self {
            client_scope: config.scope,
            endpoint: endpoint.into(),
            auth_and_handshake_revision: revision.finalize().into(),
            model: reservation
                .and_then(ContinuationReservation::model)
                .map(str::to_owned),
        }
    }
}

struct PoolEntry {
    compatibility: Option<PoolCompatibility>,
    published_turn: AtomicU64,
    connection: ConnectionHandle,
    socket_id: u64,
    created_at: tokio::time::Instant,
    idle_since: Mutex<tokio::time::Instant>,
    last_activity: AtomicU64,
}

impl PoolEntry {
    #[cfg(test)]
    fn new(ws: CodexWebSocketStream) -> Self {
        Self::with_compatibility(ws, None)
    }

    fn with_compatibility(
        ws: CodexWebSocketStream,
        compatibility: Option<PoolCompatibility>,
    ) -> Self {
        Self {
            compatibility,
            published_turn: AtomicU64::new(0),
            connection: ConnectionHandle::new(ws),
            socket_id: next_monotonic_nonzero(&NEXT_SOCKET_ID, "WebSocket ID"),
            created_at: tokio::time::Instant::now(),
            idle_since: Mutex::new(tokio::time::Instant::now()),
            last_activity: AtomicU64::new(next_pool_activity()),
        }
    }

    fn touch(&self) {
        *self.idle_since.lock().unwrap() = tokio::time::Instant::now();
        self.last_activity
            .fetch_max(next_pool_activity(), Ordering::Relaxed);
    }
}

static NEXT_SOCKET_ID: AtomicU64 = AtomicU64::new(0);
static POOL_ACTIVITY_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static WS_POOL: once_cell::sync::Lazy<Mutex<HashMap<ConversationIdentity, Arc<PoolEntry>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(test)]
static WS_POOL_TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());
static WS_CONNECT_GATE: once_cell::sync::Lazy<WebSocketConnectGate> =
    once_cell::sync::Lazy::new(|| WebSocketConnectGate::new(WEBSOCKET_CONNECT_START_SPACING));

// fetch_update supports the Rust 1.89 toolchain.
#[allow(deprecated)]
fn next_monotonic_nonzero(sequence: &AtomicU64, label: &str) -> u64 {
    let previous = sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .unwrap_or_else(|_| panic!("{label} sequence exhausted"));
    previous + 1
}

fn next_pool_activity() -> u64 {
    POOL_ACTIVITY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

pub fn clear_codex_websocket_pool_for_tests() {
    let mut guard = WS_POOL.lock().unwrap();
    guard.clear();
}

#[cfg(test)]
pub(crate) async fn lock_codex_websocket_pool_for_tests() -> tokio::sync::MutexGuard<'static, ()> {
    WS_POOL_TEST_LOCK.lock().await
}

#[cfg(test)]
pub(crate) fn pooled_socket_id_for_tests(owner: &ConversationIdentity) -> Option<u64> {
    WS_POOL
        .lock()
        .unwrap()
        .get(owner)
        .map(|entry| entry.socket_id)
}

pub fn invalidate_codex_websocket_pool_owner(owner: &ConversationIdentity) {
    let mut guard = WS_POOL.lock().unwrap();
    guard.remove(owner);
}

#[deprecated(note = "use typed conversation ownership internally")]
pub fn invalidate_codex_websocket_pool_key(session_id: &str) {
    let mut guard = WS_POOL.lock().unwrap();
    guard.retain(|owner, _| match owner {
        ConversationIdentity::Main(owner_session_id)
        | ConversationIdentity::Agent(owner_session_id, _) => owner_session_id != session_id,
    });
}

pub(crate) fn invalidate_codex_websocket_pool_turn_for_owner(
    owner: &ConversationIdentity,
    turn_id: Option<u64>,
) {
    let reservation = ContinuationReservation::for_owner_turn(Some(owner), turn_id);
    super::continuation::with_current_turn_for_owner(&reservation, || {
        invalidate_codex_websocket_pool_owner(owner)
    });
}

#[deprecated(note = "use typed conversation ownership internally")]
pub fn invalidate_codex_websocket_pool_turn(session_id: &str, turn_id: Option<u64>) {
    let owner = ConversationIdentity::Main(session_id.to_owned());
    invalidate_codex_websocket_pool_turn_for_owner(&owner, turn_id);
}

#[cfg(test)]
fn invalidate_pool_entry(owner: &ConversationIdentity, entry: &Arc<PoolEntry>) {
    let mut guard = WS_POOL.lock().unwrap();
    if guard
        .get(owner)
        .is_some_and(|pooled| Arc::ptr_eq(pooled, entry))
    {
        guard.remove(owner);
    }
}

#[cfg(test)]
fn invalidate_pool_owner(owner: Option<&ConversationIdentity>, entry: Option<&Arc<PoolEntry>>) {
    let Some(owner) = owner else {
        return;
    };
    match entry {
        Some(entry) => invalidate_pool_entry(owner, entry),
        None => invalidate_codex_websocket_pool_owner(owner),
    }
}

fn reservation_pool_owner(
    reservation: Option<&ContinuationReservation>,
) -> Option<&ConversationIdentity> {
    let reservation = reservation?;
    if !reservation.pool_enabled() {
        return None;
    }
    reservation.owner()
}

fn pool_take_for_turn(reservation: &ContinuationReservation) -> Option<Arc<PoolEntry>> {
    let owner = reservation.owner()?;
    let (entry, removed) = super::continuation::if_current_turn_for_owner(reservation, || {
        let mut pool = WS_POOL.lock().unwrap();
        let policy = crate::config::codex_cache_policy();
        let now = tokio::time::Instant::now();
        let mut removed = maintain_pool_at(&mut pool, policy, now);
        let entry = pool.remove(owner).and_then(|entry| {
            if pool_entry_expired(&entry, policy, now) {
                removed.push(entry);
                None
            } else {
                Some(entry)
            }
        });
        (entry, removed)
    })?;
    drop(removed);
    entry
}

fn take_pool_entry_for_request(
    reservation: Option<&ContinuationReservation>,
    compatibility: &PoolCompatibility,
) -> Result<Option<Arc<PoolEntry>>, CodexError> {
    let candidate = reservation.map(ContinuationReservation::candidate);
    let requires_origin = candidate
        .and_then(|candidate| candidate.previous_response_id.as_deref())
        .is_some();
    let expected_socket_id = reservation.and_then(ContinuationReservation::origin_socket_id);
    let pool_owner = reservation_pool_owner(reservation);
    let pooled = reservation
        .filter(|_| pool_owner.is_some())
        .and_then(pool_take_for_turn)
        .filter(|entry| {
            !entry.connection.is_closed()
                && entry
                    .compatibility
                    .as_ref()
                    .is_none_or(|stored| stored == compatibility)
        });

    if requires_origin
        && (pool_owner.is_none()
            || expected_socket_id.is_none()
            || pooled
                .as_ref()
                .is_none_or(|entry| Some(entry.socket_id) != expected_socket_id))
    {
        if let (Some(owner), Some(entry)) = (pool_owner, pooled.as_ref()) {
            pool_insert_if_vacant_or_same(owner.clone(), entry.clone());
        }
        return Err(continuation_socket_missing_error());
    }

    Ok(pooled)
}

fn pool_insert_for_turn(reservation: &ContinuationReservation, entry: Arc<PoolEntry>) -> bool {
    let Some(owner) = reservation_pool_owner(Some(reservation)).cloned() else {
        return false;
    };
    super::continuation::if_current_turn_for_owner(reservation, || {
        entry
            .published_turn
            .store(reservation.turn_id().unwrap_or(0), Ordering::Release);
        pool_insert_if_vacant_or_same(owner, entry)
    })
    .unwrap_or(false)
}

#[cfg(test)]
fn pool_remove_entry(owner: &ConversationIdentity, entry: &Arc<PoolEntry>) {
    invalidate_pool_entry(owner, entry);
}

pub(super) fn invalidate_codex_websocket_pool_socket(
    reservation: &ContinuationReservation,
    socket_id: Option<u64>,
) {
    let Some(owner) = reservation_pool_owner(Some(reservation)) else {
        return;
    };
    let Some(socket_id) = socket_id else {
        return;
    };
    // Check and remove under one pool lock: the same Arc can be republished by
    // a later turn between an unlocked check and removal.
    let removed = {
        let mut pool = WS_POOL.lock().unwrap();
        let matches = pool.get(owner).is_some_and(|entry| {
            let turn = entry.published_turn.load(Ordering::Acquire);
            entry.socket_id == socket_id && (turn == 0 || Some(turn) == reservation.turn_id())
        });
        if matches { pool.remove(owner) } else { None }
    };
    drop(removed);
}

fn pool_insert_if_vacant_or_same(owner: ConversationIdentity, entry: Arc<PoolEntry>) -> bool {
    let policy = crate::config::codex_cache_policy();
    let now = tokio::time::Instant::now();
    entry.touch();
    let mut guard = WS_POOL.lock().unwrap();
    let mut removed = maintain_pool_at(&mut guard, policy, now);
    let inserted = if now.saturating_duration_since(entry.created_at)
        >= Duration::from_millis(policy.max_age_ms)
    {
        false
    } else if let Some(existing) = guard.get(&owner) {
        Arc::ptr_eq(existing, &entry)
    } else {
        guard.insert(owner, entry);
        removed.extend(maintain_pool_at(&mut guard, policy, now));
        true
    };
    drop(guard);
    drop(removed);
    inserted
}

#[cfg(test)]
fn pool_insert(owner: ConversationIdentity, entry: Arc<PoolEntry>) {
    pool_insert_if_vacant_or_same(owner, entry);
}

#[cfg(test)]
fn cleanup_pool_before_connect() {
    cleanup_pool_with_policy(crate::config::codex_cache_policy());
}

fn cleanup_pool_with_policy(policy: crate::config::CodexCachePolicy) {
    let removed = maintain_pool_at(
        &mut WS_POOL.lock().unwrap(),
        policy,
        tokio::time::Instant::now(),
    );
    drop(removed);
}

fn maintain_pool_at(
    pool: &mut HashMap<ConversationIdentity, Arc<PoolEntry>>,
    policy: crate::config::CodexCachePolicy,
    now: tokio::time::Instant,
) -> Vec<Arc<PoolEntry>> {
    // Checked-out entries are absent; extra Arc owners represent active leases
    // or delivery guards. Only idle, unleased sockets count toward the cap.
    let mut expired = Vec::new();
    let mut idle = Vec::new();
    for (owner, entry) in pool.iter() {
        if Arc::strong_count(entry) != 1 {
            continue;
        }
        if pool_entry_expired(entry, policy, now) {
            expired.push(owner.clone());
        } else {
            idle.push((owner.clone(), entry.last_activity.load(Ordering::Relaxed)));
        }
    }
    idle.sort_unstable_by_key(|(_, activity)| *activity);
    expired.extend(
        idle.iter()
            .take(idle.len().saturating_sub(policy.idle_cap))
            .map(|(owner, _)| owner.clone()),
    );
    expired
        .into_iter()
        .filter_map(|owner| pool.remove(&owner))
        .collect()
}

fn pool_entry_expired(
    entry: &PoolEntry,
    policy: crate::config::CodexCachePolicy,
    now: tokio::time::Instant,
) -> bool {
    entry.connection.is_closed()
        || now.saturating_duration_since(entry.created_at)
            >= Duration::from_millis(policy.max_age_ms)
        || now.saturating_duration_since(*entry.idle_since.lock().unwrap())
            >= Duration::from_millis(policy.idle_ttl_ms)
}

struct WebSocketConnectGate {
    last_start: AsyncMutex<Option<tokio::time::Instant>>,
    start_spacing: Duration,
}

impl WebSocketConnectGate {
    fn new(start_spacing: Duration) -> Self {
        Self {
            last_start: AsyncMutex::new(None),
            start_spacing,
        }
    }

    async fn wait_to_start(&self, before_start: impl std::future::Future<Output = ()>) {
        let mut last_start = self.last_start.lock().await;
        if let Some(previous) = *last_start {
            tokio::time::sleep_until(previous + self.start_spacing).await;
        }
        before_start.await;
        *last_start = Some(tokio::time::Instant::now());
    }
}

// ---------------------------------------------------------------------------
// URL conversion
// ---------------------------------------------------------------------------

pub fn to_websocket_url(url: &str) -> Result<String, CodexWebSocketError> {
    let mut parsed = url::Url::parse(url)
        .map_err(|e| CodexWebSocketError::new(format!("Failed to parse URL: {e}")))?;
    match parsed.scheme() {
        "http" => parsed.set_scheme("ws").map_err(|_| {
            CodexWebSocketError::new("Unsupported Codex WebSocket URL scheme".to_string())
        })?,
        "https" => parsed.set_scheme("wss").map_err(|_| {
            CodexWebSocketError::new("Unsupported Codex WebSocket URL scheme".to_string())
        })?,
        "ws" | "wss" => { /* already a ws scheme */ }
        other => {
            return Err(CodexWebSocketError::new(format!(
                "Unsupported Codex WebSocket URL scheme: {other}"
            )));
        }
    }
    Ok(parsed.to_string())
}

fn to_http_upgrade_url(url: &str) -> Result<String, CodexWebSocketError> {
    let mut parsed = url::Url::parse(url)
        .map_err(|e| CodexWebSocketError::new(format!("Failed to parse URL: {e}")))?;
    match parsed.scheme() {
        "ws" => parsed.set_scheme("http").map_err(|_| {
            CodexWebSocketError::new("Unsupported Codex WebSocket URL scheme".to_string())
        })?,
        "wss" => parsed.set_scheme("https").map_err(|_| {
            CodexWebSocketError::new("Unsupported Codex WebSocket URL scheme".to_string())
        })?,
        other => {
            return Err(CodexWebSocketError::new(format!(
                "Unsupported Codex WebSocket URL scheme: {other}"
            )));
        }
    }
    Ok(parsed.to_string())
}

// ---------------------------------------------------------------------------
// Header rewriting
// ---------------------------------------------------------------------------

pub fn codex_websocket_headers(http_headers: &HeaderMap) -> HeaderMap {
    let mut ws = HeaderMap::new();
    for (key, value) in http_headers.iter() {
        let key_str = key.as_str().to_lowercase();
        if matches!(
            key_str.as_str(),
            "content-length"
                | "content-type"
                | "accept"
                | "connection"
                | "upgrade"
                | "proxy-authorization"
        ) {
            continue;
        }
        ws.insert(key.clone(), value.clone());
    }
    ws.insert("openai-beta", WEBSOCKET_PROTOCOL_HEADER.parse().unwrap());
    if !ws.contains_key("sec-websocket-key") {
        ws.insert("sec-websocket-key", generate_key().parse().unwrap());
    }
    ws
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

fn encode_sse(text: &str) -> Vec<u8> {
    let mut out = String::new();
    for line in text.lines() {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// Terminal event detection
// ---------------------------------------------------------------------------

pub(super) fn is_terminal_event(payload: &serde_json::Value) -> bool {
    super::events::event_is_terminal(payload)
}

fn is_response_event(payload: &serde_json::Value) -> bool {
    match payload.get("type").and_then(|v| v.as_str()) {
        Some("error") => true,
        Some(t) => t.starts_with("response."),
        None => false,
    }
}

fn is_previous_response_missing(payload: &serde_json::Value) -> bool {
    let error = super::events::event_error(payload);
    if error
        .and_then(|error| error.get("code"))
        .and_then(|value| value.as_str())
        == Some("previous_response_not_found")
    {
        return true;
    }
    if let Some(msg) = error
        .and_then(|error| error.get("message"))
        .and_then(|value| value.as_str())
    {
        let lower = msg.to_lowercase();
        if lower.contains("previous response") && lower.contains("not found") {
            return true;
        }
    }
    false
}

fn malformed_event_error() -> CodexError {
    CodexError {
        status: 502,
        message: "Malformed JSON in Codex WebSocket event".to_string(),
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocket,
    }
}

pub(super) fn event_error_status(payload: &serde_json::Value) -> Option<u16> {
    super::events::classify_event_failure(payload).map(|failure| failure.client_status())
}

#[allow(dead_code)]
fn extract_retry_after(payload: &serde_json::Value) -> Option<String> {
    payload
        .get("error")
        .and_then(|e| e.get("retry_after"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Main request function
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) async fn codex_websocket_request(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
    body_value: &serde_json::Value,
    ctx: &RequestContext,
    traffic: Option<&TrafficCapture>,
    connect_timeout_ms: u64,
    idle_timeout_ms: u64,
    reservation: Option<&ContinuationReservation>,
) -> Result<OwnerAwareCodexResponse, CodexError> {
    let started_at = Instant::now();
    let events = codex_websocket_event_stream(
        websocket_client,
        proxy_config,
        url,
        headers,
        body_value,
        ctx,
        ctx.traffic.clone(),
        connect_timeout_ms,
        idle_timeout_ms,
        reservation,
    )
    .await?;
    collect_codex_websocket_response(events, traffic, started_at).await
}

pub(super) async fn collect_codex_websocket_response(
    mut events: CodexWebSocketEventStream,
    traffic: Option<&TrafficCapture>,
    started_at: Instant,
) -> Result<OwnerAwareCodexResponse, CodexError> {
    let mut sse_body = Vec::new();
    let mut status = 200;
    let mut completed = false;
    while let Some(item) = events.recv().await {
        let payload = item?;
        sse_body.extend_from_slice(&encode_sse(&payload.to_string()));
        if is_terminal_event(&payload) {
            status = event_error_status(&payload).unwrap_or(200);
            completed = true;
            events.accept_terminal();
            break;
        }
    }
    if !completed {
        return Err(missing_terminal_error());
    }
    if let Some(tc) = traffic {
        write_websocket_response_capture(tc, status, started_at.elapsed(), &sse_body);
    }
    Ok(OwnerAwareCodexResponse::new(
        CodexResponse {
            body: sse_body,
            status,
            headers: vec![],
            transport: ActualTransport::WebSocket,
        },
        events.socket_id(),
    )
    .with_intervening_failure(events.had_intervening_failure()))
}

fn handle_connection_metadata(payload: &serde_json::Value) -> bool {
    match payload.get("type").and_then(serde_json::Value::as_str) {
        Some("codex.rate_limits") => {
            super::rate_limits::record_event(payload);
            true
        }
        Some("keepalive") => true,
        _ => false,
    }
}

fn pong_write_error() -> CodexError {
    CodexError {
        status: 0,
        message: "WebSocket Pong write failed".into(),
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocket,
    }
}

async fn write_request<S>(ws: &mut WebSocketStream<S>, body: String) -> Result<(), CodexError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let result = tokio::time::timeout(Duration::from_secs(15), ws.send(Message::Text(body))).await;
    match result {
        Ok(Ok(())) => Ok(()),
        _ => Err(CodexError {
            status: 0,
            message: "WebSocket request write did not complete; delivery is ambiguous".into(),
            detail: Some(WEBSOCKET_AMBIGUOUS_WRITE_DETAIL.into()),
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocket,
        }),
    }
}

pub(super) struct ReadyWebSocket {
    ws_url: String,
    entry: Arc<PoolEntry>,
    used_pooled: bool,
    reservation: Option<ContinuationReservation>,
    traffic: Option<Arc<TrafficCapture>>,
    idle_timeout_ms: u64,
    diagnostics: WsDiagnostics,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn prepare_codex_websocket(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
    req_id: Option<&str>,
    traffic: Option<Arc<TrafficCapture>>,
    reservation: Option<&ContinuationReservation>,
    connect_timeout_ms: u64,
    idle_timeout_ms: u64,
) -> Result<ReadyWebSocket, CodexError> {
    let mut diagnostics = WsDiagnostics::new(req_id, headers, reservation);
    diagnostics.logger.info("websocket_attempt_start", None);
    let ws_url = to_websocket_url(url).map_err(|error| CodexError {
        status: 0,
        message: error.message,
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    })?;
    let compatibility = PoolCompatibility::capture(proxy_config, &ws_url, headers, reservation);
    let pooled = take_pool_entry_for_request(reservation, &compatibility)
        .inspect_err(|error| diagnostics.failure("pool_checkout", &error.message))?;
    let used_pooled = pooled.is_some();
    let entry = if let Some(entry) = pooled {
        entry
    } else {
        let stream = connect_with_timeout(
            websocket_client,
            proxy_config,
            &ws_url,
            headers,
            connect_timeout_ms,
        )
        .await
        .inspect_err(|error| diagnostics.failure("connect", &error.message))?;
        Arc::new(PoolEntry::with_compatibility(stream, Some(compatibility)))
    };
    diagnostics.ready(&entry, used_pooled);
    Ok(ReadyWebSocket {
        ws_url,
        entry,
        used_pooled,
        reservation: reservation.cloned(),
        traffic,
        idle_timeout_ms,
        diagnostics,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn codex_websocket_responses_stream(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
    body: &super::translate::request::ResponsesRequest,
    ctx: &RequestContext,
    reservation: Option<&ContinuationReservation>,
) -> Result<CodexWebSocketEventStream, CodexError> {
    let mut ready = prepare_codex_websocket(
        websocket_client,
        proxy_config,
        url,
        headers,
        Some(&ctx.req_id),
        ctx.traffic.clone(),
        reservation,
        WEBSOCKET_CONNECT_TIMEOUT_MS,
        WEBSOCKET_IDLE_TIMEOUT_MS,
    )
    .await?;
    // Checkout and origin-affinity checks happen before allocating the selected wire input.
    let wire = super::client::build_websocket_request(
        body,
        reservation.map(ContinuationReservation::candidate),
    );
    ready.diagnostics.logger = ready.diagnostics.logger.child(serde_json::Map::from_iter([(
        "fullInputItems".into(),
        body.input.len().into(),
    )]));
    let json = serde_json::to_string(&wire).expect("wire request serializes");
    Ok(start_codex_websocket_events(
        ready,
        &wire,
        json,
        headers,
        reservation,
    ))
}

pub(super) fn start_codex_websocket_events(
    mut ready: ReadyWebSocket,
    body_value: &serde_json::Value,
    body_json: String,
    headers: &HeaderMap,
    reservation: Option<&ContinuationReservation>,
) -> CodexWebSocketEventStream {
    ready.diagnostics.model(body_value);
    ready.diagnostics.logger = ready.diagnostics.logger.child(serde_json::json!({
        "requestMode": if body_value.get("previous_response_id").and_then(serde_json::Value::as_str).is_some() {"delta"} else {"full"},
        "wireInputItems":body_value.get("input").and_then(serde_json::Value::as_array).map(Vec::len),
        "wireBytes":body_json.len(),
        "continuationReason":reservation.and_then(|r|r.candidate().disabled_reason.as_deref()).unwrap_or("eligible"),
    }).as_object().unwrap().clone());
    if let Some(tc) = ready.traffic.as_deref() {
        write_websocket_metadata_capture(tc, &ready.ws_url, reservation, ready.used_pooled);
        tc.write_json("020-upstream-request", body_value);
        tc.write_json(
            "021-upstream-request-metadata",
            &serde_json::json!({
                "provider": "codex",
                "transport": "websocket",
                "headers": headers_to_json(headers),
                "size": summarize_json_request_size(body_value, &body_json),
            }),
        );
    }
    let (tx, rx) = mpsc::channel(64);
    let (receiver, socket_id_publisher) = CodexWebSocketEventStream::pending(rx);
    let receiver = receiver.track_attempt(reservation, false);
    socket_id_publisher.publish_diagnostic_attempt(Some(ready.diagnostics.attempt_id));
    let tx = socket_id_publisher.sender(tx);
    let connection = ready.entry.connection.clone();
    let turn = connection::StartTurn {
        body_json,
        entry: ready.entry,
        reservation: ready.reservation,
        traffic: ready.traffic,
        idle_timeout_ms: ready.idle_timeout_ms,
        diagnostics: ready.diagnostics,
        settings: crate::config::captured_request_settings(),
        tx,
        publisher: socket_id_publisher,
    };
    connection.start(turn);
    receiver
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) async fn codex_websocket_event_stream(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
    body_value: &serde_json::Value,
    ctx: &RequestContext,
    traffic: Option<Arc<TrafficCapture>>,
    connect_timeout_ms: u64,
    idle_timeout_ms: u64,
    reservation: Option<&ContinuationReservation>,
) -> Result<CodexWebSocketEventStream, CodexError> {
    let body_json = serde_json::to_string(body_value).map_err(|error| CodexError {
        status: 500,
        message: "Failed to serialize WebSocket request".to_string(),
        detail: Some(error.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    })?;
    let ready = prepare_codex_websocket(
        websocket_client,
        proxy_config,
        url,
        headers,
        Some(&ctx.req_id),
        traffic,
        reservation,
        connect_timeout_ms,
        idle_timeout_ms,
    )
    .await?;
    Ok(start_codex_websocket_events(
        ready,
        body_value,
        body_json,
        headers,
        reservation,
    ))
}

fn continuation_socket_missing_error() -> CodexError {
    CodexError {
        status: 0,
        message: "Previous response socket is no longer available".to_string(),
        detail: Some(WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

fn missing_terminal_error() -> CodexError {
    CodexError {
        status: 0,
        message: "WebSocket connection closed before terminal Codex response event".to_string(),
        detail: Some(WEBSOCKET_MISSING_TERMINAL_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocket,
    }
}

fn response_start_timeout_error(timeout_ms: u64) -> CodexError {
    CodexError {
        status: 0,
        message: format!("WebSocket response start timeout after {timeout_ms}ms"),
        detail: Some(WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocket,
    }
}

fn write_websocket_metadata_capture(
    traffic: &TrafficCapture,
    ws_url: &str,
    reservation: Option<&ContinuationReservation>,
    pooled: bool,
) {
    let pool_owner = reservation_pool_owner(reservation);
    let continuation = reservation.map(ContinuationReservation::candidate);
    traffic.write_json(
        "022-upstream-websocket-metadata",
        &serde_json::json!({
            "provider": "codex",
            "transport": "websocket",
            "url": ws_url,
            "poolingEnabled": pool_owner.is_some(),
            "pooled": pooled,
            "continuation": {
                "previousResponseId": continuation
                    .and_then(|c| c.previous_response_id.as_deref()),
                "inputDeltaCount": continuation
                    .and_then(|c| c.input_delta.as_ref())
                    .map(|items| items.len()),
                "disabledReason": continuation
                    .and_then(|c| c.disabled_reason.as_deref()),
            },
        }),
    );
}

fn write_websocket_response_capture(
    traffic: &TrafficCapture,
    status: u16,
    elapsed: Duration,
    sse_body: &[u8],
) {
    traffic.write_json(
        "030-upstream-response-headers",
        &serde_json::json!({
            "status": status,
            "elapsedMs": elapsed.as_millis(),
            "headers": {
                "content-type": "text/event-stream",
            },
        }),
    );
    if status >= 400 {
        traffic.write_text(
            "031-upstream-error-body",
            &String::from_utf8_lossy(sse_body),
        );
    } else {
        traffic.write_bytes("032-upstream-response-body.sse", sse_body);
    }
}

// ---------------------------------------------------------------------------
// Connection helper
// ---------------------------------------------------------------------------

const MAX_HANDSHAKE_ERROR_DETAIL_BYTES: usize = 1024;
const GENERIC_HANDSHAKE_ERROR_DETAIL: &str = "WebSocket upgrade was rejected";

fn handshake_error_detail(body: Option<&[u8]>) -> String {
    let Some(value) = body.and_then(|body| serde_json::from_slice::<serde_json::Value>(body).ok())
    else {
        return GENERIC_HANDSHAKE_ERROR_DETAIL.to_string();
    };
    let Some(message) = value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .and_then(|value| value.as_str())
    else {
        return GENERIC_HANDSHAKE_ERROR_DETAIL.to_string();
    };
    let sanitized: String = message
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, ' '))
        .collect();
    let mut end = sanitized.len().min(MAX_HANDSHAKE_ERROR_DETAIL_BYTES);
    while !sanitized.is_char_boundary(end) {
        end -= 1;
    }
    sanitized[..end].to_string()
}

fn header_has_token(headers: &HeaderMap, name: &str, expected: &str) -> bool {
    headers.get_all(name).iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case(expected))
        })
    })
}

fn requested_subprotocols(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(http::header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn websocket_protocol_error(message: &str) -> CodexError {
    CodexError {
        status: 0,
        message: message.to_string(),
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

fn validate_websocket_upgrade(
    version: http::Version,
    headers: &HeaderMap,
    websocket_key: &str,
    requested_subprotocols: &[String],
) -> Result<(), CodexError> {
    if version != http::Version::HTTP_11 {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response did not use HTTP/1.1",
        ));
    }
    if !header_has_token(headers, http::header::UPGRADE.as_str(), "websocket") {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response is missing Upgrade: websocket",
        ));
    }
    if !header_has_token(headers, http::header::CONNECTION.as_str(), "upgrade") {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response is missing Connection: Upgrade",
        ));
    }

    let expected_accept = derive_accept_key(websocket_key.as_bytes());
    let mut accept_values = headers.get_all(http::header::SEC_WEBSOCKET_ACCEPT).iter();
    let accept = accept_values.next().and_then(|value| value.to_str().ok());
    if accept_values.next().is_some() || accept != Some(expected_accept.as_str()) {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response has an invalid Sec-WebSocket-Accept",
        ));
    }
    if headers.contains_key(http::header::SEC_WEBSOCKET_EXTENSIONS) {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response selected an unsolicited extension",
        ));
    }

    let mut response_protocols = headers.get_all(http::header::SEC_WEBSOCKET_PROTOCOL).iter();
    let response_protocol = response_protocols
        .next()
        .map(|value| value.to_str().map(str::trim));
    if response_protocols.next().is_some() {
        return Err(websocket_protocol_error(
            "WebSocket upgrade response contains multiple subprotocols",
        ));
    }
    match response_protocol {
        None if requested_subprotocols.is_empty() => {}
        None => {
            return Err(websocket_protocol_error(
                "WebSocket upgrade response omitted the requested subprotocol",
            ));
        }
        Some(Err(_)) => {
            return Err(websocket_protocol_error(
                "WebSocket upgrade response contains an invalid subprotocol",
            ));
        }
        Some(Ok(_)) if requested_subprotocols.is_empty() => {
            return Err(websocket_protocol_error(
                "WebSocket upgrade response selected an unsolicited subprotocol",
            ));
        }
        Some(Ok(protocol))
            if !requested_subprotocols
                .iter()
                .any(|requested| requested == protocol) =>
        {
            return Err(websocket_protocol_error(
                "WebSocket upgrade response selected an unsupported subprotocol",
            ));
        }
        Some(Ok(_)) => {}
    }

    Ok(())
}

async fn bounded_handshake_error_body(mut response: reqwest::Response) -> Vec<u8> {
    let mut body = Vec::new();
    while body.len() < MAX_HANDSHAKE_ERROR_DETAIL_BYTES {
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) | Err(_) => break,
        };
        let remaining = MAX_HANDSHAKE_ERROR_DETAIL_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    body
}

fn error_chain_contains(error: &(dyn std::error::Error + 'static), expected: &str) -> bool {
    let expected = expected.to_ascii_lowercase();
    let mut current = Some(error);
    while let Some(error) = current {
        if error.to_string().to_ascii_lowercase().contains(&expected) {
            return true;
        }
        current = error.source();
    }
    false
}

fn reqwest_handshake_error(error: reqwest::Error) -> CodexError {
    let proxy_auth_required =
        error.is_connect() && error_chain_contains(&error, "proxy authorization required");
    let proxy_tunnel_rejected =
        error.is_connect() && error_chain_contains(&error, "tunnel error: unsuccessful");
    let status = if proxy_auth_required {
        http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16()
    } else {
        error.status().map(|status| status.as_u16()).unwrap_or(0)
    };
    let message = if proxy_auth_required {
        "WebSocket proxy authentication failed"
    } else if proxy_tunnel_rejected {
        "WebSocket proxy tunnel was rejected"
    } else if error.is_timeout() {
        "WebSocket upgrade request timed out"
    } else if error.is_connect() {
        "WebSocket connection failed"
    } else {
        "WebSocket upgrade request failed"
    };
    CodexError {
        status,
        message: message.to_string(),
        detail: proxy_tunnel_rejected.then(|| WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

struct PrefixedIo {
    prefix: Vec<u8>,
    position: usize,
    inner: BoxedWebSocketIo,
}

impl AsyncRead for PrefixedIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.position < this.prefix.len() {
            let available = &this.prefix[this.position..];
            let len = available.len().min(buffer.remaining());
            buffer.put_slice(&available[..len]);
            this.position += len;
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut this.inner).poll_read(cx, buffer)
    }
}

impl AsyncWrite for PrefixedIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

fn skip_websocket_request_header(name: &http::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "upgrade"
            | "sec-websocket-key"
            | "sec-websocket-version"
            | "host"
            | "content-length"
            | "proxy-authorization"
    )
}

fn tunneled_websocket_request(
    url: &str,
    headers: &HeaderMap,
    websocket_key: &str,
) -> Result<http::Request<()>, CodexError> {
    let mut request = url
        .into_client_request()
        .map_err(|_| websocket_protocol_error("WebSocket request URL was invalid"))?;
    *request.version_mut() = http::Version::HTTP_11;
    request.headers_mut().insert(
        http::header::SEC_WEBSOCKET_KEY,
        http::HeaderValue::from_str(websocket_key)
            .map_err(|_| websocket_protocol_error("WebSocket key was invalid"))?,
    );
    for (name, value) in headers {
        if !skip_websocket_request_header(name) {
            request.headers_mut().append(name.clone(), value.clone());
        }
    }
    Ok(request)
}

fn tunnel_error(status: u16, retry_after: Option<String>) -> CodexError {
    let proxy_auth_required = status == http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16();
    CodexError {
        status,
        message: if proxy_auth_required {
            "WebSocket proxy authentication failed".to_string()
        } else {
            "WebSocket proxy tunnel was rejected".to_string()
        },
        detail: if proxy_auth_required {
            Some(GENERIC_HANDSHAKE_ERROR_DETAIL.to_string())
        } else {
            Some(WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string())
        },
        retry_after,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

fn invalid_tunnel_response(message: &str) -> CodexError {
    CodexError {
        status: 0,
        message: message.to_string(),
        detail: Some(WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

fn connect_response_header_end(response: &[u8]) -> Option<usize> {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4)
}

fn parse_connect_response_head(response: &[u8]) -> Result<(u16, Option<String>), CodexError> {
    let status_line_end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .ok_or_else(|| invalid_tunnel_response("WebSocket proxy returned an invalid response"))?;
    let status_line = std::str::from_utf8(&response[..status_line_end])
        .map_err(|_| invalid_tunnel_response("WebSocket proxy returned an invalid response"))?;
    let mut parts = status_line.split_ascii_whitespace();
    let version = parts.next().unwrap_or_default();
    let status = parts.next().unwrap_or_default();
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || status.len() != 3
        || !status.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid_tunnel_response(
            "WebSocket proxy returned an invalid response",
        ));
    }
    let status = status
        .parse::<u16>()
        .map_err(|_| invalid_tunnel_response("WebSocket proxy returned an invalid response"))?;
    let retry_after = response[status_line_end + 2..]
        .split(|byte| *byte == b'\n')
        .find_map(|line| {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let separator = line.iter().position(|byte| *byte == b':')?;
            let (name, value) = line.split_at(separator);
            if !name.eq_ignore_ascii_case(b"retry-after") {
                return None;
            }
            std::str::from_utf8(&value[1..])
                .ok()
                .map(|value| value.trim().to_string())
        });
    Ok((status, retry_after))
}

async fn establish_connect_tunnel(
    mut stream: BoxedWebSocketIo,
    authority: &str,
    basic_auth: Option<&http::HeaderValue>,
) -> Result<BoxedWebSocketIo, CodexError> {
    let mut request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n").into_bytes();
    if let Some(auth) = basic_auth {
        request.extend_from_slice(b"Proxy-Authorization: ");
        request.extend_from_slice(auth.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"\r\n");
    stream.write_all(&request).await.map_err(|_| CodexError {
        status: 0,
        message: "WebSocket proxy tunnel request failed".to_string(),
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    })?;

    let mut response = Vec::new();
    loop {
        if let Some(header_end) = connect_response_header_end(&response) {
            let (status, retry_after) = parse_connect_response_head(&response[..header_end])?;
            if (100..200).contains(&status) {
                response.drain(..header_end);
                continue;
            }
            if (200..300).contains(&status) {
                let prefix = response.split_off(header_end);
                return Ok(Box::new(PrefixedIo {
                    prefix,
                    position: 0,
                    inner: stream,
                }));
            }
            return Err(tunnel_error(status, retry_after));
        }
        if response.len() == MAX_CONNECT_RESPONSE_HEADER_BYTES {
            return Err(invalid_tunnel_response(
                "WebSocket proxy response headers were too large",
            ));
        }
        let remaining = MAX_CONNECT_RESPONSE_HEADER_BYTES - response.len();
        let mut buffer = [0_u8; 1024];
        let capacity = remaining.min(buffer.len());
        let read = stream
            .read(&mut buffer[..capacity])
            .await
            .map_err(|_| invalid_tunnel_response("WebSocket proxy response could not be read"))?;
        if read == 0 {
            return Err(invalid_tunnel_response(
                "WebSocket proxy closed the tunnel response early",
            ));
        }
        response.extend_from_slice(&buffer[..read]);
    }
}

async fn tls_connect(
    stream: BoxedWebSocketIo,
    host: &str,
    tls_config: Arc<rustls::ClientConfig>,
    peer: &str,
) -> Result<BoxedWebSocketIo, CodexError> {
    let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|_| websocket_protocol_error("WebSocket TLS host name was invalid"))?;
    let stream = tokio_rustls::TlsConnector::from(tls_config)
        .connect(server_name, stream)
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!("WebSocket TLS connection to {peer} failed"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        })?;
    Ok(Box::new(stream))
}

async fn connect_to_http_proxy(
    route: &WebSocketProxyRoute,
    tls_config: Arc<rustls::ClientConfig>,
) -> Result<BoxedWebSocketIo, CodexError> {
    let scheme = route.uri.scheme_str().unwrap_or_default();
    let host = route
        .uri
        .host()
        .ok_or_else(|| websocket_protocol_error("WebSocket proxy URL did not contain a host"))?;
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let port = route
        .uri
        .port_u16()
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    let stream = TcpStream::connect((host, port))
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: "WebSocket proxy connection failed".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        })?;
    let stream: BoxedWebSocketIo = Box::new(stream);
    if scheme == "https" {
        tls_connect(stream, host, tls_config, "proxy").await
    } else {
        Ok(stream)
    }
}

fn websocket_destination(url: &str) -> Result<(String, String), CodexError> {
    let destination = url::Url::parse(url)
        .map_err(|_| websocket_protocol_error("WebSocket destination URL was invalid"))?;
    let host = destination
        .host_str()
        .ok_or_else(|| websocket_protocol_error("WebSocket destination URL had no host"))?;
    let port = destination
        .port_or_known_default()
        .ok_or_else(|| websocket_protocol_error("WebSocket destination URL had no usable port"))?;
    let authority = match destination.host() {
        Some(url::Host::Ipv6(address)) => format!("[{address}]:{port}"),
        Some(_) => format!("{host}:{port}"),
        None => {
            return Err(websocket_protocol_error(
                "WebSocket destination URL had no host",
            ));
        }
    };
    Ok((host.to_string(), authority))
}

fn tungstenite_handshake_error(error: tokio_tungstenite::tungstenite::Error) -> CodexError {
    if let tokio_tungstenite::tungstenite::Error::Http(response) = error {
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        return CodexError {
            status,
            message: format!("WebSocket upgrade rejected with status {status}"),
            detail: Some(GENERIC_HANDSHAKE_ERROR_DETAIL.to_string()),
            retry_after,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        };
    }
    CodexError {
        status: 0,
        message: "WebSocket upgrade request failed".to_string(),
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    }
}

enum ConnectAttemptError {
    Origin(CodexError),
    ProxyTunnel(CodexError),
}

impl ConnectAttemptError {
    fn is_origin_forbidden(&self) -> bool {
        matches!(
            self,
            Self::Origin(error) if error.status == http::StatusCode::FORBIDDEN.as_u16()
        )
    }

    fn into_error(self) -> CodexError {
        match self {
            Self::Origin(error) | Self::ProxyTunnel(error) => error,
        }
    }
}

async fn connect_via_http_proxy_tunnel(
    proxy_config: &WebSocketProxyConfig,
    route: WebSocketProxyRoute,
    url: &str,
    headers: &HeaderMap,
) -> Result<CodexWebSocketStream, ConnectAttemptError> {
    let (host, authority) = websocket_destination(url).map_err(ConnectAttemptError::Origin)?;
    let stream = connect_to_http_proxy(&route, proxy_config.tls_config.clone())
        .await
        .map_err(ConnectAttemptError::ProxyTunnel)?;
    let stream = establish_connect_tunnel(stream, &authority, route.basic_auth.as_ref())
        .await
        .map_err(ConnectAttemptError::ProxyTunnel)?;
    let stream = tls_connect(
        stream,
        &host,
        proxy_config.tls_config.clone(),
        "destination",
    )
    .await
    .map_err(ConnectAttemptError::Origin)?;
    let websocket_key = generate_key();
    let subprotocols = requested_subprotocols(headers);
    let request = tunneled_websocket_request(url, headers, &websocket_key)
        .map_err(ConnectAttemptError::Origin)?;
    let (websocket, response) = tokio_tungstenite::client_async(request, stream)
        .await
        .map_err(tungstenite_handshake_error)
        .map_err(ConnectAttemptError::Origin)?;
    validate_websocket_upgrade(
        response.version(),
        response.headers(),
        &websocket_key,
        &subprotocols,
    )
    .map_err(ConnectAttemptError::Origin)?;
    Ok(websocket)
}

async fn connect_via_http_upgrade(
    websocket_client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
) -> Result<CodexWebSocketStream, CodexError> {
    let http_url = to_http_upgrade_url(url).map_err(|error| CodexError {
        status: 0,
        message: error.message,
        detail: None,
        retry_after: None,
        usage_limit: None,
        origin: CodexErrorOrigin::WebSocketHandshake,
    })?;
    let websocket_key = generate_key();
    let subprotocols = requested_subprotocols(headers);
    let mut request = websocket_client
        .get(http_url)
        .version(http::Version::HTTP_11)
        .header(http::header::CONNECTION, "Upgrade")
        .header(http::header::UPGRADE, "websocket")
        .header(http::header::SEC_WEBSOCKET_VERSION, "13")
        .header(http::header::SEC_WEBSOCKET_KEY, &websocket_key);

    for (key, value) in headers {
        if skip_websocket_request_header(key) {
            continue;
        }
        request = request.header(key.clone(), value.clone());
    }

    let response = request.send().await.map_err(reqwest_handshake_error)?;
    if response.status() != http::StatusCode::SWITCHING_PROTOCOLS {
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let detail = if status == http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16() {
            GENERIC_HANDSHAKE_ERROR_DETAIL.to_string()
        } else {
            let body = bounded_handshake_error_body(response).await;
            handshake_error_detail(Some(&body))
        };
        return Err(CodexError {
            status,
            message: format!("WebSocket upgrade rejected with status {status}"),
            detail: Some(detail),
            retry_after,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        });
    }

    validate_websocket_upgrade(
        response.version(),
        response.headers(),
        &websocket_key,
        &subprotocols,
    )?;
    let upgraded = response
        .upgrade()
        .await
        .map_err(|_| websocket_protocol_error("WebSocket upgrade stream was not available"))?;
    let upgraded: BoxedWebSocketIo = Box::new(upgraded);
    Ok(WebSocketStream::from_raw_socket(upgraded, Role::Client, None).await)
}

async fn connect_once(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
) -> Result<CodexWebSocketStream, ConnectAttemptError> {
    if let Some(route) = proxy_config
        .http_connect_route(url)
        .map_err(ConnectAttemptError::Origin)?
    {
        connect_via_http_proxy_tunnel(proxy_config, route, url, headers).await
    } else {
        connect_via_http_upgrade(websocket_client, url, headers)
            .await
            .map_err(ConnectAttemptError::Origin)
    }
}

fn connect_timeout_error(connect_timeout: Duration) -> CodexError {
    websocket_protocol_error(&format!(
        "WebSocket connect timeout after {}ms",
        connect_timeout.as_millis()
    ))
}

async fn connect_with_policy<T, P, PFut, F, Fut>(
    gate: &WebSocketConnectGate,
    connect_timeout: Duration,
    forbidden_cooldown: Duration,
    mut before_start: P,
    mut connect: F,
) -> Result<T, CodexError>
where
    P: FnMut() -> PFut,
    PFut: std::future::Future<Output = ()>,
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ConnectAttemptError>>,
{
    for attempt in 0..=1 {
        gate.wait_to_start(before_start()).await;
        let result = tokio::time::timeout(connect_timeout, connect())
            .await
            .unwrap_or_else(|_| {
                Err(ConnectAttemptError::Origin(connect_timeout_error(
                    connect_timeout,
                )))
            });
        if result
            .as_ref()
            .is_err_and(ConnectAttemptError::is_origin_forbidden)
            && attempt == 0
        {
            retry_sleep(u64::try_from(forbidden_cooldown.as_millis()).unwrap_or(u64::MAX)).await;
            continue;
        }
        return result.map_err(ConnectAttemptError::into_error);
    }
    unreachable!()
}

async fn connect_with_timeout(
    websocket_client: &reqwest::Client,
    proxy_config: &WebSocketProxyConfig,
    url: &str,
    headers: &HeaderMap,
    connect_timeout_ms: u64,
) -> Result<CodexWebSocketStream, CodexError> {
    connect_with_policy(
        &WS_CONNECT_GATE,
        Duration::from_millis(connect_timeout_ms),
        WEBSOCKET_CONNECT_FORBIDDEN_COOLDOWN,
        || async { cleanup_pool_with_policy(proxy_config.cache_policy) },
        || connect_once(websocket_client, proxy_config, url, headers),
    )
    .await
}

// ---------------------------------------------------------------------------
// Event collection
// ---------------------------------------------------------------------------

enum WebSocketRead {
    Frame(Option<Result<Message, tokio_tungstenite::tungstenite::Error>>),
    Timeout,
    KeepaliveError(String),
}

async fn read_ws_frame_with_keepalive<S>(
    ws: &mut WebSocketStream<S>,
    read_timeout: Duration,
    keepalive_interval: Duration,
) -> WebSocketRead
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let timeout = tokio::time::sleep(read_timeout);
    tokio::pin!(timeout);
    let mut keepalive = tokio::time::interval(keepalive_interval);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    keepalive.tick().await;

    loop {
        tokio::select! {
            biased;
            frame = ws.next() => return WebSocketRead::Frame(frame),
            _ = &mut timeout => return WebSocketRead::Timeout,
            _ = keepalive.tick() => {
                let send = ws.send(Message::Ping(Vec::new()));
                let send_timeout = tokio::time::sleep(WEBSOCKET_KEEPALIVE_SEND_TIMEOUT);
                tokio::pin!(send_timeout);
                tokio::select! {
                    biased;
                    result = send => {
                        if let Err(error) = result {
                            return WebSocketRead::KeepaliveError(error.to_string());
                        }
                    }
                    _ = &mut timeout => return WebSocketRead::Timeout,
                    _ = &mut send_timeout => {
                        return WebSocketRead::KeepaliveError(format!(
                            "send timed out after {}ms",
                            WEBSOCKET_KEEPALIVE_SEND_TIMEOUT.as_millis(),
                        ));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
struct WsEvent {
    payload: serde_json::Value,
}

// Constant-size stream metadata: count events, emit only first milestones/terminal/failure.
// Response bodies and headers never enter this logger.
struct WsDiagnostics {
    logger: Logger,
    started: Instant,
    events: u64,
    response_events: u64,
    attempt_id: u64,
    started_us: u64,
    phase_times_us: [Option<u64>; 4],
}

const TTFT_PHASES: [&str; 4] = [
    "response_created",
    "reasoning_started",
    "first_summary_delta",
    "first_text_delta",
];
static NEXT_DIAGNOSTIC_ATTEMPT: AtomicU64 = AtomicU64::new(0);

fn diagnostic_text(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars().filter(|ch| !ch.is_control()) {
        if output.len() + ch.len_utf8() > 256 {
            break;
        }
        output.push(ch);
    }
    output
}

impl WsDiagnostics {
    fn new(
        req_id: Option<&str>,
        headers: &HeaderMap,
        reservation: Option<&ContinuationReservation>,
    ) -> Self {
        let mut fields = serde_json::Map::new();
        if let Some(req_id) = req_id.or_else(|| {
            headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
        }) {
            fields.insert("reqId".into(), diagnostic_text(req_id).into());
        }
        if let Some(model) = reservation.and_then(ContinuationReservation::model) {
            fields.insert("model".into(), diagnostic_text(model).into());
        }
        if let Some(owner) = reservation.and_then(ContinuationReservation::owner) {
            match owner {
                ConversationIdentity::Main(session) => {
                    fields.insert("sessionId".into(), diagnostic_text(session).into());
                }
                ConversationIdentity::Agent(session, agent) => {
                    fields.insert("sessionId".into(), diagnostic_text(session).into());
                    fields.insert("agentId".into(), diagnostic_text(agent).into());
                }
            }
        }
        Self {
            logger: create_logger("codex").child(fields),
            started: Instant::now(),
            events: 0,
            response_events: 0,
            attempt_id: next_monotonic_nonzero(&NEXT_DIAGNOSTIC_ATTEMPT, "diagnostic attempt"),
            started_us: super::timing::monotonic_us(),
            phase_times_us: [None; 4],
        }
    }

    fn fields(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::from_iter([
            (
                "elapsedMs".into(),
                serde_json::json!(self.started.elapsed().as_millis()),
            ),
            ("eventCount".into(), self.events.into()),
            ("responseEventCount".into(), self.response_events.into()),
            ("attemptId".into(), self.attempt_id.into()),
            ("processId".into(), std::process::id().into()),
            ("attemptStartedUs".into(), self.started_us.into()),
            ("monotonicUs".into(), super::timing::monotonic_us().into()),
            (
                "phaseTimesUs".into(),
                serde_json::Value::Object(
                    TTFT_PHASES
                        .iter()
                        .zip(self.phase_times_us)
                        .map(|(name, time)| {
                            (
                                (*name).into(),
                                time.map_or(serde_json::Value::Null, Into::into),
                            )
                        })
                        .collect(),
                ),
            ),
        ])
    }

    fn ready(&mut self, entry: &PoolEntry, reused: bool) {
        self.logger = self.logger.child(serde_json::Map::from_iter([
            ("socketId".into(), entry.socket_id.into()),
            ("reused".into(), reused.into()),
        ]));
        self.logger.info("websocket_ready", Some(self.fields()));
    }

    fn model(&mut self, body: &serde_json::Value) {
        if let Some(model) = body.get("model").and_then(|value| value.as_str()) {
            self.logger = self.logger.child(serde_json::Map::from_iter([(
                "model".into(),
                diagnostic_text(model).into(),
            )]));
        }
    }

    fn event(&mut self, payload: &serde_json::Value) {
        self.events += 1;
        let phase = match payload.get("type").and_then(serde_json::Value::as_str) {
            Some("response.created") => Some(0),
            Some("response.output_item.added")
                if payload
                    .pointer("/item/type")
                    .and_then(serde_json::Value::as_str)
                    == Some("reasoning") =>
            {
                Some(1)
            }
            Some("response.reasoning_summary_text.delta")
                if payload
                    .get("delta")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| !s.is_empty()) =>
            {
                Some(2)
            }
            Some("response.output_text.delta")
                if payload
                    .get("delta")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| !s.is_empty()) =>
            {
                Some(3)
            }
            _ => None,
        };
        if let Some(index) = phase.filter(|&index| self.phase_times_us[index].is_none()) {
            let observed = super::timing::monotonic_us();
            self.phase_times_us[index] = Some(observed);
            let mut fields = self.fields();
            fields.insert("monotonicUs".into(), observed.into());
            fields.insert("phase".into(), TTFT_PHASES[index].into());
            fields.insert("boundary".into(), "upstream_reader".into());
            self.logger.info("websocket_phase", Some(fields));
        }
        if is_response_event(payload) {
            self.response_events += 1;
            if self.response_events == 1 {
                self.logger.info(
                    "websocket_first_response",
                    Some({
                        let mut fields = self.fields();
                        fields.insert(
                            "event".into(),
                            diagnostic_text(
                                payload
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown"),
                            )
                            .into(),
                        );
                        fields
                    }),
                );
            }
        }
        if let Some(outcome) = super::events::terminal_outcome(payload) {
            let mut fields = self.fields();
            fields.insert("phase".into(), "upstream_terminal".into());
            fields.insert("outcome".into(), outcome.into());
            fields.insert(
                "event".into(),
                diagnostic_text(
                    payload
                        .get("type")
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown"),
                )
                .into(),
            );
            fields.insert(
                "status".into(),
                event_error_status(payload).unwrap_or(200).into(),
            );
            if let Some(status) = payload
                .pointer("/response/status")
                .and_then(|value| value.as_str())
            {
                fields.insert("responseStatus".into(), diagnostic_text(status).into());
            }
            self.logger.info("websocket_terminal", Some(fields));
        }
    }

    fn timeout(&self, response_started: bool, waited: Duration, budget_ms: u64) {
        let mut fields = self.fields();
        fields.insert(
            "phase".into(),
            if response_started {
                "response_idle"
            } else {
                "response_start"
            }
            .into(),
        );
        fields.insert("waitedMs".into(), serde_json::json!(waited.as_millis()));
        fields.insert("timeoutMs".into(), budget_ms.into());
        self.logger.warn("websocket_timeout", Some(fields));
    }

    fn failure(&self, phase: &str, error: &str) {
        let mut fields = self.fields();
        fields.insert("phase".into(), phase.into());
        fields.insert("error".into(), diagnostic_text(error).into());
        self.logger.warn("websocket_error", Some(fields));
    }

    fn close(&self, close: Option<&tokio_tungstenite::tungstenite::protocol::CloseFrame<'_>>) {
        let mut fields = self.fields();
        fields.insert("terminalSeen".into(), false.into());
        if let Some(close) = close {
            fields.insert("closeCode".into(), u16::from(close.code).into());
            fields.insert("closeReason".into(), diagnostic_text(&close.reason).into());
        }
        self.logger.warn("websocket_close", Some(fields));
    }

    fn eof(&self) {
        let mut fields = self.fields();
        fields.insert("terminalSeen".into(), false.into());
        self.logger.warn("websocket_eof", Some(fields));
    }
}

#[cfg(test)]
async fn collect_ws_events<S>(
    ws: &mut WebSocketStream<S>,
    idle_timeout_ms: u64,
    pool_owner: Option<&ConversationIdentity>,
    pool_entry: Option<&Arc<PoolEntry>>,
    traffic: Option<&TrafficCapture>,
    diagnostics: Option<&mut WsDiagnostics>,
) -> Result<(Vec<u8>, Option<WsEvent>), CodexError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut fallback = WsDiagnostics::new(None, &HeaderMap::new(), None);
    let diagnostics = diagnostics.unwrap_or(&mut fallback);
    if let Some(entry) = pool_entry {
        diagnostics.logger = diagnostics.logger.child(serde_json::Map::from_iter([(
            "socketId".into(),
            entry.socket_id.into(),
        )]));
    }
    collect_ws_events_with_keepalive_interval(
        ws,
        idle_timeout_ms,
        pool_owner,
        pool_entry,
        traffic,
        WEBSOCKET_KEEPALIVE_INTERVAL,
        Some(diagnostics),
    )
    .await
}

#[cfg(test)]
async fn collect_ws_events_with_keepalive_interval<S>(
    ws: &mut WebSocketStream<S>,
    idle_timeout_ms: u64,
    pool_owner: Option<&ConversationIdentity>,
    pool_entry: Option<&Arc<PoolEntry>>,
    traffic: Option<&TrafficCapture>,
    keepalive_interval: Duration,
    diagnostics: Option<&mut WsDiagnostics>,
) -> Result<(Vec<u8>, Option<WsEvent>), CodexError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (sender, mut receiver) = mpsc::channel(64);
    let tx = EventSender {
        sender,
        aborted: Arc::new(AtomicBool::new(false)),
    };
    let read = async {
        let result = stream_ws_events_with_keepalive_interval(
            ws,
            idle_timeout_ms,
            traffic,
            &tx,
            diagnostics,
            keepalive_interval,
        )
        .await;
        drop(tx);
        result
    };
    let collect = async {
        let mut body = Vec::new();
        while let Some(item) = receiver.recv().await {
            body.extend_from_slice(&encode_sse(&item.unwrap().to_string()));
        }
        body
    };
    let ((reusable, terminal), mut body) = tokio::join!(read, collect);
    if !reusable {
        invalidate_pool_owner(pool_owner, pool_entry);
    }
    let terminal = match terminal {
        Some(item) => Some(WsEvent { payload: item? }),
        None => None,
    };
    if let Some(event) = terminal.as_ref() {
        body.extend_from_slice(&encode_sse(&event.payload.to_string()));
    }
    Ok((body, terminal))
}

async fn stream_ws_events<S>(
    ws: &mut WebSocketStream<S>,
    idle_timeout_ms: u64,
    traffic: Option<&TrafficCapture>,
    tx: &EventSender,
    diagnostics: Option<&mut WsDiagnostics>,
) -> (bool, Option<Result<serde_json::Value, CodexError>>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    stream_ws_events_with_keepalive_interval(
        ws,
        idle_timeout_ms,
        traffic,
        tx,
        diagnostics,
        WEBSOCKET_KEEPALIVE_INTERVAL,
    )
    .await
}

async fn stream_ws_events_with_keepalive_interval<S>(
    ws: &mut WebSocketStream<S>,
    idle_timeout_ms: u64,
    traffic: Option<&TrafficCapture>,
    tx: &EventSender,
    diagnostics: Option<&mut WsDiagnostics>,
    keepalive_interval: Duration,
) -> (bool, Option<Result<serde_json::Value, CodexError>>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut fallback = WsDiagnostics::new(None, &HeaderMap::new(), None);
    let diagnostics = diagnostics.unwrap_or(&mut fallback);
    let started_at = Instant::now();
    let response_event_budget = Duration::from_millis(idle_timeout_ms);
    let response_wait_started = Instant::now();
    let mut last_response_event_at = response_wait_started;
    let mut response_started = false;
    let mut status = 200u16;
    let mut reusable = false;
    let mut terminal_item = None;

    loop {
        let response_deadline_started = if response_started {
            last_response_event_at
        } else {
            response_wait_started
        };
        let read_timeout =
            match response_event_budget.checked_sub(response_deadline_started.elapsed()) {
                Some(remaining) if !remaining.is_zero() => remaining,
                _ => {
                    diagnostics.timeout(
                        response_started,
                        response_deadline_started.elapsed(),
                        idle_timeout_ms,
                    );
                    terminal_item = Some(Err(if response_started {
                        CodexError {
                            status: 0,
                            message: format!("WebSocket idle timeout after {idle_timeout_ms}ms"),
                            detail: None,
                            retry_after: None,
                            usage_limit: None,
                            origin: CodexErrorOrigin::WebSocket,
                        }
                    } else {
                        response_start_timeout_error(idle_timeout_ms)
                    }));
                    break;
                }
            };

        let frame = tokio::select! {
            biased;
            _ = tx.closed() => {
                diagnostics.failure("downstream_delivery", "consumer closed");
                break;
            },
            frame = read_ws_frame_with_keepalive(
                ws,
                read_timeout,
                keepalive_interval,
            ) => match frame {
                WebSocketRead::Frame(frame) => frame,
                WebSocketRead::Timeout => {
                    diagnostics.timeout(response_started, response_deadline_started.elapsed(), idle_timeout_ms);
                    terminal_item = Some(Err(if response_started {
                        CodexError {
                            status: 0,
                            message: format!("WebSocket idle timeout after {idle_timeout_ms}ms"),
                            detail: None,
                            retry_after: None,
                            usage_limit: None,
                            origin: CodexErrorOrigin::WebSocket,
                        }
                    } else {
                        response_start_timeout_error(idle_timeout_ms)
                    }));
                    break;
                }
                WebSocketRead::KeepaliveError(error) => {
                    diagnostics.failure("keepalive_write", &error);
                    terminal_item = Some(Err(CodexError {
                        status: 0,
                        message: format!("WebSocket keepalive error: {error}"),
                        detail: Some(WEBSOCKET_KEEPALIVE_FAILURE_DETAIL.into()),
                        retry_after: None,
                        usage_limit: None,
                        origin: CodexErrorOrigin::WebSocket,
                    }));
                    break;
                }
            },
        };

        match frame {
            Some(Ok(Message::Text(text))) => {
                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(_) => {
                        diagnostics.failure("response_parse", "malformed JSON event");
                        if let Some(tc) = traffic {
                            tc.write_json_event(
                                "040-upstream-event",
                                &serde_json::json!({
                                    "unparseable": true,
                                    "bytes": text.len(),
                                }),
                            );
                        }
                        terminal_item = Some(Err(malformed_event_error()));
                        break;
                    }
                };

                if handle_connection_metadata(&parsed) && parsed["type"] != "codex.rate_limits" {
                    continue;
                }
                diagnostics.event(&parsed);

                if let Some(tc) = traffic {
                    tc.write_json_event("040-upstream-event", &parsed);
                }

                if is_response_event(&parsed) {
                    response_started = true;
                    last_response_event_at = Instant::now();
                }
                if parsed.get("type").and_then(|value| value.as_str()) == Some("error") {
                    status = event_error_status(&parsed).unwrap_or(500);
                }

                if is_terminal_event(&parsed) {
                    if is_previous_response_missing(&parsed) {
                        terminal_item = Some(Err(CodexError {
                            status: 0,
                            message: "Previous response not found".to_string(),
                            detail: Some("previous_response_not_found".to_string()),
                            retry_after: None,
                            usage_limit: None,
                            origin: CodexErrorOrigin::WebSocket,
                        }));
                    } else {
                        reusable = super::events::event_is_success_terminal(&parsed);
                        terminal_item = Some(Ok(parsed));
                    }
                    break;
                }

                if tx.send_bounded(Ok(parsed)).await.is_err() {
                    diagnostics.failure("downstream_delivery", "consumer stalled or closed");
                    break;
                }
            }
            Some(Ok(Message::Binary(_))) => {
                diagnostics.failure("response_read", "binary frames not supported");
                terminal_item = Some(Err(CodexError {
                    status: 0,
                    message: "WebSocket binary frames not supported".to_string(),
                    detail: None,
                    retry_after: None,
                    usage_limit: None,
                    origin: CodexErrorOrigin::WebSocket,
                }));
                break;
            }
            Some(Ok(Message::Ping(data))) => {
                let pong = tokio::select! {
                    biased;
                    _ = tx.closed() => break,
                    result = tokio::time::timeout(WEBSOCKET_KEEPALIVE_SEND_TIMEOUT, ws.send(Message::Pong(data))) => result,
                };
                if !matches!(pong, Ok(Ok(()))) {
                    diagnostics.failure("pong_write", "Pong write failed or timed out");
                    terminal_item = Some(Err(pong_write_error()));
                    break;
                }
            }
            Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
            Some(Ok(Message::Close(close))) => {
                diagnostics.close(close.as_ref());
                terminal_item = Some(Err(missing_terminal_error()));
                break;
            }
            None => {
                diagnostics.eof();
                terminal_item = Some(Err(missing_terminal_error()));
                break;
            }
            Some(Err(error)) => {
                diagnostics.failure("response_read", &error.to_string());
                terminal_item = Some(Err(CodexError {
                    status: 0,
                    message: format!("WebSocket stream error: {error}"),
                    detail: None,
                    retry_after: None,
                    usage_limit: None,
                    origin: CodexErrorOrigin::WebSocket,
                }));
                break;
            }
        }
    }

    if let Some(tc) = traffic {
        // Individual events are captured above. Buffered collection alone keeps
        // a complete SSE body; live capture intentionally omits that raw mirror.
        tc.write_json(
            "030-upstream-response-headers",
            &serde_json::json!({
                "status": status, "elapsedMs": started_at.elapsed().as_millis(),
                "headers": {"content-type":"text/event-stream"},
                "bodyCapture": "per-event records; live SSE body omitted"
            }),
        );
    }
    (reusable, terminal_item)
}

fn headers_to_json(headers: &HeaderMap) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in headers.iter() {
        out.insert(
            key.to_string(),
            serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
        );
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use super::*;

    fn diagnostics_settings(root: &std::path::Path) -> Arc<crate::config::CodexRequestSettings> {
        crate::config::CodexRequestSettings::for_env(HashMap::from([
            (
                "CCP_CONFIG_DIR".into(),
                root.join("config").display().to_string(),
            ),
            (
                "XDG_STATE_HOME".into(),
                root.join("state").display().to_string(),
            ),
        ]))
    }

    fn diagnostic_records(path: &std::path::Path) -> Vec<serde_json::Value> {
        assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn diagnostic_outcomes_keep_websocket_terminal_scope_and_attempt() {
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
        ] {
            let temp = tempfile::tempdir().unwrap();
            let settings = diagnostics_settings(temp.path());
            let path = settings.clone().sync_scope(crate::logging::log_file);
            settings
                .scope(async {
                    let (io, peer) = tokio::io::duplex(16 * 1024);
                    let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
                    let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
                    for event in [
                        serde_json::json!({"type":"response.created"}),
                        terminal.clone(),
                    ] {
                        peer.send(Message::Text(event.to_string())).await.unwrap();
                    }
                    let mut diagnostics =
                        WsDiagnostics::new(Some("ws-outcome"), &HeaderMap::new(), None);
                    // Upstream failure is still a terminal event, forwarded unchanged.
                    let (bytes, _) = collect_ws_events(
                        &mut client,
                        1_000,
                        None,
                        None,
                        None,
                        Some(&mut diagnostics),
                    )
                    .await
                    .unwrap();
                    assert!(
                        String::from_utf8(bytes)
                            .unwrap()
                            .contains(terminal["type"].as_str().unwrap())
                    );
                })
                .await;
            let records = diagnostic_records(&path);
            let first = records
                .iter()
                .find(|r| r["msg"] == "websocket_first_response")
                .unwrap();
            let record = records
                .iter()
                .find(|r| r["msg"] == "websocket_terminal")
                .unwrap();
            assert_eq!(record["fields"]["phase"], "upstream_terminal");
            assert_eq!(record["fields"]["outcome"], outcome);
            assert_eq!(record["fields"]["event"], terminal["type"]);
            assert_eq!(record["fields"]["reqId"], "ws-outcome");
            assert!(record["fields"]["attemptId"].is_u64());
            assert_eq!(record["fields"]["attemptId"], first["fields"]["attemptId"]);
            assert_eq!(record["fields"]["processId"], first["fields"]["processId"]);
        }
    }

    #[tokio::test]
    async fn websocket_ttft_phases_are_once_only_and_content_free() {
        let temp = tempfile::tempdir().unwrap();
        let settings = diagnostics_settings(temp.path());
        let path = settings.clone().sync_scope(crate::logging::log_file);
        settings.scope(async {
            let (io, peer) = tokio::io::duplex(16 * 1024);
            let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
            let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
            for event in [
                serde_json::json!({"type":"response.created"}),
                serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"private-id"}}),
                serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":""}),
                serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"private-summary"}),
                serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"private-summary"}),
                serde_json::json!({"type":"response.output_text.delta","output_index":1,"delta":""}),
                serde_json::json!({"type":"response.output_text.delta","output_index":1,"delta":"private-answer"}),
                serde_json::json!({"type":"response.output_text.delta","output_index":1,"delta":"private-answer"}),
                serde_json::json!({"type":"response.completed","response":{"status":"completed"}}),
            ] {
                peer.send(Message::Text(event.to_string())).await.unwrap();
            }
            collect_ws_events(&mut client, 1_000, None, None, None, None).await.unwrap();
        }).await;
        let logs = diagnostic_records(&path);
        let phases: Vec<_> = logs
            .iter()
            .filter(|r| r["msg"] == "websocket_phase")
            .collect();
        assert_eq!(
            phases.len(),
            4,
            "each first upstream milestone must be logged once"
        );
        let mut previous = 0;
        for (record, expected) in phases.iter().zip([
            "response_created",
            "reasoning_started",
            "first_summary_delta",
            "first_text_delta",
        ]) {
            assert_eq!(record["fields"]["phase"], expected);
            assert_eq!(record["fields"]["boundary"], "upstream_reader");
            assert!(record["fields"]["attemptId"].as_u64().is_some());
            let observed = record["fields"]["monotonicUs"].as_u64().unwrap();
            assert!(observed >= previous);
            previous = observed;
        }
        let serialized = serde_json::to_string(&logs).unwrap();
        assert!(!serialized.contains("private-"));
        let terminal = logs
            .iter()
            .find(|r| r["msg"] == "websocket_terminal")
            .unwrap();
        assert!(terminal["fields"]["phaseTimesUs"]["first_summary_delta"].is_u64());
    }

    #[tokio::test]
    async fn websocket_ttft_missing_phases_remain_unknown() {
        let temp = tempfile::tempdir().unwrap();
        let settings = diagnostics_settings(temp.path());
        let path = settings.clone().sync_scope(crate::logging::log_file);
        settings
            .scope(async {
                let (io, peer) = tokio::io::duplex(1024);
                let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
                let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
                peer.send(Message::Text(
                    r#"{"type":"response.completed","response":{"status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
                collect_ws_events(&mut client, 1_000, None, None, None, None)
                    .await
                    .unwrap();
            })
            .await;
        let logs = diagnostic_records(&path);
        let terminal = logs
            .iter()
            .find(|r| r["msg"] == "websocket_terminal")
            .unwrap();
        for name in [
            "response_created",
            "reasoning_started",
            "first_summary_delta",
            "first_text_delta",
        ] {
            assert_eq!(
                terminal["fields"]["phaseTimesUs"].get(name),
                Some(&serde_json::Value::Null)
            );
        }
    }

    #[tokio::test]
    async fn websocket_diagnostics_logs_first_and_terminal_without_delta_payloads() {
        let temp = tempfile::tempdir().unwrap();
        let settings = diagnostics_settings(temp.path());
        let path = settings.clone().sync_scope(crate::logging::log_file);
        settings
            .scope(async {
                let (io, peer) = tokio::io::duplex(16 * 1024);
                let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
                let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
                peer.send(Message::Text(r#"{"type":"response.created"}"#.into()))
                    .await
                    .unwrap();
                for _ in 0..20 {
                    peer.send(Message::Text(
                        r#"{"type":"response.output_text.delta","delta":"private-output"}"#.into(),
                    ))
                    .await
                    .unwrap();
                }
                peer.send(Message::Text(
                    r#"{"type":"response.completed","response":{"status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
                collect_ws_events(&mut client, 1_000, None, None, None, None)
                    .await
                    .unwrap();
            })
            .await;
        let logs = diagnostic_records(&path);
        let logs: Vec<_> = logs
            .into_iter()
            .filter(|r| r["msg"] != "websocket_phase")
            .collect();
        assert_eq!(
            logs.len(),
            2,
            "only first response and terminal should be emitted"
        );
        assert_eq!(logs[0]["msg"], "websocket_first_response");
        assert_eq!(logs[1]["msg"], "websocket_terminal");
        assert_eq!(logs[1]["fields"]["eventCount"], 22);
        assert_eq!(logs[1]["fields"]["status"], 200);
        assert!(
            !logs
                .iter()
                .any(|log| log.to_string().contains("private-output"))
        );
    }

    #[tokio::test]
    async fn websocket_diagnostics_preserves_close_code_and_sanitizes_reason() {
        for live in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let settings = diagnostics_settings(temp.path());
            let path = settings.clone().sync_scope(crate::logging::log_file);
            settings.scope(async {
                let (io, peer) = tokio::io::duplex(1024);
                let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
                let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
                peer.send(Message::Close(Some(tokio_tungstenite::tungstenite::protocol::CloseFrame {
                    code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Policy,
                    reason: "policy\nclosed".into(),
                }))).await.unwrap();
                if live {
                    let (tx, rx) = mpsc::channel(64);
                    let (_stream, publisher) = CodexWebSocketEventStream::pending(rx);
                    let (reusable, terminal) = stream_ws_events(&mut client, 1_000, None, &publisher.sender(tx), None).await;
                    assert!(!reusable);
                    assert!(terminal.unwrap().is_err());
                } else {
                    assert!(collect_ws_events(&mut client, 1_000, None, None, None, None).await.unwrap_err().detail.as_deref() == Some(WEBSOCKET_MISSING_TERMINAL_DETAIL));
                }
            }).await;
            let logs = diagnostic_records(&path);
            let close = logs
                .iter()
                .find(|log| log["msg"] == "websocket_close")
                .expect("close metadata missing");
            assert_eq!(close["fields"]["closeCode"], 1008);
            assert_eq!(close["fields"]["closeReason"], "policyclosed");
            assert_eq!(close["fields"]["terminalSeen"], false);
        }
    }

    #[tokio::test]
    async fn websocket_diagnostics_names_timeout_phase_and_wait() {
        for (started, phase) in [(false, "response_start"), (true, "response_idle")] {
            let temp = tempfile::tempdir().unwrap();
            let settings = diagnostics_settings(temp.path());
            let path = settings.clone().sync_scope(crate::logging::log_file);
            settings
                .scope(async {
                    let (io, peer) = tokio::io::duplex(1024);
                    let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
                    let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
                    if started {
                        peer.send(Message::Text(r#"{"type":"response.created"}"#.into()))
                            .await
                            .unwrap();
                    }
                    assert!(
                        collect_ws_events(&mut client, 20, None, None, None, None)
                            .await
                            .is_err()
                    );
                })
                .await;
            let logs = diagnostic_records(&path);
            let timeout = logs
                .iter()
                .find(|log| log["msg"] == "websocket_timeout")
                .expect("timeout metadata missing");
            assert_eq!(timeout["fields"]["phase"], phase);
            assert!(timeout["fields"]["waitedMs"].as_u64().unwrap() >= 10);
            assert_eq!(timeout["fields"]["timeoutMs"], 20);
        }
    }

    #[test]
    fn provider_retry_handoff_is_attempt_local() {
        let (_tx_a, rx_a) = mpsc::channel(1);
        let (stream_a, publisher_a) = CodexWebSocketEventStream::pending(rx_a);
        let (_tx_b, rx_b) = mpsc::channel(1);
        let (_stream_b, publisher_b) = CodexWebSocketEventStream::pending(rx_b);

        assert!(!publisher_a.is_provider_retry_handoff());
        assert!(!publisher_b.is_provider_retry_handoff());
        stream_a.mark_provider_retry_handoff();
        assert!(publisher_a.is_provider_retry_handoff());
        assert!(!publisher_b.is_provider_retry_handoff());
    }

    #[tokio::test]
    async fn idle_reader_answers_ping_and_isolates_metadata_between_turns() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("idle-reader");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/responses", listener.local_addr().unwrap());
        let (idle_tx, idle_rx) = tokio::sync::oneshot::channel();
        let (pong_tx, pong_rx) = tokio::sync::oneshot::channel();
        let (close_tx, close_rx) = tokio::sync::oneshot::channel();
        let peer = tokio::spawn(async move {
            let (io, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(io).await.unwrap();
            assert!(matches!(ws.next().await, Some(Ok(Message::Text(_)))));
            ws.send(Message::Text(
                serde_json::json!({"type":"response.completed","response":{"id":"first"}})
                    .to_string(),
            ))
            .await
            .unwrap();
            idle_rx.await.unwrap();
            ws.send(Message::Text(
                serde_json::json!({"type":"codex.rate_limits","rate_limits":{"primary":{"used_percent":99}}}).to_string(),
            ))
            .await
            .unwrap();
            // Pong acknowledges the preceding idle metadata before the next create.
            ws.send(Message::Ping(vec![1, 2, 3])).await.unwrap();
            assert!(matches!(ws.next().await, Some(Ok(Message::Pong(p))) if p == [1,2,3]));
            pong_tx.send(()).unwrap();
            assert!(matches!(ws.next().await, Some(Ok(Message::Text(_)))));
            ws.send(Message::Text(
                serde_json::json!({"type":"response.completed","response":{"id":"second"}})
                    .to_string(),
            ))
            .await
            .unwrap();
            close_rx.await.unwrap();
            ws.close(None).await.unwrap();
        });
        let request = continuation_request();
        let context = RequestContext {
            req_id: "idle-test".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let mut events = codex_websocket_event_stream(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &endpoint,
            &HeaderMap::new(),
            &serde_json::json!({"type":"response.create","input":[]}),
            &context,
            None,
            1_000,
            1_000,
            Some(&reservation),
        )
        .await
        .unwrap();
        assert_eq!(
            events.recv().await.unwrap().unwrap()["type"],
            "response.completed"
        );
        let socket_id = events.socket_id();
        events.accept_terminal();
        idle_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_millis(500), pong_rx)
            .await
            .expect("idle Ping must be answered before next create")
            .unwrap();
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let mut events = codex_websocket_event_stream(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &endpoint,
            &HeaderMap::new(),
            &serde_json::json!({"type":"response.create","input":[]}),
            &context,
            None,
            1_000,
            1_000,
            Some(&reservation),
        )
        .await
        .unwrap();
        assert_eq!(
            events.recv().await.unwrap().unwrap()["type"],
            "response.completed"
        );
        assert_eq!(events.socket_id(), socket_id);
        events.accept_terminal();
        close_tx.send(()).unwrap();
        peer.await.unwrap();
        // Let the persistent reader consume the Close, without a new request/probe.
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if WS_POOL
                    .lock()
                    .unwrap()
                    .get(&owner)
                    .is_none_or(|entry| entry.connection.is_closed())
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(
            pool_take_for_turn(&reservation).is_none(),
            "closed socket must not be handed out"
        );
        invalidate_codex_websocket_pool_owner(&owner);
    }

    fn main_owner(session_id: &str) -> ConversationIdentity {
        ConversationIdentity::Main(session_id.to_string())
    }

    fn agent_owner(session_id: &str, agent_id: &str) -> ConversationIdentity {
        ConversationIdentity::Agent(session_id.to_string(), agent_id.to_string())
    }

    fn test_continuation(
        owner: Option<ConversationIdentity>,
        turn_id: Option<u64>,
        previous_response_id: Option<&str>,
        origin_socket_id: Option<u64>,
    ) -> ContinuationReservation {
        ContinuationReservation::new(
            super::super::continuation::ContinuationCandidate {
                turn_id,
                previous_response_id: previous_response_id.map(str::to_string),
                input_delta: Some(vec![]),
                input_delta_count: 0,
                disabled_reason: None,
            },
            owner,
            origin_socket_id,
        )
    }

    fn continuation_request() -> super::super::translate::request::ResponsesRequest {
        super::super::translate::request::ResponsesRequest {
            model: "gpt-5.6-sol".to_string(),
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

    fn test_websocket_client() -> reqwest::Client {
        reqwest::Client::builder()
            .http1_only()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .unwrap()
    }

    fn handshake_error(status: u16) -> CodexError {
        CodexError {
            status,
            message: format!("handshake failed with status {status}"),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        }
    }

    fn origin_forbidden_error() -> ConnectAttemptError {
        ConnectAttemptError::Origin(handshake_error(http::StatusCode::FORBIDDEN.as_u16()))
    }

    struct DropProbeIo {
        probe: Option<(Arc<AtomicBool>, Arc<AtomicBool>)>,
    }

    impl Drop for DropProbeIo {
        fn drop(&mut self) {
            if let Some((dropped, pool_was_unlocked)) = &self.probe {
                pool_was_unlocked.store(WS_POOL.try_lock().is_ok(), Ordering::SeqCst);
                dropped.store(true, Ordering::SeqCst);
            }
        }
    }

    impl AsyncRead for DropProbeIo {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buffer: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Pending
        }
    }

    impl AsyncWrite for DropProbeIo {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Ok(buffer.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    struct FailingWriteIo;

    impl AsyncRead for FailingWriteIo {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buffer: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Pending
        }
    }

    impl AsyncWrite for FailingWriteIo {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "test write failed",
            )))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    async fn raw_test_stream(
        probe: Option<(Arc<AtomicBool>, Arc<AtomicBool>)>,
    ) -> CodexWebSocketStream {
        let io: BoxedWebSocketIo = Box::new(DropProbeIo { probe });
        WebSocketStream::from_raw_socket(io, Role::Client, None).await
    }

    fn shared_pool_entry(ws: &ConnectionHandle) -> Arc<PoolEntry> {
        Arc::new(PoolEntry {
            compatibility: None,
            published_turn: AtomicU64::new(0),
            connection: ws.clone(),
            socket_id: next_monotonic_nonzero(&NEXT_SOCKET_ID, "WebSocket ID"),
            created_at: tokio::time::Instant::now(),
            idle_since: Mutex::new(tokio::time::Instant::now()),
            last_activity: AtomicU64::new(next_pool_activity()),
        })
    }

    #[tokio::test]
    async fn pooled_peer_compatibility_rejects_auth_account_endpoint_flavor_and_client_changes() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = lock_codex_websocket_pool_for_tests().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/responses", listener.local_addr().unwrap());
        let (capture_tx, mut captures) = mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let mut peers = tokio::task::JoinSet::new();
            let mut ordinal = 0;
            loop {
                let (io, _) = listener.accept().await.unwrap();
                ordinal += 1;
                let capture = capture_tx.clone();
                peers.spawn(async move {
                    let mut peer = tokio_tungstenite::accept_async(io).await.unwrap();
                    while let Some(Ok(frame)) = peer.next().await {
                        match frame {
                            Message::Ping(data) => { if peer.send(Message::Pong(data)).await.is_err() { break; } },
                            Message::Text(text) => {
                                let body: serde_json::Value = serde_json::from_str(&text).unwrap();
                                assert!(body.get("previous_response_id").is_none());
                                capture.send(ordinal).unwrap();
                                peer.send(Message::Text(serde_json::json!({"type":"response.completed","response":{"id":"ok","status":"completed"}}).to_string())).await.unwrap();
                            },
                            _ => break,
                        }
                    }
                });
            }
        });
        let owner = main_owner("compatibility-matrix");
        let request = continuation_request();
        let context = RequestContext {
            req_id: "compatibility".into(),
            session_id: Some("compatibility-matrix".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let client = test_websocket_client();
        let mut config = WebSocketProxyConfig::new(None, None, None, None);
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer revision-one".parse().unwrap());
        headers.insert("chatgpt-account-id", "account-one".parse().unwrap());
        let mut url = endpoint.clone();
        let mut previous_socket = 0;
        for change in [
            "initial", "same", "auth", "account", "endpoint", "flavor", "client",
        ] {
            match change {
                "auth" => {
                    headers.insert("authorization", "Bearer revision-two".parse().unwrap());
                }
                "account" => {
                    headers.insert("chatgpt-account-id", "account-two".parse().unwrap());
                }
                "endpoint" => {
                    url = format!("{endpoint}?route=other");
                }
                "flavor" => {
                    headers.insert(
                        "x-openai-internal-codex-responses-lite",
                        "true".parse().unwrap(),
                    );
                }
                "client" => {
                    config = WebSocketProxyConfig::new(None, None, None, None);
                }
                _ => {}
            }
            let reservation = super::super::continuation::reserve_owner_request(
                Some(&owner),
                &request,
                false,
                true,
            );
            codex_websocket_request(&client, &config, &url, &headers, &serde_json::json!({"type":"response.create","model":request.model,"input":request.input}), &context, None, 15_000, 300_000, Some(&reservation)).await.unwrap();
            let socket = captures.recv().await.unwrap();
            if change == "same" {
                assert_eq!(socket, previous_socket);
            } else {
                assert_ne!(
                    socket, previous_socket,
                    "{change} must not reuse incompatible provenance"
                );
            }
            previous_socket = socket;
        }
        invalidate_codex_websocket_pool_owner(&owner);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn dropped_consumer_interrupts_partial_request_write() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("partial-write-cancellation");
        let request = continuation_request();
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let (io, mut peer) = tokio::io::duplex(1);
        let io: BoxedWebSocketIo = Box::new(io);
        let websocket = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
        let entry = Arc::new(PoolEntry::new(websocket));
        let ready = ReadyWebSocket {
            ws_url: "ws://local-duplex".into(),
            entry,
            used_pooled: false,
            reservation: Some(reservation.clone()),
            traffic: None,
            idle_timeout_ms: 300_000,
            diagnostics: WsDiagnostics::new(None, &HeaderMap::new(), None),
        };
        let body = serde_json::json!({"type":"response.create","input":"x".repeat(4096)});
        let events = start_codex_websocket_events(
            ready,
            &body,
            body.to_string(),
            &HeaderMap::new(),
            Some(&reservation),
        );
        let mut first_byte = [0];
        peer.read_exact(&mut first_byte).await.unwrap();
        drop(events);
        let mut remainder = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), peer.read_to_end(&mut remainder))
            .await
            .expect("consumer cancellation must drop a partially written socket immediately")
            .unwrap();
        assert!(pooled_socket_id_for_tests(&owner).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn blocked_pong_write_terminates_buffered_and_live_attempts() {
        for live in [false, true] {
            let (io, peer) = tokio::io::duplex(64);
            let mut client = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
            let mut peer = WebSocketStream::from_raw_socket(peer, Role::Server, None).await;
            let reader = tokio::spawn(async move {
                if live {
                    let (tx, rx) = mpsc::channel(64);
                    let (_stream, publisher) = CodexWebSocketEventStream::pending(rx);
                    let tx = publisher.sender(tx);
                    let (reusable, terminal) =
                        stream_ws_events(&mut client, 300_000, None, &tx, None).await;
                    assert!(!reusable);
                    terminal.is_some_and(|item| item.is_err())
                } else {
                    collect_ws_events(&mut client, 300_000, None, None, None, None)
                        .await
                        .is_err()
                }
            });
            peer.send(Message::Ping(vec![1; 125])).await.unwrap();
            let explicit_failure = tokio::time::timeout(Duration::from_secs(11), reader)
                .await
                .expect(
                    "Pong write timeout must terminate the attempt rather than resume a 300s read",
                )
                .unwrap();
            assert!(
                explicit_failure,
                "failed Pong write must be an explicit failure, not EOF compatibility"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn partial_request_write_is_bounded_and_never_retryable() {
        let (io, _unread_peer) = tokio::io::duplex(1);
        let mut websocket = WebSocketStream::from_raw_socket(io, Role::Client, None).await;
        let error = write_request(&mut websocket, "request".repeat(1024))
            .await
            .unwrap_err();
        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_AMBIGUOUS_WRITE_DETAIL)
        );
        assert!(!super::super::retryable_live_start_codex_error(&error));
    }

    #[tokio::test]
    async fn stale_delivery_cleanup_preserves_same_socket_republished_by_new_turn() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("same-socket-new-turn");
        let request = continuation_request();
        let first =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let entry = Arc::new(PoolEntry::new(raw_test_stream(None).await));
        assert!(pool_insert_for_turn(&first, entry.clone()));
        let (_, rx) = mpsc::channel(64);
        let (old, publisher) = CodexWebSocketEventStream::pending(rx);
        let old = old.track_attempt(Some(&first), true);
        publisher.publish(Some(entry.socket_id));
        let newer =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let checked_out = pool_take_for_turn(&newer).unwrap();
        assert!(pool_insert_for_turn(&newer, checked_out));
        drop(old);
        assert_eq!(pooled_socket_id_for_tests(&owner), Some(entry.socket_id));
        assert!(super::super::continuation::is_current_turn_for_owner(
            &newer
        ));
        invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn connect_cleanup_uses_one_idle_cap_lru_and_skips_leases() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let ws = ConnectionHandle::new(raw_test_stream(None).await);
        {
            let mut guard = WS_POOL.lock().unwrap();
            for index in 0..64 {
                guard.insert(
                    main_owner(&format!("entry-{index:02}")),
                    shared_pool_entry(&ws),
                );
            }
        }

        cleanup_pool_before_connect();
        assert_eq!(WS_POOL.lock().unwrap().len(), 64);

        WS_POOL
            .lock()
            .unwrap()
            .insert(main_owner("entry-64"), shared_pool_entry(&ws));
        WS_POOL
            .lock()
            .unwrap()
            .insert(main_owner("entry-65"), shared_pool_entry(&ws));
        WS_POOL
            .lock()
            .unwrap()
            .get(&main_owner("entry-00"))
            .unwrap()
            .touch();
        let leased = WS_POOL
            .lock()
            .unwrap()
            .get(&main_owner("entry-01"))
            .unwrap()
            .clone();

        cleanup_pool_before_connect();

        let guard = WS_POOL.lock().unwrap();
        assert_eq!(guard.len(), 65); // 64 idle entries plus one active lease.
        assert!(guard.contains_key(&main_owner("entry-00")));
        assert!(guard.contains_key(&main_owner("entry-01")));
        assert!(!guard.contains_key(&main_owner("entry-02")));
        assert!(guard.contains_key(&main_owner("entry-03")));
        assert!(guard.contains_key(&main_owner("entry-65")));
        drop(guard);
        drop(leased);
        clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn connect_cleanup_drops_final_socket_owners_after_unlocking_pool() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let dropped = Arc::new(AtomicBool::new(false));
        let pool_was_unlocked = Arc::new(AtomicBool::new(false));
        let shared_ws = ConnectionHandle::new(raw_test_stream(None).await);
        let probe_stream =
            raw_test_stream(Some((dropped.clone(), pool_was_unlocked.clone()))).await;
        {
            let mut guard = WS_POOL.lock().unwrap();
            guard.insert(
                main_owner("entry-00"),
                Arc::new(PoolEntry::new(probe_stream)),
            );
            for index in 1..=64 {
                guard.insert(
                    main_owner(&format!("entry-{index:02}")),
                    shared_pool_entry(&shared_ws),
                );
            }
        }

        cleanup_pool_before_connect();

        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("eviction must end the reader and drop its socket");
        assert!(pool_was_unlocked.load(Ordering::SeqCst));
        clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn pool_fake_clock_distinguishes_true_idle_from_absolute_age_and_leases() {
        let ws = ConnectionHandle::new(raw_test_stream(None).await);
        let policy = crate::config::CodexCachePolicy::default();
        let fresh_idle = shared_pool_entry(&ws);
        let expired_idle = shared_pool_entry(&ws);
        let active = shared_pool_entry(&ws);
        let now = tokio::time::Instant::now();
        let mut pool = HashMap::new();
        pool.insert(main_owner("fresh"), fresh_idle);
        pool.insert(main_owner("expired"), expired_idle);
        pool.insert(main_owner("active"), active.clone());
        *pool
            .get(&main_owner("fresh"))
            .unwrap()
            .idle_since
            .lock()
            .unwrap() = now + Duration::from_secs(299);
        assert!(maintain_pool_at(&mut pool, policy, now + Duration::from_secs(299)).is_empty());
        drop(maintain_pool_at(
            &mut pool,
            policy,
            now + Duration::from_secs(300),
        ));
        assert!(pool.contains_key(&main_owner("fresh")));
        assert!(!pool.contains_key(&main_owner("expired")));
        assert!(pool.contains_key(&main_owner("active")));
        *pool
            .get(&main_owner("fresh"))
            .unwrap()
            .idle_since
            .lock()
            .unwrap() = now + Duration::from_secs(1799);
        drop(maintain_pool_at(
            &mut pool,
            policy,
            now + Duration::from_secs(1800),
        ));
        assert!(
            !pool.contains_key(&main_owner("fresh")),
            "use never extends absolute age"
        );
        assert!(pool.contains_key(&main_owner("active")));
        drop(active);
        drop(maintain_pool_at(
            &mut pool,
            policy,
            now + Duration::from_secs(1800),
        ));
        assert!(pool.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn checkout_expires_idle_origin_and_insertion_refreshes_idle_but_not_age() {
        let _registry =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let request = continuation_request();
        let owner = main_owner("clock-checkout");
        let reservation =
            super::super::continuation::reserve_owner_request(Some(&owner), &request, false, true);
        let entry = Arc::new(PoolEntry::new(raw_test_stream(None).await));
        assert!(pool_insert_for_turn(&reservation, entry));
        tokio::time::advance(Duration::from_secs(299)).await;
        let reused = pool_take_for_turn(&reservation).expect("before true idle expiry");
        assert!(pool_insert_for_turn(&reservation, reused));
        tokio::time::advance(Duration::from_secs(299)).await;
        let reused = pool_take_for_turn(&reservation).expect("reinsertion must reset idle age");
        assert!(pool_insert_for_turn(&reservation, reused));
        tokio::time::advance(Duration::from_secs(300)).await;
        assert!(pool_take_for_turn(&reservation).is_none());

        let mut entry = Arc::new(PoolEntry::new(raw_test_stream(None).await));
        Arc::get_mut(&mut entry).unwrap().created_at =
            tokio::time::Instant::now() - Duration::from_secs(1800);
        assert!(
            !pool_insert_for_turn(&reservation, entry),
            "an expired leased origin must not be reinserted"
        );
        assert!(pooled_socket_id_for_tests(&owner).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn insertion_expires_other_idle_entries_and_enforces_cap_immediately() {
        let _pool = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let ws = ConnectionHandle::new(raw_test_stream(None).await);
        pool_insert(main_owner("expired"), shared_pool_entry(&ws));
        tokio::time::advance(Duration::from_secs(300)).await;
        pool_insert(main_owner("new"), shared_pool_entry(&ws));
        assert!(pooled_socket_id_for_tests(&main_owner("expired")).is_none());
        for index in 0..64 {
            pool_insert(
                main_owner(&format!("insert-{index}")),
                shared_pool_entry(&ws),
            );
        }
        let pool = WS_POOL.lock().unwrap();
        assert_eq!(pool.len(), 64);
        assert!(!pool.contains_key(&main_owner("new")));
        assert!(pool.contains_key(&main_owner("insert-63")));
    }

    #[tokio::test(start_paused = true)]
    async fn connect_gate_spaces_starts_without_serializing_handshakes() {
        let gate = Arc::new(WebSocketConnectGate::new(Duration::from_secs(1)));
        let preflight_started = Arc::new(tokio::sync::Notify::new());
        let release_first = Arc::new(tokio::sync::Notify::new());
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();

        let first_gate = gate.clone();
        let first_preflight = preflight_started.clone();
        let first_release = release_first.clone();
        let first_tx = started_tx.clone();
        let first = tokio::spawn(async move {
            first_gate
                .wait_to_start(async move {
                    first_preflight.notify_one();
                    tokio::time::sleep(Duration::from_secs(2)).await;
                })
                .await;
            first_tx.send(tokio::time::Instant::now()).unwrap();
            first_release.notified().await;
        });
        preflight_started.notified().await;

        let second_gate = gate.clone();
        let second = tokio::spawn(async move {
            second_gate.wait_to_start(async {}).await;
            started_tx.send(tokio::time::Instant::now()).unwrap();
        });
        let first_started = started_rx.recv().await.unwrap();
        let second_started = started_rx.recv().await.unwrap();

        assert!(second_started.duration_since(first_started) >= Duration::from_secs(1));
        assert!(!first.is_finished());
        second.await.unwrap();
        release_first.notify_one();
        first.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn connect_gate_spaces_after_non_403_failure_without_retrying() {
        let gate = WebSocketConnectGate::new(Duration::from_secs(1));
        let attempts = AtomicUsize::new(0);
        let starts = Mutex::new(Vec::new());

        let error = connect_with_policy(
            &gate,
            Duration::from_secs(10),
            Duration::from_secs(3),
            || async {},
            || {
                attempts.fetch_add(1, Ordering::SeqCst);
                starts.lock().unwrap().push(tokio::time::Instant::now());
                async {
                    Err::<(), ConnectAttemptError>(ConnectAttemptError::Origin(handshake_error(
                        http::StatusCode::BAD_GATEWAY.as_u16(),
                    )))
                }
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error.status, http::StatusCode::BAD_GATEWAY.as_u16());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);

        connect_with_policy(
            &gate,
            Duration::from_secs(10),
            Duration::from_secs(3),
            || async {},
            || {
                starts.lock().unwrap().push(tokio::time::Instant::now());
                async { Ok::<(), ConnectAttemptError>(()) }
            },
        )
        .await
        .unwrap();

        let starts = starts.lock().unwrap();
        assert!(starts[1].duration_since(starts[0]) >= Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn origin_403_waits_for_cooldown_retries_once_and_returns_second_error() {
        let gate = WebSocketConnectGate::new(Duration::from_secs(1));
        let attempts = AtomicUsize::new(0);
        let starts = Mutex::new(Vec::new());

        let error = connect_with_policy(
            &gate,
            Duration::from_secs(10),
            Duration::from_secs(3),
            || async {},
            || {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                starts.lock().unwrap().push(tokio::time::Instant::now());
                async move {
                    if attempt == 0 {
                        Err::<(), ConnectAttemptError>(origin_forbidden_error())
                    } else {
                        Err::<(), ConnectAttemptError>(ConnectAttemptError::Origin(CodexError {
                            status: http::StatusCode::FORBIDDEN.as_u16(),
                            message: "second forbidden".to_string(),
                            detail: Some("second-detail".to_string()),
                            retry_after: Some("7".to_string()),
                            usage_limit: None,
                            origin: CodexErrorOrigin::WebSocketHandshake,
                        }))
                    }
                }
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.status, http::StatusCode::FORBIDDEN.as_u16());
        assert_eq!(error.message, "second forbidden");
        assert_eq!(error.detail.as_deref(), Some("second-detail"));
        assert_eq!(error.retry_after.as_deref(), Some("7"));
        assert_eq!(error.origin, CodexErrorOrigin::WebSocketHandshake);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        let starts = starts.lock().unwrap();
        assert!(starts[1].duration_since(starts[0]) >= Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn proxy_connect_403_is_not_retried() {
        let gate = WebSocketConnectGate::new(Duration::ZERO);
        let attempts = AtomicUsize::new(0);

        let error = connect_with_policy(
            &gate,
            Duration::from_secs(10),
            Duration::from_secs(3),
            || async {},
            || {
                attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err::<(), ConnectAttemptError>(ConnectAttemptError::ProxyTunnel(
                        handshake_error(http::StatusCode::FORBIDDEN.as_u16()),
                    ))
                }
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.status, http::StatusCode::FORBIDDEN.as_u16());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn explicit_proxy_connect_403_has_private_tunnel_provenance() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let proxy = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while connect_response_header_end(&request).is_none() {
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
            }
            socket
                .write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
        });
        let proxy_url = format!("http://{proxy_addr}");
        let proxy_config = WebSocketProxyConfig::new(None, Some(&proxy_url), None, None);
        let client = test_websocket_client();

        let error = match connect_once(
            &client,
            &proxy_config,
            "wss://codex.invalid/backend-api/codex/responses",
            &HeaderMap::new(),
        )
        .await
        {
            Ok(_) => panic!("proxy CONNECT rejection should fail"),
            Err(error) => error,
        };

        match error {
            ConnectAttemptError::ProxyTunnel(error) => {
                assert_eq!(error.status, http::StatusCode::FORBIDDEN.as_u16());
                assert_eq!(
                    error.detail.as_deref(),
                    Some(WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL)
                );
                assert_eq!(error.origin, CodexErrorOrigin::WebSocketHandshake);
            }
            ConnectAttemptError::Origin(_) => panic!("proxy rejection was classified as origin"),
        }
        proxy.await.unwrap();
    }

    #[tokio::test]
    async fn origin_403_with_spoofed_proxy_detail_still_retries_once() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                while connect_response_header_end(&request).is_none() {
                    let read = socket.read(&mut buffer).await.unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&buffer[..read]);
                }
                let body = format!(
                    "{{\"error\":{{\"message\":\"{WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL}\"}}}}"
                );
                let response = format!(
                    "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let client = test_websocket_client();
        let proxy_config = WebSocketProxyConfig::direct();
        let url = format!("ws://{addr}/backend-api/codex/responses");
        let headers = HeaderMap::new();
        let gate = WebSocketConnectGate::new(Duration::ZERO);

        let error = match connect_with_policy(
            &gate,
            Duration::from_secs(1),
            Duration::ZERO,
            || async {},
            || connect_once(&client, &proxy_config, &url, &headers),
        )
        .await
        {
            Ok(_) => panic!("spoofed origin rejection should fail after one retry"),
            Err(error) => error,
        };

        assert_eq!(error.status, http::StatusCode::FORBIDDEN.as_u16());
        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL)
        );
        server.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_applies_to_each_network_attempt_not_waits_or_cooldown() {
        let gate = WebSocketConnectGate::new(Duration::from_secs(1));
        let attempts = AtomicUsize::new(0);
        let started_at = tokio::time::Instant::now();

        let error = connect_with_policy(
            &gate,
            Duration::from_secs(2),
            Duration::from_secs(3),
            || async {},
            || {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    if attempt == 0 {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        Err::<(), ConnectAttemptError>(origin_forbidden_error())
                    } else {
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        Ok(())
                    }
                }
            },
        )
        .await
        .unwrap_err();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(error.message, "WebSocket connect timeout after 2000ms");
        assert_eq!(started_at.elapsed(), Duration::from_secs(6));
    }

    #[test]
    fn websocket_tls_configuration_is_shared() {
        let first = websocket_tls_config();
        let second = websocket_tls_config();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn event_error_status_requires_error_event_and_checks_numeric_fallbacks() {
        assert_eq!(
            event_error_status(&serde_json::json!({
                "type": "response.failed",
                "status": "failed",
                "status_code": 401
            })),
            Some(401)
        );
        assert_eq!(
            event_error_status(&serde_json::json!({
                "type": "response.completed",
                "status_code": 401
            })),
            None
        );
        assert_eq!(
            event_error_status(&serde_json::json!({
                "type": "error",
                "error": {"status": 401}
            })),
            Some(401)
        );
    }

    #[test]
    fn websocket_url_conversion() {
        assert_eq!(
            to_websocket_url("https://example.test/codex").unwrap(),
            "wss://example.test/codex"
        );
        assert_eq!(
            to_websocket_url("http://example.test/codex").unwrap(),
            "ws://example.test/codex"
        );
        assert_eq!(
            to_websocket_url("wss://example.test/codex").unwrap(),
            "wss://example.test/codex"
        );
        assert!(to_websocket_url("ftp://example.test/codex").is_err());
    }

    #[test]
    fn websocket_upgrade_url_preserves_authority_path_and_query() {
        assert_eq!(
            to_http_upgrade_url("wss://chatgpt.com/backend-api/codex/responses?mode=live").unwrap(),
            "https://chatgpt.com/backend-api/codex/responses?mode=live"
        );
        assert_eq!(
            to_http_upgrade_url("ws://127.0.0.1:4141/backend-api/codex/responses").unwrap(),
            "http://127.0.0.1:4141/backend-api/codex/responses"
        );
        assert_eq!(
            to_http_upgrade_url("ws://[::1]:4141/path").unwrap(),
            "http://[::1]:4141/path"
        );
    }

    #[test]
    fn validates_tokenized_websocket_upgrade_headers() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let mut headers = HeaderMap::new();
        headers.insert(http::header::UPGRADE, "h2c, WebSocket".parse().unwrap());
        headers.insert(
            http::header::CONNECTION,
            "keep-alive, Upgrade".parse().unwrap(),
        );
        headers.insert(
            http::header::SEC_WEBSOCKET_ACCEPT,
            derive_accept_key(key.as_bytes()).parse().unwrap(),
        );
        headers.insert(
            http::header::SEC_WEBSOCKET_PROTOCOL,
            "responses".parse().unwrap(),
        );

        validate_websocket_upgrade(
            http::Version::HTTP_11,
            &headers,
            key,
            &["responses".to_string()],
        )
        .unwrap();
    }

    #[test]
    fn rejects_invalid_websocket_upgrade_headers() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let valid = || {
            let mut headers = HeaderMap::new();
            headers.insert(http::header::UPGRADE, "websocket".parse().unwrap());
            headers.insert(http::header::CONNECTION, "Upgrade".parse().unwrap());
            headers.insert(
                http::header::SEC_WEBSOCKET_ACCEPT,
                derive_accept_key(key.as_bytes()).parse().unwrap(),
            );
            headers
        };

        let mut missing_upgrade = valid();
        missing_upgrade.remove(http::header::UPGRADE);
        assert!(
            validate_websocket_upgrade(http::Version::HTTP_11, &missing_upgrade, key, &[]).is_err()
        );

        let mut missing_connection = valid();
        missing_connection.remove(http::header::CONNECTION);
        assert!(
            validate_websocket_upgrade(http::Version::HTTP_11, &missing_connection, key, &[])
                .is_err()
        );

        let mut wrong_accept = valid();
        wrong_accept.insert(http::header::SEC_WEBSOCKET_ACCEPT, "wrong".parse().unwrap());
        assert!(
            validate_websocket_upgrade(http::Version::HTTP_11, &wrong_accept, key, &[]).is_err()
        );

        let unsolicited_extension = {
            let mut headers = valid();
            headers.insert(
                http::header::SEC_WEBSOCKET_EXTENSIONS,
                "permessage-deflate".parse().unwrap(),
            );
            headers
        };
        assert!(
            validate_websocket_upgrade(http::Version::HTTP_11, &unsolicited_extension, key, &[],)
                .is_err()
        );

        assert!(validate_websocket_upgrade(http::Version::HTTP_10, &valid(), key, &[]).is_err());

        let mut unsolicited_protocol = valid();
        unsolicited_protocol.insert(
            http::header::SEC_WEBSOCKET_PROTOCOL,
            "unexpected".parse().unwrap(),
        );
        assert!(
            validate_websocket_upgrade(http::Version::HTTP_11, &unsolicited_protocol, key, &[])
                .is_err()
        );
    }

    #[test]
    fn websocket_headers_rewrite_beta() {
        let mut headers = http::HeaderMap::new();
        headers.insert("openai-beta", "responses=experimental".parse().unwrap());
        headers.insert("content-length", "10".parse().unwrap());
        headers.insert("authorization", "Bearer tok".parse().unwrap());
        headers.insert(
            http::header::PROXY_AUTHORIZATION,
            "Basic dXNlcjpwYXNz".parse().unwrap(),
        );
        let ws = codex_websocket_headers(&headers);
        assert_eq!(ws.get("openai-beta").unwrap(), WEBSOCKET_PROTOCOL_HEADER);
        assert!(!ws.contains_key("content-length"));
        assert!(!ws.contains_key(http::header::PROXY_AUTHORIZATION));
        assert_eq!(ws.get("authorization").unwrap(), "Bearer tok");
    }

    #[test]
    fn websocket_headers_strips_accept() {
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::ACCEPT, "text/event-stream".parse().unwrap());
        let ws = codex_websocket_headers(&headers);
        assert!(!ws.contains_key(http::header::ACCEPT.as_str()));
    }

    #[test]
    fn websocket_headers_adds_sec_key() {
        let headers = http::HeaderMap::new();
        let ws = codex_websocket_headers(&headers);
        assert!(ws.contains_key("sec-websocket-key"));
    }

    #[test]
    fn encode_sse_single_line() {
        let result = encode_sse(r#"{"type":"test","data":"hello"}"#);
        let expected = b"data: {\"type\":\"test\",\"data\":\"hello\"}\n\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn encode_sse_multi_line() {
        let result = encode_sse("line1\nline2");
        assert_eq!(
            String::from_utf8(result).unwrap(),
            "data: line1\ndata: line2\n\n"
        );
    }

    #[test]
    fn is_terminal_event_detection() {
        let completed = serde_json::json!({"type": "response.completed"});
        assert!(is_terminal_event(&completed));

        let done = serde_json::json!({"type": "response.done"});
        assert!(is_terminal_event(&done));

        let delta = serde_json::json!({"type": "response.output_text.delta"});
        assert!(!is_terminal_event(&delta));

        let error = serde_json::json!({"type": "error", "error": {"message": "fail"}});
        assert!(is_terminal_event(&error));

        let response_error = serde_json::json!({
            "type": "response.error",
            "response": {"error": {"message": "fail"}}
        });
        assert!(is_terminal_event(&response_error));
    }

    #[test]
    fn is_response_event_detection() {
        let rate_limits = serde_json::json!({"type": "codex.rate_limits"});
        assert!(!is_response_event(&rate_limits));

        let output = serde_json::json!({"type": "response.output_text.delta"});
        assert!(is_response_event(&output));

        let error = serde_json::json!({"type": "error", "error": {"message": "fail"}});
        assert!(is_response_event(&error));
    }

    #[test]
    fn is_previous_response_missing_detection() {
        let by_code = serde_json::json!({
            "type": "error",
            "error": {"code": "previous_response_not_found", "message": "not found"}
        });
        assert!(is_previous_response_missing(&by_code));

        let by_msg = serde_json::json!({
            "type": "error",
            "error": {"message": "The previous response was not found"}
        });
        assert!(is_previous_response_missing(&by_msg));

        let nested = serde_json::json!({
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "previous_response_not_found",
                    "message": "Previous response not found"
                }
            }
        });
        assert!(is_previous_response_missing(&nested));

        let unrelated = serde_json::json!({"type": "error", "error": {"message": "rate limited"}});
        assert!(!is_previous_response_missing(&unrelated));
    }

    #[test]
    fn websocket_metadata_does_not_serialize_typed_owner() {
        let writer = crate::logging::writer::pause_for_fixture();
        let temp = tempfile::tempdir().unwrap();
        let traffic = crate::traffic::test_capture(temp.path().join("traffic"));
        let owner = agent_owner("session-secret", "agent-secret");
        let reservation = test_continuation(Some(owner), None, None, None);

        write_websocket_metadata_capture(
            &traffic,
            "wss://example.invalid/responses",
            Some(&reservation),
            false,
        );

        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let artifact = std::fs::read_dir(traffic.root())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let captured = std::fs::read_to_string(artifact).unwrap();
        assert!(captured.contains("poolingEnabled"));
        assert!(!captured.contains("poolKey"));
        assert!(!captured.contains("session-secret"));
        assert!(!captured.contains("agent-secret"));
    }

    #[tokio::test]
    async fn pool_checkout_is_exclusive_and_removal_is_identity_safe() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let first = Arc::new(PoolEntry::new(create_dummy_stream_async().await));
        let owner = main_owner("exclusive");
        pool_insert(owner.clone(), first.clone());
        assert!(Arc::ptr_eq(
            &WS_POOL.lock().unwrap().remove(&owner).unwrap(),
            &first
        ));
        assert!(WS_POOL.lock().unwrap().remove(&owner).is_none());

        let replacement = Arc::new(PoolEntry::new(create_dummy_stream_async().await));
        pool_insert(owner.clone(), replacement.clone());
        pool_remove_entry(&owner, &first);
        let reservation = test_continuation(Some(owner.clone()), None, None, None);
        invalidate_codex_websocket_pool_socket(&reservation, Some(first.socket_id));
        assert!(Arc::ptr_eq(
            WS_POOL.lock().unwrap().get(&owner).unwrap(),
            &replacement
        ));
        clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn pool_entries_receive_monotonic_nonzero_socket_ids() {
        let first = PoolEntry::new(raw_test_stream(None).await);
        let second = PoolEntry::new(raw_test_stream(None).await);

        assert_ne!(first.socket_id, 0);
        assert!(second.socket_id > first.socket_id);
    }

    #[tokio::test]
    async fn continuation_rejects_and_preserves_same_owner_replacement() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = agent_owner("replacement-session", "replacement-agent");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let reserved = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let replacement = Arc::new(PoolEntry::new(raw_test_stream(None).await));
        pool_insert(owner.clone(), replacement.clone());
        let continuation = test_continuation(
            Some(owner.clone()),
            reserved.turn_id(),
            Some("resp_origin"),
            Some(replacement.socket_id.checked_add(1).unwrap()),
        );

        let error = match prepare_codex_websocket(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            "ws://127.0.0.1:9/responses",
            &HeaderMap::new(),
            None,
            None,
            Some(&continuation),
            50,
            50,
        )
        .await
        {
            Ok(_) => panic!("replacement socket must not satisfy continuation provenance"),
            Err(error) => error,
        };

        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
        );
        assert!(!error.message.contains("replacement-session"));
        assert!(!error.message.contains("replacement-agent"));
        assert!(Arc::ptr_eq(
            WS_POOL.lock().unwrap().get(&owner).unwrap(),
            &replacement
        ));

        invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&reserved);
    }

    #[tokio::test]
    async fn dead_exact_origin_removes_only_that_arc_and_preserves_replacement() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("dead-origin-session");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let reserved = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let (io, peer) = tokio::io::duplex(1024);
        drop(peer);
        let io: BoxedWebSocketIo = Box::new(io);
        let exact = Arc::new(PoolEntry::new(
            WebSocketStream::from_raw_socket(io, Role::Client, None).await,
        ));
        while !exact.connection.is_closed() {
            tokio::task::yield_now().await;
        }
        let replacement = Arc::new(PoolEntry::new(raw_test_stream(None).await));
        pool_insert(owner.clone(), exact.clone());
        let continuation = test_continuation(
            Some(owner.clone()),
            reserved.turn_id(),
            Some("resp_exact"),
            Some(exact.socket_id),
        );

        let replacement_owner = owner.clone();
        let replacement_for_task = replacement.clone();
        let mut insert_replacement = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(1), async move {
                loop {
                    if !WS_POOL.lock().unwrap().contains_key(&replacement_owner) {
                        pool_insert(replacement_owner, replacement_for_task);
                        return;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
        });
        let error = match prepare_codex_websocket(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            "ws://127.0.0.1:9/responses",
            &HeaderMap::new(),
            None,
            None,
            Some(&continuation),
            50,
            50,
        )
        .await
        {
            Ok(_) => panic!("dead continuation origin must be rejected before request send"),
            Err(error) => error,
        };
        match tokio::time::timeout(Duration::from_secs(2), &mut insert_replacement).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(_))) => panic!(
                "replacement polling timed out; currently pooled socket ID: {:?}",
                pooled_socket_id_for_tests(&owner)
            ),
            Ok(Err(error)) => panic!(
                "replacement polling task failed ({error}); currently pooled socket ID: {:?}",
                pooled_socket_id_for_tests(&owner)
            ),
            Err(_) => {
                insert_replacement.abort();
                let abort_result = insert_replacement.await;
                panic!(
                    "replacement polling join timed out ({abort_result:?}); currently pooled socket ID: {:?}",
                    pooled_socket_id_for_tests(&owner)
                );
            }
        }

        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
        );
        assert!(Arc::ptr_eq(
            WS_POOL.lock().unwrap().get(&owner).unwrap(),
            &replacement
        ));
        assert!(!Arc::ptr_eq(
            WS_POOL.lock().unwrap().get(&owner).unwrap(),
            &exact
        ));

        invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&reserved);
    }

    #[tokio::test]
    async fn completed_terminal_is_published_after_origin_returns_to_pool() {
        assert_terminal_pool_contract("response.completed", "completed", true).await;
    }

    #[tokio::test]
    async fn done_success_reuses_origin_socket() {
        assert_terminal_pool_contract("response.done", "completed", true).await;
    }

    #[tokio::test]
    async fn completed_failed_status_never_reuses_socket() {
        assert_terminal_pool_contract("response.completed", "failed", false).await;
    }

    async fn assert_terminal_pool_contract(
        event_type: &'static str,
        status: &'static str,
        reusable: bool,
    ) {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = agent_owner("terminal-order-session", "terminal-order-agent");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (release_response_tx, release_response_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            while let Some(Ok(message)) = websocket.next().await {
                match message {
                    Message::Ping(payload) => {
                        websocket.send(Message::Pong(payload)).await.unwrap();
                    }
                    Message::Text(_) => {
                        release_response_rx.await.unwrap();
                        websocket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": event_type,
                                    "response": {"id": "resp_terminal_order", "status": status}
                                })
                                .to_string(),
                            ))
                            .await
                            .unwrap();
                        return;
                    }
                    _ => {}
                }
            }
        });
        let context = RequestContext {
            req_id: "terminal-order-request".to_string(),
            session_id: Some("header-session-is-not-pool-owner".to_string()),
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        };
        let mut events = codex_websocket_event_stream(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &format!("http://{addr}/responses"),
            &HeaderMap::new(),
            &serde_json::json!({"type":"response.create","input":[]}),
            &context,
            None,
            1_000,
            1_000,
            Some(&continuation),
        )
        .await
        .unwrap();
        assert_eq!(events.socket_id(), None);
        release_response_tx.send(()).unwrap();
        let terminal = events.recv().await.unwrap().unwrap();

        assert_eq!(
            terminal.get("type").and_then(serde_json::Value::as_str),
            Some(event_type)
        );
        let pooled = WS_POOL.lock().unwrap().get(&owner).cloned();
        if reusable {
            let pooled =
                pooled.expect("successful origin must be reusable before terminal publication");
            assert_eq!(events.socket_id(), Some(pooled.socket_id));
        } else {
            assert!(
                pooled.is_none(),
                "failed terminal must never publish reusable socket"
            );
            assert_eq!(events.socket_id(), None);
        }
        server.await.unwrap();

        invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&continuation);
    }

    #[tokio::test]
    async fn dropped_receiver_removes_completed_origin_after_terminal_publication_fails() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("dropped-terminal-receiver-session");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (release_response_tx, release_response_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            while let Some(Ok(message)) = websocket.next().await {
                match message {
                    Message::Ping(payload) => {
                        websocket.send(Message::Pong(payload)).await.unwrap();
                    }
                    Message::Text(_) => {
                        request_seen_tx.send(()).unwrap();
                        release_response_rx.await.unwrap();
                        websocket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "response.completed",
                                    "response": {"id": "resp_dropped_receiver"}
                                })
                                .to_string(),
                            ))
                            .await
                            .unwrap();
                        return;
                    }
                    _ => {}
                }
            }
        });
        let headers = HeaderMap::new();
        let body = serde_json::json!({"type":"response.create","input":[]});
        let ready = prepare_codex_websocket(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &format!("http://{addr}/responses"),
            &headers,
            None,
            None,
            Some(&continuation),
            1_000,
            1_000,
        )
        .await
        .unwrap();
        let exact = ready.entry.clone();
        let events = start_codex_websocket_events(
            ready,
            &body,
            serde_json::to_string(&body).unwrap(),
            &headers,
            Some(&continuation),
        );

        request_seen_rx.await.unwrap();
        drop(events);
        release_response_tx.send(()).unwrap();
        server.await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while Arc::strong_count(&exact) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("failed terminal publication must release the exact completed pool entry");

        assert!(
            WS_POOL
                .lock()
                .unwrap()
                .get(&owner)
                .is_none_or(|pooled| !Arc::ptr_eq(pooled, &exact)),
            "the exact completed socket must not remain pooled"
        );
        super::super::continuation::abort_continuation_for_owner(&continuation);
    }

    #[tokio::test]
    async fn dropped_receiver_after_terminal_enqueue_disposes_exact_origin() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("dropped-terminal-receiver-session");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (release_response_tx, release_response_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            while let Some(Ok(message)) = websocket.next().await {
                match message {
                    Message::Ping(payload) => {
                        websocket.send(Message::Pong(payload)).await.unwrap();
                    }
                    Message::Text(_) => {
                        request_seen_tx.send(()).unwrap();
                        release_response_rx.await.unwrap();
                        websocket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "response.completed",
                                    "response": {"id": "resp_dropped_receiver"}
                                })
                                .to_string(),
                            ))
                            .await
                            .unwrap();
                        return;
                    }
                    _ => {}
                }
            }
        });
        let headers = HeaderMap::new();
        let body = serde_json::json!({"type":"response.create","input":[]});
        let ready = prepare_codex_websocket(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &format!("http://{addr}/responses"),
            &headers,
            None,
            None,
            Some(&continuation),
            1_000,
            1_000,
        )
        .await
        .unwrap();
        let exact = ready.entry.clone();
        let mut events = start_codex_websocket_events(
            ready,
            &body,
            serde_json::to_string(&body).unwrap(),
            &headers,
            Some(&continuation),
        );

        request_seen_rx.await.unwrap();
        release_response_tx.send(()).unwrap();
        assert!(events.recv().await.unwrap().is_ok());
        assert!(pooled_socket_id_for_tests(&owner).is_some());
        drop(events);
        server.await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while Arc::strong_count(&exact) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("failed terminal publication must release the exact completed pool entry");

        assert!(
            WS_POOL
                .lock()
                .unwrap()
                .get(&owner)
                .is_none_or(|pooled| !Arc::ptr_eq(pooled, &exact)),
            "the exact completed socket must not remain pooled"
        );
        super::super::continuation::abort_continuation_for_owner(&continuation);
    }

    #[tokio::test]
    async fn terminal_capacity_is_reserved_before_pool_publication() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("dropped-terminal-receiver-session");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        invalidate_codex_websocket_pool_owner(&owner);
        let request = continuation_request();
        let continuation = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (release_response_tx, release_response_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            while let Some(Ok(message)) = websocket.next().await {
                match message {
                    Message::Ping(payload) => {
                        websocket.send(Message::Pong(payload)).await.unwrap();
                    }
                    Message::Text(_) => {
                        request_seen_tx.send(()).unwrap();
                        release_response_rx.await.unwrap();
                        for _ in 0..64 {
                            websocket.send(Message::Text(serde_json::json!({"type":"response.output_text.delta","delta":"x"}).to_string())).await.unwrap();
                        }
                        websocket
                            .send(Message::Text(
                                serde_json::json!({
                                    "type": "response.completed",
                                    "response": {"id": "resp_dropped_receiver"}
                                })
                                .to_string(),
                            ))
                            .await
                            .unwrap();
                        return;
                    }
                    _ => {}
                }
            }
        });
        let headers = HeaderMap::new();
        let body = serde_json::json!({"type":"response.create","input":[]});
        let ready = prepare_codex_websocket(
            &test_websocket_client(),
            &WebSocketProxyConfig::direct(),
            &format!("http://{addr}/responses"),
            &headers,
            None,
            None,
            Some(&continuation),
            1_000,
            1_000,
        )
        .await
        .unwrap();
        let exact = ready.entry.clone();
        let events = start_codex_websocket_events(
            ready,
            &body,
            serde_json::to_string(&body).unwrap(),
            &headers,
            Some(&continuation),
        );

        request_seen_rx.await.unwrap();
        release_response_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while events.receiver.len() < 64 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(
            pooled_socket_id_for_tests(&owner).is_none(),
            "full delivery queue must not publish an idle socket"
        );
        drop(events);
        server.await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while Arc::strong_count(&exact) > 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(pooled_socket_id_for_tests(&owner).is_none());
        invalidate_codex_websocket_pool_owner(&owner);
        super::super::continuation::abort_continuation_for_owner(&continuation);
    }

    #[tokio::test]
    async fn pool_invalidation() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let first_stream = create_dummy_stream_async().await;
        let second_stream = create_dummy_stream_async().await;
        let first_owner = agent_owner("test-session", "first-agent");
        let sibling_owner = agent_owner("test-session", "sibling-agent");
        {
            let mut guard = WS_POOL.lock().unwrap();
            guard.insert(first_owner.clone(), Arc::new(PoolEntry::new(first_stream)));
            guard.insert(
                sibling_owner.clone(),
                Arc::new(PoolEntry::new(second_stream)),
            );
        }
        assert!(WS_POOL.lock().unwrap().contains_key(&first_owner));
        assert!(WS_POOL.lock().unwrap().contains_key(&sibling_owner));

        invalidate_codex_websocket_pool_owner(&first_owner);
        assert!(!WS_POOL.lock().unwrap().contains_key(&first_owner));
        assert!(WS_POOL.lock().unwrap().contains_key(&sibling_owner));
        clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn session_key_invalidation_removes_main_and_agents_only_for_exact_session() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let session_id = "session-key";
        let main = main_owner(session_id);
        let first_agent = agent_owner(session_id, "first-agent");
        let second_agent = agent_owner(session_id, "second-agent");
        let other_session = main_owner("session-key-other");
        for owner in [
            main.clone(),
            first_agent.clone(),
            second_agent.clone(),
            other_session.clone(),
        ] {
            pool_insert(
                owner,
                Arc::new(PoolEntry::new(create_dummy_stream_async().await)),
            );
        }

        invalidate_codex_websocket_pool_key(session_id);

        let pool = WS_POOL.lock().unwrap();
        assert!(!pool.contains_key(&main));
        assert!(!pool.contains_key(&first_agent));
        assert!(!pool.contains_key(&second_agent));
        assert!(pool.contains_key(&other_session));
        drop(pool);
        clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn missing_owner_or_turn_cannot_mutate_pool_state() {
        let _registry_guard =
            super::super::continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        let owner = main_owner("missing-owner-turn-pool");
        super::super::continuation::clear_continuation_for_owner(Some(&owner));
        clear_codex_websocket_pool_for_tests();
        let request = continuation_request();
        let current = super::super::continuation::continuation_candidate_for_owner(
            Some(&owner),
            &request,
            true,
        );
        let pooled = Arc::new(PoolEntry::new(create_dummy_stream_async().await));
        pool_insert(owner.clone(), pooled.clone());
        let missing_owner = test_continuation(None, current.turn_id(), None, None);
        let missing_turn = test_continuation(Some(owner.clone()), None, None, None);

        assert!(pool_take_for_turn(&missing_owner).is_none());
        assert!(pool_take_for_turn(&missing_turn).is_none());
        assert!(!pool_insert_for_turn(
            &missing_owner,
            Arc::new(PoolEntry::new(create_dummy_stream_async().await)),
        ));
        assert!(!pool_insert_for_turn(
            &missing_turn,
            Arc::new(PoolEntry::new(create_dummy_stream_async().await)),
        ));
        invalidate_codex_websocket_pool_turn_for_owner(&owner, None);
        invalidate_codex_websocket_pool_socket(&missing_owner, Some(pooled.socket_id));

        assert!(Arc::ptr_eq(
            WS_POOL.lock().unwrap().get(&owner).unwrap(),
            &pooled
        ));
        clear_codex_websocket_pool_for_tests();
        super::super::continuation::abort_continuation_for_owner(&current);
    }

    #[tokio::test]
    async fn websocket_connect_401_is_pre_request_handshake_error() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let _ = socket.read(&mut buf).await;
            socket
                .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 13\r\n\r\npolicy denied")
                .await
                .unwrap();
        });

        let client = test_websocket_client();
        let err = match connect_with_timeout(
            &client,
            &WebSocketProxyConfig::direct(),
            &format!("ws://{addr}/backend-api/codex/responses"),
            &HeaderMap::new(),
            1_000,
        )
        .await
        {
            Ok(_) => panic!("expected unauthorized websocket handshake to fail"),
            Err(err) => err,
        };

        assert_eq!(err.status, 401);
        assert_eq!(err.detail.as_deref(), Some(GENERIC_HANDSHAKE_ERROR_DETAIL));
        assert_eq!(err.origin, CodexErrorOrigin::WebSocketHandshake);
    }

    #[tokio::test]
    async fn websocket_connect_502_preserves_retry_metadata() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let _ = socket.read(&mut buf).await;
            socket
                .write_all(
                    b"HTTP/1.1 502 Bad Gateway\r\nRetry-After: 3\r\nContent-Length: 0\r\n\r\n",
                )
                .await
                .unwrap();
        });

        let client = test_websocket_client();
        let err = match connect_with_timeout(
            &client,
            &WebSocketProxyConfig::direct(),
            &format!("ws://{addr}/backend-api/codex/responses"),
            &HeaderMap::new(),
            1_000,
        )
        .await
        {
            Ok(_) => panic!("expected websocket handshake to fail"),
            Err(err) => err,
        };

        assert_eq!(err.status, 502);
        assert_eq!(err.detail.as_deref(), Some(GENERIC_HANDSHAKE_ERROR_DETAIL));
        assert_eq!(err.retry_after.as_deref(), Some("3"));
        assert_eq!(err.origin, CodexErrorOrigin::WebSocketHandshake);
    }

    #[tokio::test]
    async fn websocket_connects_through_explicit_http_proxy() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let (captured_tx, captured_rx) = tokio::sync::oneshot::channel();
        let proxy = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
            }
            let request_text = String::from_utf8(request).unwrap();
            let key = request_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("sec-websocket-key")
                        .then(|| value.trim().to_string())
                })
                .unwrap();
            let _ = captured_tx.send(request_text);
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: {}\r\n\r\n",
                derive_accept_key(key.as_bytes())
            );
            stream.write_all(response.as_bytes()).await.unwrap();

            let mut websocket = WebSocketStream::from_raw_socket(stream, Role::Server, None).await;
            assert_eq!(
                websocket.next().await.unwrap().unwrap(),
                Message::Text("hello".to_string())
            );
            websocket
                .send(Message::Text("proxy-ok".to_string()))
                .await
                .unwrap();
        });

        let client = reqwest::Client::builder()
            .http1_only()
            .redirect(reqwest::redirect::Policy::none())
            .proxy(
                reqwest::Proxy::http(format!("http://proxy-user:proxy-pass@{proxy_addr}")).unwrap(),
            )
            .build()
            .unwrap();
        let mut websocket = connect_with_timeout(
            &client,
            &WebSocketProxyConfig::direct(),
            "ws://codex.invalid/backend-api/codex/responses",
            &HeaderMap::new(),
            2_000,
        )
        .await
        .unwrap();
        websocket
            .send(Message::Text("hello".to_string()))
            .await
            .unwrap();
        assert_eq!(
            websocket.next().await.unwrap().unwrap(),
            Message::Text("proxy-ok".to_string())
        );

        let captured = captured_rx.await.unwrap();
        assert!(
            captured.starts_with("GET http://codex.invalid/backend-api/codex/responses HTTP/1.1")
        );
        assert!(
            captured
                .to_ascii_lowercase()
                .contains("proxy-authorization: basic ")
        );
        proxy.await.unwrap();
    }

    #[tokio::test]
    async fn connect_tunnel_accepts_fragmented_non_200_success() {
        let (client, mut proxy) = tokio::io::duplex(4096);
        let proxy_task = tokio::spawn(async move {
            let mut request = Vec::new();
            let mut buffer = [0_u8; 256];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = proxy.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
            }
            assert!(request.starts_with(b"CONNECT codex.invalid:4443 HTTP/1.1\r\n"));
            proxy.write_all(b"HTTP/1.").await.unwrap();
            tokio::task::yield_now().await;
            proxy
                .write_all(b"1 204 No Content\r\nX-Proxy: ok\r\n\r\nprefixed")
                .await
                .unwrap();
        });

        let stream: BoxedWebSocketIo = Box::new(client);
        let mut stream = establish_connect_tunnel(stream, "codex.invalid:4443", None)
            .await
            .unwrap();
        let mut prefix = [0_u8; 8];
        stream.read_exact(&mut prefix).await.unwrap();
        assert_eq!(&prefix, b"prefixed");
        proxy_task.await.unwrap();
    }

    #[tokio::test]
    async fn connect_tunnel_classifies_fragmented_proxy_authentication() {
        let (client, mut proxy) = tokio::io::duplex(4096);
        tokio::spawn(async move {
            let mut request = [0_u8; 512];
            let _ = proxy.read(&mut request).await.unwrap();
            proxy.write_all(b"HTTP/1.1 4").await.unwrap();
            tokio::task::yield_now().await;
            proxy
                .write_all(b"07 Proxy Authentication Required\r\n\r\n")
                .await
                .unwrap();
        });

        let stream: BoxedWebSocketIo = Box::new(client);
        let error = match establish_connect_tunnel(stream, "codex.invalid:4443", None).await {
            Ok(_) => panic!("proxy authentication should be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error.status,
            http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16()
        );
        assert_eq!(error.message, "WebSocket proxy authentication failed");
    }

    #[test]
    fn websocket_proxy_routing_honors_no_proxy_and_leaves_socks_to_reqwest() {
        let http_proxy = "http://proxy.example:8080";
        let config = WebSocketProxyConfig::new(None, Some(http_proxy), None, None);
        assert!(config.uses_proxy_for("wss://codex.invalid/responses"));
        assert!(
            config
                .http_connect_route("wss://codex.invalid/responses")
                .unwrap()
                .is_some()
        );

        let bypass = WebSocketProxyConfig::new(None, Some(http_proxy), None, Some("codex.invalid"));
        assert!(!bypass.uses_proxy_for("wss://codex.invalid/responses"));
        assert!(
            bypass
                .http_connect_route("wss://codex.invalid/responses")
                .unwrap()
                .is_none()
        );

        let socks =
            WebSocketProxyConfig::new(None, Some("socks5h://proxy.example:1080"), None, None);
        assert!(socks.uses_proxy_for("wss://codex.invalid/responses"));
        assert!(
            socks
                .http_connect_route("wss://codex.invalid/responses")
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn websocket_wss_uses_http_connect_without_leaking_proxy_credentials() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let (captured_tx, captured_rx) = tokio::sync::oneshot::channel();
        let proxy = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
            }
            let _ = captured_tx.send(String::from_utf8(request).unwrap());
            stream
                .write_all(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n",
                )
                .await
                .unwrap();
        });

        let proxy_url = format!("http://secret-user:secret-pass@{proxy_addr}");
        let client = reqwest::Client::builder()
            .http1_only()
            .redirect(reqwest::redirect::Policy::none())
            .proxy(reqwest::Proxy::https(&proxy_url).unwrap())
            .build()
            .unwrap();
        let proxy_config = WebSocketProxyConfig::new(None, Some(&proxy_url), None, None);
        let error = match connect_with_timeout(
            &client,
            &proxy_config,
            "wss://codex.invalid:4443/backend-api/codex/responses",
            &HeaderMap::new(),
            2_000,
        )
        .await
        {
            Ok(_) => panic!("proxy rejection should fail the WebSocket connection"),
            Err(error) => error,
        };

        let captured = captured_rx.await.unwrap();
        assert!(captured.starts_with("CONNECT codex.invalid:4443 HTTP/1.1"));
        assert!(
            captured
                .to_ascii_lowercase()
                .contains("proxy-authorization: basic ")
        );
        assert!(!error.message.contains("secret-user"));
        assert!(!error.message.contains("secret-pass"));
        assert!(
            !error
                .detail
                .as_deref()
                .unwrap_or_default()
                .contains("secret")
        );
        proxy.await.unwrap();
    }

    #[tokio::test]
    async fn binary_frame_invalidates_pool_owner() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let pooled_stream = create_dummy_stream_async().await;
        let owner = agent_owner("binary-session", "binary-agent");
        {
            let mut guard = WS_POOL.lock().unwrap();
            guard.insert(owner.clone(), Arc::new(PoolEntry::new(pooled_stream)));
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            ws.send(Message::Binary(vec![1, 2, 3])).await.unwrap();
        });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
            .await
            .unwrap();
        let err = match collect_ws_events(&mut ws, 1_000, Some(&owner), None, None, None).await {
            Ok(_) => panic!("expected binary frame to fail"),
            Err(err) => err,
        };

        assert!(err.message.contains("binary frames"));
        assert!(!WS_POOL.lock().unwrap().contains_key(&owner));
    }

    #[tokio::test]
    async fn response_start_timeout_ignores_rate_limits_and_pings() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let pooled_stream = create_dummy_stream_async().await;
        let owner = main_owner("start-timeout-session");
        {
            let mut guard = WS_POOL.lock().unwrap();
            guard.insert(owner.clone(), Arc::new(PoolEntry::new(pooled_stream)));
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            ws.send(Message::Text(
                r#"{"type":"codex.rate_limits","rate_limits":{"allowed":true}}"#.into(),
            ))
            .await
            .unwrap();
            loop {
                if ws.send(Message::Ping(Vec::new())).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
            .await
            .unwrap();
        let err = match collect_ws_events(&mut ws, 50, Some(&owner), None, None, None).await {
            Ok(_) => panic!("expected response start timeout"),
            Err(err) => err,
        };

        assert_eq!(
            err.detail.as_deref(),
            Some(WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL)
        );
        assert!(!WS_POOL.lock().unwrap().contains_key(&owner));
    }

    #[tokio::test]
    async fn response_idle_timeout_ignores_pings_after_response_event() {
        let _pool_test_guard = lock_codex_websocket_pool_for_tests().await;
        clear_codex_websocket_pool_for_tests();
        let pooled_stream = create_dummy_stream_async().await;
        let owner = main_owner("response-idle-session");
        {
            let mut guard = WS_POOL.lock().unwrap();
            guard.insert(owner.clone(), Arc::new(PoolEntry::new(pooled_stream)));
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            ws.send(Message::Text(
                r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}"#
                    .into(),
            ))
            .await
            .unwrap();
            loop {
                if ws.send(Message::Ping(Vec::new())).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
            .await
            .unwrap();
        let err = match collect_ws_events(&mut ws, 50, Some(&owner), None, None, None).await {
            Ok(_) => panic!("expected response idle timeout"),
            Err(err) => err,
        };

        assert!(err.message.contains("idle timeout"));
        assert_eq!(err.detail, None);
        assert!(!WS_POOL.lock().unwrap().contains_key(&owner));
    }

    #[tokio::test(start_paused = true)]
    async fn silent_response_wait_sends_keepalive_ping() {
        let (client_io, server_io) = tokio::io::duplex(1024);
        let mut client = WebSocketStream::from_raw_socket(client_io, Role::Client, None).await;
        let mut server = WebSocketStream::from_raw_socket(server_io, Role::Server, None).await;
        let server_task = tokio::spawn(async move {
            let frame = server.next().await.unwrap().unwrap();
            assert!(matches!(frame, Message::Ping(_)));
            server
                .send(Message::Text(r#"{"type":"response.created"}"#.into()))
                .await
                .unwrap();
        });

        let frame = read_ws_frame_with_keepalive(
            &mut client,
            Duration::from_secs(60),
            WEBSOCKET_KEEPALIVE_INTERVAL,
        )
        .await;

        assert!(matches!(
            frame,
            WebSocketRead::Frame(Some(Ok(Message::Text(text))))
                if text.contains("response.created")
        ));
        server_task.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn failed_keepalive_write_returns_stable_error() {
        let mut client = WebSocketStream::from_raw_socket(FailingWriteIo, Role::Client, None).await;

        let error = match collect_ws_events_with_keepalive_interval(
            &mut client,
            60_000,
            None,
            None,
            None,
            WEBSOCKET_KEEPALIVE_INTERVAL,
            None,
        )
        .await
        {
            Ok(_) => panic!("expected keepalive failure"),
            Err(error) => error,
        };

        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_KEEPALIVE_FAILURE_DETAIL)
        );
        assert!(error.message.contains("keepalive error"));
    }

    #[tokio::test]
    async fn keepalive_pongs_do_not_extend_response_start_timeout() {
        let (client_io, server_io) = tokio::io::duplex(1024);
        let mut client = WebSocketStream::from_raw_socket(client_io, Role::Client, None).await;
        let mut server = WebSocketStream::from_raw_socket(server_io, Role::Server, None).await;
        let ping_count = Arc::new(AtomicUsize::new(0));
        let server_ping_count = ping_count.clone();
        let server_task = tokio::spawn(async move {
            while let Some(Ok(frame)) = server.next().await {
                if let Message::Ping(payload) = frame {
                    server_ping_count.fetch_add(1, Ordering::SeqCst);
                    if server.send(Message::Pong(payload)).await.is_err() {
                        break;
                    }
                }
            }
        });

        let error = match collect_ws_events_with_keepalive_interval(
            &mut client,
            50,
            None,
            None,
            None,
            Duration::from_millis(10),
            None,
        )
        .await
        {
            Ok(_) => panic!("expected response start timeout"),
            Err(error) => error,
        };

        assert_eq!(
            error.detail.as_deref(),
            Some(WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL)
        );
        assert!(ping_count.load(Ordering::SeqCst) >= 2);
        server_task.abort();
    }

    async fn create_dummy_stream_async() -> CodexWebSocketStream {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let _ = tokio_tungstenite::accept_async(socket).await;
            futures_util::future::pending::<()>().await;
        });
        let url = format!("ws://{addr}/");
        let client = test_websocket_client();
        connect_with_timeout(
            &client,
            &WebSocketProxyConfig::direct(),
            &url,
            &HeaderMap::new(),
            1_000,
        )
        .await
        .unwrap()
    }

    #[test]
    fn handshake_details_are_structured_sanitized_and_bounded() {
        let message = format!("safe\n{}", "x".repeat(MAX_HANDSHAKE_ERROR_DETAIL_BYTES * 2));
        let body = serde_json::to_vec(&serde_json::json!({
            "error": { "message": message }
        }))
        .unwrap();
        let detail = handshake_error_detail(Some(&body));
        assert!(!detail.contains('\n'));
        assert!(detail.len() <= MAX_HANDSHAKE_ERROR_DETAIL_BYTES);
        assert!(detail.starts_with("safe"));
    }

    #[test]
    fn handshake_details_reject_unstructured_and_binary_bodies() {
        for body in [
            b"<html>denied</html>".to_vec(),
            vec![0xff, 0xfe],
            b"{".to_vec(),
        ] {
            assert_eq!(
                handshake_error_detail(Some(&body)),
                GENERIC_HANDSHAKE_ERROR_DETAIL
            );
        }
    }
}
