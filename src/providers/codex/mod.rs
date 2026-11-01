pub mod auth;
pub mod chat_completions;
pub mod client;
pub mod compaction;
pub mod continuation;
pub mod count_tokens;
pub(crate) mod events;
pub mod images;
pub mod native;
pub mod performance;
pub mod rate_limits;
pub mod request_summary;
pub mod search;
mod timing;
pub mod transcription;
pub mod translate;
pub mod websocket;

use async_trait::async_trait;
use axum::Json;
use axum::body::Body;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::anthropic::error::json_error;
use crate::anthropic::schema::{CountTokensResponse, MessagesRequest};
#[cfg(test)]
use crate::anthropic::sse::parse_sse_events;
use crate::config;
use crate::logging::create_logger;
use crate::monitor::usage_from_anthropic_sse;
use crate::provider::{CliHandlers, Provider, RequestContext};
use crate::registry;
use crate::request_identity::ConversationIdentity;
use crate::retry::{compute_backoff_delay, sleep};

use self::auth::browser_login::run_browser_login;
use self::auth::device::DeviceAuthClient;
use self::auth::manager::CodexAuthManager;
use self::auth::token_store::file_store;
use self::client::CodexHttpClient;
#[cfg(test)]
use self::continuation::continuation_candidate_for_owner;
use self::continuation::{
    ContinuationReservation, abort_continuation_for_owner, record_continuation_for_owner,
};
use self::count_tokens::{CountingGate, count_translated_tokens, estimate_translated_tokens};
use self::translate::accumulate::accumulate_response_with_traffic;
use self::translate::live_stream::LiveStreamTranslator;
use self::translate::model_allowlist::{
    assert_allowed_model, full_lane_web_search_model, resolve_model_request_with_config_override,
    uses_responses_lite_with_full_lane,
};
use self::translate::reducer::finish_metadata_from_upstream;
use self::translate::request::{
    TranslateOptions, has_hosted_web_search, is_compact_messages_request, translate_request,
    translate_request_with_request_id, validate_tool_registration,
};

pub(super) const STREAM_DELIVERY_STALL_TIMEOUT: Duration = Duration::from_secs(300);

/// A continuous full queue is a stalled consumer, not an upstream inference timeout.
pub(super) trait StreamSenderExt<T> {
    fn send_bounded(&self, item: T) -> impl std::future::Future<Output = Result<(), ()>> + Send;
}

impl<T: Send> StreamSenderExt<T> for tokio::sync::mpsc::Sender<T> {
    async fn send_bounded(&self, item: T) -> Result<(), ()> {
        let permit = tokio::time::timeout(STREAM_DELIVERY_STALL_TIMEOUT, self.reserve())
            .await
            .map_err(|_| ())?
            .map_err(|_| ())?;
        permit.send(item);
        Ok(())
    }
}

const MAX_RETRYABLE_LIVE_STREAM_RETRIES: u32 = 10;
const MAX_EMPTY_COMPLETION_RETRIES: u32 = 10;
const EMPTY_CODEX_COMPLETION_DETAIL: &str = "empty_codex_completion";
const LIVE_STREAM_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
use self::translate::stream::translate_stream_bytes_with_traffic;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub(crate) fn clear_session_compaction(session_id: &str) {
    compaction::clear_compaction(session_id);
}

#[cfg(test)]
type CountHook = Arc<dyn Fn(&str) + Send + Sync>;

pub struct CodexProvider {
    client: Arc<CodexHttpClient>,
    counting_gate: Arc<CountingGate>,
    #[cfg(test)]
    count_hook: Option<CountHook>,
}

impl Default for CodexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexProvider {
    pub fn new() -> Self {
        Self {
            client: Arc::new(CodexHttpClient::new().with_early_reasoning()),
            counting_gate: CountingGate::shared(),
            #[cfg(test)]
            count_hook: None,
        }
    }
}

impl CodexProvider {
    async fn handle_count_tokens_inner(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
    ) -> Response {
        let permits = match self.counting_gate.acquire().await {
            Ok(permits) => permits,
            Err(()) => {
                return json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "overloaded_error",
                    "Local token-counting CPU capacity is exhausted; retry later",
                );
            }
        };
        let settings = config::captured_request_settings();
        let observation = performance::current();
        #[cfg(test)]
        let hook = self.count_hook.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            // These owned permits survive caller cancellation until actual work ends.
            let _permits = permits;
            let settings = performance::in_sync_scope(observation, || {
                settings.unwrap_or_else(config::CodexRequestSettings::capture)
            });
            settings.sync_scope(|| {
                let model = body.model.as_deref().unwrap_or("gpt-6-sol");
                let mut resolved = resolve_model_request_with_config_override(
                    model,
                    !body.bypass_provider_model_override,
                );
                if let Err(e) = assert_allowed_model(&resolved.model) {
                    return Err(format!(
                        "Model \"{model}\" resolves to unsupported model \"{}\"",
                        e.model
                    ));
                }
                let use_responses_lite = apply_model_lane_for_request(
                    &mut resolved.model,
                    &body,
                    config::codex_full_lane(),
                );
                #[cfg(test)]
                if let Some(hook) = &hook {
                    hook("translation");
                }
                let translated = match translate_request(
                    &body,
                    TranslateOptions {
                        session_id: None,
                        service_tier: resolved.service_tier.clone(),
                        model: resolved.model.clone(),
                        use_responses_lite,
                    },
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        return Err(e.to_string());
                    }
                };

                #[cfg(test)]
                if let Some(hook) = &hook {
                    hook("tokenizer");
                }
                let tokens = count_translated_tokens(&translated);
                Ok((resolved.model, tokens))
            })
        })
        .await;
        let (model, tokens) = match prepared {
            Ok(Ok(prepared)) => prepared,
            Ok(Err(message)) => {
                return json_error(StatusCode::BAD_REQUEST, "invalid_request_error", message);
            }
            Err(_) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Local token-counting job failed",
                );
            }
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model);
            monitor.typed_progress(
                &ctx.req_id,
                0,
                0,
                crate::monitor::UsageObservation {
                    input_tokens: Some(crate::monitor::TokenObservation::estimated(tokens)),
                    ..Default::default()
                },
            );
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }

    async fn handle_messages_inner(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
        let want_stream = body.stream;
        let model = body.model.as_deref().unwrap_or("gpt-6-sol");

        let mut resolved =
            resolve_model_request_with_config_override(model, !body.bypass_provider_model_override);
        if let Err(e) = assert_allowed_model(&resolved.model) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Model \"{model}\" resolves to unsupported model \"{}\"",
                    e.model
                ),
            );
        }
        if !is_compact_messages_request(&body) && search::is_standalone_search_request(&body) {
            if let Err(error) = validate_tool_registration(&body) {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    error.to_string(),
                );
            }
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.model_resolved(&ctx.req_id, &resolved.model);
            }
            let (search_request, query) = match search::build_search_request(
                &body,
                &resolved.model,
                ctx.session_id.as_deref(),
            ) {
                Ok(request) => request,
                Err(error) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        error.to_string(),
                    );
                }
            };
            let log = create_logger("codex");
            let started_at = Instant::now();
            log.info(
                "codex_standalone_search_started",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                    ("model".to_string(), serde_json::json!(&resolved.model)),
                    ("stream".to_string(), serde_json::json!(want_stream)),
                ])),
            );
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.upstream_started(&ctx.req_id);
            }
            let search_response = match self.client.post_search(&search_request, &ctx).await {
                Ok(response) => response,
                Err(error) => {
                    log.warn(
                        "codex_standalone_search_failed",
                        Some(serde_json::Map::from_iter([
                            ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                            ("model".to_string(), serde_json::json!(&resolved.model)),
                            ("status".to_string(), serde_json::json!(error.status)),
                            (
                                "ms".to_string(),
                                serde_json::json!(started_at.elapsed().as_millis()),
                            ),
                        ])),
                    );
                    return map_codex_error_to_response(&error);
                }
            };
            log.info(
                "codex_standalone_search_completed",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                    ("model".to_string(), serde_json::json!(&resolved.model)),
                    (
                        "resultCount".to_string(),
                        serde_json::json!(search_response.results.as_ref().map(Vec::len)),
                    ),
                    (
                        "ms".to_string(),
                        serde_json::json!(started_at.elapsed().as_millis()),
                    ),
                ])),
            );
            let input_tokens = search::search_request_input_tokens(&search_request);
            let output_tokens = search::search_response_output_tokens(&search_response);
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.usage_updated(&ctx.req_id, Some(input_tokens), Some(output_tokens));
            }
            return search::anthropic_search_response_with_request_id(
                &search_response,
                &query,
                &message_id,
                model,
                want_stream,
                input_tokens,
                ctx.traffic.as_deref(),
                Some(&ctx.req_id),
            );
        }
        let use_responses_lite =
            apply_model_lane_for_request(&mut resolved.model, &body, config::codex_full_lane());
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
        }

        let translated = match translate_request_with_request_id(
            &body,
            TranslateOptions {
                session_id: ctx.session_id.clone(),
                service_tier: resolved.service_tier.clone(),
                model: resolved.model.clone(),
                use_responses_lite,
            },
            Some(&ctx.req_id),
        ) {
            Ok(t) => t,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                );
            }
        };

        let previous_response_id_enabled = config::codex_previous_response_id();
        let continuation = continuation::reserve_owner_request(
            conversation_identity.as_ref(),
            &translated,
            previous_response_id_enabled,
            self.client.websocket_reuse(),
        );
        let continuation = if is_compact_messages_request(&body) {
            continuation.without_transcript_recording()
        } else {
            continuation
        };
        let turn_id = continuation.turn_id();
        let configured_transport = config::codex_transport();
        let transport = configured_transport.as_str();
        let upstream_started_at = Instant::now();
        let log = create_logger("codex");
        let req_id = ctx.req_id.clone();
        log.info(
            "codex_upstream_request_started",
            Some(serde_json::Map::from_iter([
                ("reqId".to_string(), serde_json::json!(&req_id)),
                (
                    "requestKind".to_string(),
                    serde_json::json!(if is_compact_messages_request(&body) {
                        "compact"
                    } else {
                        "messages"
                    }),
                ),
                ("transport".to_string(), serde_json::json!(transport)),
                ("model".to_string(), serde_json::json!(&resolved.model)),
                ("stream".to_string(), serde_json::json!(want_stream)),
                (
                    "reasoning".to_string(),
                    serde_json::json!(&translated.reasoning),
                ),
                (
                    "requestedProxySpeed".to_string(),
                    serde_json::json!(
                        body.proxy_speed_snapshot
                            .map(crate::proxy_speed::ProxySpeedMode::as_str)
                    ),
                ),
                (
                    "serviceTier".to_string(),
                    serde_json::json!(&translated.service_tier),
                ),
                (
                    "inputItems".to_string(),
                    serde_json::json!(translated.input.len()),
                ),
                (
                    "websocketReuseEnabled".to_string(),
                    serde_json::json!(self.client.websocket_reuse()),
                ),
                (
                    "poolingEnabled".to_string(),
                    serde_json::json!(continuation.pool_enabled()),
                ),
                (
                    "responsesLite".to_string(),
                    serde_json::json!(use_responses_lite),
                ),
                (
                    "previousResponseIdEnabled".to_string(),
                    serde_json::json!(previous_response_id_enabled),
                ),
                (
                    "hasPreviousResponseId".to_string(),
                    serde_json::json!(continuation.candidate().previous_response_id.is_some()),
                ),
                (
                    "inputDeltaCount".to_string(),
                    serde_json::json!(continuation.candidate().input_delta.as_ref().map(Vec::len)),
                ),
                ("turnId".to_string(), serde_json::json!(turn_id)),
            ])),
        );

        let client = self.client.clone();
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        if want_stream {
            let stream_request = Arc::new(translated);
            let response = live_stream_response(
                client,
                message_id,
                model,
                ctx,
                stream_request,
                continuation,
                configured_transport,
                tail_is_successful_tool_result(&body),
            )
            .await;
            log.info(
                "codex_upstream_response_ready",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&req_id)),
                    ("transport".to_string(), serde_json::json!(transport)),
                    ("phase".to_string(), serde_json::json!("response_ready")),
                    (
                        "outcome".to_string(),
                        serde_json::json!(if response.status().is_success() {
                            "pending"
                        } else {
                            "failed"
                        }),
                    ),
                    (
                        "status".to_string(),
                        serde_json::json!(response.status().as_u16()),
                    ),
                    (
                        "ms".to_string(),
                        serde_json::json!(upstream_started_at.elapsed().as_millis()),
                    ),
                ])),
            );
            return response;
        }

        let request_continuation = continuation.clone();
        let mut continuation = Some(continuation);
        let mut attempt = 0_u32;
        let mut empty_completion_attempts = 0_u32;
        let successful_tool_tail = tail_is_successful_tool_result(&body);
        let upstream = loop {
            let response = match client
                .post_codex_for_owner(&translated, &ctx, continuation.as_ref())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    log.warn(
                        "codex_upstream_request_failed",
                        Some(serde_json::Map::from_iter([
                            ("reqId".to_string(), serde_json::json!(&req_id)),
                            ("transport".to_string(), serde_json::json!(transport)),
                            ("status".to_string(), serde_json::json!(e.status)),
                            (
                                "origin".to_string(),
                                serde_json::json!(format!("{:?}", e.origin)),
                            ),
                            ("error".to_string(), serde_json::json!(&e.message)),
                            (
                                "ms".to_string(),
                                serde_json::json!(upstream_started_at.elapsed().as_millis()),
                            ),
                        ])),
                    );
                    abort_continuation_for_owner(&request_continuation);
                    return map_codex_error_to_response(&e);
                }
            };
            if response.had_intervening_failure {
                empty_completion_attempts = 0;
            }
            let output = buffered_output_summary(&response.body, Some(&ctx.req_id));
            if let Some(output) = output.as_ref()
                && output.thinking_only
            {
                if let Some(usage) = output.usage.as_ref() {
                    record_buffered_usage(&ctx, usage);
                }
                abort_continuation_for_owner(&request_continuation);
                return map_live_stream_error_to_response(
                    &translate::live_stream::LiveStreamError::LocalProtocol(
                        "Codex completed without producing an answer or tools".to_string(),
                    ),
                );
            }
            if !output.is_some_and(|output| output.empty) {
                break response;
            }
            let confirmed_empty = buffered_terminal_is_completed(&response.body);
            if accept_empty_completion(successful_tool_tail, empty_completion_attempts)
                && confirmed_empty
            {
                // A repeated empty success can end a fully resolved tool exchange.
                break response;
            }
            // A successful terminal event with no output would translate into
            // an empty end_turn; retry with full context instead.
            let error = empty_buffered_completion_error();
            empty_completion_attempts =
                advance_empty_success_sequence(empty_completion_attempts, confirmed_empty, false);
            drop_live_continuation_for_retry(&mut continuation);
            if attempt >= MAX_EMPTY_COMPLETION_RETRIES {
                log.warn(
                    "codex_empty_completion_exhausted",
                    Some(empty_completion_exhausted_fields(
                        &req_id,
                        transport,
                        &resolved.model,
                        attempt + 1,
                        "max_attempts",
                        upstream_started_at.elapsed().as_millis(),
                    )),
                );
                abort_continuation_for_owner(&request_continuation);
                return map_codex_error_to_response(&error);
            }
            let delay = compute_backoff_delay(attempt, None);
            if delay.exceeds_budget {
                log.warn(
                    "codex_empty_completion_exhausted",
                    Some(empty_completion_exhausted_fields(
                        &req_id,
                        transport,
                        &resolved.model,
                        attempt + 1,
                        "backoff_budget",
                        upstream_started_at.elapsed().as_millis(),
                    )),
                );
                abort_continuation_for_owner(&request_continuation);
                return map_codex_error_to_response(&error);
            }
            log.warn(
                "codex_empty_completion_retry",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&req_id)),
                    (
                        "configuredTransport".to_string(),
                        serde_json::json!(transport),
                    ),
                    ("model".to_string(), serde_json::json!(&resolved.model)),
                    ("failedAttempt".to_string(), serde_json::json!(attempt + 1)),
                    (
                        "maxAttempts".to_string(),
                        serde_json::json!(MAX_EMPTY_COMPLETION_RETRIES + 1),
                    ),
                    ("delayMs".to_string(), serde_json::json!(delay.wait_ms)),
                    (
                        "ms".to_string(),
                        serde_json::json!(upstream_started_at.elapsed().as_millis()),
                    ),
                ])),
            );
            attempt += 1;
            sleep(delay.wait_ms).await;
        };
        if let Some(failure) = events::first_event_failure(&upstream.body) {
            abort_continuation_for_owner(&request_continuation);
            return map_codex_event_failure_to_response(&failure);
        }
        log.info(
            "codex_upstream_response_received",
            Some(serde_json::Map::from_iter([
                ("reqId".to_string(), serde_json::json!(&req_id)),
                ("transport".to_string(), serde_json::json!(transport)),
                ("status".to_string(), serde_json::json!(upstream.status)),
                (
                    "bodyBytes".to_string(),
                    serde_json::json!(upstream.body.len()),
                ),
                (
                    "ms".to_string(),
                    serde_json::json!(upstream_started_at.elapsed().as_millis()),
                ),
            ])),
        );

        if want_stream {
            let estimated_input_tokens = count_translated_tokens(&translated);
            let sse_bytes = match translate_stream_bytes_with_traffic(
                &upstream.body,
                &message_id,
                model,
                estimated_input_tokens,
                ctx.traffic.as_deref(),
            ) {
                Ok(b) => b,
                Err(e) => {
                    abort_continuation_for_owner(&request_continuation);
                    return map_codex_translation_error_to_response(&e);
                }
            };
            if let Some(monitor) = ctx.monitor.as_ref() {
                let (input_tokens, output_tokens) = usage_from_anthropic_sse(&sse_bytes);
                monitor.stream_progress(
                    &ctx.req_id,
                    sse_bytes.len() as u64,
                    count_sse_events(&sse_bytes),
                    input_tokens,
                    output_tokens,
                );
            }
            update_continuation_from_upstream(
                &ctx,
                &request_continuation,
                &translated,
                &upstream.body,
                upstream.socket_id,
            );

            let headers = [
                (http::header::CONTENT_TYPE, "text/event-stream"),
                (http::header::CACHE_CONTROL, "no-cache"),
                (http::header::CONNECTION, "keep-alive"),
            ];
            (headers, sse_bytes).into_response()
        } else {
            match accumulate_response_with_traffic(
                &upstream.body,
                &message_id,
                model,
                ctx.traffic.as_deref(),
            ) {
                Ok(json) => {
                    if let Some(monitor) = ctx.monitor.as_ref() {
                        monitor.usage_updated(
                            &ctx.req_id,
                            json.pointer("/usage/input_tokens").and_then(|v| v.as_u64()),
                            json.pointer("/usage/output_tokens")
                                .and_then(|v| v.as_u64()),
                        );
                    }
                    update_continuation_from_upstream(
                        &ctx,
                        &request_continuation,
                        &translated,
                        &upstream.body,
                        upstream.socket_id,
                    );
                    (StatusCode::OK, Json(json)).into_response()
                }
                Err(e) => {
                    abort_continuation_for_owner(&request_continuation);
                    map_codex_translation_error_to_response(&e)
                }
            }
        }
    }
}

#[async_trait]
impl Provider for CodexProvider {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn supported_models(&self) -> Vec<String> {
        registry::expand_codex_models()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &CODEX_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        config::request_scope(async {
            crate::provider::request_scoped_response(
                self.handle_messages_inner(body, ctx, None).await,
            )
        })
        .await
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        config::request_scope(async {
            crate::provider::request_scoped_response(
                self.handle_messages_inner(body, ctx, conversation_identity)
                    .await,
            )
        })
        .await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        crate::provider::request_scoped_response(self.handle_count_tokens_inner(body, ctx).await)
    }
}

/// Picks the upstream model and lane for a request. Hosted web_search must
/// run on the full Responses API (the lite lane rejects hosted tools), and
/// lite-only models like gpt-5.6-luna don't exist there, so such requests
/// are upgraded to a full-lane model. Returns whether to use the lite lane.
fn apply_model_lane_for_request(
    model: &mut String,
    body: &MessagesRequest,
    full_lane: bool,
) -> bool {
    if !is_compact_messages_request(body) && has_hosted_web_search(body) {
        *model = full_lane_web_search_model(model).to_string();
        return false;
    }
    uses_responses_lite_with_full_lane(model, full_lane)
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    String::from_utf8_lossy(bytes).matches("event:").count() as u64
}

struct LiveRequestStateCleanup {
    continuation: ContinuationReservation,
    armed: bool,
}

impl LiveRequestStateCleanup {
    fn new(continuation: ContinuationReservation) -> Self {
        Self {
            continuation,
            armed: true,
        }
    }

    fn abort(&mut self) {
        if self.armed {
            abort_continuation_for_owner(&self.continuation);
            self.armed = false;
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LiveRequestStateCleanup {
    fn drop(&mut self) {
        if self.armed {
            abort_continuation_for_owner(&self.continuation);
        }
    }
}

enum LiveStreamStart {
    Response(Response),
    Retry {
        error: client::CodexError,
        full_context_retry_attempted: bool,
        confirmed_empty_success: bool,
        intervening_failure: bool,
    },
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_response(
    client: Arc<CodexHttpClient>,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_body: impl Into<Arc<translate::request::ResponsesRequest>>,
    continuation: ContinuationReservation,
    transport: config::CodexTransport,
    successful_tool_tail: bool,
) -> Response {
    let request_body = request_body.into();
    let estimated_input_tokens = estimate_translated_tokens(&request_body);
    let model = model.to_string();
    let started_at = Instant::now();
    let request_continuation = continuation.clone();
    let mut empty_completion_attempts = 0_u32;
    let mut cleanup = LiveRequestStateCleanup::new(request_continuation.clone());
    let mut attempt = 0_u32;
    let mut continuation = Some(continuation);

    loop {
        let attempt_started = Instant::now();
        create_logger("codex").info(
            "codex_live_attempt_started",
            Some(
                serde_json::json!({
                    "reqId": ctx.req_id,
                    "attempt": attempt + 1,
                    "transport": transport.as_str(),
                    "model": request_body.model,
                    "estimatedInputTokens": estimated_input_tokens,
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
        );
        let upstream_events = match transport {
            config::CodexTransport::Http => {
                client
                    .stream_codex_http_events_for_owner(&request_body, &ctx)
                    .await
            }
            config::CodexTransport::WebSocket => {
                client
                    .stream_codex_websocket_events_shared_for_owner(
                        Arc::clone(&request_body),
                        &ctx,
                        continuation.as_ref(),
                    )
                    .await
            }
            config::CodexTransport::Auto => {
                client
                    .stream_codex_auto_events_shared_for_owner(
                        Arc::clone(&request_body),
                        &ctx,
                        continuation.as_ref(),
                    )
                    .await
            }
        };
        if let Err(error) = &upstream_events {
            log_live_attempt_failure(&ctx, error, attempt + 1, attempt_started.elapsed());
        }
        let upstream_events = match upstream_events {
            Ok(events) => events,
            Err(err) if err.origin == client::CodexErrorOrigin::Http => {
                cleanup.abort();
                return map_codex_error_to_response(&err);
            }
            Err(err) if retryable_live_start_codex_error(&err) => {
                empty_completion_attempts = 0;
                let dropped = drop_live_continuation_for_retry(&mut continuation);
                if dropped && is_missing_previous_response_error(&err) {
                    attempt += 1;
                    continue;
                }
                if attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    cleanup.abort();
                    return map_codex_error_to_response(&err);
                }
                let delay = compute_backoff_delay(attempt, err.retry_after.as_deref());
                if delay.exceeds_budget {
                    cleanup.abort();
                    return map_codex_error_to_response(&err);
                }
                attempt += 1;
                sleep(delay.wait_ms).await;
                continue;
            }
            Err(err) => {
                cleanup.abort();
                return map_codex_error_to_response(&err);
            }
        };

        match live_stream_response_once(
            upstream_events,
            message_id.clone(),
            &model,
            ctx.clone(),
            request_continuation.clone(),
            request_body.clone(),
            estimated_input_tokens,
            transport,
            accept_empty_completion(successful_tool_tail, empty_completion_attempts),
        )
        .await
        {
            LiveStreamStart::Response(response) => {
                cleanup.disarm();
                return response;
            }
            LiveStreamStart::Retry {
                error,
                full_context_retry_attempted,
                confirmed_empty_success,
                intervening_failure,
            } => {
                log_live_attempt_failure(&ctx, &error, attempt + 1, attempt_started.elapsed());
                // The incremental HTTP reader performs its own bounded
                // pre-semantic retries so it can stop immediately when the
                // consumer disappears. Do not multiply that exhausted retry
                // loop by the provider-level WebSocket retry policy.
                if error.origin == client::CodexErrorOrigin::Http {
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                empty_completion_attempts = advance_empty_success_sequence(
                    empty_completion_attempts,
                    confirmed_empty_success,
                    intervening_failure,
                );
                let dropped = drop_live_continuation_for_retry(&mut continuation);
                if full_context_retry_attempted && client::is_continuation_retry_error(&error) {
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                if dropped && is_missing_previous_response_error(&error) {
                    attempt += 1;
                    continue;
                }
                if attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    if is_empty_completion_error(&error) {
                        create_logger("codex").warn(
                            "codex_empty_completion_exhausted",
                            Some(empty_completion_exhausted_fields(
                                &ctx.req_id,
                                transport.as_str(),
                                &model,
                                attempt + 1,
                                "max_attempts",
                                started_at.elapsed().as_millis(),
                            )),
                        );
                    }
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                let delay = compute_backoff_delay(attempt, error.retry_after.as_deref());
                if delay.exceeds_budget {
                    if is_empty_completion_error(&error) {
                        create_logger("codex").warn(
                            "codex_empty_completion_exhausted",
                            Some(empty_completion_exhausted_fields(
                                &ctx.req_id,
                                transport.as_str(),
                                &model,
                                attempt + 1,
                                "backoff_budget",
                                started_at.elapsed().as_millis(),
                            )),
                        );
                    }
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                attempt += 1;
                sleep(delay.wait_ms).await;
            }
        }
    }
}

fn log_live_attempt_failure(
    ctx: &RequestContext,
    error: &client::CodexError,
    attempt: u32,
    elapsed: Duration,
) {
    create_logger("codex").warn(
        "codex_live_attempt_failed",
        Some(
            serde_json::json!({
                "reqId": ctx.req_id,
                "attempt": attempt,
                "ms": elapsed.as_millis(),
                "status": error.status,
                "origin": format!("{:?}", error.origin),
                "detail": error.detail,
                "error": error.message,
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    );
}

fn log_live_stream_failure(ctx: &RequestContext, phase: &str, error: &str) {
    create_logger("codex").warn(
        "codex_live_stream_failed",
        Some(
            serde_json::json!({"reqId": ctx.req_id, "phase": phase, "outcome": "failed", "error": error})
                .as_object()
                .unwrap()
                .clone(),
        ),
    );
}

fn provider_retry(
    upstream_events: &websocket::CodexWebSocketEventStream,
    error: client::CodexError,
) -> LiveStreamStart {
    provider_retry_with_empty_confirmation(upstream_events, error, false)
}

fn provider_retry_with_empty_confirmation(
    upstream_events: &websocket::CodexWebSocketEventStream,
    error: client::CodexError,
    confirmed_empty_success: bool,
) -> LiveStreamStart {
    let full_context_retry_attempted = upstream_events.used_full_context_retry();
    let intervening_failure = upstream_events.had_intervening_failure();
    upstream_events.mark_provider_retry_handoff();
    LiveStreamStart::Retry {
        error,
        full_context_retry_attempted,
        confirmed_empty_success,
        intervening_failure,
    }
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_response_once(
    mut upstream_events: websocket::CodexWebSocketEventStream,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_continuation: ContinuationReservation,
    request_body: impl Into<Arc<translate::request::ResponsesRequest>>,
    estimated_input_tokens: u64,
    configured_transport: config::CodexTransport,
    accept_empty: bool,
) -> LiveStreamStart {
    let request_body = request_body.into();
    let yield_diagnostics = ThinkingYieldDiagnostics {
        logger: create_logger("codex").child(serde_json::Map::from_iter([
            ("reqId".into(), ctx.req_id.clone().into()),
            ("processId".into(), std::process::id().into()),
        ])),
        attempt_id: upstream_events.diagnostic_attempt_handle(),
        recorded: false,
    };
    let mut translator = LiveStreamTranslator::with_estimated_input_tokens(
        message_id,
        model.to_string(),
        estimated_input_tokens,
    )
    .with_early_reasoning();
    translator.set_request_id(&ctx.req_id);
    let replay_needed = request_continuation.records_transcript();
    translator = translator.with_replay_budget(
        replay_needed.then(|| config::codex_cache_policy().owner_bytes as usize),
    );
    // A published thinking or tool start already makes a transparent retry unsafe.
    // Every branch that consumes pending_chunk returns, so it is never flushed twice.
    translator.stage_capture(ctx.traffic.is_some());
    let mut pending_progress = translate::live_stream::EmissionProgress::default();
    let mut pending_chunk = Vec::new();
    let mut generation_started = false;
    let mut staged_control = false;

    while let Some(item) = upstream_events.recv().await {
        let payload = match item {
            Ok(payload) => payload,
            Err(err) => {
                if retryable_live_start_codex_error(&err) {
                    return provider_retry(&upstream_events, err);
                }
                abort_continuation_for_owner(&request_continuation);
                return LiveStreamStart::Response(map_codex_error_to_response(&err));
            }
        };
        if !generation_started && codex_generation_event(&payload) {
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.generation_started(&ctx.req_id);
            }
            generation_started = true;
        }

        let (chunk, terminal, progress) =
            match translate_live_stream_payload(&mut translator, &payload, None) {
                Ok(result) => result,
                Err(message) => {
                    // A spent subscription window reopens hours from now, so the
                    // retry budget can only burn the request down to the same 429.
                    // Report it once, with the reset clock upstream supplied.
                    if let Some(limit) = events::usage_limit_from_event(&payload) {
                        abort_continuation_for_owner(&request_continuation);
                        return LiveStreamStart::Response(usage_limit_response(&limit));
                    }
                    if let Some(failure) = events::classify_event_failure(&payload) {
                        record_event_failure_diagnostics(
                            &ctx,
                            &failure,
                            &payload,
                            translator.has_semantic_output(),
                        );
                        if failure.retryable() {
                            return provider_retry(
                                &upstream_events,
                                codex_event_failure_error(
                                    &failure,
                                    client::CodexErrorOrigin::WebSocket,
                                ),
                            );
                        }
                        abort_continuation_for_owner(&request_continuation);
                        record_live_stream_progress(&ctx, &[], translator.take_progress());
                        return LiveStreamStart::Response(map_codex_event_failure_to_response(
                            &failure,
                        ));
                    }
                    abort_continuation_for_owner(&request_continuation);
                    record_live_stream_progress(&ctx, &[], translator.take_progress());
                    return LiveStreamStart::Response(map_live_stream_error_to_response(&message));
                }
            };
        let control_ping = matches!(
            payload.get("type").and_then(serde_json::Value::as_str),
            Some("keepalive" | "response.created" | "response.in_progress" | "codex.rate_limits")
        );
        if !control_ping || !staged_control {
            if !translator.has_semantic_output()
                && chunk.len() > (8 * 1024 * 1024usize).saturating_sub(pending_chunk.len())
            {
                abort_continuation_for_owner(&request_continuation);
                return LiveStreamStart::Response(map_live_stream_error_to_response(
                    &translate::live_stream::LiveStreamError::LocalProtocol(
                        "Codex pre-semantic output buffer limit exceeded".to_string(),
                    ),
                ));
            }
            pending_chunk.extend_from_slice(&chunk);
            pending_progress.append(progress);
            staged_control |= control_ping;
        }
        if terminal
            && is_codex_success_terminal_event(&payload)
            && !translator.has_semantic_output()
            && !translator.had_compatibility_skips()
            && !(accept_empty
                && terminal_status_is_completed(&payload)
                && !upstream_events.had_intervening_failure())
        {
            create_logger("codex").warn(
                "codex_empty_completion_retry",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                    (
                        "configuredTransport".to_string(),
                        serde_json::json!(configured_transport.as_str()),
                    ),
                    ("model".to_string(), serde_json::json!(model)),
                    ("stream".to_string(), serde_json::json!(true)),
                ])),
            );
            return provider_retry_with_empty_confirmation(
                &upstream_events,
                empty_live_completion_error(),
                terminal_status_is_completed(&payload),
            );
        }
        if translator.has_semantic_output() && !pending_chunk.is_empty() {
            translator.stage_capture(false);
            let thinking_started = pending_progress.thinking_started;
            record_live_stream_progress(&ctx, &pending_chunk, pending_progress);
            if terminal {
                update_continuation_from_finish(
                    &request_continuation,
                    &request_body,
                    translator.finish_metadata(),
                    upstream_events.socket_id(),
                );
                return LiveStreamStart::Response(single_live_stream_response(
                    LiveChunk::new(pending_chunk, thinking_started),
                    upstream_events,
                    LiveRequestStateCleanup::new(request_continuation.clone()),
                    yield_diagnostics,
                ));
            }
            return LiveStreamStart::Response(remaining_live_stream_response(
                upstream_events,
                translator,
                LiveChunk::new(pending_chunk, thinking_started),
                ctx,
                request_continuation,
                request_body,
                yield_diagnostics,
            ));
        }
        if terminal {
            update_continuation_from_finish(
                &request_continuation,
                &request_body,
                translator.finish_metadata(),
                upstream_events.socket_id(),
            );
            if pending_chunk.is_empty() {
                upstream_events.accept_terminal();
                return LiveStreamStart::Response(empty_live_stream_response());
            }
            translator.stage_capture(false);
            let thinking_started = pending_progress.thinking_started;
            record_live_stream_progress(&ctx, &pending_chunk, pending_progress);
            return LiveStreamStart::Response(single_live_stream_response(
                LiveChunk::new(pending_chunk, thinking_started),
                upstream_events,
                LiveRequestStateCleanup::new(request_continuation.clone()),
                yield_diagnostics,
            ));
        }
    }

    if upstream_events.delivery_aborted() {
        abort_continuation_for_owner(&request_continuation);
        return LiveStreamStart::Response(map_codex_failure_to_response(
            "Downstream delivery stalled",
        ));
    }

    provider_retry(
        &upstream_events,
        client::CodexError {
            status: 0,
            message: "WebSocket connection closed before terminal Codex response event".to_string(),
            detail: Some(websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocket,
        },
    )
}

/// Empty success is eligible only when the last exchange resolves every tool call.
/// Text and system messages may contain new instructions, even when added by hooks.
fn tail_is_successful_tool_result(body: &MessagesRequest) -> bool {
    use std::collections::HashSet;

    let [.., assistant, user] = body.messages.as_slice() else {
        return false;
    };
    if assistant.role != "assistant" || user.role != "user" {
        return false;
    }
    let (Some(calls), Some(results)) = (assistant.content.as_array(), user.content.as_array())
    else {
        return false;
    };
    let mut pending = HashSet::new();
    for block in calls {
        if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
            let Some(id) = block
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|id| !id.is_empty())
            else {
                return false;
            };
            if !pending.insert(id) {
                return false;
            }
        }
    }
    if pending.is_empty() {
        return false;
    }
    for block in results {
        match block.get("type").and_then(|v| v.as_str()) {
            Some("text")
                if block
                    .get("text")
                    .and_then(|v| v.as_str())
                    .is_some_and(|text| text.trim().is_empty()) => {}
            Some("tool_result") => {
                if !matches!(
                    block.get("is_error"),
                    None | Some(serde_json::Value::Bool(false))
                ) {
                    return false;
                }
                let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) else {
                    return false;
                };
                if !pending.remove(id) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    pending.is_empty()
}

/// An empty end_turn must not hide a missing or contradictory upstream status.
fn terminal_status_is_completed(payload: &serde_json::Value) -> bool {
    payload.pointer("/response/status").and_then(|v| v.as_str()) == Some("completed")
}

fn buffered_terminal_is_completed(upstream_sse: &[u8]) -> bool {
    crate::anthropic::sse::parse_sse_events(upstream_sse)
        .into_iter()
        .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.data).ok())
        .rfind(events::event_is_terminal)
        .is_some_and(|payload| {
            is_codex_success_terminal_event(&payload) && terminal_status_is_completed(&payload)
        })
}

fn advance_empty_success_sequence(previous: u32, confirmed: bool, interrupted: bool) -> u32 {
    if !confirmed {
        0
    } else if interrupted {
        1
    } else {
        previous.saturating_add(1)
    }
}

/// Retry once with full context before permitting an empty successful tool follow-up.
/// Repetition is a heuristic, not proof that the tool delivered a user-visible answer.
fn accept_empty_completion(successful_tool_tail: bool, empty_attempts: u32) -> bool {
    successful_tool_tail && empty_attempts >= 1
}

fn empty_live_completion_error() -> client::CodexError {
    client::CodexError {
        status: 503,
        message: "Codex completed without producing output".to_string(),
        detail: Some(EMPTY_CODEX_COMPLETION_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: client::CodexErrorOrigin::WebSocket,
    }
}

fn codex_generation_event(payload: &serde_json::Value) -> bool {
    !matches!(
        payload.get("type").and_then(|value| value.as_str()),
        Some("codex.rate_limits" | "keepalive") | None
    )
}

fn translate_live_stream_payload(
    translator: &mut LiveStreamTranslator,
    payload: &serde_json::Value,
    traffic: Option<&crate::traffic::TrafficCapture>,
) -> Result<
    (Vec<u8>, bool, translate::live_stream::EmissionProgress),
    translate::live_stream::LiveStreamError,
> {
    rate_limits::record_event(payload);
    let chunk = translator.accept(payload, traffic)?;
    let terminal = events::event_is_terminal(payload) || translator.is_finished();
    Ok((chunk, terminal, translator.take_progress()))
}

fn record_live_stream_progress(
    ctx: &RequestContext,
    chunk: &[u8],
    progress: translate::live_stream::EmissionProgress,
) {
    if let Some(traffic) = ctx.traffic.as_ref() {
        for event in progress.captured_events {
            traffic.write_json_event("050-downstream-event", &event);
        }
    }
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.typed_progress(
            &ctx.req_id,
            chunk.len() as u64,
            progress.events,
            progress.usage,
        );
        if let Some(error) = progress.failed {
            monitor.stream_failed(&ctx.req_id, error);
        }
    }
}

struct LiveChunk {
    bytes: Bytes,
    thinking_started: bool,
}

impl LiveChunk {
    fn new(bytes: Vec<u8>, thinking_started: bool) -> Self {
        Self {
            bytes: Bytes::from(bytes),
            thinking_started,
        }
    }
}

struct ThinkingYieldDiagnostics {
    logger: crate::logging::Logger,
    attempt_id: Arc<std::sync::atomic::AtomicU64>,
    recorded: bool,
}

impl ThinkingYieldDiagnostics {
    fn yielded(&mut self, chunk: &LiveChunk) {
        if chunk.thinking_started && !self.recorded {
            self.recorded = true;
            let attempt = self.attempt_id.load(std::sync::atomic::Ordering::Acquire);
            self.logger.info(
                "codex_thinking_start_yielded",
                Some(serde_json::Map::from_iter([
                    ("monotonicUs".into(), timing::monotonic_us().into()),
                    ("boundary".into(), "provider_body_yield".into()),
                    ("phase".into(), "thinking_started".into()),
                    ("outcome".into(), "pending".into()),
                    ("attemptId".into(), (attempt != 0).then_some(attempt).into()),
                ])),
            );
        }
    }
}

fn single_live_stream_response(
    chunk: LiveChunk,
    upstream: websocket::CodexWebSocketEventStream,
    cleanup: LiveRequestStateCleanup,
    diagnostics: ThinkingYieldDiagnostics,
) -> Response {
    event_stream_response(futures_util::stream::unfold(
        (Some(chunk), upstream, cleanup, diagnostics),
        |(chunk, mut upstream, mut cleanup, mut diagnostics)| async move {
            if let Some(chunk) = chunk {
                diagnostics.yielded(&chunk);
                Some((
                    Ok::<Bytes, std::io::Error>(chunk.bytes),
                    (None, upstream, cleanup, diagnostics),
                ))
            } else {
                upstream.accept_terminal();
                cleanup.disarm();
                None
            }
        },
    ))
}

fn empty_live_stream_response() -> Response {
    event_stream_response(futures_util::stream::empty::<Result<Bytes, std::io::Error>>())
}

#[allow(clippy::too_many_arguments)]
fn remaining_live_stream_response(
    mut upstream_events: websocket::CodexWebSocketEventStream,
    mut translator: LiveStreamTranslator,
    first_chunk: LiveChunk,
    ctx: RequestContext,
    request_continuation: ContinuationReservation,
    request_body: impl Into<Arc<translate::request::ResponsesRequest>>,
    diagnostics: ThinkingYieldDiagnostics,
) -> Response {
    let request_body = request_body.into();
    let delivery_guard = upstream_events.delivery_guard();
    let body_cleanup = LiveRequestStateCleanup::new(request_continuation.clone());
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<LiveChunk, std::io::Error>>(64);
    tokio::spawn(crate::config::inherit_request_scope(async move {
        if tx.send_bounded(Ok(first_chunk)).await.is_err() {
            abort_continuation_for_owner(&request_continuation);
            return;
        }
        let mut heartbeat = tokio::time::interval(LIVE_STREAM_HEARTBEAT_INTERVAL);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        heartbeat.tick().await;
        loop {
            let item = tokio::select! {
                biased;
                _ = tx.closed() => {
                    abort_continuation_for_owner(
                        &request_continuation);
                    return;
                }
                item = upstream_events.recv() => item,
                _ = heartbeat.tick() => {
                    let chunk = translator.ping_chunk(ctx.traffic.as_deref());
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk, translator.take_progress());
                        if tx.send_bounded(Ok(LiveChunk::new(chunk, false))).await.is_err() {
                            abort_continuation_for_owner(
                                &request_continuation);
                            return;
                        }
                    }
                    continue;
                }
            };
            let Some(item) = item else {
                if upstream_events.delivery_aborted() {
                    log_live_stream_failure(
                        &ctx,
                        "delivery",
                        "Upstream delivery stalled before terminal",
                    );
                    if let Some(monitor) = ctx.monitor.as_ref() {
                        monitor.stream_failed(
                            &ctx.req_id,
                            "Upstream delivery stalled before terminal",
                        );
                    }
                    abort_continuation_for_owner(&request_continuation);
                    return;
                }
                break;
            };
            match item {
                Ok(payload) => {
                    let (chunk, terminal, progress) = match translate_live_stream_payload(
                        &mut translator,
                        &payload,
                        ctx.traffic.as_deref(),
                    ) {
                        Ok(result) => result,
                        Err(message) => {
                            log_live_stream_failure(&ctx, "translation", message.as_str());
                            abort_continuation_for_owner(&request_continuation);
                            let failure = events::classify_event_failure(&payload);
                            if let Some(failure) = failure.as_ref() {
                                record_event_failure_diagnostics(
                                    &ctx,
                                    failure,
                                    &payload,
                                    translator.has_semantic_output(),
                                );
                            }
                            let error_type = failure
                                .as_ref()
                                .map_or("api_error", events::CodexEventFailure::client_error_type);
                            let error_message = failure
                                .as_ref()
                                .map_or(message.as_str(), |failure| failure.message.as_str());
                            let chunk = translator.error_chunk(
                                error_message,
                                error_type,
                                ctx.traffic.as_deref(),
                            );
                            if !chunk.is_empty() {
                                record_live_stream_progress(
                                    &ctx,
                                    &chunk,
                                    translator.take_progress(),
                                );
                                let _ = tx.send_bounded(Ok(LiveChunk::new(chunk, false))).await;
                            }
                            return;
                        }
                    };
                    if terminal {
                        upstream_events.accept_terminal();
                        update_continuation_from_finish(
                            &request_continuation,
                            &request_body,
                            translator.finish_metadata(),
                            upstream_events.socket_id(),
                        );
                    }
                    if !chunk.is_empty() {
                        let thinking_started = progress.thinking_started;
                        record_live_stream_progress(&ctx, &chunk, progress);
                        if tx
                            .send_bounded(Ok(LiveChunk::new(chunk, thinking_started)))
                            .await
                            .is_err()
                        {
                            abort_continuation_for_owner(&request_continuation);
                            return;
                        }
                    }
                    if terminal {
                        return;
                    }
                }
                Err(err) => {
                    abort_continuation_for_owner(&request_continuation);
                    log_live_stream_failure(&ctx, "upstream", codex_error_message(&err));
                    let error_type = codex_stream_error_type(&err);
                    let chunk = translator.error_chunk(
                        codex_error_message(&err),
                        error_type,
                        ctx.traffic.as_deref(),
                    );
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk, translator.take_progress());
                        let _ = tx.send_bounded(Ok(LiveChunk::new(chunk, false))).await;
                    }
                    return;
                }
            }
        }

        abort_continuation_for_owner(&request_continuation);
        log_live_stream_failure(
            &ctx,
            "eof",
            "Upstream event stream closed before terminal Codex response event",
        );
        let chunk = translator.error_chunk(
            "Upstream event stream closed before terminal Codex response event",
            "api_error",
            ctx.traffic.as_deref(),
        );
        if !chunk.is_empty() {
            record_live_stream_progress(&ctx, &chunk, translator.take_progress());
            let _ = tx.send_bounded(Ok(LiveChunk::new(chunk, false))).await;
        }
    }));

    let stream = futures_util::stream::unfold(
        (rx, delivery_guard, body_cleanup, diagnostics),
        |(mut rx, mut guard, mut cleanup, mut diagnostics)| async {
            match rx.recv().await {
                Some(item) => {
                    let item = item.map(|chunk| {
                        diagnostics.yielded(&chunk);
                        chunk.bytes
                    });
                    Some((item, (rx, guard, cleanup, diagnostics)))
                }
                None => {
                    if let Some(guard) = guard.as_mut() {
                        guard.accept();
                    }
                    cleanup.disarm();
                    None
                }
            }
        },
    );
    event_stream_response(stream)
}

fn event_stream_response<S>(stream: S) -> Response
where
    S: futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    let headers = [
        (http::header::CONTENT_TYPE, "text/event-stream"),
        (http::header::CACHE_CONTROL, "no-cache"),
        (http::header::CONNECTION, "keep-alive"),
    ];
    (headers, Body::from_stream(stream)).into_response()
}

/// True when *error* is the empty-completion 503 rather than some other
/// retryable failure sharing the live-stream retry loop.
fn is_empty_completion_error(error: &client::CodexError) -> bool {
    error.detail.as_deref() == Some(EMPTY_CODEX_COMPLETION_DETAIL)
}

/// Log fields for giving up on an empty completion, on either exit.
///
/// `reason` separates the two: `"max_attempts"` when the retry budget ran
/// out, `"backoff_budget"` when the next delay would have exceeded the
/// backoff budget. They produce the same 503, so the log is the only place
/// the difference is visible.
fn empty_completion_exhausted_fields(
    req_id: &str,
    configured_transport: &str,
    model: &str,
    attempts: u32,
    reason: &str,
    elapsed_ms: u128,
) -> serde_json::Map<String, serde_json::Value> {
    serde_json::Map::from_iter([
        ("reqId".to_string(), serde_json::json!(req_id)),
        (
            "configuredTransport".to_string(),
            serde_json::json!(configured_transport),
        ),
        ("model".to_string(), serde_json::json!(model)),
        ("attempts".to_string(), serde_json::json!(attempts)),
        ("reason".to_string(), serde_json::json!(reason)),
        ("ms".to_string(), serde_json::json!(elapsed_ms)),
    ])
}

fn empty_buffered_completion_error() -> client::CodexError {
    client::CodexError {
        status: 503,
        message: "Codex completed without producing output".to_string(),
        detail: Some(EMPTY_CODEX_COMPLETION_DETAIL.to_string()),
        retry_after: None,
        usage_limit: None,
        origin: match config::codex_transport() {
            config::CodexTransport::Http => client::CodexErrorOrigin::BufferedHttp,
            _ => client::CodexErrorOrigin::BufferedWebSocket,
        },
    }
}

/// True when the buffered upstream body ended in a successful terminal event
/// without ever producing semantic output (text, thinking, tool, web search).
#[cfg(test)]
fn is_empty_codex_success_completion(upstream_sse: &[u8]) -> bool {
    buffered_output_summary(upstream_sse, None).is_some_and(|output| output.empty)
}

struct BufferedOutputSummary {
    empty: bool,
    thinking_only: bool,
    usage: Option<translate::reducer::CodexUsage>,
}

fn buffered_output_summary(
    upstream_sse: &[u8],
    req_id: Option<&str>,
) -> Option<BufferedOutputSummary> {
    use self::translate::event::TranslationEvent;
    let (events, metadata) = self::translate::reducer::translate_upstream_bytes_with_request_id(
        upstream_sse,
        translate::IncompleteResponsePolicy::Error,
        req_id,
    )
    .ok()?;
    let metadata = metadata?;
    if !metadata.stream_success {
        return None;
    }
    let answer_or_tool = events.iter().any(|event| match event {
        TranslationEvent::TextDelta { text, .. } => !text.is_empty(),
        TranslationEvent::ToolStart { .. } | TranslationEvent::SearchComplete { .. } => true,
        _ => false,
    });
    let thinking = events
        .iter()
        .any(|event| matches!(event, TranslationEvent::ThinkingStart { .. }));
    let compatibility_output = events.iter().any(|event| {
        matches!(event, TranslationEvent::Skipped(skip) if skip.reason != "redundant_snapshot")
    });
    Some(BufferedOutputSummary {
        empty: !answer_or_tool && !thinking && !compatibility_output,
        thinking_only: thinking && !answer_or_tool,
        usage: metadata.usage,
    })
}

fn is_codex_success_terminal_event(payload: &serde_json::Value) -> bool {
    events::event_is_success_terminal(payload)
}

fn retryable_live_start_codex_error(err: &client::CodexError) -> bool {
    if err.origin.is_protocol_rejection() {
        return false;
    }
    if err.detail.as_deref() == Some(websocket::WEBSOCKET_AMBIGUOUS_WRITE_DETAIL) {
        return false;
    }
    if err.origin == client::CodexErrorOrigin::WebSocketHandshake {
        if err.detail.as_deref() == Some(websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL) {
            return false;
        }
        return err.status == 0 || matches!(err.status, 429 | 500 | 502 | 503 | 504 | 529);
    }
    if err.detail.as_deref() == Some(websocket::WEBSOCKET_KEEPALIVE_FAILURE_DETAIL) {
        return true;
    }
    matches!(err.status, 429 | 500 | 502 | 503 | 504 | 529)
        || (err.status == 0 && retryable_live_message(codex_error_message(err)))
}

fn is_missing_previous_response_error(err: &client::CodexError) -> bool {
    matches!(
        err.detail.as_deref(),
        Some("previous_response_not_found")
            | Some(websocket::WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
    )
}

fn drop_live_continuation_for_retry(continuation: &mut Option<ContinuationReservation>) -> bool {
    if continuation
        .as_ref()
        .and_then(|reservation| reservation.candidate().previous_response_id.as_deref())
        .is_none()
    {
        return false;
    }

    if let Some(reservation) = continuation.as_ref() {
        *continuation = Some(reservation.full_context_retry());
    }
    true
}

fn retryable_live_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "overloaded",
        "rate limit",
        "you can retry your request",
        "temporarily unavailable",
        "timed out",
        "connection closed",
        "connection reset",
        "broken pipe",
        "epipe",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn record_event_failure_diagnostics(
    ctx: &RequestContext,
    failure: &events::CodexEventFailure,
    payload: &serde_json::Value,
    committed: bool,
) {
    let mut diagnostics = failure.diagnostics(
        "live_provider",
        "event_failure",
        Some(&ctx.req_id),
        payload.get("type").and_then(serde_json::Value::as_str),
        committed,
    );
    diagnostics["message"] = serde_json::json!(&failure.message);
    diagnostics["reqId"] = serde_json::json!(&ctx.req_id);
    create_logger("codex").warn("codex_event_failed", diagnostics.as_object().cloned());
    if let Some(traffic) = ctx.traffic.as_deref() {
        traffic.write_json("060-codex-event-error", &diagnostics);
    }
}

fn codex_event_failure_error(
    failure: &events::CodexEventFailure,
    origin: client::CodexErrorOrigin,
) -> client::CodexError {
    client::CodexError {
        status: failure.status,
        message: failure.message.clone(),
        detail: Some(failure.message.clone()),
        retry_after: failure.retry_after.clone(),
        usage_limit: None,
        origin,
    }
}

fn codex_stream_error_type(err: &client::CodexError) -> &'static str {
    events::client_error_type(err.status)
}

fn record_buffered_usage(ctx: &RequestContext, usage: &translate::reducer::CodexUsage) {
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.typed_progress(
            &ctx.req_id,
            0,
            0,
            crate::monitor::UsageObservation {
                input_tokens: usage
                    .input_tokens
                    .map(crate::monitor::TokenObservation::provider),
                cached_input_tokens: usage.input_tokens_details_cached,
                output_tokens: usage
                    .output_tokens
                    .map(crate::monitor::TokenObservation::provider),
                reasoning_tokens: usage.output_tokens_details_reasoning,
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_continuation_from_upstream(
    ctx: &RequestContext,
    continuation: &ContinuationReservation,
    request_body: &translate::request::ResponsesRequest,
    upstream_body: &[u8],
    socket_id: Option<u64>,
) {
    let finish = finish_metadata_from_upstream(upstream_body).ok().flatten();
    if let Some(usage) = finish.as_ref().and_then(|finish| finish.usage.as_ref()) {
        record_buffered_usage(ctx, usage);
    }
    update_continuation_from_finish(continuation, request_body, finish.as_ref(), socket_id);
}

#[allow(clippy::too_many_arguments)]
fn update_continuation_from_finish(
    continuation: &ContinuationReservation,
    request_body: &translate::request::ResponsesRequest,
    finish: Option<&translate::finish::FinishMetadata>,
    socket_id: Option<u64>,
) {
    match finish {
        Some(finish) if finish.stream_success => {
            // Recording is optional; successful requests still own their socket.
            if !continuation.records_transcript() {
                return;
            }
            if finish.continuation_eligible {
                record_continuation_for_owner(
                    continuation,
                    request_body,
                    finish.response_id.as_deref(),
                    socket_id,
                    &finish.output_items,
                );
            } else {
                abort_continuation_for_owner(continuation);
            }
        }
        _ => {
            abort_continuation_for_owner(continuation);
        }
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Answer a spent subscription window with the rate limit headers the client
/// reads, so it can name the exhausted window and show when it reopens.
///
/// `Retry-After` is deliberately absent: clients sleep for its full value, and
/// here that is hours. `x-should-retry: false` stops the retry loop instead,
/// which is the honest signal — a spent window does not reopen on a backoff.
fn usage_limit_response(limit: &events::CodexUsageLimit) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-should-retry"),
        HeaderValue::from_static("false"),
    );
    headers.insert(
        HeaderName::from_static("anthropic-ratelimit-unified-status"),
        HeaderValue::from_static("rejected"),
    );
    if let Some(resets_at) = limit.resets_at
        && let Ok(value) = HeaderValue::from_str(&resets_at.to_string())
    {
        headers.insert(
            HeaderName::from_static("anthropic-ratelimit-unified-reset"),
            value,
        );
    }
    if let Some(window) = limit.window {
        headers.insert(
            HeaderName::from_static("anthropic-ratelimit-unified-representative-claim"),
            HeaderValue::from_static(window.claim()),
        );
    }

    (
        headers,
        json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            &limit.message,
        ),
    )
        .into_response()
}

fn map_codex_error_to_response(err: &client::CodexError) -> Response {
    if err.origin.is_protocol_rejection() {
        return local_protocol_rejection_response(&err.message);
    }
    if let Some(limit) = err.usage_limit.as_ref() {
        return usage_limit_response(limit);
    }
    let message = codex_error_message(err);
    if err.detail.as_deref() == Some(EMPTY_CODEX_COMPLETION_DETAIL) {
        // The proxy has already spent its own retry budget on this request, so
        // a client retry only multiplies it: Claude Code resends a 503 up to
        // ten times, each one repeating the proxy's full retry loop.
        return (
            [(
                HeaderName::from_static("x-should-retry"),
                HeaderValue::from_static("false"),
            )],
            json_error(StatusCode::SERVICE_UNAVAILABLE, "api_error", &err.message),
        )
            .into_response();
    }

    let status = StatusCode::from_u16(err.status)
        .ok()
        .filter(|status| status.is_client_error() || status.is_server_error())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = json_error(
        status,
        events::client_error_type(status.as_u16()),
        events::sanitize_error_message(message),
    );
    if err.detail.as_deref() == Some(websocket::WEBSOCKET_AMBIGUOUS_WRITE_DETAIL) {
        // The request may already be executing upstream. Match the proxy's
        // no-replay decision at the SDK boundary too.
        response.headers_mut().insert(
            HeaderName::from_static("x-should-retry"),
            HeaderValue::from_static("false"),
        );
    }
    if let Some(retry_after) = err.retry_after.as_deref() {
        ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
    } else {
        response
    }
}

fn map_codex_translation_error_to_response(error: &anyhow::Error) -> Response {
    if let Some(error) = error.downcast_ref::<translate::reducer::UpstreamStreamError>() {
        let status = error
            .diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.client_status)
            .unwrap_or(502);
        let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
        return json_error(
            status,
            events::client_error_type(status.as_u16()),
            events::sanitize_error_message(&error.message),
        );
    }
    map_codex_failure_to_response(&error.to_string())
}

fn local_protocol_rejection_response(message: &str) -> Response {
    (
        [("x-should-retry", "false")],
        json_error(StatusCode::BAD_GATEWAY, "api_error", message),
    )
        .into_response()
}

fn map_live_stream_error_to_response(error: &translate::live_stream::LiveStreamError) -> Response {
    match error {
        translate::live_stream::LiveStreamError::LocalProtocol(message) => {
            local_protocol_rejection_response(message)
        }
        translate::live_stream::LiveStreamError::Upstream(message) => {
            map_codex_failure_to_response(message)
        }
    }
}

fn map_codex_failure_to_response(message: &str) -> Response {
    if is_context_window_overflow(message) {
        json_error(StatusCode::PAYLOAD_TOO_LARGE, "request_too_large", message)
    } else {
        json_error(StatusCode::BAD_GATEWAY, "api_error", message)
    }
}

fn map_codex_event_failure_to_response(failure: &events::CodexEventFailure) -> Response {
    let status = StatusCode::from_u16(failure.client_status()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response = json_error(status, failure.client_error_type(), &failure.message);
    if let Some(retry_after) = failure.retry_after.as_deref() {
        ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
    } else {
        response
    }
}

fn is_context_window_overflow(message: &str) -> bool {
    message.to_ascii_lowercase().contains("context window")
}

fn codex_error_message(err: &client::CodexError) -> &str {
    // Statusless transport errors use detail as an internal category marker.
    if err.status == 0 && !err.message.trim().is_empty() {
        return &err.message;
    }
    err.detail
        .as_deref()
        .filter(|detail| !detail.trim().is_empty())
        .or_else(|| (!err.message.trim().is_empty()).then_some(err.message.as_str()))
        .unwrap_or("Upstream error")
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

pub(crate) struct CodexCli;

impl CliHandlers for CodexCli {
    fn login(&self) -> Result<(), anyhow::Error> {
        let tokens = run_browser_login()?;
        let store = file_store();
        let manager = CodexAuthManager::new(store);
        let saved = manager.persist_initial_tokens(&tokens)?;
        print!(
            "{}",
            format_auth_saved_output(&manager.store.auth_path(), saved.account_id.as_deref())
        );
        Ok(())
    }

    fn device(&self) -> Result<(), anyhow::Error> {
        let tokens = DeviceAuthClient::new().run()?;
        let store = file_store();
        let manager = CodexAuthManager::new(store);
        let saved = manager.persist_initial_tokens(&tokens)?;
        print!(
            "{}",
            format_auth_saved_output(&manager.store.auth_path(), saved.account_id.as_deref())
        );
        Ok(())
    }

    fn status(&self) -> Result<(), anyhow::Error> {
        let store = file_store();
        let stored = store.load_auth()?;
        match stored {
            Some(auth) => {
                println!(
                    "Account: {}",
                    auth.account_id.as_deref().unwrap_or("(none)")
                );
                println!("{}", format_expiry(auth.expires, now_ms()));
                println!("Storage: {}", store.auth_path());
                Ok(())
            }
            None => {
                anyhow::bail!("Not authenticated");
            }
        }
    }

    fn logout(&self) -> Result<(), anyhow::Error> {
        let store = file_store();
        store.clear_auth()?;
        println!("Logged out");
        Ok(())
    }
}

pub(crate) static CODEX_CLI: CodexCli = CodexCli;

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn format_expiry(expires: u64, now: u64) -> String {
    let remaining = (i128::from(expires) - i128::from(now)).div_euclid(1000);
    let iso = time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(expires) * 1_000_000)
        .ok()
        .and_then(|dt| {
            let fmt = time::format_description::parse_borrowed::<2>(
                "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z",
            )
            .ok()?;
            dt.format(&fmt).ok()
        })
        .unwrap_or_else(|| "invalid".to_string());
    format!("Expires: {iso} (in {remaining}s)")
}

fn format_auth_saved_output(auth_path: &str, account_id: Option<&str>) -> String {
    let mut out = format!("Auth saved in {auth_path}\n");
    if let Some(account_id) = account_id {
        out.push_str(&format!("Account: {account_id}\n"));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn ambiguous_websocket_write_disables_client_retry() {
        let error = super::client::CodexError {
            status: 502,
            message: "WebSocket request write did not complete; delivery is ambiguous".into(),
            detail: Some(super::websocket::WEBSOCKET_AMBIGUOUS_WRITE_DETAIL.into()),
            retry_after: None,
            usage_limit: None,
            origin: super::client::CodexErrorOrigin::WebSocket,
        };
        let response = super::map_codex_error_to_response(&error);
        assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            response
                .headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
    }
    use futures_util::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use tokio::net::{TcpListener, TcpStream};
    use tokio_tungstenite::tungstenite::Message;

    use super::*;
    use translate::request::ServiceTier;

    fn live_test_request(text: &str) -> translate::request::ResponsesRequest {
        translate::request::ResponsesRequest {
            model: "gpt-5.6-sol".to_string(),
            instructions: None,
            input: vec![translate::request::ResponsesInputItem::Message {
                role: "user".to_string(),
                content: vec![translate::request::ResponsesContentPart::InputText {
                    text: text.to_string(),
                }],
            }],
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: translate::request::ResponsesText {
                verbosity: None,
                format: None,
            },
            reasoning: None,
        }
    }

    fn live_test_context(session_id: &str) -> RequestContext {
        RequestContext {
            req_id: format!("request-{session_id}"),
            session_id: Some(session_id.to_string()),
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    #[tokio::test]
    async fn buffered_thinking_only_fails_and_keeps_provider_compute_usage() {
        for (summary, carrier) in [(true, false), (false, true), (true, true)] {
            for answer in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let settings =
                    config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                        (
                            "CCP_CONFIG_DIR".into(),
                            dir.path().to_string_lossy().into_owned(),
                        ),
                        (
                            "XDG_STATE_HOME".into(),
                            dir.path().to_string_lossy().into_owned(),
                        ),
                        ("CCP_CODEX_TRANSPORT".into(), "http".into()),
                        ("CCP_CODEX_PREVIOUS_RESPONSE_ID".into(), "0".into()),
                    ]));
                let mut events = vec![
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs"}}),
                ];
                if summary {
                    events.push(serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"summary"}));
                }
                let mut done = serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs"}});
                if carrier {
                    done["item"]["encrypted_content"] = serde_json::json!("opaque");
                }
                events.push(done);
                if answer {
                    events.extend([
                        serde_json::json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m"}}),
                        serde_json::json!({"type":"response.output_text.delta","output_index":1,"delta":"answer"}),
                        serde_json::json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"m","content":[{"type":"output_text","text":"answer"}]}}),
                    ]);
                }
                events.push(serde_json::json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{
                    "input_tokens":11,"input_tokens_details":{"cached_tokens":2},
                    "output_tokens":9,"output_tokens_details":{"reasoning_tokens":9}
                }}}));
                let events = upstream_sse(&events);
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let url = format!("http://{}", listener.local_addr().unwrap());
                let peer_app = axum::Router::new().fallback(axum::routing::post(move || {
                    let events = events.clone();
                    async move { ([("content-type", "text/event-stream")], events) }
                }));
                let peer = tokio::spawn(async move {
                    axum::serve(listener, peer_app).await.unwrap();
                });
                let provider = CodexProvider {
                    client: authenticated_live_test_client(url),
                    counting_gate: CountingGate::new(),
                    count_hook: None,
                };
                let monitor = crate::monitor::MonitorHandle::default();
                let mut ctx = live_test_context("thinking-buffered");
                monitor.request_started(
                    &ctx.req_id,
                    ctx.session_id.clone(),
                    None,
                    crate::monitor::EndpointKind::Messages,
                );
                ctx.monitor = Some(monitor.clone());
                let body = serde_json::from_value(serde_json::json!({
                    "model":"gpt-6-sol","stream":false,"max_tokens":64,
                    "messages":[{"role":"user","content":"hello"}]
                }))
                .unwrap();
                let response = settings.scope(provider.handle_messages(body, ctx)).await;
                assert_eq!(
                    response.status().is_success(),
                    answer,
                    "summary={summary} carrier={carrier}"
                );
                let value = serde_json::to_value(crate::monitor::snapshot::MonitorSnapshot::from(
                    monitor.snapshot(),
                ))
                .unwrap();
                assert_eq!(value["active"][0]["input_tokens"], 11);
                assert_eq!(value["active"][0]["output_tokens"], 9);
                assert_eq!(value["active"][0]["usage"]["cached_input_tokens"], 2);
                assert_eq!(value["active"][0]["usage"]["reasoning_tokens"], 9);
                peer.abort();
            }
        }
    }

    #[tokio::test]
    async fn cached_input_is_reported_for_buffered_messages_from_finish_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            (
                "CCP_CONFIG_DIR".into(),
                dir.path().to_string_lossy().into_owned(),
            ),
            (
                "XDG_STATE_HOME".into(),
                dir.path().to_string_lossy().into_owned(),
            ),
            ("CCP_CODEX_TRANSPORT".into(), "http".into()),
        ]));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let events = upstream_sse(&[
            serde_json::json!({"type":"response.output_item.added","output_index":0,
                "item":{"type":"message","id":"m"}}),
            serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"hello"}),
            serde_json::json!({"type":"response.output_item.done","output_index":0,
                "item":{"type":"message","id":"m","content":[{"type":"output_text","text":"hello"}]}}),
            serde_json::json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{
                "input_tokens":790845,"output_tokens":10,"input_tokens_details":{"cached_tokens":782848}}}}),
        ]);
        let peer_app = axum::Router::new().fallback(axum::routing::post(move || {
            let events = events.clone();
            async move { ([("content-type", "text/event-stream")], events) }
        }));
        let peer = tokio::spawn(async move {
            axum::serve(listener, peer_app).await.unwrap();
        });
        let provider = CodexProvider {
            client: authenticated_live_test_client(url),
            counting_gate: CountingGate::new(),
            count_hook: None,
        };
        let monitor = crate::monitor::MonitorHandle::default();
        let mut ctx = live_test_context("cached-buffered");
        monitor.request_started(
            &ctx.req_id,
            ctx.session_id.clone(),
            None,
            crate::monitor::EndpointKind::Messages,
        );
        ctx.monitor = Some(monitor.clone());
        let request = serde_json::from_value(serde_json::json!({
            "model":"gpt-6-sol","max_tokens":10,"stream":false,
            "messages":[{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = settings.scope(provider.handle_messages(request, ctx)).await;
        peer.abort();
        assert_eq!(response.status(), StatusCode::OK);
        let value = serde_json::to_value(crate::monitor::snapshot::MonitorSnapshot::from(
            monitor.snapshot(),
        ))
        .unwrap();
        assert_eq!(value["active"][0]["input_tokens"], 790845);
        assert_eq!(value["active"][0]["usage"]["cached_input_tokens"], 782848);
        assert_eq!(value["sessions"][0]["cached_input_tokens"], 782848);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn count_preparation_runs_outside_tokio_worker() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"codex":{"model":"gpt-6-sol"}}"#,
        )
        .unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([(
            "CCP_CONFIG_DIR".into(),
            dir.path().to_string_lossy().into_owned(),
        )]));
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"codex":{"model":"gpt-6-astra"}}"#,
        )
        .unwrap();
        let worker = std::thread::current().id();
        let phases = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = phases.clone();
        let provider = CodexProvider {
            client: authenticated_live_test_client("http://127.0.0.1:1".into()),
            counting_gate: CountingGate::new(),
            count_hook: Some(Arc::new(move |phase| {
                assert_ne!(
                    std::thread::current().id(),
                    worker,
                    "{phase} blocks the Tokio worker"
                );
                assert!(
                    config::captured_request_settings().is_some(),
                    "blocking preparation lost its settings scope"
                );
                assert_eq!(
                    config::codex_model().as_deref(),
                    Some("gpt-6-sol"),
                    "count reloaded a captured snapshot"
                );
                observed.lock().unwrap().push(phase.to_owned());
            })),
        };
        let body = serde_json::from_value(serde_json::json!({
            "model":"gpt-6-sol", "max_tokens":10,
            "messages":[{"role":"user","content":"hello"}]
        }))
        .unwrap();
        let response = settings
            .scope(provider.handle_count_tokens(body, live_test_context("off-worker")))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(*phases.lock().unwrap(), ["translation", "tokenizer"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_count_captures_settings_inside_blocking_preparation() {
        assert!(config::captured_request_settings().is_none());
        let worker = std::thread::current().id();
        let provider = CodexProvider {
            client: authenticated_live_test_client("http://127.0.0.1:1".into()),
            counting_gate: CountingGate::new(),
            count_hook: Some(Arc::new(move |_| {
                assert_ne!(std::thread::current().id(), worker);
                assert!(config::captured_request_settings().is_some());
            })),
        };
        let body = serde_json::from_value(serde_json::json!({"model":"gpt-6-sol","max_tokens":10,"messages":[{"role":"user","content":"hello"}]})).unwrap();
        let response = provider
            .handle_count_tokens(body, live_test_context("direct-count"))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(config::captured_request_settings().is_none());
    }

    #[tokio::test]
    async fn count_endpoint_preserves_original_o200k_fixture_values() {
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([(
            "CCP_CONFIG_DIR".into(),
            dir.path().to_string_lossy().into_owned(),
        )]));
        let provider: Arc<dyn Provider> = Arc::new(CodexProvider {
            client: authenticated_live_test_client("http://127.0.0.1:1".into()),
            counting_gate: CountingGate::new(),
            count_hook: None,
        });
        let registry = Arc::new(registry::Registry::from_providers(
            config::AliasProvider::Codex,
            [provider],
        ));
        let app =
            crate::server::app_with_features(registry, None, crate::server::AppFeatures::default());
        let replay = translate::reasoning_signature::ReasoningReplay {
            id: "r".into(),
            encrypted_content: "A".repeat(4000),
        };
        let signature =
            translate::reasoning_signature::encode_reasoning_signature(&replay).unwrap();
        let fixtures = [
            serde_json::json!({"model":"gpt-6-sol","max_tokens":10,"messages":[{"role":"user","content":"汉字测试上下文压缩".repeat(384)}]}),
            serde_json::json!({"model":"gpt-6-sol","max_tokens":10,"tools":[{"name":"fixture","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"messages":[{"role":"user","content":[{"type":"text","text":"caption"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"YQ=="}}]},{"role":"assistant","content":[{"type":"thinking","thinking":"","signature":signature}]}]}),
        ];
        // Independently captured with the original estimator at 5ec30cb.
        for (fixture, want) in fixtures.into_iter().zip([2697, 2635]) {
            let request = http::Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(Body::from(fixture.to_string()))
                .unwrap();
            let response = settings
                .clone()
                .scope(app.clone().oneshot(request))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(wire["input_tokens"], want);
        }
    }

    #[tokio::test]
    async fn local_count_is_displayed_as_estimated_without_billable_session_usage() {
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([(
            "CCP_CONFIG_DIR".into(),
            dir.path().to_string_lossy().into_owned(),
        )]));
        let provider: Arc<dyn Provider> = Arc::new(CodexProvider {
            client: authenticated_live_test_client("http://127.0.0.1:1".into()),
            counting_gate: CountingGate::new(),
            count_hook: None,
        });
        let monitor = crate::monitor::MonitorHandle::default();
        let registry = Arc::new(registry::Registry::from_providers(
            config::AliasProvider::Codex,
            [provider],
        ));
        let app = crate::server::app_with_features(
            registry,
            Some(monitor.clone()),
            crate::server::AppFeatures::default(),
        );
        let request = http::Request::builder().method("POST").uri("/v1/messages/count_tokens")
            .header("content-type", "application/json").header("x-claude-code-session-id", "local-count-provenance")
            .body(Body::from(serde_json::json!({"model":"gpt-6-sol","max_tokens":10,"messages":[{"role":"user","content":"hello"}]}).to_string())).unwrap();
        let response = settings.scope(app.oneshot(request)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let count = wire["input_tokens"].as_u64().unwrap();
        assert!(count > 0);
        let state = monitor.snapshot();
        assert_eq!(state.recent[0].input_tokens, Some(count));
        assert_eq!(
            state.recent[0].usage.input_tokens.unwrap().provenance,
            crate::monitor::UsageProvenance::Estimated
        );
        assert_eq!(state.sessions[0].input_tokens, 0);
        assert_eq!(state.sessions[0].output_tokens, 0);
    }

    struct CountStall {
        released: std::sync::Mutex<bool>,
        changed: std::sync::Condvar,
        started: std::sync::atomic::AtomicUsize,
    }

    impl CountStall {
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.changed.notify_all();
        }
    }

    struct ReleaseCountStall(Arc<CountStall>);
    impl Drop for ReleaseCountStall {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    async fn count_and_stream_mixed_load(phase: &'static str) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            (
                "CCP_CONFIG_DIR".into(),
                dir.path().to_string_lossy().into_owned(),
            ),
            (
                "XDG_STATE_HOME".into(),
                dir.path().to_string_lossy().into_owned(),
            ),
            ("CCP_CODEX_TRANSPORT".into(), "http".into()),
            ("CCP_CODEX_MODEL".into(), "gpt-6-sol".into()),
        ]));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let events = upstream_sse(&[
            serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}),
            serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"responsive"}),
            serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m","content":[{"type":"output_text","text":"responsive"}]}}),
            serde_json::json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":7,"output_tokens":2}}}),
        ]);
        let peer_app = axum::Router::new().fallback(axum::routing::post(move || {
            let events = events.clone();
            async move { ([("content-type", "text/event-stream")], events) }
        }));
        let peer = tokio::spawn(async move {
            axum::serve(listener, peer_app).await.unwrap();
        });
        let stall = Arc::new(CountStall {
            released: std::sync::Mutex::new(false),
            changed: std::sync::Condvar::new(),
            started: AtomicUsize::new(0),
        });
        let _release = ReleaseCountStall(stall.clone());
        let blocked = stall.clone();
        let worker = std::thread::current().id();
        let provider: Arc<dyn Provider> = Arc::new(CodexProvider {
            client: authenticated_live_test_client(base_url),
            counting_gate: CountingGate::new(),
            count_hook: Some(Arc::new(move |at| {
                assert_ne!(std::thread::current().id(), worker, "{at} is inline");
                if at == phase {
                    blocked.started.fetch_add(1, Ordering::SeqCst);
                    let released = blocked.released.lock().unwrap();
                    let (released, timeout) = blocked
                        .changed
                        .wait_timeout_while(released, Duration::from_secs(15), |done| !*done)
                        .unwrap();
                    assert!(
                        *released && !timeout.timed_out(),
                        "test did not release count job"
                    );
                }
            })),
        });
        let registry = Arc::new(registry::Registry::from_providers(
            config::AliasProvider::Codex,
            [provider],
        ));
        let app =
            crate::server::app_with_features(registry, None, crate::server::AppFeatures::default());
        let payload = serde_json::json!({"model":"gpt-6-sol", "max_tokens":10, "messages":[{"role":"user","content":"Unicode 汉字 function f(a){return a+1;} ".repeat(4096)}]}).to_string();
        let count_request = |app: axum::Router| {
            let payload = payload.clone();
            let settings = settings.clone();
            tokio::spawn(settings.scope(async move {
                app.oneshot(
                    http::Request::builder()
                        .method("POST")
                        .uri("/v1/messages/count_tokens")
                        .header("content-type", "application/json")
                        .body(Body::from(payload))
                        .unwrap(),
                )
                .await
                .unwrap()
            }))
        };
        let mut tasks = vec![count_request(app.clone()), count_request(app.clone())];
        tokio::time::timeout(Duration::from_secs(2), async {
            while stall.started.load(Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("two count jobs must start");
        for _ in 0..4 {
            tasks.push(count_request(app.clone()));
        }
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        for _ in 0..2 {
            let response = tokio::time::timeout(Duration::from_secs(1), count_request(app.clone()))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(response.headers().get("retry-after").is_none());
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(error["error"]["type"], "overloaded_error");
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("Local token-counting CPU")
            );
        }
        let health = tokio::time::timeout(
            Duration::from_secs(1),
            app.clone().oneshot(
                http::Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let stream_request = http::Request::builder().method("POST").uri("/v1/messages").header("content-type", "application/json").body(Body::from(serde_json::json!({"model":"gpt-6-sol","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hello"}]}).to_string())).unwrap();
        let stream = tokio::time::timeout(
            Duration::from_secs(2),
            settings.clone().scope(app.clone().oneshot(stream_request)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(stream.status(), StatusCode::OK);
        let bytes = tokio::time::timeout(
            Duration::from_secs(2),
            axum::body::to_bytes(stream.into_body(), usize::MAX),
        )
        .await
        .unwrap()
        .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("responsive"));
        assert!(text.contains("\"input_tokens\":7"));
        assert_eq!(stall.started.load(Ordering::SeqCst), 2);
        // Cancel both executing callers: their jobs must still own capacity.
        tasks[0].abort();
        tasks[1].abort();
        let response = count_request(app.clone()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        // A cancelled waiter frees admission without spawning a phantom job.
        tasks[2].abort();
        let _ = (&mut tasks[2]).await;
        let replacement = count_request(app.clone());
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        assert!(!replacement.is_finished());
        assert_eq!(stall.started.load(Ordering::SeqCst), 2);
        stall.release();
        for task in tasks.into_iter().skip(3).chain([replacement]) {
            let response = tokio::time::timeout(Duration::from_secs(10), task)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        assert_eq!(stall.started.load(Ordering::SeqCst), 6);
        peer.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn count_and_stream_mixed_load_stalled_translation() {
        count_and_stream_mixed_load("translation").await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn count_and_stream_mixed_load_stalled_tokenizer() {
        count_and_stream_mixed_load("tokenizer").await;
    }

    fn authenticated_live_test_client(base_url: String) -> Arc<CodexHttpClient> {
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            base_url,
            1_000,
            1_000,
        )
        .with_early_reasoning();
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "test".to_string(),
                refresh: String::new(),
                expires: u64::MAX,
                account_id: Some("acct".to_string()),
            });
        Arc::new(client)
    }

    async fn next_live_websocket_request(
        websocket: &mut tokio_tungstenite::WebSocketStream<TcpStream>,
    ) -> serde_json::Value {
        loop {
            match websocket.next().await {
                Some(Ok(Message::Ping(payload))) => {
                    websocket.send(Message::Pong(payload)).await.unwrap();
                }
                Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).unwrap(),
                other => panic!("unexpected WebSocket request frame: {other:?}"),
            }
        }
    }

    async fn emit_live_event(
        websocket: &mut tokio_tungstenite::WebSocketStream<TcpStream>,
        event: &serde_json::Value,
    ) {
        websocket
            .send(Message::Text(event.to_string()))
            .await
            .unwrap();
    }

    fn upstream_sse(events: &[serde_json::Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for event in events {
            bytes.extend_from_slice(format!("data: {event}\n\n").as_bytes());
        }
        bytes
    }

    #[test]
    fn nonzero_status_without_detail_preserves_message() {
        for detail in [None, Some("".to_string()), Some("  ".to_string())] {
            let error = client::CodexError {
                detail,
                ..client::CodexError::new(502, "synthetic failure".to_string())
            };
            assert_eq!(codex_error_message(&error), "synthetic failure");
        }
    }

    #[tokio::test]
    async fn cross_path_error_types_are_consistent() {
        for (status, want) in [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (413, "request_too_large"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (529, "overloaded_error"),
            (502, "api_error"),
        ] {
            let error = client::CodexError::new(status, "synthetic failure".to_string());
            assert_eq!(codex_stream_error_type(&error), want, "{status}");
            let response = map_codex_error_to_response(&error);
            assert_eq!(response.status().as_u16(), status);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["type"], want, "{status}");
            assert_eq!(value["error"]["message"], "synthetic failure", "{status}");
            let upstream = upstream_sse(&[
                serde_json::json!({"type":"error","status":status,"message":"synthetic failure"}),
            ]);
            let error =
                translate::accumulate::accumulate_response(&upstream, "msg_1", "gpt-6.1-sol")
                    .unwrap_err();
            let response = map_codex_translation_error_to_response(&error);
            assert_eq!(response.status().as_u16(), status);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["type"], want, "buffered {status}");
            assert_eq!(
                value["error"]["message"], "synthetic failure",
                "buffered {status}"
            );
        }
    }

    #[tokio::test]
    async fn task3_live_skip_only_terminal_returns_without_retry() {
        use http_body_util::BodyExt as _;
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        for event in [
            serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_call","id":"skip"}}),
            serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_call","id":"skip"}}),
            serde_json::json!({"type":"response.completed","response":{"status":"completed"}}),
        ] {
            tx.send(Ok(event)).await.unwrap();
        }
        drop(tx);
        let (rx, _) = websocket::CodexWebSocketEventStream::pending(rx);
        let request = live_test_request("fixture");
        let response = match live_stream_response_once(
            rx,
            "m".into(),
            "model",
            live_test_context("task3-skips"),
            ContinuationReservation::for_owner_turn(None, None),
            request,
            1,
            config::CodexTransport::Http,
            false,
        )
        .await
        {
            LiveStreamStart::Response(response) => response,
            LiveStreamStart::Retry { .. } => {
                panic!("successful skip-only terminal must not reissue upstream")
            }
        };
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("message_stop"));
        assert!(!text.contains("content_block_start"));
    }

    #[test]
    fn task3_skip_only_success_does_not_retry_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_call","id":"skip"}}),
            serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_call","id":"skip"}}),
            serde_json::json!({"type":"response.completed","response":{"status":"completed"}}),
        ]);
        assert!(translate::accumulate::accumulate_response(&body, "m", "model").is_ok());
        assert!(!is_empty_codex_success_completion(&body));
        let no_terminal = upstream_sse(&[serde_json::json!({"type":"response.future.progress"})]);
        assert!(translate::accumulate::accumulate_response(&no_terminal, "m", "model").is_err());
        let failed = upstream_sse(&[
            serde_json::json!({"type":"response.future.progress"}),
            serde_json::json!({"type":"response.failed","response":{"status":"failed","error":{"message":"failure"}}}),
        ]);
        assert!(translate::accumulate::accumulate_response(&failed, "m", "model").is_err());
    }

    #[test]
    fn terminal_only_completed_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "incomplete_details": null, "usage": {"input_tokens": 5, "output_tokens": 0}}
        })]);
        assert!(is_empty_codex_success_completion(&body));
    }

    #[test]
    fn terminal_only_done_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.done",
            "response": {"id": "resp_1", "usage": {}}
        })]);
        assert!(is_empty_codex_success_completion(&body));
    }

    #[test]
    fn empty_message_item_is_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"}
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(is_empty_codex_success_completion(&body));
    }

    #[test]
    fn upstream_with_text_is_not_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"}
            }),
            serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "hello"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(!is_empty_codex_success_completion(&body));
    }

    #[test]
    fn upstream_with_tool_call_is_not_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(!is_empty_codex_success_completion(&body));
    }

    #[test]
    fn terminal_only_incomplete_upstream_is_not_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.incomplete",
            "response": {"id": "resp_1", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {}}
        })]);
        assert!(!is_empty_codex_success_completion(&body));
    }

    #[test]
    fn upstream_without_terminal_event_is_not_empty_completion() {
        assert!(!is_empty_codex_success_completion(&upstream_sse(&[])));
    }

    fn request_with_tail(messages: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "gpt-5.6-luna",
            "messages": messages
        }))
        .unwrap()
    }

    #[test]
    fn successful_tool_result_tail_is_detected() {
        let body = request_with_tail(serde_json::json!([
            {"role":"user","content":"ping"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"reply","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sent"}]}
        ]));
        assert!(tail_is_successful_tool_result(&body));
    }

    #[test]
    fn error_tool_result_tail_is_excluded() {
        let body = request_with_tail(serde_json::json!([
            {"role":"user","content":"ping"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"run","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"boom","is_error":true}]}
        ]));
        assert!(!tail_is_successful_tool_result(&body));
    }

    #[test]
    fn non_tool_tails_are_excluded() {
        // plain-text user tail
        let text = request_with_tail(serde_json::json!([{"role":"user","content":"ping"}]));
        assert!(!tail_is_successful_tool_result(&text));
        // assistant tail
        let assistant = request_with_tail(serde_json::json!([
            {"role":"user","content":"ping"},
            {"role":"assistant","content":[{"type":"text","text":"pong"}]}
        ]));
        assert!(!tail_is_successful_tool_result(&assistant));
        // tool_result present but NOT the last block
        let not_last = request_with_tail(serde_json::json!([
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"sent"},
                {"type":"text","text":"and also this"}
            ]}
        ]));
        assert!(!tail_is_successful_tool_result(&not_last));
        // empty conversation
        let empty = request_with_tail(serde_json::json!([]));
        assert!(!tail_is_successful_tool_result(&empty));
    }

    #[test]
    fn tool_tail_requires_complete_successful_matching_results() {
        let calls = serde_json::json!({"role":"assistant","content":[
            {"type":"tool_use","id":"a","name":"reply","input":{}},
            {"type":"tool_use","id":"b","name":"reply","input":{}}
        ]});
        let results = serde_json::json!([
            {"type":"tool_result","tool_use_id":"a","content":"sent"},
            {"type":"tool_result","tool_use_id":"b","content":"sent"}
        ]);
        let make = |content| {
            request_with_tail(serde_json::json!([
                calls.clone(), {"role":"user","content":content}
            ]))
        };
        assert!(tail_is_successful_tool_result(&make(results.clone())));
        let mut whitespace = results.clone();
        whitespace
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"type":"text","text":" \n\t"}));
        assert!(tail_is_successful_tool_result(&make(whitespace)));
        let mut cases = vec![
            serde_json::json!([results[0]]),
            serde_json::json!([results[0], results[0]]),
        ];
        for key in ["is_error", "tool_use_id"] {
            let mut invalid = results.clone();
            invalid[0][key] = if key == "is_error" {
                serde_json::json!(true)
            } else {
                serde_json::json!("unknown")
            };
            cases.push(invalid);
        }
        for text in [
            "continue working",
            "<system-reminder>PostToolUse hook: continue</system-reminder>",
        ] {
            let mut invalid = results.clone();
            invalid
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({"type":"text","text":text}));
            cases.push(invalid);
        }
        for case in cases {
            assert!(
                !tail_is_successful_tool_result(&make(case.clone())),
                "{case}"
            );
        }
        for role in ["user", "system"] {
            let mut body = make(results.clone());
            let trailing = request_with_tail(
                serde_json::json!([{"role":role,"content":"PostToolUse hook: continue"}]),
            );
            body.messages.extend(trailing.messages);
            assert!(!tail_is_successful_tool_result(&body));
        }
    }

    #[tokio::test]
    async fn empty_acceptance_requires_explicit_completed_status() {
        for status in [
            serde_json::json!("cancelled"),
            serde_json::json!("in_progress"),
            serde_json::Value::Null,
        ] {
            let (tx, rx, ctx, request_body) = empty_terminal_fixture();
            let payload = serde_json::json!({
                "type":"response.completed", "response":{"id":"r", "status":status,"output":[]}
            });
            assert!(!buffered_terminal_is_completed(&upstream_sse(
                std::slice::from_ref(&payload)
            )));
            tx.send(Ok(payload)).await.unwrap();
            match live_stream_response_once(
                rx,
                "msg_test".into(),
                "gpt-5.6-sol",
                ctx,
                ContinuationReservation::for_owner_turn(None, None),
                request_body,
                0,
                config::CodexTransport::WebSocket,
                true,
            )
            .await
            {
                LiveStreamStart::Retry {
                    confirmed_empty_success,
                    ..
                } => assert!(!confirmed_empty_success),
                LiveStreamStart::Response(_) => {
                    panic!("non-completed status must not end the turn")
                }
            }
        }
        assert!(!buffered_terminal_is_completed(&upstream_sse(&[
            serde_json::json!({
                "type":"response.done", "response":{"output":[]}
            })
        ])));
    }

    #[tokio::test]
    async fn accepting_empty_does_not_accept_failed_or_incomplete_terminals() {
        for status in ["failed", "incomplete"] {
            let (tx, rx, ctx, request_body) = empty_terminal_fixture();
            let payload = serde_json::json!({
                "type":"response.done", "response":{"id":"r", "status":status,"output":[]}
            });
            assert!(!is_empty_codex_success_completion(&upstream_sse(
                std::slice::from_ref(&payload)
            )));
            tx.send(Ok(payload)).await.unwrap();
            match live_stream_response_once(
                rx,
                "msg_test".into(),
                "gpt-5.6-sol",
                ctx,
                ContinuationReservation::for_owner_turn(None, None),
                request_body,
                0,
                config::CodexTransport::WebSocket,
                true,
            )
            .await
            {
                LiveStreamStart::Retry { error, .. } => assert!(!is_empty_completion_error(&error)),
                LiveStreamStart::Response(response) => assert!(!response.status().is_success()),
            }
        }
    }

    #[test]
    fn empty_success_sequence_requires_confirmed_consecutive_completions() {
        assert_eq!(advance_empty_success_sequence(1, false, false), 0);
        assert_eq!(advance_empty_success_sequence(1, true, true), 1);
        assert_eq!(advance_empty_success_sequence(1, true, false), 2);
        assert_eq!(advance_empty_success_sequence(0, true, false), 1);
        assert_eq!(advance_empty_success_sequence(2, false, true), 0);
    }

    #[test]
    fn empty_completion_is_accepted_only_after_one_retry_and_only_with_the_tail() {
        // Require a full-context retry before permitting empty success.
        assert!(
            !accept_empty_completion(true, 0),
            "first empty must retry once"
        );
        assert!(
            accept_empty_completion(true, 1),
            "second consecutive empty is the turn's end"
        );
        assert!(accept_empty_completion(true, 2));
        assert!(
            !accept_empty_completion(false, 0),
            "no tool tail: empty completion is not eligible"
        );
        assert!(
            !accept_empty_completion(false, 5),
            "no tool tail: never accepted"
        );
    }

    fn empty_terminal_fixture() -> (
        tokio::sync::mpsc::Sender<Result<serde_json::Value, client::CodexError>>,
        websocket::CodexWebSocketEventStream,
        RequestContext,
        translate::request::ResponsesRequest,
    ) {
        let body = request_with_tools(serde_json::json!([]));
        let request_body = translate_request(
            &body,
            TranslateOptions {
                session_id: None,
                service_tier: None,
                model: "gpt-5.6-sol".to_string(),
                use_responses_lite: true,
            },
        )
        .unwrap();
        let ctx = RequestContext {
            req_id: "empty-terminal".to_string(),
            session_id: None,
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        };
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let (rx, _) = websocket::CodexWebSocketEventStream::pending(rx);
        (tx, rx, ctx, request_body)
    }

    #[tokio::test]
    async fn empty_terminal_without_accept_flag_still_retries() {
        let (tx, rx, ctx, request_body) = empty_terminal_fixture();
        tx.send(Ok(serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}}
        })))
        .await
        .unwrap();
        let continuation = ContinuationReservation::for_owner_turn(None, None);
        match live_stream_response_once(
            rx,
            "msg_test".to_string(),
            "claude-opus-4-8",
            ctx,
            continuation,
            request_body,
            0,
            config::CodexTransport::WebSocket,
            false,
        )
        .await
        {
            LiveStreamStart::Retry { error, .. } => {
                assert_eq!(error.detail.as_deref(), Some(EMPTY_CODEX_COMPLETION_DETAIL));
            }
            LiveStreamStart::Response(_) => {
                panic!("an empty completion without the accept flag must stay a retry")
            }
        }
    }

    #[tokio::test]
    async fn empty_terminal_with_accept_flag_is_a_valid_response() {
        use http_body_util::BodyExt as _;
        let (tx, rx, ctx, request_body) = empty_terminal_fixture();
        tx.send(Ok(serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}}
        })))
        .await
        .unwrap();
        let continuation = ContinuationReservation::for_owner_turn(None, None);
        let response = match live_stream_response_once(
            rx,
            "msg_test".to_string(),
            "claude-opus-4-8",
            ctx,
            continuation,
            request_body,
            0,
            config::CodexTransport::WebSocket,
            true,
        )
        .await
        {
            LiveStreamStart::Response(response) => response,
            LiveStreamStart::Retry { error, .. } => {
                panic!(
                    "accept flag set: the empty completion must be delivered, got retry: {error}"
                )
            }
        };
        let mut collected = Vec::new();
        let mut body = response.into_body();
        while let Ok(Some(frame)) =
            tokio::time::timeout(Duration::from_millis(300), body.frame()).await
        {
            if let Ok(data) = frame.unwrap().into_data() {
                collected.extend_from_slice(&data);
            }
        }
        let sse = String::from_utf8(collected).unwrap();
        assert!(
            sse.contains("event: message_stop"),
            "delivered stream must terminate properly, got: {sse:?}"
        );
    }

    fn request_with_tools(tools: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": tools
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn task4_invalid_registration_is_local_before_regular_or_standalone_upstream() {
        let dir = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            ("CCP_CONFIG_DIR".into(), dir.path().display().to_string()),
            ("XDG_STATE_HOME".into(), dir.path().display().to_string()),
            ("CCP_CODEX_TRANSPORT".into(), "http".into()),
        ]));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client =
            authenticated_live_test_client(format!("http://{}", listener.local_addr().unwrap()));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peer_calls = calls.clone();
        let app = axum::Router::new().fallback(axum::routing::post(
            move |uri: axum::http::Uri,
                  headers: axum::http::HeaderMap,
                  axum::Json(body): axum::Json<serde_json::Value>| {
                let calls = peer_calls.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    assert_eq!(uri.path(), "/alpha/search");
                    assert_eq!(headers[http::header::AUTHORIZATION], "Bearer test");
                    assert_eq!(body["model"], "gpt-6-luna");
                    assert!(body.get("reasoning").is_none());
                    assert_eq!(
                        body["commands"]["search_query"][0]["q"],
                        "synthetic forced query"
                    );
                    assert_eq!(
                        body["settings"]["filters"]["allowed_domains"],
                        serde_json::json!(["example.test"])
                    );
                    assert_eq!(body["max_output_tokens"], 2500);
                    axum::Json(serde_json::json!({
                        "encrypted_output":"opaque-provider-must-not-escape",
                        "output":"Actual search answer",
                        "results":[{"url":"https://example.test/source","title":"Fixture"}]
                    }))
                }
            },
        ));
        let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = CodexProvider {
            client,
            counting_gate: CountingGate::new(),
            count_hook: None,
        };
        for (tools, choice) in [
            (
                serde_json::json!([{"name":"ToolSearch","input_schema":{"type":"object"}}]),
                serde_json::json!({"type":"tool","name":"ToolSearch"}),
            ),
            (
                serde_json::json!([{"type":"web_fetch_20250910","name":"native"}]),
                serde_json::json!({"type":"auto"}),
            ),
            (
                serde_json::json!([{"type":"web_search_20250305","name":"web_search"},{"type":"web_fetch_20250910","name":"native"}]),
                serde_json::json!({"type":"tool","name":"web_search"}),
            ),
        ] {
            let request = serde_json::from_value(serde_json::json!({
                "model":"gpt-6-sol","messages":[{"role":"user","content":"synthetic query"}],
                "tools":tools,"tool_choice":choice
            }))
            .unwrap();
            let response = settings
                .clone()
                .scope(provider.handle_messages(request, live_test_context("task4-invalid")))
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        let request = serde_json::from_value(serde_json::json!({
            "model":"gpt-6-luna", "stream":true,
            "messages":[{"role":"user","content":"Perform a web search for the query: synthetic forced query"}],
            "tools":[{"type":"web_search_20250305","name":"web_search","allowed_domains":["example.test"]}],
            "tool_choice":{"type":"tool","name":"web_search"}
        })).unwrap();
        let mut context = live_test_context("task5-forced-search");
        context.req_id = "task5-forced-search".into();
        let response = settings
            .clone()
            .scope(provider.handle_messages(request, context))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("Actual search answer"), "{text}");
        assert!(
            text.contains("[Fixture](https://example.test/source)"),
            "{text}"
        );
        assert!(text.contains(r#""type":"server_tool_use""#), "{text}");
        assert!(text.contains("web_search_tool_result"), "{text}");
        assert!(!text.contains("opaque-provider-must-not-escape"), "{text}");
        assert_eq!(text.matches("event: message_start").count(), 1);
        assert_eq!(text.matches("event: message_stop").count(), 1);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "no additional Responses generation"
        );
        assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
        let path = settings.clone().sync_scope(crate::logging::log_file);
        let logs = std::fs::read_to_string(path).unwrap();
        let records: Vec<serde_json::Value> = logs
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|record: &serde_json::Value| record["fields"]["reqId"] == "task5-forced-search")
            .collect();
        let skips: Vec<_> = records
            .iter()
            .filter(|r| r["msg"] == "codex_event_skipped")
            .collect();
        assert_eq!(skips.len(), 1);
        assert_eq!(
            skips[0]["fields"]["event_type"],
            "standalone_search.native_pair"
        );
        assert_eq!(
            skips[0]["fields"]["reason"],
            "anthropic_search_opaque_replay_unavailable"
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| r["msg"] == "codex_event_skipped_summary")
                .count(),
            1
        );
        assert!(!logs.contains("opaque-provider-must-not-escape"));
        peer.abort();
    }

    #[test]
    fn web_search_requests_leave_lite_lane_and_upgrade_luna() {
        let body = request_with_tools(serde_json::json!([
            {"type":"web_search_20250305", "name":"web_search"}
        ]));
        for (resolved, expected) in [
            ("gpt-5.6-luna", "gpt-5.6-sol"),
            ("gpt-5.6-sol", "gpt-5.6-sol"),
            ("gpt-5.6-terra", "gpt-5.6-terra"),
            ("gpt-6-luna", "gpt-6-sol"),
            ("gpt-6-sol", "gpt-6-sol"),
            ("gpt-6.1-terra", "gpt-6.1-terra"),
            ("gpt-5.4", "gpt-5.4"),
        ] {
            for full_lane in [false, true] {
                let mut model = resolved.to_string();
                let lite = apply_model_lane_for_request(&mut model, &body, full_lane);
                assert!(!lite, "{resolved} with web_search must use the full lane");
                assert_eq!(model, expected);
            }
        }
    }

    #[test]
    fn compact_with_unused_hosted_search_keeps_model_and_lane() {
        let mut body = request_with_tools(serde_json::json!([
            {"type":"web_search_20250305", "name":"web_search"}
        ]));
        body.extra.insert(
            "system".into(),
            serde_json::json!(
                "You are a helpful AI assistant tasked with summarizing conversations."
            ),
        );
        for (resolved, full_lane, expected_lite) in [
            ("gpt-5.6-luna", false, true),
            ("gpt-5.6-luna", true, true),
            ("gpt-5.6-sol", false, true),
            ("gpt-5.6-sol", true, false),
            ("gpt-5.4", true, false),
        ] {
            let mut model = resolved.to_string();
            assert_eq!(
                apply_model_lane_for_request(&mut model, &body, full_lane),
                expected_lite
            );
            assert_eq!(model, resolved);
        }
    }

    #[test]
    fn requests_without_web_search_keep_model_and_lite_lane() {
        let body = request_with_tools(serde_json::json!([
            {"name":"Bash", "input_schema":{}}
        ]));
        for (resolved, lite_expected) in [
            ("gpt-5.6-luna", true),
            ("gpt-5.6-sol", true),
            ("gpt-6-luna", true),
            ("gpt-6-sol", true),
            ("gpt-6.1-terra", true),
            ("gpt-5.4", false),
        ] {
            let mut model = resolved.to_string();
            let lite = apply_model_lane_for_request(&mut model, &body, false);
            assert_eq!(model, resolved, "model must not change without web_search");
            assert_eq!(lite, lite_expected);
        }
    }

    #[test]
    fn full_lane_translation_preserves_parallel_tool_restrictions() {
        for model in [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6.1-sol",
            "gpt-6.1-terra",
            "gpt-5.6-luna",
            "gpt-6-luna",
        ] {
            for full_lane in [false, true] {
                for disable_parallel in [false, true] {
                    let mut body = request_with_tools(serde_json::json!([
                        {"name":"Read", "input_schema":{"type":"object"}}
                    ]));
                    body.extra.insert("tool_choice".to_string(),
                        serde_json::json!({"type":"auto", "disable_parallel_tool_use":disable_parallel}),
                    );
                    let mut resolved = model.to_string();
                    let lite = apply_model_lane_for_request(&mut resolved, &body, full_lane);
                    let translated = translate_request(
                        &body,
                        TranslateOptions {
                            session_id: None,
                            service_tier: None,
                            model: resolved,
                            use_responses_lite: lite,
                        },
                    )
                    .unwrap();
                    assert_eq!(translated.model, model);
                    assert_eq!(translated.client_metadata.is_some(), lite);
                    assert_eq!(translated.parallel_tool_calls, !lite && !disable_parallel);
                    assert_eq!(translated.tools.is_some(), !lite);
                }
            }
        }
    }

    #[test]
    fn terra_translation_preserves_model_and_tier_across_lanes() {
        for requested in ["gpt-6.1-terra", "gpt-6.1-terra-fast"] {
            for hosted_search in [false, true] {
                let tools = if hosted_search {
                    serde_json::json!([{"type":"web_search_20250305", "name":"web_search"}])
                } else {
                    serde_json::json!([])
                };
                let mut body = request_with_tools(tools);
                body.model = Some(requested.to_string());
                let mut resolved = resolve_model_request_with_config_override(requested, false);
                let lite = apply_model_lane_for_request(&mut resolved.model, &body, false);
                let translated = translate_request(
                    &body,
                    TranslateOptions {
                        session_id: None,
                        service_tier: resolved.service_tier,
                        model: resolved.model,
                        use_responses_lite: lite,
                    },
                )
                .unwrap();
                assert_eq!(translated.model, "gpt-6.1-terra");
                assert_eq!(translated.client_metadata.is_some(), !hosted_search);
                assert_eq!(
                    translated.service_tier,
                    requested
                        .ends_with("-fast")
                        .then_some(ServiceTier::Priority)
                );
            }
        }
    }

    #[test]
    fn generation_timing_ignores_control_events() {
        assert!(!codex_generation_event(&serde_json::json!({
            "type": "codex.rate_limits"
        })));
        assert!(!codex_generation_event(&serde_json::json!({
            "type": "keepalive"
        })));
        assert!(codex_generation_event(&serde_json::json!({
            "type": "response.created"
        })));
    }

    #[test]
    fn estimate321_then_actual300_reports300() {
        let monitor = crate::monitor::MonitorHandle::new(10);
        monitor.request_started(
            "usage",
            Some("session".into()),
            None,
            crate::monitor::EndpointKind::Messages,
        );
        let ctx = RequestContext {
            req_id: "usage".into(),
            session_id: Some("session".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: Some(monitor.clone()),
        };
        let mut translator = LiveStreamTranslator::with_estimated_input_tokens("msg", "model", 321);
        let initial = translator.ping_chunk(None);
        record_live_stream_progress(&ctx, &initial, translator.take_progress());
        assert_eq!(
            monitor.snapshot().sessions[0].input_tokens,
            0,
            "estimate entered authoritative totals"
        );
        let terminal = translator.accept(&serde_json::json!({"type":"response.completed", "response":{"status":"completed", "usage":{"input_tokens":300,"output_tokens":48}}}), None).unwrap();
        record_live_stream_progress(&ctx, &terminal, translator.take_progress());
        assert_eq!(monitor.snapshot().active[0].input_tokens, Some(300));
        assert_eq!(monitor.snapshot().sessions[0].input_tokens, 300);
    }

    #[test]
    fn live_stream_progress_records_terminal_usage() {
        let monitor = crate::monitor::MonitorHandle::new(10);
        monitor.request_started(
            "request",
            None,
            None,
            crate::monitor::EndpointKind::Messages,
        );
        let ctx = RequestContext {
            req_id: "request".to_string(),
            session_id: None,
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: Some(monitor.clone()),
        };

        let mut translator = LiveStreamTranslator::new("msg", "model");
        let chunk = translator.accept(&serde_json::json!({"type":"response.completed", "response":{"status":"completed", "usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":8},"output_tokens":48}}}), None).unwrap();
        record_live_stream_progress(&ctx, &chunk, translator.take_progress());

        let state = monitor.snapshot();
        assert_eq!(state.active[0].input_tokens, Some(12));
        assert_eq!(state.active[0].usage.cached_input_tokens, Some(8));
        assert_eq!(state.sessions[0].input_tokens, 12);
        assert_eq!(state.sessions[0].cached_input_tokens, Some(8));
        assert_eq!(state.active[0].output_tokens, Some(48));
    }

    // Exercise the real transport gate, provider translator, body, and state cleanup.
    async fn assert_live_protocol_rejection(
        events: Vec<serde_json::Value>,
        pre_output: bool,
        websocket: bool,
    ) {
        use http_body_util::BodyExt as _;
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let session = format!("task5-protocol-{websocket}-{pre_output}");
        let owner = ConversationIdentity::Main(session.clone());
        let request = live_test_request("hello");
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            if websocket {
                let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                let _ = next_live_websocket_request(&mut ws).await;
                for event in events {
                    if ws.send(Message::Text(event.to_string())).await.is_err() {
                        break;
                    }
                }
                let _ = ws.close(None).await;
            } else {
                let mut request = [0_u8; 8192];
                assert!(socket.read(&mut request).await.unwrap() > 0);
                socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await.unwrap();
                for event in events {
                    let event = format!("data: {event}\n\n");
                    let chunk = format!("{:x}\r\n{event}\r\n", event.len());
                    if socket.write_all(chunk.as_bytes()).await.is_err() {
                        break;
                    }
                }
                let _ = socket.write_all(b"0\r\n\r\n").await;
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(200), listener.accept())
                    .await
                    .is_err(),
                "local protocol rejection must make exactly one attempt"
            );
        });
        let response = tokio::time::timeout(
            Duration::from_secs(3),
            live_stream_response(
                authenticated_live_test_client(format!("http://{address}/responses")),
                "msg".into(),
                &request.model.clone(),
                live_test_context(&session),
                request,
                reservation.clone(),
                if websocket {
                    config::CodexTransport::WebSocket
                } else {
                    config::CodexTransport::Http
                },
                false,
            ),
        )
        .await
        .expect("protocol rejection must not enter retry backoff");
        if pre_output {
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            assert_eq!(
                response
                    .headers()
                    .get("x-should-retry")
                    .and_then(|v| v.to_str().ok()),
                Some("false")
            );
        } else {
            assert_eq!(response.status(), StatusCode::OK);
        }
        let bytes = tokio::time::timeout(Duration::from_secs(3), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        let body = String::from_utf8_lossy(&bytes);
        let errors: Vec<serde_json::Value> = if pre_output {
            vec![serde_json::from_slice(&bytes).unwrap()]
        } else {
            parse_sse_events(&bytes)
                .into_iter()
                .filter_map(|event| serde_json::from_str(&event.data).ok())
                .collect()
        };
        assert!(
            errors
                .iter()
                .any(|event| event["type"] == "error" && event["error"]["type"] == "api_error"),
            "expected explicit api_error: {body}"
        );
        assert!(
            !body.contains("message_stop"),
            "must not emit a successful ending: {body}"
        );
        assert!(!body.contains("\"stop_reason\":\"tool_use\""));
        assert!(!body.contains("\"stop_reason\":\"end_turn\""));
        server.await.unwrap();
        assert!(!continuation::is_current_turn_for_owner(&reservation));
        assert!(!continuation::has_continuation_for_owner_for_tests(&owner));
        assert!(websocket::pooled_socket_id_for_tests(&owner).is_none());
    }

    #[tokio::test]
    async fn local_protocol_open_outputs_fail_through_http_and_ws_provider() {
        let _registry = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = websocket::lock_codex_websocket_pool_for_tests().await;
        for websocket in [false, true] {
            for events in [
                vec![
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"Read","call_id":"c"}}),
                    serde_json::json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"file_path\":\"/tmp/a\""}),
                    serde_json::json!({"type":"response.completed","response":{"id":"r","status":"completed"}}),
                ],
                vec![
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}),
                    serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"unfinished"}),
                    serde_json::json!({"type":"response.completed","response":{"id":"r","status":"completed"}}),
                ],
                vec![
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}),
                    serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"visible"}),
                    serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[{"type":"output_audio"}]}}),
                    serde_json::json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","name":"Bash","call_id":"c"}}),
                    serde_json::json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","name":"Bash","call_id":"c","arguments":"{}"}}),
                ],
            ] {
                assert_live_protocol_rejection(events, false, websocket).await;
            }
        }
    }

    #[tokio::test]
    async fn local_protocol_identity_guard_is_nonretryable_through_http_and_ws_provider() {
        let _registry = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = websocket::lock_codex_websocket_pool_for_tests().await;
        for websocket in [false, true] {
            assert_live_protocol_rejection(vec![
                serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"x".repeat(4097)}}),
                serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"hello"}),
            ], true, websocket).await;
        }
    }

    #[tokio::test]
    async fn structural_overflow_is_nonretryable_through_http_and_ws_provider() {
        use http_body_util::BodyExt as _;
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        for websocket in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (response_done_tx, response_done_rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                if websocket {
                    let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                    let _ = next_live_websocket_request(&mut ws).await;
                    for index in 0..1025 {
                        let event = serde_json::json!({"type":"response.output_item.added","output_index":index,"item":{"type":"message"}});
                        if ws.send(Message::Text(event.to_string())).await.is_err() {
                            break;
                        }
                    }
                } else {
                    let mut request = [0_u8; 8192];
                    assert!(socket.read(&mut request).await.unwrap() > 0);
                    socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await.unwrap();
                    for index in 0..1025 {
                        let event = format!(
                            "data: {{\"type\":\"response.output_item.added\",\"output_index\":{index},\"item\":{{\"type\":\"message\"}}}}\n\n"
                        );
                        let chunk = format!("{:x}\r\n{event}\r\n", event.len());
                        if socket.write_all(chunk.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                }
                tokio::select! {
                    biased;
                    accepted = listener.accept() => {
                        panic!("resource overflow must not retry: {accepted:?}");
                    }
                    done = response_done_rx => done.expect("response task dropped"),
                }
            });
            let client = authenticated_live_test_client(format!("http://{address}/responses"));
            let request = live_test_request("hello");
            // The process-wide WebSocket start gate spaces all tests by one
            // second, so this is a deadlock watchdog, not a latency assertion.
            let response = tokio::time::timeout(
                Duration::from_secs(120),
                live_stream_response(
                    client,
                    "msg".into(),
                    &request.model.clone(),
                    live_test_context("task5-overflow"),
                    request,
                    ContinuationReservation::for_owner_turn(None, None),
                    if websocket {
                        config::CodexTransport::WebSocket
                    } else {
                        config::CodexTransport::Http
                    },
                    false,
                ),
            )
            .await
            .expect("resource failure response timed out");
            let _ = response_done_tx.send(());
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            assert_eq!(response.headers().get("x-should-retry").unwrap(), "false");
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert!(String::from_utf8_lossy(&bytes).contains("structural event limit exceeded"));
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["type"], "api_error");
            assert!(!String::from_utf8_lossy(&bytes).contains("message_stop"));
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn live_provider_quota_is_recorded_once_and_control_pings_coalesce() {
        use http_body_util::BodyExt as _;
        rate_limits::TEST_RECORD_CALLS.with(|calls| calls.set(0));
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for _ in 0..10_000 {
                tx.send(Ok(
                    serde_json::json!({"type":"codex.rate_limits","rate_limits":{"allowed":true}}),
                ))
                .await
                .unwrap();
            }
            for event in [
                serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}),
                serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"ok"}),
                serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}),
                serde_json::json!({"type":"response.completed","response":{"id":"r"}}),
            ] {
                tx.send(Ok(event)).await.unwrap();
            }
        });
        let (events, _) = websocket::CodexWebSocketEventStream::pending(rx);
        let request = live_test_request("hello");
        let monitor = crate::monitor::MonitorHandle::default();
        let mut ctx = live_test_context("task5-quota");
        monitor.request_started(
            &ctx.req_id,
            ctx.session_id.clone(),
            None,
            crate::monitor::EndpointKind::Messages,
        );
        ctx.monitor = Some(monitor.clone());
        let response = match live_stream_response_once(
            events,
            "msg".into(),
            &request.model.clone(),
            ctx,
            ContinuationReservation::for_owner_turn(None, None),
            request,
            12,
            config::CodexTransport::Http,
            false,
        )
        .await
        {
            LiveStreamStart::Response(response) => response,
            LiveStreamStart::Retry { .. } => panic!("unexpected retry"),
        };
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let events = parse_sse_events(&bytes);
        let state = monitor.snapshot();
        assert_eq!(
            state.active[0].stream_chunks,
            events.len() as u64,
            "coalesced control metadata must match committed bytes"
        );
        assert_eq!(state.active[0].streamed_bytes, bytes.len() as u64);
        assert_eq!(state.active[0].output_tokens, None);
        assert_eq!(state.sessions[0].input_tokens, 0);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event.as_deref() == Some("ping"))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event.as_deref() == Some("message_start"))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event.as_deref() == Some("content_block_start"))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event.as_deref() == Some("message_stop"))
                .count(),
            1
        );
        rate_limits::TEST_RECORD_CALLS.with(|calls| {
            assert_eq!(
                calls.get(),
                10_000,
                "terminal finalization must not reparse quota events"
            )
        });
    }

    #[tokio::test]
    async fn previous_disabled_portable_summary_survives_ordinary_live_turn() {
        use http_body_util::BodyExt as _;
        use translate::request::{ResponsesContentPart, ResponsesInputItem};

        let session = "task5-compaction-three-turns";
        let summary = "The portable summary preserves all prior context for this test.";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for turn in 0..3 {
                let (socket, _) = listener.accept().await.unwrap();
                let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                let request = next_live_websocket_request(&mut ws).await;
                assert!(request.get("previous_response_id").is_none());
                if turn > 0 {
                    assert!(
                        request["input"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|item| item.to_string().contains(summary))
                    );
                }
                let answer = if turn == 0 {
                    summary
                } else {
                    "ordinary answer"
                };
                for event in [
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}),
                    serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":answer}),
                    serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}),
                    serde_json::json!({"type":"response.completed","response":{"id":format!("r{turn}"),"status":"completed"}}),
                ] {
                    emit_live_event(&mut ws, &event).await;
                }
            }
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let initial = live_test_request("compact the prior context");
        for turn in 0..3 {
            let request = if turn == 0 {
                initial.clone()
            } else {
                let mut logical = live_test_request(summary);
                logical.input.push(ResponsesInputItem::Message {
                    role: "user".to_string(),
                    content: vec![ResponsesContentPart::InputText {
                        text: format!("next {turn}"),
                    }],
                });
                logical
            };
            let response = live_stream_response(
                client.clone(),
                format!("msg{turn}"),
                &request.model.clone(),
                live_test_context(session),
                request,
                ContinuationReservation::for_owner_turn(None, None),
                config::CodexTransport::WebSocket,
                false,
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert!(String::from_utf8_lossy(&bytes).contains("message_stop"));
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn early_reasoning_reaches_body_before_summary_and_stops_retries() {
        use http_body_util::BodyExt as _;
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let _registry = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool = websocket::lock_codex_websocket_pool_for_tests().await;
        for transport in [
            config::CodexTransport::Http,
            config::CodexTransport::WebSocket,
            config::CodexTransport::Auto,
        ] {
            for outcome in [
                "success",
                "error",
                "eof",
                "signature-only",
                "empty-completed",
                "no-replay-text",
            ] {
                let session = format!("early-reasoning-{}-{outcome}", transport.as_str());
                let owner = ConversationIdentity::Main(session.clone());
                let request = live_test_request("hello");
                let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (release, gate) = tokio::sync::oneshot::channel::<()>();
                let (finished, body_done) = tokio::sync::oneshot::channel::<()>();
                let server = tokio::spawn(async move {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let head = vec![
                        serde_json::json!({"type":"response.created","response":{"id":"resp_early"}}),
                        serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"r"}}),
                    ];
                    let tail = match outcome {
                        "eof" => vec![],
                        "error" => vec![
                            serde_json::json!({"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"transient failure after reasoning"}}}),
                        ],
                        "empty-completed" => vec![
                            serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"r","summary":[]}}),
                            serde_json::json!({"type":"response.completed","response":{"id":"resp_early","status":"completed"}}),
                        ],
                        _ => {
                            let mut events = vec![];
                            if outcome == "success" {
                                events.push(serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"Real summary"}));
                            }
                            let reasoning = if outcome == "no-replay-text" {
                                serde_json::json!({"type":"reasoning","id":"r","summary":[]})
                            } else {
                                serde_json::json!({"type":"reasoning","id":"r","encrypted_content":"opaque"})
                            };
                            events.extend([
                                serde_json::json!({"type":"response.output_item.done","output_index":0,"item":reasoning}),
                                serde_json::json!({"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"m"}}),
                                serde_json::json!({"type":"response.output_text.delta","output_index":1,"delta":"Answer"}),
                                serde_json::json!({"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"m"}}),
                                serde_json::json!({"type":"response.completed","response":{"id":"resp_early","status":"completed"}}),
                            ]);
                            events
                        }
                    };
                    if matches!(transport, config::CodexTransport::Http) {
                        let mut request = Vec::new();
                        let mut buf = [0u8; 4096];
                        loop {
                            let n = socket.read(&mut buf).await.unwrap();
                            assert!(n > 0);
                            request.extend_from_slice(&buf[..n]);
                            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                                let headers =
                                    String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                                let length: usize = headers
                                    .lines()
                                    .find_map(|line| line.strip_prefix("content-length:"))
                                    .unwrap()
                                    .trim()
                                    .parse()
                                    .unwrap();
                                if request.len() >= end + 4 + length {
                                    break;
                                }
                            }
                        }
                        socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await.unwrap();
                        for event in head {
                            let data = format!("data: {event}\n\n");
                            socket
                                .write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes())
                                .await
                                .unwrap();
                        }
                        gate.await.unwrap();
                        for event in tail {
                            let data = format!("data: {event}\n\n");
                            socket
                                .write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes())
                                .await
                                .unwrap();
                        }
                        socket.write_all(b"0\r\n\r\n").await.unwrap();
                        drop(socket);
                    } else {
                        let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                        let _ = next_live_websocket_request(&mut ws).await;
                        for event in head {
                            emit_live_event(&mut ws, &event).await;
                        }
                        gate.await.unwrap();
                        for event in tail {
                            emit_live_event(&mut ws, &event).await;
                        }
                        let _ = ws.close(None).await;
                    }
                    tokio::select! {
                        _ = listener.accept() => panic!("no retry after a published reasoning start"),
                        _ = body_done => {},
                    }
                    assert!(
                        tokio::time::timeout(Duration::from_millis(50), listener.accept())
                            .await
                            .is_err()
                    );
                });
                let response = tokio::time::timeout(
                    Duration::from_secs(3),
                    live_stream_response(
                        authenticated_live_test_client(format!("http://{address}/responses")),
                        "msg_early".into(),
                        "gpt-6.1-sol",
                        live_test_context(&session),
                        request,
                        reservation.clone(),
                        transport,
                        false,
                    ),
                )
                .await
                .expect("reasoning start must open body while summary is gated");
                assert_eq!(response.status(), StatusCode::OK);
                let mut body = response.into_body();
                let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap()
                    .into_data()
                    .unwrap();
                let first_text = String::from_utf8_lossy(&first);
                assert!(first_text.contains("\"type\":\"thinking\""), "{first_text}");
                assert!(!first_text.contains("thinking_delta"));
                assert!(!first_text.contains("signature_delta"));
                release.send(()).unwrap();
                let tail = tokio::time::timeout(Duration::from_secs(2), body.collect())
                    .await
                    .unwrap()
                    .unwrap()
                    .to_bytes();
                let mut wire = first.to_vec();
                wire.extend_from_slice(&tail);
                let events = parse_sse_events(&wire);
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e.event.as_deref() == Some("message_start"))
                        .count(),
                    1
                );
                for event in &events {
                    let data: serde_json::Value = serde_json::from_str(&event.data).unwrap();
                    assert_eq!(event.event.as_deref(), data["type"].as_str());
                    assert!(!data["type"].as_str().unwrap().starts_with("response."));
                }
                let text = String::from_utf8_lossy(&wire);
                if matches!(outcome, "error" | "eof" | "empty-completed") {
                    assert!(text.contains("event: error"));
                    assert!(!text.contains("message_stop"));
                    assert!(!text.contains("signature_delta"));
                    assert!(!continuation::has_continuation_for_owner_for_tests(&owner));
                } else {
                    assert!(text.contains("Answer"));
                    assert!(text.contains("message_stop"));
                    assert_eq!(text.matches("event: content_block_start").count(), 2);
                    assert_eq!(
                        text.matches("signature_delta").count(),
                        usize::from(outcome != "no-replay-text")
                    );
                }
                finished.send(()).unwrap();
                server.await.unwrap();
                continuation::clear_continuation_for_owner(Some(&owner));
            }
        }
    }

    #[tokio::test]
    async fn live_stream_response_emits_downstream_frames_before_terminal_event() {
        use http_body_util::BodyExt as _;

        let body = request_with_tools(serde_json::json!([]));
        let request_body = translate_request(
            &body,
            TranslateOptions {
                session_id: None,
                service_tier: None,
                model: "gpt-5.6-sol".to_string(),
                use_responses_lite: true,
            },
        )
        .unwrap();
        let ctx = RequestContext {
            req_id: "incremental-http".to_string(),
            session_id: None,
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        };
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.send(Ok(serde_json::json!({"type": "keepalive"})))
            .await
            .unwrap();
        tx.send(Ok(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_up"}
        })))
        .await
        .unwrap();
        tx.send(Ok(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "first"
        })))
        .await
        .unwrap();

        let (rx, _) = websocket::CodexWebSocketEventStream::pending(rx);
        let continuation = ContinuationReservation::for_owner_turn(None, None);
        let response = match live_stream_response_once(
            rx,
            "msg_test".to_string(),
            "claude-opus-4-8",
            ctx,
            continuation,
            request_body,
            12,
            config::CodexTransport::Http,
            false,
        )
        .await
        {
            LiveStreamStart::Response(response) => response,
            LiveStreamStart::Retry { error, .. } => panic!("unexpected retry: {error}"),
        };
        let mut body = response.into_body();
        let first = tokio::time::timeout(Duration::from_millis(200), body.frame())
            .await
            .expect("initial downstream frame must be available immediately")
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        let first = String::from_utf8(first.to_vec()).unwrap();
        assert!(first.contains("event: message_start"));
        assert!(first.contains("event: ping"));
        assert!(first.contains("event: content_block_start"));
        assert!(first.contains("event: content_block_delta"));

        tx.send(Ok(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "second"
        })))
        .await
        .unwrap();
        let second = tokio::time::timeout(Duration::from_millis(200), body.frame())
            .await
            .expect("text delta must arrive before the terminal event")
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert!(
            String::from_utf8(second.to_vec())
                .unwrap()
                .contains("event: content_block_delta")
        );

        for payload in [
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "incomplete_details": null,
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                }
            }),
        ] {
            tx.send(Ok(payload)).await.unwrap();
        }
        drop(tx);
        while let Some(frame) = body.frame().await {
            frame.unwrap();
        }
    }

    #[tokio::test]
    async fn ttft_attempt_id_survives_live_websocket_coordinator() {
        use http_body_util::BodyExt as _;
        let temp = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            (
                "CCP_CONFIG_DIR".into(),
                temp.path().join("config").display().to_string(),
            ),
            (
                "XDG_STATE_HOME".into(),
                temp.path().join("state").display().to_string(),
            ),
        ]));
        let path = settings.clone().sync_scope(crate::logging::log_file);
        settings.scope(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (resume, wait_for_reader) = tokio::sync::oneshot::channel();
            let peer = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                next_live_websocket_request(&mut ws).await;
                let mut wait_for_reader = Some(wait_for_reader);
                for event in [
                    serde_json::json!({"type":"response.created"}),
                    serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_ttft"}}),
                    serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"local summary"}),
                    serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_ttft","encrypted_content":"opaque"}}),
                    serde_json::json!({"type":"response.completed","response":{"id":"r_ttft","status":"completed"}}),
                ] {
                    if event["type"] == "response.reasoning_summary_text.delta" {
                        wait_for_reader.take().unwrap().await.unwrap();
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    emit_live_event(&mut ws, &event).await;
                }
            });
            let client = authenticated_live_test_client(format!("http://{address}/responses"));
            let response = tokio::spawn(config::inherit_request_scope(live_stream_response(client, "msg_ttft".into(), "gpt-6.1-sol",
                live_test_context("ttft-coordinator"), live_test_request("hello"),
                ContinuationReservation::for_owner_turn(None, None), config::CodexTransport::WebSocket, false)));
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
                    let records = std::fs::read_to_string(&path).unwrap_or_default();
                    if records.lines().filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                        .any(|r| r["msg"]=="websocket_phase" && r["fields"]["phase"]=="reasoning_started") { break; }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }).await.expect("reader must record reasoning before summary is released");
            let response = tokio::time::timeout(Duration::from_secs(1), response)
                .await.expect("reasoning start must publish the response before summary")
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let mut body = response.into_body();
            let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
            let first = String::from_utf8_lossy(&first);
            assert!(first.contains("\"type\":\"thinking\""));
            assert!(!first.contains("local summary"));
            resume.send(()).unwrap();
            body.collect().await.unwrap();
            peer.await.unwrap();
        }).await;
        assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
        let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let input = records
            .iter()
            .find(|r| {
                r["msg"] == "websocket_phase" && r["fields"]["phase"] == "first_summary_delta"
            })
            .unwrap();
        let output = records
            .iter()
            .find(|r| r["msg"] == "codex_thinking_start_yielded")
            .unwrap();
        assert!(input["fields"]["attemptId"].is_u64());
        assert_eq!(
            output["fields"]["attemptId"], input["fields"]["attemptId"],
            "reader and body must identify the same transport attempt"
        );
        let reasoning = records
            .iter()
            .find(|r| r["msg"] == "websocket_phase" && r["fields"]["phase"] == "reasoning_started")
            .unwrap();
        let reasoning_us = reasoning["fields"]["monotonicUs"].as_u64().unwrap();
        let summary_us = input["fields"]["monotonicUs"].as_u64().unwrap();
        let yielded_us = output["fields"]["monotonicUs"].as_u64().unwrap();
        assert!(
            summary_us - reasoning_us >= 100_000,
            "reader timestamps must retain the scripted upstream wait"
        );
        assert!(
            yielded_us >= reasoning_us && yielded_us < summary_us,
            "body yield must follow reasoning start and precede the gated summary"
        );
        println!(
            "ttft_sample {}",
            serde_json::json!({
                "attemptId": input["fields"]["attemptId"], "processId": input["fields"]["processId"],
                "reasoningToSummaryUs": summary_us-reasoning_us,
                "reasoningToProviderThinkingYieldUs": yielded_us-reasoning_us,
                "providerThinkingYieldToSummaryUs": summary_us-yielded_us,
            })
        );
    }

    #[tokio::test]
    async fn ttft_thinking_yield_is_logged_when_body_is_polled_not_when_translated() {
        use http_body_util::BodyExt as _;
        let temp = tempfile::tempdir().unwrap();
        let settings = config::CodexRequestSettings::for_env(std::collections::HashMap::from([
            (
                "CCP_CONFIG_DIR".into(),
                temp.path().join("config").display().to_string(),
            ),
            (
                "XDG_STATE_HOME".into(),
                temp.path().join("state").display().to_string(),
            ),
        ]));
        let path = settings.clone().sync_scope(crate::logging::log_file);
        settings.scope(async {
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            for event in [
                serde_json::json!({"type":"response.created"}),
                serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}),
                serde_json::json!({"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"private-summary"}),
            ] { tx.send(Ok(event)).await.unwrap(); }
            let (upstream, _) = websocket::CodexWebSocketEventStream::pending(rx);
            let mut ctx = live_test_context("ttft-test");
            ctx.req_id = "ttft-body-poll".into();
            let response = match live_stream_response_once(
                upstream, "msg_ttft".into(), "gpt-6.1-sol", ctx,
                ContinuationReservation::for_owner_turn(None, None), live_test_request("hello"), 12,
                config::CodexTransport::Http, false,
            ).await {
                LiveStreamStart::Response(r) => r,
                LiveStreamStart::Retry { error, .. } => panic!("unexpected retry: {error}"),
            };
            let records = || {
                assert!(crate::logging::flush_diagnostics(Duration::from_secs(2)));
                std::fs::read_to_string(&path).unwrap_or_default().lines()
                    .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                    .filter(|r| r["msg"] == "codex_thinking_start_yielded").collect::<Vec<_>>()
            };
            assert!(records().is_empty(), "translation is not a body yield");
            let mut body = response.into_body();
            let frame = body.frame().await.unwrap().unwrap().into_data().unwrap();
            let first = String::from_utf8_lossy(&frame);
            assert!(first.contains("\"type\":\"thinking\""));
            assert!(!first.contains("private-summary"));
            let second = body.frame().await.unwrap().unwrap().into_data().unwrap();
            assert!(String::from_utf8_lossy(&second).contains("private-summary"));
            let records = records();
            assert_eq!(records.len(), 1, "first thinking frame needs a body-poll timestamp");
            assert_eq!(records[0]["fields"]["reqId"], "ttft-body-poll");
            assert_eq!(records[0]["fields"]["boundary"], "provider_body_yield");
            assert_eq!(records[0]["fields"]["phase"], "thinking_started");
            assert_eq!(records[0]["fields"]["outcome"], "pending");
            assert!(records[0]["fields"]["monotonicUs"].as_u64().is_some());
            assert!(!records[0].to_string().contains("private-summary"));
            drop(body);
            drop(tx);
        }).await;
    }

    #[test]
    fn supported_models_includes_fast_variants() {
        let provider = CodexProvider::new();
        let models = provider.supported_models();
        assert!(models.contains(&"gpt-5.6-sol".to_string()));
        assert!(models.contains(&"gpt-5.6-sol-fast".to_string()));
        assert!(models.contains(&"gpt-5.6-terra".to_string()));
        assert!(models.contains(&"gpt-5.6-luna".to_string()));
        assert!(models.contains(&"gpt-6-sol".to_string()));
        assert!(models.contains(&"gpt-6-sol-fast".to_string()));
        assert!(models.contains(&"gpt-6-luna".to_string()));
        assert!(models.contains(&"gpt-6.1-sol".to_string()));
        assert!(models.contains(&"gpt-6.1-sol-fast".to_string()));
        assert!(models.contains(&"gpt-6.1-terra".to_string()));
        assert!(models.contains(&"gpt-6.1-terra-fast".to_string()));
        assert!(!models.contains(&"gpt-6.1-terra-ultrafast".to_string()));
        assert!(models.contains(&"gpt-6.1-sol-ultrafast".to_string()));
        assert!(models.contains(&"gpt-6.1-terra".to_string()));
        assert!(models.contains(&"gpt-6.1-terra-fast".to_string()));
        assert!(!models.contains(&"gpt-6.1-terra-ultrafast".to_string()));
        assert!(models.contains(&"gpt-5.4".to_string()));
        assert!(models.contains(&"gpt-5.4-mini".to_string()));
        assert!(models.contains(&"gpt-6-astra-fast".to_string()));
        assert!(models.contains(&"gpt-6-astra-ultrafast".to_string()));
        assert!(!models.contains(&"gpt-6-sol-ultrafast".to_string()));
    }

    #[test]
    fn format_auth_saved_output_with_account() {
        assert_eq!(
            format_auth_saved_output("/tmp/auth.json", Some("acct_1")),
            "Auth saved in /tmp/auth.json\nAccount: acct_1\n"
        );
    }

    #[test]
    fn format_auth_saved_output_without_account() {
        assert_eq!(
            format_auth_saved_output("/tmp/auth.json", None),
            "Auth saved in /tmp/auth.json\n"
        );
    }

    #[test]
    fn format_expiry_with_future_expiry() {
        // 2100-01-01T00:00:00Z in ms
        let expires = 4102444800000;
        let now = 4102444790000; // 10s before
        let output = format_expiry(expires, now);
        assert!(output.starts_with("Expires: 2100-01-01T00:00:00.000Z (in "));
        assert!(output.ends_with("s)"));
    }

    #[test]
    fn format_expiry_with_past_expiry() {
        // 2000-01-01T00:00:00Z in ms
        let expires = 946684800000;
        let now = 946684810000; // 10s after
        let output = format_expiry(expires, now);
        assert!(output.starts_with("Expires: 2000-01-01T00:00:00.000Z (in -"));
    }

    #[tokio::test]
    async fn live_upstream_status_and_retry_after_are_preserved() {
        let err = client::CodexError {
            status: 422,
            message: "invalid request".to_string(),
            detail: Some("invalid request".to_string()),
            retry_after: Some("7".to_string()),
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };
        let response = map_codex_error_to_response(&err);
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            response.headers().get(http::header::RETRY_AFTER).unwrap(),
            "7"
        );
    }

    #[tokio::test]
    async fn statusless_codex_error_returns_source_message() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket connect error: HTTP error: 502 Bad Gateway".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocket,
        };

        let response = map_codex_error_to_response(&err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body.pointer("/error/message").and_then(|v| v.as_str()),
            Some("WebSocket connect error: HTTP error: 502 Bad Gateway")
        );
    }

    #[tokio::test]
    async fn empty_live_completion_maps_to_explicit_service_unavailable() {
        let err = empty_live_completion_error();

        assert_eq!(err.status, 503);
        assert_eq!(err.detail.as_deref(), Some(EMPTY_CODEX_COMPLETION_DETAIL));
        let response = map_codex_error_to_response(&err);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["x-should-retry"], "false");
    }

    #[test]
    fn empty_completion_exhausted_fields_distinguish_the_two_give_up_reasons() {
        let max = empty_completion_exhausted_fields(
            "req-1",
            "auto",
            "gpt-6-sol",
            11,
            "max_attempts",
            157_000,
        );
        assert_eq!(max["reqId"], "req-1");
        assert_eq!(max["configuredTransport"], "auto");
        assert!(max.get("transport").is_none());
        assert_eq!(max["model"], "gpt-6-sol");
        assert_eq!(max["attempts"], 11);
        assert_eq!(max["reason"], "max_attempts");
        assert_eq!(max["ms"], 157_000);

        let budget = empty_completion_exhausted_fields(
            "req-1",
            "websocket",
            "gpt-6-sol",
            4,
            "backoff_budget",
            20_000,
        );
        assert_eq!(budget["reason"], "backoff_budget");
    }

    #[test]
    fn live_start_statusless_websocket_handshake_error_is_retryable() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket connect timeout after 15000ms".to_string(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(retryable_live_start_codex_error(&err));
    }

    #[test]
    fn live_start_proxy_tunnel_rejection_is_not_retryable() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket proxy tunnel was rejected".to_string(),
            detail: Some(websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(!retryable_live_start_codex_error(&err));
    }

    #[test]
    fn live_start_keepalive_failure_is_retryable() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket keepalive error: test write failed".to_string(),
            detail: Some(websocket::WEBSOCKET_KEEPALIVE_FAILURE_DETAIL.to_string()),
            retry_after: None,
            usage_limit: None,
            origin: client::CodexErrorOrigin::WebSocket,
        };

        assert!(retryable_live_start_codex_error(&err));
    }

    #[test]
    fn live_start_payload_retry_detection_uses_event_failure_classification() {
        assert!(
            events::classify_event_failure(&serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {"limit_reached": true}
            }))
            .is_none()
        );
        assert!(
            events::classify_event_failure(&serde_json::json!({
                "type": "response.failed",
                "response": {"error": {"type": "overloaded_error", "message": "overloaded"}}
            }))
            .is_some_and(|failure| failure.retryable())
        );
        assert!(
            !events::classify_event_failure(&serde_json::json!({
                "type": "response.failed",
                "response": {"error": {"status": 400, "code": "invalid_prompt", "message": "bad request"}}
            }))
            .is_some_and(|failure| failure.retryable())
        );
    }

    async fn run_live_failure_case(
        session_id: &str,
        event: serde_json::Value,
        expected_attempts: usize,
    ) -> Response {
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for _ in 0..expected_attempts {
                let (socket, _) = listener.accept().await.unwrap();
                let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
                let _ = next_live_websocket_request(&mut websocket).await;
                emit_live_event(&mut websocket, &event).await;
                drop(websocket);
            }
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            live_stream_response(
                client,
                "message".to_string(),
                &request.model,
                live_test_context(session_id),
                request.clone(),
                continuation.clone(),
                config::CodexTransport::WebSocket,
                false,
            ),
        )
        .await
        .expect("live failure case timed out");
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .expect("live failure server timed out")
            .expect("live failure server failed");

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        response
    }

    #[tokio::test]
    async fn compact_partial_summary_eof_is_error() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        for (suffix, terminal) in [
            ("eof", None),
            (
                "error",
                Some(serde_json::json!({
                    "type": "response.failed", "response": {"status": "failed", "error": {
                        "type": "server_error", "message": "summary interrupted"
                    }}
                })),
            ),
            (
                "incomplete",
                Some(serde_json::json!({
                    "type": "response.incomplete", "response": {"status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"}}
                })),
            ),
        ] {
            let session_id = format!("compact-partial-summary-{suffix}");
            let owner = ConversationIdentity::Main(session_id.clone());
            continuation::clear_continuation_for_owner(Some(&owner));
            websocket::invalidate_codex_websocket_pool_owner(&owner);
            let mut request = live_test_request("Summarize with the user's focus");
            request.instructions =
                Some("You are a helpful AI assistant tasked with summarizing conversations".into());
            let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
                let _ = next_live_websocket_request(&mut websocket).await;
                emit_live_event(
                    &mut websocket,
                    &serde_json::json!({
                        "type": "response.output_item.added", "output_index": 0,
                        "item": {"type": "message", "id": "summary-partial"}
                    }),
                )
                .await;
                emit_live_event(
                    &mut websocket,
                    &serde_json::json!({
                        "type": "response.output_text.delta", "output_index": 0,
                        "delta": "<summary>partial"
                    }),
                )
                .await;
                release_rx.await.unwrap();
                if let Some(terminal) = terminal {
                    emit_live_event(&mut websocket, &terminal).await;
                }
                websocket.close(None).await.unwrap();
            });
            let client = authenticated_live_test_client(format!("http://{addr}/responses"));
            let response = tokio::time::timeout(
                Duration::from_secs(2),
                live_stream_response(
                    client,
                    "message".into(),
                    &request.model,
                    live_test_context(&session_id),
                    request.clone(),
                    continuation.clone(),
                    config::CodexTransport::WebSocket,
                    false,
                ),
            )
            .await
            .expect("partial summary must be published before upstream termination");
            assert_eq!(response.status(), StatusCode::OK);
            let mut body = response.into_body();
            let first = tokio::time::timeout(Duration::from_secs(1), body.frame())
                .await
                .expect("partial summary must stream before EOF")
                .unwrap()
                .unwrap()
                .into_data()
                .unwrap();
            let first = std::str::from_utf8(&first).unwrap();
            assert!(first.contains("<summary>partial"));
            assert!(!first.contains("event: error"));
            assert!(!first.contains("event: message_stop"));
            release_tx.send(()).unwrap();
            let remaining = tokio::time::timeout(Duration::from_secs(2), body.collect())
                .await
                .expect("partial summary failure must finish without a replay")
                .unwrap()
                .to_bytes();
            let remaining = std::str::from_utf8(&remaining).unwrap();
            assert!(remaining.contains("event: error"), "{suffix}: {remaining}");
            assert!(
                !remaining.contains("<summary>partial"),
                "partial text must not replay"
            );
            assert!(!remaining.contains("event: message_stop"));
            assert!(!remaining.contains("</summary>"));
            assert!(!continuation::is_current_turn_for_owner(&continuation));
            server.await.unwrap();
            websocket::invalidate_codex_websocket_pool_owner(&owner);
        }
    }

    #[tokio::test]
    async fn usage_limit_fast_fails_websocket_and_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let response = run_live_failure_case(
            "live-usage-limit-cleanup",
            serde_json::json!({
                "type": "error",
                "status_code": 429,
                "error": {
                    "type": "usage_limit_reached",
                    "message": "The usage limit has been reached",
                    "resets_at": 1788879437u64,
                    "resets_in_seconds": 9568
                },
                "headers": {
                    "X-Codex-Primary-Window-Minutes": "300",
                    "X-Codex-Primary-Reset-After-Seconds": "9569",
                    "X-Codex-Secondary-Window-Minutes": "10080",
                    "X-Codex-Secondary-Reset-After-Seconds": "596369"
                }
            }),
            1,
        )
        .await;

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["x-should-retry"], "false");
        assert_eq!(
            response.headers()["anthropic-ratelimit-unified-status"],
            "rejected"
        );
        assert_eq!(
            response.headers()["anthropic-ratelimit-unified-reset"],
            "1788879437"
        );
        assert_eq!(
            response.headers()["anthropic-ratelimit-unified-representative-claim"],
            "five_hour"
        );
        assert!(!response.headers().contains_key(http::header::RETRY_AFTER));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["message"], "The usage limit has been reached");
    }

    #[tokio::test]
    async fn dropping_live_stream_during_retry_backoff_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-retry-backoff-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (event_sent_tx, event_sent_rx) = tokio::sync::oneshot::channel();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "codex.rate_limits",
                    "rate_limits": {"allowed": false, "limit_reached": true}
                }),
            )
            .await;
            event_sent_tx.send(()).unwrap();
            drop(websocket);
            socket_closed_tx.send(()).unwrap();
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let task_request = request.clone();
        let task_continuation = continuation.clone();
        let response_task = tokio::spawn(async move {
            let model = task_request.model.clone();
            live_stream_response(
                client,
                "message".to_string(),
                &model,
                live_test_context(session_id),
                task_request,
                task_continuation,
                config::CodexTransport::WebSocket,
                false,
            )
            .await
        });

        event_sent_rx.await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), socket_closed_rx)
            .await
            .expect("retry handoff did not close the abandoned attempt socket")
            .expect("retry handoff socket-close sender dropped");
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        assert!(
            !response_task.is_finished(),
            "logical request must still be waiting in retry backoff"
        );
        response_task.abort();
        assert!(response_task.await.unwrap_err().is_cancelled());

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn dropping_live_response_body_after_first_chunk_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-response-body-drop-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "id": "msg-partial"}
                }),
            )
            .await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "partial"
                }),
            )
            .await;
            while websocket.next().await.is_some() {}
            socket_closed_tx.send(()).unwrap();
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            live_stream_response(
                client,
                "message".to_string(),
                &request.model,
                live_test_context(session_id),
                request.clone(),
                continuation.clone(),
                config::CodexTransport::WebSocket,
                false,
            ),
        )
        .await
        .expect("live response did not publish the first chunk");
        let mut body = response.into_body();
        tokio::time::timeout(std::time::Duration::from_secs(1), body.frame())
            .await
            .expect("first downstream chunk timed out")
            .expect("live response body ended before the first chunk")
            .expect("first downstream chunk failed");
        drop(body);

        tokio::time::timeout(std::time::Duration::from_secs(1), socket_closed_rx)
            .await
            .expect("dropping the downstream body did not close the upstream socket")
            .expect("socket-close acknowledgement sender dropped");
        assert!(!continuation::is_current_turn_for_owner(&continuation));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn dropping_body_after_reader_reinsertion_aborts_exact_socket() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-completed-body-drop-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "id": "msg-partial"}
                }),
            )
            .await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "partial"
                }),
            )
            .await;
            emit_live_event(&mut websocket, &serde_json::json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg-partial"}})).await;
            emit_live_event(&mut websocket, &serde_json::json!({"type":"response.completed","response":{"id":"finished-but-undelivered","status":"completed"}})).await;
            while websocket.next().await.is_some() {}
            socket_closed_tx.send(()).unwrap();
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            live_stream_response(
                client,
                "message".to_string(),
                &request.model,
                live_test_context(session_id),
                request.clone(),
                continuation.clone(),
                config::CodexTransport::WebSocket,
                false,
            ),
        )
        .await
        .expect("live response did not publish the first chunk");
        let mut body = response.into_body();
        tokio::time::timeout(std::time::Duration::from_secs(1), body.frame())
            .await
            .expect("first downstream chunk timed out")
            .expect("live response body ended before the first chunk")
            .expect("first downstream chunk failed");
        tokio::time::timeout(Duration::from_secs(1), async {
            while !continuation::has_continuation_for_owner_for_tests(&owner) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reader must reinsert and finish before consumer cancellation");
        assert!(websocket::pooled_socket_id_for_tests(&owner).is_some());
        drop(body);

        tokio::time::timeout(std::time::Duration::from_secs(1), socket_closed_rx)
            .await
            .expect("dropping the downstream body did not close the upstream socket")
            .expect("socket-close acknowledgement sender dropped");
        assert!(!continuation::is_current_turn_for_owner(&continuation));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn stale_request_cleanup_preserves_newer_turn() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let session_id = "stale-live-request-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        let request = live_test_request("one");
        let stale_continuation = continuation_candidate_for_owner(Some(&owner), &request, true);
        let stale_cleanup = LiveRequestStateCleanup::new(stale_continuation);

        let newer_continuation = continuation_candidate_for_owner(Some(&owner), &request, true);
        drop(stale_cleanup);

        assert!(continuation::is_current_turn_for_owner(&newer_continuation));
        abort_continuation_for_owner(&newer_continuation);
    }

    #[tokio::test]
    async fn retry_exhaustion_aborts_live_request_state_after_eleven_attempts() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        count_tokens::observations::BPE_CALLS.set(0);
        count_tokens::observations::ESTIMATES.set(0);
        let status = run_live_failure_case(
            "live-retry-exhaustion-cleanup",
            serde_json::json!({
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "status": 429,
                        "code": "rate_limit_exceeded",
                        "message": "rate limit reached",
                        "retry_after_seconds": 0
                    }
                }
            }),
            11,
        )
        .await
        .status();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            count_tokens::observations::ESTIMATES.get(),
            1,
            "retry re-estimated logical input"
        );
        assert_eq!(
            count_tokens::observations::BPE_CALLS.get(),
            0,
            "live retry ran inline BPE"
        );
    }

    #[tokio::test]
    async fn excessive_retry_after_aborts_live_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let status = run_live_failure_case(
            "live-excessive-retry-after-cleanup",
            serde_json::json!({
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "status": 429,
                        "code": "rate_limit_exceeded",
                        "message": "rate limit reached",
                        "retry_after_seconds": 31
                    }
                }
            }),
            1,
        )
        .await
        .status();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn nonretryable_live_error_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let status = run_live_failure_case(
            "live-nonretryable-cleanup",
            serde_json::json!({
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "status": 400,
                        "code": "invalid_prompt",
                        "message": "invalid request"
                    }
                }
            }),
            1,
        )
        .await
        .status();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn cancellation_while_replacement_startup_is_blocked_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-blocked-replacement-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (replacement_accepted_tx, replacement_accepted_rx) = tokio::sync::oneshot::channel();
        let (release_replacement_tx, release_replacement_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            let _ = next_live_websocket_request(&mut first_websocket).await;
            emit_live_event(
                &mut first_websocket,
                &serde_json::json!({
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "status": 429,
                            "message": "rate limit exceeded",
                            "retry_after_seconds": 0
                        }
                    }
                }),
            )
            .await;
            drop(first_websocket);

            let (_replacement_socket, _) = listener.accept().await.unwrap();
            replacement_accepted_tx.send(()).unwrap();
            let _ = release_replacement_rx.await;
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let task_request = request.clone();
        let task_continuation = continuation.clone();
        let response_task = tokio::spawn(async move {
            let model = task_request.model.clone();
            live_stream_response(
                client,
                "message".to_string(),
                &model,
                live_test_context(session_id),
                task_request,
                task_continuation,
                config::CodexTransport::WebSocket,
                false,
            )
            .await
        });

        // Other parallel tests also queue at the process-wide, one-second
        // WebSocket start gate; the acknowledgement is the actual assertion.
        tokio::time::timeout(std::time::Duration::from_secs(120), replacement_accepted_rx)
            .await
            .expect("replacement startup did not reach the blocked handshake")
            .expect("replacement startup acknowledgement sender dropped");
        response_task.abort();
        assert!(response_task.await.unwrap_err().is_cancelled());
        let _ = release_replacement_tx.send(());
        server.await.unwrap();

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }
}
