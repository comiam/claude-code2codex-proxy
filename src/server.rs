use crate::{
    anthropic::{MAX_ANTHROPIC_REQUEST_BYTES, json_error},
    logging::{Logger, REDACT_KEYS, create_logger},
    monitor::{DeliveryEvent, EndpointKind, MonitorHandle},
    openai_compat::{
        MAX_OPENAI_REQUEST_BYTES, OpenAiError, OpenAiSurface,
        request::{extract_model, parse_request},
        stream::openai_response as render_openai_response,
    },
    project,
    provider::RequestContext,
    providers::codex::{
        chat_completions::{ChatCompletionsBackend, request::translate_request},
        images::{
            CodexImagesBackend, ImageOperation, ImageRequestError, MAX_EDIT_REQUEST_BYTES,
            MAX_GENERATION_REQUEST_BYTES, MultipartEditInput, UploadedImage, image_error_response,
            prepare_json_request, prepare_multipart_edit,
        },
        native::{
            CodexNativeBackend, NativeResponseOutcome, openai_error, validate_native_request_model,
        },
        transcription::{
            CodexTranscriptionBackend, MAX_TRANSCRIPTION_REQUEST_BYTES, TranscriptionRequestError,
            prepare_transcription, transcription_error_response,
        },
        translate::request::is_compact_messages_request,
    },
    registry::{Registry, normalize_incoming_model},
    request_identity::{CLAUDE_AGENT_HEADER, CLAUDE_PARENT_AGENT_HEADER, ConversationIdentity},
    session::{self, SessionState},
    traffic::{TrafficCaptureOptions, create_traffic_capture},
};
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Query, State},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    serve::ListenerExt,
};
use http_body_util::{BodyExt, StreamBody};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use uuid::Uuid;

const CLAUDE_AUTO_REVIEW_SYSTEM_PREFIX: &str =
    "You are a security monitor for autonomous AI coding agents.";
const CODEX_AUTO_REVIEW_MODEL: &str = "gpt-6-luna";

#[derive(Debug, Clone, PartialEq, Eq)]
struct AutoReviewRoute {
    requested_model: String,
    override_model: String,
}

fn is_claude_auto_review_request(body: &crate::anthropic::schema::MessagesRequest) -> bool {
    if body.stream {
        return false;
    }

    let has_tools = body
        .extra
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if has_tools {
        return false;
    }

    body.extra
        .get("system")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| {
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.starts_with(CLAUDE_AUTO_REVIEW_SYSTEM_PREFIX))
            })
        })
}

fn apply_auto_review_model(
    body: &mut crate::anthropic::schema::MessagesRequest,
    count_tokens: bool,
    configured_model: Option<&str>,
    original_provider: &str,
) -> Option<AutoReviewRoute> {
    if count_tokens || !is_claude_auto_review_request(body) {
        return None;
    }

    let override_model = configured_model
        .filter(|model| !model.is_empty())
        .or((original_provider == "codex").then_some(CODEX_AUTO_REVIEW_MODEL))?;
    let route = AutoReviewRoute {
        requested_model: body.model.clone()?,
        override_model: override_model.to_string(),
    };
    body.model = Some(route.override_model.clone());
    Some(route)
}

pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub monitor: Option<MonitorHandle>,
}

pub async fn serve(config: ServerConfig) -> anyhow::Result<()> {
    serve_inner(config, std::future::pending::<()>()).await
}

pub async fn serve_with_shutdown(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    serve_inner(config, shutdown).await
}

async fn serve_inner(
    config: ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let listener = bind_proxy_listener(&config.bind_address, config.port).await?;
    serve_listener(listener, config.monitor, shutdown).await
}

pub async fn bind_proxy_listener(bind_address: &str, port: u16) -> anyhow::Result<TcpListener> {
    let ip = bind_address
        .parse::<std::net::IpAddr>()
        .map_err(|err| anyhow::anyhow!("invalid proxy bind address {bind_address:?}: {err}"))?;
    let addr = std::net::SocketAddr::new(ip, port);
    TcpListener::bind(addr)
        .await
        .map_err(|err| anyhow::anyhow!("failed to bind proxy listener on {addr}: {err}"))
}

pub async fn serve_listener(
    listener: TcpListener,
    monitor: Option<MonitorHandle>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let local_addr = listener.local_addr()?;
    let port = local_addr.port();
    create_logger("server").info(
        "server listening",
        Some(serde_json::Map::from_iter([
            ("port".to_string(), json!(port)),
            (
                "bindAddress".to_string(),
                json!(local_addr.ip().to_string()),
            ),
            (
                "logDir".to_string(),
                json!(
                    crate::paths::log_file()
                        .parent()
                        .map(|path| path.display().to_string())
                ),
            ),
        ])),
    );
    let app = app_with_monitor(Arc::new(Registry::with_default_alias()), monitor);
    axum::serve(
        listener.tap_io(|socket| {
            if let Err(error) = socket.set_nodelay(true) {
                create_logger("server").warn(
                    "downstream_tcp_nodelay_failed",
                    Some(serde_json::Map::from_iter([(
                        "error".to_string(),
                        json!(error.to_string()),
                    )])),
                );
            }
        }),
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await?;
    Ok(())
}

pub fn app(registry: Arc<Registry>) -> Router {
    app_with_features(
        registry,
        None,
        AppFeatures {
            responses_api: crate::config::codex_responses_api(),
            images_api: crate::config::codex_images_api(),
            transcriptions_api: crate::config::codex_transcriptions_api(),
        },
    )
}

pub fn app_with_monitor(registry: Arc<Registry>, monitor: Option<MonitorHandle>) -> Router {
    app_with_features(
        registry,
        monitor,
        AppFeatures {
            responses_api: crate::config::codex_responses_api(),
            images_api: crate::config::codex_images_api(),
            transcriptions_api: crate::config::codex_transcriptions_api(),
        },
    )
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AppFeatures {
    pub responses_api: bool,
    pub images_api: bool,
    pub transcriptions_api: bool,
}

pub fn app_with_options(
    registry: Arc<Registry>,
    monitor: Option<MonitorHandle>,
    responses_api: bool,
) -> Router {
    app_with_features(
        registry,
        monitor,
        AppFeatures {
            responses_api,
            images_api: false,
            transcriptions_api: false,
        },
    )
}

pub fn app_with_features(
    registry: Arc<Registry>,
    monitor: Option<MonitorHandle>,
    features: AppFeatures,
) -> Router {
    let native_responses = features
        .responses_api
        .then(|| Arc::new(CodexNativeBackend::new()));
    let chat_completions = features
        .responses_api
        .then(|| Arc::new(ChatCompletionsBackend::new()));
    let images = if features.images_api {
        match CodexImagesBackend::new() {
            Ok(backend) => Some(Arc::new(backend)),
            Err(error) => {
                create_logger("server").warn(
                    "codex images backend disabled by invalid configuration",
                    Some(Map::from_iter([("error".to_string(), json!(error))])),
                );
                None
            }
        }
    } else {
        None
    };
    let transcriptions = features
        .transcriptions_api
        .then(|| Arc::new(CodexTranscriptionBackend::new()));
    let state = Arc::new(AppState {
        registry,
        monitor,
        native_responses,
        chat_completions,
        images,
        transcriptions,
    });
    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/monitor", get(handler_monitor))
        .route("/usage", get(handler_usage))
        .route("/v1/messages", post(handler_messages))
        .route("/v1/messages/count_tokens", post(handler_count_tokens))
        .route("/v1/models", get(handler_models));
    let router = if features.responses_api {
        router
            .route("/v1/responses", post(handler_responses))
            .route("/v1/chat/completions", post(handler_chat_completions))
    } else {
        router
    };
    let router = if features.images_api {
        router
            .route("/v1/images/generations", post(handler_image_generation))
            .route(
                "/v1/images/edits",
                post(handler_image_edit).layer(DefaultBodyLimit::max(MAX_EDIT_REQUEST_BYTES)),
            )
    } else {
        router
    };
    let router = if features.transcriptions_api {
        router.route(
            "/v1/audio/transcriptions",
            post(handler_transcription)
                .layer(DefaultBodyLimit::max(MAX_TRANSCRIPTION_REQUEST_BYTES)),
        )
    } else {
        router
    };
    router.fallback(fallback_handler).with_state(state)
}

#[derive(Clone)]
struct AppState {
    registry: Arc<Registry>,
    monitor: Option<MonitorHandle>,
    native_responses: Option<Arc<CodexNativeBackend>>,
    chat_completions: Option<Arc<ChatCompletionsBackend>>,
    images: Option<Arc<CodexImagesBackend>>,
    transcriptions: Option<Arc<CodexTranscriptionBackend>>,
}

async fn handler_monitor(
    State(state): State<Arc<AppState>>,
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Response {
    let Some(axum::Extension(axum::extract::ConnectInfo(peer))) = peer else {
        return json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Monitor access requires a local connection",
        );
    };
    if !peer.ip().to_canonical().is_loopback() {
        return json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Monitor access requires a local connection",
        );
    }
    match &state.monitor {
        Some(monitor) => (
            [(http::header::CACHE_CONTROL, "no-store")],
            Json(crate::monitor::snapshot::MonitorResponse::from(
                monitor.snapshot(),
            )),
        )
            .into_response(),
        None => json_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Monitor collection is disabled",
        ),
    }
}

async fn healthz() -> Json<serde_json::Value> {
    Json(json!({ "ok": true }))
}

async fn handler_usage() -> Json<serde_json::Value> {
    Json(json!({
        "codex": crate::providers::codex::rate_limits::latest(),
    }))
}

#[derive(serde::Deserialize)]
struct ModelsQuery {
    limit: Option<usize>,
}

/// Anthropic-shaped model listing so Claude Code's gateway model discovery
/// (`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`) finds the proxy's models.
/// Claude Code only adds entries whose id starts with `claude` or `anthropic`,
/// so the Anthropic-style aliases are what surface in its `/model` picker;
/// raw provider ids are still listed for other Anthropic-compatible clients.
async fn handler_models(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ModelsQuery>,
) -> Json<serde_json::Value> {
    let mut data: Vec<Value> = state
        .registry
        .all_supported_models()
        .into_iter()
        .map(|(model, provider)| {
            json!({
                "type": "model",
                "object": "model",
                "id": model,
                "display_name": format!("{model} ({provider})"),
            })
        })
        .collect();
    let has_more = query.limit.is_some_and(|limit| data.len() > limit);
    if let Some(limit) = query.limit {
        data.truncate(limit);
    }
    Json(json!({
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": data.first().and_then(|entry| entry.get("id")).cloned(),
        "last_id": data.last().and_then(|entry| entry.get("id")).cloned(),
    }))
}

async fn handler_messages(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let mut response = dispatch_request(state, req, false).await;
    crate::providers::codex::rate_limits::apply_anthropic_headers(response.headers_mut());
    response
}

async fn handler_count_tokens(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    dispatch_request(state, req, true).await
}

async fn handler_transcription(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    crate::config::request_scope(async {
        crate::provider::request_scoped_response(
            handler_transcription_inner(State(state), req).await,
        )
    })
    .await
}

async fn handler_transcription_inner(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let headers = req.headers().clone();
    log.info(
        "request",
        Some(Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!("POST")),
            ("path".to_string(), json!("/v1/audio/transcriptions")),
            ("query".to_string(), json!({})),
        ])),
    );
    let session_id = native_session_id(&headers);
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(
            &req_id,
            session_id.clone(),
            None,
            EndpointKind::Transcriptions,
        );
        monitor.provider_selected(&req_id, "codex", "codex-transcribe", None);
    }
    let request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());
    if req.uri().query().is_some() {
        let response = transcription_error_response(TranscriptionRequestError::invalid(
            "Transcription endpoint does not accept query parameters",
            None,
        ));
        log_native_request_completed(
            &log,
            &req_id,
            "transcriptions",
            Some("codex-transcribe"),
            response.status(),
            started_at,
        );
        return monitor_response_body(response, request_guard);
    }
    let content_type = headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.starts_with("multipart/form-data") {
        let response = transcription_error_response(TranscriptionRequestError {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            message: "Transcription request must use multipart/form-data".to_string(),
            param: None,
            code: "unsupported_media_type",
        });
        log_native_request_completed(
            &log,
            &req_id,
            "transcriptions",
            Some("codex-transcribe"),
            response.status(),
            started_at,
        );
        return monitor_response_body(response, request_guard);
    }
    let mut multipart = match Multipart::from_request(req, &()).await {
        Ok(multipart) => multipart,
        Err(error) => {
            let status = axum::response::IntoResponse::into_response(error).status();
            let response = transcription_error_response(TranscriptionRequestError {
                status: if status == StatusCode::PAYLOAD_TOO_LARGE {
                    StatusCode::PAYLOAD_TOO_LARGE
                } else {
                    StatusCode::BAD_REQUEST
                },
                message: if status == StatusCode::PAYLOAD_TOO_LARGE {
                    "Transcription request exceeded the size limit".to_string()
                } else {
                    "Invalid multipart transcription request".to_string()
                },
                param: None,
                code: if status == StatusCode::PAYLOAD_TOO_LARGE {
                    "request_too_large"
                } else {
                    "invalid_multipart"
                },
            });
            log_native_request_completed(
                &log,
                &req_id,
                "transcriptions",
                Some("codex-transcribe"),
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let mut audio = None;
    let mut filename = None;
    let mut audio_content_type = None;
    let mut language = None;
    while let Some(field) = match multipart.next_field().await {
        Ok(field) => field,
        Err(_) => {
            let response = transcription_error_response(TranscriptionRequestError::invalid(
                "Invalid multipart transcription request",
                None,
            ));
            return monitor_response_body(response, request_guard);
        }
    } {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                if audio.is_some() {
                    let response =
                        transcription_error_response(TranscriptionRequestError::invalid(
                            "Transcription request must contain one 'file' field",
                            Some("file"),
                        ));
                    return monitor_response_body(response, request_guard);
                }
                filename = field.file_name().map(str::to_string);
                audio_content_type = field.content_type().map(str::to_string);
                audio = match field.bytes().await {
                    Ok(bytes) => Some(bytes),
                    Err(_) => {
                        let response =
                            transcription_error_response(TranscriptionRequestError::invalid(
                                "Failed to read uploaded audio",
                                Some("file"),
                            ));
                        return monitor_response_body(response, request_guard);
                    }
                };
            }
            "language" => {
                language = match multipart_text(field, "language").await {
                    Ok(value) => Some(value),
                    Err(error) => {
                        return monitor_response_body(
                            transcription_error_response(TranscriptionRequestError {
                                status: error.status,
                                message: error.message,
                                param: error.param,
                                code: error.code.unwrap_or("invalid_multipart"),
                            }),
                            request_guard,
                        );
                    }
                };
            }
            "model" => {
                if multipart_text(field, "model").await.is_err() {
                    let response =
                        transcription_error_response(TranscriptionRequestError::invalid(
                            "Invalid multipart 'model' field",
                            Some("model"),
                        ));
                    return monitor_response_body(response, request_guard);
                }
            }
            _ => {
                let response = transcription_error_response(TranscriptionRequestError::invalid(
                    format!("Unsupported multipart field '{name}'"),
                    None,
                ));
                return monitor_response_body(response, request_guard);
            }
        }
    }
    let prepared = match prepare_transcription(audio, filename, audio_content_type, language) {
        Ok(prepared) => prepared,
        Err(error) => {
            let response = transcription_error_response(error);
            log_native_request_completed(
                &log,
                &req_id,
                "transcriptions",
                Some("codex-transcribe"),
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: None,
        provider: "codex".to_string(),
        traffic: None,
        monitor: state.monitor.clone(),
    };
    let response = match state.transcriptions.as_ref() {
        Some(backend) => backend.handle(prepared, context).await,
        None => transcription_error_response(TranscriptionRequestError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Codex transcription API is unavailable".to_string(),
            param: None,
            code: "transcriptions_api_unavailable",
        }),
    };
    log_native_request_completed(
        &log,
        &req_id,
        "transcriptions",
        Some("codex-transcribe"),
        response.status(),
        started_at,
    );
    monitor_response_body(response, request_guard)
}

async fn handler_image_generation(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    dispatch_image_request(state, req, ImageOperation::Generation).await
}

async fn handler_image_edit(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    dispatch_image_request(state, req, ImageOperation::Edit).await
}

async fn dispatch_image_request(
    state: Arc<AppState>,
    req: Request<Body>,
    operation: ImageOperation,
) -> Response {
    crate::config::request_scope(async {
        crate::provider::request_scoped_response(
            dispatch_image_request_inner(state, req, operation).await,
        )
    })
    .await
}

async fn dispatch_image_request_inner(
    state: Arc<AppState>,
    req: Request<Body>,
    operation: ImageOperation,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let path = uri.path().to_string();
    log.info(
        "request",
        Some(Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!("POST")),
            ("path".to_string(), json!(&path)),
            ("query".to_string(), json!(redacted_query(&uri))),
        ])),
    );
    let session_id = native_session_id(&headers);
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(&req_id, session_id.clone(), None, EndpointKind::Images);
    }
    let request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());

    if uri.query().is_some() {
        let response = image_error_response(ImageRequestError {
            status: StatusCode::BAD_REQUEST,
            message: "Image endpoints do not accept query parameters".to_string(),
            param: None,
            code: Some("invalid_request"),
        });
        log_native_request_completed(
            &log,
            &req_id,
            operation.label(),
            None,
            response.status(),
            started_at,
        );
        return monitor_response_body(response, request_guard);
    }
    let content_type = headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let prepared = if content_type.starts_with("application/json") {
        let limit = match operation {
            ImageOperation::Generation => MAX_GENERATION_REQUEST_BYTES,
            ImageOperation::Edit => MAX_EDIT_REQUEST_BYTES,
        };
        let body = match axum::body::to_bytes(req.into_body(), limit).await {
            Ok(body) => body,
            Err(_) => {
                let response = image_error_response(ImageRequestError {
                    status: StatusCode::PAYLOAD_TOO_LARGE,
                    message: "Image request exceeded the size limit".to_string(),
                    param: None,
                    code: Some("request_too_large"),
                });
                log_native_request_completed(
                    &log,
                    &req_id,
                    operation.label(),
                    None,
                    response.status(),
                    started_at,
                );
                return monitor_response_body(response, request_guard);
            }
        };
        prepare_json_request(operation, &body)
    } else if operation == ImageOperation::Edit && content_type.starts_with("multipart/form-data") {
        let multipart = match Multipart::from_request(req, &()).await {
            Ok(multipart) => multipart,
            Err(_) => {
                let response = image_error_response(ImageRequestError {
                    status: StatusCode::BAD_REQUEST,
                    message: "Invalid multipart image edit request".to_string(),
                    param: None,
                    code: Some("invalid_multipart"),
                });
                log_native_request_completed(
                    &log,
                    &req_id,
                    operation.label(),
                    None,
                    response.status(),
                    started_at,
                );
                return monitor_response_body(response, request_guard);
            }
        };
        parse_multipart_image_edit(multipart).await
    } else {
        let response = image_error_response(ImageRequestError {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            message: if operation == ImageOperation::Edit {
                "Image edit request must use application/json or multipart/form-data".to_string()
            } else {
                "Image generation request must use application/json".to_string()
            },
            param: None,
            code: Some("unsupported_media_type"),
        });
        log_native_request_completed(
            &log,
            &req_id,
            operation.label(),
            None,
            response.status(),
            started_at,
        );
        return monitor_response_body(response, request_guard);
    };
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            let response = image_error_response(error);
            log_native_request_completed(
                &log,
                &req_id,
                operation.label(),
                None,
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let model = prepared.model.clone();
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.provider_selected(&req_id, "codex", &model, None);
    }
    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: None,
        provider: "codex".to_string(),
        traffic: None,
        monitor: state.monitor.clone(),
    };
    let response = match state.images.as_ref() {
        Some(backend) => backend.handle(operation, prepared, context).await,
        None => image_error_response(ImageRequestError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Codex Images API is unavailable".to_string(),
            param: None,
            code: Some("images_api_unavailable"),
        }),
    };
    log_native_request_completed(
        &log,
        &req_id,
        operation.label(),
        Some(&model),
        response.status(),
        started_at,
    );
    monitor_response_body(response, request_guard)
}

async fn parse_multipart_image_edit(
    mut multipart: Multipart,
) -> Result<crate::providers::codex::images::PreparedImageRequest, ImageRequestError> {
    let mut input = MultipartEditInput::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| ImageRequestError {
            status: StatusCode::BAD_REQUEST,
            message: "Invalid multipart image edit request".to_string(),
            param: None,
            code: Some("invalid_multipart"),
        })?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "image" | "image[]" => {
                let bytes = field.bytes().await.map_err(|_| ImageRequestError {
                    status: StatusCode::BAD_REQUEST,
                    message: "Failed to read uploaded image".to_string(),
                    param: Some("image"),
                    code: Some("invalid_image"),
                })?;
                input.images.push(UploadedImage {
                    bytes, // EFFICIENCY: field.bytes() already returns owned Bytes, avoid to_vec() copy
                });
            }
            "prompt" => {
                input.prompt = Some(multipart_text(field, "prompt").await?);
            }
            "model" => {
                input.model = Some(multipart_text(field, "model").await?);
            }
            "background" => {
                input.background = Some(multipart_text(field, "background").await?);
            }
            "quality" => {
                input.quality = Some(multipart_text(field, "quality").await?);
            }
            "size" => {
                input.size = Some(multipart_text(field, "size").await?);
            }
            "n" => {
                let raw = multipart_text(field, "n").await?;
                input.n = Some(raw.parse::<u8>().map_err(|_| ImageRequestError {
                    status: StatusCode::BAD_REQUEST,
                    message: "'n' must be an integer between 1 and 10".to_string(),
                    param: Some("n"),
                    code: Some("invalid_request"),
                })?);
            }
            "mask" => {
                return Err(ImageRequestError {
                    status: StatusCode::BAD_REQUEST,
                    message: "Image masks are not supported by the Codex image backend".to_string(),
                    param: Some("mask"),
                    code: Some("unsupported_parameter"),
                });
            }
            _ => {
                return Err(ImageRequestError {
                    status: StatusCode::BAD_REQUEST,
                    message: format!("Unsupported multipart field '{name}'"),
                    param: None,
                    code: Some("unsupported_parameter"),
                });
            }
        }
    }
    prepare_multipart_edit(input)
}

async fn multipart_text(
    field: axum::extract::multipart::Field<'_>,
    param: &'static str,
) -> Result<String, ImageRequestError> {
    field.text().await.map_err(|_| ImageRequestError {
        status: StatusCode::BAD_REQUEST,
        message: format!("Invalid multipart '{param}' field"),
        param: Some(param),
        code: Some("invalid_multipart"),
    })
}

async fn handler_responses(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    crate::config::request_scope(async {
        crate::provider::request_scoped_response(handler_responses_inner(State(state), req).await)
    })
    .await
}

async fn handler_responses_inner(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let path = uri.path().to_string();
    let query = redacted_query(&uri);
    log.info(
        "request",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!(method.as_str())),
            ("path".to_string(), json!(&path)),
            ("query".to_string(), json!(&query)),
        ])),
    );

    let session_id = native_session_id(&headers);
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(&req_id, session_id.clone(), None, EndpointKind::Responses);
    }
    let request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_OPENAI_REQUEST_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            let response = openai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "Request body exceeded the size limit".to_string(),
                None,
                Some("request_too_large"),
            );
            log_native_request_completed(
                &log,
                &req_id,
                "responses",
                None,
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let body: Value = match parse_native_json_body(&body_bytes) {
        Ok(body) => body,
        Err(response) => {
            log_native_request_completed(
                &log,
                &req_id,
                "responses",
                None,
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let (requested_model, normalized_model) = match extract_model(&body) {
        Ok(model) => model,
        Err(error) => return monitor_response_body(error.response(), request_guard),
    };
    let now = current_millis();
    let session_state = session::existing_session(session_id.as_deref(), now);
    let affinity = session_state
        .as_ref()
        .and_then(|session| session.affinity_provider.as_ref());
    let Some(provider) = state
        .registry
        .provider_for_model(&normalized_model, affinity)
    else {
        let response = OpenAiError::invalid(
            format!(
                "Unknown model \"{normalized_model}\". {}",
                state.registry.unknown_model_message()
            ),
            Some("model"),
        )
        .response();
        return monitor_response_body(response, request_guard);
    };
    let parsed = if provider.name() == "codex" {
        if let Err(response) = validate_native_request_model(&body) {
            return monitor_response_body(response, request_guard);
        }
        None
    } else {
        match parse_request(
            OpenAiSurface::Responses,
            body.clone(),
            provider.name(),
            session_id.as_deref(),
        ) {
            Ok(parsed) => Some(parsed),
            Err(error) => return monitor_response_body(error.response(), request_guard),
        }
    };
    if provider.name() != "codex"
        && let Some(session_id) = session_id.as_deref()
    {
        crate::providers::codex::clear_session_compaction(session_id);
    }
    let current = session::record_session_request(
        session_id.as_deref(),
        session_state.as_ref(),
        provider.name(),
        &normalized_model,
        now,
    );
    let effort = parsed
        .as_ref()
        .and_then(|parsed| {
            parsed
                .messages
                .extra
                .get("output_config")
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str)
        })
        .or_else(|| body.pointer("/reasoning/effort").and_then(Value::as_str))
        .map(str::to_string);
    if let Some(monitor) = state.monitor.as_ref() {
        if let Some(current) = current.as_ref() {
            monitor.session_sequence_resolved(&req_id, current.seq);
        }
        monitor.provider_selected(&req_id, provider.name(), &normalized_model, effort);
    }
    let traffic = create_traffic_capture(TrafficCaptureOptions {
        req_id: req_id.clone(),
        session_id: session_id.clone(),
        session_seq: current.as_ref().map(|session| session.seq),
        provider: Some(provider.name().to_string()),
        state_dir_override: None,
    })
    .map(Arc::new);
    if let Some(capture) = traffic.as_ref() {
        if let Some(monitor) = state.monitor.as_ref() {
            monitor.traffic_capture_path(&req_id, capture.root().to_path_buf());
        }
        capture.write_json(
            "000-metadata",
            &json!({
                "reqId": &req_id,
                "sessionId": &session_id,
                "sessionSeq": current.as_ref().map(|session| session.seq),
                "kind": "responses",
                "provider": provider.name(),
                "model": &normalized_model,
                "requestedModel": &requested_model,
                "method": method.as_str(),
                "path": &path,
                "query": &query,
                "headers": headers_to_record(&headers),
            }),
        );
        capture.write_json("010-openai-responses-request", &body);
        if let Some(parsed) = parsed.as_ref() {
            capture.write_json(
                "015-anthropic-request",
                &serde_json::to_value(&parsed.messages).unwrap_or_else(|_| json!({})),
            );
        }
    }
    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: current.map(|session| session.seq),
        provider: provider.name().to_string(),
        traffic: traffic.clone(),
        monitor: state.monitor.clone(),
    };
    let response = if let Some(parsed) = parsed {
        match provider
            .generate_anthropic_stream(parsed.messages, context)
            .await
        {
            Ok(generation) => {
                if let Some(capture) = traffic.as_ref() {
                    capture.write_json(
                        "016-model-resolution",
                        &json!({"requestedModel":requested_model,"resolvedModel":generation.resolved_model}),
                    );
                }
                match render_openai_response(
                    OpenAiSurface::Responses,
                    generation,
                    parsed.stream,
                    parsed.include_usage,
                    parsed.response_metadata,
                    traffic.clone(),
                )
                .await
                {
                    Ok(response) => response,
                    Err(error) => error.response(),
                }
            }
            Err(error) => OpenAiError::from(error).response(),
        }
    } else {
        match state.native_responses.as_ref() {
            Some(backend) => backend.handle(body, context).await,
            None => openai_error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                "Native Responses API is disabled",
                None,
                None,
            ),
        }
    };
    log_routed_openai_request_completed(
        &log,
        &req_id,
        "responses",
        provider.name(),
        Some(&normalized_model),
        response.status(),
        started_at,
    );
    monitor_response_body(response, request_guard)
}

async fn handler_chat_completions(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    crate::config::request_scope(async {
        crate::provider::request_scoped_response(
            handler_chat_completions_inner(State(state), req).await,
        )
    })
    .await
}

async fn handler_chat_completions_inner(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let path = uri.path().to_string();
    let query = redacted_query(&uri);
    log.info(
        "request",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!(method.as_str())),
            ("path".to_string(), json!(&path)),
            ("query".to_string(), json!(&query)),
        ])),
    );

    let session_id = native_session_id(&headers);
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(
            &req_id,
            session_id.clone(),
            None,
            EndpointKind::ChatCompletions,
        );
    }
    let request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_OPENAI_REQUEST_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            let response = openai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "Request body exceeded the size limit".to_string(),
                None,
                Some("request_too_large"),
            );
            log_native_request_completed(
                &log,
                &req_id,
                "chat_completions",
                None,
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let body = match parse_native_json_body(&body_bytes) {
        Ok(body) => body,
        Err(response) => {
            log_native_request_completed(
                &log,
                &req_id,
                "chat_completions",
                None,
                response.status(),
                started_at,
            );
            return monitor_response_body(response, request_guard);
        }
    };
    let (requested_model, normalized_model) = match extract_model(&body) {
        Ok(model) => model,
        Err(error) => return monitor_response_body(error.response(), request_guard),
    };
    let now = current_millis();
    let session_state = session::existing_session(session_id.as_deref(), now);
    let affinity = session_state
        .as_ref()
        .and_then(|session| session.affinity_provider.as_ref());
    let Some(provider) = state
        .registry
        .provider_for_model(&normalized_model, affinity)
    else {
        let response = OpenAiError::invalid(
            format!(
                "Unknown model \"{normalized_model}\". {}",
                state.registry.unknown_model_message()
            ),
            Some("model"),
        )
        .response();
        return monitor_response_body(response, request_guard);
    };
    let (translated, parsed) = if provider.name() == "codex" {
        let translated = match translate_request(body.clone()) {
            Ok(translated) => translated,
            Err(error) => return monitor_response_body(error.response(), request_guard),
        };
        (Some(translated), None)
    } else {
        let parsed = match parse_request(
            OpenAiSurface::ChatCompletions,
            body.clone(),
            provider.name(),
            session_id.as_deref(),
        ) {
            Ok(parsed) => parsed,
            Err(error) => return monitor_response_body(error.response(), request_guard),
        };
        (None, Some(parsed))
    };
    if provider.name() != "codex"
        && let Some(session_id) = session_id.as_deref()
    {
        crate::providers::codex::clear_session_compaction(session_id);
    }
    let current = session::record_session_request(
        session_id.as_deref(),
        session_state.as_ref(),
        provider.name(),
        &normalized_model,
        now,
    );
    let effort = translated
        .as_ref()
        .and_then(|translated| translated.effort.clone())
        .or_else(|| {
            parsed.as_ref().and_then(|parsed| {
                parsed
                    .messages
                    .extra
                    .get("output_config")
                    .and_then(|value| value.get("effort"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
        });
    if let Some(monitor) = state.monitor.as_ref() {
        if let Some(current) = current.as_ref() {
            monitor.session_sequence_resolved(&req_id, current.seq);
        }
        monitor.provider_selected(&req_id, provider.name(), &normalized_model, effort.clone());
    }
    let traffic = create_traffic_capture(TrafficCaptureOptions {
        req_id: req_id.clone(),
        session_id: session_id.clone(),
        session_seq: current.as_ref().map(|session| session.seq),
        provider: Some(provider.name().to_string()),
        state_dir_override: None,
    })
    .map(Arc::new);
    if let Some(capture) = traffic.as_ref() {
        if let Some(monitor) = state.monitor.as_ref() {
            monitor.traffic_capture_path(&req_id, capture.root().to_path_buf());
        }
        capture.write_json(
            "000-metadata",
            &json!({
                "reqId": &req_id,
                "sessionId": &session_id,
                "sessionSeq": current.as_ref().map(|session| session.seq),
                "kind": "chat_completions",
                "provider": provider.name(),
                "model": &normalized_model,
                "requestedModel": &requested_model,
                "effort": &effort,
                "method": method.as_str(),
                "path": &path,
                "query": &query,
                "headers": headers_to_record(&headers),
            }),
        );
        capture.write_json("010-openai-chat-completions-request", &body);
        if let Some(translated) = translated.as_ref() {
            capture.write_json("020-upstream-request", &translated.upstream);
        }
        if let Some(parsed) = parsed.as_ref() {
            capture.write_json(
                "015-anthropic-request",
                &serde_json::to_value(&parsed.messages).unwrap_or_else(|_| json!({})),
            );
        }
    }
    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: current.map(|session| session.seq),
        provider: provider.name().to_string(),
        traffic: traffic.clone(),
        monitor: state.monitor.clone(),
    };
    let response = if let Some(translated) = translated {
        match state.chat_completions.as_ref() {
            Some(backend) => backend.handle(translated, context).await,
            None => openai_error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                "Chat Completions API is disabled",
                None,
                None,
            ),
        }
    } else {
        let parsed = parsed.expect("non-Codex request was parsed");
        match provider
            .generate_anthropic_stream(parsed.messages, context)
            .await
        {
            Ok(generation) => {
                if let Some(capture) = traffic.as_ref() {
                    capture.write_json(
                        "016-model-resolution",
                        &json!({"requestedModel":requested_model,"resolvedModel":generation.resolved_model}),
                    );
                }
                match render_openai_response(
                    OpenAiSurface::ChatCompletions,
                    generation,
                    parsed.stream,
                    parsed.include_usage,
                    parsed.response_metadata,
                    traffic.clone(),
                )
                .await
                {
                    Ok(response) => response,
                    Err(error) => error.response(),
                }
            }
            Err(error) => OpenAiError::from(error).response(),
        }
    };
    log_routed_openai_request_completed(
        &log,
        &req_id,
        "chat_completions",
        provider.name(),
        Some(&normalized_model),
        response.status(),
        started_at,
    );
    monitor_response_body(response, request_guard)
}

fn native_session_id(headers: &http::HeaderMap) -> Option<String> {
    [
        "x-claude-code-session-id",
        "session_id",
        "x-client-request-id",
    ]
    .into_iter()
    .find_map(|name| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

#[allow(clippy::result_large_err)]
fn parse_native_json_body(body: &[u8]) -> Result<Value, Response> {
    if body.is_empty() {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Invalid JSON: empty body",
            None,
            Some("invalid_json"),
        ));
    }
    serde_json::from_slice(body).map_err(|error| {
        openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Invalid JSON: {error}"),
            None,
            Some("invalid_json"),
        )
    })
}

fn log_routed_openai_request_completed(
    log: &Logger,
    req_id: &str,
    endpoint: &str,
    provider: &str,
    model: Option<&str>,
    status: StatusCode,
    started_at: Instant,
) {
    log.info(
        "request_completed",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(req_id)),
            ("endpoint".to_string(), json!(endpoint)),
            ("provider".to_string(), json!(provider)),
            ("model".to_string(), json!(model)),
            ("countTokens".to_string(), json!(false)),
            ("status".to_string(), json!(status.as_u16())),
            ("ms".to_string(), json!(started_at.elapsed().as_millis())),
            ("timingPhase".to_string(), json!("response-ready")),
            ("phase".to_string(), json!("response_ready")),
            (
                "outcome".to_string(),
                json!(if status.is_success() {
                    "pending"
                } else {
                    "failed"
                }),
            ),
        ])),
    );
}

fn log_native_request_completed(
    log: &Logger,
    req_id: &str,
    endpoint: &str,
    model: Option<&str>,
    status: StatusCode,
    started_at: Instant,
) {
    log_routed_openai_request_completed(log, req_id, endpoint, "codex", model, status, started_at);
}

async fn dispatch_request(
    state: Arc<AppState>,
    req: Request<Body>,
    count_tokens: bool,
) -> Response {
    crate::config::request_scope(async {
        crate::provider::request_scoped_response(
            dispatch_request_inner(state, req, count_tokens).await,
        )
    })
    .await
}

async fn dispatch_request_inner(
    state: Arc<AppState>,
    req: Request<Body>,
    count_tokens: bool,
) -> Response {
    let started_at = Instant::now();
    let log = create_logger("server");
    let req_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();
    let conversation_identity = (!count_tokens)
        .then(|| ConversationIdentity::from_headers(&headers))
        .flatten();
    let path = uri.path().to_string();
    let query = redacted_query(&uri);
    let endpoint = if count_tokens {
        EndpointKind::CountTokens
    } else {
        EndpointKind::Messages
    };
    log.info(
        "request",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(&req_id)),
            ("method".to_string(), json!(method.as_str())),
            ("path".to_string(), json!(&path)),
            ("query".to_string(), json!(&query)),
        ])),
    );
    let session_id = headers
        .get("x-claude-code-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    // Provider and telemetry context uses the raw session header; trusted
    // session state uses the canonical control-command identity when available.
    let state_session_id = match conversation_identity.as_ref() {
        Some(ConversationIdentity::Main(session) | ConversationIdentity::Agent(session, _)) => {
            Some(session.as_str())
        }
        None => session_id.as_deref(),
    };
    if let Some(monitor) = state.monitor.as_ref() {
        monitor.request_started(&req_id, session_id.clone(), None, endpoint);
    }
    let mut request_guard = RequestMonitorGuard::new(state.monitor.clone(), req_id.clone());
    let now = current_millis();
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_ANTHROPIC_REQUEST_BYTES).await
    {
        Ok(bytes) => bytes,
        Err(_) => {
            let response = json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                "Request body exceeded the size limit",
            );
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            request_guard.failed(
                response.status(),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Invalid JSON")
                    .to_string(),
            );
            return with_request_id(response, &req_id);
        }
    };

    let mut body: crate::anthropic::schema::MessagesRequest = match parse_json_body(&body_bytes) {
        Ok(body) => body,
        Err(response) => {
            let status = response.status();
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                *response,
            )
            .await;
            request_guard.failed(
                status,
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Invalid JSON")
                    .to_string(),
            );
            return with_request_id(response, &req_id);
        }
    };

    if let Some(thread) = body.extra.remove("thread")
        && thread.get("type").and_then(Value::as_str) != Some("create")
    {
        let message = "capability_rejected: beta_header:message-threads-2026-08-12";
        let response = json_error(StatusCode::BAD_REQUEST, "invalid_request_error", message);
        log_request_completed(
            &log,
            RequestLogContext {
                req_id: &req_id,
                provider: None,
                model: None,
                count_tokens,
                status: response.status(),
                started_at,
            },
        );
        let (response, details) = record_failed_response(
            &log,
            FailedResponseLogContext {
                req_id: &req_id,
                provider: None,
                model: None,
                count_tokens,
                started_at,
            },
            response,
        )
        .await;
        request_guard.failed(
            response.status(),
            details
                .as_ref()
                .map(|details| details.message.clone())
                .unwrap_or_else(|| message.to_string()),
        );
        return with_request_id(response, &req_id);
    }

    if let Some(project) = project::name_from_request(
        body.extra.get("system"),
        body.messages.iter().rev().map(|message| &message.content),
    ) && let Some(monitor) = state.monitor.as_ref()
    {
        monitor.project_resolved(&req_id, project);
    }

    let model = match body.model.as_deref() {
        Some(model) => model,
        None => {
            let response = json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Missing \"model\" in request body. {}",
                    state.registry.unknown_model_message()
                ),
            );
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: None,
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            request_guard.failed(
                response.status(),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Missing model")
                    .to_string(),
            );
            return with_request_id(response, &req_id);
        }
    };

    let mut normalized_model = normalize_incoming_model(model);
    body.model = Some(normalized_model.clone());
    let session_state = if let Some(session_id) = state_session_id {
        session::existing_session(Some(session_id), now)
    } else {
        None
    };
    let session_affinity = session_state
        .as_ref()
        .and_then(|state| state.affinity_provider.as_ref());
    let original_provider = state
        .registry
        .provider_for_model(&normalized_model, session_affinity);
    let configured_auto_review_model = crate::config::auto_review_model();
    let auto_review_route = original_provider.as_ref().and_then(|provider| {
        apply_auto_review_model(
            &mut body,
            count_tokens,
            configured_auto_review_model.as_deref(),
            provider.name(),
        )
    });
    if auto_review_route.is_some() {
        normalized_model = normalize_incoming_model(body.model.as_deref().expect("override model"));
        body.model = Some(normalized_model.clone());
    }

    let provider = if auto_review_route.is_some() {
        state.registry.provider_for_model(&normalized_model, None)
    } else {
        original_provider
    };

    let provider = match provider {
        Some(provider) => provider,
        None => {
            log.warn(
                "unknown model",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), json!(&req_id)),
                    ("model".to_string(), json!(&normalized_model)),
                ])),
            );
            let response = json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Unknown model \"{normalized_model}\". {}",
                    state.registry.unknown_model_message()
                ),
            );
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: Some(&normalized_model),
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            let (response, details) = record_failed_response(
                &log,
                FailedResponseLogContext {
                    req_id: &req_id,
                    provider: None,
                    model: Some(&normalized_model),
                    count_tokens,
                    started_at,
                },
                response,
            )
            .await;
            request_guard.failed(
                response.status(),
                details
                    .as_ref()
                    .map(|details| details.message.as_str())
                    .unwrap_or("Unknown model")
                    .to_string(),
            );
            return with_request_id(response, &req_id);
        }
    };

    // Parse only the current explicitly expanded user invocation. Classifier
    // requests must never receive an acknowledgment as a security-review answer.
    let control = if count_tokens {
        Ok(None)
    } else {
        crate::proxy_speed::parse_current_command(&body, &headers)
            .map_err(|error| error.to_string())
            .and_then(|command| {
                if command.is_some() {
                    if auto_review_route.is_some() {
                        return Err("Proxy-speed cannot be changed by an auxiliary auto-review request".to_string());
                    }
                    if provider.name() != "codex" {
                        return Err("Proxy-speed applies only to Codex models".to_string());
                    }
                    let resolved = crate::providers::codex::translate::model_allowlist::resolve_model_request_with_config_override(&normalized_model, true);
                    crate::providers::codex::translate::model_allowlist::assert_allowed_model(&resolved.model)
                        .map_err(|_| format!("Unsupported resolved Codex model: {}", resolved.model))?;
                    crate::providers::translate_shared::read_effort(&body).map_err(|error| error.to_string())?;
                }
                Ok(command)
            })
    };
    let command = match control {
        Ok(command) => command,
        Err(message) => {
            let response = json_error(StatusCode::BAD_REQUEST, "invalid_request_error", &message);
            log_request_completed(
                &log,
                RequestLogContext {
                    req_id: &req_id,
                    provider: Some(provider.name()),
                    model: Some(&normalized_model),
                    count_tokens,
                    status: response.status(),
                    started_at,
                },
            );
            request_guard.failed(response.status(), message);
            return with_request_id(response, &req_id);
        }
    };
    if let Some(command) = command {
        let current = session::record_session_request_with_speed(
            Some(&command.session_id),
            session_state.as_ref(),
            provider.name(),
            &normalized_model,
            command.mode,
            now,
        )
        .expect("validated command has a session");
        log.info(
            "proxy_speed_changed",
            Some(Map::from_iter([
                ("reqId".into(), json!(&req_id)),
                ("sessionId".into(), json!(&command.session_id)),
                ("sessionSeq".into(), json!(current.seq)),
                ("mode".into(), json!(command.mode.as_str())),
            ])),
        );
        if let Some(monitor) = state.monitor.as_ref() {
            monitor.session_sequence_resolved(&req_id, current.seq);
        }
        let response =
            crate::proxy_speed::acknowledgment_response(&command, &normalized_model, body.stream);
        log_request_completed(
            &log,
            RequestLogContext {
                req_id: &req_id,
                provider: None,
                model: Some(&normalized_model),
                count_tokens,
                status: response.status(),
                started_at,
            },
        );
        return monitor_response_body(response, request_guard);
    }
    // Provider/counting copies never contain executable control history. The
    // source request is not rewritten, and ordinary quoted/tool data stays data.
    body = crate::proxy_speed::sanitized_request(&body);

    body.bypass_provider_model_override = auto_review_route.is_some() && provider.name() == "codex";

    if let Some(route) = auto_review_route.as_ref() {
        log.info(
            "auto-review route selected",
            Some(Map::from_iter([
                ("reqId".to_string(), json!(&req_id)),
                ("requestedModel".to_string(), json!(&route.requested_model)),
                ("overrideModel".to_string(), json!(&route.override_model)),
                ("provider".to_string(), json!(provider.name())),
            ])),
        );
    }

    let effort = crate::providers::translate_shared::read_effort(&body)
        .ok()
        .flatten()
        .map(str::to_string);
    let current = session::record_session_request_with_affinity_update(
        state_session_id,
        session_state.as_ref(),
        provider.name(),
        &normalized_model,
        auto_review_route.is_none(),
        now,
    );
    if !count_tokens
        && provider.name() == "codex"
        && auto_review_route.is_none()
        && conversation_identity.is_some()
    {
        body.proxy_speed_snapshot = current.as_ref().and_then(|state| state.proxy_speed);
    }
    if let Some(monitor) = state.monitor.as_ref() {
        if let Some(current) = current.as_ref() {
            monitor.session_sequence_resolved(&req_id, current.seq);
        }
        monitor.provider_selected(&req_id, provider.name(), &normalized_model, effort);
    }

    let traffic = create_traffic_capture(TrafficCaptureOptions {
        req_id: req_id.clone(),
        session_id: session_id.clone(),
        session_seq: current.as_ref().map(|s| s.seq),
        provider: Some(provider.name().to_string()),
        state_dir_override: None,
    })
    .map(Arc::new);

    if let Some(capture) = traffic.as_ref() {
        if let Some(monitor) = state.monitor.as_ref() {
            monitor.traffic_capture_path(&req_id, capture.root().to_path_buf());
        }
        capture.write_json(
            "000-metadata",
            &json!({
                "reqId": &req_id,
                "sessionId": &session_id,
                "sessionSeq": current.as_ref().map(|s| s.seq),
                "kind": if count_tokens { "count_tokens" } else { "messages" },
                "provider": provider.name(),
                "model": &normalized_model,
                "method": method.as_str(),
                "path": &path,
                "query": &query,
                "headers": headers_to_record(&headers),
            }),
        );
        capture.write_json(
            "010-anthropic-request",
            &serde_json::to_value(&body).unwrap_or_else(|_| json!({})),
        );
    }

    let context = RequestContext {
        req_id: req_id.clone(),
        session_id,
        session_seq: current.map(|s| s.seq),
        provider: provider.name().to_string(),
        traffic,
        monitor: state.monitor.clone(),
    };

    let response = if count_tokens {
        provider.handle_count_tokens(body, context).await
    } else {
        let stream = body.stream;
        let first_heartbeat_after =
            if provider.name() == "codex" && is_compact_messages_request(&body) {
                std::time::Duration::ZERO
            } else {
                PENDING_PROVIDER_HEARTBEAT
            };
        let request_provider = Arc::clone(&provider);
        let response = async move {
            request_provider
                .handle_messages_with_conversation_identity(
                    body,
                    context,
                    if auto_review_route.is_some() {
                        None
                    } else {
                        conversation_identity
                    },
                )
                .await
        };
        if stream {
            keep_pending_provider_alive(
                response,
                log.clone(),
                PendingProviderContext {
                    req_id: req_id.clone(),
                    provider: provider.name(),
                    model: normalized_model.clone(),
                    started_at,
                    first_heartbeat_after,
                },
            )
            .await
        } else {
            response.await
        }
    };
    log_request_completed(
        &log,
        RequestLogContext {
            req_id: &req_id,
            provider: Some(provider.name()),
            model: Some(&normalized_model),
            count_tokens,
            status: response.status(),
            started_at,
        },
    );
    let status = response.status();
    if status.is_success() {
        return monitor_response_body(response, request_guard);
    }

    let (response, details) = record_failed_response(
        &log,
        FailedResponseLogContext {
            req_id: &req_id,
            provider: Some(provider.name()),
            model: Some(&normalized_model),
            count_tokens,
            started_at,
        },
        response,
    )
    .await;
    if let Some(details) = details.as_ref() {
        request_guard.failed(status, details.message.clone());
    } else {
        request_guard.failed(status, format!("HTTP {}", status.as_u16()));
    }
    with_request_id(response, &req_id)
}

const PENDING_PROVIDER_HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(15);
const PENDING_PROVIDER_PING: &[u8] = b"event: ping\ndata: {\"type\":\"ping\"}\n\n";

struct PendingProviderContext {
    first_heartbeat_after: std::time::Duration,
    req_id: String,
    provider: &'static str,
    model: String,
    started_at: Instant,
}

#[derive(Clone, Default)]
struct PendingProviderOutcome(Arc<std::sync::OnceLock<(StatusCode, String)>>);

impl PendingProviderOutcome {
    fn failure(&self) -> Option<(StatusCode, String)> {
        self.0.get().cloned()
    }
}

/// Preserve ordinary HTTP responses while the provider is quick. Once waiting
/// requires a heartbeat, own the same future in the body so downstream
/// cancellation drops it, including any provider retry/connection guards.
async fn keep_pending_provider_alive(
    response: impl Future<Output = Response> + Send + 'static,
    log: Logger,
    ctx: PendingProviderContext,
) -> Response {
    let mut response = Box::pin(response);
    tokio::select! {
        biased;
        response = &mut response => return response,
        _ = tokio::time::sleep(ctx.first_heartbeat_after) => {}
    }

    let log = log.child(Map::from_iter([
        ("reqId".into(), json!(ctx.req_id)),
        ("provider".into(), json!(ctx.provider)),
        ("model".into(), json!(ctx.model)),
    ]));
    log.info(
        "provider_pending_heartbeat_started",
        Some(Map::from_iter([(
            "ms".into(),
            json!(ctx.started_at.elapsed().as_millis()),
        )])),
    );
    let outcome = PendingProviderOutcome::default();
    let pending_outcome = outcome.clone();
    let prepare_body = async move {
        let response = response.await;
        let status = response.status();
        log.info(
            "pending_provider_response_ready",
            Some(Map::from_iter([
                ("status".into(), json!(status.as_u16())),
                ("ms".into(), json!(ctx.started_at.elapsed().as_millis())),
                ("headers".into(), headers_to_record(response.headers())),
            ])),
        );
        if status.is_success() {
            return response.into_body();
        }
        let (response, details) = record_failed_response(
            &log,
            FailedResponseLogContext {
                req_id: &ctx.req_id,
                provider: Some(ctx.provider),
                model: Some(&ctx.model),
                count_tokens: false,
                started_at: ctx.started_at,
            },
            response,
        )
        .await;
        let message = details
            .map(|details| details.message)
            .unwrap_or_else(|| format!("HTTP {}", status.as_u16()));
        let _ = pending_outcome.0.set((status, message.clone()));
        let bytes = response
            .into_body()
            .collect()
            .await
            .map(|body| body.to_bytes())
            .unwrap_or_default();
        let mut error = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .filter(|value| {
                value["error"]["type"].is_string() && value["error"]["message"].is_string()
            })
            .unwrap_or_else(|| {
                let kind = match status.as_u16() {
                    400 => "invalid_request_error",
                    401 => "authentication_error",
                    403 => "permission_error",
                    404 => "not_found_error",
                    413 => "request_too_large",
                    429 => "rate_limit_error",
                    529 => "overloaded_error",
                    _ => "api_error",
                };
                json!({"error": {"type": kind, "message": message}})
            });
        error["type"] = json!("error");
        Body::from(format!("event: error\ndata: {error}\n\n"))
    };
    let mut heartbeat = tokio::time::interval(PENDING_PROVIDER_HEARTBEAT);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let stream = futures_util::stream::unfold(
        (Some(Box::pin(prepare_body)), None::<Body>, heartbeat),
        |(mut pending, mut body, mut heartbeat)| async move {
            if let Some(response) = pending.as_mut() {
                tokio::select! {
                    biased;
                    ready = response => { body = Some(ready); pending = None; }
                    _ = heartbeat.tick() => {
                        return Some((Ok(http_body::Frame::data(bytes::Bytes::from_static(PENDING_PROVIDER_PING))), (pending, body, heartbeat)));
                    }
                }
            }
            body.as_mut()
                .expect("provider body is ready")
                .frame()
                .await
                .map(|frame| (frame, (pending, body, heartbeat)))
        },
    );
    let mut response = (
        [
            (http::header::CONTENT_TYPE, "text/event-stream"),
            (http::header::CACHE_CONTROL, "no-cache"),
            (http::header::CONNECTION, "keep-alive"),
        ],
        Body::new(StreamBody::new(stream)),
    )
        .into_response();
    response.extensions_mut().insert(outcome);
    response
}

/// Header Claude Code reads to populate the `requestId` field it writes into
/// every transcript record.
const REQUEST_ID_HEADER: &str = "request-id";

/// Supply an id for transcript de-duplication, preserving any upstream id.
fn stamp_request_id(headers: &mut http::HeaderMap, req_id: &str) {
    if !headers.contains_key(REQUEST_ID_HEADER)
        && let Ok(value) = http::HeaderValue::from_str(req_id)
    {
        headers.insert(REQUEST_ID_HEADER, value);
    }
}

fn with_request_id(mut response: Response, req_id: &str) -> Response {
    stamp_request_id(response.headers_mut(), req_id);
    response
}

fn monitor_response_body(response: Response, mut guard: RequestMonitorGuard) -> Response {
    guard.body_pending = true;
    guard.delivery(DeliveryEvent::ResponseReady);
    let status = response.status();
    let outcome = response
        .extensions()
        .get::<NativeResponseOutcome>()
        .cloned();
    let pending_outcome = response
        .extensions()
        .get::<PendingProviderOutcome>()
        .cloned();
    let (mut parts, body) = response.into_parts();
    // Stamped on the parts before the body is streamed, so a streaming SSE
    // response carries the header too.
    stamp_request_id(&mut parts.headers, &guard.req_id);
    let stream = futures_util::stream::unfold(
        (body, guard, outcome, pending_outcome),
        move |(mut body, mut guard, outcome, pending_outcome)| async move {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Some((status, error)) = pending_outcome
                        .as_ref()
                        .and_then(PendingProviderOutcome::failure)
                    {
                        guard.failed(status, error);
                    }
                    guard.delivery(DeliveryEvent::BodyYield {
                        bytes: frame.data_ref().map_or(0, |data| data.len() as u64),
                    });
                    Some((Ok(frame), (body, guard, outcome, pending_outcome)))
                }
                Some(Err(err)) => {
                    guard.failed(status, err.to_string());
                    Some((Err(err), (body, guard, outcome, pending_outcome)))
                }
                None => {
                    guard.delivery(DeliveryEvent::BodyFinished);
                    if let Some(message) = outcome.as_ref().and_then(NativeResponseOutcome::failure)
                    {
                        guard.failed(status, message);
                    } else if status.is_success() {
                        guard.completed(status);
                    } else {
                        guard.failed(status, format!("HTTP {}", status.as_u16()));
                    }
                    None
                }
            }
        },
    );
    Response::from_parts(parts, Body::new(StreamBody::new(stream)))
}

struct RequestLogContext<'a> {
    req_id: &'a str,
    provider: Option<&'a str>,
    model: Option<&'a str>,
    count_tokens: bool,
    status: StatusCode,
    started_at: Instant,
}

fn log_request_completed(log: &Logger, ctx: RequestLogContext<'_>) {
    log.info(
        "request_completed",
        Some(serde_json::Map::from_iter([
            ("reqId".to_string(), json!(ctx.req_id)),
            ("provider".to_string(), json!(ctx.provider)),
            ("model".to_string(), json!(ctx.model)),
            ("countTokens".to_string(), json!(ctx.count_tokens)),
            ("status".to_string(), json!(ctx.status.as_u16())),
            (
                "ms".to_string(),
                json!(ctx.started_at.elapsed().as_millis()),
            ),
            ("timingPhase".to_string(), json!("response-ready")),
            ("phase".to_string(), json!("response_ready")),
            (
                "outcome".to_string(),
                json!(if ctx.status.is_success() {
                    "pending"
                } else {
                    "failed"
                }),
            ),
        ])),
    );
}

struct FailedResponseLogContext<'a> {
    req_id: &'a str,
    provider: Option<&'a str>,
    model: Option<&'a str>,
    count_tokens: bool,
    started_at: Instant,
}

struct FailedResponseDetails {
    message: String,
}

async fn record_failed_response(
    log: &Logger,
    ctx: FailedResponseLogContext<'_>,
    response: Response,
) -> (Response, Option<FailedResponseDetails>) {
    if response.status().is_success() {
        return (response, None);
    }

    let status = response.status();
    let (parts, body) = response.into_parts();
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            log.info(
                "request_failed",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), json!(ctx.req_id)),
                    ("provider".to_string(), json!(ctx.provider)),
                    ("model".to_string(), json!(ctx.model)),
                    ("countTokens".to_string(), json!(ctx.count_tokens)),
                    ("status".to_string(), json!(status.as_u16())),
                    (
                        "ms".to_string(),
                        json!(ctx.started_at.elapsed().as_millis()),
                    ),
                    ("bodyReadError".to_string(), json!(err.to_string())),
                ])),
            );
            return (Response::from_parts(parts, Body::empty()), None);
        }
    };

    let response_body = response_body_value(&bytes);
    let message = error_message_from_response(&response_body)
        .unwrap_or_else(|| format!("HTTP {}", status.as_u16()));
    let document = json!({
        "reqId": ctx.req_id,
        "provider": ctx.provider,
        "model": ctx.model,
        "countTokens": ctx.count_tokens,
        "status": status.as_u16(),
        "elapsedMs": ctx.started_at.elapsed().as_millis(),
        "message": message,
        "response": response_body,
    });
    let error_file = write_error_capture(ctx.req_id, &document);

    let mut fields = serde_json::Map::from_iter([
        ("reqId".to_string(), json!(ctx.req_id)),
        ("provider".to_string(), json!(ctx.provider)),
        ("model".to_string(), json!(ctx.model)),
        ("countTokens".to_string(), json!(ctx.count_tokens)),
        ("status".to_string(), json!(status.as_u16())),
        (
            "ms".to_string(),
            json!(ctx.started_at.elapsed().as_millis()),
        ),
        ("message".to_string(), json!(message)),
    ]);
    if let Some(path) = error_file.as_ref() {
        fields.insert("errorFile".to_string(), json!(path.display().to_string()));
    }
    log.info("request_failed", Some(fields));

    (
        Response::from_parts(parts, Body::from(bytes)),
        Some(FailedResponseDetails { message }),
    )
}

fn response_body_value(bytes: &[u8]) -> Value {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) => json!({ "json": value }),
        Err(_) => json!({ "text": String::from_utf8_lossy(bytes) }),
    }
}

fn error_message_from_response(response_body: &Value) -> Option<String> {
    response_body
        .get("json")
        .and_then(|body| body.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| {
            response_body
                .get("text")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .map(std::string::ToString::to_string)
}

fn write_error_capture(req_id: &str, document: &Value) -> Option<PathBuf> {
    use crate::logging::writer::{self, Record};
    if crate::bounded_json_size(document, writer::MAX_RECORD_BYTES - 4096)
        .ok()
        .flatten()
        .is_none()
    {
        writer::global().omit(None);
        return None;
    }
    let dir = crate::paths::state_dir().join("errors");
    let path = dir.join(format!(
        "{}-{}.json",
        current_millis(),
        sanitize_path_part(req_id)
    ));
    let Some(payload) = writer::bounded_json(&crate::traffic::redact_traffic(document), true)
    else {
        writer::global().omit(None);
        return None;
    };
    writer::global()
        .enqueue(Record::file(path.clone(), payload))
        .then_some(path)
}

fn sanitize_path_part(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

struct RequestMonitorGuard {
    monitor: Option<MonitorHandle>,
    req_id: String,
    body_pending: bool,
    request_finalized: bool,
    started_at: Instant,
}

impl RequestMonitorGuard {
    fn new(monitor: Option<MonitorHandle>, req_id: String) -> Self {
        Self {
            monitor,
            req_id,
            body_pending: false,
            request_finalized: false,
            started_at: Instant::now(),
        }
    }

    fn delivery(&mut self, event: DeliveryEvent) {
        if matches!(event, DeliveryEvent::BodyFinished) {
            self.body_pending = false;
        }
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.delivery(&self.req_id, event);
        }
    }

    fn completed(&mut self, status: StatusCode) {
        if self.request_finalized {
            return;
        }
        self.request_finalized = true;
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.request_completed(&self.req_id, status.as_u16(), None, None);
        }
    }

    fn failed(&mut self, status: StatusCode, error: String) {
        if self.request_finalized {
            return;
        }
        self.request_finalized = true;
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.request_failed(&self.req_id, Some(status.as_u16()), error);
        }
    }
}

impl Drop for RequestMonitorGuard {
    fn drop(&mut self) {
        if self.body_pending || !self.request_finalized {
            create_logger("server").info(
                "downstream_request_cancelled",
                Some(Map::from_iter([
                    ("reqId".into(), json!(self.req_id)),
                    (
                        "elapsedMs".into(),
                        json!(self.started_at.elapsed().as_millis()),
                    ),
                    (
                        "phase".into(),
                        json!(if self.body_pending {
                            "response-body"
                        } else {
                            "request-handler"
                        }),
                    ),
                ])),
            );
        }
        if let Some(monitor) = self.monitor.as_ref() {
            if self.body_pending {
                monitor.delivery(&self.req_id, DeliveryEvent::BodyCancelled);
            }
            if !self.request_finalized {
                monitor.request_abandoned(&self.req_id, "Request future ended before completion");
            }
        }
    }
}

fn headers_to_record(headers: &http::HeaderMap) -> Value {
    let mut out = Map::new();
    for (key, value) in headers {
        if let Ok(raw) = value.to_str() {
            let value = if matches!(
                key.as_str(),
                CLAUDE_AGENT_HEADER | CLAUDE_PARENT_AGENT_HEADER
            ) {
                format!("[redacted len={}]", raw.len())
            } else {
                raw.to_string()
            };
            out.insert(key.as_str().to_string(), Value::String(value));
        }
    }
    Value::Object(out)
}

fn redacted_query(uri: &http::Uri) -> Value {
    let mut out = Map::new();
    let Some(query) = uri.query() else {
        return Value::Object(out);
    };
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let key = key.into_owned();
        let lower = key.to_lowercase();
        let value = if REDACT_KEYS.contains(&lower.as_str()) {
            Value::String(format!("[redacted len={}]", value.len()))
        } else {
            Value::String(value.into_owned())
        };
        out.insert(key, value);
    }
    Value::Object(out)
}

fn parse_json_body<T>(body: &[u8]) -> Result<T, Box<Response>>
where
    T: DeserializeOwned,
{
    if body.is_empty() {
        return Err(Box::new(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Invalid JSON: empty body",
        )));
    }

    serde_json::from_slice::<T>(body).map_err(|err| {
        Box::new(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Invalid JSON: {err}"),
        ))
    })
}

async fn fallback_handler(method: axum::http::Method, uri: axum::http::Uri) -> Response {
    json_error(
        StatusCode::NOT_FOUND,
        "not_found",
        format!("No route for {method} {}", uri.path()),
    )
}

fn current_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[allow(dead_code)]
fn _unused(session_state: Option<&SessionState>) {
    let _ = session_state;
}

#[cfg(test)]
mod request_id_header_tests {
    #[test]
    fn diagnostic_outcomes_keep_response_readiness_separate_from_completion() {
        let temp = tempfile::tempdir().unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
                ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
            ]));
        let path = settings.sync_scope(|| {
            let log = crate::logging::create_logger("server");
            for (status, suffix) in [(StatusCode::OK, "ok"), (StatusCode::BAD_GATEWAY, "error")] {
                super::log_request_completed(
                    &log,
                    super::RequestLogContext {
                        req_id: &format!("messages-{suffix}"),
                        provider: Some("codex"),
                        model: Some("gpt-6.1-sol"),
                        count_tokens: false,
                        status,
                        started_at: std::time::Instant::now(),
                    },
                );
                super::log_native_request_completed(
                    &log,
                    &format!("native-{suffix}"),
                    "responses",
                    Some("gpt-6.1-sol"),
                    status,
                    std::time::Instant::now(),
                );
                super::log_routed_openai_request_completed(
                    &log,
                    &format!("routed-{suffix}"),
                    "chat/completions",
                    "opencode",
                    Some("model"),
                    status,
                    std::time::Instant::now(),
                );
            }
            crate::logging::log_file()
        });
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 6);
        for (suffix, status, outcome) in [("ok", 200, "pending"), ("error", 502, "failed")] {
            for route in ["messages", "native", "routed"] {
                let req_id = format!("{route}-{suffix}");
                let record = records
                    .iter()
                    .find(|r| r["fields"]["reqId"] == req_id)
                    .unwrap();
                assert_eq!(record["msg"], "request_completed");
                assert_eq!(record["fields"]["phase"], "response_ready");
                assert_eq!(record["fields"]["outcome"], outcome);
                assert_eq!(record["fields"]["timingPhase"], "response-ready");
                assert_eq!(record["fields"]["status"], status);
                assert!(record["fields"]["ms"].is_number());
            }
        }
    }

    #[test]
    fn cancelled_guards_log_phase_but_finalized_http_errors_do_not() {
        let temp = tempfile::tempdir().unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
                ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
            ]));
        let path = settings.sync_scope(|| {
            drop(RequestMonitorGuard::new(None, "handler-cancel".into()));
            drop(monitor_response_body(
                Response::new(Body::empty()),
                RequestMonitorGuard::new(None, "body-cancel".into()),
            ));
            let mut error = RequestMonitorGuard::new(None, "http-error".into());
            error.failed(StatusCode::UNAUTHORIZED, "expired".into());
            drop(error);
            crate::logging::log_file()
        });
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        let records: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .expect("cancellation diagnostic must be persisted")
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        for (req_id, phase) in [
            ("handler-cancel", "request-handler"),
            ("body-cancel", "response-body"),
        ] {
            let record = records
                .iter()
                .find(|record| record["fields"]["reqId"] == req_id)
                .expect("missing cancellation record");
            assert_eq!(record["msg"], "downstream_request_cancelled");
            assert_eq!(record["fields"]["phase"], phase);
            assert!(record["fields"]["elapsedMs"].is_number());
        }
        assert!(
            records
                .iter()
                .all(|record| record["fields"]["reqId"] != "http-error")
        );
    }

    #[test]
    fn error_capture_uses_shared_writer_instead_of_inline_filesystem() {
        let writer = crate::logging::writer::pause_for_fixture();
        let temp = tempfile::tempdir().unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                ("CCP_CONFIG_DIR".into(), temp.path().display().to_string()),
                ("XDG_STATE_HOME".into(), temp.path().display().to_string()),
            ]));
        let path = settings
            .sync_scope(|| super::write_error_capture("fixture", &serde_json::json!({"safe":true})))
            .unwrap();
        assert!(
            !path.exists(),
            "error capture performed synchronous filesystem I/O"
        );
        drop(writer);
        assert!(crate::logging::flush_diagnostics(
            std::time::Duration::from_secs(2)
        ));
        assert!(path.exists());
    }
    use super::{REQUEST_ID_HEADER, RequestMonitorGuard, monitor_response_body};
    use axum::body::Body;
    use axum::response::Response;
    use http::{HeaderValue, StatusCode};

    fn guard(req_id: &str) -> RequestMonitorGuard {
        RequestMonitorGuard::new(None, req_id.to_string())
    }

    #[test]
    fn abandoned_request_future_does_not_claim_body_cancellation() {
        use crate::monitor::{EndpointKind, MonitorHandle};
        let monitor = MonitorHandle::default();
        monitor.request_started("future", None, None, EndpointKind::Messages);
        drop(RequestMonitorGuard::new(
            Some(monitor.clone()),
            "future".into(),
        ));
        let state = monitor.snapshot();
        assert!(state.recent[0].delivery.response_ready_at.is_none());
        assert!(state.recent[0].delivery.body_cancelled_at.is_none());
    }

    fn body_with_frame_error(monitor: crate::monitor::MonitorHandle, req_id: &str) -> Body {
        use http_body_util::StreamBody;
        let frames = futures_util::stream::iter([
            Ok(http_body::Frame::data(bytes::Bytes::from_static(b"early"))),
            Err(std::io::Error::other("fixture body failure")),
        ]);
        monitor_response_body(
            Response::new(Body::new(StreamBody::new(frames))),
            RequestMonitorGuard::new(Some(monitor), req_id.into()),
        )
        .into_body()
    }

    #[tokio::test]
    async fn body_frame_error_then_eof_preserves_failure_and_records_finish_once() {
        use crate::monitor::{EndpointKind, MonitorHandle, RequestStatus};
        use http_body_util::BodyExt;
        let monitor = MonitorHandle::default();
        monitor.request_started("error-eof", None, None, EndpointKind::Messages);
        let mut body = body_with_frame_error(monitor.clone(), "error-eof");
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            "early"
        );
        assert!(body.frame().await.unwrap().is_err());
        let original_failure = monitor.snapshot().recent[0].clone();
        assert_eq!(original_failure.status, RequestStatus::Failed);
        assert_eq!(
            original_failure.error.as_deref(),
            Some("fixture body failure")
        );
        assert!(body.frame().await.is_none());
        drop(body);
        let state = monitor.snapshot();
        assert_eq!(state.recent.len(), 1);
        let done = &state.recent[0];
        assert_eq!(done.status, RequestStatus::Failed);
        assert_eq!(done.error, original_failure.error);
        assert_eq!(done.finished_at, original_failure.finished_at);
        assert!(
            done.delivery.body_finished_at.is_some(),
            "EOF after frame error must be observed"
        );
        assert!(done.delivery.body_cancelled_at.is_none());
        assert_eq!(done.delivery.yielded_bytes, 5);
    }

    #[tokio::test]
    async fn body_frame_error_then_drop_preserves_failure_and_records_cancel_once() {
        use crate::monitor::{EndpointKind, MonitorHandle, RequestStatus};
        use http_body_util::BodyExt;
        let monitor = MonitorHandle::default();
        monitor.request_started("error-drop", None, None, EndpointKind::Messages);
        let mut body = body_with_frame_error(monitor.clone(), "error-drop");
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            "early"
        );
        assert!(body.frame().await.unwrap().is_err());
        let original_failure = monitor.snapshot().recent[0].clone();
        drop(body);
        let state = monitor.snapshot();
        assert_eq!(state.recent.len(), 1);
        let done = &state.recent[0];
        assert_eq!(done.status, RequestStatus::Failed);
        assert_eq!(done.error.as_deref(), Some("fixture body failure"));
        assert_eq!(done.finished_at, original_failure.finished_at);
        assert!(
            done.delivery.body_cancelled_at.is_some(),
            "drop after frame error must be observed"
        );
        assert!(done.delivery.body_finished_at.is_none());
        assert_eq!(done.delivery.yielded_bytes, 5);
    }

    #[tokio::test]
    async fn response_ready_is_separate_from_body_yield_and_cancel() {
        use crate::monitor::{EndpointKind, MonitorHandle, RequestStatus};
        use futures_util::StreamExt;
        use http_body_util::BodyExt;
        let monitor = MonitorHandle::default();
        monitor.request_started("timing", None, None, EndpointKind::Messages);
        let response = Response::new(Body::from_stream(
            futures_util::stream::once(async {
                Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"early"))
            })
            .chain(futures_util::stream::pending()),
        ));
        let mut body = monitor_response_body(
            response,
            RequestMonitorGuard::new(Some(monitor.clone()), "timing".into()),
        )
        .into_body();
        let snapshot: crate::monitor::snapshot::MonitorSnapshot = monitor.snapshot().into();
        let value = serde_json::to_value(snapshot).unwrap();
        assert!(
            value["active"][0]["delivery"]["response_ready_at"].is_object(),
            "response ready not recorded"
        );
        assert!(value["active"][0]["delivery"]["first_body_yield_at"].is_null());
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            "early"
        );
        drop(body);
        let done = &monitor.snapshot().recent[0];
        assert_eq!(done.status, RequestStatus::Failed);
        assert!(done.delivery.first_body_yield_at.is_some());
        assert!(done.delivery.body_cancelled_at.is_some());
        assert!(done.delivery.body_finished_at.is_none());
        assert_eq!(done.delivery.yielded_bytes, 5);
    }

    #[tokio::test]
    async fn body_monitor_preserves_data_and_trailers_and_marks_eof() {
        use crate::monitor::{EndpointKind, MonitorHandle};
        use http_body_util::{BodyExt, StreamBody};
        let monitor = MonitorHandle::default();
        monitor.request_started("frames", None, None, EndpointKind::Messages);
        let mut trailers = http::HeaderMap::new();
        trailers.insert("x-fixture-trailer", HeaderValue::from_static("done"));
        let frames = futures_util::stream::iter([
            Ok::<_, std::io::Error>(http_body::Frame::data(bytes::Bytes::from_static(b"early"))),
            Ok(http_body::Frame::trailers(trailers)),
        ]);
        let response = Response::new(Body::new(StreamBody::new(frames)));
        let mut body = monitor_response_body(
            response,
            RequestMonitorGuard::new(Some(monitor.clone()), "frames".into()),
        )
        .into_body();
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            "early"
        );
        assert_eq!(
            body.frame()
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap()["x-fixture-trailer"],
            "done"
        );
        assert!(body.frame().await.is_none());
        let state = monitor.snapshot();
        assert!(state.recent[0].delivery.body_finished_at.is_some());
        assert!(state.recent[0].delivery.body_cancelled_at.is_none());
        assert_eq!(state.recent[0].delivery.yielded_bytes, 5);
        assert_eq!(state.recent[0].delivery.yielded_frames, 2);
    }

    // Claude Code populates its transcript `requestId` from this header.
    // Without it, consumers that de-duplicate transcript records by request id
    // count every request twice, because a transcript legitimately repeats a
    // record and the id is what resolves the repeat.
    #[test]
    fn stamps_the_request_id_on_a_response() {
        let response = Response::builder()
            .status(StatusCode::OK)
            .body(Body::from("{}"))
            .unwrap();
        let stamped = monitor_response_body(response, guard("req-abc-123"));
        assert_eq!(
            stamped.headers().get(REQUEST_ID_HEADER).unwrap(),
            "req-abc-123"
        );
    }

    // Streaming responses carry it too: the header is applied to the parts
    // before the body is streamed, which is why the body-level message id is
    // not a substitute for SSE.
    #[test]
    fn stamps_a_streaming_response_before_the_body() {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from("event: message_start\n"))
            .unwrap();
        let stamped = monitor_response_body(response, guard("req-stream-1"));
        assert_eq!(
            stamped.headers().get(REQUEST_ID_HEADER).unwrap(),
            "req-stream-1"
        );
    }

    // An upstream that already supplied one owns it; overwriting would relabel
    // a real provider id with a local uuid.
    #[test]
    fn does_not_clobber_an_upstream_supplied_id() {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(
                REQUEST_ID_HEADER,
                HeaderValue::from_static("upstream-owned"),
            )
            .body(Body::from("{}"))
            .unwrap();
        let stamped = monitor_response_body(response, guard("local-uuid"));
        assert_eq!(
            stamped.headers().get(REQUEST_ID_HEADER).unwrap(),
            "upstream-owned"
        );
    }
}

#[cfg(test)]
mod auto_review_tests {
    use super::{apply_auto_review_model, headers_to_record, is_claude_auto_review_request};
    use crate::anthropic::schema::MessagesRequest;
    use crate::request_identity::{CLAUDE_AGENT_HEADER, CLAUDE_PARENT_AGENT_HEADER};
    use http::{HeaderMap, HeaderValue};
    use serde_json::json;

    fn request(system: &str, stream: bool, tools: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "max_tokens": 2112,
            "stream": stream,
            "system": [{"type": "text", "text": system}],
            "messages": [{"role": "user", "content": "review this Bash command"}],
            "tools": tools
        }))
        .unwrap()
    }

    #[test]
    fn detects_claude_auto_review_classifier() {
        let body = request(
            "You are a security monitor for autonomous AI coding agents.\n\n## Context",
            false,
            json!([]),
        );
        assert!(is_claude_auto_review_request(&body));
    }

    #[test]
    fn ignores_normal_streaming_and_tool_using_requests() {
        assert!(!is_claude_auto_review_request(&request(
            "You are an interactive coding agent.",
            false,
            json!([]),
        )));
        assert!(!is_claude_auto_review_request(&request(
            "You are a security monitor for autonomous AI coding agents.",
            true,
            json!([]),
        )));
        assert!(!is_claude_auto_review_request(&request(
            "You are a security monitor for autonomous AI coding agents.",
            false,
            json!([{"name": "Bash"}]),
        )));
    }

    #[test]
    fn codex_classifier_defaults_to_luna() {
        let mut classifier = request(
            "You are a security monitor for autonomous AI coding agents.",
            false,
            json!([]),
        );
        let route = apply_auto_review_model(&mut classifier, false, None, "codex")
            .expect("classifier should be routed");
        assert_eq!(route.requested_model, "gpt-5.6-sol");
        assert_eq!(route.override_model, "gpt-6-luna");
        assert_eq!(classifier.model.as_deref(), Some("gpt-6-luna"));
    }

    #[test]
    fn non_codex_classifier_keeps_requested_model_without_override() {
        let mut classifier = request(
            "You are a security monitor for autonomous AI coding agents.",
            false,
            json!([]),
        );
        classifier.model = Some("kimi-for-coding".to_string());
        assert!(apply_auto_review_model(&mut classifier, false, None, "kimi").is_none());
        assert_eq!(classifier.model.as_deref(), Some("kimi-for-coding"));
    }

    #[test]
    fn configured_model_overrides_provider_default() {
        let mut classifier = request(
            "You are a security monitor for autonomous AI coding agents.",
            false,
            json!([]),
        );
        let route = apply_auto_review_model(&mut classifier, false, Some("grok-4.5"), "codex")
            .expect("configured classifier should be routed");
        assert_eq!(route.override_model, "grok-4.5");
        assert_eq!(classifier.model.as_deref(), Some("grok-4.5"));
    }

    #[test]
    fn grok_session_classifier_routes_by_requested_model_or_explicit_override() {
        use crate::config::AliasProvider;
        use crate::registry::Registry;
        use crate::session::{record_session_request, record_session_request_with_affinity_update};

        let registry = Registry::new(AliasProvider::Codex);
        let session_id = "grok-session-classifier-routing";
        let initial = record_session_request(Some(session_id), None, "grok", "grok-4.6", 1)
            .expect("Grok session");
        assert_eq!(initial.affinity_provider, None);

        for (requested, configured, expected_provider, expected_model) in [
            ("claude-sonnet-4-6", None, "codex", "gpt-6-luna"),
            ("grok-4.6", None, "grok", "grok-4.6"),
            ("claude-sonnet-4-6", Some("grok-4.6"), "grok", "grok-4.6"),
            ("gpt-6-luna", Some("grok-4.6"), "grok", "grok-4.6"),
        ] {
            let mut classifier = request(
                "You are a security monitor for autonomous AI coding agents.",
                false,
                json!([]),
            );
            classifier.model = Some(requested.to_string());
            let original = registry
                .provider_for_model(requested, initial.affinity_provider.as_ref())
                .expect("original provider");
            let route =
                apply_auto_review_model(&mut classifier, false, configured, original.name());
            let model = classifier.model.as_deref().expect("classifier model");
            let provider = if route.is_some() {
                registry.provider_for_model(model, None).expect("override")
            } else {
                original
            };
            assert_eq!(provider.name(), expected_provider);
            assert_eq!(model, expected_model);

            let after_review = record_session_request_with_affinity_update(
                Some(session_id),
                Some(&initial),
                provider.name(),
                model,
                route.is_none(),
                2,
            )
            .expect("session after review");
            assert_eq!(after_review.affinity_provider, None);
        }
    }

    #[test]
    fn traffic_headers_redact_agent_lineage_ids() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-claude-code-session-id",
            HeaderValue::from_static("session-visible"),
        );
        headers.insert(
            CLAUDE_AGENT_HEADER,
            HeaderValue::from_static("agent-secret"),
        );
        headers.insert(
            CLAUDE_PARENT_AGENT_HEADER,
            HeaderValue::from_static("parent-secret"),
        );

        let captured = headers_to_record(&headers);
        assert_eq!(
            captured["x-claude-code-session-id"],
            json!("session-visible")
        );
        assert_eq!(captured[CLAUDE_AGENT_HEADER], json!("[redacted len=12]"));
        assert_eq!(
            captured[CLAUDE_PARENT_AGENT_HEADER],
            json!("[redacted len=13]")
        );
        let serialized = captured.to_string();
        assert!(!serialized.contains("agent-secret"));
        assert!(!serialized.contains("parent-secret"));
    }

    #[test]
    fn count_tokens_keeps_requested_model() {
        let mut classifier = request(
            "You are a security monitor for autonomous AI coding agents.",
            false,
            json!([]),
        );
        assert!(
            apply_auto_review_model(&mut classifier, true, Some("grok-4.5"), "codex").is_none()
        );
        assert_eq!(classifier.model.as_deref(), Some("gpt-5.6-sol"));
    }
}
