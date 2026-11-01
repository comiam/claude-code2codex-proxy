use crate::anthropic::schema::MessagesRequest;
use crate::monitor::MonitorHandle;
use crate::request_identity::ConversationIdentity;
use crate::traffic::TrafficCapture;
use anyhow::Result;
use async_trait::async_trait;
use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use clap::Subcommand;
use std::sync::Arc;

#[derive(Debug, Clone, Subcommand)]
pub enum AuthCommand {
    /// Sign in using browser-based authentication
    Login,
    /// Sign in using a device code
    Device,
    /// Show the current authentication status
    Status,
    /// Delete stored authentication credentials
    Logout,
}

// async_trait adds #[must_use] to methods returning already-must-use futures.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_models(&self) -> Vec<String>;
    fn cli(&self) -> &'static dyn CliHandlers;
    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response;

    async fn handle_messages_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        let _ = conversation_identity;
        self.handle_messages(body, ctx).await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response;

    async fn generate_anthropic_stream(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        Err(ProviderError::new(
            StatusCode::NOT_IMPLEMENTED,
            ProviderErrorKind::InvalidRequest,
            format!(
                "provider '{}' does not support OpenAI-compatible generation",
                self.name()
            ),
        ))
    }
}

pub enum GenerationBody {
    BufferedSse(Bytes),
    LiveSse(Body),
}

pub struct Generation {
    pub body: GenerationBody,
    pub resolved_model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Authentication,
    Permission,
    RateLimit,
    InvalidRequest,
    Api,
}

#[derive(Debug, Clone)]
pub struct ProviderError {
    pub status: StatusCode,
    pub kind: ProviderErrorKind,
    pub message: String,
    pub retry_after: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ProviderError {
    pub fn new(status: StatusCode, kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            retry_after: None,
            param: None,
            code: None,
        }
    }

    pub fn error_type(&self) -> &'static str {
        match self.kind {
            ProviderErrorKind::Authentication => "authentication_error",
            ProviderErrorKind::Permission => "permission_error",
            ProviderErrorKind::RateLimit => "rate_limit_error",
            ProviderErrorKind::InvalidRequest => "invalid_request_error",
            ProviderErrorKind::Api => "api_error",
        }
    }
}

pub trait CliHandlers: Send + Sync {
    fn login(&self) -> Result<()>;
    fn device(&self) -> Result<()>;
    fn status(&self) -> Result<()>;
    fn logout(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub req_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub provider: String,
    pub traffic: Option<Arc<TrafficCapture>>,
    pub monitor: Option<MonitorHandle>,
}

/// Body polling and destruction outlive the handler future. Keep the same
/// request settings around both, including cancellation of a pending frame.
pub(crate) fn request_scoped_response(response: Response) -> Response {
    use axum::body::HttpBody;
    let Some(settings) = crate::config::captured_request_settings() else {
        return response;
    };
    struct ScopedBody {
        inner: Option<Body>,
        settings: Arc<crate::config::CodexRequestSettings>,
    }
    impl HttpBody for ScopedBody {
        type Data = Bytes;
        type Error = axum::Error;
        fn poll_frame(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<http_body::Frame<Bytes>, Self::Error>>> {
            let body = self.get_mut();
            Arc::clone(&body.settings).sync_scope(|| {
                std::pin::Pin::new(body.inner.as_mut().expect("scoped body is present"))
                    .poll_frame(cx)
            })
        }
        fn is_end_stream(&self) -> bool {
            Arc::clone(&self.settings).sync_scope(|| {
                self.inner
                    .as_ref()
                    .expect("scoped body is present")
                    .is_end_stream()
            })
        }
        fn size_hint(&self) -> http_body::SizeHint {
            Arc::clone(&self.settings).sync_scope(|| {
                self.inner
                    .as_ref()
                    .expect("scoped body is present")
                    .size_hint()
            })
        }
    }
    impl Drop for ScopedBody {
        fn drop(&mut self) {
            let inner = self.inner.take();
            Arc::clone(&self.settings).sync_scope(|| drop(inner));
        }
    }
    let (parts, inner) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(ScopedBody {
            inner: Some(inner),
            settings,
        }),
    )
}

#[cfg(test)]
mod request_scope_tests {
    use super::*;
    use http_body_util::BodyExt;
    use std::sync::Mutex;

    #[tokio::test]
    async fn scoped_response_body_preserves_size_hint_and_end_stream() {
        use axum::body::HttpBody;
        let dir = tempfile::TempDir::new().unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([(
                "CCP_CONFIG_DIR".into(),
                dir.path().to_string_lossy().into_owned(),
            )]));
        settings
            .scope(async {
                let response = request_scoped_response(Response::new(Body::from("abc")));
                assert_eq!(response.body().size_hint().exact(), Some(3));
                assert!(!response.body().is_end_stream());
                let empty = request_scoped_response(Response::new(Body::empty()));
                assert!(empty.body().is_end_stream());
            })
            .await;
    }

    #[tokio::test]
    async fn scoped_response_body_preserves_trailers() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([(
                "CCP_CONFIG_DIR".into(),
                dir.path().to_string_lossy().into_owned(),
            )]));
        let mut trailers = http::HeaderMap::new();
        trailers.insert("x-fixture", http::HeaderValue::from_static("complete"));
        let response = settings
            .scope(async {
                let frames = futures_util::stream::iter(vec![
                    Ok::<_, std::io::Error>(http_body::Frame::data(Bytes::from_static(b"data"))),
                    Ok(http_body::Frame::trailers(trailers.clone())),
                ]);
                request_scoped_response(Response::new(Body::new(http_body_util::StreamBody::new(
                    frames,
                ))))
            })
            .await;
        let mut body = response.into_body();
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            Bytes::from_static(b"data")
        );
        assert_eq!(
            body.frame()
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap(),
            trailers
        );
        assert!(body.frame().await.is_none());
    }

    #[tokio::test]
    async fn response_body_poll_and_drop_retain_request_policy_and_destinations() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"codex":{"model":"gpt-6-sol"},"log":{"verbose":true}}"#,
        )
        .unwrap();
        let settings =
            crate::config::CodexRequestSettings::for_env(std::collections::HashMap::from([
                (
                    "CCP_CONFIG_DIR".into(),
                    dir.path().to_string_lossy().into_owned(),
                ),
                (
                    "XDG_STATE_HOME".into(),
                    dir.path().join("state").to_string_lossy().into_owned(),
                ),
            ]));
        struct Watch(Arc<Mutex<Vec<std::path::PathBuf>>>);
        impl Drop for Watch {
            fn drop(&mut self) {
                self.0.lock().unwrap().push(crate::logging::log_file());
            }
        }
        for cancel in [false, true] {
            let drops = Arc::new(Mutex::new(Vec::new()));
            let response = Arc::clone(&settings)
                .scope(async {
                    let stream = futures_util::stream::unfold(
                        (Watch(Arc::clone(&drops)), false),
                        move |(watch, sent)| async move {
                            if sent && cancel {
                                std::future::pending().await
                            } else if sent {
                                None
                            } else {
                                assert!(crate::config::log_verbose());
                                let model = crate::config::codex_model().unwrap();
                                Some((Ok::<_, std::io::Error>(Bytes::from(model)), (watch, true)))
                            }
                        },
                    );
                    request_scoped_response(Response::new(Body::from_stream(stream)))
                })
                .await;
            let mut body = response.into_body();
            let frame = body.frame().await.unwrap().unwrap();
            assert_eq!(frame.into_data().unwrap(), Bytes::from_static(b"gpt-6-sol"));
            if cancel {
                assert!(futures_util::poll!(body.frame()).is_pending());
            } else {
                assert!(body.frame().await.is_none());
            }
            drop(body);
            assert_eq!(
                *drops.lock().unwrap(),
                vec![dir.path().join("state/claude-code-proxy/proxy.log")]
            );
        }
    }
}
