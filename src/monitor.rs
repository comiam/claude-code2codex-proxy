use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

mod mock;
pub mod remote;
pub mod snapshot;

pub use mock::{MockMonitor, mock_state};

const DEFAULT_RECENT_LIMIT: usize = 200;
pub const SESSION_TOKEN_BUCKET_SECS: u64 = 10;
pub const SESSION_OUTPUT_HISTORY_BUCKETS: u64 = 60 * 60 / SESSION_TOKEN_BUCKET_SECS;
const SESSION_USAGE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
const SESSION_PRUNE_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageProvenance {
    Estimated,
    Provider,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenObservation {
    pub tokens: u64,
    pub provenance: UsageProvenance,
}

impl TokenObservation {
    pub fn estimated(tokens: u64) -> Self {
        Self {
            tokens,
            provenance: UsageProvenance::Estimated,
        }
    }
    pub fn provider(tokens: u64) -> Self {
        Self {
            tokens,
            provenance: UsageProvenance::Provider,
        }
    }
}

/// Cumulative provider counts. Output is the provider total, including any
/// reasoning and tool tokens; absent breakdowns remain unknown.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UsageObservation {
    pub input_tokens: Option<TokenObservation>,
    /// Reported cached input, included in the provider's gross input total.
    #[serde(default)]
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<TokenObservation>,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeliveryTiming {
    pub response_ready_at: Option<SystemTime>,
    pub first_body_yield_at: Option<SystemTime>,
    pub body_finished_at: Option<SystemTime>,
    pub body_cancelled_at: Option<SystemTime>,
    pub yielded_bytes: u64,
    pub yielded_frames: u64,
}

#[derive(Debug, Clone)]
pub enum DeliveryEvent {
    ResponseReady,
    BodyYield { bytes: u64 },
    BodyFinished,
    BodyCancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Messages,
    CountTokens,
    Responses,
    ChatCompletions,
    Images,
    Transcriptions,
}

impl EndpointKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::CountTokens => "count_tokens",
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
            Self::Images => "images",
            Self::Transcriptions => "transcriptions",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Started,
    ProviderSelected,
    Compacting,
    Upstream,
    Streaming,
    Completed,
    Failed,
}

impl RequestStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::ProviderSelected => "selected",
            Self::Compacting => "compacting",
            Self::Upstream => "upstream",
            Self::Streaming => "streaming",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub enum MonitorEvent {
    TypedProgress {
        request_id: String,
        bytes: u64,
        events: u64,
        usage: UsageObservation,
    },
    Delivery {
        request_id: String,
        event: DeliveryEvent,
    },
    StreamFailed {
        request_id: String,
        error: String,
    },
    RequestStarted {
        request_id: String,
        session_id: Option<String>,
        session_seq: Option<u64>,
        endpoint: EndpointKind,
    },
    ProjectResolved {
        request_id: String,
        project: String,
    },
    SessionSequenceResolved {
        request_id: String,
        session_seq: u64,
    },
    ProviderSelected {
        request_id: String,
        provider: String,
        model: String,
        effort: Option<String>,
    },
    ModelResolved {
        request_id: String,
        model: String,
    },
    CompactionStarted {
        request_id: String,
    },
    UpstreamStarted {
        request_id: String,
    },
    GenerationStarted {
        request_id: String,
    },
    TrafficCapturePath {
        request_id: String,
        path: PathBuf,
    },
    StreamProgress {
        request_id: String,
        bytes: u64,
        chunks: u64,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    UsageUpdated {
        request_id: String,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    RequestCompleted {
        request_id: String,
        http_status: u16,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    RequestFailed {
        request_id: String,
        http_status: Option<u16>,
        error: String,
    },
    RequestAbandoned {
        request_id: String,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub struct ActiveRequest {
    pub request_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub project: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub endpoint: EndpointKind,
    pub started_at: SystemTime,
    started_instant: Instant,
    pub generation_started_at: Option<SystemTime>,
    generation_started_instant: Option<Instant>,
    generation_initial_output_tokens: u64,
    pub generation_finished_at: Option<SystemTime>,
    pub generation_duration: Option<Duration>,
    pub status: RequestStatus,
    pub streamed_bytes: u64,
    pub stream_chunks: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub error: Option<String>,
    pub traffic_capture_path: Option<PathBuf>,
    pub usage: UsageObservation,
    pub delivery: DeliveryTiming,
}

impl ActiveRequest {
    pub fn elapsed(&self) -> Duration {
        self.started_instant.elapsed()
    }

    pub fn rate(&self) -> Throughput {
        throughput(
            self.output_tokens
                .filter(|_| {
                    self.usage
                        .output_tokens
                        .is_none_or(|value| value.provenance == UsageProvenance::Provider)
                })
                .and_then(|tokens| tokens.checked_sub(self.generation_initial_output_tokens)),
            self.streamed_bytes,
            self.stream_chunks,
            self.generation_duration.unwrap_or(Duration::ZERO),
        )
    }
}

#[derive(Debug, Clone)]
pub struct CompletedRequest {
    pub request_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub project: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub endpoint: EndpointKind,
    pub started_at: SystemTime,
    pub finished_at: SystemTime,
    pub generation_started_at: Option<SystemTime>,
    generation_started_instant: Option<Instant>,
    generation_initial_output_tokens: u64,
    pub generation_finished_at: Option<SystemTime>,
    pub generation_duration: Option<Duration>,
    pub status: RequestStatus,
    pub http_status: Option<u16>,
    pub latency: Duration,
    pub streamed_bytes: u64,
    pub stream_chunks: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub error: Option<String>,
    pub traffic_capture_path: Option<PathBuf>,
    pub usage: UsageObservation,
    pub delivery: DeliveryTiming,
}

impl CompletedRequest {
    pub fn rate(&self) -> Throughput {
        throughput(
            self.output_tokens
                .filter(|_| {
                    self.usage
                        .output_tokens
                        .is_none_or(|value| value.provenance == UsageProvenance::Provider)
                })
                .and_then(|tokens| tokens.checked_sub(self.generation_initial_output_tokens)),
            self.streamed_bytes,
            self.stream_chunks,
            self.generation_duration.unwrap_or(Duration::ZERO),
        )
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "unit", content = "value", rename_all = "snake_case")]
pub enum Throughput {
    TokensPerSecond(f64),
    BytesPerSecond(f64),
    EventsPerSecond(f64),
    None,
}

impl Throughput {
    pub fn label(&self) -> String {
        match self {
            Self::TokensPerSecond(value) => format!("{value:.1} tok/s"),
            Self::BytesPerSecond(value) if *value >= 1024.0 => {
                format!("{:.1} KB/s", value / 1024.0)
            }
            Self::BytesPerSecond(value) => format!("{value:.0} B/s"),
            Self::EventsPerSecond(value) => format!("{value:.1} ev/s"),
            Self::None => "-".to_string(),
        }
    }

    pub fn label_with_basis(&self) -> String {
        match self {
            Self::TokensPerSecond(value) => format!("{value:.1} provider-total tok/s"),
            _ => self.label(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitorState {
    pub started_at: SystemTime,
    pub uptime: Duration,
    pub sessions: Vec<SessionSummary>,
    pub active: Vec<ActiveRequest>,
    pub recent: Vec<CompletedRequest>,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: Option<String>,
    pub project: Option<String>,
    pub active_count: usize,
    pub request_count: usize,
    pub failure_count: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub last_seen: SystemTime,
    pub input_tokens: u64,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: u64,
    pub output_token_samples: Vec<(SystemTime, u64)>,
    rate_output_tokens: u64,
    pub generation_duration: Duration,
    pub last_status: String,
}

impl SessionSummary {
    pub fn rate(&self) -> Throughput {
        throughput(
            Some(self.rate_output_tokens).filter(|tokens| *tokens > 0),
            0,
            0,
            self.generation_duration,
        )
    }

    pub fn label(&self) -> String {
        self.session_id
            .clone()
            .unwrap_or_else(|| "no-session".to_string())
    }
}

#[derive(Debug)]
struct MonitorStore {
    started_at: SystemTime,
    started_instant: Instant,
    last_session_prune_instant: Instant,
    active: HashMap<String, ActiveRequest>,
    recent: VecDeque<CompletedRequest>,
    session_usage: HashMap<Option<String>, SessionUsage>,
    session_output_buckets: HashMap<Option<String>, Vec<(u64, u64)>>,
    recent_limit: usize,
}

#[derive(Debug)]
struct SessionUsage {
    input_tokens: u64,
    cached_input_tokens: Option<u64>,
    output_tokens: u64,
    last_seen: SystemTime,
}

impl Default for SessionUsage {
    fn default() -> Self {
        Self {
            input_tokens: 0,
            cached_input_tokens: None,
            output_tokens: 0,
            last_seen: SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitorHandle {
    store: Arc<Mutex<MonitorStore>>,
}

impl Default for MonitorHandle {
    fn default() -> Self {
        Self::new(DEFAULT_RECENT_LIMIT)
    }
}

impl MonitorHandle {
    pub fn new(recent_limit: usize) -> Self {
        Self {
            store: Arc::new(Mutex::new(MonitorStore {
                started_at: SystemTime::now(),
                started_instant: Instant::now(),
                last_session_prune_instant: Instant::now(),
                active: HashMap::new(),
                recent: VecDeque::new(),
                session_usage: HashMap::new(),
                session_output_buckets: HashMap::new(),
                recent_limit,
            })),
        }
    }

    pub fn publish(&self, event: MonitorEvent) {
        if let Ok(mut store) = self.store.lock() {
            store.apply(event);
        }
    }

    pub fn snapshot(&self) -> MonitorState {
        match self.store.lock() {
            Ok(mut store) => store.snapshot(),
            Err(_) => MonitorState {
                started_at: SystemTime::now(),
                uptime: Duration::ZERO,
                sessions: Vec::new(),
                active: Vec::new(),
                recent: Vec::new(),
            },
        }
    }

    pub fn request_started(
        &self,
        request_id: impl Into<String>,
        session_id: Option<String>,
        session_seq: Option<u64>,
        endpoint: EndpointKind,
    ) {
        self.publish(MonitorEvent::RequestStarted {
            request_id: request_id.into(),
            session_id,
            session_seq,
            endpoint,
        });
    }

    pub fn project_resolved(&self, request_id: impl Into<String>, project: impl Into<String>) {
        self.publish(MonitorEvent::ProjectResolved {
            request_id: request_id.into(),
            project: project.into(),
        });
    }

    pub fn session_sequence_resolved(&self, request_id: impl Into<String>, session_seq: u64) {
        self.publish(MonitorEvent::SessionSequenceResolved {
            request_id: request_id.into(),
            session_seq,
        });
    }

    pub fn provider_selected(
        &self,
        request_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        effort: Option<String>,
    ) {
        self.publish(MonitorEvent::ProviderSelected {
            request_id: request_id.into(),
            provider: provider.into(),
            model: model.into(),
            effort,
        });
    }

    pub fn model_resolved(&self, request_id: impl Into<String>, model: impl Into<String>) {
        self.publish(MonitorEvent::ModelResolved {
            request_id: request_id.into(),
            model: model.into(),
        });
    }

    pub fn compaction_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::CompactionStarted {
            request_id: request_id.into(),
        });
    }

    pub fn upstream_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::UpstreamStarted {
            request_id: request_id.into(),
        });
    }

    pub fn generation_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::GenerationStarted {
            request_id: request_id.into(),
        });
    }

    pub fn traffic_capture_path(&self, request_id: impl Into<String>, path: PathBuf) {
        self.publish(MonitorEvent::TrafficCapturePath {
            request_id: request_id.into(),
            path,
        });
    }

    pub fn stream_progress(
        &self,
        request_id: impl Into<String>,
        bytes: u64,
        chunks: u64,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::StreamProgress {
            request_id: request_id.into(),
            bytes,
            chunks,
            input_tokens,
            output_tokens,
        });
    }

    pub fn usage_updated(
        &self,
        request_id: impl Into<String>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::UsageUpdated {
            request_id: request_id.into(),
            input_tokens,
            output_tokens,
        });
    }

    pub fn typed_progress(
        &self,
        request_id: impl Into<String>,
        bytes: u64,
        events: u64,
        usage: UsageObservation,
    ) {
        self.publish(MonitorEvent::TypedProgress {
            request_id: request_id.into(),
            bytes,
            events,
            usage,
        });
    }

    pub fn delivery(&self, request_id: impl Into<String>, event: DeliveryEvent) {
        self.publish(MonitorEvent::Delivery {
            request_id: request_id.into(),
            event,
        });
    }

    pub fn stream_failed(&self, request_id: impl Into<String>, error: impl Into<String>) {
        self.publish(MonitorEvent::StreamFailed {
            request_id: request_id.into(),
            error: error.into(),
        });
    }

    pub fn request_completed(
        &self,
        request_id: impl Into<String>,
        http_status: u16,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::RequestCompleted {
            request_id: request_id.into(),
            http_status,
            input_tokens,
            output_tokens,
        });
    }

    pub fn request_failed(
        &self,
        request_id: impl Into<String>,
        http_status: Option<u16>,
        error: impl Into<String>,
    ) {
        self.publish(MonitorEvent::RequestFailed {
            request_id: request_id.into(),
            http_status,
            error: error.into(),
        });
    }

    pub fn request_abandoned(&self, request_id: impl Into<String>, error: impl Into<String>) {
        self.publish(MonitorEvent::RequestAbandoned {
            request_id: request_id.into(),
            error: error.into(),
        });
    }
}

impl MonitorStore {
    fn apply(&mut self, event: MonitorEvent) {
        match event {
            MonitorEvent::TypedProgress {
                request_id,
                bytes,
                events,
                usage,
            } => {
                let update = if let Some(active) = self.active.get_mut(&request_id) {
                    if bytes > 0 {
                        active.status = RequestStatus::Streaming;
                        if active.generation_started_instant.is_none() {
                            active.generation_started_at = Some(SystemTime::now());
                            active.generation_started_instant = Some(Instant::now());
                        }
                        active.generation_finished_at = Some(SystemTime::now());
                        active.generation_duration = active
                            .generation_started_instant
                            .map(|start| start.elapsed());
                    }
                    active.streamed_bytes = active.streamed_bytes.saturating_add(bytes);
                    active.stream_chunks = active.stream_chunks.saturating_add(events);
                    let delta = apply_usage(
                        &mut active.input_tokens,
                        &mut active.output_tokens,
                        &mut active.usage,
                        usage,
                    );
                    Some((active.session_id.clone(), delta, None))
                } else if let Some(done) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    done.streamed_bytes = done.streamed_bytes.saturating_add(bytes);
                    done.stream_chunks = done.stream_chunks.saturating_add(events);
                    let delta = apply_usage(
                        &mut done.input_tokens,
                        &mut done.output_tokens,
                        &mut done.usage,
                        usage,
                    );
                    Some((
                        done.session_id.clone(),
                        delta,
                        Some(done.generation_finished_at.unwrap_or(done.finished_at)),
                    ))
                } else {
                    None
                };
                if let Some((session, (input, output, cached), timestamp)) = update {
                    self.record_session_usage(session.clone(), input, output, cached);
                    if let Some(timestamp) = timestamp
                        && output > 0
                    {
                        self.record_session_output(session, timestamp, output);
                    }
                }
            }
            MonitorEvent::Delivery { request_id, event } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    apply_delivery(&mut active.delivery, event);
                } else if let Some(done) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    apply_delivery(&mut done.delivery, event);
                }
            }
            MonitorEvent::StreamFailed { request_id, error } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.error = Some(error);
                } else if let Some(done) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    done.status = RequestStatus::Failed;
                    done.error = Some(error);
                }
            }
            MonitorEvent::RequestStarted {
                request_id,
                session_id,
                session_seq,
                endpoint,
            } => {
                self.active.insert(
                    request_id.clone(),
                    ActiveRequest {
                        request_id,
                        session_id,
                        session_seq,
                        project: None,
                        provider: None,
                        model: None,
                        effort: None,
                        endpoint,
                        started_at: SystemTime::now(),
                        started_instant: Instant::now(),
                        generation_started_at: None,
                        generation_started_instant: None,
                        generation_initial_output_tokens: 0,
                        generation_finished_at: None,
                        generation_duration: None,
                        status: RequestStatus::Started,
                        streamed_bytes: 0,
                        stream_chunks: 0,
                        input_tokens: None,
                        output_tokens: None,
                        error: None,
                        traffic_capture_path: None,
                        usage: UsageObservation::default(),
                        delivery: DeliveryTiming::default(),
                    },
                );
            }
            MonitorEvent::ProjectResolved {
                request_id,
                project,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.project = Some(project);
                }
            }
            MonitorEvent::SessionSequenceResolved {
                request_id,
                session_seq,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.session_seq = Some(session_seq);
                }
            }
            MonitorEvent::ProviderSelected {
                request_id,
                provider,
                model,
                effort,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.provider = Some(provider);
                    active.model = Some(model);
                    active.effort = effort;
                    active.status = RequestStatus::ProviderSelected;
                }
            }
            MonitorEvent::ModelResolved { request_id, model } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.model = Some(match active.model.take() {
                        Some(incoming) if incoming != model => format!("{incoming} → {model}"),
                        Some(incoming) => incoming,
                        None => model,
                    });
                }
            }
            MonitorEvent::CompactionStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Compacting;
                }
            }
            MonitorEvent::UpstreamStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Upstream;
                }
            }
            MonitorEvent::GenerationStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.generation_started_at = Some(SystemTime::now());
                    active.generation_started_instant = Some(Instant::now());
                    active.generation_initial_output_tokens = active
                        .output_tokens
                        .filter(|_| {
                            active
                                .usage
                                .output_tokens
                                .is_none_or(|value| value.provenance == UsageProvenance::Provider)
                        })
                        .unwrap_or(0);
                    active.generation_finished_at = None;
                    active.generation_duration = None;
                }
            }
            MonitorEvent::TrafficCapturePath { request_id, path } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.traffic_capture_path = Some(path);
                }
            }
            MonitorEvent::StreamProgress {
                request_id,
                bytes,
                chunks,
                input_tokens,
                output_tokens,
            } => {
                let mut usage_update = None;
                let mut history_update = None;
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Streaming;
                    if active.generation_started_instant.is_none() {
                        active.generation_started_at = Some(SystemTime::now());
                        active.generation_started_instant = Some(Instant::now());
                        active.generation_initial_output_tokens =
                            output_tokens.or(active.output_tokens).unwrap_or(0);
                    } else {
                        active.generation_finished_at = Some(SystemTime::now());
                        active.generation_duration = active
                            .generation_started_instant
                            .map(|started| started.elapsed());
                    }
                    active.streamed_bytes = active.streamed_bytes.saturating_add(bytes);
                    active.stream_chunks = active.stream_chunks.saturating_add(chunks);
                    let input_delta = apply_observation(
                        &mut active.input_tokens,
                        &mut active.usage.input_tokens,
                        input_tokens.map(TokenObservation::provider),
                    );
                    let output_delta = apply_observation(
                        &mut active.output_tokens,
                        &mut active.usage.output_tokens,
                        output_tokens.map(TokenObservation::provider),
                    );
                    usage_update = Some((active.session_id.clone(), input_delta, output_delta));
                } else if let Some(completed) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    if let Some(started) = completed.generation_started_instant {
                        completed.generation_finished_at = Some(SystemTime::now());
                        completed.generation_duration = Some(started.elapsed());
                    }
                    completed.streamed_bytes = completed.streamed_bytes.saturating_add(bytes);
                    completed.stream_chunks = completed.stream_chunks.saturating_add(chunks);
                    let input_delta = apply_observation(
                        &mut completed.input_tokens,
                        &mut completed.usage.input_tokens,
                        input_tokens.map(TokenObservation::provider),
                    );
                    let output_delta = apply_observation(
                        &mut completed.output_tokens,
                        &mut completed.usage.output_tokens,
                        output_tokens.map(TokenObservation::provider),
                    );
                    usage_update = Some((completed.session_id.clone(), input_delta, output_delta));
                    if output_delta > 0 {
                        history_update = Some((
                            completed.session_id.clone(),
                            completed
                                .generation_finished_at
                                .unwrap_or(completed.finished_at),
                            output_delta,
                        ));
                    }
                }
                if let Some((session_id, input_delta, output_delta)) = usage_update {
                    self.record_session_usage(session_id, input_delta, output_delta, None);
                }
                if let Some((session_id, timestamp, tokens)) = history_update {
                    self.record_session_output(session_id, timestamp, tokens);
                }
            }
            MonitorEvent::UsageUpdated {
                request_id,
                input_tokens,
                output_tokens,
            } => {
                let mut usage_update = None;
                let mut history_update = None;
                if let Some(active) = self.active.get_mut(&request_id) {
                    if output_tokens.is_some()
                        && let Some(started) = active.generation_started_instant
                    {
                        active.generation_finished_at = Some(SystemTime::now());
                        active.generation_duration = Some(started.elapsed());
                    }
                    let input_delta = apply_observation(
                        &mut active.input_tokens,
                        &mut active.usage.input_tokens,
                        input_tokens.map(TokenObservation::provider),
                    );
                    let output_delta = apply_observation(
                        &mut active.output_tokens,
                        &mut active.usage.output_tokens,
                        output_tokens.map(TokenObservation::provider),
                    );
                    usage_update = Some((active.session_id.clone(), input_delta, output_delta));
                } else if let Some(completed) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    if output_tokens.is_some()
                        && let Some(started) = completed.generation_started_instant
                    {
                        completed.generation_finished_at = Some(SystemTime::now());
                        completed.generation_duration = Some(started.elapsed());
                    }
                    let input_delta = apply_observation(
                        &mut completed.input_tokens,
                        &mut completed.usage.input_tokens,
                        input_tokens.map(TokenObservation::provider),
                    );
                    let output_delta = apply_observation(
                        &mut completed.output_tokens,
                        &mut completed.usage.output_tokens,
                        output_tokens.map(TokenObservation::provider),
                    );
                    usage_update = Some((completed.session_id.clone(), input_delta, output_delta));
                    if output_delta > 0 {
                        history_update = Some((
                            completed.session_id.clone(),
                            completed
                                .generation_finished_at
                                .unwrap_or(completed.finished_at),
                            output_delta,
                        ));
                    }
                }
                if let Some((session_id, input_delta, output_delta)) = usage_update {
                    self.record_session_usage(session_id, input_delta, output_delta, None);
                }
                if let Some((session_id, timestamp, tokens)) = history_update {
                    self.record_session_output(session_id, timestamp, tokens);
                }
            }
            MonitorEvent::RequestCompleted {
                request_id,
                http_status,
                input_tokens,
                output_tokens,
            } => {
                self.finish(
                    &request_id,
                    RequestStatus::Completed,
                    Some(http_status),
                    input_tokens,
                    output_tokens,
                    None,
                );
            }
            MonitorEvent::RequestFailed {
                request_id,
                http_status,
                error,
            } => {
                self.finish(
                    &request_id,
                    RequestStatus::Failed,
                    http_status,
                    None,
                    None,
                    Some(error),
                );
            }
            MonitorEvent::RequestAbandoned { request_id, error } => {
                self.finish_active(
                    &request_id,
                    RequestStatus::Failed,
                    None,
                    None,
                    None,
                    Some(error),
                );
            }
        }
        self.prune_session_history_if_due(SystemTime::now());
    }

    fn finish_active(
        &mut self,
        request_id: &str,
        status: RequestStatus,
        http_status: Option<u16>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        error: Option<String>,
    ) {
        if self.active.contains_key(request_id) {
            self.finish(
                request_id,
                status,
                http_status,
                input_tokens,
                output_tokens,
                error,
            );
        }
    }

    fn finish(
        &mut self,
        request_id: &str,
        status: RequestStatus,
        http_status: Option<u16>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        error: Option<String>,
    ) {
        let mut active = self
            .active
            .remove(request_id)
            .unwrap_or_else(|| ActiveRequest {
                request_id: request_id.to_string(),
                session_id: None,
                session_seq: None,
                project: None,
                provider: None,
                model: None,
                effort: None,
                endpoint: EndpointKind::Messages,
                started_at: SystemTime::now(),
                started_instant: Instant::now(),
                generation_started_at: None,
                generation_started_instant: None,
                generation_initial_output_tokens: 0,
                generation_finished_at: None,
                generation_duration: None,
                status: RequestStatus::Started,
                streamed_bytes: 0,
                stream_chunks: 0,
                input_tokens: None,
                output_tokens: None,
                error: None,
                traffic_capture_path: None,
                usage: UsageObservation::default(),
                delivery: DeliveryTiming::default(),
            });
        if output_tokens.is_some()
            && let Some(started) = active.generation_started_instant
        {
            active.generation_finished_at = Some(SystemTime::now());
            active.generation_duration = Some(started.elapsed());
        }
        let input_delta = apply_observation(
            &mut active.input_tokens,
            &mut active.usage.input_tokens,
            input_tokens.map(TokenObservation::provider),
        );
        let output_delta = apply_observation(
            &mut active.output_tokens,
            &mut active.usage.output_tokens,
            output_tokens.map(TokenObservation::provider),
        );
        self.record_session_usage(active.session_id.clone(), input_delta, output_delta, None);
        let status = if status == RequestStatus::Completed && active.error.is_some() {
            RequestStatus::Failed
        } else {
            status
        };
        let completed = CompletedRequest {
            request_id: active.request_id,
            session_id: active.session_id,
            session_seq: active.session_seq,
            project: active.project,
            provider: active.provider,
            model: active.model,
            effort: active.effort,
            endpoint: active.endpoint,
            started_at: active.started_at,
            finished_at: SystemTime::now(),
            generation_started_at: active.generation_started_at,
            generation_started_instant: active.generation_started_instant,
            generation_initial_output_tokens: active.generation_initial_output_tokens,
            generation_finished_at: active.generation_finished_at,
            generation_duration: active.generation_duration,
            status,
            http_status,
            latency: active.started_instant.elapsed(),
            streamed_bytes: active.streamed_bytes,
            stream_chunks: active.stream_chunks,
            input_tokens: active.input_tokens,
            output_tokens: active.output_tokens,
            error: error.or(active.error),
            traffic_capture_path: active.traffic_capture_path,
            usage: active.usage,
            delivery: active.delivery,
        };
        if let Some(tokens) = completed.output_tokens.filter(|tokens| {
            *tokens > 0
                && completed
                    .usage
                    .output_tokens
                    .is_none_or(|value| value.provenance == UsageProvenance::Provider)
        }) {
            self.record_session_output(
                completed.session_id.clone(),
                completed
                    .generation_finished_at
                    .unwrap_or(completed.finished_at),
                tokens,
            );
        }
        self.recent.push_front(completed);
        while self.recent.len() > self.recent_limit {
            self.recent.pop_back();
        }
    }

    fn record_session_usage(
        &mut self,
        session_id: Option<String>,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: Option<u64>,
    ) {
        let usage = self.session_usage.entry(session_id).or_default();
        usage.input_tokens = usage.input_tokens.saturating_add(input_tokens);
        usage.output_tokens = usage.output_tokens.saturating_add(output_tokens);
        if let Some(tokens) = cached_input_tokens {
            usage.cached_input_tokens = Some(
                usage
                    .cached_input_tokens
                    .unwrap_or(0)
                    .saturating_add(tokens),
            );
        }
        usage.last_seen = SystemTime::now();
    }

    fn record_session_output(
        &mut self,
        session_id: Option<String>,
        timestamp: SystemTime,
        tokens: u64,
    ) {
        let bucket = session_token_bucket(timestamp);
        let buckets = self.session_output_buckets.entry(session_id).or_default();
        match buckets.binary_search_by_key(&bucket, |(bucket, _)| *bucket) {
            Ok(index) => buckets[index].1 = buckets[index].1.saturating_add(tokens),
            Err(index) => buckets.insert(index, (bucket, tokens)),
        }
    }

    fn prune_session_history_if_due(&mut self, now: SystemTime) {
        if self.last_session_prune_instant.elapsed() >= SESSION_PRUNE_INTERVAL {
            self.prune_session_history(now);
        }
    }

    fn prune_session_history(&mut self, now: SystemTime) {
        let retained_sessions: HashSet<_> = self
            .active
            .values()
            .map(|request| request.session_id.clone())
            .chain(self.recent.iter().map(|request| request.session_id.clone()))
            .collect();
        self.session_usage.retain(|session_id, usage| {
            let retained_by_age = match now.duration_since(usage.last_seen) {
                Ok(age) => age < SESSION_USAGE_RETENTION,
                Err(_) => true,
            };
            retained_sessions.contains(session_id) || retained_by_age
        });
        let first_bucket = first_session_output_bucket(now);
        for buckets in self.session_output_buckets.values_mut() {
            buckets.retain(|(bucket, _)| *bucket >= first_bucket);
        }
        self.session_output_buckets
            .retain(|_, buckets| !buckets.is_empty());
        self.last_session_prune_instant = Instant::now();
    }

    fn snapshot(&mut self) -> MonitorState {
        let now = SystemTime::now();
        self.prune_session_history_if_due(now);
        let mut active: Vec<_> = self.active.values().cloned().collect();
        active.sort_by_key(|request| request.started_at);
        let sessions = session_summaries(
            &active,
            &self.recent,
            &self.session_usage,
            &self.session_output_buckets,
            now,
        );
        MonitorState {
            started_at: self.started_at,
            uptime: self.started_instant.elapsed(),
            sessions,
            active,
            recent: self.recent.iter().cloned().collect(),
        }
    }
}

fn session_summaries(
    active: &[ActiveRequest],
    recent: &VecDeque<CompletedRequest>,
    session_usage: &HashMap<Option<String>, SessionUsage>,
    session_output_buckets: &HashMap<Option<String>, Vec<(u64, u64)>>,
    now: SystemTime,
) -> Vec<SessionSummary> {
    let mut sessions: HashMap<Option<String>, SessionSummary> = HashMap::new();
    for request in recent.iter().rev() {
        let entry = sessions
            .entry(request.session_id.clone())
            .or_insert_with(|| SessionSummary {
                session_id: request.session_id.clone(),
                project: request.project.clone(),
                active_count: 0,
                request_count: 0,
                failure_count: 0,
                provider: None,
                model: None,
                effort: None,
                last_seen: request.finished_at,
                input_tokens: 0,
                cached_input_tokens: None,
                output_tokens: 0,
                output_token_samples: Vec::new(),
                rate_output_tokens: 0,
                generation_duration: Duration::ZERO,
                last_status: "-".to_string(),
            });
        entry.request_count += 1;
        if request.status == RequestStatus::Failed {
            entry.failure_count += 1;
        }
        entry.project = request.project.clone().or(entry.project.clone());
        entry.provider = request.provider.clone().or(entry.provider.clone());
        entry.model = request.model.clone().or(entry.model.clone());
        entry.effort = request.effort.clone().or(entry.effort.clone());
        entry.last_seen = max_system_time(entry.last_seen, request.finished_at);
        if let (Some(tokens), Some(duration)) = (
            request
                .output_tokens
                .filter(|_| {
                    request
                        .usage
                        .output_tokens
                        .is_none_or(|value| value.provenance == UsageProvenance::Provider)
                })
                .and_then(|tokens| tokens.checked_sub(request.generation_initial_output_tokens))
                .filter(|tokens| *tokens > 0),
            request
                .generation_duration
                .filter(|duration| !duration.is_zero()),
        ) {
            entry.rate_output_tokens = entry.rate_output_tokens.saturating_add(tokens);
            entry.generation_duration = entry.generation_duration.saturating_add(duration);
        }
        entry.last_status = request.status.label().to_string();
    }

    for request in active {
        let entry = sessions
            .entry(request.session_id.clone())
            .or_insert_with(|| SessionSummary {
                session_id: request.session_id.clone(),
                project: request.project.clone(),
                active_count: 0,
                request_count: 0,
                failure_count: 0,
                provider: None,
                model: None,
                effort: None,
                last_seen: request.started_at,
                input_tokens: 0,
                cached_input_tokens: None,
                output_tokens: 0,
                output_token_samples: Vec::new(),
                rate_output_tokens: 0,
                generation_duration: Duration::ZERO,
                last_status: "-".to_string(),
            });
        entry.active_count += 1;
        entry.request_count += 1;
        entry.project = request.project.clone().or(entry.project.clone());
        entry.provider = request.provider.clone().or(entry.provider.clone());
        entry.model = request.model.clone().or(entry.model.clone());
        entry.effort = request.effort.clone().or(entry.effort.clone());
        entry.last_seen = max_system_time(entry.last_seen, request.started_at);
        if let (Some(tokens), Some(duration)) = (
            request
                .output_tokens
                .filter(|_| {
                    request
                        .usage
                        .output_tokens
                        .is_none_or(|value| value.provenance == UsageProvenance::Provider)
                })
                .and_then(|tokens| tokens.checked_sub(request.generation_initial_output_tokens))
                .filter(|tokens| *tokens > 0),
            request
                .generation_duration
                .filter(|duration| !duration.is_zero()),
        ) {
            entry.rate_output_tokens = entry.rate_output_tokens.saturating_add(tokens);
            entry.generation_duration = entry.generation_duration.saturating_add(duration);
        }
        entry.last_status = request.status.label().to_string();
    }

    let first_bucket = first_session_output_bucket(now);
    for (session_id, session) in &mut sessions {
        if let Some(usage) = session_usage.get(session_id) {
            session.input_tokens = usage.input_tokens;
            session.cached_input_tokens = usage.cached_input_tokens;
            session.output_tokens = usage.output_tokens;
        }
        if let Some(buckets) = session_output_buckets.get(session_id) {
            session.output_token_samples = buckets
                .iter()
                .filter(|(bucket, _)| *bucket >= first_bucket)
                .map(|(bucket, tokens)| (session_token_bucket_start(*bucket), *tokens))
                .collect();
        }
    }

    let mut out: Vec<_> = sessions.into_values().collect();
    out.sort_by_key(SessionSummary::label);
    out
}

fn session_token_bucket(timestamp: SystemTime) -> u64 {
    timestamp
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
        / SESSION_TOKEN_BUCKET_SECS
}

fn session_token_bucket_start(bucket: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(bucket.saturating_mul(SESSION_TOKEN_BUCKET_SECS))
}

fn first_session_output_bucket(now: SystemTime) -> u64 {
    session_token_bucket(now).saturating_sub(SESSION_OUTPUT_HISTORY_BUCKETS.saturating_sub(1))
}

fn apply_usage(
    input: &mut Option<u64>,
    output: &mut Option<u64>,
    current: &mut UsageObservation,
    incoming: UsageObservation,
) -> (u64, u64, Option<u64>) {
    let input_delta = apply_observation(input, &mut current.input_tokens, incoming.input_tokens);
    let output_delta =
        apply_observation(output, &mut current.output_tokens, incoming.output_tokens);
    if incoming.reasoning_tokens.is_some() {
        current.reasoning_tokens = incoming.reasoning_tokens;
    }
    let cached_delta = incoming.cached_input_tokens.map(|tokens| {
        let previous = current.cached_input_tokens.unwrap_or(0);
        let total = previous.max(tokens);
        current.cached_input_tokens = Some(total);
        total.saturating_sub(previous)
    });
    (input_delta, output_delta, cached_delta)
}

fn apply_observation(
    display: &mut Option<u64>,
    current: &mut Option<TokenObservation>,
    incoming: Option<TokenObservation>,
) -> u64 {
    let Some(incoming) = incoming else {
        return 0;
    };
    let previous_provider = match *current {
        Some(value) if value.provenance == UsageProvenance::Estimated => None,
        _ => *display,
    };
    if incoming.provenance == UsageProvenance::Estimated {
        if previous_provider.is_none() {
            *display = Some(incoming.tokens);
            *current = Some(incoming);
        }
        return 0;
    }
    let tokens = previous_provider.unwrap_or(0).max(incoming.tokens);
    *display = Some(tokens);
    *current = Some(TokenObservation::provider(tokens));
    tokens.saturating_sub(previous_provider.unwrap_or(0))
}

fn apply_delivery(timing: &mut DeliveryTiming, event: DeliveryEvent) {
    let now = SystemTime::now();
    match event {
        DeliveryEvent::ResponseReady => {
            timing.response_ready_at.get_or_insert(now);
        }
        DeliveryEvent::BodyYield { bytes } => {
            timing.first_body_yield_at.get_or_insert(now);
            timing.yielded_bytes = timing.yielded_bytes.saturating_add(bytes);
            timing.yielded_frames = timing.yielded_frames.saturating_add(1);
        }
        DeliveryEvent::BodyFinished => {
            timing.body_finished_at.get_or_insert(now);
        }
        DeliveryEvent::BodyCancelled => {
            timing.body_cancelled_at.get_or_insert(now);
        }
    }
}

fn max_system_time(left: SystemTime, right: SystemTime) -> SystemTime {
    if right.duration_since(left).is_ok() {
        right
    } else {
        left
    }
}

pub fn throughput(
    output_tokens: Option<u64>,
    streamed_bytes: u64,
    stream_chunks: u64,
    elapsed: Duration,
) -> Throughput {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return Throughput::None;
    }
    if let Some(tokens) = output_tokens.filter(|tokens| *tokens > 0) {
        return Throughput::TokensPerSecond(tokens as f64 / secs);
    }
    if streamed_bytes > 0 {
        return Throughput::BytesPerSecond(streamed_bytes as f64 / secs);
    }
    if stream_chunks > 0 {
        return Throughput::EventsPerSecond(stream_chunks as f64 / secs);
    }
    Throughput::None
}

pub fn usage_from_anthropic_sse(bytes: &[u8]) -> (Option<u64>, Option<u64>) {
    let text = String::from_utf8_lossy(bytes);
    let mut input_tokens = None;
    let mut output_tokens = None;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data.trim()) else {
            continue;
        };
        for usage in [
            value.pointer("/usage"),
            value.pointer("/delta/usage"),
            value.pointer("/message/usage"),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(tokens) = usage.get("input_tokens").and_then(|value| value.as_u64()) {
                input_tokens = Some(tokens);
            }
            if let Some(tokens) = usage.get("output_tokens").and_then(|value| value.as_u64()) {
                output_tokens = Some(tokens);
            }
        }
    }
    (input_tokens, output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_input_counts_once_and_survives_recent_eviction() {
        let monitor = MonitorHandle::new(1);
        monitor.request_started("cached", Some("s".into()), None, EndpointKind::Messages);
        let usage: UsageObservation = serde_json::from_value(serde_json::json!({
            "input_tokens": {"tokens": 790845, "provenance": "provider"},
            "cached_input_tokens": 782848
        }))
        .unwrap();
        monitor.typed_progress("cached", 10, 1, usage.clone());
        monitor.typed_progress("cached", 0, 0, usage.clone());
        monitor.request_completed("cached", 200, Some(790845), None);
        monitor.typed_progress("cached", 0, 0, usage.clone());
        let before =
            serde_json::to_value(snapshot::MonitorSnapshot::from(monitor.snapshot())).unwrap();
        assert_eq!(before["recent"][0]["usage"]["cached_input_tokens"], 782848);
        assert_eq!(before["sessions"][0]["cached_input_tokens"], 782848);
        monitor.request_started("next", Some("s".into()), None, EndpointKind::Messages);
        monitor.request_completed("next", 200, Some(1), None);
        monitor.typed_progress("cached", 0, 0, usage);
        let after =
            serde_json::to_value(snapshot::MonitorSnapshot::from(monitor.snapshot())).unwrap();
        assert_eq!(after["recent"][0]["request_id"], "next");
        assert_eq!(after["sessions"][0]["request_count"], 1);
        assert_eq!(after["sessions"][0]["input_tokens"], 790846);
        assert_eq!(after["sessions"][0]["cached_input_tokens"], 782848);
        monitor.request_started("more", Some("s".into()), None, EndpointKind::Messages);
        for cached in [64, 256, 32] {
            monitor.typed_progress(
                "more",
                0,
                0,
                UsageObservation {
                    input_tokens: Some(TokenObservation::provider(400)),
                    cached_input_tokens: Some(cached),
                    ..Default::default()
                },
            );
        }
        monitor.request_completed("more", 200, Some(400), None);
        let state = monitor.snapshot();
        assert_eq!(state.recent[0].usage.cached_input_tokens, Some(256));
        assert_eq!(state.sessions[0].input_tokens, 791246);
        assert_eq!(state.sessions[0].cached_input_tokens, Some(783104));
    }

    #[test]
    fn cached_input_distinguishes_unknown_and_reported_zero() {
        let monitor = MonitorHandle::default();
        monitor.request_started(
            "unknown",
            Some("unknown".into()),
            None,
            EndpointKind::Messages,
        );
        monitor.request_completed("unknown", 200, Some(10), None);
        monitor.request_started("zero", Some("zero".into()), None, EndpointKind::Messages);
        let zero = serde_json::from_value(serde_json::json!({"cached_input_tokens": 0})).unwrap();
        monitor.typed_progress("zero", 0, 0, zero);
        monitor.typed_progress("zero", 0, 0, UsageObservation::default());
        monitor.request_completed("zero", 200, Some(10), None);
        let value =
            serde_json::to_value(snapshot::MonitorSnapshot::from(monitor.snapshot())).unwrap();
        assert_eq!(value["recent"][0]["usage"]["cached_input_tokens"], 0);
        assert_eq!(
            value["recent"][1]["usage"]["cached_input_tokens"],
            serde_json::Value::Null
        );
        assert_eq!(
            value["sessions"][0]["cached_input_tokens"],
            serde_json::Value::Null
        );
        assert_eq!(value["sessions"][1]["cached_input_tokens"], 0);
    }

    #[test]
    fn typed_usage_replaces_estimates_and_counts_cumulative_provider_fields_once() {
        let monitor = MonitorHandle::default();
        monitor.request_started("r", Some("s".into()), None, EndpointKind::Messages);
        monitor.typed_progress(
            "r",
            10,
            2,
            UsageObservation {
                input_tokens: Some(TokenObservation::estimated(321)),
                output_tokens: Some(TokenObservation::estimated(99)),
                ..Default::default()
            },
        );
        assert_eq!(monitor.snapshot().sessions[0].input_tokens, 0);
        assert_eq!(monitor.snapshot().sessions[0].output_tokens, 0);
        let actual = UsageObservation {
            input_tokens: Some(TokenObservation::provider(300)),
            output_tokens: Some(TokenObservation::provider(48)),
            reasoning_tokens: None,
            ..Default::default()
        };
        monitor.typed_progress("r", 5, 1, actual.clone());
        monitor.typed_progress("r", 0, 0, actual.clone());
        monitor.request_completed("r", 200, None, None);
        monitor.typed_progress("r", 0, 0, actual);
        let state = monitor.snapshot();
        assert_eq!(state.recent[0].input_tokens, Some(300));
        assert_eq!(state.recent[0].output_tokens, Some(48));
        assert_eq!(state.recent[0].usage.reasoning_tokens, None);
        assert_eq!(state.recent[0].stream_chunks, 3);
        assert_eq!(state.sessions[0].input_tokens, 300);
        assert_eq!(state.sessions[0].output_tokens, 48);
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, count)| count)
                .sum::<u64>(),
            48
        );
    }

    #[test]
    fn unknown_usage_is_distinct_from_provider_zero_and_estimate_only_history() {
        let monitor = MonitorHandle::default();
        monitor.request_started("r", None, None, EndpointKind::Messages);
        monitor.typed_progress(
            "r",
            1,
            1,
            UsageObservation {
                input_tokens: Some(TokenObservation::estimated(321)),
                ..Default::default()
            },
        );
        assert_eq!(monitor.snapshot().active[0].output_tokens, None);
        monitor.typed_progress(
            "r",
            0,
            0,
            UsageObservation {
                output_tokens: Some(TokenObservation::provider(0)),
                ..Default::default()
            },
        );
        assert_eq!(monitor.snapshot().active[0].output_tokens, Some(0));
        assert_eq!(
            monitor.snapshot().active[0]
                .usage
                .input_tokens
                .unwrap()
                .provenance,
            UsageProvenance::Estimated
        );
        monitor.request_completed("r", 200, None, None);
        assert_eq!(monitor.snapshot().sessions[0].input_tokens, 0);
        assert!(
            monitor.snapshot().sessions[0]
                .output_token_samples
                .is_empty()
        );
    }

    #[test]
    fn started_requests_appear_active() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(3),
            EndpointKind::Messages,
        );
        let state = monitor.snapshot();
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.active[0].request_id, "r1");
        assert_eq!(state.active[0].session_id.as_deref(), Some("s1"));
        assert_eq!(state.active[0].session_seq, Some(3));
    }

    #[test]
    fn uptime_uses_the_service_monotonic_clock() {
        let monitor = MonitorHandle::new(10);
        {
            let mut store = monitor.store.lock().unwrap();
            store.started_at = SystemTime::now() + Duration::from_secs(3_600);
            store.started_instant = Instant::now() - Duration::from_secs(42);
        }

        let state = monitor.snapshot();

        assert!(state.started_at > SystemTime::now());
        assert!(state.uptime >= Duration::from_secs(42));
        assert!(state.uptime < Duration::from_secs(43));
    }

    #[test]
    fn resolved_model_appends_to_incoming_alias() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "claude-sonnet-4-6", None);
        monitor.model_resolved("r1", "gpt-5.4");

        let state = monitor.snapshot();
        assert_eq!(
            state.active[0].model.as_deref(),
            Some("claude-sonnet-4-6 → gpt-5.4")
        );
    }

    #[test]
    fn identical_resolved_model_is_shown_once() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.6-sol", None);
        monitor.model_resolved("r1", "gpt-5.6-sol");

        let state = monitor.snapshot();
        assert_eq!(state.active[0].model.as_deref(), Some("gpt-5.6-sol"));
    }

    #[test]
    fn compaction_started_marks_request_compacting() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.6-sol", None);
        monitor.compaction_started("r1");

        let state = monitor.snapshot();
        assert_eq!(state.active[0].status, RequestStatus::Compacting);
        assert_eq!(state.sessions[0].last_status, "compacting");
    }

    #[test]
    fn generation_baseline_pairs_total_usage_with_the_full_observed_interval() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.generation_started("r1");
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));
        monitor.request_completed("r1", 200, None, None);

        let request = &monitor.snapshot().recent[0];

        assert!(
            request
                .generation_duration
                .is_some_and(|duration| !duration.is_zero())
        );
        assert!(matches!(request.rate(), Throughput::TokensPerSecond(_)));
    }

    #[test]
    fn first_stream_progress_has_no_rate_without_an_interval() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));

        let state = monitor.snapshot();
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.active[0].rate(), Throughput::None);
    }

    #[test]
    fn late_stream_progress_extends_stream_timing_without_extending_request_latency() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.stream_progress("r1", 100, 1, Some(0), Some(0));
        monitor.request_completed("r1", 200, None, None);
        let completed = monitor.snapshot().recent[0].clone();
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));

        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent[0].streamed_bytes, 150);
        assert_eq!(state.recent[0].stream_chunks, 2);
        assert_eq!(state.recent[0].input_tokens, Some(1_225));
        assert_eq!(state.recent[0].output_tokens, Some(141));
        assert_eq!(state.recent[0].finished_at, completed.finished_at);
        assert_eq!(state.recent[0].latency, completed.latency);
        assert!(state.recent[0].generation_duration > completed.generation_duration);
        assert!(matches!(
            state.recent[0].rate(),
            Throughput::TokensPerSecond(_)
        ));
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            141
        );
    }

    #[test]
    fn completed_requests_leave_active_and_enter_recent() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.5", Some("high".to_string()));
        monitor.request_completed("r1", 200, Some(10), Some(20));
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].provider.as_deref(), Some("codex"));
        assert_eq!(state.recent[0].effort.as_deref(), Some("high"));
        assert_eq!(state.recent[0].output_tokens, Some(20));
    }

    #[test]
    fn failed_requests_preserve_error_summary() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_failed("r1", Some(400), "Unknown model");
        let state = monitor.snapshot();
        assert_eq!(state.recent[0].status, RequestStatus::Failed);
        assert_eq!(state.recent[0].http_status, Some(400));
        assert_eq!(state.recent[0].error.as_deref(), Some("Unknown model"));
    }

    #[test]
    fn abandoned_requests_leave_active_once() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_abandoned("r1", "request dropped");
        monitor.request_abandoned("r1", "request dropped again");
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].status, RequestStatus::Failed);
        assert_eq!(state.recent[0].http_status, None);
        assert_eq!(state.recent[0].error.as_deref(), Some("request dropped"));
    }

    #[test]
    fn completed_requests_ignore_late_abandonment() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_completed("r1", 200, None, None);
        monitor.request_abandoned("r1", "request dropped");
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].status, RequestStatus::Completed);
    }

    #[test]
    fn bounded_recent_history_drops_oldest() {
        let monitor = MonitorHandle::new(2);
        for id in ["r1", "r2", "r3"] {
            monitor.request_started(id, None, None, EndpointKind::Messages);
            monitor.request_completed(id, 200, None, None);
        }
        let state = monitor.snapshot();
        let ids: Vec<_> = state
            .recent
            .iter()
            .map(|request| request.request_id.as_str())
            .collect();
        assert_eq!(ids, vec!["r3", "r2"]);
    }

    #[test]
    fn throughput_selects_best_available_signal() {
        let elapsed = Duration::from_secs(2);
        assert_eq!(
            throughput(Some(84), 1024, 10, elapsed),
            Throughput::TokensPerSecond(42.0)
        );
        assert_eq!(
            throughput(None, 2048, 10, elapsed),
            Throughput::BytesPerSecond(1024.0)
        );
        assert_eq!(
            throughput(None, 0, 36, elapsed),
            Throughput::EventsPerSecond(18.0)
        );
    }

    #[test]
    fn sse_usage_extracts_final_message_delta_tokens() {
        let sse = br#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"output_tokens":48}}

"#;
        assert_eq!(usage_from_anthropic_sse(sse), (Some(12), Some(48)));
    }

    fn completed_request(
        request_id: &str,
        session_id: &str,
        output_tokens: u64,
        latency: Duration,
        generation_duration: Option<Duration>,
    ) -> CompletedRequest {
        CompletedRequest {
            request_id: request_id.to_string(),
            session_id: Some(session_id.to_string()),
            session_seq: None,
            project: None,
            provider: Some("codex".to_string()),
            model: Some("gpt-5.6-sol".to_string()),
            effort: None,
            endpoint: EndpointKind::Messages,
            started_at: SystemTime::UNIX_EPOCH,
            finished_at: SystemTime::UNIX_EPOCH + latency,
            generation_started_at: generation_duration.map(|_| SystemTime::UNIX_EPOCH),
            generation_started_instant: None,
            generation_initial_output_tokens: 0,
            generation_finished_at: generation_duration
                .map(|duration| SystemTime::UNIX_EPOCH + duration),
            generation_duration,
            status: RequestStatus::Completed,
            http_status: Some(200),
            latency,
            streamed_bytes: 0,
            stream_chunks: 0,
            input_tokens: None,
            output_tokens: Some(output_tokens),
            error: None,
            traffic_capture_path: None,
            usage: UsageObservation::default(),
            delivery: DeliveryTiming::default(),
        }
    }

    fn session_summaries_for_requests(recent: &VecDeque<CompletedRequest>) -> Vec<SessionSummary> {
        let mut usage = HashMap::<Option<String>, SessionUsage>::new();
        for request in recent {
            let entry = usage.entry(request.session_id.clone()).or_default();
            entry.input_tokens = entry
                .input_tokens
                .saturating_add(request.input_tokens.unwrap_or(0));
            entry.output_tokens = entry
                .output_tokens
                .saturating_add(request.output_tokens.unwrap_or(0));
        }
        session_summaries(&[], recent, &usage, &HashMap::new(), SystemTime::now())
    }

    #[test]
    fn completed_request_rate_uses_stream_interval_instead_of_request_latency() {
        let request = completed_request(
            "r1",
            "s1",
            120,
            Duration::from_secs(30),
            Some(Duration::from_secs(4)),
        );

        assert_eq!(request.rate(), Throughput::TokensPerSecond(30.0));
    }

    #[test]
    fn request_rate_uses_token_delta_from_the_initial_observation() {
        let mut request = completed_request(
            "r1",
            "s1",
            120,
            Duration::from_secs(30),
            Some(Duration::from_secs(4)),
        );
        request.generation_initial_output_tokens = 20;

        assert_eq!(request.rate(), Throughput::TokensPerSecond(25.0));
    }

    #[test]
    fn session_rate_combines_request_tokens_and_generation_intervals() {
        let recent = VecDeque::from([
            completed_request(
                "r2",
                "s1",
                50,
                Duration::from_secs(40),
                Some(Duration::from_secs(1)),
            ),
            completed_request(
                "r1",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        let sessions = session_summaries_for_requests(&recent);

        assert_eq!(sessions[0].output_tokens, 150);
        assert_eq!(sessions[0].generation_duration, Duration::from_secs(5));
        assert_eq!(sessions[0].rate(), Throughput::TokensPerSecond(30.0));
    }

    #[test]
    fn output_without_observed_stream_interval_has_no_output_rate() {
        let request = completed_request("r1", "s1", 120, Duration::from_secs(30), None);
        let recent = VecDeque::from([request.clone()]);

        assert_eq!(request.rate(), Throughput::None);
        assert_eq!(
            session_summaries_for_requests(&recent)[0].rate(),
            Throughput::None
        );
    }

    #[test]
    fn session_rate_excludes_interval_without_output_usage() {
        let mut tokenless = completed_request(
            "tokenless",
            "s1",
            0,
            Duration::from_secs(30),
            Some(Duration::from_secs(100)),
        );
        tokenless.output_tokens = None;
        let recent = VecDeque::from([
            tokenless,
            completed_request(
                "measured",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        assert_eq!(
            session_summaries_for_requests(&recent)[0].rate(),
            Throughput::TokensPerSecond(25.0)
        );
    }

    #[test]
    fn session_rate_excludes_output_without_a_matching_stream_interval() {
        let recent = VecDeque::from([
            completed_request("buffered", "s1", 900, Duration::from_secs(30), None),
            completed_request(
                "streamed",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        let session = &session_summaries_for_requests(&recent)[0];

        assert_eq!(session.output_tokens, 1_000);
        assert_eq!(session.rate(), Throughput::TokensPerSecond(25.0));
    }

    #[test]
    fn session_summaries_group_recent_and_active_requests() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.project_resolved("r1", "example");
        monitor.provider_selected("r1", "codex", "gpt-5.5", None);
        monitor.request_completed("r1", 200, Some(10), Some(20));
        monitor.request_started(
            "r2",
            Some("s1".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        monitor.provider_selected("r2", "codex", "gpt-5.5", Some("xhigh".to_string()));
        let state = monitor.snapshot();
        assert_eq!(state.sessions.len(), 1);
        assert_eq!(state.sessions[0].label(), "s1");
        assert_eq!(state.sessions[0].project.as_deref(), Some("example"));
        assert_eq!(state.sessions[0].request_count, 2);
        assert_eq!(state.sessions[0].active_count, 1);
        assert_eq!(state.sessions[0].effort.as_deref(), Some("xhigh"));
        assert_eq!(state.sessions[0].output_tokens, 20);
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .collect::<Vec<_>>(),
            vec![20]
        );
    }

    #[test]
    fn session_output_history_survives_request_eviction() {
        let monitor = MonitorHandle::new(1);
        for (request_id, tokens) in [("oldest", 20), ("newest", 80)] {
            monitor.request_started(
                request_id,
                Some("s1".to_string()),
                None,
                EndpointKind::Messages,
            );
            monitor.request_completed(request_id, 200, Some(tokens * 10), Some(tokens));
        }

        let state = monitor.snapshot();

        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.sessions[0].input_tokens, 1_000);
        assert_eq!(state.sessions[0].output_tokens, 100);
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            100
        );
    }

    #[test]
    fn session_output_history_prunes_old_buckets_without_changing_totals() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", Some("s1".to_string()), None, EndpointKind::Messages);
        monitor.request_completed("r1", 200, Some(100), Some(20));

        let now = SystemTime::now();
        let current_bucket = session_token_bucket(now);
        {
            let mut store = monitor.store.lock().unwrap();
            store
                .session_output_buckets
                .get_mut(&Some("s1".into()))
                .unwrap()
                .insert(
                    0,
                    (
                        current_bucket.saturating_sub(SESSION_OUTPUT_HISTORY_BUCKETS),
                        900,
                    ),
                );
            store.prune_session_history(now);
            assert_eq!(
                store
                    .session_output_buckets
                    .get(&Some("s1".into()))
                    .unwrap()
                    .len(),
                1
            );
        }

        let session = &monitor.snapshot().sessions[0];

        assert_eq!(session.input_tokens, 100);
        assert_eq!(session.output_tokens, 20);
        assert_eq!(
            session
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            20
        );
        assert!(
            session
                .output_token_samples
                .iter()
                .all(|(timestamp, _)| session_token_bucket(*timestamp)
                    >= current_bucket.saturating_sub(SESSION_OUTPUT_HISTORY_BUCKETS - 1))
        );
    }

    #[test]
    fn snapshot_filters_old_buckets_before_prune_is_due() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", Some("s1".to_string()), None, EndpointKind::Messages);
        monitor.request_completed("r1", 200, Some(100), Some(20));

        let now = SystemTime::now();
        let current_bucket = session_token_bucket(now);
        let old_bucket = current_bucket.saturating_sub(SESSION_OUTPUT_HISTORY_BUCKETS);
        {
            let mut store = monitor.store.lock().unwrap();
            store
                .session_output_buckets
                .get_mut(&Some("s1".into()))
                .unwrap()
                .insert(0, (old_bucket, 900));
            store.last_session_prune_instant = Instant::now();
        }

        let session = &monitor.snapshot().sessions[0];

        assert_eq!(session.output_tokens, 20);
        assert_eq!(
            session
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            20
        );
        let store = monitor.store.lock().unwrap();
        assert!(
            store
                .session_output_buckets
                .get(&Some("s1".into()))
                .unwrap()
                .iter()
                .any(|(bucket, tokens)| *bucket == old_bucket && *tokens == 900)
        );
    }

    #[test]
    fn inactive_session_accounting_expires_without_affecting_active_totals() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "active",
            Some("active-session".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.usage_updated("active", Some(12), Some(7));
        monitor.request_started(
            "old",
            Some("old-session".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.request_completed("old", 200, Some(30), Some(11));

        let old_timestamp = SystemTime::now() - SESSION_USAGE_RETENTION - Duration::from_secs(1);
        let old_bucket = session_token_bucket(old_timestamp);
        {
            let mut store = monitor.store.lock().unwrap();
            store.recent.clear();
            store
                .session_usage
                .get_mut(&Some("active-session".into()))
                .unwrap()
                .last_seen = old_timestamp;
            store
                .session_usage
                .get_mut(&Some("old-session".into()))
                .unwrap()
                .last_seen = old_timestamp;
            store
                .session_output_buckets
                .insert(Some("old-session".into()), vec![(old_bucket, 11)]);
            store.last_session_prune_instant = Instant::now() - SESSION_PRUNE_INTERVAL;
        }

        let state = monitor.snapshot();

        assert_eq!(state.sessions.len(), 1);
        assert_eq!(state.sessions[0].label(), "active-session");
        assert_eq!(state.sessions[0].input_tokens, 12);
        assert_eq!(state.sessions[0].output_tokens, 7);
        let store = monitor.store.lock().unwrap();
        assert!(
            !store
                .session_usage
                .contains_key(&Some("old-session".into()))
        );
        assert!(
            !store
                .session_output_buckets
                .contains_key(&Some("old-session".into()))
        );
    }

    #[test]
    fn session_usage_ignores_decreasing_request_observations() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.usage_updated("r1", Some(100), Some(20));
        monitor.usage_updated("r1", Some(90), Some(15));
        monitor.usage_updated("r1", Some(120), Some(25));

        let active = monitor.snapshot();
        assert_eq!(active.active[0].input_tokens, Some(120));
        assert_eq!(active.active[0].output_tokens, Some(25));
        assert_eq!(active.sessions[0].input_tokens, 120);
        assert_eq!(active.sessions[0].output_tokens, 25);

        monitor.request_completed("r1", 200, Some(80), Some(10));
        let completed = monitor.snapshot();
        assert_eq!(completed.recent[0].input_tokens, Some(120));
        assert_eq!(completed.recent[0].output_tokens, Some(25));
        assert_eq!(completed.sessions[0].input_tokens, 120);
        assert_eq!(completed.sessions[0].output_tokens, 25);
    }

    #[test]
    fn compaction_preserves_cumulative_session_usage() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "before",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.request_completed("before", 200, Some(100), Some(20));
        monitor.request_started(
            "compact",
            Some("s1".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        monitor.compaction_started("compact");
        monitor.request_completed("compact", 200, Some(40), Some(10));

        let state = monitor.snapshot();
        assert_eq!(state.sessions[0].input_tokens, 140);
        assert_eq!(state.sessions[0].output_tokens, 30);
    }

    #[test]
    fn session_sequence_restart_preserves_cumulative_usage() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "before",
            Some("s1".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.session_sequence_resolved("before", 7);
        monitor.request_completed("before", 200, Some(100), Some(20));

        monitor.request_started(
            "after",
            Some("s1".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.session_sequence_resolved("after", 1);
        monitor.request_completed("after", 200, Some(25), Some(5));

        let state = monitor.snapshot();
        assert_eq!(state.sessions[0].input_tokens, 125);
        assert_eq!(state.sessions[0].output_tokens, 25);
    }

    #[test]
    fn session_order_is_stable_across_activity() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("session-b".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.request_started(
            "r2",
            Some("session-a".to_string()),
            Some(1),
            EndpointKind::Messages,
        );

        let first: Vec<_> = monitor
            .snapshot()
            .sessions
            .iter()
            .map(SessionSummary::label)
            .collect();
        monitor.request_completed("r1", 200, None, None);
        monitor.request_started(
            "r3",
            Some("session-b".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        let second: Vec<_> = monitor
            .snapshot()
            .sessions
            .iter()
            .map(SessionSummary::label)
            .collect();

        assert_eq!(first, vec!["session-a", "session-b"]);
        assert_eq!(second, first);
    }
}
