//! Opt-in, request-local observations for synthetic performance experiments.
//! Missing timestamps/counters mean unknown; consumer yield is not wire delivery.
use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Measurements {
    pub preparation_start: Option<Duration>,
    pub preparation_end: Option<Duration>,
    pub gate_start: Option<Duration>,
    pub gate_end: Option<Duration>,
    pub handshake_end: Option<Duration>,
    pub validation_end: Option<Duration>,
    pub response_headers: Option<Duration>,
    pub first_upstream_event: Option<Duration>,
    pub first_upstream_text: Option<Duration>,
    pub first_semantic: Option<Duration>,
    pub first_downstream_yield: Option<Duration>,
    /// Upstream terminal consumed by the HTTP reader; not downstream body EOF.
    pub body_complete: Option<Duration>,
    pub body_cancel: Option<Duration>,
    /// HTTP reader task finalization; not server request finalization.
    pub finalization: Option<Duration>,
    /// HTTP ResponsesRequest JSON bytes; excludes other serializers.
    pub serialized_bytes: u64,
    /// HTTP body_json.to_string bytes; excludes provider/continuation handoff clones.
    pub full_request_clone_bytes: u64,
    pub config_reads: u64,
    pub config_parses: u64,
    pub attempts: u64,
}

#[derive(Clone)]
pub(crate) struct Observation {
    started: Instant,
    measurements: Arc<Mutex<Measurements>>,
}
tokio::task_local! { static CURRENT: Observation; }

pub async fn observe<F: Future>(future: F) -> (F::Output, Measurements) {
    let observation = Observation {
        started: Instant::now(),
        measurements: Arc::new(Mutex::new(Measurements::default())),
    };
    let output = CURRENT.scope(observation.clone(), future).await;
    let measurements = observation.measurements.lock().unwrap().clone();
    (output, measurements)
}

pub(crate) fn current() -> Option<Observation> {
    CURRENT.try_with(Clone::clone).ok()
}

pub(crate) fn record(update: impl FnOnce(&mut Measurements, Duration)) {
    let _ = CURRENT.try_with(|observation| {
        if let Ok(mut measurements) = observation.measurements.lock() {
            update(&mut measurements, observation.started.elapsed());
        }
    });
}

pub(crate) async fn in_scope<F: Future>(observation: Option<Observation>, future: F) -> F::Output {
    match observation {
        Some(observation) => CURRENT.scope(observation, future).await,
        None => future.await,
    }
}

pub(crate) fn in_sync_scope<T>(observation: Option<Observation>, run: impl FnOnce() -> T) -> T {
    match observation {
        Some(observation) => CURRENT.sync_scope(observation, run),
        None => run(),
    }
}

pub(crate) struct ReaderFinalization;
impl Drop for ReaderFinalization {
    fn drop(&mut self) {
        record(|measurements, elapsed| {
            measurements.finalization = Some(elapsed);
        });
    }
}
