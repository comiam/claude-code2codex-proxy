//! One process-wide, best-effort diagnostic writer. Queue limits exclude one
//! in-flight record (at most MAX_RECORD_BYTES), and fixed queue/thread storage.
use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub const MAX_QUEUE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_QUEUE_RECORDS: usize = 2048;
pub const MAX_RECORD_BYTES: usize = 2 * 1024 * 1024;
/// Shared decimal 1 GB ceiling for this writer's managed diagnostic files.
pub const MAX_DIAGNOSTIC_DISK_BYTES: u64 = 1_000_000_000;

#[derive(Debug, Default)]
pub(crate) struct CaptureProgress {
    pub accepted: AtomicU64,
    pub dropped: AtomicU64,
    pub failed: AtomicU64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticsSnapshot {
    pub accepted: u64,
    pub dropped: u64,
    pub failed: u64,
    pub written: u64,
    pub queued_bytes: u64,
    pub queued_records: u64,
    pub inflight_bytes: u64,
}

#[derive(Default)]
struct Counters {
    accepted: AtomicU64,
    dropped: AtomicU64,
    failed: AtomicU64,
    written: AtomicU64,
    queued_bytes: AtomicU64,
    queued_records: AtomicU64,
    inflight_bytes: AtomicU64,
}

pub(crate) struct Record {
    path: Box<Path>,
    payload: Box<[u8]>,
    append: bool,
    stderr: bool,
    capture: Option<Arc<CaptureProgress>>,
    completion: bool,
    #[cfg(test)]
    pause: Option<Arc<FixturePause>>,
}
impl Record {
    // Only the logger and the traffic module's redacted boundary construct records.
    pub(super) fn log(path: PathBuf, payload: Vec<u8>, stderr: bool) -> Self {
        Self {
            path: path.into_boxed_path(),
            payload: payload.into_boxed_slice(),
            append: true,
            stderr,
            capture: None,
            completion: false,
            #[cfg(test)]
            pause: None,
        }
    }
    pub(crate) fn capture(
        path: PathBuf,
        payload: Vec<u8>,
        progress: Arc<CaptureProgress>,
        completion: bool,
    ) -> Self {
        Self {
            path: path.into_boxed_path(),
            payload: payload.into_boxed_slice(),
            append: false,
            stderr: false,
            capture: Some(progress),
            completion,
            #[cfg(test)]
            pause: None,
        }
    }
    pub(crate) fn file(path: PathBuf, payload: Vec<u8>) -> Self {
        let mut record = Self::log(path, payload, false);
        record.append = false;
        record
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.path.as_os_str().len()
            + self.payload.len()
            + self.capture.as_ref().map_or(0, |_| {
                std::mem::size_of::<CaptureProgress>() + 2 * std::mem::size_of::<usize>()
            })
    }
}

#[derive(Default)]
struct Queue {
    records: VecDeque<Record>,
    bytes: usize,
    closed: bool,
    completed: u64,
}
struct Shared {
    queue: Mutex<Queue>,
    changed: Condvar,
    counters: Counters,
}
pub(crate) struct Writer {
    shared: Arc<Shared>,
}

impl Writer {
    fn start() -> Self {
        let mut retention = Retention::new(MAX_DIAGNOSTIC_DISK_BYTES);
        Self::with_sink(move |record| write_record_with_retention(record, Some(&mut retention)))
    }
    fn with_sink(mut sink: impl FnMut(&Record) -> io::Result<()> + Send + 'static) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            changed: Condvar::new(),
            counters: Counters::default(),
        });
        let worker = shared.clone();
        // A detached native thread cannot hold Tokio runtime teardown hostage.
        let spawned = std::thread::Builder::new()
            .name("diagnostic-writer".into())
            .spawn(move || {
                loop {
                    let record = {
                        let mut queue = worker.queue.lock().unwrap_or_else(|e| e.into_inner());
                        while queue.records.is_empty() && !queue.closed {
                            queue = worker
                                .changed
                                .wait(queue)
                                .unwrap_or_else(|e| e.into_inner());
                        }
                        let Some(record) = queue.records.pop_front() else {
                            worker.changed.notify_all();
                            return;
                        };
                        queue.bytes -= record.bytes();
                        worker
                            .counters
                            .queued_bytes
                            .store(queue.bytes as u64, Ordering::Relaxed);
                        worker
                            .counters
                            .queued_records
                            .store(queue.records.len() as u64, Ordering::Relaxed);
                        worker
                            .counters
                            .inflight_bytes
                            .store(record.bytes() as u64, Ordering::Relaxed);
                        record
                    };
                    // No filesystem or stderr operation runs under the queue mutex.
                    match sink(&record) {
                        Err(error) if error.kind() == io::ErrorKind::QuotaExceeded => {
                            worker.counters.dropped.fetch_add(1, Ordering::Relaxed);
                            if let Some(capture) = &record.capture {
                                capture.dropped.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(_) => {
                            worker.counters.failed.fetch_add(1, Ordering::Relaxed);
                            if let Some(capture) = &record.capture {
                                capture.failed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Ok(()) => {
                            worker.counters.written.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    drop(record);
                    worker.counters.inflight_bytes.store(0, Ordering::Relaxed);
                    let mut queue = worker.queue.lock().unwrap_or_else(|e| e.into_inner());
                    queue.completed += 1;
                    worker.changed.notify_all();
                }
            });
        if spawned.is_err() {
            shared
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .closed = true;
            shared.counters.failed.fetch_add(1, Ordering::Relaxed);
        }
        Self { shared }
    }

    pub(crate) fn omit(&self, capture: Option<&CaptureProgress>) {
        self.shared.counters.dropped.fetch_add(1, Ordering::Relaxed);
        if let Some(capture) = capture {
            capture.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn enqueue(&self, record: Record) -> bool {
        let bytes = record.bytes();
        if bytes <= MAX_RECORD_BYTES
            && let Ok(mut queue) = self.shared.queue.try_lock()
            && !queue.closed
            && queue.records.len() < MAX_QUEUE_RECORDS
            && queue.bytes.saturating_add(bytes) <= MAX_QUEUE_BYTES
        {
            if let Some(capture) = &record.capture {
                capture.accepted.fetch_add(1, Ordering::Relaxed);
            }
            queue.bytes += bytes;
            queue.records.push_back(record);
            self.shared
                .counters
                .accepted
                .fetch_add(1, Ordering::Relaxed);
            self.shared
                .counters
                .queued_bytes
                .store(queue.bytes as u64, Ordering::Relaxed);
            self.shared
                .counters
                .queued_records
                .store(queue.records.len() as u64, Ordering::Relaxed);
            self.shared.changed.notify_one();
            return true;
        }
        self.omit(record.capture.as_deref());
        false
    }

    pub(crate) fn snapshot(&self) -> DiagnosticsSnapshot {
        let c = &self.shared.counters;
        DiagnosticsSnapshot {
            accepted: c.accepted.load(Ordering::Relaxed),
            dropped: c.dropped.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            written: c.written.load(Ordering::Relaxed),
            queued_bytes: c.queued_bytes.load(Ordering::Relaxed),
            queued_records: c.queued_records.load(Ordering::Relaxed),
            inflight_bytes: c.inflight_bytes.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn flush(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let target = self.shared.counters.accepted.load(Ordering::Relaxed);
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while queue.completed < target {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            let (next, result) = self
                .shared
                .changed
                .wait_timeout(queue, remaining)
                .unwrap_or_else(|e| e.into_inner());
            queue = next;
            if result.timed_out() && queue.completed < target {
                return false;
            }
        }
        true
    }

    pub(crate) fn shutdown(&self, timeout: Duration) -> bool {
        {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.closed = true;
            self.shared.changed.notify_all();
        }
        self.flush(timeout.min(Duration::from_secs(2)))
    }
}

static WRITER: OnceLock<Writer> = OnceLock::new();
pub(crate) fn global() -> &'static Writer {
    WRITER.get_or_init(Writer::start)
}
pub fn diagnostics_snapshot() -> DiagnosticsSnapshot {
    WRITER
        .get()
        .map_or_else(DiagnosticsSnapshot::default, Writer::snapshot)
}
/// Explicit fixture drain; request and body delivery must never call this.
#[doc(hidden)]
pub fn flush_diagnostics(timeout: Duration) -> bool {
    WRITER.get().is_none_or(|writer| writer.flush(timeout))
}
pub fn shutdown_diagnostics() -> bool {
    WRITER
        .get()
        .is_none_or(|writer| writer.shutdown(Duration::from_secs(2)))
}

// Only native-worker code scans, prunes or accounts for disk usage. Roots are
// discovered from known destinations, never from arbitrary directory contents.
struct Retention {
    limit: u64,
    total: u64,
    roots: Vec<PathBuf>,
    rescan: bool,
}

impl Retention {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            total: 0,
            roots: Vec::new(),
            rescan: false,
        }
    }

    fn reserve(&mut self, record: &Record, bytes: u64) -> io::Result<Option<u64>> {
        let Some(root) = diagnostic_root(record) else {
            return Ok(None);
        };
        if bytes > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::QuotaExceeded,
                "diagnostic disk budget exceeded",
            ));
        }
        if !self.roots.contains(&root) {
            self.roots.push(root);
            self.rescan = true;
        }
        let mut previous = regular_file_len(&record.path)?;
        let projected = |total: u64, previous: u64| {
            total
                .saturating_sub(previous)
                .saturating_add(if record.append {
                    previous.saturating_add(bytes)
                } else {
                    bytes
                })
        };
        if self.rescan || projected(self.total, previous) > self.limit {
            let mut files = Vec::new();
            for root in &self.roots {
                collect_diagnostics(root, &mut files)?;
            }
            self.total = files
                .iter()
                .map(|file| file.1)
                .fold(0u64, u64::saturating_add);
            self.rescan = false;
            if projected(self.total, previous) > self.limit {
                files.sort_unstable_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
                for (path, _, _) in files {
                    if projected(self.total, previous) <= self.limit {
                        break;
                    }
                    // Recheck the file and every ancestor before removal; never
                    // traverse a symlink or remove an unrelated file by age.
                    reject_symlink_ancestors(&path)?;
                    let meta = match std::fs::symlink_metadata(&path) {
                        Ok(meta) => meta,
                        // A member eviction may already have invalidated this marker.
                        Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(error),
                    };
                    if !meta.file_type().is_file() {
                        continue;
                    }
                    let invalidated = invalidate_capture_completion(&path)?;
                    self.total = self.total.saturating_sub(invalidated);
                    if invalidated > 0 && record.completion {
                        previous = regular_file_len(&record.path)?;
                    }
                    std::fs::remove_file(&path)?;
                    self.total = self.total.saturating_sub(meta.len());
                    if path == record.path.as_ref() {
                        previous = 0;
                    }
                }
            }
        }
        if projected(self.total, previous) > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::QuotaExceeded,
                "diagnostic disk budget unavailable",
            ));
        }
        Ok(Some(projected(self.total, previous)))
    }
}

fn diagnostic_root(record: &Record) -> Option<PathBuf> {
    if record.append && record.path.file_name()? == "proxy.log" {
        return record.path.parent().map(Path::to_path_buf);
    }
    if record.capture.is_some() {
        return record
            .path
            .ancestors()
            .find(|path| {
                path.file_name().is_some_and(|name| name == "traffic")
                    && record.path.strip_prefix(path).is_ok_and(|relative| {
                        let parts: Vec<_> = relative.components().collect();
                        matches!(parts.len(), 3 | 4)
                            && traffic_request_name(Path::new(parts[1].as_os_str()))
                            && (parts.len() == 3 || parts[2].as_os_str() == "events")
                    })
            })
            .and_then(Path::parent)
            .map(Path::to_path_buf);
    }
    let parent = record.path.parent()?;
    (parent.file_name()? == "errors" && diagnostic_error_name(&record.path))
        .then(|| parent.parent().map(Path::to_path_buf))
        .flatten()
}

fn diagnostic_error_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.ends_with(".json")
                && name.split_once('-').is_some_and(|(timestamp, suffix)| {
                    !timestamp.is_empty()
                        && timestamp.bytes().all(|b| b.is_ascii_digit())
                        && !suffix.is_empty()
                })
        })
}

fn traffic_artifact_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.split_once('-').is_some_and(|(ordinal, _)| {
                ordinal.len() >= 6 && ordinal.bytes().all(|b| b.is_ascii_digit())
            }) && matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("json" | "txt" | "sse" | "bin")
            )
        })
}

fn traffic_request_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.split_once('-').is_some_and(|(ordinal, suffix)| {
                ordinal.len() >= 6
                    && ordinal.bytes().all(|b| b.is_ascii_digit())
                    && suffix.contains('-')
            })
        })
}

fn capture_completion_name(path: &Path) -> bool {
    traffic_artifact_name(path)
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.split_once('-')
                    .is_some_and(|(_, suffix)| suffix == "capture-completion.json")
            })
}

// Evicting a completed capture's member invalidates its claim of completeness.
// Discover markers from the already-known capture layout, without a registry.
fn invalidate_capture_completion(member: &Path) -> io::Result<u64> {
    if !traffic_artifact_name(member) || capture_completion_name(member) {
        return Ok(0);
    }
    let Some(parent) = member.parent() else {
        return Ok(0);
    };
    let Some(request) = (if parent.file_name().is_some_and(|name| name == "events") {
        parent.parent()
    } else {
        Some(parent)
    }) else {
        return Ok(0);
    };
    if !traffic_request_name(request)
        || request
            .parent()
            .and_then(Path::parent)
            .is_none_or(|path| path.file_name().is_none_or(|name| name != "traffic"))
    {
        return Ok(0);
    }
    let mut freed = 0;
    for entry in std::fs::read_dir(request)? {
        let path = entry?.path();
        if !capture_completion_name(&path) {
            continue;
        }
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_file() {
            std::fs::remove_file(&path)?;
            freed += meta.len();
        }
    }
    Ok(freed)
}

fn retained_capture_records(request: &Path) -> io::Result<u64> {
    let mut files = Vec::new();
    collect_capture_files(request, 1, true, &mut files)?;
    Ok(files
        .iter()
        .filter(|(path, _, _)| !capture_completion_name(path))
        .count() as u64)
}

fn reject_symlink_ancestors(path: &Path) -> io::Result<()> {
    for ancestor in path.ancestors() {
        match std::fs::symlink_metadata(ancestor) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "diagnostic symlink destination",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn regular_file_len(path: &Path) -> io::Result<u64> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_file() => Ok(meta.len()),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "diagnostic destination is not a regular file",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error),
    }
}

type DiagnosticFile = (PathBuf, u64, std::time::SystemTime);
fn collect_diagnostics(root: &Path, files: &mut Vec<DiagnosticFile>) -> io::Result<()> {
    reject_symlink_ancestors(root)?;
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_file() {
            let name = entry.file_name();
            let name = name.to_str().unwrap_or("");
            if name == "proxy.log"
                || name.strip_prefix("proxy.").is_some_and(|suffix| {
                    !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
                })
            {
                files.push((
                    path,
                    meta.len(),
                    meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                ));
            }
        } else if meta.file_type().is_dir() && entry.file_name() == "errors" {
            collect_capture_files(&path, 0, false, files)?;
        } else if meta.file_type().is_dir() && entry.file_name() == "traffic" {
            collect_capture_files(&path, 3, true, files)?;
        }
    }
    Ok(())
}

fn collect_capture_files(
    dir: &Path,
    depth: u8,
    traffic: bool,
    files: &mut Vec<DiagnosticFile>,
) -> io::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_file()
            && (!traffic || depth <= 1)
            && if traffic {
                traffic_artifact_name(&path)
            } else {
                diagnostic_error_name(&path)
            }
        {
            files.push((
                path,
                meta.len(),
                meta.modified().unwrap_or(std::time::UNIX_EPOCH),
            ));
        } else if meta.file_type().is_dir()
            && match depth {
                3 => true,                        // sanitized session directory
                2 => traffic_request_name(&path), // generated sequence-provider-request directory
                1 => entry.file_name() == "events",
                _ => false,
            }
        {
            collect_capture_files(&path, depth - 1, traffic, files)?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn write_record(record: &Record) -> io::Result<()> {
    write_record_with_retention(record, None)
}

fn write_record_with_retention(
    record: &Record,
    mut retention: Option<&mut Retention>,
) -> io::Result<()> {
    #[cfg(test)]
    if let Some(pause) = &record.pause {
        let mut state = pause.state.lock().unwrap();
        state.0 = true;
        pause.changed.notify_all();
        let (state, _) = pause
            .changed
            .wait_timeout_while(state, Duration::from_secs(10), |state| !state.1)
            .unwrap();
        return if state.1 {
            Ok(())
        } else {
            Err(io::Error::other("fixture pause timed out"))
        };
    }
    let mut failed = false;
    if record.stderr {
        failed |= io::stderr().write_all(&record.payload).is_err();
    }
    let file_result = (|| {
        reject_symlink_ancestors(&record.path)?;
        let mut marker = None;
        let mut expected_members = 0;
        if record.completion {
            let capture = record.capture.as_ref().expect("completion has capture");
            let mut value: serde_json::Value = serde_json::from_slice(&record.payload)?;
            let dropped = capture.dropped.load(Ordering::Relaxed);
            let failed = capture.failed.load(Ordering::Relaxed);
            // Capture's payload counts members before completion itself is accepted.
            expected_members = value
                .get("accepted_records")
                .and_then(|value| value.as_u64())
                .unwrap_or_else(|| capture.accepted.load(Ordering::Relaxed).saturating_sub(1));
            value["accepted_records"] = expected_members.into();
            value["dropped"] = dropped.into();
            value["failed"] = failed.into();
            // Reserve the largest final representation before pruning, including
            // counter digit growth and the extra byte in false versus true.
            value["complete"] = false.into();
            value["retained_records"] = u64::MAX.into();
            value["missing_records"] = u64::MAX.into();
            marker = Some(value);
        }
        let reserved_bytes = if let Some(marker) = marker.as_ref() {
            serde_json::to_vec(marker)?.len() as u64
        } else {
            record.payload.len() as u64
        };
        let reserved_total = retention
            .as_mut()
            .map(|retention| retention.reserve(record, reserved_bytes))
            .transpose()?
            .flatten();
        let completion = if let Some(mut marker) = marker {
            let request = record
                .path
                .parent()
                .ok_or_else(|| io::Error::other("capture completion has no parent"))?;
            let retained = retained_capture_records(request)?;
            let missing = expected_members.saturating_sub(retained);
            marker["retained_records"] = retained.into();
            marker["missing_records"] = missing.into();
            marker["complete"] =
                (missing == 0 && marker["dropped"] == 0 && marker["failed"] == 0).into();
            Some(serde_json::to_vec(&marker)?)
        } else {
            None
        };
        let payload = completion.as_deref().unwrap_or(&record.payload);
        if let Some(parent) = record.path.parent() {
            super::create_dir(parent, 0o700)?;
        }
        if record.append
            && std::fs::metadata(&record.path).is_ok_and(|m| m.len() > super::MAX_LOG_BYTES)
        {
            super::rotate_file(&record.path)?;
        }
        let mut options = OpenOptions::new();
        options
            .create(true)
            .write(true)
            .append(record.append)
            .truncate(!record.append);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&record.path)?;
        super::set_mode(&record.path, 0o600);
        file.write_all(payload)?;
        if let Some(total) = reserved_total
            && let Some(retention) = retention.as_mut()
        {
            retention.total = total
                .saturating_sub(reserved_bytes)
                .saturating_add(payload.len() as u64);
        }
        Ok::<_, io::Error>(())
    })();
    if file_result.is_err()
        && let Some(retention) = retention.as_mut()
    {
        retention.rescan = true;
    }
    if failed {
        Err(io::Error::other("diagnostic stderr failed"))
    } else {
        file_result
    }
}

// Fixture-only admission barrier: holds the native writer OUTSIDE its queue
// while a small multi-artifact fixture emits. Production admission stays lossy.
#[cfg(test)]
struct FixturePause {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}
#[cfg(test)]
pub(crate) struct FixtureWriterPause(Arc<FixturePause>);
#[cfg(test)]
impl Drop for FixtureWriterPause {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().1 = true;
        self.0.changed.notify_all();
    }
}
#[cfg(test)]
pub(crate) fn pause_for_fixture() -> FixtureWriterPause {
    assert!(flush_diagnostics(Duration::from_secs(2)));
    let pause = Arc::new(FixturePause {
        state: Mutex::new((false, false)),
        changed: Condvar::new(),
    });
    let guard = FixtureWriterPause(pause.clone());
    let mut record = Record::log(PathBuf::new(), Vec::new(), false);
    record.pause = Some(pause.clone());
    assert!(global().enqueue(record));
    let (state, _) = pause
        .changed
        .wait_timeout_while(
            pause.state.lock().unwrap(),
            Duration::from_secs(2),
            |state| !state.0,
        )
        .unwrap();
    assert!(state.0, "fixture writer did not reach its barrier");
    drop(state);
    guard
}

/// Serialize without ever allocating a full oversized diagnostic copy.
pub(crate) fn bounded_json(value: &impl serde::Serialize, pretty: bool) -> Option<Vec<u8>> {
    struct Limited(Vec<u8>);
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.0.len().saturating_add(bytes.len()) > MAX_RECORD_BYTES - 4096 {
                return Err(io::Error::other("diagnostic record too large"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited(Vec::new());
    let result = if pretty {
        serde_json::to_writer_pretty(&mut output, value)
    } else {
        serde_json::to_writer(&mut output, value)
    };
    result.ok().map(|()| output.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small disk budget keeps retention fixtures cheap.
    fn retained_writer(limit: u64) -> Writer {
        let mut retention = Retention::new(limit);
        Writer::with_sink(move |record| write_record_with_retention(record, Some(&mut retention)))
    }

    #[test]
    fn retention_prunes_oldest_diagnostics_across_log_errors_and_traffic() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir(root.join("errors")).unwrap();
        std::fs::create_dir_all(root.join("traffic/s/000001-codex-r/events")).unwrap();
        let oldest = root.join("proxy.1");
        std::fs::write(&oldest, [b'x'; 30]).unwrap();
        std::fs::File::open(&oldest)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        let error = root.join("errors/2-r.json");
        let event = root.join("traffic/s/000001-codex-r/events/000001-event.json");
        let binary = root.join("traffic/s/000001-codex-r/000002-032-upstream-response-body.bin");
        std::fs::write(&error, [b'e'; 20]).unwrap();
        std::fs::write(&event, [b't'; 20]).unwrap();
        std::fs::write(&binary, [b'b'; 20]).unwrap();
        let foreign = root.join("proxy.notes");
        std::fs::write(&foreign, [b'f'; 100]).unwrap();
        let writer = retained_writer(100);
        assert!(writer.enqueue(Record::log(root.join("proxy.log"), vec![b'l'; 20], false)));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert!(
            !oldest.exists(),
            "oldest managed artifact must release shared disk budget"
        );
        assert!(error.exists());
        assert!(event.exists());
        assert!(binary.exists());
        assert_eq!(std::fs::read(foreign).unwrap().len(), 100);
        assert_eq!(writer.snapshot().written, 1);
    }

    #[test]
    fn retention_drops_oversized_record_without_deleting_existing_diagnostics() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("proxy.log");
        std::fs::write(&path, b"existing").unwrap();
        let writer = retained_writer(16);
        assert!(writer.enqueue(Record::log(path.clone(), vec![b'x'; 17], false)));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert_eq!(std::fs::read(path).unwrap(), b"existing");
        assert_eq!(writer.snapshot().dropped, 1);
        assert_eq!(writer.snapshot().written, 0);
    }

    #[cfg(unix)]
    #[test]
    fn retention_never_follows_diagnostic_symlinks() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("auth.json");
        std::fs::write(&secret, b"secret").unwrap();
        symlink(&secret, temp.path().join("proxy.1")).unwrap();
        symlink(outside.path(), temp.path().join("errors")).unwrap();
        let writer = retained_writer(8);
        assert!(writer.enqueue(Record::log(
            temp.path().join("proxy.log"),
            b"ok".to_vec(),
            false
        )));
        assert!(writer.flush(Duration::from_secs(1)));
        assert!(writer.enqueue(Record::file(
            temp.path().join("errors/1-r.json"),
            b"overwrite".to_vec()
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert_eq!(std::fs::read(secret).unwrap(), b"secret");
        assert!(!outside.path().join("1-r.json").exists());
        assert!(
            std::fs::symlink_metadata(temp.path().join("proxy.1"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(writer.snapshot().failed, 1);
    }

    #[test]
    fn retention_uses_one_budget_across_managed_roots() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let first_path = first.path().join("proxy.log");
        let second_path = second.path().join("proxy.log");
        let writer = retained_writer(40);
        assert!(writer.enqueue(Record::log(first_path.clone(), vec![b'a'; 30], false)));
        assert!(writer.flush(Duration::from_secs(1)));
        std::fs::File::open(&first_path)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        assert!(writer.enqueue(Record::log(second_path.clone(), vec![b'b'; 30], false)));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert!(
            !first_path.exists(),
            "a second state root must not get a second budget"
        );
        assert_eq!(std::fs::read(second_path).unwrap().len(), 30);
    }

    #[test]
    fn retention_preserves_foreign_files_inside_traffic_directories() {
        let temp = tempfile::tempdir().unwrap();
        let foreign = temp.path().join("traffic/s/notes/000001-draft.json");
        let managed = temp
            .path()
            .join("traffic/s/000001-codex-r/000001-request.json");
        std::fs::create_dir_all(foreign.parent().unwrap()).unwrap();
        std::fs::create_dir_all(managed.parent().unwrap()).unwrap();
        std::fs::write(&foreign, [b'f'; 30]).unwrap();
        std::fs::File::open(&foreign)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        std::fs::write(&managed, [b't'; 30]).unwrap();
        let writer = retained_writer(20);
        assert!(writer.enqueue(Record::log(
            temp.path().join("proxy.log"),
            vec![b'l'; 10],
            false
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert!(
            foreign.exists(),
            "generated-looking files in foreign directories must remain"
        );
        assert!(!managed.exists());
    }

    #[test]
    fn retention_counts_capture_when_session_is_named_traffic() {
        let temp = tempfile::tempdir().unwrap();
        let request = temp.path().join("traffic/traffic/000001-codex-r");
        std::fs::create_dir_all(&request).unwrap();
        let old = request.join("000001-request.json");
        std::fs::write(&old, [b'o'; 20]).unwrap();
        let writer = retained_writer(30);
        assert!(writer.enqueue(Record::capture(
            request.join("000002-response.json"),
            vec![b'n'; 20],
            Arc::new(CaptureProgress::default()),
            false,
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert!(
            !old.exists(),
            "a session named traffic must not change the state root"
        );
        assert_eq!(writer.snapshot().written, 1);
    }

    #[test]
    fn retention_completion_reports_gaps_and_later_eviction_removes_marker() {
        for prune_at_completion in [true, false] {
            let temp = tempfile::tempdir().unwrap();
            let request = temp.path().join("traffic/s/000001-codex-r");
            let first = request.join("000001-request.json");
            let second = request.join("events/000002-event.json");
            let marker = request.join("000003-capture-completion.json");
            let progress = Arc::new(CaptureProgress::default());
            let writer = retained_writer(if prune_at_completion { 250 } else { 300 });
            for (path, bytes) in [
                (&first, if prune_at_completion { 150 } else { 20 }),
                (&second, if prune_at_completion { 50 } else { 20 }),
            ] {
                assert!(writer.enqueue(Record::capture(
                    path.clone(),
                    vec![b'x'; bytes],
                    progress.clone(),
                    false
                )));
                assert!(writer.flush(Duration::from_secs(1)));
            }
            std::fs::File::open(&first)
                .unwrap()
                .set_modified(std::time::UNIX_EPOCH)
                .unwrap();
            let completion = serde_json::json!({
                "kind":"diagnostic_capture_completion", "complete":false,
                "accepted_records":progress.accepted.load(Ordering::Relaxed), "dropped":0, "failed":0,
            });
            assert!(writer.enqueue(Record::capture(
                marker.clone(),
                serde_json::to_vec(&completion).unwrap(),
                progress,
                true
            )));
            assert!(writer.flush(Duration::from_secs(1)));
            let completion: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
            if prune_at_completion {
                assert!(
                    !first.exists(),
                    "completion admission must evict the older member at the cap"
                );
                assert!(second.exists());
                assert_eq!(
                    completion["complete"], false,
                    "an accepted but evicted member makes capture incomplete"
                );
                assert_eq!(completion["retained_records"], 1);
                assert_eq!(completion["missing_records"], 1);
                assert!(
                    std::fs::metadata(&marker).unwrap().len()
                        + std::fs::metadata(&second).unwrap().len()
                        <= 250
                );
            } else {
                assert_eq!(completion["complete"], true);
                assert!(writer.enqueue(Record::log(
                    temp.path().join("proxy.log"),
                    vec![b'l'; 200],
                    false
                )));
                assert!(writer.flush(Duration::from_secs(1)));
                assert!(
                    !first.exists(),
                    "later log admission must evict the older capture member"
                );
                assert!(
                    !marker.exists(),
                    "later member eviction must invalidate its complete marker"
                );
                assert!(second.exists());
            }
            assert!(writer.shutdown(Duration::from_secs(1)));
        }
    }

    fn record(bytes: usize) -> Record {
        Record::log("fixture.log".into(), vec![b'x'; bytes], false)
    }

    #[test]
    fn queued_logs_bind_destination_redaction_and_stderr_policy_at_emit() {
        let pause = pause_for_fixture();
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let settings = |path: &Path, verbose: bool| {
            let mut env = std::collections::HashMap::from([
                ("CCP_CONFIG_DIR".into(), path.display().to_string()),
                ("XDG_STATE_HOME".into(), path.display().to_string()),
            ]);
            if verbose {
                env.insert("CCP_LOG_VERBOSE".into(), "1".into());
            }
            crate::config::CodexRequestSettings::for_env(env)
        };
        let a = settings(first.path(), false);
        let b = settings(second.path(), true);
        let first_path = a.clone().sync_scope(super::super::log_file);
        let second_path = b.clone().sync_scope(super::super::log_file);
        let fields = || {
            serde_json::Map::from_iter([
                ("text".into(), serde_json::Value::String("x".repeat(6000))),
                ("encrypted_content".into(), "opaque-secret".into()),
            ])
        };
        let logger = super::super::create_logger("emission-policy");
        {
            let _suppression = super::super::suppress_stderr();
            a.sync_scope(|| logger.warn("first", Some(fields())));
        }
        b.sync_scope(|| logger.info("second", Some(fields())));
        {
            let queue = global().shared.queue.lock().unwrap();
            let first = queue
                .records
                .iter()
                .find(|record| record.path.as_ref() == first_path)
                .unwrap();
            assert!(
                !first.stderr,
                "worker must retain the earlier suppression decision"
            );
        }
        drop(pause);
        assert!(flush_diagnostics(Duration::from_secs(2)));
        let first: serde_json::Value =
            serde_json::from_slice(&std::fs::read(first_path).unwrap()).unwrap();
        let second: serde_json::Value =
            serde_json::from_slice(&std::fs::read(second_path).unwrap()).unwrap();
        assert!(
            first["fields"]["text"]
                .as_str()
                .unwrap()
                .ends_with("…[2000 more]")
        );
        assert_eq!(second["fields"]["text"].as_str().unwrap().len(), 6000);
        assert_eq!(first["fields"]["encrypted_content"], "[redacted len=13]");
    }

    #[test]
    fn stalled_sink_bounds_bytes_records_and_metadata_without_producer_wait() {
        for payload in [1, 1024 * 1024] {
            let (entered_tx, entered_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let mut first = true;
            let writer = Writer::with_sink(move |_| {
                if first {
                    first = false;
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                }
                Ok(())
            });
            assert!(writer.enqueue(record(payload)));
            entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            for _ in 0..if payload == 1 {
                MAX_QUEUE_RECORDS + 1
            } else {
                12
            } {
                writer.enqueue(record(payload));
            }
            let stats = writer.snapshot();
            assert!(stats.dropped > 0);
            assert!(stats.queued_bytes <= MAX_QUEUE_BYTES as u64);
            assert!(stats.queued_records <= MAX_QUEUE_RECORDS as u64);
            assert!(
                stats.inflight_bytes > payload as u64,
                "retained metadata was not counted"
            );
            assert!(stats.inflight_bytes <= MAX_RECORD_BYTES as u64);
            assert!(!writer.flush(Duration::from_millis(10)));
            release_tx.send(()).unwrap();
            assert!(writer.shutdown(Duration::from_secs(1)));
        }
    }

    #[test]
    fn contended_closed_and_metadata_oversized_records_drop_observably() {
        let writer = Writer::with_sink(|_| Ok(()));
        let lock = writer.shared.queue.lock().unwrap();
        assert!(!writer.enqueue(record(1)));
        drop(lock);
        assert!(!writer.enqueue(Record::log(
            PathBuf::from("x".repeat(MAX_RECORD_BYTES)),
            vec![0],
            false
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert!(!writer.enqueue(record(1)));
        assert_eq!(writer.snapshot().dropped, 3);
    }

    #[test]
    fn failed_sink_does_not_hide_loss_or_stop_following_records() {
        let writer = Writer::with_sink(|_| Err(io::Error::other("scripted disk failure")));
        // A transient try-lock loss is legal; use one record to test the sink.
        assert!(writer.enqueue(record(1)));
        assert!(writer.flush(Duration::from_secs(1)));
        assert_eq!(writer.snapshot().failed, 1);
        assert_eq!(writer.snapshot().written, 0);
        assert!(writer.enqueue(record(1)));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert_eq!(writer.snapshot().failed, 2);
    }

    #[test]
    fn completion_follows_records_and_reports_previous_write_failure() {
        let temp = tempfile::tempdir().unwrap();
        let progress = Arc::new(CaptureProgress::default());
        let writer = Writer::with_sink(write_record);
        let bad = temp.path().join("bad");
        std::fs::create_dir(&bad).unwrap();
        assert!(writer.enqueue(Record::capture(
            bad,
            b"{}".to_vec(),
            progress.clone(),
            false
        )));
        assert!(writer.flush(Duration::from_secs(1)));
        let marker = temp.path().join("marker.json");
        assert!(writer.enqueue(Record::capture(
            marker.clone(),
            b"{}".to_vec(),
            progress,
            true
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        let marker: serde_json::Value =
            serde_json::from_slice(&std::fs::read(marker).unwrap()).unwrap();
        assert_eq!(marker["complete"], false);
        assert_eq!(marker["failed"], 1);
    }

    #[test]
    fn writer_preserves_rotation_format_and_private_permissions() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("proxy.log");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(super::super::MAX_LOG_BYTES + 1)
            .unwrap();
        let writer = Writer::with_sink(write_record);
        assert!(writer.enqueue(Record::log(
            path.clone(),
            b"{\"msg\":\"hello\"}\n".to_vec(),
            false
        )));
        assert!(writer.shutdown(Duration::from_secs(1)));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"msg\":\"hello\"}\n"
        );
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(temp.path()).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
    }
}
