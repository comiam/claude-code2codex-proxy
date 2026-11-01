//! One owner of the WebSocket for its entire lifetime, including idle time.
use super::*;

#[derive(Clone)]
pub(super) struct ConnectionHandle {
    commands: mpsc::Sender<StartTurn>,
    closed: Arc<AtomicBool>,
}

pub(super) struct StartTurn {
    pub body_json: String,
    pub entry: Arc<PoolEntry>,
    pub reservation: Option<ContinuationReservation>,
    pub traffic: Option<Arc<TrafficCapture>>,
    pub idle_timeout_ms: u64,
    pub diagnostics: WsDiagnostics,
    pub settings: Option<Arc<crate::config::CodexRequestSettings>>,
    pub tx: EventSender,
    pub publisher: CodexWebSocketSocketIdPublisher,
}

// The driver never holds its own sender or a strong idle PoolEntry reference.
struct ClosedOnDrop(Arc<AtomicBool>);
impl Drop for ClosedOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl ConnectionHandle {
    pub(super) fn new(ws: CodexWebSocketStream) -> Self {
        let (commands, receiver) = mpsc::channel(1);
        let closed = Arc::new(AtomicBool::new(false));
        tokio::spawn(drive(ws, receiver, ClosedOnDrop(closed.clone())));
        Self { commands, closed }
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire) || self.commands.is_closed()
    }

    pub(super) fn start(&self, turn: StartTurn) {
        // Exclusive pool checkout guarantees at most one pending turn.
        if let Err(error) = self.commands.try_send(turn) {
            let turn = error.into_inner();
            let error = prewrite_closed_error(&turn);
            let _ = turn.tx.try_send(Err(error));
        }
    }
}

async fn drive(
    mut ws: CodexWebSocketStream,
    mut commands: mpsc::Receiver<StartTurn>,
    closed: ClosedOnDrop,
) {
    let mut idle_logger = create_logger("codex");
    let policy = crate::config::codex_cache_policy();
    let absolute_deadline = tokio::time::Instant::now() + Duration::from_millis(policy.max_age_ms);
    let mut idle_deadline = tokio::time::Instant::now() + Duration::from_millis(policy.idle_ttl_ms);
    loop {
        let turn = tokio::select! {
            biased;
            _ = tokio::time::sleep_until(idle_deadline.min(absolute_deadline)) => break,
            // Process already available idle frames before accepting the next create.
            frame = read_ws_frame_with_keepalive(&mut ws, Duration::from_secs(1800), WEBSOCKET_KEEPALIVE_INTERVAL) => {
                if !handle_idle_frame(&mut ws, frame, &idle_logger).await { break; }
                continue;
            }
            command = commands.recv() => match command { Some(turn) => turn, None => break },
        };
        idle_logger = turn.diagnostics.logger.clone();
        let settings = turn.settings.clone();
        let run = run_turn(&mut ws, turn);
        let reusable = match settings {
            Some(settings) => settings.scope(run).await,
            None => run.await,
        };
        if !reusable {
            break;
        }
        idle_deadline = tokio::time::Instant::now() + Duration::from_millis(policy.idle_ttl_ms);
    }
    drop(closed);
    // Any prepared turn racing an idle close gets a pre-write error, never a lost request.
    commands.close();
    while let Some(turn) = commands.recv().await {
        let error = prewrite_closed_error(&turn);
        let _ = turn.tx.try_send(Err(error));
    }
}

fn prewrite_closed_error(turn: &StartTurn) -> CodexError {
    if turn
        .reservation
        .as_ref()
        .is_some_and(|r| r.candidate().previous_response_id.is_some())
    {
        continuation_socket_missing_error()
    } else {
        CodexError {
            status: 0,
            message: "WebSocket closed before request write".into(),
            detail: None,
            retry_after: None,
            usage_limit: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        }
    }
}

async fn handle_idle_frame(
    ws: &mut CodexWebSocketStream,
    frame: WebSocketRead,
    logger: &Logger,
) -> bool {
    let (kind, error) = match frame {
        WebSocketRead::Frame(Some(Ok(Message::Ping(data)))) => {
            if matches!(
                tokio::time::timeout(
                    WEBSOCKET_KEEPALIVE_SEND_TIMEOUT,
                    ws.send(Message::Pong(data))
                )
                .await,
                Ok(Ok(()))
            ) {
                return true;
            }
            ("ping", "Pong write failed or timed out".to_owned())
        }
        WebSocketRead::Frame(Some(Ok(Message::Pong(_)))) => return true,
        WebSocketRead::Frame(Some(Ok(Message::Text(text)))) => {
            let payload = serde_json::from_str::<serde_json::Value>(&text).ok();
            if payload.as_ref().is_some_and(handle_connection_metadata) {
                return true;
            }
            let kind = payload
                .as_ref()
                .and_then(|p| p.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("malformed");
            logger.warn(
                "websocket_idle_error",
                Some(
                    serde_json::json!({"frame":"text","event":diagnostic_text(kind)})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            );
            return false;
        }
        WebSocketRead::Frame(Some(Ok(Message::Close(close)))) => {
            logger.info("websocket_idle_close", Some(serde_json::json!({"frame":"close","closeCode":close.as_ref().map(|c|u16::from(c.code)),"closeReason":close.as_ref().map(|c|diagnostic_text(&c.reason))}).as_object().unwrap().clone()));
            return false;
        }
        WebSocketRead::Frame(Some(Ok(Message::Binary(_)))) => {
            ("binary", "Unexpected binary frame".to_owned())
        }
        WebSocketRead::Frame(Some(Err(e))) => ("read", e.to_string()),
        WebSocketRead::KeepaliveError(e) => ("keepalive", e),
        _ => ("eof", "Idle socket ended".to_owned()),
    };
    logger.warn(
        "websocket_idle_error",
        Some(
            serde_json::json!({"frame":kind,"error":diagnostic_text(&error)})
                .as_object()
                .unwrap()
                .clone(),
        ),
    );
    false
}

async fn run_turn(ws: &mut CodexWebSocketStream, mut turn: StartTurn) -> bool {
    let sent = tokio::select! {
        biased;
        _ = turn.tx.closed() => return false,
        result = write_request(ws, turn.body_json) => result,
    };
    if let Err(error) = sent {
        turn.diagnostics.failure("request_write", &error.message);
        let _ = turn.tx.send_bounded(Err(error)).await;
        return false;
    }
    turn.diagnostics
        .logger
        .info("websocket_request_sent", Some(turn.diagnostics.fields()));
    let (reusable, terminal) = stream_ws_events(
        ws,
        turn.idle_timeout_ms,
        turn.traffic.as_deref(),
        &turn.tx,
        Some(&mut turn.diagnostics),
    )
    .await;
    let Some(terminal) = terminal else {
        return false;
    };
    // Reserve before publication. A full terminal queue must not advertise an idle socket.
    let permit = match tokio::time::timeout(
        super::super::STREAM_DELIVERY_STALL_TIMEOUT,
        turn.tx.reserve(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        _ => {
            turn.publisher.abort_delivery();
            return false;
        }
    };
    let inserted = reusable
        && turn
            .reservation
            .as_ref()
            .is_some_and(|r| pool_insert_for_turn(r, turn.entry.clone()));
    turn.publisher
        .publish(inserted.then_some(turn.entry.socket_id));
    permit.send(terminal);
    // A receiver dropped before ID publication may have observed no socket ID.
    // Its shared abort flag closes that publication race; exact-turn cleanup
    // still preserves a socket already checked out/republished by a newer turn.
    if turn.publisher.delivery_aborted.load(Ordering::SeqCst)
        && inserted
        && let Some(reservation) = turn.reservation.as_ref()
    {
        invalidate_codex_websocket_pool_socket(reservation, Some(turn.entry.socket_id));
    }
    // No await between insertion, socket-ID publication and terminal delivery.
    reusable && inserted
}
