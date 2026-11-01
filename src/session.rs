use crate::config::AliasProvider;
use crate::proxy_speed::ProxySpeedMode;
use crate::registry::normalize_incoming_model;
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

const SESSION_IDLE_TTL_MS: u64 = 30 * 60 * 1000;
pub const MAX_SESSIONS: usize = 10_000;

#[derive(Debug, Clone)]
pub struct SessionState {
    pub seq: u64,
    pub affinity_provider: Option<AliasProvider>,
    pub proxy_speed: Option<ProxySpeedMode>,
    pub last_seen: u64,
}

#[derive(Default)]
struct SessionStore {
    map: HashMap<String, SessionState>,
    order: VecDeque<String>,
}

static SESSIONS: LazyLock<Mutex<SessionStore>> =
    LazyLock::new(|| Mutex::new(SessionStore::default()));

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

pub fn existing_session(session_id: Option<&str>, now: u64) -> Option<SessionState> {
    let id = session_id?;
    SESSIONS
        .lock()
        .expect("session lock")
        .existing_session(id, now)
}

pub fn existing_session_now(session_id: Option<&str>) -> Option<SessionState> {
    existing_session(session_id, now_millis())
}

pub fn record_session_request(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    now: u64,
) -> Option<SessionState> {
    record_session_request_with_affinity_update(session_id, prior, provider_name, model, true, now)
}

pub(crate) fn record_session_request_with_affinity_update(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    let id = session_id?;
    let mut store = SESSIONS.lock().expect("session lock");
    Some(store.record_request(
        id,
        prior,
        provider_name,
        model,
        SessionUpdate {
            update_affinity,
            proxy_speed: None,
        },
        now,
    ))
}

struct SessionUpdate {
    update_affinity: bool,
    proxy_speed: Option<ProxySpeedMode>,
}

impl SessionStore {
    fn record_request(
        &mut self,
        id: &str,
        prior: Option<&SessionState>,
        provider_name: &str,
        model: &str,
        update: SessionUpdate,
        now: u64,
    ) -> SessionState {
        let stored = self.existing_session(id, now);
        let mut next = stored
            .or_else(|| {
                prior
                    .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS)
                    .cloned()
                    .map(|mut state| {
                        // An evicted/expired/restarted store must not recover speed
                        // from an in-flight request's stale snapshot.
                        state.proxy_speed = None;
                        state
                    })
            })
            .unwrap_or(SessionState {
                seq: 0,
                affinity_provider: None,
                proxy_speed: None,
                last_seen: now,
            });
        next.seq += 1;
        next.last_seen = now;
        if let Some(mode) = update.proxy_speed {
            next.proxy_speed = Some(mode);
        }
        if update.update_affinity
            && is_alias_routable_provider(provider_name)
            && !crate::registry::is_anthropic_alias(normalize_incoming_model(model).as_str())
        {
            next.affinity_provider = Some(match provider_name {
                "codex" => AliasProvider::Codex,
                "kimi" => AliasProvider::Kimi,
                _ => next.affinity_provider.unwrap_or(AliasProvider::Codex),
            });
        }
        if !self.map.contains_key(id) {
            self.order.push_back(id.to_string());
        }
        self.map.insert(id.to_string(), next.clone());
        while self.order.len() > MAX_SESSIONS {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
            } else {
                break;
            }
        }
        next
    }

    fn existing_session(&mut self, id: &str, now: u64) -> Option<SessionState> {
        let state = self.map.get(id).cloned()?;
        if now.saturating_sub(state.last_seen) > SESSION_IDLE_TTL_MS {
            self.map.remove(id);
            self.order.retain(|item| item != id);
            return None;
        }
        Some(state)
    }
}

/// Atomically accepts a control command and records its request. The returned
/// state is a snapshot; callers must validate command identity before calling.
pub fn record_session_request_with_speed(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    mode: ProxySpeedMode,
    now: u64,
) -> Option<SessionState> {
    let id = session_id?;
    let mut store = SESSIONS.lock().expect("session lock");
    Some(store.record_request(
        id,
        prior,
        provider_name,
        model,
        SessionUpdate {
            update_affinity: true,
            proxy_speed: Some(mode),
        },
        now,
    ))
}

fn is_alias_routable_provider(name: &str) -> bool {
    matches!(name, "codex" | "kimi")
}

#[cfg(test)]
pub fn reset_sessions_for_test() {
    let mut store = SESSIONS.lock().expect("session lock");
    store.map.clear();
    store.order.clear();
}

pub fn affinity_provider_from_session(session: &SessionState) -> Option<AliasProvider> {
    session.affinity_provider
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn record(
        store: &mut SessionStore,
        id: &str,
        prior: Option<&SessionState>,
        mode: Option<ProxySpeedMode>,
        now: u64,
    ) -> SessionState {
        store.record_request(
            id,
            prior,
            "codex",
            "gpt-5.6-sol",
            SessionUpdate {
                update_affinity: true,
                proxy_speed: mode,
            },
            now,
        )
    }

    #[test]
    fn speed_distinguishes_unset_off_and_literal_fast_modes() {
        let mut store = SessionStore::default();
        let initial = record(&mut store, "a", None, None, 1);
        assert_eq!(initial.proxy_speed, None);
        for (now, mode) in [
            (2, ProxySpeedMode::Off),
            (3, ProxySpeedMode::Fast),
            (4, ProxySpeedMode::Ultrafast),
        ] {
            let state = record(&mut store, "a", Some(&initial), Some(mode), now);
            assert_eq!(state.proxy_speed, Some(mode));
            assert_eq!(state.seq, now);
            assert_eq!(state.last_seen, now);
        }
    }

    #[test]
    fn speed_is_isolated_and_survives_ordinary_and_auxiliary_requests() {
        let mut store = SessionStore::default();
        let a = record(&mut store, "a", None, Some(ProxySpeedMode::Fast), 1);
        let b = record(&mut store, "b", None, None, 2);
        assert_eq!(b.proxy_speed, None);
        let ordinary = record(&mut store, "a", Some(&a), None, 3);
        assert_eq!(ordinary.proxy_speed, Some(ProxySpeedMode::Fast));
        let auxiliary = store.record_request(
            "a",
            Some(&a),
            "kimi",
            "kimi-for-coding",
            SessionUpdate {
                update_affinity: false,
                proxy_speed: None,
            },
            4,
        );
        assert_eq!(auxiliary.proxy_speed, Some(ProxySpeedMode::Fast));
        assert_eq!(auxiliary.affinity_provider, Some(AliasProvider::Codex));
        assert_eq!(store.existing_session("b", 4).unwrap().proxy_speed, None);
    }

    #[test]
    fn stale_ordinary_request_cannot_restore_prior_speed_and_snapshot_does_not_change() {
        let mut store = SessionStore::default();
        let initial = record(&mut store, "a", None, Some(ProxySpeedMode::Fast), 1);
        let command = record(
            &mut store,
            "a",
            Some(&initial),
            Some(ProxySpeedMode::Off),
            2,
        );
        let completed_late = record(&mut store, "a", Some(&initial), None, 3);
        assert_eq!(completed_late.proxy_speed, Some(ProxySpeedMode::Off));
        assert_eq!(completed_late.seq, command.seq + 1);
        assert_eq!(initial.proxy_speed, Some(ProxySpeedMode::Fast));
    }

    #[test]
    fn speed_expires_with_idle_ttl_and_stale_prior_does_not_resurrect_it() {
        let mut store = SessionStore::default();
        let prior = record(&mut store, "a", None, Some(ProxySpeedMode::Ultrafast), 10);
        assert_eq!(
            store
                .existing_session("a", 10 + SESSION_IDLE_TTL_MS)
                .unwrap()
                .proxy_speed,
            Some(ProxySpeedMode::Ultrafast)
        );
        assert!(
            store
                .existing_session("a", 11 + SESSION_IDLE_TTL_MS)
                .is_none()
        );
        let renewed = record(
            &mut store,
            "a",
            Some(&prior),
            None,
            11 + SESSION_IDLE_TTL_MS,
        );
        assert_eq!(renewed.proxy_speed, None);
        assert_eq!(renewed.seq, 1);
    }

    #[test]
    fn ordinary_and_control_acceptance_refresh_idle_lifetime_atomically() {
        let mut store = SessionStore::default();
        record(&mut store, "a", None, Some(ProxySpeedMode::Fast), 1);
        let ordinary = record(&mut store, "a", None, None, SESSION_IDLE_TTL_MS);
        assert_eq!(ordinary.proxy_speed, Some(ProxySpeedMode::Fast));
        assert_eq!(ordinary.last_seen, SESSION_IDLE_TTL_MS);
        let command = record(
            &mut store,
            "a",
            None,
            Some(ProxySpeedMode::Off),
            2 * SESSION_IDLE_TTL_MS,
        );
        assert_eq!(command.seq, 3);
        assert_eq!(command.last_seen, 2 * SESSION_IDLE_TTL_MS);
        assert_eq!(
            store
                .existing_session("a", 3 * SESSION_IDLE_TTL_MS)
                .unwrap()
                .proxy_speed,
            Some(ProxySpeedMode::Off)
        );
    }

    #[test]
    fn bounded_eviction_removes_speed_even_with_unexpired_prior() {
        let mut store = SessionStore::default();
        let prior = record(&mut store, "oldest", None, Some(ProxySpeedMode::Fast), 1);
        for offset in 0..MAX_SESSIONS {
            record(&mut store, &format!("other-{offset}"), None, None, 1);
        }
        assert_eq!(store.map.len(), MAX_SESSIONS);
        assert_eq!(store.order.len(), MAX_SESSIONS);
        assert!(store.existing_session("oldest", 2).is_none());
        let renewed = record(&mut store, "oldest", Some(&prior), None, 2);
        assert_eq!(renewed.proxy_speed, None);
        assert_eq!(store.map.len(), MAX_SESSIONS);
    }

    #[test]
    fn fresh_store_does_not_import_speed_from_prior_state() {
        let prior = record(
            &mut SessionStore::default(),
            "a",
            None,
            Some(ProxySpeedMode::Ultrafast),
            1,
        );
        let mut restarted = SessionStore::default();
        let renewed = record(&mut restarted, "a", Some(&prior), None, 2);
        assert_eq!(renewed.proxy_speed, None);
        assert_eq!(
            record(
                &mut restarted,
                "a",
                Some(&prior),
                Some(ProxySpeedMode::Off),
                3
            )
            .proxy_speed,
            Some(ProxySpeedMode::Off)
        );
    }

    #[test]
    fn concurrent_commands_return_snapshots_in_atomic_acceptance_order() {
        let store = Arc::new(Mutex::new(SessionStore::default()));
        let prior = record(
            &mut store.lock().unwrap(),
            "a",
            None,
            Some(ProxySpeedMode::Fast),
            1,
        );
        let barrier = Arc::new(Barrier::new(17));
        let handles: Vec<_> = (0..16)
            .map(|index| {
                let store = store.clone();
                let barrier = barrier.clone();
                let prior = prior.clone();
                std::thread::spawn(move || {
                    let mode = [
                        ProxySpeedMode::Off,
                        ProxySpeedMode::Fast,
                        ProxySpeedMode::Ultrafast,
                    ][index % 3];
                    barrier.wait();
                    let accepted =
                        record(&mut store.lock().unwrap(), "a", Some(&prior), Some(mode), 2);
                    assert_eq!(accepted.proxy_speed, Some(mode));
                    accepted
                })
            })
            .collect();
        barrier.wait();
        let mut snapshots: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        snapshots.sort_by_key(|snapshot| snapshot.seq);
        assert_eq!(
            snapshots
                .iter()
                .map(|snapshot| snapshot.seq)
                .collect::<Vec<_>>(),
            (2..=17).collect::<Vec<_>>()
        );
        let latest = store.lock().unwrap().existing_session("a", 2).unwrap();
        assert_eq!(latest.seq, 17);
        assert_eq!(latest.proxy_speed, snapshots.last().unwrap().proxy_speed);
    }

    #[test]
    fn public_speed_entry_point_records_request_and_preserves_legacy_callers() {
        let id = "session-speed-public-entry-test";
        let accepted = record_session_request_with_speed(
            Some(id),
            None,
            "codex",
            "gpt-5.6-sol",
            ProxySpeedMode::Ultrafast,
            1,
        )
        .unwrap();
        assert_eq!(accepted.seq, 1);
        assert_eq!(accepted.proxy_speed, Some(ProxySpeedMode::Ultrafast));
        assert_eq!(
            existing_session(Some(id), 1).unwrap().proxy_speed,
            Some(ProxySpeedMode::Ultrafast)
        );
        let ordinary =
            record_session_request(Some(id), Some(&accepted), "codex", "gpt-5.6-sol", 2).unwrap();
        assert_eq!(ordinary.seq, 2);
        assert_eq!(ordinary.proxy_speed, Some(ProxySpeedMode::Ultrafast));
        assert!(
            record_session_request_with_speed(
                None,
                None,
                "codex",
                "gpt-5.6-sol",
                ProxySpeedMode::Off,
                3
            )
            .is_none()
        );
    }

    #[test]
    fn concurrent_requests_increment_latest_sequence() {
        let session_id = "session-concurrent-sequence-test";
        let initial = record_session_request(Some(session_id), None, "codex", "gpt-5.6-sol", 1)
            .expect("initial session");
        let handles: Vec<_> = (0..16)
            .map(|offset| {
                let stale = initial.clone();
                std::thread::spawn(move || {
                    record_session_request(
                        Some(session_id),
                        Some(&stale),
                        "codex",
                        "gpt-5.6-sol",
                        2 + offset,
                    )
                    .expect("recorded session")
                    .seq
                })
            })
            .collect();
        let mut sequences: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("session thread"))
            .collect();
        sequences.sort_unstable();
        assert_eq!(sequences, (2..=17).collect::<Vec<_>>());
    }

    #[test]
    fn auxiliary_request_does_not_change_session_affinity() {
        let session_id = "session-affinity-auxiliary-request-test";
        let initial = record_session_request(Some(session_id), None, "codex", "gpt-5.6-sol", 1)
            .expect("initial session");
        assert_eq!(initial.affinity_provider, Some(AliasProvider::Codex));

        let after_review = record_session_request_with_affinity_update(
            Some(session_id),
            Some(&initial),
            "kimi",
            "kimi-for-coding",
            false,
            2,
        )
        .expect("updated session");
        assert_eq!(after_review.seq, 2);
        assert_eq!(after_review.affinity_provider, Some(AliasProvider::Codex));
    }
}
