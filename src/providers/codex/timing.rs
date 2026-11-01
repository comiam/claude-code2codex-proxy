//! One process-local monotonic clock shared by reader and body-poll diagnostics.
//! These timestamps never imply network acknowledgement or UI rendering.
use std::sync::OnceLock;
use std::time::Instant;

pub(super) fn monotonic_us() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_micros()
        .min(u64::MAX as u128) as u64
}
