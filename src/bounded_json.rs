use serde::Serialize;

/// Count JSON bytes without retaining JSON. None means budget overflow, while
/// Err remains a genuine serializer failure. Stop immediately on overflow.
pub(crate) fn bounded_json_size<T: Serialize + ?Sized>(
    value: &T,
    budget: usize,
) -> Result<Option<usize>, serde_json::Error> {
    struct Counter {
        bytes: usize,
        budget: usize,
        overflow: bool,
    }
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.budget.saturating_sub(self.bytes) {
                self.overflow = true;
                return Err(std::io::Error::other("JSON size budget exceeded"));
            }
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter {
        bytes: 0,
        budget,
        overflow: false,
    };
    match serde_json::to_writer(&mut counter, value) {
        Ok(()) => Ok(Some(counter.bytes)),
        Err(_) if counter.overflow => Ok(None),
        Err(error) => Err(error),
    }
}
