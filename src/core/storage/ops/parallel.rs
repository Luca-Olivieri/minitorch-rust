//! Worker-count policy for storage kernels that split across the batch axis.
//!
//! The kernels use `std::thread::scope`, so there is no pool to build and no
//! dependency to add. The count is read from `MINITORCH_THREADS` when set,
//! which serves two purposes: a single binary can be measured at several widths
//! without a rebuild, and `MINITORCH_THREADS=1` produces an exact serial
//! baseline for A/B comparison in the same build. That matters more than usual
//! here, because run-to-run control drift on the development machine is around
//! 1%, which is the same size as several candidate optimisations.
//!
//! Threads are spawned per kernel call rather than reused, so a kernel should
//! only be split when its work comfortably exceeds the spawn cost.

use std::sync::OnceLock;
use std::thread::available_parallelism;

fn configured_workers() -> usize {
    static CONFIGURED: OnceLock<usize> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("MINITORCH_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or_else(|| available_parallelism().map(|n| n.get()).unwrap_or(1))
    })
}

/// Workers to use for a kernel that can split `items` independent items.
///
/// Never more than the number of items, and never fewer than one.
pub(crate) fn worker_count(items: usize) -> usize {
    configured_workers().min(items).max(1)
}
