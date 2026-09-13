pub mod core;
pub mod data;

// Internal unit tests (exercise crate-private machinery, e.g. storage kernels).
// Integration tests that drive the public API live in the `tests/` directory.
#[cfg(test)]
mod unit_tests;
