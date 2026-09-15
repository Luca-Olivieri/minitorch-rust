//! Internal unit tests for the library.
//!
//! These exercise crate-internal machinery (e.g. `TensorStorage` kernels) that is
//! deliberately not part of the public API, so they are compiled as part of the
//! library itself (`#[cfg(test)]`) rather than as integration tests. Integration
//! tests live in `tests/` and only use the public API.

mod dtype;
mod storage;
mod tensor_casts;
mod tensor_generics;
