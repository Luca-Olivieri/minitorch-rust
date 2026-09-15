//! Dtype casts for tensors: allocate a new tensor whose elements are converted
//! element-wise.
//!
//! Mirroring the element-level three-tier conversion model, legality is decided
//! at compile time by which trait impl exists — mixing dtypes is never a silent
//! coercion:
//!
//! - [`crate::core::dtype::CastFrom`] — exact/infallible, always available as
//!   `.cast::<U>()`.
//! - [`crate::core::dtype::LossyCastFrom`] — total/rounding, available as
//!   `.cast_lossy::<U>()` only with the `allow_lossy_casts` feature.
//! - [`crate::core::dtype::DangerousCastFrom`] — partial/panicking, available as
//!   `.cast_dangerous::<U>()` only with the `allow_dangerous_casts` feature.

use crate::core::dtype::{CastFrom, Dtype};
#[cfg(feature = "allow_dangerous_casts")]
use crate::core::dtype::DangerousCastFrom;
#[cfg(feature = "allow_lossy_casts")]
use crate::core::dtype::LossyCastFrom;
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::{FreeTensor, GraphTensor};
use std::rc::Rc;

/// Build a fresh, contiguous storage of dtype `U` by converting every logical
/// element of `storage` (which may be a strided view) with `convert`.
pub(crate) fn cast_storage<T: Dtype, U: Dtype>(
    storage: &TensorStorage<T>,
    convert: impl Fn(T) -> U,
) -> TensorStorage<U> {
    let mut buffer: Vec<U> = Vec::with_capacity(storage.numel);
    for flat in storage.strided_indices() {
        buffer.push(convert(storage.buffer[flat]));
    }
    TensorStorage::from_buffer(storage.shape.clone(), buffer)
}

impl<T: Dtype> GraphTensor<T> {
    /// Element-wise exact conversion to dtype `U`, allocating a new tensor.
    ///
    /// Compile-time legality: this method only exists where `U: CastFrom<T>`
    /// holds (exact, infallible conversions). The result keeps the source's
    /// `requires_grad` flag but carries no graph edge — casts are currently a
    /// non-differentiable data transform.
    pub fn cast<U: CastFrom<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise lossy (but total) conversion to dtype `U`, allocating a new
    /// tensor. Compiled only with the `allow_lossy_casts` feature.
    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy<U: LossyCastFrom<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::lossy_cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise dangerous (panicking on out-of-domain inputs) conversion to
    /// dtype `U`, allocating a new tensor. Compiled only with the
    /// `allow_dangerous_casts` feature.
    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous<U: DangerousCastFrom<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::dangerous_cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }
}

impl<T: Dtype> FreeTensor<T> {
    /// Element-wise exact conversion to dtype `U`, allocating a new tensor.
    ///
    /// Compile-time legality: this method only exists where `U: CastFrom<T>`
    /// holds (exact, infallible conversions).
    pub fn cast<U: CastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        FreeTensor {
            node: Box::new(node),
        }
    }

    /// Element-wise lossy (but total) conversion to dtype `U`, allocating a new
    /// tensor. Compiled only with the `allow_lossy_casts` feature.
    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy<U: LossyCastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::lossy_cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        FreeTensor {
            node: Box::new(node),
        }
    }

    /// Element-wise dangerous (panicking on out-of-domain inputs) conversion to
    /// dtype `U`, allocating a new tensor. Compiled only with the
    /// `allow_dangerous_casts` feature.
    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous<U: DangerousCastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::dangerous_cast_from);
        let node = TensorNode {
            storage,
            requires_grad: self.node.requires_grad,
            grad_fn: None,
        };
        FreeTensor {
            node: Box::new(node),
        }
    }
}