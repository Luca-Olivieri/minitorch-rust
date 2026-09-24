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
//! - [`crate::core::dtype::DangerousCastFrom`] — partial/panicking, available
//!   as `.cast_dangerous::<U>()` only with the `allow_dangerous_casts` feature.

use crate::core::autograd::grad_fn::{BackwardOpKind, maybe_edge};
#[cfg(feature = "allow_dangerous_casts")]
use crate::core::dtype::DangerousCastFrom;
#[cfg(feature = "allow_lossy_casts")]
use crate::core::dtype::LossyCastFrom;
use crate::core::dtype::{CastBackward, CastFrom, Dtype, Numeric};
use crate::core::node::{AutogradMeta, TensorNode};
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
    pub fn cast<U: CastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        self.cast_with_mode(false)
    }

    pub fn cast_with_mode<U: CastFrom<T> + CastBackward<T>>(
        &self,
        no_grad: bool,
    ) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::cast_from);
        let autograd = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp, no_grad).map(AutogradMeta::Node)
        } else {
            None
        };
        let node = TensorNode { storage, autograd };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise lossy conversion to dtype `U`.
    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy<U: LossyCastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        self.cast_lossy_with_mode(false)
    }

    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy_with_mode<U: LossyCastFrom<T> + CastBackward<T>>(
        &self,
        no_grad: bool,
    ) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::lossy_cast_from);
        let autograd = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp, no_grad).map(AutogradMeta::Node)
        } else {
            None
        };
        let node = TensorNode { storage, autograd };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise dangerous conversion to dtype `U`.
    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous<U: DangerousCastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        self.cast_dangerous_with_mode(false)
    }

    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous_with_mode<U: DangerousCastFrom<T> + CastBackward<T>>(
        &self,
        no_grad: bool,
    ) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::dangerous_cast_from);
        let autograd = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp, no_grad).map(AutogradMeta::Node)
        } else {
            None
        };
        let node = TensorNode { storage, autograd };
        GraphTensor {
            node: Rc::new(node),
        }
    }
}

impl GraphTensor<bool> {
    /// Reinterpret this boolean tensor elementwise as numeric 1/0 in dtype `T`.
    #[doc(hidden)]
    pub fn as_numeric<T: Numeric>(&self) -> GraphTensor<T> {
        let storage = cast_storage(
            &self.node.storage,
            |b: bool| {
                if b { T::ONE } else { T::ZERO }
            },
        );
        let node = TensorNode {
            storage,
            autograd: None,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }
}

impl<T: Dtype> FreeTensor<T> {
    /// Element-wise exact conversion to dtype `U`, allocating a new tensor.
    pub fn cast<U: CastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::cast_from);
        let node = TensorNode {
            storage,
            autograd: self.node.requires_grad().then_some(AutogradMeta::Leaf),
        };
        FreeTensor {
            node: Box::new(node),
        }
    }

    /// Element-wise lossy conversion to dtype `U`.
    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy<U: LossyCastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::lossy_cast_from);
        let node = TensorNode {
            storage,
            autograd: self.node.requires_grad().then_some(AutogradMeta::Leaf),
        };
        FreeTensor {
            node: Box::new(node),
        }
    }

    /// Element-wise dangerous conversion to dtype `U`.
    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous<U: DangerousCastFrom<T>>(&self) -> FreeTensor<U> {
        let storage = cast_storage(&self.node.storage, U::dangerous_cast_from);
        let node = TensorNode {
            storage,
            autograd: self.node.requires_grad().then_some(AutogradMeta::Leaf),
        };
        FreeTensor {
            node: Box::new(node),
        }
    }
}
