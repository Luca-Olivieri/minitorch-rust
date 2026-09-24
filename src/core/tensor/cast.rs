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

use crate::core::autograd::grad_fn::{BackwardOpKind, maybe_edge};
#[cfg(feature = "allow_dangerous_casts")]
use crate::core::dtype::DangerousCastFrom;
#[cfg(feature = "allow_lossy_casts")]
use crate::core::dtype::LossyCastFrom;
use crate::core::dtype::{CastBackward, CastFrom, Dtype, Numeric};
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
    /// holds (exact, infallible conversions). When the conversion is also
    /// differentiable — both dtypes are floats and the reverse cast is legal in
    /// this build, witnessed by `U: CastBackward<T>` — the result carries a graph
    /// edge whose backward applies the reverse conversion to the upstream
    /// gradient. Otherwise the cast is a graph boundary (`requires_grad` does not
    /// propagate).
    pub fn cast<U: CastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::cast_from);
        let grad_fn = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp)
        } else {
            None
        };
        let node = TensorNode {
            storage,
            requires_grad: grad_fn.is_some() && self.node.requires_grad,
            no_grad: self.node.no_grad,
            grad_fn,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise lossy (but total) conversion to dtype `U`, allocating a new
    /// tensor. Compiled only with the `allow_lossy_casts` feature.
    ///
    /// Differentiable exactly like [`GraphTensor::cast`]: with the feature on,
    /// the narrowing `f64 -> f32` edge exists and its backward is the exact
    /// `f32 -> f64` widening.
    #[cfg(feature = "allow_lossy_casts")]
    pub fn cast_lossy<U: LossyCastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::lossy_cast_from);
        let grad_fn = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp)
        } else {
            None
        };
        let node = TensorNode {
            storage,
            requires_grad: grad_fn.is_some() && self.node.requires_grad,
            no_grad: self.node.no_grad,
            grad_fn,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }

    /// Element-wise dangerous (panicking on out-of-domain inputs) conversion to
    /// dtype `U`, allocating a new tensor. Compiled only with the
    /// `allow_dangerous_casts` feature.
    ///
    /// Never differentiable: no float↔float pair is dangerous.
    #[cfg(feature = "allow_dangerous_casts")]
    pub fn cast_dangerous<U: DangerousCastFrom<T> + CastBackward<T>>(&self) -> GraphTensor<U> {
        let storage = cast_storage(&self.node.storage, U::dangerous_cast_from);
        let grad_fn = if U::GRAD_EDGE {
            maybe_edge(&[self], BackwardOpKind::CastOp)
        } else {
            None
        };
        let node = TensorNode {
            storage,
            requires_grad: grad_fn.is_some() && self.node.requires_grad,
            no_grad: self.node.no_grad,
            grad_fn,
        };
        GraphTensor {
            node: Rc::new(node),
        }
    }
}

impl GraphTensor<bool> {
    /// Reinterpret this boolean tensor elementwise as numeric 1/0 in dtype `T`
    /// (`T::ONE`/`T::ZERO`).
    ///
    /// Unlike `cast::<T>()`, this does **not** route through the cast-trait
    /// table: `bool` has no numeric value of its own (it is `Dtype`-only), so
    /// the reinterpretation is available for *every* `T: Numeric` with no edge.
    /// It is used by the mask-based backward rules (`maximum`/`max`), which are
    /// dispatched only when a float backward run actually materializes them.
    /// Internal: public only for cross-module visibility (tensor ops and
    /// autograd rules).
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
            requires_grad: false,
            no_grad: self.node.no_grad,
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
            no_grad: self.node.no_grad,
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
            no_grad: self.node.no_grad,
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
            no_grad: self.node.no_grad,
            grad_fn: None,
        };
        FreeTensor {
            node: Box::new(node),
        }
    }
}
