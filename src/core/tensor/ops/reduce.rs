use std::rc::Rc;

use crate::core::autograd::grad_fn::{BackwardOpKind, maybe_edge};
use crate::core::dtype::{Float, Numeric};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::storage::ops::reduce::{OneHotLabel, resolve_dims};
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl<T: Numeric> GraphTensor<T> {
    /// Sum over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent, e.g.
    /// `x.sum(&[0, 2])`. An empty slice aggregates over all dimensions, like
    /// PyTorch's `torch.sum(x, dim=None)`.
    ///
    /// With `keepdim` the reduced dimensions are retained as size 1 (so
    /// `x.sum(&[0, 2], true)` on `[2, 2, 3]` yields `[1, 2, 1]`), which keeps
    /// the result broadcastable against the input.
    pub fn sum(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| {
                let reduced = TensorStorage::sum(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(BackwardOpKind::SumOp {
                dims: dims.clone(),
                keepdim,
            }),
            &[self],
        )
    }

    /// Max over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent. An
    /// empty slice aggregates over all dimensions. Gradient flows to every
    /// element that attains the per-slice maximum (like `torch.amax`). See
    /// [`Self::sum`] for the `keepdim` semantics.
    pub fn max(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| {
                let reduced = TensorStorage::max(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(BackwardOpKind::MaxOp {
                dims: dims.clone(),
                keepdim,
            }),
            &[self],
        )
    }

    /// Argmax along `dim`, yielding the per-slice flat index as a `f64` tensor
    /// (labels are consumed by the dtype they are cast to). Non-differentiable.
    pub fn argmax(&self, dim: usize, keepdim: bool) -> GraphTensor<f64> {
        let reduced = TensorStorage::argmax(&self.node.storage, dim);
        let out_store = if keepdim {
            TensorStorage::unsqueeze(&reduced, dim)
        } else {
            reduced
        };

        let out_node = TensorNode {
            storage: out_store,
            requires_grad: false,
            grad_fn: None,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }

    /// Direct [m,k] x [k,n] -> [m,n] gemm. 1D operands are treated as [1,k]
    /// / [k,1] rows/columns (NumPy semantics) and the corresponding axis of the
    /// result is squeezed away.
    pub fn matmul(a: &GraphTensor<T>, b: &GraphTensor<T>) -> GraphTensor<T> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        if !((a_ndim == 1 || a_ndim == 2) && (b_ndim == 1 || b_ndim == 2)) {
            panic!(
                "matmul requires 1D or 2D tensors, got {}D and {}D",
                a_ndim, b_ndim
            );
        }

        // Convert 1D inputs to 2D views: a [K] -> [1,K], b [K] -> [K,1]
        let a2 = if a_ndim == 1 {
            a.unsqueeze(0)
        } else {
            a.copy_s()
        };
        let b2 = if b_ndim == 1 {
            b.unsqueeze(1)
        } else {
            b.copy_s()
        };

        let a2_shape = a2.shape(); // [m, k]
        let b2_shape = b2.shape(); // [k, n]

        let k = a2_shape[1];
        let kb = b2_shape[0];

        if k != kb {
            panic!("matmul inner dimensions must match ({} != {})", k, kb);
        }

        // Direct [m,k] x [k,n] -> [m,n] kernel.
        let out_store = TensorStorage::matmul(&a2.node.storage, &b2.node.storage);

        // Differentiable only for float dtypes (edge-gated); `requires_grad`
        // follows the edge.
        let grad_fn = maybe_edge(&[a, b], BackwardOpKind::MatmulOp);
        let requires_grad = grad_fn.is_some() && (a.requires_grad() || b.requires_grad());

        let out_node = TensorNode {
            storage: out_store,
            requires_grad,
            grad_fn,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}

impl<T: Float> GraphTensor<T> {
    /// Mean over every dimension in `dims` at once. An empty slice aggregates
    /// over all dimensions. See [`crate::core::tensor::GraphTensor::sum`] for
    /// the `keepdim` semantics.
    pub fn mean(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        let count: usize = dims.iter().map(|&d| self.shape()[d]).product();
        let count_t = T::from_f64(count as f64);
        &self.sum(&dims, keepdim) / count_t
    }
}

impl<T: Numeric + OneHotLabel> GraphTensor<T> {
    /// One-hot encode the labels in `self` into a fresh tensor of shape
    /// `self.shape ++ [num_classes]`. Labels may be floats (validated integral,
    /// non-negative) or integers; the output is `f64` (`1.0`/`0.0`), the
    /// representation the softmax-loss path consumes. Non-differentiable.
    pub fn one_hot(&self, num_classes: usize) -> GraphTensor<f64> {
        let out_storage = TensorStorage::one_hot(&self.node.storage, num_classes);

        let out_node = TensorNode {
            storage: out_storage,
            requires_grad: false,
            grad_fn: None,
        };

        GraphTensor::<f64> {
            node: Rc::new(out_node),
        }
    }
}
