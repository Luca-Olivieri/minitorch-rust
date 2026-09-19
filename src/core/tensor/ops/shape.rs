use std::rc::Rc;

use crate::core::autograd::grad_fn::BackwardOpKind;
use crate::core::dtype::{Dtype, Numeric};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl<T: Numeric> GraphTensor<T> {
    pub fn unsqueeze(&self, dim: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::unsqueeze(ops[0], dim),
            Some(BackwardOpKind::UnsqueezeOp { dim }),
            &[self],
        )
    }

    pub fn squeeze(&self, dim: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::squeeze(ops[0], dim),
            Some(BackwardOpKind::SqueezeOp { dim }),
            &[self],
        )
    }

    pub fn transpose(&self, dim_a: usize, dim_b: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::transpose(ops[0], dim_a, dim_b),
            Some(BackwardOpKind::TransposeOp { dim_a, dim_b }),
            &[self],
        )
    }

    pub fn expand(&self, dim: usize, times: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::expand(ops[0], dim, times),
            Some(BackwardOpKind::ExpandOp { dim }),
            &[self],
        )
    }

    pub fn broadcast(&self, b: &GraphTensor<T>) -> GraphTensor<T> {
        Self::broadcast_to_shape(self, b.shape())
    }

    pub fn broadcast_to_shape(&self, target_shape: &[usize]) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::broadcast_to_shape(ops[0], target_shape),
            Some(BackwardOpKind::BroadcastOp {
                old_shape: self.shape().clone(),
            }),
            &[self],
        )
    }

    /// Zero-pad every dimension by `pads[d] = (before, after)` elements.
    ///
    /// Padded regions read as zero; the gradient discards them via an inverse
    /// slice back to the original shape. `pads` must provide exactly one pair
    /// per dimension.
    pub fn pad(&self, pads: &[(usize, usize)]) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::pad(ops[0], pads),
            Some(BackwardOpKind::PadOp {
                pads: pads.to_vec(),
            }),
            &[self],
        )
    }

    /// Extract the window `[start, start + len)` along every dimension,
    /// materializing it into a fresh contiguous tensor.
    ///
    /// The gradient places `in_grad` back into the full shape at the slice
    /// offsets, leaving the excluded elements with zero gradient.
    pub fn slice(&self, ranges: &[(usize, usize)]) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::slice(ops[0], ranges),
            Some(BackwardOpKind::SliceOp {
                ranges: ranges.to_vec(),
            }),
            &[self],
        )
    }

    /// Reinterpret the logical elements under a new shape, materializing into a
    /// fresh contiguous tensor. `new_shape` must preserve the element count.
    pub fn reshape(&self, new_shape: &[usize]) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::reshape(ops[0], new_shape),
            Some(BackwardOpKind::ReshapeOp {
                new_shape: new_shape.to_vec(),
            }),
            &[self],
        )
    }
}

impl<T: Dtype> GraphTensor<T> {
    /// Stack a list of tensors along a new dimension 0.
    ///
    /// All tensors must have the same shape. The output has shape
    /// `[N, original_shape...]` where `N` is the number of inputs.
    /// Non-differentiable: stacking is a pure data layout op.
    pub fn stack(tensors: &[GraphTensor<T>]) -> GraphTensor<T> {
        if tensors.is_empty() {
            panic!("Cannot stack an empty list of tensors.");
        }

        let storages: Vec<&TensorStorage<T>> = tensors.iter().map(|t| &t.node.storage).collect();
        let out_store = TensorStorage::stack(&storages);

        // Non-differentiable: a graph boundary (no edge, no grad propagation).
        let out_node = TensorNode {
            storage: out_store,
            requires_grad: false,
            grad_fn: None,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}
