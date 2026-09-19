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