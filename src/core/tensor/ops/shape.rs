use std::rc::Rc;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::grad_fn::NBackwardOp;
use crate::core::autograd::ops::shape::{
    BroadcastOp, ExpandOp, SqueezeOp, TransposeOp, UnsqueezeOp,
};
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
            Some(|operands: [GraphTensor<T>; 1]| {
                Box::new(NBackwardOp::<UnsqueezeOp, 1, T> {
                    operands,
                    op: UnsqueezeOp { dim },
                }) as Box<dyn GradFnTrait<T>>
            }),
            &[self],
        )
    }

    pub fn squeeze(&self, dim: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::squeeze(ops[0], dim),
            Some(|operands: [GraphTensor<T>; 1]| {
                Box::new(NBackwardOp::<SqueezeOp, 1, T> {
                    operands,
                    op: SqueezeOp { dim },
                }) as Box<dyn GradFnTrait<T>>
            }),
            &[self],
        )
    }

    pub fn transpose(&self, dim_a: usize, dim_b: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::transpose(ops[0], dim_a, dim_b),
            Some(|operands: [GraphTensor<T>; 1]| {
                Box::new(NBackwardOp::<TransposeOp, 1, T> {
                    operands,
                    op: TransposeOp { dim_a, dim_b },
                }) as Box<dyn GradFnTrait<T>>
            }),
            &[self],
        )
    }

    pub fn expand(&self, dim: usize, times: usize) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::expand(ops[0], dim, times),
            Some(|operands: [GraphTensor<T>; 1]| {
                Box::new(NBackwardOp::<ExpandOp, 1, T> {
                    operands,
                    op: ExpandOp { dim },
                }) as Box<dyn GradFnTrait<T>>
            }),
            &[self],
        )
    }

    pub fn broadcast(&self, b: &GraphTensor<T>) -> GraphTensor<T> {
        Self::broadcast_to_shape(self, b.shape())
    }

    pub fn broadcast_to_shape(&self, target_shape: &[usize]) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::broadcast_to_shape(ops[0], target_shape),
            Some(|operands: [GraphTensor<T>; 1]| {
                let old_shape = operands[0].shape().clone();
                Box::new(NBackwardOp::<BroadcastOp, 1, T> {
                    operands,
                    op: BroadcastOp { old_shape },
                }) as Box<dyn GradFnTrait<T>>
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

        let requires_grad = tensors.iter().any(|t| t.requires_grad());

        let out_node = TensorNode {
            storage: out_store,
            requires_grad,
            grad_fn: None,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}