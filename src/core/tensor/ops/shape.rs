use std::rc::Rc;

use crate::core::GraphTensor;
use crate::core::node::TensorNode;
use crate::core::tensor::AbstractTensor;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::shape::{
    BackwardBroadcast, BackwardCopyD, BackwardExpand, BackwardSqueeze, BackwardTranspose,
    BackwardUnsqueeze, BroadcastOp, CopyDOp, ExpandOp, SqueezeOp, TransposeOp, UnsqueezeOp,
};
use crate::core::storage::TensorStorage;
use crate::core::tensor::ops::math::apply_tensor_op;

impl GraphTensor {
    pub fn copy_d(&self) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::copy_d(ops[0]),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardCopyD {
                    operands: operands,
                    op: CopyDOp {},
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn unsqueeze(&self, dim: usize) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::unsqueeze(ops[0], dim),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardUnsqueeze {
                    operands: operands,
                    op: UnsqueezeOp { dim },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn squeeze(&self, dim: usize) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::squeeze(ops[0], dim),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardSqueeze {
                    operands: operands,
                    op: SqueezeOp { dim },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn transpose(&self) -> GraphTensor {
        if self.shape().len() != 2 {
            panic!(
                "transpose() requires a 2D tensor, got shape {:?}.",
                self.shape()
            );
        }

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::transpose(ops[0]),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardTranspose {
                    operands,
                    op: TransposeOp {},
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn expand(&self, dim: usize, times: usize) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::expand(ops[0], dim, times),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardExpand {
                    operands: operands,
                    op: ExpandOp { dim },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn broadcast(&self, b: &GraphTensor) -> GraphTensor {
        Self::broadcast_to_shape(&self, &b.shape())
    }

    pub fn broadcast_to_shape(&self, target_shape: &Vec<usize>) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::broadcast_to_shape(ops[0], target_shape),
            Some(|operands: [GraphTensor; 1]| {
                let old_shape = operands[0].shape().clone();
                Box::new(BackwardBroadcast {
                    operands,
                    op: BroadcastOp { old_shape },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    /// Stack a list of tensors along a new dimension 0.
    ///
    /// All tensors must have the same shape. The output has shape
    /// `[N, original_shape...]` where `N` is the number of inputs.
    pub fn stack(tensors: &[GraphTensor]) -> GraphTensor {
        if tensors.is_empty() {
            panic!("Cannot stack an empty list of tensors.");
        }

        let storages: Vec<&TensorStorage> = tensors.iter().map(|t| &t.node.storage).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_forward() {
        let a = GraphTensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let b = a.broadcast_to_shape(&vec![2, 3]);

        assert_eq!(b.shape(), &vec![2, 3]);
        assert_eq!(*b.at(&vec![0, 0]), 1.0);
        assert_eq!(*b.at(&vec![1, 1]), 2.0);
        assert_eq!(*b.at(&vec![1, 2]), 3.0);
    }

    #[test]
    fn broadcast_sums_grad_over_expanded_axes() {
        let a = GraphTensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let b = a.broadcast_to_shape(&vec![2, 3]);

        let grads = b.backward(true);
        let da = grads.get(&a.to_key()).unwrap();

        assert_eq!(da.shape(), &vec![3]);
        assert_eq!(*da.at(&vec![0]), 2.0);
        assert_eq!(*da.at(&vec![1]), 2.0);
        assert_eq!(*da.at(&vec![2]), 2.0);
    }

    #[test]
    #[should_panic]
    fn broadcast_incompatible_shape_panics() {
        let a = GraphTensor::from_vec(vec![1.0, 2.0, 3.0], false);
        a.broadcast_to_shape(&vec![2, 2]);
    }
}
