use std::rc::Rc;

use crate::core::GraphTensor;
use crate::core::node::TensorNode;
use crate::core::tensor::AbstractTensor;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::shape::{
    BackwardCopyD, BackwardExpand, BackwardSqueeze, BackwardTranspose, BackwardUnsqueeze, CopyDOp,
    ExpandOp, SqueezeOp, TransposeOp, UnsqueezeOp,
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
