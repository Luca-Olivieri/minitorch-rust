use crate::core::GraphTensor;

use crate::core::autograd::ops::shape::{
    BackwardCopyD, BackwardExpand, BackwardSqueeze, BackwardUnsqueeze, CopyDOp, ExpandOp, SqueezeOp, UnsqueezeOp
};
use crate::core::storage::TensorStorage;
use crate::core::tensor::ops::math::apply_tensor_op;
use crate::core::autograd::grad_fn::GradFnTrait;

impl GraphTensor {

    pub fn copy_d(&self) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::copy_d(ops[0]),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardCopyD{operands: operands, op: CopyDOp{}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }

    pub fn unsqueeze(
        &self,
        dim: usize
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::unsqueeze(ops[0], dim),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardUnsqueeze{operands: operands, op: UnsqueezeOp{dim}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }

    pub fn squeeze(
        &self,
        dim: usize
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::squeeze(ops[0], dim),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardSqueeze{operands: operands, op: SqueezeOp{dim}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }

    pub fn expand(
        &self,
        dim: usize,
        times: usize
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::expand(ops[0], dim, times),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardExpand{operands: operands, op: ExpandOp{dim}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }
}
