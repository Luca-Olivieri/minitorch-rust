use crate::core::GraphTensor;

use crate::core::autograd::ops::reduce::{BackwardSumDim, SumDimOp};
use crate::core::storage::TensorStorage;
use crate::core::tensor::{AbstractTensor, Tensor};
use crate::core::tensor::ops::math::apply_tensor_op;
use crate::core::autograd::grad_fn::GradFnTrait;

impl GraphTensor {

    pub fn sum_dim(
        &self,
        dim: usize,
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::sum_dim(ops[0], dim),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDim{operands: operands, op: SumDimOp { dim, original_times: self.shape()[dim]}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }

    pub fn sum(
        &self,
    ) -> GraphTensor {

        let mut out: GraphTensor = self.copy_d();

        for _ in 0..self.shape().len() {
            out = out.sum_dim(0);
        }

        out
    }
}
