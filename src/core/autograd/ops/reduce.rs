use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::autograd::grad_fn::*;

// TODO there should be no SumDim, there shoulb just one Sum operation, and it should allow to sum over multiple dimension or all dimensions.

#[derive(Debug)]
pub struct SumDimOp {
    pub dim: usize,
    pub original_times: usize
}

pub type BackwardSumDim = NBackwardOp<SumDimOp, 1>;

impl GradRule<1> for SumDimOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(operands[0].requires_grad().then(|| {
            in_grad.unsqueeze(self.dim).expand(self.dim, self.original_times)
        }));

        out_grads
    }
}
