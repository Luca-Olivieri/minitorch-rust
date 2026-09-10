use crate::core::autograd::grad_fn::*;
use crate::core::tensor::{AbstractTensor, GraphTensor};

// TODO there should be no SumDim, there shoulb just one Sum operation, and it should allow to sum over multiple dimension or all dimensions.

#[derive(Debug)]
pub struct SumOp {}

pub type BackwardSum = NBackwardOp<SumOp, 1>;

impl GradRule<1> for SumOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            let mut g = in_grad.copy_s();
            for (i, &size) in operands[0].shape().iter().enumerate() {
                g = g.unsqueeze(i).expand(i, size);
            }
            g
        }));
    }
}

#[derive(Debug)]
pub struct MaxDimOp {
    pub dim: usize,
}

pub type BackwardMaxDim = NBackwardOp<MaxDimOp, 1>;

impl GradRule<1> for MaxDimOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        let input = &operands[0];
        let size = input.shape()[self.dim];

        // mask of the elements that reach the max (a >= max(value)); no element can exceed it
        let max_val = input.max_dim(self.dim);
        let max_b = max_val.unsqueeze(self.dim).expand(self.dim, size);
        let mask = input.gte(&max_b);

        // broadcast the upstream gradient back to the input shape
        let grad_b = in_grad.unsqueeze(self.dim).expand(self.dim, size);

        out.push(input.requires_grad().then(|| &grad_b * &mask));
    }
}

#[derive(Debug)]
pub struct SumDimOp {
    pub dim: usize,
    pub original_times: usize,
}

pub type BackwardSumDim = NBackwardOp<SumDimOp, 1>;

impl GradRule<1> for SumDimOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            in_grad
                .unsqueeze(self.dim)
                .expand(self.dim, self.original_times)
        }));
    }
}
