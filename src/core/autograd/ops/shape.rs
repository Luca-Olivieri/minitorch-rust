use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::autograd::grad_fn::*;

#[derive(Debug)]
pub struct CopyDOp {}

pub type BackwardCopyD = NBackwardOp<CopyDOp, 1>;

impl GradRule<1> for CopyDOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(operands[0].requires_grad().then(|| {
            in_grad.copy_d()
        }));

        out_grads
    }
}

#[derive(Debug)]
pub struct UnsqueezeOp {
    pub dim: usize,
}

pub type BackwardUnsqueeze = NBackwardOp<UnsqueezeOp, 1>;

impl GradRule<1> for UnsqueezeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(operands[0].requires_grad().then(|| {
            in_grad.squeeze(self.dim)
        }));

        out_grads
    }
}

#[derive(Debug)]
pub struct SqueezeOp {
    pub dim: usize,
}

pub type BackwardSqueeze = NBackwardOp<SqueezeOp, 1>;

impl GradRule<1> for SqueezeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(operands[0].requires_grad().then(|| {
            in_grad.unsqueeze(self.dim)
        }));

        out_grads
    }
}

#[derive(Debug)]
pub struct ExpandOp {
    pub dim: usize,
}

pub type BackwardExpand = NBackwardOp<ExpandOp, 1>;

impl GradRule<1> for ExpandOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(operands[0].requires_grad().then(|| {
            in_grad.sum_dim(self.dim).unsqueeze(self.dim) // TODO implement a flag to keep the dimension
        }));

        out_grads
    }
}
