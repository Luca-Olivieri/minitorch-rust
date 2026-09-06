use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::autograd::grad_fn::*;

#[derive(Debug)]
pub struct CopyDOp {}

pub type BackwardCopyD = NBackwardOp<CopyDOp, 1>;

impl GradRule<1> for CopyDOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| in_grad.copy_d()));
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
        in_grad: &GraphTensor,
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| in_grad.squeeze(self.dim)));
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
        in_grad: &GraphTensor,
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| in_grad.unsqueeze(self.dim)));
    }
}

#[derive(Debug)]
pub struct TransposeOp {}

pub type BackwardTranspose = NBackwardOp<TransposeOp, 1>;

impl GradRule<1> for TransposeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| in_grad.transpose()));
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
        in_grad: &GraphTensor,
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| {
            in_grad.sum_dim(self.dim).unsqueeze(self.dim) // TODO implement a flag to keep the dimension
        }));
    }
}
