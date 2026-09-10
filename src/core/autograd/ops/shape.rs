use crate::core::autograd::grad_fn::*;
use crate::core::tensor::{AbstractTensor, GraphTensor};

#[derive(Debug)]
pub struct CopyDOp {}

pub type BackwardCopyD = NBackwardOp<CopyDOp, 1>;

impl GradRule<1> for CopyDOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
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
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| in_grad.squeeze(self.dim)),
        );
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
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| in_grad.unsqueeze(self.dim)),
        );
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
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
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
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            in_grad.sum_dim(self.dim).unsqueeze(self.dim) // TODO implement a flag to keep the dimension
        }));
    }
}

#[derive(Debug)]
pub struct BroadcastOp {
    pub old_shape: Vec<usize>,
}

pub type BackwardBroadcast = NBackwardOp<BroadcastOp, 1>;

impl GradRule<1> for BroadcastOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        // The forward op produced `in_grad.shape()` from an operand of shape
        // `self.old_shape`; the gradient is the sum of `in_grad` over every
        // axis that broadcasting expanded.
        out.push(
            operands[0]
                .requires_grad()
                .then(|| reduce_grad_to_shape(in_grad, &self.old_shape, retain_graph)),
        );
    }
}
