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
        out: &mut Vec<Option<GraphTensor>>
    ) {
        out.push(operands[0].requires_grad().then(|| {
            let new_shape = in_grad.shape();
            let old_shape = &self.old_shape;

            // Prepend 1s to old_shape so it has the same ndim as new_shape (NumPy convention)
            let mut aligned_old = vec![1usize; new_shape.len() - old_shape.len()];
            aligned_old.extend_from_slice(old_shape);

            let mut g = in_grad.copy_s();
            // Sum over every dimension that was broadcast (old=1, new>1), right to left
            // to avoid index shifting issues
            for d in (0..new_shape.len()).rev() {
                if aligned_old[d] == 1 && new_shape[d] > 1 {
                    g = g.sum_dim(d);
                }
            }
            g
        }));
    }
}
