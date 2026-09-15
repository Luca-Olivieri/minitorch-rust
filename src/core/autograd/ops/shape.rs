use crate::core::{
    GraphTensor,
    autograd::grad_fn::*,
    dtype::Numeric,
    tensor::AbstractTensor,
};

#[derive(Debug)]
pub struct CopyDOp {}

impl<T: Numeric> GradRule<1, T> for CopyDOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| in_grad.copy_d()));
    }
}

#[derive(Debug)]
pub struct UnsqueezeOp {
    pub dim: usize,
}

impl<T: Numeric> GradRule<1, T> for UnsqueezeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
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

impl<T: Numeric> GradRule<1, T> for SqueezeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| in_grad.unsqueeze(self.dim)),
        );
    }
}

#[derive(Debug)]
pub struct TransposeOp {
    pub dim_a: usize,
    pub dim_b: usize,
}

impl<T: Numeric> GradRule<1, T> for TransposeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // transpose is its own inverse: transposing the same dims twice is a no-op
        out.push(
            operands[0]
                .requires_grad()
                .then(|| in_grad.transpose(self.dim_a, self.dim_b)),
        );
    }
}

#[derive(Debug)]
pub struct ExpandOp {
    pub dim: usize,
}

impl<T: Numeric> GradRule<1, T> for ExpandOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            // keepdim: summing an expanded (stride-0) axis yields a size-1 axis
            // in the same position, which is exactly the operand's original shape
            in_grad.sum(&[self.dim], true)
        }));
    }
}

#[derive(Debug)]
pub struct BroadcastOp {
    pub old_shape: Vec<usize>,
}

impl<T: Numeric> GradRule<1, T> for BroadcastOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
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