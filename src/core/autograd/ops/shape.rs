use std::rc::Rc;

use crate::core::{
    GraphTensor,
    autograd::grad_fn::*,
    dtype::Numeric,
    node::TensorNode,
    storage::TensorStorage,
    tensor::{AbstractTensor, TensorNodeAccess},
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

#[derive(Debug)]
pub struct PadOp {
    pub pads: Vec<(usize, usize)>,
}

impl<T: Numeric> GradRule<1, T> for PadOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = pad(x, pads): only the interior (the original x) contributes to y,
        // so dy/dx is the upstream gradient with the pad region cut away.
        let shape = operands[0].shape();
        let ranges: Vec<(usize, usize)> = shape
            .iter()
            .enumerate()
            .map(|(d, &len)| (self.pads[d].0, len))
            .collect();
        out.push(operands[0].requires_grad().then(|| in_grad.slice(&ranges)));
    }
}

#[derive(Debug)]
pub struct SliceOp {
    pub ranges: Vec<(usize, usize)>,
}

impl<T: Numeric> GradRule<1, T> for SliceOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = slice(x, ranges): x's elements outside the window are ignored, so
        // their gradient is 0; the window's gradient is `in_grad` dropped back
        // into the full shape at the slice offsets (a zero-pad). The pad "after"
        // the window is `len - (start + slice_len)`, i.e. the elements past the
        // slice end.
        let x = &operands[0];
        let pads: Vec<(usize, usize)> = x
            .shape()
            .iter()
            .enumerate()
            .map(|(d, &len)| (self.ranges[d].0, len - self.ranges[d].0 - self.ranges[d].1))
            .collect();
        out.push(x.requires_grad().then(|| in_grad.pad(&pads)));
    }
}

#[derive(Debug)]
pub struct StridedSliceOp {
    pub ranges: Vec<(usize, usize, usize)>,
}

impl<T: Numeric> GradRule<1, T> for StridedSliceOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            let dy_storage = &TensorNodeAccess::get_node(in_grad).storage;
            let dx = TensorStorage::slice_strided_backward(
                dy_storage,
                operands[0].shape(),
                &self.ranges,
            );

            // The scatter is a graph boundary; the first-order gradient is
            // materialized directly rather than through a higher-order unslice.
            let out_node = TensorNode {
                storage: dx,
                requires_grad: false,
                no_grad: false,
                grad_fn: None,
            };

            GraphTensor {
                node: Rc::new(out_node),
            }
        }));
    }
}

#[derive(Debug)]
pub struct ReshapeOp {
    pub new_shape: Vec<usize>,
}

impl<T: Numeric> GradRule<1, T> for ReshapeOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = reshape(x, new_shape): the inverse of a reshape is a reshape back
        // to the operand's own shape (element counts match by construction).
        let shape = operands[0].shape().clone();
        out.push(operands[0].requires_grad().then(|| in_grad.reshape(&shape)));
    }
}
