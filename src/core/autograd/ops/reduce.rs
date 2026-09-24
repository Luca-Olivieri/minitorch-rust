use std::rc::Rc;

use crate::core::{
    GraphTensor,
    autograd::grad_fn::*,
    dtype::{Float, Numeric},
    node::TensorNode,
    storage::TensorStorage,
    tensor::{AbstractTensor, TensorNodeAccess},
};

// There is a single Sum op that reduces an arbitrary subset of dimensions; summing
// over all dimensions or a single dimension are just special cases of it. `dims` are
// indices into the *input* (pre-reduction) tensor, sorted ascending and de-duplicated.

#[derive(Debug)]
pub struct SumOp {
    pub dims: Vec<usize>,
    pub keepdim: bool,
}

impl<T: Numeric> GradRule<1, T> for SumOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            // Gradient of a sum is the upstream gradient replicated over every
            // reduced dimension (via stride-0 expand views). With keepdim the
            // reduced axes are already present (size 1) in `in_grad`, so only
            // expanding is needed; otherwise they are re-inserted first.
            let mut g = in_grad.copy_s();
            let shape = operands[0].shape();
            for &d in &self.dims {
                if self.keepdim {
                    g = g.expand(d, shape[d]);
                } else {
                    g = g.unsqueeze(d).expand(d, shape[d]);
                }
            }
            g
        }));
    }
}

#[derive(Debug)]
pub struct MaxOp {
    pub dims: Vec<usize>,
    pub keepdim: bool,
}

impl<T: Numeric> GradRule<1, T> for MaxOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        let input = &operands[0];
        let shape = input.shape();

        // Broadcast the per-slice maximum back to the input shape.
        let mut max_val = input.max(&self.dims, false);
        for &d in &self.dims {
            max_val = max_val.unsqueeze(d).expand(d, shape[d]);
        }

        // mask of the elements that reach the max (a >= max(value)); no element
        // can exceed it. The mask is `bool`; the gradient math runs in `T`, so
        // it is re-interpreted as 1/0 via `as_numeric::<T>()`.
        let mask = input.gte(&max_val).as_numeric::<T>();

        // broadcast the upstream gradient back to the input shape; with keepdim
        // the reduced axes are already present (size 1) in `in_grad`
        let mut grad_b = in_grad.copy_s();
        for &d in &self.dims {
            if self.keepdim {
                grad_b = grad_b.expand(d, shape[d]);
            } else {
                grad_b = grad_b.unsqueeze(d).expand(d, shape[d]);
            }
        }

        out.push(input.requires_grad().then(|| &grad_b * &mask));
    }
}

#[derive(Debug)]
pub struct AvgPool2dOp {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
}

impl<T: Float> GradRule<1, T> for AvgPool2dOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            let dy_storage = &TensorNodeAccess::get_node(in_grad).storage;
            let dx = TensorStorage::avg_pool2d_backward(
                dy_storage,
                operands[0].shape(),
                self.kernel,
                self.stride,
            );

            // The produced gradient is a graph boundary: pooling's backward is a
            // scatter, and its transposed (`unpool`) operator is a different
            // transform, so there is no edge that could backpropagate through it
            // correctly. `requires_grad = false` surfaces an attempted
            // higher-order pass through pooling as a clear panic.
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
pub struct MaxPool2dOp {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub max_indices: Rc<Vec<Vec<usize>>>,
}

impl<T: Float> GradRule<1, T> for MaxPool2dOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            let dy_storage = &TensorNodeAccess::get_node(in_grad).storage;
            let dx = TensorStorage::max_pool2d_backward(
                dy_storage,
                operands[0].shape(),
                self.kernel,
                self.stride,
                self.max_indices.as_slice(),
            );

            // As with average pooling, the scatter is a graph boundary and is
            // intentionally not differentiable a second time.
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
