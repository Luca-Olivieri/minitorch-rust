use std::rc::Rc;

use crate::core::autograd::grad_fn::{Conv2dProfile, GradRule};
use crate::core::dtype::Float;
use crate::core::node::TensorNode;
use crate::core::storage::ops::conv::{conv2d_backward, conv2d_backward_profiled};
use crate::core::tensor::{AbstractTensor, GraphTensor, TensorNodeAccess};

/// Backward rule for a direct, groups-one 2D convolution.
#[derive(Debug)]
pub struct Conv2dOp {
    pub(crate) stride: (usize, usize),
    pub(crate) padding: ((usize, usize), (usize, usize)),
    pub(crate) dilation: (usize, usize),
    pub(crate) profile: Option<Conv2dProfile>,
}

impl<T: Float> GradRule<2, T> for Conv2dOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        if !operands[0].requires_grad() && !operands[1].requires_grad() {
            out.push(None);
            out.push(None);
            return;
        }

        let input_storage = &TensorNodeAccess::get_node(&operands[0]).storage;
        let weight_storage = &TensorNodeAccess::get_node(&operands[1]).storage;
        let grad_output_storage = &TensorNodeAccess::get_node(in_grad).storage;
        let backward = if let Some(profile) = &self.profile {
            let (backward, timing) = conv2d_backward_profiled(
                grad_output_storage,
                input_storage,
                weight_storage,
                self.stride,
                self.padding,
                self.dilation,
            );
            profile.borrow_mut().push(timing.sections());
            backward
        } else {
            conv2d_backward(
                grad_output_storage,
                input_storage,
                weight_storage,
                self.stride,
                self.padding,
                self.dilation,
            )
        };

        let make_tensor = |storage| GraphTensor {
            node: Rc::new(TensorNode {
                storage,
                autograd: None,
            }),
        };
        out.push(
            operands[0]
                .requires_grad()
                .then(|| make_tensor(backward.grad_input)),
        );
        out.push(
            operands[1]
                .requires_grad()
                .then(|| make_tensor(backward.grad_weight)),
        );
    }
}
