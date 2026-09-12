use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::math::{
    AddOp, BackwardAdd, BackwardDiv, BackwardExp, BackwardLn, BackwardMaximum, BackwardMul,
    BackwardNeg, BackwardPow, BackwardSub, DivOp, ExpOp, LnOp, MaximumOp, MulOp, NegOp, PowOp,
    SubOp,
};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::extract_requires_grad;
use crate::core::tensor::GraphTensor;

impl GraphTensor {
    impl_tensor_unary_op!(ln, TensorStorage::ln, BackwardLn, LnOp);
    impl_tensor_unary_op!(exp, TensorStorage::exp, BackwardExp, ExpOp);
    impl_tensor_binary_op!(pow, TensorStorage::pow, BackwardPow, PowOp);
    impl_tensor_binary_op!(maximum, TensorStorage::maximum, BackwardMaximum, MaximumOp);

    pub fn norm(&self) -> f64 {
        (self * self).sum(&[], false).item().sqrt()
    }
    pub fn dist(a: &GraphTensor, b: &GraphTensor) -> f64 {
        (a - b).norm()
    }

    pub fn sub_scaled(&self, other: &GraphTensor, scale: f64) -> GraphTensor {
        // Computes self - scale * other in a single fused pass (no intermediate).
        apply_tensor_op(
            |ops: &[&TensorStorage; 2]| TensorStorage::sub_scaled(ops[0], ops[1], scale),
            None::<fn([GraphTensor; 2]) -> Box<dyn GradFnTrait>>,
            &[self, other],
        )
    }
}

impl_tensor_binary_ops! {
    Add, add, TensorStorage::add,  BackwardAdd, AddOp;
    Sub, sub, TensorStorage::sub,  BackwardSub, SubOp;
    Mul, mul, TensorStorage::mul,  BackwardMul, MulOp;
    Div, div, TensorStorage::div,  BackwardDiv, DivOp;
}

impl_tensor_unary_ops! {
    Neg, neg, TensorStorage::neg, BackwardNeg, NegOp;
}

pub fn apply_tensor_op<F, G, const N: usize>(
    op: F,
    grad_fn: Option<G>,
    operands: &[&GraphTensor; N],
) -> GraphTensor
where
    F: Fn(&[&TensorStorage; N]) -> TensorStorage,
    G: FnOnce([GraphTensor; N]) -> Box<dyn GradFnTrait>,
{
    // NumPy-style right-aligned broadcasting: every operand is expanded to a
    // common shape (via stride-0 views) before the storage op runs.
    let target_shape = broadcast_shape(&(*operands).map(|o| &o.node.storage.shape));

    let needs_broadcast = operands
        .iter()
        .any(|o| o.node.storage.shape != target_shape);

    // The broadcast views for operands whose shape differs from the common shape.
    // Declared here so the references in `storages` below outlive the `if` block.
    let mut owned: Vec<TensorStorage> = Vec::with_capacity(N);
    let storages: [&TensorStorage; N] = if needs_broadcast {
        for o in operands {
            owned.push(o.node.storage.broadcast_to_shape(&target_shape));
        }
        std::array::from_fn(|i| &owned[i])
    } else {
        std::array::from_fn(|i| &operands[i].node.storage)
    };

    let out_store = op(&storages);

    // Only generate a grad_fn if one was provided
    let grad_fn_opt = grad_fn.map(|g| {
        let new_operands: [GraphTensor; N] = std::array::from_fn(|i| operands[i].copy_s());
        g(new_operands)
    });

    let out_node = TensorNode {
        storage: out_store,
        requires_grad: extract_requires_grad(operands),
        grad_fn: grad_fn_opt,
    };

    GraphTensor {
        node: Rc::new(out_node),
    }
}

// Compute the result shape of broadcasting all the given shapes together,
// following NumPy's right-aligned semantics (dimensions of size 1 stretch to
// the other operand's size; a missing leading dim acts as 1).
fn broadcast_shape(shapes: &[&Vec<usize>]) -> Vec<usize> {
    let ndim = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut out = vec![1usize; ndim];

    for s in shapes {
        let pad = ndim - s.len();
        for (i, &dim) in s.iter().enumerate() {
            let d = pad + i;
            if out[d] != dim && out[d] != 1 && dim != 1 {
                panic!(
                    "Shapes {:?} cannot be broadcast together: dim {} is {} but the common shape requires {}.",
                    shapes, d, dim, out[d]
                );
            }
            out[d] = out[d].max(dim);
        }
    }

    out
}

// TODO make this scalar operations into a macro
impl Add<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn add(self, other: f64) -> GraphTensor {
        let other_t = GraphTensor::new(vec![], other, false);
        self + &other_t
    }
}

impl Sub<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn sub(self, other: f64) -> GraphTensor {
        let other_t = GraphTensor::new(vec![], other, false);
        self - &other_t
    }
}

impl Mul<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn mul(self, other: f64) -> GraphTensor {
        let other_t = GraphTensor::new(vec![], other, false);
        self * &other_t
    }
}

impl Div<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn div(self, other: f64) -> GraphTensor {
        let other_t = GraphTensor::new(vec![], other, false);
        self / &other_t
    }
}
