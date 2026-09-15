use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::grad_fn::NBackwardOp;
use crate::core::autograd::ops::math::{
    AddOp, DivOp, ExpOp, LnOp, MaximumOp, MulOp, NegOp, PowOp, SqrtOp, SubOp,
};
use crate::core::dtype::{Dtype, Float, Numeric};
use crate::core::storage::TensorStorage;
use crate::core::tensor::extract_requires_grad;
use crate::core::tensor::GraphTensor;

// Math ops live in two dtype homes, chosen by what their backward math needs:
//
// - Numeric home: add/mul (signed-agnostic, gradient is a broadcast/passthrough)
//   and maximum (mask-based gradient needs only comparisons).
// - Float home: sub/div (backward uses neg/div), neg, and the transcendental
//   pow/ln/exp/sqrt (float-only kernels and derivatives).

impl<T: Numeric> GraphTensor<T> {
    impl_tensor_binary_method!(maximum, TensorStorage::maximum, MaximumOp);

    /// out[i] = self[i] - scale * other[i], fused into a single storage pass.
    /// Non-differentiable (a linear combination used by the optimizers).
    pub fn sub_scaled(&self, other: &GraphTensor<T>, scale: T) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::sub_scaled(ops[0], ops[1], scale),
            None::<fn([GraphTensor<T>; 2]) -> Box<dyn GradFnTrait<T>>>,
            &[self, other],
        )
    }
}

impl_tensor_binary_ops! {
    Numeric, Add, add, TensorStorage::add,  AddOp;
    Numeric, Mul, mul, TensorStorage::mul,  MulOp;
    Float,   Sub, sub, TensorStorage::sub,  SubOp;
    Float,   Div, div, TensorStorage::div,  DivOp;
}

impl_tensor_unary_ops! {
    Float, Neg, neg, TensorStorage::neg, NegOp;
}

impl_tensor_scalar_ops! {
    Numeric, Add, add;
    Numeric, Mul, mul;
    Float,   Sub, sub;
    Float,   Div, div;
}

impl<T: Float> GraphTensor<T> {
    impl_tensor_unary_method!(ln, TensorStorage::ln, LnOp);
    impl_tensor_unary_method!(exp, TensorStorage::exp, ExpOp);
    impl_tensor_unary_method!(sqrt, TensorStorage::sqrt, SqrtOp);
    impl_tensor_binary_method!(pow, TensorStorage::pow, PowOp);

    pub fn norm(&self) -> GraphTensor<T> {
        (self * self).sum(&[], false).sqrt()
    }

    pub fn dist(a: &GraphTensor<T>, b: &GraphTensor<T>) -> GraphTensor<T> {
        (a - b).norm()
    }
}

pub(crate) fn apply_tensor_op<T: Dtype, F, G, const N: usize>(
    op: F,
    grad_fn: Option<G>,
    operands: &[&GraphTensor<T>; N],
) -> GraphTensor<T>
where
    F: Fn(&[&TensorStorage<T>; N]) -> TensorStorage<T>,
    G: FnOnce([GraphTensor<T>; N]) -> Box<dyn GradFnTrait<T>>,
{
    // NumPy-style right-aligned broadcasting: every operand is expanded to a
    // common shape (via stride-0 views) before the storage op runs.
    let target_shape = broadcast_shape(&(*operands).map(|o| o.node.storage.shape.as_slice()));

    let needs_broadcast = operands
        .iter()
        .any(|o| o.node.storage.shape != target_shape);

    // The broadcast views for operands whose shape differs from the common shape.
    // Declared here so the references in `storages` below outlive the `if` block.
    let mut owned: Vec<TensorStorage<T>> = Vec::with_capacity(N);
    let storages: [&TensorStorage<T>; N] = if needs_broadcast {
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
        let new_operands: [GraphTensor<T>; N] = std::array::from_fn(|i| operands[i].copy_s());
        g(new_operands)
    });

    let out_node = crate::core::node::TensorNode {
        storage: out_store,
        requires_grad: extract_requires_grad(operands),
        grad_fn: grad_fn_opt,
    };

    GraphTensor {
        node: std::rc::Rc::new(out_node),
    }
}

// Compute the result shape of broadcasting all the given shapes together,
// following NumPy's right-aligned semantics (dimensions of size 1 stretch to
// the other operand's size; a missing leading dim acts as 1).
fn broadcast_shape(shapes: &[&[usize]]) -> Vec<usize> {
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