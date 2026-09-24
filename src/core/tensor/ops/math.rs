use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::core::autograd::grad_fn::{BackwardOpKind, maybe_edge};
use crate::core::dtype::{Dtype, Float, Numeric, Signed};
use crate::core::node::AutogradMeta;
use crate::core::storage::TensorStorage;
use crate::core::tensor::GraphTensor;

// Math ops live in dtype homes, chosen by what their *forward* kernels need.
// Forward edges only record a `BackwardSource`, so no op needs a `GradRule`
// bound to be constructed; the rule is materialized at backward time (T: Float).
// Edges are dtype-gated (`maybe_edge` / `T::DIFFERENTIABLE`): a `Numeric` op on
// an integer is forward-only and records no edge.
//
// - Numeric home: add/mul/sub/div (kernels are signed-agnostic, `sub`/`div`
//   wrap/truncate on integers like PyTorch's uint arithmetic), maximum
//   (mask-based gradient) and the modul/sub_scaled boundaries.
// - Signed home: neg (unsigned negation is undefined — std has no `Neg` for
//   `u8`…`u64`) and abs (`Signed::abs`; differentiable for floats, boundary for
//   signed ints).
// - Float home: pow/ln/exp/sqrt and norm/dist (float-only kernels and
//   derivatives).

impl<T: Numeric> GraphTensor<T> {
    impl_tensor_binary_method!(maximum, TensorStorage::maximum, MaximumOp);

    /// `out[i] = self[i] % other[i]` (truncated remainder). Non-differentiable:
    /// the truncated division remainder is a sawtooth whose derivative is not
    /// well-defined, so it is a graph boundary like `maximum`.
    pub fn modul(&self, other: &GraphTensor<T>) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::modul(&[ops[0], ops[1]]),
            None,
            false,
            &[self, other],
        )
    }

    /// out[i] = self[i] - scale * other[i], fused into a single storage pass.
    /// Non-differentiable (a linear combination used by the optimizers).
    pub fn sub_scaled(&self, other: &GraphTensor<T>, scale: T) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::sub_scaled(ops[0], ops[1], scale),
            None,
            false,
            &[self, other],
        )
    }
}

impl<T: Signed> GraphTensor<T> {
    /// Elementwise `|self|`. Differentiable for floats (gradient = sign, 0 at
    /// zero); a graph boundary for signed integers (`T::DIFFERENTIABLE` is
    /// false, so no edge is recorded and `requires_grad` does not propagate).
    pub fn abs(&self) -> GraphTensor<T> {
        self.abs_with_mode(false)
    }

    pub fn abs_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::abs(&[ops[0]]),
            Some(BackwardOpKind::AbsOp),
            no_grad,
            &[self],
        )
    }
}

impl_tensor_binary_ops! {
    Numeric, Add, add, TensorStorage::add,  AddOp;
    Numeric, Mul, mul, TensorStorage::mul,  MulOp;
    Numeric, Sub, sub, TensorStorage::sub,  SubOp;
    Numeric, Div, div, TensorStorage::div,  DivOp;
}

impl_tensor_unary_ops! {
    Signed, Neg, neg, TensorStorage::neg, NegOp;
}

impl_tensor_scalar_ops! {
    Numeric, Add, add;
    Numeric, Mul, mul;
    Numeric, Sub, sub;
    Numeric, Div, div;
}

impl<T: Numeric> GraphTensor<T> {
    pub fn maximum_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::maximum(&[ops[0], ops[1]]),
            Some(BackwardOpKind::MaximumOp),
            no_grad,
            &[self, other],
        )
    }

    pub fn add_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::add(&[ops[0], ops[1]]),
            Some(BackwardOpKind::AddOp),
            no_grad,
            &[self, other],
        )
    }

    pub fn mul_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::mul(&[ops[0], ops[1]]),
            Some(BackwardOpKind::MulOp),
            no_grad,
            &[self, other],
        )
    }

    pub fn sub_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::sub(&[ops[0], ops[1]]),
            Some(BackwardOpKind::SubOp),
            no_grad,
            &[self, other],
        )
    }

    pub fn div_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::div(&[ops[0], ops[1]]),
            Some(BackwardOpKind::DivOp),
            no_grad,
            &[self, other],
        )
    }
}

impl<T: Signed> GraphTensor<T> {
    pub fn neg_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::neg(&[ops[0]]),
            Some(BackwardOpKind::NegOp),
            no_grad,
            &[self],
        )
    }
}

impl<T: Float> GraphTensor<T> {
    impl_tensor_unary_method!(ln, TensorStorage::ln, LnOp);
    impl_tensor_unary_method!(exp, TensorStorage::exp, ExpOp);
    impl_tensor_unary_method!(sqrt, TensorStorage::sqrt, SqrtOp);
    impl_tensor_binary_method!(pow, TensorStorage::pow, PowOp);

    pub fn ln_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::ln(&[ops[0]]),
            Some(BackwardOpKind::LnOp),
            no_grad,
            &[self],
        )
    }

    pub fn exp_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::exp(&[ops[0]]),
            Some(BackwardOpKind::ExpOp),
            no_grad,
            &[self],
        )
    }

    pub fn sqrt_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::sqrt(&[ops[0]]),
            Some(BackwardOpKind::SqrtOp),
            no_grad,
            &[self],
        )
    }

    pub fn pow_with_mode(&self, other: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::pow(&[ops[0], ops[1]]),
            Some(BackwardOpKind::PowOp),
            no_grad,
            &[self, other],
        )
    }

    pub fn norm(&self) -> GraphTensor<T> {
        self.norm_with_mode(false)
    }

    pub fn norm_with_mode(&self, no_grad: bool) -> GraphTensor<T> {
        self.mul_with_mode(self, no_grad)
            .sum_with_mode(&[], false, no_grad)
            .sqrt_with_mode(no_grad)
    }

    pub fn dist(a: &GraphTensor<T>, b: &GraphTensor<T>) -> GraphTensor<T> {
        Self::dist_with_mode(a, b, false)
    }

    pub fn dist_with_mode(a: &GraphTensor<T>, b: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        a.sub_with_mode(b, no_grad).norm_with_mode(no_grad)
    }
}

pub(crate) fn apply_tensor_op<T: Dtype, F, const N: usize>(
    op: F,
    grad_op: Option<BackwardOpKind>,
    no_grad: bool,
    operands: &[&GraphTensor<T>; N],
) -> GraphTensor<T>
where
    F: Fn(&[&TensorStorage<T>; N]) -> TensorStorage<T>,
{
    let autograd = grad_op
        .and_then(|op| maybe_edge(operands, op, no_grad))
        .map(AutogradMeta::Node);

    with_broadcast_operands(operands, |storages: &[&TensorStorage<T>; N]| {
        let out_store = op(storages);
        let out_node = crate::core::node::TensorNode {
            storage: out_store,
            autograd,
        };

        GraphTensor {
            node: std::rc::Rc::new(out_node),
        }
    })
}

/// Dtype-changing elementwise op: operands all share the input dtype `T`, the
/// result is a new, disconnected `GraphTensor<U>` (no gradient edge — the
/// writers are comparison masks, i.e. step functions, so nothing is differenti
/// able here). Used by comparisons `(T, T) -> bool`.
pub(crate) fn apply_tensor_op_into<T: Dtype, U: Dtype, F, const N: usize>(
    op: F,
    operands: &[&GraphTensor<T>; N],
) -> GraphTensor<U>
where
    F: Fn(&[&TensorStorage<T>; N]) -> TensorStorage<U>,
{
    with_broadcast_operands(operands, |storages: &[&TensorStorage<T>; N]| {
        let out_store = op(storages);
        let out_node = crate::core::node::TensorNode {
            storage: out_store,
            autograd: None,
        };

        GraphTensor {
            node: std::rc::Rc::new(out_node),
        }
    })
}

/// Broadcast every operand to the common NumPy-style shape, then hand the
/// prepared storage views to `f`. The broadcast views live in a `Vec` that is
/// local to this closure, so `f` runs (and produces its result) before they are
/// dropped.
fn with_broadcast_operands<T: Dtype, const N: usize, R>(
    operands: &[&GraphTensor<T>; N],
    f: impl FnOnce(&[&TensorStorage<T>; N]) -> R,
) -> R {
    // NumPy-style right-aligned broadcasting: every operand is expanded to a
    // common shape (via stride-0 views) before the storage op runs.
    let target_shape = broadcast_shape(&(*operands).map(|o| o.node.storage.shape.as_slice()));

    let needs_broadcast = operands
        .iter()
        .any(|o| o.node.storage.shape != target_shape);

    // The broadcast views for operands whose shape differs from the common shape.
    let mut owned: Vec<TensorStorage<T>> = Vec::with_capacity(N);
    let storages: [&TensorStorage<T>; N] = if needs_broadcast {
        for o in operands {
            owned.push(o.node.storage.broadcast_to_shape(&target_shape));
        }
        std::array::from_fn(|i| &owned[i])
    } else {
        std::array::from_fn(|i| &operands[i].node.storage)
    };

    f(&storages)
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
