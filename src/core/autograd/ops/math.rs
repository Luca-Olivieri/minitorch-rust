use crate::core::{
    GraphTensor,
    autograd::grad_fn::*,
    dtype::{Float, Numeric, Signed},
    tensor::AbstractTensor,
};

// The grad-rule for each op is dispatched from `BackwardSource::into_grad_fn`
// only when a backward run materializes it (T: Float), so the rule's bound is
// no longer tied to the op's forward home:
//
// - Numeric-homed rules (add/mul/maximum/matmul) are usable for any integer and
//   float Tensor; integer tensors carry the same forward graph structure even
//   though a backward *run* only exists for floats.
// - Signed-homed rules (sub/div/neg) negate a gradient, so they apply to floats
//   and signed integers — while their forward ops (`sub`/`div`) now live in the
//   Numeric home and are simply forward-only for unsigned integers.
// - Float-homed rules (ln/exp/sqrt/pow) use float-only kernels, matching their
//   Float-homed forward ops.

/// Build a scalar node of dtype `T` from a plain `f64` constant.
fn scalar<T: Float>(x: f64) -> GraphTensor<T> {
    GraphTensor::new(Vec::new(), T::from_f64(x), false)
}

#[derive(Debug)]
pub struct NegOp;

impl<T: Signed> GradRule<1, T> for NegOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(operands[0].requires_grad().then(|| -in_grad));
    }
}

#[derive(Debug)]
pub struct AbsOp;

impl<T: Signed> GradRule<1, T> for AbsOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = |a|:  dy/da = sign(a) = (a > 0) - (a < 0); the subgradient is 0 at
        // a == 0 (both masks are false). Comparisons yield `bool` masks, which
        // are re-interpreted as 1/0 in `T` via `as_numeric`.
        let a = &operands[0];
        let zero = GraphTensor::<T>::new(Vec::new(), T::ZERO, false);

        let pos = a.gt(&zero).as_numeric::<T>();
        let neg = a.lt(&zero).as_numeric::<T>();

        out.push(a.requires_grad().then(|| in_grad * &(&pos - &neg)));
    }
}

#[derive(Debug)]
pub struct LnOp;

impl<T: Float> GradRule<1, T> for LnOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = ln(a)
        // dy/da = 1 / a
        let a = &operands[0];

        out.push(a.requires_grad().then(|| in_grad / a));
    }
}

#[derive(Debug)]
pub struct ExpOp;

impl<T: Float> GradRule<1, T> for ExpOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = exp(a) = e^a
        // dy/da = e^a = y
        out.push(
            operands[0]
                .requires_grad()
                .then(|| in_grad * &operands[0].exp()),
        );
    }
}

#[derive(Debug)]
pub struct SqrtOp;

impl<T: Float> GradRule<1, T> for SqrtOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 1],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = sqrt(a)
        // dy/da = 1 / (2 * sqrt(a)) = 1 / (2 * y)
        let a = &operands[0];
        let two = scalar::<T>(2.0);

        out.push(a.requires_grad().then(|| in_grad / &(&a.sqrt() * &two)));
    }
}

#[derive(Debug)]
pub struct MatmulOp;

impl<T: Numeric> GradRule<2, T> for MatmulOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| grad_a(&operands[0], &operands[1], in_grad)),
        );
        out.push(
            operands[1]
                .requires_grad()
                .then(|| grad_b(&operands[0], &operands[1], in_grad)),
        );
    }
}

// Turn a 1D tensor [k] into a [1, k] "row" view.
fn into_row<T: Numeric>(t: &GraphTensor<T>) -> (GraphTensor<T>, bool) {
    if t.shape().len() == 1 {
        (t.unsqueeze(0), true)
    } else {
        (t.copy_s(), false)
    }
}

// Turn a 1D tensor [k] into a [k, 1] "column" view.
fn into_col<T: Numeric>(t: &GraphTensor<T>) -> (GraphTensor<T>, bool) {
    if t.shape().len() == 1 {
        (t.unsqueeze(1), true)
    } else {
        (t.copy_s(), false)
    }
}

// Reshape the upstream gradient to the 2D [m, n] shape the kernel expects.
fn grad_to_2d<T: Numeric>(
    a_ndim: usize,
    b_ndim: usize,
    in_grad: &GraphTensor<T>,
) -> GraphTensor<T> {
    match (a_ndim, b_ndim) {
        (1, 1) => in_grad.unsqueeze(0).unsqueeze(1),
        (1, _) => in_grad.unsqueeze(0),
        (_, 1) => in_grad.unsqueeze(1),
        (_, _) => in_grad.copy_s(),
    }
}

// dL/dA = in_grad @ B^T
fn grad_a<T: Numeric>(
    a: &GraphTensor<T>,
    b: &GraphTensor<T>,
    in_grad: &GraphTensor<T>,
) -> GraphTensor<T> {
    let (_, a_was_1d) = into_row(a);
    let (b2, _) = into_col(b);

    let g = GraphTensor::matmul(
        &grad_to_2d(a.shape().len(), b.shape().len(), in_grad),
        &b2.transpose(0, 1),
    );

    if a_was_1d { g.squeeze(0) } else { g }
}

// dL/dB = A^T @ in_grad
fn grad_b<T: Numeric>(
    a: &GraphTensor<T>,
    b: &GraphTensor<T>,
    in_grad: &GraphTensor<T>,
) -> GraphTensor<T> {
    let (a2, _) = into_row(a);
    let (_, b_was_1d) = into_col(b);

    let g = GraphTensor::matmul(
        &a2.transpose(0, 1),
        &grad_to_2d(a.shape().len(), b.shape().len(), in_grad),
    );

    if b_was_1d { g.squeeze(1) } else { g }
}

#[derive(Debug)]
pub struct AddOp;

impl<T: Numeric> GradRule<2, T> for AddOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| reduce_grad_to_shape(in_grad, operands[0].shape(), retain_graph)),
        );
        out.push(
            operands[1]
                .requires_grad()
                .then(|| reduce_grad_to_shape(in_grad, operands[1].shape(), retain_graph)),
        );
    }
}

#[derive(Debug)]
pub struct SubOp;

impl<T: Signed> GradRule<2, T> for SubOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.push(
            operands[0]
                .requires_grad()
                .then(|| reduce_grad_to_shape(in_grad, operands[0].shape(), retain_graph)),
        );
        out.push(
            operands[1]
                .requires_grad()
                .then(|| -&reduce_grad_to_shape(in_grad, operands[1].shape(), retain_graph)),
        );
    }
}

#[derive(Debug)]
pub struct MulOp;

impl<T: Numeric> GradRule<2, T> for MulOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // d/dx0 (x0*x1) = in_grad * x1, d/dx1 = in_grad * x0
        out.push(operands[0].requires_grad().then(|| {
            reduce_grad_to_shape(&(in_grad * &operands[1]), operands[0].shape(), retain_graph)
        }));
        out.push(operands[1].requires_grad().then(|| {
            reduce_grad_to_shape(&(in_grad * &operands[0]), operands[1].shape(), retain_graph)
        }));
    }
}

#[derive(Debug)]
pub struct DivOp;

impl<T: Signed> GradRule<2, T> for DivOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = a / b
        // dy/da = 1/b            -> grad_a = in_grad / b
        // dy/db = -a/b^2         -> grad_b = -(in_grad * a) / (b * b)
        out.push(operands[0].requires_grad().then(|| {
            reduce_grad_to_shape(&(in_grad / &operands[1]), operands[0].shape(), retain_graph)
        }));
        out.push(operands[1].requires_grad().then(|| {
            let num = -&(in_grad * &operands[0]);
            let rhs = &num / &(&operands[1] * &operands[1]);
            reduce_grad_to_shape(&rhs, operands[1].shape(), retain_graph)
        }));
    }
}

#[derive(Debug)]
pub struct PowOp;

impl<T: Float> GradRule<2, T> for PowOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = b.powf(e)
        // dy/db = e * b^(e-1)
        // dy/de = b^e * ln(b)
        let base = &operands[0];
        let exp = &operands[1];
        let one = scalar::<T>(1.0);

        out.push(base.requires_grad().then(|| {
            let exp_minus_one = exp - &one;
            let rhs = &(in_grad * exp) * &base.pow(&exp_minus_one);
            reduce_grad_to_shape(&rhs, base.shape(), retain_graph)
        }));

        out.push(exp.requires_grad().then(|| {
            // ln(b) computed as log base e of b, reusing the log op's convention: a.log(b)
            let ln_base = base.ln();
            let y = base.pow(exp);
            let rhs = &(in_grad * &y) * &ln_base;
            reduce_grad_to_shape(&rhs, exp.shape(), retain_graph)
        }));
    }
}

#[derive(Debug)]
pub struct MaximumOp;

impl<T: Numeric> GradRule<2, T> for MaximumOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor<T>; 2],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        // y = max(a, b): gradient flows to whichever operand attained the max.
        //   dy/da = (a >= b), dy/db = (a < b)
        // Comparisons yield `bool` masks; the gradient math runs in `T`, so the
        // masks are re-interpreted as 1/0 via `as_numeric::<T>()`.
        let a = &operands[0];
        let b = &operands[1];

        out.push(a.requires_grad().then(|| {
            reduce_grad_to_shape(
                &(in_grad * &a.gte(b).as_numeric::<T>()),
                a.shape(),
                retain_graph,
            )
        }));
        out.push(b.requires_grad().then(|| {
            reduce_grad_to_shape(
                &(in_grad * &a.lt(b).as_numeric::<T>()),
                b.shape(),
                retain_graph,
            )
        }));
    }
}

// void BackwardMaximum::compute_operands_grad(
//         const Tensor& out
// ) {
//     Tensor& a = m_operands[0];
//     Tensor& b = m_operands[1];
//     Tensor a_mask = a > b;
//     Tensor b_mask = a <= b;
//     a.accumulate_grad(a_mask * out.grad());
//     b.accumulate_grad(b_mask * out.grad());
// }
