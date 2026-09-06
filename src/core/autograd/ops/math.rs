use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::autograd::grad_fn::*;

#[derive(Debug)]
pub struct NegOp;
pub type BackwardNeg = NBackwardOp<NegOp, 1>;

impl GradRule<1> for NegOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        vec![operands[0].requires_grad().then(|| -in_grad)]
    }
}

#[derive(Debug)]
pub struct LnOp;
pub type BackwardLn = NBackwardOp<LnOp, 1>;
impl GradRule<1> for LnOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = ln(a)
        // dy/da = 1 / a
        let a = &operands[0];
        let mut out_grads = Vec::with_capacity(1);

        // TODO should do inplace operations if retain_graph = False

        out_grads.push(a.requires_grad().then(|| {
            in_grad / a
        }));

        out_grads
    }
}

#[derive(Debug)]
pub struct ExpOp;
pub type BackwardExp = NBackwardOp<ExpOp, 1>;
impl GradRule<1> for ExpOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = exp(a) = e^a
        // dy/da = e^a = y
        let mut out_grads = Vec::with_capacity(1);
        out_grads.push(operands[0].requires_grad().then(|| in_grad * &operands[0].exp()));
        out_grads
    }
}

#[derive(Debug)]
pub struct MatmulOp;
pub type BackwardMatmul = NBackwardOp<MatmulOp, 2>;

impl GradRule<2> for MatmulOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        out_grads.push(operands[0].requires_grad().then(|| grad_a(&operands[0], &operands[1], in_grad)));
        out_grads.push(operands[1].requires_grad().then(|| grad_b(&operands[0], &operands[1], in_grad)));
        out_grads
    }
}

// Turn a 1D tensor [k] into a [1, k] "row" view.
fn into_row(t: &GraphTensor) -> (GraphTensor, bool) {
    if t.shape().len() == 1 {
        (t.unsqueeze(0), true)
    } else {
        (t.copy_s(), false)
    }
}

// Turn a 1D tensor [k] into a [k, 1] "column" view.
fn into_col(t: &GraphTensor) -> (GraphTensor, bool) {
    if t.shape().len() == 1 {
        (t.unsqueeze(1), true)
    } else {
        (t.copy_s(), false)
    }
}

// Reshape the upstream gradient to the 2D [m, n] shape the kernel expects.
fn grad_to_2d(
    a_ndim: usize,
    b_ndim: usize,
    in_grad: &GraphTensor
) -> GraphTensor {
    match (a_ndim, b_ndim) {
        (1, 1) => in_grad.unsqueeze(0).unsqueeze(1),
        (1, _) => in_grad.unsqueeze(0),
        (_, 1) => in_grad.unsqueeze(1),
        (_, _) => in_grad.copy_s(),
    }
}

// dL/dA = in_grad @ B^T
fn grad_a(a: &GraphTensor, b: &GraphTensor, in_grad: &GraphTensor) -> GraphTensor {
    let (_, a_was_1d) = into_row(a);
    let (b2, _) = into_col(b);

    let g = GraphTensor::matmul(&grad_to_2d(a.shape().len(), b.shape().len(), in_grad), &b2.transpose());

    if a_was_1d { g.squeeze(0) } else { g }
}

// dL/dB = A^T @ in_grad
fn grad_b(a: &GraphTensor, b: &GraphTensor, in_grad: &GraphTensor) -> GraphTensor {
    let (a2, _) = into_row(a);
    let (_, b_was_1d) = into_col(b);

    let g = GraphTensor::matmul(&a2.transpose(), &grad_to_2d(a.shape().len(), b.shape().len(), in_grad));

    if b_was_1d { g.squeeze(1) } else { g }
}

#[derive(Debug)]
pub struct AddOp;
pub type BackwardAdd = NBackwardOp<AddOp, 2>;

impl GradRule<2> for AddOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        out_grads.push(operands[0].requires_grad().then(|| in_grad.copy_d()));
        out_grads.push(operands[1].requires_grad().then(|| in_grad.copy_d()));
        out_grads
    }
}

#[derive(Debug)]
pub struct SubOp;
pub type BackwardSub = NBackwardOp<SubOp, 2>;

impl GradRule<2> for SubOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        out_grads.push(operands[0].requires_grad().then(|| in_grad.copy_d()));
        out_grads.push(operands[1].requires_grad().then(|| -in_grad));
        out_grads
    }
}

#[derive(Debug)]
pub struct MulOp;
pub type BackwardMul = NBackwardOp<MulOp, 2>;

impl GradRule<2> for MulOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        // d/dx0 (x0*x1) = in_grad * x1, d/dx1 = in_grad * x0
        out_grads.push(operands[0].requires_grad().then(|| in_grad * &operands[1]));
        out_grads.push(operands[1].requires_grad().then(|| in_grad * &operands[0]));
        out_grads
    }
}

#[derive(Debug)]
pub struct DivOp;
pub type BackwardDiv = NBackwardOp<DivOp, 2>;
impl GradRule<2> for DivOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = a / b
        // dy/da = 1/b            -> grad_a = in_grad / b
        // dy/db = -a/b^2         -> grad_b = -(in_grad * a) / (b * b)
        let mut out_grads = Vec::with_capacity(2);
        out_grads.push(operands[0].requires_grad().then(|| in_grad / &operands[1]));
        out_grads.push(operands[1].requires_grad().then(|| {
            &(-&(in_grad * &operands[0])) / &(&operands[1] * &operands[1])
        }));
        out_grads
    }
}

#[derive(Debug)]
pub struct PowOp;
pub type BackwardPow = NBackwardOp<PowOp, 2>;
impl GradRule<2> for PowOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = b.powf(e)
        // dy/db = e * b^(e-1)
        // dy/de = b^e * ln(b)
        let base = &operands[0];
        let exp = &operands[1];
        let mut out_grads = Vec::with_capacity(2);

        out_grads.push(base.requires_grad().then(|| {
            let exp_minus_one = exp - 1.0;
            &(in_grad * exp) * &base.pow(&exp_minus_one)
        }));

        out_grads.push(exp.requires_grad().then(|| {
            // ln(b) computed as log base e of b, reusing the log op's convention: a.log(b)
            let ln_base = base.ln();
            let y = base.pow(exp);
            &(in_grad * &y) * &ln_base
        }));

        out_grads
    }
}

#[derive(Debug)]
pub struct MaximumOp;
pub type BackwardMaximum = NBackwardOp<MaximumOp, 2>;
impl GradRule<2> for MaximumOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = b.powf(e)
        // dy/db = e * b^(e-1)
        // dy/de = b^e * ln(b)
        let a = &operands[0];
        let b = &operands[1];
        let mut out_grads = Vec::with_capacity(2);

        out_grads.push(a.requires_grad().then(|| { in_grad * &a.gte(b) }));
        out_grads.push(b.requires_grad().then(|| { in_grad * &a.lt(b) })); // TODO implement a NOT operator

        out_grads
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
