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

        out_grads.push(a.requires_grad().then(|| {
            in_grad / a
        }));

        out_grads
    }
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
