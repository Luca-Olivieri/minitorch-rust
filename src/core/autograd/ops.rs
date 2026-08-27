use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::autograd::grad_fn::*;

#[derive(Debug)]
pub struct NegOp;
pub type BackwardNeg = NBackwardOp<NegOp, 1>;

impl GradRule<1> for NegOp {
    fn compute_grad(
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
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        // y = ln(a)
        // dy/da = 1 / a
        let a = &operands[0];
        let mut out_grads = Vec::with_capacity(1);

        out_grads.push(a.requires_grad().then(|| {
            in_grad / &a
        }));

        out_grads
    }
}

// #[derive(Debug)]
// pub struct AbsOp;
// pub type BackwardAbs = NBackwardOp<AbsOp, 1>;

// impl GradRule<1> for AbsOp {
//     fn compute_grad(
//         operands: &[GraphTensor; 1],
//         in_grad: &GraphTensor
//     ) -> Vec<Option<GraphTensor>> {
//         todo!()
//     }
// }

#[derive(Debug)]
pub struct AddOp;
pub type BackwardAdd = NBackwardOp<AddOp, 2>;

impl GradRule<2> for AddOp {
    fn compute_grad(
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        let ones = GraphTensor::new(operands[0].shape().clone(), 1.0, false); // TODO use a utility to deep copy instead
        out_grads.push(operands[0].requires_grad().then(|| in_grad * &ones));
        out_grads.push(operands[1].requires_grad().then(|| in_grad * &ones));
        out_grads
    }
}

#[derive(Debug)]
pub struct SubOp;
pub type BackwardSub = NBackwardOp<SubOp, 2>;

impl GradRule<2> for SubOp {
    fn compute_grad(
        operands: &[GraphTensor; 2],
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {
        let mut out_grads = Vec::with_capacity(2);
        let ones = GraphTensor::new(operands[0].shape().clone(), 1.0, false); // TODO use a utility to deep copy instead
        let minus_ones = GraphTensor::new(operands[0].shape().clone(), -1.0, false); // TODO use a utility to deep copy instead
        out_grads.push(operands[0].requires_grad().then(|| in_grad * &ones));
        out_grads.push(operands[1].requires_grad().then(|| in_grad * &minus_ones));
        out_grads
    }
}

#[derive(Debug)]
pub struct MulOp;
pub type BackwardMul = NBackwardOp<MulOp, 2>;

impl GradRule<2> for MulOp {
    fn compute_grad(
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
            let ones = GraphTensor::new(exp.shape().clone(), 1.0, false); // TODO use a utility to deep copy instead
            let exp_minus_one = exp - &ones;
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
