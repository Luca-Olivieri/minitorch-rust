use crate::core::{GraphTensor, tensor::AbstractTensor};
use std::marker::PhantomData;

/// Generic backward-op container: stores operands, arity N, delegates
/// gradient computation to the zero-sized policy type `Op`.
#[derive(Debug)]
pub struct NBackwardOp<Op, const N: usize> {
    pub(crate) operands: [GraphTensor; N],
    pub(crate) _marker: PhantomData<Op>,
}

impl<Op, const N: usize> NBackwardOp<Op, N> {
    pub fn new(operands: [GraphTensor; N]) -> Self {
        Self { operands, _marker: PhantomData }
    }
}

pub trait GradFnTrait: HasOperands + ComputesGrads + std::fmt::Debug {}
impl<T: HasOperands + ComputesGrads + std::fmt::Debug> GradFnTrait for T {}

pub trait HasOperands {
    fn get_operands(&self) -> &[GraphTensor];
}

pub trait ComputesGrads {
    fn compute_operands_grad(&self, in_grad: &GraphTensor) -> Vec<Option<GraphTensor>>;
}

// Written ONCE, for every N and every Op.
impl<Op, const N: usize> HasOperands for NBackwardOp<Op, N> {
    fn get_operands(&self) -> &[GraphTensor] {
        &self.operands
    }
}

/// The actual per-op math lives here. `Op` types are zero-sized markers.
pub trait GradRule<const N: usize> {
    fn compute_grad(
        operands: &[GraphTensor; N],
        in_grad: &GraphTensor,
    ) -> Vec<Option<GraphTensor>>;
}

// Written ONCE, for every Op that implements GradRule<N>.
impl<Op: GradRule<N>, const N: usize> ComputesGrads for NBackwardOp<Op, N> {
    fn compute_operands_grad(&self, in_grad: &GraphTensor) -> Vec<Option<GraphTensor>> {
        Op::compute_grad(&self.operands, in_grad)
    }
}

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
