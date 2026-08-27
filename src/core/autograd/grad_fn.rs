use crate::core::{GraphTensor};
use std::marker::PhantomData;

/// Generic backward-op container: stores operands, arity N, delegates
/// gradient computation to the zero-sized policy type `Op`.
#[derive(Debug)]
pub struct NBackwardOp<Op, const N: usize> {
    pub(crate) operands: [GraphTensor; N],
    pub(crate) _marker: PhantomData<Op>,
}

impl<Op, const N: usize> NBackwardOp<Op, N> {}

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
