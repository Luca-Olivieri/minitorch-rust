use crate::core::{GraphTensor, tensor::AbstractTensor};
use std::fmt::Debug;

/// Generic backward-op container: stores operands, arity N, and the operation state.
#[derive(Debug)]
pub struct NBackwardOp<Op, const N: usize> {
    pub(crate) operands: [GraphTensor; N],
    pub(crate) op: Op, // Holds the actual struct (and its fields like `dim`)
}

pub trait GradFnTrait: HasOperands + ComputesGrads + Debug {}
impl<T: HasOperands + ComputesGrads + Debug> GradFnTrait for T {}

pub trait HasOperands {
    fn get_operands(&self) -> &[GraphTensor];
}

pub trait ComputesGrads {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor,
        retain_graph: bool,
    ) -> Vec<Option<GraphTensor>>;
}

impl<Op, const N: usize> HasOperands for NBackwardOp<Op, N> {
    fn get_operands(&self) -> &[GraphTensor] {
        &self.operands
    }
}

/// The actual per-op math lives here.
/// It takes `&self` so it can access operation-specific parameters (like `dim`).
pub trait GradRule<const N: usize> {
    fn compute_grad(
        &self, // <-- Added `&self` to access struct fields
        operands: &[GraphTensor; N],
        in_grad: &GraphTensor,
    ) -> Vec<Option<GraphTensor>>;
}

impl<Op: GradRule<N>, const N: usize> ComputesGrads for NBackwardOp<Op, N> {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor,
        retain_graph: bool,
    ) -> Vec<Option<GraphTensor>> {
        // Call it as a method on `self.op`
        let mut grads = self.op.compute_grad(&self.operands, in_grad);

        for grad in grads.iter_mut() {
            if !retain_graph && let Some(g) = grad {
                g.get_node_mut().grad_fn = None;
            }
        }

        grads
    }
}
