use crate::core::{tensor::AbstractTensor, GraphTensor};
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
    /// Compute the gradients for each operand, appending exactly N entries to
    /// `out` (per-operand), reusing the caller-provided buffer to avoid per-node
    /// allocations during the backward walk. The caller clears `out` first.
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    );
}

impl<Op, const N: usize> HasOperands for NBackwardOp<Op, N> {
    fn get_operands(&self) -> &[GraphTensor] {
        &self.operands
    }
}

/// The actual per-op math lives here.
/// It takes `&self` so it can access operation-specific parameters (like `dim`).
/// It appends exactly N per-operand entries to `out`.
pub trait GradRule<const N: usize> {
    fn compute_grad(
        &self, // <-- Added `&self` to access struct fields
        operands: &[GraphTensor; N],
        in_grad: &GraphTensor,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    );
}

impl<Op: GradRule<N>, const N: usize> ComputesGrads for NBackwardOp<Op, N> {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.clear();
        self.op
            .compute_grad(&self.operands, in_grad, retain_graph, out);

        if !retain_graph {
            for grad in out.iter_mut() {
                if let Some(g) = grad {
                    g.get_node_mut().grad_fn = None;
                }
            }
        }
    }
}

/// Reduce an upstream gradient back to the shape of the tensor it belongs to.
///
/// The forward op broadcasts its operands to a common shape, so the gradient
/// arriving at an operand may be (strictly) larger than the operand itself.
/// The gradient of a broadcast is a sum along every axis that was expanded,
/// i.e. every axis where the operand's aligned dimension is 1 but the gradient's
/// is larger.
///
/// `sum(dims)` removes the reduced axes, so the axis is re-inserted (keepdim) to
/// keep positions stable while iterating, and any leading axes that only existed
/// because the target shape was rank-deficient are squeezed away at the end.
/// When `in_grad` already has `target_shape`, this is a no-op.
pub fn reduce_grad_to_shape(
    in_grad: &GraphTensor,
    target_shape: &[usize],
    retain_graph: bool,
) -> GraphTensor {
    let bs = in_grad.shape();
    if bs.as_slice() == target_shape {
        // On the higher-order path the graph structure is preserved, so the
        // operand gradient can simply share `in_grad`'s node (no allocation).
        if retain_graph {
            return in_grad.copy_s();
        }
        // First-order path: deep-copy, because the caller clears `grad_fn` on
        // the result in place, which requires an exclusively-owned node (the
        // shared seed grad must not be mutated).
        return in_grad.copy_d();
    }

    let nd = bs.len() - target_shape.len();

    // Right-align the target shape (NumPy convention): prepend the missing dims.
    let mut aligned = vec![1usize; nd];
    aligned.extend_from_slice(target_shape);

    let mut g = in_grad.copy_s();
    for d in (0..bs.len()).rev() {
        if aligned[d] == 1 && bs[d] > 1 {
            g = g.sum(&[d], true);
        }
    }

    // Remove the leading dims that only existed because the target was rank-deficient.
    for _ in 0..nd {
        g = g.squeeze(0);
    }
    g
}
