use std::fmt;

use crate::core::autograd::ops::math::{
    AddOp, DivOp, ExpOp, LnOp, MatmulOp, MaximumOp, MulOp, NegOp, PowOp, SqrtOp, SubOp,
};
use crate::core::autograd::ops::reduce::{MaxOp, SumOp};
use crate::core::autograd::ops::shape::{
    BroadcastOp, CopyDOp, ExpandOp, SqueezeOp, TransposeOp, UnsqueezeOp,
};
use crate::core::{
    GraphTensor,
    dtype::{Dtype, Float, Numeric},
    tensor::{AbstractTensor, TensorNodeAccess},
};

/// Deferred gradient source for a forward node.
///
/// A forward op records only its operands and an op *marker* — building the
/// edge requires no [`GradRule`] bound at all. That is the whole point: the
/// forward home of an op (e.g. `sub` on every [`Numeric`]) is decoupled from
/// what its backward math needs (a [`GradRule`] bound like `T: Signed`).
///
/// The concrete rule is materialized — [`BackwardSource::into_grad_fn`] — only
/// at backward-build time, when `T: Float` is known and every rule's
/// `Numeric`/`Signed`/`Float` bound is satisfiable.
pub(crate) struct BackwardSource<T: Dtype = f64> {
    operands: Vec<GraphTensor<T>>,
    op: BackwardOpKind,
}

impl<T: Dtype> Clone for BackwardSource<T> {
    fn clone(&self) -> Self {
        Self {
            operands: self.operands.iter().map(|o| o.copy_s()).collect(),
            op: self.op.clone(),
        }
    }
}

impl<T: Dtype> fmt::Debug for BackwardSource<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // GraphTensor's Debug needs a styler bound; print counts instead so the
        // node-level Debug impl stays usable for every dtype.
        f.debug_struct("BackwardSource")
            .field("op", &self.op)
            .field("operands", &self.operands.len())
            .finish()
    }
}

impl<T: Dtype> BackwardSource<T> {
    pub(crate) fn new(operands: Vec<GraphTensor<T>>, op: BackwardOpKind) -> Self {
        Self { operands, op }
    }

    pub(crate) fn operands(&self) -> &[GraphTensor<T>] {
        &self.operands
    }
}

/// Marker for the differentiable op an edge records. Parametrized variants hold
/// the op's own state (e.g. the reduced dims of `sum`). Names mirror the rule
/// structs in `ops/{math,reduce,shape}.rs` so forward code refers to one ident.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone)]
pub(crate) enum BackwardOpKind {
    AddOp,
    MulOp,
    SubOp,
    DivOp,
    NegOp,
    LnOp,
    ExpOp,
    SqrtOp,
    PowOp,
    MatmulOp,
    MaximumOp,
    SumOp {
        dims: Vec<usize>,
        keepdim: bool,
    },
    MaxOp {
        dims: Vec<usize>,
        keepdim: bool,
    },
    CopyDOp,
    UnsqueezeOp {
        dim: usize,
    },
    SqueezeOp {
        dim: usize,
    },
    TransposeOp {
        dim_a: usize,
        dim_b: usize,
    },
    ExpandOp {
        dim: usize,
    },
    BroadcastOp {
        old_shape: Vec<usize>,
    },
}

impl<T: Float> BackwardSource<T> {
    /// Rebuild the concrete `NBackwardOp` (operands + rule struct) the forward
    /// op would previously have boxed. Backward passes only run for float
    /// dtypes, so every rule's `Numeric`/`Signed`/`Float` bound holds here.
    pub(crate) fn into_grad_fn(self) -> Box<dyn GradFnTrait<T>> {
        fn box_rule<Op, const N: usize, T: Float>(
            operands: Vec<GraphTensor<T>>,
            op: Op,
        ) -> Box<dyn GradFnTrait<T>>
        where
            Op: GradRule<N, T> + fmt::Debug + 'static,
        {
            let operands: [GraphTensor<T>; N] = match operands.try_into() {
                Ok(operands) => operands,
                Err(_) => panic!("BackwardSource operand count must match the op arity"),
            };
            Box::new(NBackwardOp { operands, op }) as Box<dyn GradFnTrait<T>>
        }

        let BackwardSource { operands, op } = self;
        match op {
            BackwardOpKind::AddOp => box_rule::<AddOp, 2, T>(operands, AddOp),
            BackwardOpKind::MulOp => box_rule::<MulOp, 2, T>(operands, MulOp),
            BackwardOpKind::SubOp => box_rule::<SubOp, 2, T>(operands, SubOp),
            BackwardOpKind::DivOp => box_rule::<DivOp, 2, T>(operands, DivOp),
            BackwardOpKind::MaximumOp => box_rule::<MaximumOp, 2, T>(operands, MaximumOp),
            BackwardOpKind::MatmulOp => box_rule::<MatmulOp, 2, T>(operands, MatmulOp {}),
            BackwardOpKind::PowOp => box_rule::<PowOp, 2, T>(operands, PowOp),
            BackwardOpKind::NegOp => box_rule::<NegOp, 1, T>(operands, NegOp),
            BackwardOpKind::LnOp => box_rule::<LnOp, 1, T>(operands, LnOp),
            BackwardOpKind::ExpOp => box_rule::<ExpOp, 1, T>(operands, ExpOp),
            BackwardOpKind::SqrtOp => box_rule::<SqrtOp, 1, T>(operands, SqrtOp),
            BackwardOpKind::SumOp { dims, keepdim } => {
                box_rule::<SumOp, 1, T>(operands, SumOp { dims, keepdim })
            }
            BackwardOpKind::MaxOp { dims, keepdim } => {
                box_rule::<MaxOp, 1, T>(operands, MaxOp { dims, keepdim })
            }
            BackwardOpKind::CopyDOp => box_rule::<CopyDOp, 1, T>(operands, CopyDOp {}),
            BackwardOpKind::UnsqueezeOp { dim } => {
                box_rule::<UnsqueezeOp, 1, T>(operands, UnsqueezeOp { dim })
            }
            BackwardOpKind::SqueezeOp { dim } => {
                box_rule::<SqueezeOp, 1, T>(operands, SqueezeOp { dim })
            }
            BackwardOpKind::TransposeOp { dim_a, dim_b } => {
                box_rule::<TransposeOp, 1, T>(operands, TransposeOp { dim_a, dim_b })
            }
            BackwardOpKind::ExpandOp { dim } => box_rule::<ExpandOp, 1, T>(operands, ExpandOp { dim }),
            BackwardOpKind::BroadcastOp { old_shape } => {
                box_rule::<BroadcastOp, 1, T>(operands, BroadcastOp { old_shape })
            }
        }
    }
}

/// Generic backward-op container: stores operands, arity N, the operation state,
/// and the dtype `T` of the operands. Autograd only ever *runs* for float dtypes
/// (see `impl<T: Float> GraphTensor<T>`), and `BackwardSource` freezes *building*
/// an edge independently of any rule — but the concrete `NBackwardOp` is produced
/// by [`BackwardSource::into_grad_fn`] at backward time.
pub struct NBackwardOp<Op, const N: usize, T: Dtype = f64> {
    pub(crate) operands: [GraphTensor<T>; N],
    pub(crate) op: Op, // Holds the actual struct (and its fields like `dim`)
}

impl<Op: fmt::Debug, const N: usize, T: Dtype> fmt::Debug for NBackwardOp<Op, N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NBackwardOp")
            .field("op", &self.op)
            .finish_non_exhaustive()
    }
}

pub trait GradFnTrait<T: Dtype = f64>:
    HasOperands<T> + ComputesGrads<T> + std::fmt::Debug
{
}
impl<T: Dtype, G: HasOperands<T> + ComputesGrads<T> + std::fmt::Debug> GradFnTrait<T> for G {}

pub trait HasOperands<T: Dtype = f64> {
    fn get_operands(&self) -> &[GraphTensor<T>];
}

pub trait ComputesGrads<T: Dtype = f64> {
    /// Compute the gradients for each operand, appending exactly N entries to
    /// `out` (per-operand), reusing the caller-provided buffer to avoid per-node
    /// allocations during the backward walk. The caller clears `out` first.
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    );
}

impl<Op, const N: usize, T: Dtype> HasOperands<T> for NBackwardOp<Op, N, T> {
    fn get_operands(&self) -> &[GraphTensor<T>] {
        &self.operands
    }
}

/// The actual per-op math lives here.
/// It takes `&self` so it can access operation-specific parameters (like `dim`).
/// It appends exactly N per-operand entries to `out`.
pub trait GradRule<const N: usize, T: Dtype = f64> {
    fn compute_grad(
        &self, // <-- Added `&self` to access struct fields
        operands: &[GraphTensor<T>; N],
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    );
}

impl<Op: GradRule<N, T>, const N: usize, T: Dtype> ComputesGrads<T> for NBackwardOp<Op, N, T> {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor<T>,
        retain_graph: bool,
        out: &mut Vec<Option<GraphTensor<T>>>,
    ) {
        out.clear();
        self.op
            .compute_grad(&self.operands, in_grad, retain_graph, out);

        if !retain_graph {
            for g in out.iter_mut().flatten() {
                g.get_node_mut().grad_fn = None;
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
///
/// Every backward path is, by construction, a linear combination over the
/// Numeric-homed forward ops (sum/squeeze/copy), hence the `Numeric` bound:
/// integer dtypes stay graph-structurally reducible even though a *backward run*
/// only exists for floats.
pub fn reduce_grad_to_shape<T: Numeric>(
    in_grad: &GraphTensor<T>,
    target_shape: &[usize],
    retain_graph: bool,
) -> GraphTensor<T> {
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