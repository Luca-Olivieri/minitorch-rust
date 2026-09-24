use std::fmt;
use std::rc::Rc;

use crate::core::autograd::erased::{ErasedHandle, ErasedTensor, FloatKind, FloatRepr, GradValue};
use crate::core::autograd::ops::math::{
    AbsOp, AddOp, DivOp, ExpOp, LnOp, MatmulOp, MaximumOp, MulOp, NegOp, PowOp, SqrtOp, SubOp,
};
use crate::core::autograd::ops::reduce::{AvgPool2dOp, MaxOp, MaxPool2dOp, SumOp};
use crate::core::autograd::ops::shape::{
    BroadcastOp, CopyDOp, ExpandOp, PadOp, ReshapeOp, SliceOp, SqueezeOp, StridedSliceOp,
    TransposeOp, UnsqueezeOp,
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
/// The operands are stored *dtype-erased* ([`ErasedHandle`]) so an edge can
/// connect nodes of different float dtypes (a differentiable cast). The concrete
/// rule is materialized —
/// [`BackwardSource::into_grad_fn`] — only at backward-build time, when the
/// node's dtype is known and every rule's `Numeric`/`Signed`/`Float` bound is
/// satisfiable.
pub(crate) struct BackwardSource {
    operands: Vec<ErasedHandle>,
    op: BackwardOpKind,
}

impl Clone for BackwardSource {
    fn clone(&self) -> Self {
        Self {
            operands: self.operands.clone(),
            op: self.op.clone(),
        }
    }
}

impl fmt::Debug for BackwardSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // GraphTensor's Debug needs a styler bound; print counts instead so the
        // node-level Debug impl stays usable for every dtype.
        f.debug_struct("BackwardSource")
            .field("op", &self.op)
            .field("operands", &self.operands.len())
            .finish()
    }
}

impl BackwardSource {
    pub(crate) fn new(operands: Vec<ErasedHandle>, op: BackwardOpKind) -> Self {
        Self { operands, op }
    }

    pub(crate) fn operands(&self) -> &[ErasedHandle] {
        &self.operands
    }
}

/// Build the deferred gradient edge for a differentiable op, or `None` when
/// the operand dtype does not participate in autograd or the computation is
/// explicitly marked no-grad.
///
/// This is the single gate for "non-differentiable operations produce no graph
/// edges": `T::DIFFERENTIABLE` is a compile-time constant, so for integer/bool
/// instantiations the whole branch is eliminated and no operands are erased.
pub(crate) fn maybe_edge<T: Dtype, const N: usize>(
    operands: &[&GraphTensor<T>; N],
    op: BackwardOpKind,
) -> Option<Box<BackwardSource>> {
    if T::DIFFERENTIABLE && operands.iter().all(|operand| !operand.is_no_grad()) {
        Some(Box::new(BackwardSource::new(
            operands.iter().map(|o| ErasedHandle::erase(o)).collect(),
            op,
        )))
    } else {
        None
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
    AbsOp,
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
    /// 2D average pooling: `out[b, c, oh, ow]` is the mean of the window at
    /// `(oh * stride.0, ow * stride.1)`. The gradient scatters the upstream
    /// gradient, divided by the window size, over every covered input position.
    AvgPool2dOp {
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    /// 2D maximum pooling. The cached groups contain logical input indices for
    /// every maximum in each output window, so tied maxima can share the
    /// upstream gradient without rescanning the input.
    MaxPool2dOp {
        kernel: (usize, usize),
        stride: (usize, usize),
        max_indices: Rc<Vec<Vec<usize>>>,
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
    /// Zero-pad every dimension by `pads[d] = (before, after)` elements.
    PadOp {
        pads: Vec<(usize, usize)>,
    },
    /// Extract the window `[start, start + len)` along every dimension.
    SliceOp {
        ranges: Vec<(usize, usize)>,
    },
    /// Extract a strided window from every dimension. The gradient scatters
    /// upstream values back to the sampled input positions.
    StridedSliceOp {
        ranges: Vec<(usize, usize, usize)>,
    },
    /// Reinterpret the logical elements under a new shape (materializing).
    ReshapeOp {
        new_shape: Vec<usize>,
    },
    /// A differentiable dtype cast. The destination dtype is the node's own
    /// dtype; the source is recovered from the single operand, so neither has to
    /// be stored here.
    CastOp,
}

impl BackwardSource {
    /// Rebuild the concrete, typed `NBackwardOp` (operands + rule struct) for a
    /// node of dtype `T`. Called at backward-build time, so every rule's
    /// `Numeric`/`Signed`/`Float` bound holds.
    ///
    /// Every operand must have the node's dtype: only a differentiable cast
    /// connects different dtypes, and it uses its own rule.
    pub(crate) fn into_grad_fn<T: FloatRepr>(self) -> Box<dyn GradFnTrait<T>> {
        let operands = self
            .operands
            .into_iter()
            .map(|handle| T::take_erased(handle.into_erased()))
            .collect();
        box_grad_rule(operands, self.op)
    }
}

/// Box the concrete rule struct for `op` over typed operands.
fn box_grad_rule<T: Float>(
    operands: Vec<GraphTensor<T>>,
    op: BackwardOpKind,
) -> Box<dyn GradFnTrait<T>> {
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

    match op {
        BackwardOpKind::AddOp => box_rule::<AddOp, 2, T>(operands, AddOp),
        BackwardOpKind::MulOp => box_rule::<MulOp, 2, T>(operands, MulOp),
        BackwardOpKind::SubOp => box_rule::<SubOp, 2, T>(operands, SubOp),
        BackwardOpKind::DivOp => box_rule::<DivOp, 2, T>(operands, DivOp),
        BackwardOpKind::MaximumOp => box_rule::<MaximumOp, 2, T>(operands, MaximumOp),
        BackwardOpKind::MatmulOp => box_rule::<MatmulOp, 2, T>(operands, MatmulOp {}),
        BackwardOpKind::PowOp => box_rule::<PowOp, 2, T>(operands, PowOp),
        BackwardOpKind::NegOp => box_rule::<NegOp, 1, T>(operands, NegOp),
        BackwardOpKind::AbsOp => box_rule::<AbsOp, 1, T>(operands, AbsOp),
        BackwardOpKind::LnOp => box_rule::<LnOp, 1, T>(operands, LnOp),
        BackwardOpKind::ExpOp => box_rule::<ExpOp, 1, T>(operands, ExpOp),
        BackwardOpKind::SqrtOp => box_rule::<SqrtOp, 1, T>(operands, SqrtOp),
        BackwardOpKind::SumOp { dims, keepdim } => {
            box_rule::<SumOp, 1, T>(operands, SumOp { dims, keepdim })
        }
        BackwardOpKind::MaxOp { dims, keepdim } => {
            box_rule::<MaxOp, 1, T>(operands, MaxOp { dims, keepdim })
        }
        BackwardOpKind::AvgPool2dOp { kernel, stride } => {
            box_rule::<AvgPool2dOp, 1, T>(operands, AvgPool2dOp { kernel, stride })
        }
        BackwardOpKind::MaxPool2dOp {
            kernel,
            stride,
            max_indices,
        } => box_rule::<MaxPool2dOp, 1, T>(
            operands,
            MaxPool2dOp {
                kernel,
                stride,
                max_indices,
            },
        ),
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
        BackwardOpKind::PadOp { pads } => box_rule::<PadOp, 1, T>(operands, PadOp { pads }),
        BackwardOpKind::SliceOp { ranges } => {
            box_rule::<SliceOp, 1, T>(operands, SliceOp { ranges })
        }
        BackwardOpKind::StridedSliceOp { ranges } => {
            box_rule::<StridedSliceOp, 1, T>(operands, StridedSliceOp { ranges })
        }
        BackwardOpKind::ReshapeOp { new_shape } => {
            box_rule::<ReshapeOp, 1, T>(operands, ReshapeOp { new_shape })
        }
        // Cast edges are the one heterogeneous case; they never reach here (see
        // `materialize_grad_fn`).
        BackwardOpKind::CastOp => unreachable!("cast edges use ErasedCastGradFn"),
    }
}

/// The erased, object-safe form of a node's backward rule.
///
/// The plan stores one per inner node and calls it uniformly with erased
/// gradient slots; the typed rule and its reusable scratch buffer live behind
/// the `f32`/`f64` dispatch in [`TypedGradFn`].
pub(crate) trait ErasedGradFn {
    fn compute(
        &mut self,
        in_grad: &GradValue,
        retain_graph: bool,
        out: &mut Vec<Option<GradValue>>,
    );
}

/// A concrete `GradFnTrait<T>` plus a reusable per-operand scratch buffer, seen
/// through the erased [`ErasedGradFn`] interface.
struct TypedGradFn<T: FloatRepr> {
    inner: Box<dyn GradFnTrait<T>>,
    scratch: Vec<Option<GraphTensor<T>>>,
}

impl<T: FloatRepr> ErasedGradFn for TypedGradFn<T> {
    fn compute(
        &mut self,
        in_grad: &GradValue,
        retain_graph: bool,
        out: &mut Vec<Option<GradValue>>,
    ) {
        let typed_in = T::ref_grad_value(in_grad);
        self.inner
            .compute_operands_grad(typed_in, retain_graph, &mut self.scratch);
        // Hand the typed results to the plan as erased slots, recycling the
        // scratch buffer (no per-node allocation after warmup).
        out.clear();
        out.extend(
            self.scratch
                .drain(..)
                .map(|grad| grad.map(T::into_grad_value)),
        );
    }
}

/// Backward rule for a differentiable cast.
///
/// The upstream gradient has the destination dtype and the output must have the
/// source dtype (the invariant "a gradient has its input's dtype"). The reverse
/// conversion is applied by reusing the *forward* cast op, so a higher-order
/// differentiation through a cast keeps working: the reverse cast itself gets an
/// edge.
struct ErasedCastGradFn {
    src: FloatKind,
    dst: FloatKind,
}

impl ErasedGradFn for ErasedCastGradFn {
    fn compute(
        &mut self,
        in_grad: &GradValue,
        _retain_graph: bool,
        out: &mut Vec<Option<GradValue>>,
    ) {
        let grad = match (self.src, self.dst) {
            (FloatKind::F32, FloatKind::F32) => match in_grad {
                GradValue::F32(g) => GradValue::F32(g.copy_s()),
                GradValue::F64(_) => unreachable!("cast grad dtype mismatch"),
            },
            (FloatKind::F64, FloatKind::F64) => match in_grad {
                GradValue::F64(g) => GradValue::F64(g.copy_s()),
                GradValue::F32(_) => unreachable!("cast grad dtype mismatch"),
            },
            (FloatKind::F64, FloatKind::F32) => match in_grad {
                GradValue::F32(g) => GradValue::F64(g.cast::<f64>()),
                GradValue::F64(_) => unreachable!("cast grad dtype mismatch"),
            },
            (FloatKind::F32, FloatKind::F64) => {
                #[cfg(feature = "allow_lossy_casts")]
                {
                    match in_grad {
                        GradValue::F64(g) => GradValue::F32(g.cast_lossy::<f32>()),
                        GradValue::F32(_) => unreachable!("cast grad dtype mismatch"),
                    }
                }
                #[cfg(not(feature = "allow_lossy_casts"))]
                {
                    let _ = in_grad;
                    unreachable!("f32 -> f64 cast edge requires allow_lossy_casts")
                }
            }
        };

        out.clear();
        out.push(Some(grad));
    }
}

/// Materialize a node's deferred edge into its erased backward rule.
///
/// The `f32`/`f64` match is the only place the scheduler picks a concrete float
/// for a node; everything downstream is erased.
pub(crate) fn materialize_grad_fn(node: &ErasedTensor) -> Option<Box<dyn ErasedGradFn>> {
    fn make<T: FloatRepr>(source: BackwardSource) -> Box<dyn ErasedGradFn> {
        Box::new(TypedGradFn::<T> {
            inner: source.into_grad_fn::<T>(),
            scratch: Vec::new(),
        })
    }

    let dst = node.kind();
    match node {
        ErasedTensor::F32(g) => {
            let source = g.node.grad_fn.as_ref()?;
            if matches!(&source.op, BackwardOpKind::CastOp) {
                return Some(cast_grad_fn(source, dst));
            }
            Some(make::<f32>((**source).clone()))
        }
        ErasedTensor::F64(g) => {
            let source = g.node.grad_fn.as_ref()?;
            if matches!(&source.op, BackwardOpKind::CastOp) {
                return Some(cast_grad_fn(source, dst));
            }
            Some(make::<f64>((**source).clone()))
        }
    }
}

/// Build the erased rule for a cast edge, reading the source dtype off its single
/// operand (the node's own dtype is the destination).
fn cast_grad_fn(source: &BackwardSource, dst: FloatKind) -> Box<dyn ErasedGradFn> {
    let src = source.operands()[0].clone().into_erased().kind();
    Box::new(ErasedCastGradFn { src, dst })
}

/// Generic backward-op container: stores operands, arity N, the operation state,
/// and the dtype `T` of the operands. Autograd only ever *runs* for float dtypes
/// (see `impl<T: FloatRepr> GraphTensor<T>`), and `BackwardSource` freezes
/// *building* an edge independently of any rule — but the concrete
/// `NBackwardOp`, fully typed in `T`, is produced by
/// [`BackwardSource::into_grad_fn`] at backward time and then hidden behind the
/// erased `ErasedGradFn` the plan holds.
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

pub trait GradFnTrait<T: Dtype = f64>: HasOperands<T> + ComputesGrads<T> + std::fmt::Debug {}
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
