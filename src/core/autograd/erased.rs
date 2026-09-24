//! Dtype erasure for the autograd scheduler.
//!
//! Autograd stays *generic* everywhere the math is: storage, ops and [`GradRule`]
//! (`crate::core::autograd::grad_fn::GradRule`) are monomorphized per dtype.
//! Only the scheduler's bookkeeping is erased, so that a single backward pass
//! can mix the two differentiable dtypes — which is exactly what a
//! differentiable `cast` produces.
//!
//! The dtype universe of autograd is closed: only `f32` and `f64` can carry a
//! gradient (`Dtype::DIFFERENTIABLE`), so erasure is a total two-variant enum,
//! not a dynamic downcast in the hot path. Forward edges are nevertheless built
//! in generic code (`maybe_edge<T: Dtype>`), where `T` may be *any* dtype, so
//! [`crate::core::autograd::grad_fn::BackwardSource`] stores the untyped
//! [`ErasedHandle`] and the plan checks it into an [`ErasedTensor`] once.
//!
//! [`GradRule`]: crate::core::autograd::grad_fn::GradRule

use std::any::Any;
use std::rc::Rc;

use crate::core::dtype::{Dtype, Float};
use crate::core::node::TensorNode;
use crate::core::tensor::{AbstractTensor, GraphTensor};

/// The closed set of gradient dtypes, as a value.
///
/// The erased cast rule must pick a concrete reverse conversion from the
/// source/destination pair it was built for without being generic over either
/// dtype (a cast is the one edge that connects two different floats), so the
/// pair is carried as values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatKind {
    F32,
    F64,
}

/// A gradient slot: the dtype-erased accumulator the scheduler writes into.
///
/// Every contribution to a given node has that node's dtype (the invariant
/// "the gradient w.r.t. an input has that input's dtype"), so adding two slots
/// only ever meets matching variants; a mismatch is an engine bug, not a user
/// error.
pub(crate) enum GradValue {
    F32(GraphTensor<f32>),
    F64(GraphTensor<f64>),
}

impl GradValue {
    /// Share the underlying node (a gradient alias, not a copy).
    pub(crate) fn copy_s(&self) -> GradValue {
        match self {
            GradValue::F32(g) => GradValue::F32(g.copy_s()),
            GradValue::F64(g) => GradValue::F64(g.copy_s()),
        }
    }
}

/// A dtype-erased handle to a differentiable forward node.
///
/// Doubles as the plan's node key (identity is the underlying `Rc` pointer) and
/// as the package of accessors the walk needs, so the plan never names a
/// concrete float outside materialization.
pub(crate) enum ErasedTensor {
    F32(GraphTensor<f32>),
    F64(GraphTensor<f64>),
}

impl ErasedTensor {
    pub(crate) fn requires_grad(&self) -> bool {
        match self {
            ErasedTensor::F32(g) => g.requires_grad(),
            ErasedTensor::F64(g) => g.requires_grad(),
        }
    }

    /// A node without an edge is a leaf for the output-map filter.
    pub(crate) fn is_leaf(&self) -> bool {
        match self {
            ErasedTensor::F32(g) => g.node.grad_fn().is_none(),
            ErasedTensor::F64(g) => g.node.grad_fn().is_none(),
        }
    }

    pub(crate) fn shape(&self) -> &Vec<usize> {
        match self {
            ErasedTensor::F32(g) => g.shape(),
            ErasedTensor::F64(g) => g.shape(),
        }
    }

    /// The concrete float this node carries.
    pub(crate) fn kind(&self) -> FloatKind {
        match self {
            ErasedTensor::F32(_) => FloatKind::F32,
            ErasedTensor::F64(_) => FloatKind::F64,
        }
    }

    /// Identity used to de-duplicate nodes during the graph walk.
    pub(crate) fn ptr(&self) -> *const () {
        match self {
            ErasedTensor::F32(g) => Rc::as_ptr(&g.node) as *const (),
            ErasedTensor::F64(g) => Rc::as_ptr(&g.node) as *const (),
        }
    }

    /// The node's edge operands, erased; empty for a leaf.
    pub(crate) fn erased_operands(&self) -> Vec<ErasedTensor> {
        let source = match self {
            ErasedTensor::F32(g) => g.node.grad_fn(),
            ErasedTensor::F64(g) => g.node.grad_fn(),
        };
        source
            .map(|src| {
                src.operands()
                    .iter()
                    .map(|h| h.clone().into_erased())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Seed gradient: an all-ones tensor of this node's dtype and shape.
    pub(crate) fn one_grad(&self, retain_graph: bool) -> GradValue {
        let shape = self.shape().clone();
        match self {
            ErasedTensor::F32(_) => GradValue::F32(GraphTensor::new(shape, 1.0f32, retain_graph)),
            ErasedTensor::F64(_) => GradValue::F64(GraphTensor::new(shape, 1.0f64, retain_graph)),
        }
    }
}

/// An untyped `Rc` handle to a forward node.
///
/// Forward edges are built in generic code, so erasure cannot dispatch on the
/// operand dtype at construction time. Storing `Rc<dyn Any>` is valid for every
/// [`Dtype`] (they are all `'static`); the plan interprets it once it knows
/// which floats the graph contains.
#[derive(Clone)]
pub(crate) struct ErasedHandle(Rc<dyn Any>);

impl ErasedHandle {
    pub(crate) fn erase<T: Dtype>(t: &GraphTensor<T>) -> Self {
        ErasedHandle(t.node.clone() as Rc<dyn Any>)
    }

    /// Check the untyped handle into the closed float enum. Only ever called on
    /// handles produced for a differentiable (`f32`/`f64`) edge.
    pub(crate) fn into_erased(self) -> ErasedTensor {
        match self.0.downcast::<TensorNode<f32>>() {
            Ok(node) => ErasedTensor::F32(GraphTensor { node }),
            Err(rc) => match rc.downcast::<TensorNode<f64>>() {
                Ok(node) => ErasedTensor::F64(GraphTensor { node }),
                Err(_) => unreachable!("autograd edges only exist for f32/f64 nodes"),
            },
        }
    }
}

/// Compile-time bridge between a concrete float and its erased representation.
///
/// Crate-internal: it exists only because `Float` is sealed to `f32`/`f64` while
/// the erasure enums are open to the compiler, which cannot prove the two sets
/// coincide. The panic arms are therefore unreachable by construction.
pub(crate) trait FloatRepr: Float {
    fn into_erased(g: GraphTensor<Self>) -> ErasedTensor;
    fn take_erased(e: ErasedTensor) -> GraphTensor<Self>;
    fn into_grad_value(g: GraphTensor<Self>) -> GradValue;
    fn ref_grad_value(g: &GradValue) -> &GraphTensor<Self>;
    /// Non-panicking lookup: `None` when the slot holds another float.
    fn try_ref_grad_value(g: &GradValue) -> Option<&GraphTensor<Self>>;
}

macro_rules! impl_float_repr {
    ($t:ty, $variant:ident) => {
        impl FloatRepr for $t {
            fn into_erased(g: GraphTensor<Self>) -> ErasedTensor {
                ErasedTensor::$variant(g)
            }
            fn take_erased(e: ErasedTensor) -> GraphTensor<Self> {
                match e {
                    ErasedTensor::$variant(g) => g,
                    _ => panic!("autograd dtype mismatch: expected {}", stringify!($t)),
                }
            }
            fn into_grad_value(g: GraphTensor<Self>) -> GradValue {
                GradValue::$variant(g)
            }
            fn ref_grad_value(g: &GradValue) -> &GraphTensor<Self> {
                match g {
                    GradValue::$variant(g) => g,
                    _ => panic!("autograd dtype mismatch: expected {}", stringify!($t)),
                }
            }
            fn try_ref_grad_value(g: &GradValue) -> Option<&GraphTensor<Self>> {
                match g {
                    GradValue::$variant(g) => Some(g),
                    _ => None,
                }
            }
        }
    };
}

impl_float_repr!(f32, F32);
impl_float_repr!(f64, F64);
