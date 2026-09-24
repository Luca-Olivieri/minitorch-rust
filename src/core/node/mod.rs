use crate::core::autograd::grad_fn::BackwardSource;
use crate::core::dtype::Dtype;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub(crate) struct TensorNode<T: Dtype = f64> {
    pub(crate) storage: TensorStorage<T>,
    pub(crate) requires_grad: bool,
    pub(crate) no_grad: bool,
    // Deferred gradient source, dtype-erased so an edge may connect operands of
    // different float dtypes (differentiable casts, Stage 6c). The rule is
    // materialized (BackwardSource -> NBackwardOp) only when a backward pass
    // actually runs.
    pub(crate) grad_fn: Option<Box<BackwardSource>>,
}

impl<T: Dtype> TensorNode<T> {
    pub(crate) fn new(shape: Vec<usize>, fill_value: T, requires_grad: bool) -> Self {
        Self {
            storage: TensorStorage::new(shape, fill_value),
            requires_grad,
            no_grad: false,
            grad_fn: None,
        }
    }

    pub(crate) fn from_storage(storage: TensorStorage<T>, requires_grad: bool) -> Self {
        Self {
            storage,
            requires_grad,
            no_grad: false,
            grad_fn: None,
        }
    }
}
