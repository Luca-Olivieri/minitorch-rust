use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub(crate) struct TensorNode {
    pub(crate) storage: TensorStorage,
    pub(crate) requires_grad: bool,
    pub(crate) grad_fn: Option<Box<dyn GradFnTrait>>,
}

impl TensorNode {
    pub(crate) fn new(shape: Vec<usize>, fill_value: f64, requires_grad: bool) -> Self {
        Self {
            storage: TensorStorage::new(shape, fill_value),
            requires_grad,
            grad_fn: None,
        }
    }

    pub(crate) fn from_storage(storage: TensorStorage, requires_grad: bool) -> Self {
        Self {
            storage,
            requires_grad,
            grad_fn: None,
        }
    }
}
