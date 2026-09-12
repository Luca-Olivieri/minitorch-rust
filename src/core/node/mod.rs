use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub struct TensorNode {
    pub storage: TensorStorage,
    pub(crate) requires_grad: bool,
    pub(super) grad_fn: Option<Box<dyn GradFnTrait>>,
}

impl TensorNode {
    pub fn new(shape: Vec<usize>, fill_value: f64, requires_grad: bool) -> Self {
        Self {
            storage: TensorStorage::new(shape, fill_value),
            requires_grad,
            grad_fn: None,
        }
    }

    pub fn from_storage(storage: TensorStorage, requires_grad: bool) -> Self {
        Self {
            storage,
            requires_grad,
            grad_fn: None,
        }
    }
}
