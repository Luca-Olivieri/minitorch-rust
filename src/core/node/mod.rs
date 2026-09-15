use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::dtype::Dtype;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub(crate) struct TensorNode<T: Dtype = f64> {
    pub(crate) storage: TensorStorage<T>,
    pub(crate) requires_grad: bool,
    pub(crate) grad_fn: Option<Box<dyn GradFnTrait<T>>>,
}

impl<T: Dtype> TensorNode<T> {
    pub(crate) fn new(shape: Vec<usize>, fill_value: T, requires_grad: bool) -> Self {
        Self {
            storage: TensorStorage::new(shape, fill_value),
            requires_grad,
            grad_fn: None,
        }
    }

    pub(crate) fn from_storage(storage: TensorStorage<T>, requires_grad: bool) -> Self {
        Self {
            storage,
            requires_grad,
            grad_fn: None,
        }
    }
}
