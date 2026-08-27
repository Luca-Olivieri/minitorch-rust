use crate::core::storage::TensorStorage;
use crate::core::autograd::grad_fn::GradFnTrait;

#[derive(Debug)]
pub struct TensorNode {
    pub(crate) storage: TensorStorage,
    pub(crate) requires_grad: bool,
    pub(super) grad_fn: Option<Box<dyn GradFnTrait>>
}

impl TensorNode {

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool
    ) -> Self {

        Self {
            storage: TensorStorage::new(shape, fill_value),
            requires_grad,
            grad_fn: None
        }
    }
}
