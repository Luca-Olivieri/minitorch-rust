use crate::core::tensor_storage::TensorStorage;

#[derive(Debug)]
pub struct TensorNode {
    pub(crate) storage: TensorStorage,
//     std::unique_ptr<GradFn> m_grad_fn { nullptr };
//     std::shared_ptr<Tensor> m_grad { nullptr };
    pub(crate) requires_grad: bool
}

impl TensorNode {

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool
    ) -> Self {

        Self {
            storage: TensorStorage::new(shape, fill_value),
        //     m_grad_fn{ nullptr },
        //     m_grad{ nullptr },
            requires_grad
        }
    }
}
