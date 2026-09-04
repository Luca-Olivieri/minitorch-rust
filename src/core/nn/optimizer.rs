use crate::core::{GraphTensor};

pub trait Optimizer {

    fn step(
        &self,
        tensor: &GraphTensor,
        grad: &GraphTensor,
    ) -> GraphTensor;
}

pub struct SGD {
    base_lr: f64
}

impl SGD {
    pub fn new(base_lr: f64) -> Self {
        Self { base_lr }
    }
}

impl Optimizer for SGD {

    fn step(
        &self,
        tensor: &GraphTensor,
        grad: &GraphTensor,
    ) -> GraphTensor {
        tensor - &(grad * self.base_lr)
    }
}

// void Optimizer::zero_grad() {
//     for (auto& [name, tensor] : m_parameters) {
//         if (tensor.m_node->m_grad) {
//             tensor.reset_grad();
//         }
//     }
// }
