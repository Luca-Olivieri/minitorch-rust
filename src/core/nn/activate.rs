use crate::core::{GraphTensor, nn::module::{Forward1, Module, ModuleBase}, tensor::AbstractTensor};

pub struct ReLU {
    base: ModuleBase,
}

impl Module for ReLU {
    fn base(&self) -> &ModuleBase { &self.base }
    fn base_mut(&mut self) -> &mut ModuleBase { &mut self.base }
}

impl ReLU {

    pub fn new() -> Self {
        Self { base: ModuleBase::new() }
    }
}

impl Forward1 for ReLU {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let ones = GraphTensor::new(input.shape().clone(), 0.0, false); // TODO substitute with tensor broadcasted from f64 value of 1.0
        GraphTensor::maximum(input, &ones)
    }
}
