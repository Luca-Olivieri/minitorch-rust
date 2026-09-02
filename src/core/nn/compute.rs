use rand::rngs::StdRng;

use crate::core::{GraphTensor, nn::module::{Forward1, Module, ModuleBase}, tensor::AbstractTensor};

pub struct Linear {
    base: ModuleBase,
}

impl Module for Linear {
    fn base(&self) -> &ModuleBase { &self.base }
    fn base_mut(&mut self) -> &mut ModuleBase { &mut self.base }
}

impl Linear {

    const WEIGHT: &str = "weight";
    const BIAS: &str = "bias";

    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        rng: StdRng
    ) -> Self {

        let mut base = ModuleBase::new();

        // Xavier/Glorot uniform initialization to break symmetry between units
        let w_shape = vec![in_features, out_features];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng); // TODO set rng

        base.parameters.insert(String::from(Self::WEIGHT), weight);

        if has_bias {
            let b_shape = vec![out_features];
            let bias = GraphTensor::new(b_shape, 0.0, true);
            base.parameters.insert(String::from(Self::BIAS), bias);
        }

        Self { base }
    }
}

impl Forward1 for Linear {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let weight = *self.parameters().get(Self::WEIGHT).unwrap();
        let bias = *self.parameters().get(Self::BIAS).unwrap(); // FIXME crashes if has_bias = false

        let mult = GraphTensor::matmul(input, weight);

        match input.shape().len()  {
           1 => &mult + bias,
           _ => &mult + &bias.unsqueeze(0).expand(0, input.shape()[0])
        }
    }
}
