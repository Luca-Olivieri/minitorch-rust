use rand::rngs::StdRng;

use crate::core::{
    GraphTensor,
    nn::module::{Forward1, Module},
};

pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
}

impl Module for Linear {
    fn for_each_own_param(&self, f: &mut dyn FnMut(&str, &GraphTensor)) {
        f("weight", &self.weight);
        if let Some(bias) = &self.bias {
            f("bias", bias);
        }
    }

    fn for_each_own_param_mut(&mut self, f: &mut dyn FnMut(&str, &mut GraphTensor)) {
        f("weight", &mut self.weight);
        if let Some(bias) = &mut self.bias {
            f("bias", bias);
        }
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool, rng: StdRng) -> Self {
        // Xavier/Glorot uniform initialization to break symmetry between units
        let w_shape = vec![in_features, out_features];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng); // TODO set rng

        let bias = if has_bias {
            Some(GraphTensor::new(vec![out_features], 0.0, true))
        } else {
            None
        };

        Self { weight, bias }
    }
}

impl Forward1 for Linear {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let mult = GraphTensor::matmul(input, &self.weight);

        // `b` has shape [out_features], `mult` [batch, out_features]: the `+`
        // broadcasts the bias across the batch dim automatically.
        match &self.bias {
            None => mult,
            Some(b) => &mult + b,
        }
    }
}
