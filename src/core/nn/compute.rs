use rand::rngs::StdRng;

use crate::core::{GraphTensor, nn::module::Forward1};
use crate::module;

module! {
    Linear {
        params {
            weight,
        },
        optional_params {
            bias,
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
