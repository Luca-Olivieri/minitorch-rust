use std::collections::HashMap;

use rand::rngs::StdRng;

use crate::core::{
    nn::module::{Forward1, Module},
    GraphTensor,
};

pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
}

impl Module for Linear {
    fn params(&self) -> HashMap<String, &GraphTensor> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("weight"), &self.weight);
        if let Some(bias) = &self.bias {
            out_map.insert(String::from("bias"), bias);
        }

        out_map
    }

    fn params_mut(&mut self) -> HashMap<String, &mut GraphTensor> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("weight"), &mut self.weight);
        if let Some(bias) = &mut self.bias {
            out_map.insert(String::from("bias"), bias);
        }

        out_map
    }

    fn parts_mut(
        &mut self,
    ) -> (
        HashMap<String, &mut GraphTensor>,
        HashMap<String, &mut dyn Module>,
    ) {
        let mut out_params_map = HashMap::new();
        out_params_map.insert(String::from("weight"), &mut self.weight);
        if let Some(bias) = &mut self.bias {
            out_params_map.insert(String::from("bias"), bias);
        }

        (out_params_map, HashMap::new())
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
