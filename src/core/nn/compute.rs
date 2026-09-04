use std::collections::HashMap;

use rand::rngs::StdRng;

use crate::core::{GraphTensor, nn::module::{Forward1, Module}, tensor::AbstractTensor};

pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>
}

impl Module for Linear {

    fn params(
        &self
    ) -> HashMap<String, &GraphTensor> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("weight"), &self.weight);
        if self.bias.is_none() {
            out_map.insert(String::from("bias"), self.bias.as_ref().unwrap());
        }

        out_map
    }

    fn params_mut(
        &mut self
    ) -> HashMap<String, &mut GraphTensor> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("weight"), &mut self.weight);
        if self.bias.is_none() {
            out_map.insert(String::from("bias"), self.bias.as_mut().unwrap());
        }

        out_map
    }
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        rng: StdRng
    ) -> Self {
        // Xavier/Glorot uniform initialization to break symmetry between units
        let w_shape = vec![in_features, out_features];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng); // TODO set rng

        let bias = if has_bias { Some(GraphTensor::new(vec![out_features], 0.0, true)) } else { None };

        Self { weight, bias }
    }
}

impl Forward1 for Linear {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let mult = GraphTensor::matmul(input, &self.weight);

        match &self.bias {
            None => mult,
            Some(b) => match input.shape().len()  {
               1 => &mult + b,
               _ => &mult + &b.unsqueeze(0).expand(0, input.shape()[0])
            }
        }
    }
}
