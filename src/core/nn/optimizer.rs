use std::collections::HashMap;

use crate::core::{GraphTensor, autograd::TensorKey};

pub trait Optimizer {

    fn step(
        &self,
        params: HashMap<String, &mut GraphTensor>,
        grads_map: &HashMap<TensorKey, GraphTensor>
    );
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
        params: HashMap<String, &mut GraphTensor>,
        grads_map: &HashMap<TensorKey, GraphTensor>
    ) {
        for param in params.into_values() {
            if let Some(g) = grads_map.get(&param.to_key()) {
                let delta = &(g * self.base_lr);
                *param = (&*param - delta).detach(true);
            }
        }
    }
}
