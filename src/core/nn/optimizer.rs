use crate::core::{autograd::GradMap, nn::module::Module};

pub trait Optimizer {
    fn step(&self, model: &mut dyn Module, grads_map: &GradMap);
}

pub struct SGD {
    base_lr: f64,
}

impl SGD {
    pub fn new(base_lr: f64) -> Self {
        Self { base_lr }
    }
}

impl Optimizer for SGD {
    fn step(&self, model: &mut dyn Module, grads_map: &GradMap) {
        model.for_each_param_mut(&mut |_, param| {
            if let Some(g) = grads_map.get(param) {
                // param = param - lr * grad, fused in a single pass
                let updated = param.sub_scaled(g, self.base_lr).detach(true);
                *param = updated;
            }
        });
    }
}
