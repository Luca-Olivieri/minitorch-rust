use std::collections::HashMap;

use crate::core::{GraphTensor, tensor::AbstractTensor};

// TODO should Module have both parameters AND modules (not just one of those at a time)?

pub trait Module {

    fn set_requires_grad(
        &mut self,
        requires_grad: bool,
        recursive: bool
    ) {
        for p in self.params_mut().values_mut() {
            p.set_requires_grad(requires_grad);
        }

        if recursive {
            for box_m in self.modules_mut().values_mut() {
                box_m.set_requires_grad(requires_grad, true);
            }
        }
    }

    fn all_params(&self) -> HashMap<String, &GraphTensor> {
        let mut out_map = HashMap::new();

        // add this module's parameters first
        for (p_name, p_tensor) in &self.params() {
            out_map.insert(String::from(p_name), *p_tensor);
        }

        // then recursively add child modules' parameters
        for (child_mod_name, child_mod) in &self.modules() {

            let child_params = child_mod.all_params();
            for (c_mod_name, c_box_mod) in child_params {
                let mut full_name = String::from(child_mod_name);
                if !c_mod_name.is_empty() {
                    full_name.push_str(".");
                    full_name.push_str(&c_mod_name);
                }
                out_map.insert(full_name, c_box_mod);
            }
        }

        out_map
    }



    fn all_params_mut(&mut self) -> HashMap<String, &mut GraphTensor> {
        let mut out_map = HashMap::new();
        let (params, modules) = self.parts_mut();
        for (p_name, p_tensor) in params {
            out_map.insert(p_name, p_tensor);
        }
        for (mod_name, box_mod) in modules {
            for (c_name, c_tensor) in box_mod.all_params_mut() {
                let mut full_name = mod_name.clone();
                if !c_name.is_empty() {
                    full_name.push('.');
                    full_name.push_str(&c_name);
                }
                out_map.insert(full_name, c_tensor);
            }
        }
        out_map
    }

    // To implement for each impl struct

    fn params(
        &self
    ) -> HashMap<String, &GraphTensor> { HashMap::new() }

    fn params_mut(
        &mut self
    ) -> HashMap<String, &mut GraphTensor> { HashMap::new() }

    fn modules(
        &self
    ) -> HashMap<String, & dyn Module> { HashMap::new() }

    fn modules_mut(
        &mut self
    ) -> HashMap<String, &mut dyn Module> { HashMap::new() }

    fn parts_mut(&mut self) -> (HashMap<String, &mut GraphTensor>, HashMap<String, &mut dyn Module>) {
        (HashMap::new(), HashMap::new())
    }
}

pub trait Forward1 {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor;
}
