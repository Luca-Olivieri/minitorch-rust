use crate::core::{GraphTensor, tensor::AbstractTensor};

pub trait Module {
    fn for_each_param(&self, f: &mut dyn FnMut(&str, &GraphTensor)) {
        self.for_each_own_param(f);
        self.for_each_own_module(&mut |name, child| {
            child.for_each_param(&mut |param_name, param| {
                f(&join_path(name, param_name), param);
            });
        });
    }

    fn for_each_param_mut(&mut self, f: &mut dyn FnMut(&str, &mut GraphTensor)) {
        self.for_each_own_param_mut(f);
        self.for_each_own_module_mut(&mut |name, child| {
            child.for_each_param_mut(&mut |param_name, param| {
                f(&join_path(name, param_name), param);
            });
        });
    }

    fn for_each_module(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        self.for_each_own_module(&mut |name, child| {
            f(name, child);
            child.for_each_module(&mut |sub_name, sub_child| {
                f(&join_path(name, sub_name), sub_child);
            });
        });
    }

    fn for_each_module_mut(&mut self, f: &mut dyn FnMut(&str, &mut dyn Module)) {
        self.for_each_own_module_mut(&mut |name, child| {
            f(name, child);
            child.for_each_module_mut(&mut |sub_name, sub_child| {
                f(&join_path(name, sub_name), sub_child);
            });
        });
    }

    fn set_requires_grad(&mut self, requires_grad: bool, recursive: bool) {
        if recursive {
            self.for_each_param_mut(&mut |_, param| param.set_requires_grad(requires_grad));
        } else {
            self.for_each_own_param_mut(&mut |_, param| {
                param.set_requires_grad(requires_grad);
            });
        }
    }

    // To implement for each impl struct. These visit the module's own
    // parameters / submodules only, without recursing.

    fn for_each_own_param(&self, _f: &mut dyn FnMut(&str, &GraphTensor)) {}

    fn for_each_own_param_mut(&mut self, _f: &mut dyn FnMut(&str, &mut GraphTensor)) {}

    fn for_each_own_module(&self, _f: &mut dyn FnMut(&str, &dyn Module)) {}

    fn for_each_own_module_mut(&mut self, _f: &mut dyn FnMut(&str, &mut dyn Module)) {}

    // Dynamic lookup of directly-registered submodules by name. Overridden by
    // macro-generated container impls; leaf modules have no submodules.

    fn module(&self, _name: &str) -> Option<&dyn Module> {
        None
    }

    fn module_mut(&mut self, _name: &str) -> Option<&mut dyn Module> {
        None
    }

    // Dotted-path lookup across nested containers, e.g. "encoder.block.0".

    fn module_path(&self, path: &str) -> Option<&dyn Module> {
        match path.split_once('.') {
            Some((head, rest)) => self.module(head)?.module_path(rest),
            None => self.module(path),
        }
    }

    fn module_path_mut(&mut self, path: &str) -> Option<&mut dyn Module> {
        match path.split_once('.') {
            Some((head, rest)) => self.module_mut(head)?.module_path_mut(rest),
            None => self.module_mut(path),
        }
    }

    fn param(&self, _name: &str) -> Option<&GraphTensor> {
        None
    }

    fn param_mut(&mut self, _name: &str) -> Option<&mut GraphTensor> {
        None
    }

    fn param_path(&self, path: &str) -> Option<&GraphTensor> {
        match path.split_once('.') {
            Some((head, rest)) => self.module(head)?.param_path(rest),
            None => self.param(path),
        }
    }

    fn param_path_mut(&mut self, path: &str) -> Option<&mut GraphTensor> {
        match path.split_once('.') {
            Some((head, rest)) => self.module_mut(head)?.param_path_mut(rest),
            None => self.param_mut(path),
        }
    }
}

fn join_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else if name.is_empty() {
        parent.to_string()
    } else {
        format!("{parent}.{name}")
    }
}

pub trait Forward1 {
    fn forward(&self, input: &GraphTensor) -> GraphTensor;
}
