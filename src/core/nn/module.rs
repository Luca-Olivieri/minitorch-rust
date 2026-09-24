use crate::core::GraphTensor;

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

    // TODO right now, the param is detached, bumping the RC of the buffer, then bringing it back since the old param si discarded
    // is there a way (that makes sense) to express this with move semantics?

    fn set_requires_grad(&mut self, requires_grad: bool, recursive: bool) {
        // Params are immutable GraphTensors, so "changing" the flag rebinds the
        // slot to a fresh detached leaf rather than mutating the shared node.
        // This rebind changes node identity, orphaning any graph already built on
        // the old node, so it is only permitted on a module that no operation has
        // touched yet.
        self.assert_not_captured(recursive);

        if recursive {
            self.for_each_param_mut(&mut |_, param| *param = param.detach(requires_grad));
        } else {
            self.for_each_own_param_mut(&mut |_, param| *param = param.detach(requires_grad));
        }
    }

    /// Propagate a training/evaluation mode change to child modules.
    ///
    /// Stateless modules use this default unchanged. Stateful modules such as
    /// Dropout override it to update their local behavior flag.
    fn set_training(&mut self, training: bool) {
        self.for_each_own_module_mut(&mut |_, child| {
            child.set_training(training);
        });
    }

    /// Business rule: a module's params may be (re)frozen only while none of
    /// them has been captured into a graph. The first op that uses a param
    /// snapshots it via `copy_s`, so "captured" is exactly "the module is no
    /// longer the sole reference" to the param's node.
    fn assert_not_captured(&self, recursive: bool) {
        let check = &mut |name: &str, param: &GraphTensor| {
            // TODO: It's a runtime/panic rule, not compile-time. If you want misuses impossible at compile time,
            // the alternative is a type-state split — e.g. XORClassifier in a "configurable" state exposing set_requires_grad,
            // plus a consuming .activate()/.build() that returns the ready wrapper which alone implements Forward1.
            // Stronger, but a bigger refactor across the module! macro, Forward1, and Optimizer::step(&mut dyn Module) plumbing.
            assert!(
                param.is_unique_ref(),
                "Module::set_requires_grad: param `{name}` has already been captured in a \
                 computation graph. Freeze parameters before building the graph (e.g. before \
                 the first `forward`)."
            );
        };

        if recursive {
            self.for_each_param(check);
        } else {
            self.for_each_own_param(check);
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

/// A forward-capable module that can be stored in a dynamic container.
///
/// This is intentionally separate from [`Module`]: not every module needs a
/// `forward` implementation, while a heterogeneous sequence must require both
/// capabilities from each child.
pub trait Layer: Forward1 + Module {}

impl<T> Layer for T where T: Forward1 + Module {}
