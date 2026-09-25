use std::time::{Duration, Instant};

use crate::core::{
    GraphTensor,
    nn::module::{Forward1, Layer, Module},
};

/// A dynamically dispatched sequence of forward modules.
///
/// Each child must implement both [`Forward1`] and [`Module`], so the container
/// can both run a forward pass and expose the children to the module registry
/// and optimizers. Child names are their zero-based positions, giving paths
/// such as `0.weight` and nested paths such as `0.1.weight`.
pub struct DynSequential {
    layers: Vec<Box<dyn Layer>>,
}

impl DynSequential {
    /// Create an empty sequence. An empty sequence is an identity transform.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Create a sequence from already-boxed dynamic layers.
    pub fn from_layers(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    /// Append a layer using the consuming builder style.
    pub fn with<L>(mut self, layer: L) -> Self
    where
        L: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Default for DynSequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for DynSequential {
    fn for_each_own_module(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        for (index, layer) in self.layers.iter().enumerate() {
            f(&index.to_string(), layer.as_ref());
        }
    }

    fn for_each_own_module_mut(&mut self, f: &mut dyn FnMut(&str, &mut dyn Module)) {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            f(&index.to_string(), layer.as_mut());
        }
    }

    fn module(&self, name: &str) -> Option<&dyn Module> {
        let index = name.parse::<usize>().ok()?;
        self.layers
            .get(index)
            .map(|layer| layer.as_ref() as &dyn Module)
    }

    fn module_mut(&mut self, name: &str) -> Option<&mut dyn Module> {
        let index = name.parse::<usize>().ok()?;
        self.layers
            .get_mut(index)
            .map(|layer| layer.as_mut() as &mut dyn Module)
    }
}

impl Forward1 for DynSequential {
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        let Some(first) = self.layers.first() else {
            return if no_grad {
                input.detach(false)
            } else {
                input.copy_s()
            };
        };

        let mut output = first.forward(input, no_grad);
        for layer in self.layers.iter().skip(1) {
            output = layer.forward(&output, no_grad);
        }
        output
    }
}

impl DynSequential {
    /// Run the sequence and return the elapsed time for each child layer.
    ///
    /// This is intended for opt-in profiling. The normal [`Forward1::forward`]
    /// path remains timer-free.
    pub fn forward_with_timings(
        &self,
        input: &GraphTensor,
        no_grad: bool,
    ) -> (GraphTensor, Vec<Duration>) {
        let Some(first) = self.layers.first() else {
            let output = if no_grad {
                input.detach(false)
            } else {
                input.copy_s()
            };
            return (output, Vec::new());
        };

        let mut timings = Vec::with_capacity(self.layers.len());
        let start = Instant::now();
        let mut output = first.forward(input, no_grad);
        timings.push(start.elapsed());

        for layer in self.layers.iter().skip(1) {
            let start = Instant::now();
            output = layer.forward(&output, no_grad);
            timings.push(start.elapsed());
        }

        (output, timings)
    }
}
