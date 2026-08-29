use std::os::raw;
use std::rc::Rc;

use crate::core::GraphTensor;

use crate::core::autograd::ops::reduce::{BackwardSumDim, SumDimOp};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::ops::math::apply_tensor_op;
use crate::core::autograd::grad_fn::GradFnTrait;

impl GraphTensor {

    pub fn sum_dim(
        &self,
        dim: usize,
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::sum_dim(ops[0], dim),
            |operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDim{operands: operands, op: SumDimOp { dim, original_times: self.shape()[dim]}}) as Box<dyn GradFnTrait>
            },
            &[self],
        )
    }

    pub fn sum(
        &self,
    ) -> GraphTensor {

        let mut out = self.copy_s();

        for _ in 0..self.shape().len() {
            out = out.sum_dim(0);
        }

        out
    }

    pub fn mean_dim(
        &self,
        dim: usize,
    ) -> GraphTensor {

        &self.sum_dim(dim) / (self.shape()[dim] as f64)
    }

    pub fn mean(
        &self
    ) -> GraphTensor {

        let mut out = self.copy_s();

        for _ in 0..self.shape().len() {
            out = out.mean_dim(0);
        }

        out
    }

    // TODO should I kep all the logic here or place the TensorStorage part in that class?
    pub fn one_hot(
        &self,
        num_classes: usize // TODO is this the best integer type here?
    ) -> GraphTensor {
        let in_shape = &self.shape();
        let in_numel = self.numel();

        // Build output shape by appending classes as the last dimension
        let mut out_shape = (*in_shape).clone();
        out_shape.push(num_classes);

        let mut out_storage = TensorStorage::new(out_shape, 0.0); // TODO see if you can have this uninit

        // Helper: convert logical index -> multi-dim coordinates for input
        for i in 0..in_numel {
            let raw_value = self.node.storage[i];
            if raw_value.fract() != 0.0 {
                panic!("One-hotted tensor has value {raw_value} with fractional part at index {i}.")
            }

            if raw_value < 0.0 {
                panic!("One-hotted tensor has negative value {raw_value} at index {i}.")
            }

            let cls = raw_value as usize; // TODO is this the best integer type here?

            if cls >= num_classes {
                panic!("One-hotting with num_classes={} but tensor has value {} at index {}", num_classes-1, raw_value, i)
            }

            if in_shape.is_empty() {
                // Scalar input -> output is 1D of length num_classes
                let out_md = vec![cls];
                out_storage[&out_md] = 1.0;
                continue;
            }

            // compute multi-dim coords for the input logical index
            let mut in_md = vec![0; in_shape.len()]; // TODO I would like to use with_capacity() here but the reverse() gives mi problems
            let mut curr = i;
            for d in (0..=in_shape.len()-1).rev() {
                in_md[d] = curr % in_shape[d];
                curr /= in_shape[d];
            }

            // append class dim
            in_md.push(cls);
            out_storage[&in_md] = 1.0;
        }

        let out_node = TensorNode {
            storage: out_storage,
            requires_grad: false,
            grad_fn: None,
        };

        Self { node: Rc::new(out_node) }
    }
}
