use std::rc::Rc;

use crate::core::GraphTensor;

use crate::core::autograd::ops::math::{BackwardMatmul, MatmulOp};
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
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDim {
                    operands,
                    op: SumDimOp { dim, original_times: self.shape()[dim] },
                }) as Box<dyn GradFnTrait>
            }),
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

    pub fn argmax(
        &self,
        dim: usize,
    ) -> GraphTensor {

        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::argmax(ops[0], dim),
            None::<fn([GraphTensor; 1]) -> Box<dyn GradFnTrait>>,
            &[self],
        )
    }

    // TODO should I kep all the logic here or place the TensorStorage part in that class?
    pub fn one_hot(
        &self,
        num_classes: usize // TODO is this the best integer type here?
    ) -> GraphTensor {
        let in_shape = self.shape();
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

    pub fn matmul(
        a: &GraphTensor,
        b: &GraphTensor
    ) -> GraphTensor {
        let a_shape = a.shape();
        let b_shape = b.shape();

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        if !((a_ndim == 1 || a_ndim == 2) && (b_ndim == 1 || b_ndim == 2)) {
            panic!("matmul requires 1D or 2D tensors, got {}D and {}D", a_ndim, b_ndim);
        }

        // Convert 1D inputs to 2D views: a [K] -> [1,K], b [K] -> [K,1]
        let a2 = if a_ndim == 1 { a.unsqueeze(0) } else { a.copy_s() };
        let b2 = if b_ndim == 1 { b.unsqueeze(1) } else { b.copy_s() };

        let a2_shape = a2.shape(); // [m, k]
        let b2_shape = b2.shape(); // [k, n]

        let k = a2_shape[1];
        let kb = b2_shape[0];

        if k != kb {
            panic!("matmul inner dimensions must match ({} != {})", k, kb);
        }

        // Direct [m,k] x [k,n] -> [m,n] kernel.
        let out_store = TensorStorage::matmul(&a2.node.storage, &b2.node.storage);

        // Only attach a grad_fn if at least one operand requires gradients.
        let requires_grad = a.requires_grad() || b.requires_grad();
        let grad_fn = requires_grad.then(|| {
            Box::new(BackwardMatmul {
                operands: [a.copy_s(), b.copy_s()],
                op: MatmulOp {},
            }) as Box<dyn GradFnTrait>
        });

        let out_node = TensorNode {
            storage: out_store,
            requires_grad,
            grad_fn,
        };

        GraphTensor { node: Rc::new(out_node) }
    }
}
