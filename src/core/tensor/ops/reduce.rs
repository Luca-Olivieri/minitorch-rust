use std::rc::Rc;

use crate::core::GraphTensor;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::math::{BackwardMatmul, MatmulOp};
use crate::core::autograd::ops::reduce::{BackwardMaxDims, BackwardSumDims, MaxDimsOp, SumDimsOp};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::storage::ops::reduce::normalize_dims;
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl GraphTensor {
    pub fn sum_dim(&self, dim: usize) -> GraphTensor {
        self.sum_dims(&[dim])
    }

    /// Sum over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent, e.g.
    /// `x.sum_dims(&[0, 2])`. An empty slice is a no-op copy.
    pub fn sum_dims(&self, dims: &[usize]) -> GraphTensor {
        let dims = normalize_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::sum_dims(ops[0], &dims),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDims {
                    operands,
                    op: SumDimsOp { dims: dims.clone() },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn sum(&self) -> GraphTensor {
        let dims: Vec<usize> = (0..self.shape().len()).collect();
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::sum_all(ops[0]),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDims {
                    operands,
                    op: SumDimsOp { dims },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn mean_dim(&self, dim: usize) -> GraphTensor {
        &self.sum_dim(dim) / (self.shape()[dim] as f64)
    }

    /// Mean over every dimension in `dims` at once.
    pub fn mean_dims(&self, dims: &[usize]) -> GraphTensor {
        let dims = normalize_dims(dims, self.shape());
        let count: usize = dims.iter().map(|&d| self.shape()[d]).product();
        &self.sum_dims(&dims) / (count as f64)
    }

    pub fn mean(&self) -> GraphTensor {
        &self.sum() / (self.numel() as f64)
    }

    pub fn max_dim(&self, dim: usize) -> GraphTensor {
        self.max_dims(&[dim])
    }

    /// Max over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent. Gradient
    /// flows to every element that attains the per-slice maximum (like
    /// `torch.amax`).
    pub fn max_dims(&self, dims: &[usize]) -> GraphTensor {
        let dims = normalize_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::max_dims(ops[0], &dims),
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardMaxDims {
                    operands,
                    op: MaxDimsOp { dims: dims.clone() },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn argmax(&self, dim: usize) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| TensorStorage::argmax(ops[0], dim),
            None::<fn([GraphTensor; 1]) -> Box<dyn GradFnTrait>>,
            &[self],
        )
    }

    // TODO should I kep all the logic here or place the TensorStorage part in that class?
    pub fn one_hot(
        &self,
        num_classes: usize, // TODO is this the best integer type here?
    ) -> GraphTensor {
        let in_shape = self.shape();
        let in_numel = self.numel();

        // Build output shape by appending classes as the last dimension
        let mut out_shape = (*in_shape).clone();
        out_shape.push(num_classes);

        let mut out_storage = TensorStorage::new(out_shape, 0.0); // TODO see if you can have this uninit

        // `out_storage` is freshly allocated, so its Rc is unique and mutable.
        let out_buf = out_storage.buffer_mut();

        // The output is contiguous with the input coords followed by the class dim,
        // so the flat output index of (input logical index i, class cls) is i*num_classes + cls.
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
                panic!(
                    "One-hotting with num_classes={} but tensor has value {} at index {}",
                    num_classes - 1,
                    raw_value,
                    i
                )
            }

            out_buf[i * num_classes + cls] = 1.0;
        }

        let out_node = TensorNode {
            storage: out_storage,
            requires_grad: false,
            grad_fn: None,
        };

        Self {
            node: Rc::new(out_node),
        }
    }

    pub fn matmul(a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let a_shape = a.shape();
        let b_shape = b.shape();

        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        if !((a_ndim == 1 || a_ndim == 2) && (b_ndim == 1 || b_ndim == 2)) {
            panic!(
                "matmul requires 1D or 2D tensors, got {}D and {}D",
                a_ndim, b_ndim
            );
        }

        // Convert 1D inputs to 2D views: a [K] -> [1,K], b [K] -> [K,1]
        let a2 = if a_ndim == 1 {
            a.unsqueeze(0)
        } else {
            a.copy_s()
        };
        let b2 = if b_ndim == 1 {
            b.unsqueeze(1)
        } else {
            b.copy_s()
        };

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

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // shape [2, 2, 3] with values i*6 + j*3 + k
    fn tensor() -> GraphTensor {
        GraphTensor::wrap(
            vec![
                vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]],
                vec![vec![6.0, 7.0, 8.0], vec![9.0, 10.0, 11.0]],
            ],
            false,
        )
    }

    #[test]
    fn sum_over_non_adjacent_dims() {
        let out = tensor().sum_dims(&[0, 2]);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 24.0);
        assert_eq!(*out.at(&vec![1]), 42.0);
    }

    #[test]
    fn sum_over_adjacent_dims() {
        let out = tensor().sum_dims(&[1, 2]);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 15.0);
        assert_eq!(*out.at(&vec![1]), 51.0);
    }

    #[test]
    fn sum_over_empty_dims_is_noop() {
        let out = tensor().sum_dims(&[]);
        assert_eq!(out.shape(), &vec![2, 2, 3]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(*out.at(&vec![i, j, k]), (i * 6 + j * 3 + k) as f64);
                }
            }
        }
    }

    #[test]
    fn sum_dim_matches_sum_dims_single() {
        let a = tensor().sum_dim(1);
        let b = tensor().sum_dims(&[1]);
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.shape(), &vec![2, 3]);
        assert_eq!(a.numel(), b.numel());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*a.at(&vec![i, j]), *b.at(&vec![i, j]));
            }
        }
    }

    #[test]
    fn sum_all_total() {
        let out = tensor().sum();
        assert_eq!(out.shape(), &vec![]);
        assert_eq!(*out.at(&vec![]), 66.0);
    }

    #[test]
    fn mean_over_non_adjacent_dims() {
        let out = tensor().mean_dims(&[0, 2]);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 4.0);
        assert_eq!(*out.at(&vec![1]), 7.0);
    }

    #[test]
    fn max_over_non_adjacent_dims() {
        let out = tensor().max_dims(&[0, 2]);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 8.0);
        assert_eq!(*out.at(&vec![1]), 11.0);
    }

    #[test]
    fn max_over_adjacent_dims() {
        let out = tensor().max_dims(&[1, 2]);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 5.0);
        assert_eq!(*out.at(&vec![1]), 11.0);
    }

    #[test]
    fn sum_backward_broadcasts_grad_to_all_reduced_dims() {
        let mut a = tensor();
        a.set_requires_grad(true);
        let out = a.sum_dims(&[0, 2]);

        let grads = out.backward(true);
        let da = grads.get(&a.to_key()).unwrap();
        assert_eq!(da.shape(), &vec![2, 2, 3]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(*da.at(&vec![i, j, k]), 1.0);
                }
            }
        }
    }

    #[test]
    fn max_backward_flows_only_to_maxima() {
        let mut a = tensor();
        a.set_requires_grad(true);
        let out = a.max_dims(&[0, 2]);

        let grads = out.backward(true);
        let da = grads.get(&a.to_key()).unwrap();
        assert_eq!(da.shape(), &vec![2, 2, 3]);

        // max over {0, 2} per j: 8 for j=0 (at i=1,k=2), 11 for j=1 (at i=1,k=2)
        let expected = vec![
            vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]],
        ];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(*da.at(&vec![i, j, k]), expected[i][j][k]);
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn duplicate_dims_panic() {
        tensor().sum_dims(&[0, 0]);
    }

    #[test]
    #[should_panic]
    fn out_of_range_dim_panics() {
        tensor().max_dims(&[3]);
    }
}
