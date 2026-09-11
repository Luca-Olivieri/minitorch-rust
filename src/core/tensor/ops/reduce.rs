use std::rc::Rc;

use crate::core::GraphTensor;

use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::autograd::ops::math::{BackwardMatmul, MatmulOp};
use crate::core::autograd::ops::reduce::{BackwardMaxDims, BackwardSumDims, MaxDimsOp, SumDimsOp};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::storage::ops::reduce::resolve_dims;
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl GraphTensor {
    /// Sum over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent, e.g.
    /// `x.sum(&[0, 2])`. An empty slice aggregates over all dimensions, like
    /// PyTorch's `torch.sum(x, dim=None)`.
    ///
    /// With `keepdim` the reduced dimensions are retained as size 1 (so
    /// `x.sum(&[0, 2], true)` on `[2, 2, 3]` yields `[1, 2, 1]`), which keeps
    /// the result broadcastable against the input.
    pub fn sum(&self, dims: &[usize], keepdim: bool) -> GraphTensor {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| {
                let reduced = TensorStorage::sum(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardSumDims {
                    operands,
                    op: SumDimsOp {
                        dims: dims.clone(),
                        keepdim,
                    },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    /// Mean over every dimension in `dims` at once. An empty slice aggregates
    /// over all dimensions. See [`Self::sum`] for the `keepdim` semantics.
    pub fn mean(&self, dims: &[usize], keepdim: bool) -> GraphTensor {
        let dims = resolve_dims(dims, self.shape());
        let count: usize = dims.iter().map(|&d| self.shape()[d]).product();
        &self.sum(&dims, keepdim) / (count as f64)
    }

    /// Max over every dimension in `dims` at once.
    ///
    /// `dims` are indices into the input tensor and may be non-adjacent. An
    /// empty slice aggregates over all dimensions. Gradient flows to every
    /// element that attains the per-slice maximum (like `torch.amax`). See
    /// [`Self::sum`] for the `keepdim` semantics.
    pub fn max(&self, dims: &[usize], keepdim: bool) -> GraphTensor {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| {
                let reduced = TensorStorage::max(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(|operands: [GraphTensor; 1]| {
                Box::new(BackwardMaxDims {
                    operands,
                    op: MaxDimsOp {
                        dims: dims.clone(),
                        keepdim,
                    },
                }) as Box<dyn GradFnTrait>
            }),
            &[self],
        )
    }

    pub fn argmax(&self, dim: usize, keepdim: bool) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 1]| {
                let reduced = TensorStorage::argmax(ops[0], dim);
                if keepdim {
                    TensorStorage::unsqueeze(&reduced, dim)
                } else {
                    reduced
                }
            },
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
        let out = tensor().sum(&[0, 2], false);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 24.0);
        assert_eq!(*out.at(&vec![1]), 42.0);
    }

    #[test]
    fn sum_over_adjacent_dims() {
        let out = tensor().sum(&[1, 2], false);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 15.0);
        assert_eq!(*out.at(&vec![1]), 51.0);
    }

    #[test]
    fn sum_empty_dims_aggregates_over_all() {
        let out = tensor().sum(&[], false);
        assert_eq!(out.shape(), &vec![]);
        assert_eq!(*out.at(&vec![]), 66.0);

        // empty and explicit-all must agree
        let all = tensor().sum(&[0, 1, 2], false);
        assert_eq!(*out.at(&vec![]), *all.at(&vec![]));
    }

    #[test]
    fn sum_single_dim_uses_single_kernel() {
        // dispatch: single dim -> sum over dim 1 per-slice columns
        let a = tensor().sum(&[1], false);
        assert_eq!(a.shape(), &vec![2, 3]);
        assert_eq!(*a.at(&vec![0, 0]), 3.0);
        assert_eq!(*a.at(&vec![0, 1]), 5.0);
        assert_eq!(*a.at(&vec![0, 2]), 7.0);
        assert_eq!(*a.at(&vec![1, 0]), 15.0);
        assert_eq!(*a.at(&vec![1, 1]), 17.0);
        assert_eq!(*a.at(&vec![1, 2]), 19.0);

        // single-dim chains must equal the generic one-shot multi-dim reduction
        let one_shot = tensor().sum(&[0, 1, 2], false);
        let chained = tensor().sum(&[1], false).sum(&[0], false).sum(&[0], false);
        assert_eq!(*one_shot.at(&vec![]), *chained.at(&vec![]));
        assert_eq!(*one_shot.at(&vec![]), 66.0);
    }

    #[test]
    fn sum_all_total() {
        let out = tensor().sum(&[0, 1, 2], false);
        assert_eq!(out.shape(), &vec![]);
        assert_eq!(*out.at(&vec![]), 66.0);
    }

    #[test]
    fn sum_keepdim_keeps_reduced_axes() {
        let out = tensor().sum(&[0, 2], true);
        assert_eq!(out.shape(), &vec![1, 2, 1]);
        assert_eq!(*out.at(&vec![0, 0, 0]), 24.0);
        assert_eq!(*out.at(&vec![0, 1, 0]), 42.0);
    }

    #[test]
    fn sum_keepdim_empty_dims_keeps_all_axes() {
        let out = tensor().sum(&[], true);
        assert_eq!(out.shape(), &vec![1, 1, 1]);
        assert_eq!(*out.at(&vec![0, 0, 0]), 66.0);
    }

    #[test]
    fn mean_over_non_adjacent_dims() {
        let out = tensor().mean(&[0, 2], false);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 4.0);
        assert_eq!(*out.at(&vec![1]), 7.0);
    }

    #[test]
    fn mean_empty_dims_aggregates_over_all() {
        let out = tensor().mean(&[], false);
        assert_eq!(out.shape(), &vec![]);
        assert_eq!(*out.at(&vec![]), 5.5);
    }

    #[test]
    fn mean_keepdim_single_dim() {
        let out = tensor().mean(&[1], true);
        assert_eq!(out.shape(), &vec![2, 1, 3]);
        assert_eq!(*out.at(&vec![0, 0, 0]), 1.5);
        assert_eq!(*out.at(&vec![0, 0, 1]), 2.5);
        assert_eq!(*out.at(&vec![1, 0, 2]), 9.5);
    }

    #[test]
    fn max_over_non_adjacent_dims() {
        let out = tensor().max(&[0, 2], false);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 8.0);
        assert_eq!(*out.at(&vec![1]), 11.0);
    }

    #[test]
    fn max_over_adjacent_dims() {
        let out = tensor().max(&[1, 2], false);
        assert_eq!(out.shape(), &vec![2]);
        assert_eq!(*out.at(&vec![0]), 5.0);
        assert_eq!(*out.at(&vec![1]), 11.0);
    }

    #[test]
    fn max_empty_dims_aggregates_over_all() {
        let out = tensor().max(&[], false);
        assert_eq!(out.shape(), &vec![]);
        assert_eq!(*out.at(&vec![]), 11.0);
    }

    #[test]
    fn max_keepdim_keeps_reduced_axes() {
        let out = tensor().max(&[0, 2], true);
        assert_eq!(out.shape(), &vec![1, 2, 1]);
        assert_eq!(*out.at(&vec![0, 0, 0]), 8.0);
        assert_eq!(*out.at(&vec![0, 1, 0]), 11.0);
    }

    #[test]
    fn argmax_keepdim_keeps_axis() {
        let out = tensor().argmax(1, true);
        assert_eq!(out.shape(), &vec![2, 1, 3]);
        assert_eq!(*out.at(&vec![0, 0, 0]), 1.0);
        assert_eq!(*out.at(&vec![1, 0, 2]), 1.0);

        let squeezed = tensor().argmax(1, false);
        assert_eq!(squeezed.shape(), &vec![2, 3]);
        assert_eq!(*squeezed.at(&vec![0, 0]), 1.0);
    }

    #[test]
    fn sum_backward_broadcasts_grad_to_all_reduced_dims() {
        let mut a = tensor();
        a.set_requires_grad(true);
        let out = a.sum(&[0, 2], false);

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
    fn sum_backward_through_keepdim_broadcasts_to_input() {
        let mut a = tensor();
        a.set_requires_grad(true);
        // keepdim output [1,2,1] broadcasts against a [2,2,3] multipliers tensor
        let out = &a.sum(&[0, 2], true)
            * &GraphTensor::wrap(
                vec![
                    vec![vec![2.0, 2.0, 2.0], vec![3.0, 3.0, 3.0]],
                    vec![vec![5.0, 5.0, 5.0], vec![7.0, 7.0, 7.0]],
                ],
                false,
            );
        assert_eq!(out.shape(), &vec![2, 2, 3]);

        let grads = out.backward(true);
        let da = grads.get(&a.to_key()).unwrap();
        assert_eq!(da.shape(), &vec![2, 2, 3]);
        // `out = s * m` with s = keepdim sum over {0,2}, m the multipliers:
        // backprop through `*` sums m over the axes `s` lacks, so each
        // (i, j, k) receives the per-j aggregate Σ_{i,k} m[i,j,k] = 21 / 30.
        let expected = vec![
            vec![vec![21.0, 21.0, 21.0], vec![30.0, 30.0, 30.0]],
            vec![vec![21.0, 21.0, 21.0], vec![30.0, 30.0, 30.0]],
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
    fn sum_empty_dims_backward_is_ones() {
        let mut a = tensor();
        a.set_requires_grad(true);
        let out = a.sum(&[], false);

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
        let out = a.max(&[0, 2], false);

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
    fn max_backward_through_keepdim_flows_only_to_maxima() {
        let mut a = tensor();
        a.set_requires_grad(true);
        // same as max backward but the reduce keeps its axes ([1,2,1])
        let out = &a.max(&[0, 2], true)
            * &GraphTensor::wrap(
                vec![
                    vec![vec![2.0, 2.0, 2.0], vec![3.0, 3.0, 3.0]],
                    vec![vec![5.0, 5.0, 5.0], vec![7.0, 7.0, 7.0]],
                ],
                false,
            );

        let grads = out.backward(true);
        let da = grads.get(&a.to_key()).unwrap();
        assert_eq!(da.shape(), &vec![2, 2, 3]);
        // analogous to the sum case: the max-slice gradient ds[j] = Σ_{i,k} m[i,j,k]
        // (21 / 30) is expanded over {0,2} and gated by the per-slice max mask
        let expected = vec![
            vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 21.0], vec![0.0, 0.0, 30.0]],
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
        tensor().sum(&[0, 0], false);
    }

    #[test]
    #[should_panic]
    fn out_of_range_dim_panics() {
        tensor().max(&[3], false);
    }
}
