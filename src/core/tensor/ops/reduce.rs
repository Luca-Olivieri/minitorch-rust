use std::rc::Rc;

use crate::core::autograd::grad_fn::{BackwardOpKind, maybe_edge};
use crate::core::dtype::{Float, Numeric};
use crate::core::node::{AutogradMeta, TensorNode};
use crate::core::storage::TensorStorage;
use crate::core::storage::ops::reduce::{OneHotLabel, resolve_dims};
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl<T: Numeric> GraphTensor<T> {
    /// Sum over every dimension in `dims` at once.
    pub fn sum(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        self.sum_with_mode(dims, keepdim, false)
    }

    pub fn sum_with_mode(&self, dims: &[usize], keepdim: bool, no_grad: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| {
                let reduced = TensorStorage::sum(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(BackwardOpKind::SumOp {
                dims: dims.clone(),
                keepdim,
            }),
            no_grad,
            &[self],
        )
    }

    /// Max over every dimension in `dims` at once.
    pub fn max(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        self.max_with_mode(dims, keepdim, false)
    }

    pub fn max_with_mode(&self, dims: &[usize], keepdim: bool, no_grad: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| {
                let reduced = TensorStorage::max(ops[0], &dims);
                if keepdim {
                    TensorStorage::unsqueeze_at(&reduced, &dims)
                } else {
                    reduced
                }
            },
            Some(BackwardOpKind::MaxOp {
                dims: dims.clone(),
                keepdim,
            }),
            no_grad,
            &[self],
        )
    }

    /// Argmax along `dim`, yielding the per-slice flat index as an `f64` tensor.
    pub fn argmax(&self, dim: usize, keepdim: bool) -> GraphTensor<f64> {
        let reduced = TensorStorage::argmax(&self.node.storage, dim);
        let out_store = if keepdim {
            TensorStorage::unsqueeze(&reduced, dim)
        } else {
            reduced
        };

        let out_node = TensorNode {
            storage: out_store,
            autograd: None,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }

    /// Direct [m,k] x [k,n] -> [m,n] GEMM.
    pub fn matmul(a: &GraphTensor<T>, b: &GraphTensor<T>) -> GraphTensor<T> {
        Self::matmul_with_mode(a, b, false)
    }

    pub fn matmul_with_mode(
        a: &GraphTensor<T>,
        b: &GraphTensor<T>,
        no_grad: bool,
    ) -> GraphTensor<T> {
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

        // Convert 1D inputs to 2D views: a [K] -> [1,K], b [K] -> [K,1].
        let a2 = if a_ndim == 1 {
            a.unsqueeze_with_mode(0, no_grad)
        } else {
            a.copy_s()
        };
        let b2 = if b_ndim == 1 {
            b.unsqueeze_with_mode(1, no_grad)
        } else {
            b.copy_s()
        };

        let a2_shape = a2.shape();
        let b2_shape = b2.shape();
        let k = a2_shape[1];
        let kb = b2_shape[0];

        if k != kb {
            panic!("matmul inner dimensions must match ({} != {}).", k, kb);
        }

        let out_store = TensorStorage::matmul(&a2.node.storage, &b2.node.storage);
        let autograd =
            maybe_edge(&[a, b], BackwardOpKind::MatmulOp, no_grad).map(AutogradMeta::Node);
        let out_node = TensorNode {
            storage: out_store,
            autograd,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}

impl<T: Float> GraphTensor<T> {
    /// Mean over every dimension in `dims` at once.
    pub fn mean(&self, dims: &[usize], keepdim: bool) -> GraphTensor<T> {
        self.mean_with_mode(dims, keepdim, false)
    }

    pub fn mean_with_mode(&self, dims: &[usize], keepdim: bool, no_grad: bool) -> GraphTensor<T> {
        let dims = resolve_dims(dims, self.shape());
        let count: usize = dims.iter().map(|&d| self.shape()[d]).product();
        let count_t = T::from_f64(count as f64);
        self.sum_with_mode(&dims, keepdim, no_grad)
            .div_with_mode(&GraphTensor::new(vec![], count_t, false), no_grad)
    }

    /// 2D average pooling over a `[batch, channel, height, width]` input.
    pub fn avg_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> GraphTensor<T> {
        self.avg_pool2d_with_mode(kernel, stride, false)
    }

    pub fn avg_pool2d_with_mode(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        no_grad: bool,
    ) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::avg_pool2d(ops[0], kernel, stride),
            Some(BackwardOpKind::AvgPool2dOp { kernel, stride }),
            no_grad,
            &[self],
        )
    }

    /// 2D maximum pooling over a `[batch, channel, height, width]` input.
    pub fn max_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> GraphTensor<T> {
        self.max_pool2d_with_mode(kernel, stride, false)
    }

    pub fn max_pool2d_with_mode(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        no_grad: bool,
    ) -> GraphTensor<T> {
        if no_grad || !self.requires_grad() {
            let storage = TensorStorage::max_pool2d(&self.node.storage, kernel, stride);
            let out_node = TensorNode {
                storage,
                autograd: None,
            };
            return GraphTensor {
                node: Rc::new(out_node),
            };
        }

        let (storage, max_indices) =
            TensorStorage::max_pool2d_with_indices(&self.node.storage, kernel, stride);
        let grad_op = BackwardOpKind::MaxPool2dOp {
            kernel,
            stride,
            max_indices: Rc::new(max_indices),
        };
        let autograd = maybe_edge(&[self], grad_op, no_grad).map(AutogradMeta::Node);
        let out_node = TensorNode { storage, autograd };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}

impl<T: Numeric + OneHotLabel> GraphTensor<T> {
    /// One-hot encode labels into a fresh tensor of shape
    /// `self.shape ++ [num_classes]`.
    pub fn one_hot(&self, num_classes: usize) -> GraphTensor<f64> {
        let out_storage = TensorStorage::one_hot(&self.node.storage, num_classes);

        let out_node = TensorNode {
            storage: out_storage,
            autograd: None,
        };

        GraphTensor::<f64> {
            node: Rc::new(out_node),
        }
    }
}
