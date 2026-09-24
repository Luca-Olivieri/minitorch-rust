use std::rc::Rc;

use crate::core::autograd::grad_fn::BackwardOpKind;
use crate::core::dtype::{Dtype, Numeric};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::AbstractTensor;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl<T: Numeric> GraphTensor<T> {
    pub fn unsqueeze(&self, dim: usize) -> GraphTensor<T> {
        self.unsqueeze_with_mode(dim, false)
    }

    pub fn unsqueeze_with_mode(&self, dim: usize, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::unsqueeze(ops[0], dim),
            Some(BackwardOpKind::UnsqueezeOp { dim }),
            no_grad,
            &[self],
        )
    }

    pub fn squeeze(&self, dim: usize) -> GraphTensor<T> {
        self.squeeze_with_mode(dim, false)
    }

    pub fn squeeze_with_mode(&self, dim: usize, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::squeeze(ops[0], dim),
            Some(BackwardOpKind::SqueezeOp { dim }),
            no_grad,
            &[self],
        )
    }

    pub fn transpose(&self, dim_a: usize, dim_b: usize) -> GraphTensor<T> {
        self.transpose_with_mode(dim_a, dim_b, false)
    }

    pub fn transpose_with_mode(&self, dim_a: usize, dim_b: usize, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::transpose(ops[0], dim_a, dim_b),
            Some(BackwardOpKind::TransposeOp { dim_a, dim_b }),
            no_grad,
            &[self],
        )
    }

    pub fn expand(&self, dim: usize, times: usize) -> GraphTensor<T> {
        self.expand_with_mode(dim, times, false)
    }

    pub fn expand_with_mode(&self, dim: usize, times: usize, no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::expand(ops[0], dim, times),
            Some(BackwardOpKind::ExpandOp { dim }),
            no_grad,
            &[self],
        )
    }

    pub fn broadcast(&self, b: &GraphTensor<T>) -> GraphTensor<T> {
        self.broadcast_with_mode(b, false)
    }

    pub fn broadcast_with_mode(&self, b: &GraphTensor<T>, no_grad: bool) -> GraphTensor<T> {
        self.broadcast_to_shape_with_mode(b.shape(), no_grad)
    }

    pub fn broadcast_to_shape(&self, target_shape: &[usize]) -> GraphTensor<T> {
        self.broadcast_to_shape_with_mode(target_shape, false)
    }

    pub fn broadcast_to_shape_with_mode(
        &self,
        target_shape: &[usize],
        no_grad: bool,
    ) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::broadcast_to_shape(ops[0], target_shape),
            Some(BackwardOpKind::BroadcastOp {
                old_shape: self.shape().clone(),
            }),
            no_grad,
            &[self],
        )
    }

    /// Zero-pad every dimension by `pads[d] = (before, after)` elements.
    ///
    /// Padded regions read as zero; the gradient discards them via an inverse
    /// slice back to the original shape. `pads` must have exactly one pair per
    /// dimension; padded regions read as `T::ZERO`.
    pub fn pad(&self, pads: &[(usize, usize)]) -> GraphTensor<T> {
        self.pad_with_mode(pads, false)
    }

    pub fn pad_with_mode(&self, pads: &[(usize, usize)], no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::pad(ops[0], pads),
            Some(BackwardOpKind::PadOp {
                pads: pads.to_vec(),
            }),
            no_grad,
            &[self],
        )
    }

    /// Extract the window `[start, start + len)` along every dimension,
    /// materializing it into a fresh contiguous tensor.
    pub fn slice(&self, ranges: &[(usize, usize)]) -> GraphTensor<T> {
        self.slice_with_mode(ranges, false)
    }

    pub fn slice_with_mode(&self, ranges: &[(usize, usize)], no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::slice(ops[0], ranges),
            Some(BackwardOpKind::SliceOp {
                ranges: ranges.to_vec(),
            }),
            no_grad,
            &[self],
        )
    }

    /// Extract a strided window from every dimension, materializing the result
    /// into a fresh contiguous buffer.
    pub fn slice_strided(&self, ranges: &[(usize, usize, usize)]) -> GraphTensor<T> {
        self.slice_strided_with_mode(ranges, false)
    }

    pub fn slice_strided_with_mode(
        &self,
        ranges: &[(usize, usize, usize)],
        no_grad: bool,
    ) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::slice_strided(ops[0], ranges),
            Some(BackwardOpKind::StridedSliceOp {
                ranges: ranges.to_vec(),
            }),
            no_grad,
            &[self],
        )
    }

    /// Reinterpret the logical elements under a new shape, materializing into
    /// a fresh contiguous buffer.
    pub fn reshape(&self, new_shape: &[usize]) -> GraphTensor<T> {
        self.reshape_with_mode(new_shape, false)
    }

    pub fn reshape_with_mode(&self, new_shape: &[usize], no_grad: bool) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 1]| TensorStorage::reshape(ops[0], new_shape),
            Some(BackwardOpKind::ReshapeOp {
                new_shape: new_shape.to_vec(),
            }),
            no_grad,
            &[self],
        )
    }
}

impl<T: Dtype> GraphTensor<T> {
    /// Stack a list of tensors along a new dimension 0.
    ///
    /// All input tensors must have identical shapes. The output has shape
    /// `[N, original_shape...]` where `N` is `tensors.len()`.
    pub fn stack(tensors: &[GraphTensor<T>]) -> GraphTensor<T> {
        if tensors.is_empty() {
            panic!("Cannot stack an empty list of tensors.");
        }

        let storages: Vec<&TensorStorage<T>> = tensors.iter().map(|t| &t.node.storage).collect();
        let out_store = TensorStorage::stack(&storages);

        // Non-differentiable: stacking is a pure data layout op.
        let out_node = TensorNode {
            storage: out_store,
            autograd: None,
        };

        GraphTensor {
            node: Rc::new(out_node),
        }
    }
}
