use crate::core::dtype::{Float, Numeric};
use crate::core::storage::TensorStorage;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op_into;

impl<T: Numeric> GraphTensor<T> {
    // Comparisons yield a `GraphTensor<bool>` mask: operands share the input
    // dtype `T`, the result is always `bool`. Non-differentiable (a mask is a
    // step function), but they feed gradients of `maximum`/`max` by casting
    // back into `T` (`bool -> T` is the always-compiled exact cast).
    impl_tensor_bool_binary_method!(gt, TensorStorage::gt);
    impl_tensor_bool_binary_method!(gte, TensorStorage::gte);
    impl_tensor_bool_binary_method!(lt, TensorStorage::lt);
    impl_tensor_bool_binary_method!(lte, TensorStorage::lte);
}

impl<T: Float> GraphTensor<T> {
    /// Elementwise `|self - other| <= atol + rtol * |other|` with PyTorch's
    /// default tolerances (rtol = 1e-5, atol = 1e-8). Non-differentiable;
    /// returns a `bool` mask.
    pub fn is_close(&self, other: &GraphTensor<T>) -> GraphTensor<bool> {
        self.is_close_with(other, 1e-5, 1e-8)
    }

    /// Elementwise `|self - other| <= atol + rtol * |other|` with explicit
    /// tolerances. Non-differentiable; returns a `bool` mask.
    pub fn is_close_with(&self, other: &GraphTensor<T>, rtol: f64, atol: f64) -> GraphTensor<bool> {
        apply_tensor_op_into(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::is_close(ops[0], ops[1], rtol, atol),
            &[self, other],
        )
    }
}
