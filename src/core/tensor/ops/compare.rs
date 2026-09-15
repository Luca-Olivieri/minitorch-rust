use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::dtype::{Float, Numeric};
use crate::core::storage::TensorStorage;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl<T: Numeric> GraphTensor<T> {
    // Comparisons yield a 1/0 mask **in the input dtype** (`T::ONE`/`T::ZERO`),
    // matching the storage kernels. They are non-differentiable (a mask is a
    // step function), but they feed gradients of `maximum`/`max`.
    impl_tensor_binary_method!(gt, TensorStorage::gt);
    impl_tensor_binary_method!(gte, TensorStorage::gte);
    impl_tensor_binary_method!(lt, TensorStorage::lt);
    impl_tensor_binary_method!(lte, TensorStorage::lte);
}

impl<T: Float> GraphTensor<T> {
    /// Elementwise `|self - other| <= atol + rtol * |other|` with PyTorch's
    /// default tolerances (rtol = 1e-5, atol = 1e-8). Non-differentiable.
    pub fn is_close(&self, other: &GraphTensor<T>) -> GraphTensor<T> {
        self.is_close_with(other, 1e-5, 1e-8)
    }

    /// Elementwise `|self - other| <= atol + rtol * |other|` with explicit
    /// tolerances. Non-differentiable.
    pub fn is_close_with(&self, other: &GraphTensor<T>, rtol: f64, atol: f64) -> GraphTensor<T> {
        apply_tensor_op(
            |ops: &[&TensorStorage<T>; 2]| TensorStorage::is_close(ops[0], ops[1], rtol, atol),
            None::<fn([GraphTensor<T>; 2]) -> Box<dyn GradFnTrait<T>>>,
            &[self, other],
        )
    }
}