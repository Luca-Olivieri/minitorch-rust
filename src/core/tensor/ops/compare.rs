use crate::core::tensor::GraphTensor;
use crate::core::storage::TensorStorage;

use crate::core::tensor::ops::math::apply_tensor_op;
use crate::core::autograd::grad_fn::GradFnTrait;


impl GraphTensor {
    impl_tensor_binary_op!(gt, TensorStorage::gt);
    impl_tensor_binary_op!(gte, TensorStorage::gte);
    impl_tensor_binary_op!(lt, TensorStorage::lt);
    impl_tensor_binary_op!(lte, TensorStorage::lte);

    /// Elementwise `|self - other| <= atol + rtol * |other|` with PyTorch's
    /// default tolerances (rtol = 1e-5, atol = 1e-8). Non-differentiable.
    pub fn is_close(
        &self,
        other: &GraphTensor
    ) -> GraphTensor {
        self.is_close_with(other, 1e-5, 1e-8)
    }

    /// Elementwise `|self - other| <= atol + rtol * |other|` with explicit
    /// tolerances. Non-differentiable.
    pub fn is_close_with(
        &self,
        other: &GraphTensor,
        rtol: f64,
        atol: f64
    ) -> GraphTensor {
        apply_tensor_op(
            |ops: &[&TensorStorage; 2]| TensorStorage::is_close(ops[0], ops[1], rtol, atol),
            None::<fn([GraphTensor; 2]) -> Box<dyn GradFnTrait>>,
            &[self, other],
        )
    }
}
