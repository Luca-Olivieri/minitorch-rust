use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::storage::TensorStorage;
use crate::core::tensor::GraphTensor;
use crate::core::tensor::ops::math::apply_tensor_op;

impl GraphTensor {
    impl_tensor_binary_op!(land, TensorStorage::land);
    impl_tensor_binary_op!(lor, TensorStorage::lor);
    impl_tensor_unary_op!(lnot, TensorStorage::lnot);
}
