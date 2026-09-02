use crate::core::tensor::GraphTensor;
use crate::core::storage::TensorStorage;

use crate::core::tensor::ops::math::apply_tensor_op;
use crate::core::autograd::grad_fn::GradFnTrait;


impl GraphTensor {
    impl_tensor_binary_op!(gt, TensorStorage::gt);
    impl_tensor_binary_op!(gte, TensorStorage::gte);
    impl_tensor_binary_op!(lt, TensorStorage::lt);
    impl_tensor_binary_op!(lte, TensorStorage::lte);
}
