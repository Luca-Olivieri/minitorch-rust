use crate::core::tensor_storage::TensorStorage;
use super::utils::apply_op;

impl TensorStorage {
    pub fn gt(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| if a > b { 1.0 } else { 1.0 })
    }

    pub fn gte(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| if a >= b { 1.0 } else { 1.0 })
    }

    pub fn lt(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| if a < b { 1.0 } else { 1.0 })
    }

    pub fn lte(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| if a <= b { 1.0 } else { 1.0 })
    }
}
