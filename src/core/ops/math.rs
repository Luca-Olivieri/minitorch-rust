use crate::core::tensor_storage::TensorStorage;
use super::utils::apply_op;

impl TensorStorage {
    pub fn add(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        // apply_op(&[&a, &b], |vals| vals.iter().sum())
        apply_op(&[&a, &b], |[a, b]| a + b)
    }

    pub fn minus(a: &TensorStorage) -> TensorStorage {
        apply_op(&[&a], |[a]| -a)
    }

    // TODO is this function even needed?
    pub fn sub(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| a - b)
    }

    pub fn mult(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| a * b)
    }

    pub fn div(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| a / b)
    }

     // TODO find a better namea: &TensorStorage, b: &TensorStorage) -> TensorStorage
    pub fn modul(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
        apply_op(&[&a, &b], |[a, b]| a % b)
    }

    pub fn pow(base: &TensorStorage, exp: &TensorStorage) -> TensorStorage {
        apply_op(&[&base, &exp], |[b, e]| b.powf(e))
    }

    pub fn log(arg: &TensorStorage, base: &TensorStorage) -> TensorStorage {
        apply_op(&[&arg, &base], |[a, b]| a.log(b))
    }
}
