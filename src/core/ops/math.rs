use crate::core::tensor_storage::TensorStorage;
use super::utils::apply_op;

macro_rules! impl_storage_elemwise_op {
    ($name:ident, ($($arg:ident),+), $body:expr) => {
        pub fn $name(operands: &[&TensorStorage; impl_storage_elemwise_op!(@count $($arg),+)]) -> TensorStorage {
            let [$($arg),+] = operands;
            apply_op(&[$(*$arg),+], |[$($arg),+]| $body)
        }
    };
    (@count $($arg:ident),+) => {
        <[()]>::len(&[$(impl_storage_elemwise_op!(@unit $arg)),+])
    };
    (@unit $arg:ident) => { () };
}

impl TensorStorage {
    impl_storage_elemwise_op!(add, (a, b), a + b);
    impl_storage_elemwise_op!(minus, (a), -a);
    impl_storage_elemwise_op!(sub, (a, b), a - b);
    impl_storage_elemwise_op!(mult, (a, b), a * b);
    impl_storage_elemwise_op!(div, (a, b), a / b);
    impl_storage_elemwise_op!(modul, (a, b), a % b);
    impl_storage_elemwise_op!(pow, (b, e), b.powf(e));
    impl_storage_elemwise_op!(log, (a, b), a.log(b));
    impl_storage_elemwise_op!(maximum, (a, b), if a > b { a } else { b });
}
