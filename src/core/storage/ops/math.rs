use crate::core::storage::TensorStorage;
use super::utils::apply_op;

macro_rules! impl_storage_elemwise_ops {
    ($($name:ident, ($($arg:ident),+), $body:expr);+ $(;)?) => {
        $(
            impl_storage_elemwise_op!($name, ($($arg),+), $body);
        )+
    };
}

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
    impl_storage_elemwise_ops!{
        add,     (a, b), a + b;
        neg,     (a), -a;
        sub,     (a, b), a - b;
        mul,    (a, b), a * b;
        div,     (a, b), a / b;
        modul,   (a, b), a % b;
        pow,     (b, e), b.powf(e);
        ln,     (a), a.ln();
        abs,     (a), a.abs();
        sqrt,     (a), a.sqrt();
        maximum, (a, b), if a > b { a } else { b };
    }
}
