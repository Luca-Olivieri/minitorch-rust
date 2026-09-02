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
            crate::core::storage::ops::utils::apply_op(&[$(*$arg),+], |[$($arg),+]| $body)
        }
    };
    (@count $($arg:ident),+) => {
        <[()]>::len(&[$(impl_storage_elemwise_op!(@unit $arg)),+])
    };
    (@unit $arg:ident) => { () };
}
