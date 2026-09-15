macro_rules! impl_storage_elemwise_ops {
    ($storage:ty; $($name:ident, ($($arg:ident),+), $body:expr);+ $(;)?) => {
        $(
            impl_storage_elemwise_op!($storage, $name, ($($arg),+), $body);
        )+
    };
}

macro_rules! impl_storage_elemwise_op {
    ($storage:ty, $name:ident, ($($arg:ident),+), $body:expr) => {
        #[allow(dead_code)]
        pub fn $name(operands: &[&$storage; impl_storage_elemwise_op!(@count $($arg),+)]) -> $storage {
            let [$($arg),+] = operands;
            crate::core::storage::ops::utils::apply_op(&[$(*$arg),+], |&[$($arg),+]| $body)
        }
    };
    (@count $($arg:ident),+) => {
        <[()]>::len(&[$(impl_storage_elemwise_op!(@unit $arg)),+])
    };
    (@unit $arg:ident) => { () };
}