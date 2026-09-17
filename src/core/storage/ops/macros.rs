macro_rules! impl_storage_elemwise_ops {
    // Same output dtype as the input (arithmetic home, `U = T`).
    ($storage:ty; $($name:ident, ($($arg:ident),+), $body:expr);+ $(;)?) => {
        $(
            impl_storage_elemwise_op!($storage, $name, ($($arg),+), $body);
        )+
    };
    // Different output dtype `$out` (e.g. comparisons `(T, T) -> bool`).
    ($storage:ty => $out:ty; $($name:ident, ($($arg:ident),+), $body:expr);+ $(;)?) => {
        $(
            impl_storage_elemwise_op!($storage => $out, $name, ($($arg),+), $body);
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
    ($storage:ty => $out:ty, $name:ident, ($($arg:ident),+), $body:expr) => {
        #[allow(dead_code)]
        pub fn $name(operands: &[&$storage; impl_storage_elemwise_op!(@count $($arg),+)]) -> $crate::core::storage::TensorStorage<$out> {
            let [$($arg),+] = operands;
            crate::core::storage::ops::utils::apply_op(&[$(*$arg),+], |&[$($arg),+]| $body)
        }
    };
    (@count $($arg:ident),+) => {
        <[()]>::len(&[$(impl_storage_elemwise_op!(@unit $arg)),+])
    };
    (@unit $arg:ident) => { () };
}