use crate::core::storage::TensorStorage;

impl TensorStorage {
    impl_storage_elemwise_ops!{
        gt,     (a, b), if a > b { 1.0 } else { 0.0 };
        gte,     (a, b), if a >= b { 1.0 } else { 0.0 };
        lt,     (a, b), if a < b { 1.0 } else { 0.0 };
        lte,     (a, b), if a <= b { 1.0 } else { 0.0 };
    }
}
