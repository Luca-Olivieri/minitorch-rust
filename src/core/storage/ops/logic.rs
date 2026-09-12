use crate::core::storage::TensorStorage;

impl TensorStorage {
    impl_storage_elemwise_ops! {
        land, (a, b), if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 };
        lor,  (a, b), if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 };
        lnot, (a), if a == 0.0 { 1.0 } else { 0.0 };
    }
}
