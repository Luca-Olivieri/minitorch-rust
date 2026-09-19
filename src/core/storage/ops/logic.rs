use crate::core::storage::TensorStorage;

// Logical operations are boolean-native: they target `Tensor<bool>` directly
// (matching the dtype design), so the old numeric `!= 0.0` f64 forms are gone.
impl TensorStorage<bool> {
    impl_storage_elemwise_ops!(TensorStorage<bool>;
        land, (a, b), a && b;
        lor,  (a, b), a || b;
        lnot, (a), !a;
    );
}
