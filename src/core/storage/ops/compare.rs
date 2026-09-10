use crate::core::storage::TensorStorage;

impl TensorStorage {
    impl_storage_elemwise_ops! {
        gt,     (a, b), if a > b { 1.0 } else { 0.0 };
        gte,     (a, b), if a >= b { 1.0 } else { 0.0 };
        lt,     (a, b), if a < b { 1.0 } else { 0.0 };
        lte,     (a, b), if a <= b { 1.0 } else { 0.0 };
    }

    /// out[i] = 1.0 if |a[i] - b[i]| <= atol + rtol * |b[i]|, else 0.0,
    /// following PyTorch's `torch.isclose` semantics (with `equal_nan = false`).
    pub fn is_close(a: &TensorStorage, b: &TensorStorage, rtol: f64, atol: f64) -> TensorStorage {
        crate::core::storage::ops::utils::apply_op(&[a, b], |[av, bv]: [f64; 2]| {
            if (av - bv).abs() <= atol + rtol * bv.abs() {
                1.0
            } else {
                0.0
            }
        })
    }
}
