use crate::core::dtype::{Float, Numeric};
use crate::core::storage::TensorStorage;

impl<T: Numeric> TensorStorage<T> {
    // Comparisons yield a 1/0 mask **in the input dtype** (T::ONE/T::ZERO), so
    // they stay closed over the tensor's dtype: no silent lossy promotion to
    // f64. On the f64 path this is exactly the old `1.0`/`0.0` behavior.
    impl_storage_elemwise_ops!(TensorStorage<T>;
        gt,     (a, b), if a > b { T::ONE } else { T::ZERO };
        gte,     (a, b), if a >= b { T::ONE } else { T::ZERO };
        lt,     (a, b), if a < b { T::ONE } else { T::ZERO };
        lte,     (a, b), if a <= b { T::ONE } else { T::ZERO };
    );
}

impl<T: Float> TensorStorage<T> {
    /// out[i] = 1.0 if |a[i] - b[i]| <= atol + rtol * |b[i]|, else 0.0,
    /// following PyTorch's `torch.isclose` semantics (with `equal_nan = false`).
    /// The mask uses the input dtype (`T::ONE`/`T::ZERO`); tolerances stay `f64`.
    pub fn is_close(a: &TensorStorage<T>, b: &TensorStorage<T>, rtol: f64, atol: f64) -> TensorStorage<T> {
        crate::core::storage::ops::utils::apply_op(&[a, b], |&[av, bv]| {
            let avf = av.to_f64();
            let bvf = bv.to_f64();
            if (avf - bvf).abs() <= atol + rtol * bvf.abs() {
                T::ONE
            } else {
                T::ZERO
            }
        })
    }
}