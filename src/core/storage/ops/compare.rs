use crate::core::dtype::{Float, Numeric};
use crate::core::storage::TensorStorage;
use crate::core::storage::ops::utils::apply_op;

impl<T: Numeric> TensorStorage<T> {
    // Comparisons yield a `bool` mask: the operands share the input dtype `T`,
    // the result is always `bool`. Values combine with numeric tensors through
    // the exact `bool -> T` cast (`T::cast_from`, always infallible).
    impl_storage_elemwise_ops!(TensorStorage<T> => bool;
        gt,     (a, b), a > b;
        gte,    (a, b), a >= b;
        lt,     (a, b), a < b;
        lte,    (a, b), a <= b;
    );
}

impl<T: Float> TensorStorage<T> {
    /// out[i] = true if |a[i] - b[i]| <= atol + rtol * |b[i]|, else false,
    /// following PyTorch's `torch.isclose` semantics (with `equal_nan =
    /// false`). Operands share the float dtype `T`; the result is `bool`.
    /// Tolerances stay `f64`.
    pub fn is_close(
        a: &TensorStorage<T>,
        b: &TensorStorage<T>,
        rtol: f64,
        atol: f64,
    ) -> TensorStorage<bool> {
        apply_op(&[a, b], |&[av, bv]| {
            let avf = av.to_f64();
            let bvf = bv.to_f64();
            (avf - bvf).abs() <= atol + rtol * bvf.abs()
        })
    }
}
