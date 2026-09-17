use crate::core::dtype::{Float, Numeric, Signed};
use crate::core::storage::TensorStorage;

impl<T: Numeric> TensorStorage<T> {
    // Not every storage-level op is reachable from the user-facing tensor layer yet,
    // so unused ones are explicitly allowed here.
    //
    // add/sub/mul/div/modul/maximum are signed-agnostic, so they live on the full
    // `Numeric` family: integer tensors get them too (with integer overflow semantics).
    impl_storage_elemwise_ops!(TensorStorage<T>;
        add,     (a, b), a + b;
        sub,     (a, b), a - b;
        mul,    (a, b), a * b;
        div,     (a, b), a / b;
        modul,   (a, b), a % b;
        maximum, (a, b), if a > b { a } else { b };
    );

    /// out[i] = a[i] - scale * b[i], fused into a single pass.
    pub fn sub_scaled(a: &TensorStorage<T>, b: &TensorStorage<T>, scale: T) -> TensorStorage<T> {
        crate::core::storage::ops::utils::apply_op(&[a, b], |&[av, bv]| av - scale * bv)
    }

    /// Direct [m,k] x [k,n] -> [m,n] GEMM kernel.
    pub fn matmul(a: &TensorStorage<T>, b: &TensorStorage<T>) -> TensorStorage<T> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            panic!(
                "TensorStorage::matmul requires 2D operands, got {:?} and {:?}.",
                a.shape, b.shape
            );
        }

        if a.shape[1] != b.shape[0] {
            panic!(
                "matmul inner dimensions must match ({} != {}).",
                a.shape[1], b.shape[0]
            );
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        // `out_buf` is freshly allocated and uniquely owned, so it is mutable.
        let mut out_buf = vec![T::ZERO; m * n];

        let a_buf = &a.buffer;
        let b_buf = &b.buffer;

        let a_off = a.offset;
        let b_off = b.offset;
        let (a_s0, a_s1) = (a.strides[0], a.strides[1]);
        let (b_s0, b_s1) = (b.strides[0], b.strides[1]);

        for i in 0..m {
            let a_row_base = a_off + i * a_s0;
            let out_row_base = i * n;
            for kk in 0..k {
                let a_val = a_buf[a_row_base + kk * a_s1];
                let b_row_base = b_off + kk * b_s0;
                for j in 0..n {
                    out_buf[out_row_base + j] += a_val * b_buf[b_row_base + j * b_s1];
                }
            }
        }

        TensorStorage::from_buffer(vec![m, n], out_buf)
    }
}

impl<T: Signed> TensorStorage<T> {
    // `neg` needs a signed notion (undefined for unsigned integers), so it
    // lives on `Signed` (floats and signed integers).
    impl_storage_elemwise_ops!(TensorStorage<T>;
        neg,     (a), -a;
    );
}

impl<T: Float> TensorStorage<T> {
    // `abs` is reachable only from float ops today (kept float-only); and
    // pow/ln/exp/sqrt are transcendental, so float-only.
    impl_storage_elemwise_ops!(TensorStorage<T>;
        abs,     (a), a.abs();
        pow,     (b, e), b.powf(e);
        ln,     (a), a.ln();
        exp,     (a), a.exp();
        sqrt,     (a), a.sqrt();
    );
}