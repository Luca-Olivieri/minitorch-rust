use std::rc::Rc;

use crate::core::storage::TensorStorage;

impl TensorStorage {
    impl_storage_elemwise_ops!{
        add,     (a, b), a + b;
        neg,     (a), -a;
        sub,     (a, b), a - b;
        mul,    (a, b), a * b;
        div,     (a, b), a / b;
        modul,   (a, b), a % b;
        pow,     (b, e), b.powf(e);
        ln,     (a), a.ln();
        exp,     (a), a.exp();
        abs,     (a), a.abs();
        sqrt,     (a), a.sqrt();
        maximum, (a, b), if a > b { a } else { b };
    }

    /// Direct [m,k] x [k,n] -> [m,n] GEMM kernel.
    pub fn matmul(
        a: &TensorStorage,
        b: &TensorStorage
    ) -> TensorStorage {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            panic!("TensorStorage::matmul requires 2D operands, got {:?} and {:?}.", a.shape, b.shape);
        }

        if a.shape[1] != b.shape[0] {
            panic!("matmul inner dimensions must match ({} != {}).", a.shape[1], b.shape[0]);
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        let mut out = TensorStorage::new(vec![m, n], 0.0);

        // `out` is freshly allocated, so its Rc is unique and mutable.
        let out_buf = Rc::get_mut(&mut out.buffer).unwrap();
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

        out
    }
}
