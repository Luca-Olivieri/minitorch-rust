use crate::core::storage::TensorStorage;
use super::utils::apply_op;

macro_rules! impl_storage_elemwise_ops {
    ($($name:ident, ($($arg:ident),+), $body:expr);+ $(;)?) => {
        $(
            impl_storage_elemwise_op!($name, ($($arg),+), $body);
        )+
    };
}

macro_rules! impl_storage_elemwise_op {
    ($name:ident, ($($arg:ident),+), $body:expr) => {
        pub fn $name(operands: &[&TensorStorage; impl_storage_elemwise_op!(@count $($arg),+)]) -> TensorStorage {
            let [$($arg),+] = operands;
            apply_op(&[$(*$arg),+], |[$($arg),+]| $body)
        }
    };
    (@count $($arg:ident),+) => {
        <[()]>::len(&[$(impl_storage_elemwise_op!(@unit $arg)),+])
    };
    (@unit $arg:ident) => { () };
}

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
        abs,     (a), a.abs();
        sqrt,     (a), a.sqrt();
        maximum, (a, b), if a > b { a } else { b };
    }
}

impl TensorStorage {
    pub fn sum_dim(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        if dim >= a.shape.len() {
            panic!("Reduction dimension {} out of range for shape {:?}.", dim, a.shape);
        }

         // build output shape
        let out_shape = Self::reduce_shape(&a.shape, dim);

        let mut out = TensorStorage::new(out_shape, 0.0);

        // build input md index with dim inserted
        let mut in_md = vec![0; a.shape.len()];
        // iterate over output logical indices
        for out_i in 0..out.numel {

            let out_md = out.logic_to_md(out_i);

            // in_md follows out_md but skips reduced dimension
            Self::populate_in_md_for_accum(&mut in_md, &out_md, dim);

            let mut acc = 0.0;
            for r in 0..a.shape[dim] {
                in_md[dim] = r;
                acc += a[&in_md]
            }

            out[out_i] = acc;
        }

        out
    }

    pub fn sum(
        operands: &[&TensorStorage]
    ) -> TensorStorage {

        let a = operands[0];

        let mut out: TensorStorage = a.copy_d();

        for _ in 0..a.shape.len() {
            out = TensorStorage::sum_dim(&out, 0);
        }

        out
    }

    fn populate_in_md_for_accum(
        in_md: &mut [usize],
        out_md: &[usize],
        dim: usize
    ) {
        // copy output coords into input coords (skip reduced dim)
        let mut j = 0;
        for i in 0..in_md.len() {
            if i == dim {
                continue;
            }
            in_md[i] = out_md[j];
            j += 1;
        }
    }

    fn reduce_shape(
        shape: &Vec<usize>,
        dim: usize
    ) -> Vec<usize> {
        let mut out_shape = shape.clone();
        out_shape.remove(dim);
        out_shape
    }
}
