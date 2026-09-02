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
        abs,     (a), a.abs();
        sqrt,     (a), a.sqrt();
        maximum, (a, b), if a > b { a } else { b };
    }
}
