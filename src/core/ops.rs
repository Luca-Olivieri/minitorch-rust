use crate::core::tensor_storage::TensorStorage;

impl TensorStorage {

    pub fn add(
        a: &TensorStorage,
        b: &TensorStorage,
    ) -> TensorStorage {
        apply_op(&[a, b], |vals| vals.iter().sum())
    }
}

/// op receives a slice: one value per input tensor, at the same flat index
pub fn apply_op<F>(
    operands: &[&TensorStorage],
    op: F
) -> TensorStorage
where
    F: Fn(&[f64]) -> f64,
{
    let first = operands[0];

    if !operands.iter().all(|t| t.shape == first.shape) {
        panic!("Shapes must match for element-wise operation");
    }

    let mut out = TensorStorage::new(first.shape.clone(), 0.0);
    let mut buf = vec![0.0; operands.len()]; // reused each iteration, no per-i alloc

    for i in 0..first.numel {
        for (slot, t) in buf.iter_mut().zip(operands.iter()) {
            *slot = t[i];
        }
        out[i] = op(&buf);
    }
    out
}
