use crate::core::tensor_storage::TensorStorage;

pub fn apply_op<F, const N: usize>(
    operands: &[&TensorStorage; N],
    op: F
) -> TensorStorage
where
    F: Fn([f64; N]) -> f64,
{
    let first = operands[0];
    let mut out = TensorStorage::new(first.shape.clone(), 0.0);

    for i in 0..first.numel {
        let vals: [f64; N] = std::array::from_fn(|j| operands[j][i]);
        out[i] = op(vals);
    }
    out
}
