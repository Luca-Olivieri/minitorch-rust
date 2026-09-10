use crate::core::storage::TensorStorage;

pub fn apply_op<F, const N: usize>(operands: &[&TensorStorage; N], op: F) -> TensorStorage
where
    F: Fn([f64; N]) -> f64, // TODO should I pass this slice as a reference?
{
    let first = operands[0];

    // capacity == numel, so the pushes below never reallocate.
    let mut out_buf = Vec::with_capacity(first.numel);

    if operands.iter().all(|o| o.contiguous) {
        // contiguous fast path: logical index == flat index (+ offset)
        let offsets: [usize; N] = std::array::from_fn(|j| operands[j].offset);
        for i in 0..first.numel {
            let vals: [f64; N] = std::array::from_fn(|j| operands[j].buffer[offsets[j] + i]);
            out_buf.push(op(vals));
        }
    } else {
        // strided path: resolve each logical index through the tensor's strides
        for i in 0..first.numel {
            let vals: [f64; N] = std::array::from_fn(|j| operands[j][i]);
            out_buf.push(op(vals));
        }
    }

    TensorStorage::from_buffer(first.shape.clone(), out_buf)
}
