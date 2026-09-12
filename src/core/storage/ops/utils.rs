use crate::core::storage::iter::StridedIter;
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
    } else if let Some(run) = common_inner_run(operands) {
        // inner-run path: every operand is flat-contiguous over the trailing
        // `run` logical elements, so the inner loop reads `base + k` linearly
        // (auto-vectorizes) and only the outer dims need a strided odometer.
        // Covers strided views whose innermost dim kept stride 1, e.g.
        // broadcasts along the outer dims or bias-expanded [*,...,K] views.
        let outer_numel = first.numel / run;
        let mut k = 0usize;
        let mut prod = 1usize;
        while prod < run {
            k += 1;
            prod *= first.shape[first.shape.len() - k];
        }
        let outer_ndim = first.shape.len() - k;
        let outer_shape = &first.shape[..outer_ndim];

        let mut iters: [StridedIter; N] = std::array::from_fn(|j| {
            StridedIter::new(outer_shape, &operands[j].strides[..outer_ndim], operands[j].offset, outer_numel)
        });
        for _ in 0..outer_numel {
            let bases: [usize; N] = std::array::from_fn(|j| iters[j].next().unwrap());
            for i in 0..run {
                let vals: [f64; N] =
                    std::array::from_fn(|j| operands[j].buffer[bases[j] + i]);
                out_buf.push(op(vals));
            }
        }
    } else {
        // mixed path: contiguous operands keep the flat `offset + i` access,
        // strided operands walk an odometer (O(1) amortized per element,
        // instead of per-element div/mod logical indexing).
        let offsets: [usize; N] = std::array::from_fn(|j| operands[j].offset);
        let mut iters: [Option<StridedIter>; N] =
            std::array::from_fn(|j| (!operands[j].contiguous).then(|| operands[j].strided_indices()));
        for i in 0..first.numel {
            let vals: [f64; N] = std::array::from_fn(|j| match &mut iters[j] {
                Some(it) => operands[j].buffer[it.next().unwrap()],
                None => operands[j].buffer[offsets[j] + i],
            });
            out_buf.push(op(vals));
        }
    }

    TensorStorage::from_buffer(first.shape.clone(), out_buf)
}

/// Length of the innermost run over which every operand is flat-contiguous.
///
/// Walks trailing dims from the innermost out, requiring the canonical stride
/// (`shape[d+1] * ...`); the run is the product of the matching trailing dims.
/// Returns `None` when the common run is only 1 flat element, i.e. some operand
/// has a non-1 innermost stride (e.g. a transposed view), where the odometer
/// path below is the best we can do.
fn common_inner_run(operands: &[&TensorStorage]) -> Option<usize> {
    let shape = &operands[0].shape;
    if shape.is_empty() {
        return None;
    }

    let mut run = usize::MAX;
    for o in operands {
        let mut inner = 1usize;
        let mut stride_needed = 1usize;
        for d in (0..shape.len()).rev() {
            if o.strides[d] == stride_needed {
                inner *= shape[d];
                stride_needed *= shape[d];
            } else {
                break;
            }
        }
        run = run.min(inner);
    }

    (run >= 2).then_some(run)
}
