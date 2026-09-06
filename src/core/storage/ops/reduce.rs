use crate::core::storage::TensorStorage;
use crate::core::storage::ops::shape::squeeze_shape;

impl TensorStorage {
    /// Reduce over every dimension in a single pass, yielding a scalar `[]`.
    pub fn sum_all(
        a: &TensorStorage
    ) -> TensorStorage {
        let total = if a.contiguous {
            a.buffer[a.offset..a.offset + a.numel].iter().sum()
        } else {
            (0..a.numel).map(|i| a[i]).sum()
        };

        TensorStorage::from_buffer(Vec::new(), vec![total])
    }

    pub fn sum_dim(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        reduce_dim(a, dim, |v| v, |acc, v| acc + v, |acc| acc)
    }

    pub fn max_dim(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        reduce_dim(a, dim, |v| v, |acc, v| if v > acc { v } else { acc }, |acc| acc)
    }

    pub fn argmax(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        reduce_dim(
            a,
            dim,
            |v| (v, 0usize),
            |(v, i), val| if val > v { (val, i + 1) } else { (v, i) },
            |(_, i)| i as f64,
        )
    }
}

/// Shared single-dimension reduction.
///
/// Visits every output slice (the shape without `dim`, in row-major order), seeds an
/// accumulator `A` with the first element along `dim`, folds the remaining elements
/// with `fold`, and writes `finish(acc)` to the corresponding output slot. The inner
/// loop is a flat-buffer `acc = fold(acc, next)` walk, so per-slice logic just
/// describes how to combine values.
fn reduce_dim<A, F, G>(
    a: &TensorStorage,
    dim: usize,
    seed: impl Fn(f64) -> A,
    fold: F,
    finish: G,
) -> TensorStorage
where
    F: Fn(A, f64) -> A,
    G: Fn(A) -> f64,
{
    if dim >= a.shape.len() {
        panic!("Reduction dimension {} out of range for shape {:?}.", dim, a.shape);
    }

    // build output shape
    let out_shape = squeeze_shape(&a.shape, dim);

    let mut out = TensorStorage::new(out_shape, 0.0);

    let out_numel = out.numel;
    let out_buf = out.buffer_mut();
    let a_buf = &a.buffer;

    // consecutive elements along the reduced dim are `reduced_stride` apart in flat space
    let reduced_stride = a.strides[dim];
    let bases = base_offsets(a, dim);

    // iterate over output logical indices
    for out_i in 0..out_numel {
        let mut f = bases[out_i];
        let mut acc = seed(a_buf[f]);
        for _ in 1..a.shape[dim] {
            f += reduced_stride;
            acc = fold(acc, a_buf[f]);
        }

        out_buf[out_i] = finish(acc);
    }

    out
}

/// Flat offset of the first element of each slice along `dim`, walked in output
/// row-major order.
///
/// Uses an odometer to avoid any per-element division or multi-dim index allocation.
fn base_offsets(a: &TensorStorage, dim: usize) -> Vec<usize> {
    let out_numel = a.numel / a.shape[dim];
    let out_ndim = a.shape.len() - 1;
    let mut bases = Vec::with_capacity(out_numel);

    let mut base = a.offset;
    let mut coords = vec![0usize; out_ndim];

    // for each output dim k (input dims skipping `dim`): its size and flat stride
    let mut sizes = Vec::with_capacity(out_ndim);
    let mut mults = Vec::with_capacity(out_ndim);
    for d in 0..a.shape.len() {
        if d == dim {
            continue;
        }
        sizes.push(a.shape[d]);
        mults.push(a.strides[d]);
    }

    for out_i in 0..out_numel {
        bases.push(base);

        if out_i == out_numel - 1 {
            break;
        }

        // advance the odometer: increment the least significant output coordinate,
        // cascading a carry to the next coordinate on overflow.
        let mut k = out_ndim - 1;
        'carry: loop {
            coords[k] += 1;
            base += mults[k];
            if coords[k] < sizes[k] {
                break 'carry;
            }
            coords[k] = 0;
            base -= mults[k] * sizes[k];
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }

    bases
}
