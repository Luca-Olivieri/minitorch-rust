use std::rc::Rc;

use crate::core::storage::TensorStorage;
use crate::core::storage::ops::shape::squeeze_shape;

impl TensorStorage {
    pub fn sum_dim(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        if dim >= a.shape.len() {
            panic!("Reduction dimension {} out of range for shape {:?}.", dim, a.shape);
        }

         // build output shape
        let out_shape = squeeze_shape(&a.shape, dim);

        let mut out = TensorStorage::new(out_shape, 0.0);

        // `out` is freshly allocated, so its Rc is unique and mutable.
        let out_buf = Rc::get_mut(&mut out.buffer).unwrap();
        let a_buf = &a.buffer;

        // consecutive elements along the reduced dim are `reduced_stride` apart in flat space
        let reduced_stride = a.strides[dim];
        let bases = base_offsets(a, dim);

        // iterate over output logical indices
        for out_i in 0..out.numel {
            let mut acc = 0.0;
            let mut f = bases[out_i];
            for _ in 0..a.shape[dim] {
                acc += a_buf[f];
                f += reduced_stride;
            }

            out_buf[out_i] = acc;
        }

        out
    }

    pub fn argmax(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        if dim >= a.shape.len() {
            panic!("Reduction dimension {} out of range for shape {:?}.", dim, a.shape);
        }

        if a.shape[dim] == 0 {
            panic!("Cannot perform argmax on an empty dimension.");
        }

        // build output shape
        let out_shape = squeeze_shape(&a.shape, dim);

        let mut out = TensorStorage::new(out_shape, 0.0);

        let out_buf = Rc::get_mut(&mut out.buffer).unwrap();
        let a_buf = &a.buffer;

        // consecutive elements along the reduced dim are `reduced_stride` apart in flat space
        let reduced_stride = a.strides[dim];
        let bases = base_offsets(a, dim);

        // iterate over output logical indices
        for out_i in 0..out.numel {
            // initialize max tracking with the first element along the dimension
            let mut f = bases[out_i];
            let mut max_val = a_buf[f];
            let mut max_idx = 0;

            // iterate through the remaining elements in the dimension
            for r in 1..a.shape[dim] {
                f += reduced_stride;
                let val = a_buf[f];

                if val > max_val {
                    max_val = val;
                    max_idx = r;
                }
            }

            // store the index as a float (assuming TensorStorage holds floats)
            out_buf[out_i] = max_idx as f64;
        }

        out
    }
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
