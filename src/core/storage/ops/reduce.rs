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

            // build input md index with dim inserted
            let mut in_md = vec![0; a.shape.len()];

            // iterate over output logical indices
            for out_i in 0..out.numel {

                let out_md = out.logic_to_md(out_i);

                // in_md follows out_md but skips reduced dimension
                Self::populate_in_md_for_accum(&mut in_md, &out_md, dim);

                // initialize max tracking with the first element along the dimension
                in_md[dim] = 0;
                let mut max_val = a[&in_md];
                let mut max_idx = 0;

                // iterate through the remaining elements in the dimension
                for r in 1..a.shape[dim] {
                    in_md[dim] = r;
                    let val = a[&in_md];

                    if val > max_val {
                        max_val = val;
                        max_idx = r;
                    }
                }

                // store the index as a float (assuming TensorStorage holds floats)
                out[out_i] = max_idx as f64;
            }

            out
        }

    pub(crate) fn populate_in_md_for_accum(
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
}
