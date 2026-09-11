use crate::core::storage::TensorStorage;

impl TensorStorage {
    /// Sum over every dimension in `dims`, yielding a tensor with those
    /// dimensions removed. An empty `dims` aggregates over all dimensions.
    ///
    /// Dispatch:
    /// - all dims     -> flat/scaled `sum_all` pass (no intermediate allocations)
    /// - a single dim -> tight strided loop (`reduce_single_dim`)
    /// - otherwise    -> general stride-odometer pass (`reduce_dims`)
    pub fn sum(a: &TensorStorage, dims: &[usize]) -> TensorStorage {
        let dims = resolve_dims(dims, &a.shape);

        if dims.len() == a.shape.len() {
            Self::sum_all(a)
        } else {
            reduce_dims(a, &dims, |v| v, |acc, v| acc + v, |acc| acc)
        }
    }

    /// Max over every dimension in `dims`, yielding the per-slice maximum over
    /// their combined cartesian product. An empty `dims` aggregates over all
    /// dimensions. A single reduced dim uses the tight strided loop.
    pub fn max(a: &TensorStorage, dims: &[usize]) -> TensorStorage {
        let dims = resolve_dims(dims, &a.shape);

        if dims.len() == a.shape.len() {
            Self::max_all(a)
        } else {
            reduce_dims(
                a,
                &dims,
                |v| v,
                |acc, v| if v > acc { v } else { acc },
                |acc| acc,
            )
        }
    }

    pub fn argmax(a: &TensorStorage, dim: usize) -> TensorStorage {
        reduce_dims(
            a,
            &[dim],
            |v| (v, 0usize),
            |(v, i), val| if val > v { (val, i + 1) } else { (v, i) },
            |(_, i)| i as f64,
        )
    }

    /// Reduce over every dimension in a single pass, yielding a scalar `[]`.
    fn sum_all(a: &TensorStorage) -> TensorStorage {
        if a.contiguous {
            // Flat slice sum: auto-vectorizes cleanly and does not allocate.
            let total = a.buffer[a.offset..a.offset + a.numel].iter().sum();
            TensorStorage::from_buffer(Vec::new(), vec![total])
        } else {
            // Strided view: walk the reduced dims with the stride odometer instead
            // of per-element div/mod logical indexing (~3x faster in practice).
            let dims: Vec<usize> = (0..a.shape.len()).collect();
            reduce_dims(a, &dims, |v| v, |acc, v| acc + v, |acc| acc)
        }
    }

    /// Max over every dimension in a single pass, yielding a scalar `[]`.
    fn max_all(a: &TensorStorage) -> TensorStorage {
        if a.contiguous {
            // Flat slice fold: no stride bookkeeping, vectorizes the fcmp/select.
            let total = a.buffer[a.offset..a.offset + a.numel]
                .iter()
                .fold(f64::NEG_INFINITY, |acc, v| if *v > acc { *v } else { acc });
            TensorStorage::from_buffer(Vec::new(), vec![total])
        } else {
            // Strided view: same odometer walk as `sum_all`.
            let dims: Vec<usize> = (0..a.shape.len()).collect();
            reduce_dims(
                a,
                &dims,
                |v| v,
                |acc, v| if v > acc { v } else { acc },
                |acc| acc,
            )
        }
    }
}

/// Shared multi-dimension reduction.
///
/// Visits every output slice (the shape without `dims`, in row-major order), seeds an
/// accumulator `A` with the first element of the slice, folds the remaining elements
/// of the cartesian product over the reduced dims with `fold`, and writes `finish(acc)`
/// to the corresponding output slot. Each reduced dimension contributes its own stride,
/// so the reduced dims do not need to be adjacent.
///
/// `dims` must not contain duplicates and each must be in range. Callers
/// [`resolve_dims`] first, so an empty `dims` means "all dims" there.
fn reduce_dims<A, F, G>(
    a: &TensorStorage,
    dims: &[usize],
    seed: impl Fn(f64) -> A,
    fold: F,
    finish: G,
) -> TensorStorage
where
    F: Fn(A, f64) -> A,
    G: Fn(A) -> f64,
{
    let dims = normalize_dims(dims, &a.shape);

    // single reduced dim: the tight strided loop beats the general odometer for
    // cache-resident tensors, and it is by far the most common case.
    if let [dim] = dims.as_slice() {
        return reduce_single_dim(a, *dim, seed, fold, finish);
    }

    // build output shape
    let out_shape = squeezed_shape(&a.shape, &dims);

    let mut out = TensorStorage::new(out_shape, 0.0);

    let out_numel = out.numel;
    let out_buf = out.buffer_mut();
    let a_buf = &a.buffer;

    // elements along the reduced dims are `reduced_strides[k]` apart in flat space
    let reduced_total: usize = dims.iter().map(|&d| a.shape[d]).product();
    let reduced_sizes: Vec<usize> = dims.iter().map(|&d| a.shape[d]).collect();
    let reduced_strides: Vec<usize> = dims.iter().map(|&d| a.strides[d]).collect();
    let bases = base_offsets(a, &dims);

    // odometer over the reduced dims; reset per output slice
    let mut coords = vec![0usize; dims.len()];

    // iterate over output logical indices
    for out_i in 0..out_numel {
        let mut f = bases[out_i];
        let mut acc = seed(a_buf[f]);

        coords.fill(0);
        for _ in 1..reduced_total {
            // advance the odometer: increment the least significant reduced
            // coordinate, cascading a carry to the next on overflow.
            let mut k = dims.len() - 1;
            loop {
                coords[k] += 1;
                f += reduced_strides[k];
                if coords[k] < reduced_sizes[k] {
                    break;
                }
                coords[k] = 0;
                f -= reduced_strides[k] * reduced_sizes[k];
                if k == 0 {
                    unreachable!("reduced_total bounds the cartesian product");
                }
                k -= 1;
            }

            acc = fold(acc, a_buf[f]);
        }

        out_buf[out_i] = finish(acc);
    }

    out
}

/// Tight single-dimension reduction: consecutive elements along the reduced dim
/// are `a.strides[dim]` apart in flat space, so the inner loop is just a strided
/// walk with no odometer bookkeeping.
fn reduce_single_dim<A, F, G>(
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
    // build output shape
    let out_shape = squeezed_shape(&a.shape, &[dim]);

    let mut out = TensorStorage::new(out_shape, 0.0);

    let out_numel = out.numel;
    let out_buf = out.buffer_mut();
    let a_buf = &a.buffer;

    let reduced_stride = a.strides[dim];
    let bases = base_offsets(a, &[dim]);

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

/// Flat offset of the first element of each output slice, walked in output
/// row-major order.
///
/// Uses an odometer over the non-reduced dims to avoid any per-element division
/// or multi-dim index allocation.
fn base_offsets(a: &TensorStorage, dims: &[usize]) -> Vec<usize> {
    let reduced_total: usize = dims.iter().map(|&d| a.shape[d]).product();
    let out_numel = a.numel / reduced_total;
    let out_ndim = a.shape.len() - dims.len();
    let mut bases = Vec::with_capacity(out_numel);

    let mut base = a.offset;
    let mut coords = vec![0usize; out_ndim];

    // for each output dim k (input dims not in `dims`): its size and flat stride
    let mut sizes = Vec::with_capacity(out_ndim);
    let mut mults = Vec::with_capacity(out_ndim);
    for (d, &shape_d) in a.shape.iter().enumerate() {
        if !dims.contains(&d) {
            sizes.push(shape_d);
            mults.push(a.strides[d]);
        }
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

/// Normalize `dims`: bounds-check, de-duplicate and sort. Unlike
/// [`resolve_dims`], an empty list stays empty (the identity/no-op reduction).
pub(crate) fn normalize_dims(dims: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(dims.len());
    for &d in dims {
        if d >= shape.len() {
            panic!(
                "Reduction dimension {} out of range for shape of length {:?}.",
                d, shape
            );
        }
        if out.contains(&d) {
            panic!("Reduction dimension {d} appears more than once.");
        }
        out.push(d);
    }
    out.sort_unstable();
    out
}

/// Normalize `dims` and treat an empty list as "all dimensions" (like PyTorch's
/// `dim=None`), i.e. the full reduction.
pub(crate) fn resolve_dims(dims: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut out = normalize_dims(dims, shape);
    if out.is_empty() {
        out.extend(0..shape.len());
    }
    out
}

/// The shape of the reduced tensor: the input shape with `dims` removed.
fn squeezed_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter(|(d, _)| !dims.contains(d))
        .map(|(_, &s)| s)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::*;

    // The `max` kernel before `max_all` existed: the generic reduce_dims odometer.
    fn old_max_odometer(a: &TensorStorage) -> TensorStorage {
        let dims: Vec<usize> = (0..a.shape.len()).collect();
        reduce_dims(
            a,
            &dims,
            |v| v,
            |acc, v| if v > acc { v } else { acc },
            |acc| acc,
        )
    }

    // Time one variant over `iters` runs, returning ns/op.
    fn time(f: impl Fn() -> TensorStorage, iters: usize) -> Duration {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        t.elapsed() / iters as u32
    }

    fn bench_one(name: &str, a: &TensorStorage) {
        const ITERS: usize = 100;

        // correctness: both produce the same scalar max
        let flat = TensorStorage::max_all(a).buffer.as_ref()[0];
        let odometer = old_max_odometer(a).buffer.as_ref()[0];
        assert_eq!(flat, odometer);

        // warm up both code paths so the comparison is steady-state
        for _ in 0..10 {
            black_box(old_max_odometer(a));
            black_box(TensorStorage::max_all(a));
        }

        let t_odo = time(|| old_max_odometer(a), ITERS).as_secs_f64() * 1e3;
        let t_flat = time(|| TensorStorage::max_all(a), ITERS).as_secs_f64() * 1e3;
        eprintln!(
            "{name:28} odometer {:>9.3}ms   max_all {:>9.3}ms   ratio {:.2}x",
            t_odo,
            t_flat,
            t_odo / t_flat
        );
    }

    #[test]
    fn max_all_matches_reduce_dims() {
        let a = TensorStorage::from_buffer(vec![3, 4], (1..=12).map(|x| x as f64).collect());
        assert_eq!(TensorStorage::max_all(&a).buffer.as_ref()[0], 12.0);
        let t = TensorStorage::transpose(&a);
        assert!(!t.contiguous);
        assert_eq!(TensorStorage::max_all(&t).buffer.as_ref()[0], 12.0);
    }

    /// Run with: cargo test --release -- --ignored --nocapture bench_max_all
    #[test]
    #[ignore]
    fn bench_max_all() {
        const M: usize = 1 << 20; // 1 Mi elems = 8 MB of f64

        // cache-resident 2D (goes through the general odometer, not the 1D tight loop)
        let a = TensorStorage::from_buffer(vec![64, 16384], (0..M).map(|i| i as f64).collect());
        bench_one("contig [64,16384] (8MB)", &a);

        // cache-resident 3D
        let a = TensorStorage::from_buffer(vec![64, 64, 256], (0..M).map(|i| i as f64).collect());
        bench_one("contig [64,64,256] (8MB)", &a);

        // larger than L2 (128MB)
        let a = TensorStorage::from_buffer(
            vec![1024, 1024, 16],
            (0..M * 16).map(|i| i as f64).collect(),
        );
        bench_one("contig [1024,1024,16] (128MB)", &a);

        // strided view of an 8MB buffer (transpose): both paths use the odometer
        let c = TensorStorage::from_buffer(vec![1024, 1024], (0..M).map(|i| i as f64).collect());
        let t = TensorStorage::transpose(&c);
        bench_one("strided [1024,1024]^T (8MB)", &t);
    }

    #[test]
    fn sum_strided_view_empty_dims_means_all() {
        // 2x2 contiguous, then transposed into a [2,2] strided view.
        let a = TensorStorage::from_buffer(vec![2, 2], (1..=4).map(|x| x as f64).collect());
        let t = TensorStorage::transpose(&a);
        assert!(!t.contiguous);

        let sum = TensorStorage::sum(&t, &[]);
        assert_eq!(sum.shape, Vec::<usize>::new());
        assert_eq!(sum.buffer.as_ref()[0], 10.0);
    }

    #[test]
    fn sum_empty_dims_matches_explicit_all_dims() {
        let a = TensorStorage::from_buffer(vec![2, 2, 2], (1..=8).map(|x| x as f64).collect());

        let via_empty = TensorStorage::sum(&a, &[]);
        let via_all = TensorStorage::sum(&a, &[0, 1, 2]);
        assert_eq!(via_empty.buffer.as_ref()[0], 36.0);
        assert_eq!(via_empty.buffer.as_ref()[0], via_all.buffer.as_ref()[0]);
        assert_eq!(via_empty.shape, Vec::<usize>::new());
    }

    #[test]
    fn sum_single_dim_and_multi_dims_agree() {
        let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
        // a = [[1,2,3],[4,5,6]]

        let single = TensorStorage::sum(&a, &[0]); // [5,7,9]
        assert_eq!(single.shape, vec![3]);
        assert_eq!(single.buffer.as_ref(), &[5.0, 7.0, 9.0]);

        // single-dim then all-dims equals reducing over {0,1} at once
        let via_multi = TensorStorage::sum(&a, &[0, 1]);
        let via_two_step = TensorStorage::sum(&single, &[0]);
        assert_eq!(via_multi.buffer.as_ref(), via_two_step.buffer.as_ref());
        assert_eq!(via_multi.buffer.as_ref(), &[21.0]);

        // dim order does not matter (sorted internally)
        let swapped = TensorStorage::sum(&a, &[1, 0]);
        assert_eq!(swapped.buffer.as_ref(), &[21.0]);
    }
}
