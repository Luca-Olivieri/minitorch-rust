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
    pub fn max_all(a: &TensorStorage) -> TensorStorage {
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
///
/// Exposed for the kernel benchmarks in `tests/storage/reduce.rs`.
#[doc(hidden)]
pub fn reduce_dims<A, F, G>(
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
