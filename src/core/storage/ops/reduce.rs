use crate::core::dtype::{Dtype, Float, Numeric};
use crate::core::storage::TensorStorage;

/// Flat maximum-index metadata for a 2D max-pooling output.
///
/// `indices` stores all logical input indices that attain a window maximum in
/// output order. `offsets[i]..offsets[i + 1]` is the tied-index group for
/// output element `i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaxPool2dMetadata {
    pub(crate) indices: Vec<usize>,
    pub(crate) offsets: Vec<usize>,
}

impl<T: Numeric> TensorStorage<T> {
    /// Sum over every dimension in `dims`, yielding a tensor with those
    /// dimensions removed. An empty `dims` aggregates over all dimensions.
    ///
    /// Dispatch:
    /// - all dims     -> flat/scaled `sum_all` pass (no intermediate allocations)
    /// - a single dim -> tight strided loop (`reduce_single_dim`)
    /// - otherwise    -> general stride-odometer pass (`reduce_dims`)
    pub fn sum(a: &TensorStorage<T>, dims: &[usize]) -> TensorStorage<T> {
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
    pub fn max(a: &TensorStorage<T>, dims: &[usize]) -> TensorStorage<T> {
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

    /// Argmax along `dim`, yielding the per-slice flat index as an `f64` tensor.
    ///
    /// The output is `f64` (not `T`) because index-valued results are used as
    /// labels/masks on the float path; keeping them in this library's float
    /// idiom leaves the dtype choice to the caller's `.cast`.
    pub fn argmax(a: &TensorStorage<T>, dim: usize) -> TensorStorage<f64> {
        reduce_dims(
            a,
            &[dim],
            |v| (v, 0usize),
            |(v, i), val| if val > v { (val, i + 1) } else { (v, i) },
            |(_, i)| i as f64,
        )
    }

    /// One-hot encode `source`'s labels into a fresh contiguous storage of shape
    /// `source.shape ++ [num_classes]`, validating the labels as they are read.
    ///
    /// Labels may be floats or integers; the output is `f64` (`1.0`/`0.0`), the
    /// representation the softmax-loss path consumes.
    pub fn one_hot(source: &TensorStorage<T>, num_classes: usize) -> TensorStorage<f64>
    where
        T: OneHotLabel,
    {
        let mut out_shape = source.shape.clone();
        out_shape.push(num_classes);

        // Validate while reading; the flat output index of (input i, class c) is
        // `i * num_classes + c`, so materializing the labels lets `from_fn` write
        // the output buffer exactly once (no up-front zero-fill).
        let labels: Vec<usize> = source
            .strided_indices()
            .enumerate()
            .map(|(i, f)| {
                let cls = source.buffer[f].as_class();
                if cls >= num_classes {
                    panic!(
                        "One-hotting with num_classes={} but found an out-of-range label at index {}.",
                        num_classes, i
                    )
                }
                cls
            })
            .collect();

        TensorStorage::from_fn(out_shape, |f| {
            if labels[f / num_classes] == f % num_classes {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Reduce over every dimension in a single pass, yielding a scalar `[]`.
    fn sum_all(a: &TensorStorage<T>) -> TensorStorage<T> {
        if a.contiguous {
            // Flat slice fold: auto-vectorizes cleanly and does not allocate.
            let total = a.buffer[a.offset..a.offset + a.numel]
                .iter()
                .fold(T::ZERO, |acc, &v| acc + v);
            TensorStorage::from_buffer(Vec::new(), vec![total])
        } else {
            // Strided view: walk the reduced dims with the stride odometer instead
            // of per-element div/mod logical indexing (~3x faster in practice).
            let dims: Vec<usize> = (0..a.shape.len()).collect();
            reduce_dims(a, &dims, |v| v, |acc, v| acc + v, |acc| acc)
        }
    }

    /// Max over every dimension in a single pass, yielding a scalar `[]`.
    pub fn max_all(a: &TensorStorage<T>) -> TensorStorage<T> {
        if a.contiguous {
            // Flat slice fold seeded with the first element: no stride
            // bookkeeping, vectorizes the cmp/select.
            let slice = &a.buffer[a.offset..a.offset + a.numel];
            let total = slice
                .iter()
                .skip(1)
                .fold(slice[0], |acc, v| if *v > acc { *v } else { acc });
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

impl<T: Float> TensorStorage<T> {
    /// 2D average pooling over a `[batch, channel, height, width]` input.
    ///
    /// Every `kernel = (kh, kw)` window starting at spatial position
    /// `(oh * stride.0, ow * stride.1)` is averaged, producing a fresh
    /// `[batch, channel, out_h, out_w]` output with
    /// `out_h = (height - kh) / stride.0 + 1` and
    /// `out_w = (width - kw) / stride.1 + 1` (PyTorch's floor mode; the window
    /// must fit entirely in the input — there is no padding).
    ///
    /// The input may be a strided view (e.g. a transposed batch): the window is
    /// walked through `a`'s own strides, so logical layout is preserved.
    pub fn avg_pool2d(
        a: &TensorStorage<T>,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> TensorStorage<T> {
        if a.shape.len() != 4 {
            panic!(
                "avg_pool2d expects a [batch, channel, height, width] input, got shape {:?}.",
                a.shape
            );
        }

        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            panic!(
                "avg_pool2d kernel and stride components must be nonzero, got kernel {kernel:?} stride {stride:?}."
            );
        }

        let (h, w) = (a.shape[2], a.shape[3]);
        if kh > h || kw > w {
            panic!(
                "avg_pool2d kernel {kernel:?} extends past the input height x width ({h} x {w})."
            );
        }

        let (b, c) = (a.shape[0], a.shape[1]);
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        let inv_kernel = T::from_f64(1.0 / (kh * kw) as f64);

        let mut out_buf = Vec::with_capacity(b * c * out_h * out_w);
        let (s0, s1, s2, s3) = (a.strides[0], a.strides[1], a.strides[2], a.strides[3]);

        for batch in 0..b {
            let b_base = a.offset + batch * s0;
            for chan in 0..c {
                let plane = b_base + chan * s1;
                for oh in 0..out_h {
                    let h_base = plane + oh * sh * s2;
                    for ow in 0..out_w {
                        let w_base = h_base + ow * sw * s3;
                        let mut acc = T::ZERO;
                        for i in 0..kh {
                            let row = w_base + i * s2;
                            for j in 0..kw {
                                acc += a.buffer[row + j * s3];
                            }
                        }
                        out_buf.push(acc * inv_kernel);
                    }
                }
            }
        }

        TensorStorage::from_buffer(vec![b, c, out_h, out_w], out_buf)
    }

    /// 2D maximum pooling over a `[batch, channel, height, width]` input.
    ///
    /// This value-only entry point does not build maximum-index metadata. The
    /// graph-backed tensor operation uses [`Self::max_pool2d_with_indices`] when
    /// its backward pass needs tied-maximum locations.
    pub fn max_pool2d(
        a: &TensorStorage<T>,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> TensorStorage<T> {
        Self::max_pool2d_impl(a, kernel, stride, false).0
    }

    /// 2D maximum pooling plus flat metadata for every maximum in each output
    /// window.
    ///
    /// Indices are flat indices in the input's logical row-major shape, not
    /// physical buffer offsets. This lets the cached maxima work for strided
    /// input views as well as contiguous storage.
    pub fn max_pool2d_with_indices(
        a: &TensorStorage<T>,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> (TensorStorage<T>, MaxPool2dMetadata) {
        Self::max_pool2d_impl(a, kernel, stride, true)
    }

    fn max_pool2d_impl(
        a: &TensorStorage<T>,
        kernel: (usize, usize),
        stride: (usize, usize),
        collect_indices: bool,
    ) -> (TensorStorage<T>, MaxPool2dMetadata) {
        if a.shape.len() != 4 {
            panic!(
                "max_pool2d expects a [batch, channel, height, width] input, got shape {:?}.",
                a.shape
            );
        }

        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            panic!(
                "max_pool2d kernel and stride components must be nonzero, got kernel {kernel:?} stride {stride:?}."
            );
        }

        let (b, c) = (a.shape[0], a.shape[1]);
        let (h, w) = (a.shape[2], a.shape[3]);
        if kh > h || kw > w {
            panic!(
                "max_pool2d kernel {kernel:?} extends past the input height x width ({h} x {w})."
            );
        }

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let out_numel = b * c * out_h * out_w;
        let mut out_buf = Vec::with_capacity(out_numel);
        let mut metadata = MaxPool2dMetadata {
            indices: Vec::new(),
            offsets: Vec::new(),
        };
        if collect_indices {
            metadata.indices.reserve(out_numel);
            metadata.offsets.reserve(out_numel + 1);
            metadata.offsets.push(0);
        }
        let (s0, s1, s2, s3) = (a.strides[0], a.strides[1], a.strides[2], a.strides[3]);

        for batch in 0..b {
            let b_base = a.offset + batch * s0;
            for chan in 0..c {
                let plane = b_base + chan * s1;
                for oh in 0..out_h {
                    let h_base = plane + oh * sh * s2;
                    for ow in 0..out_w {
                        let w_base = h_base + ow * sw * s3;
                        let window_start = ((batch * c + chan) * h + oh * sh) * w + ow * sw;
                        let mut max_value = a.buffer[w_base];
                        let group_start = metadata.indices.len();
                        if collect_indices {
                            metadata.indices.push(window_start);
                        }

                        for i in 0..kh {
                            let row = w_base + i * s2;
                            for j in 0..kw {
                                if i == 0 && j == 0 {
                                    continue;
                                }

                                let value = a.buffer[row + j * s3];
                                let input_index = window_start + i * w + j;
                                if value > max_value {
                                    max_value = value;
                                    if collect_indices {
                                        metadata.indices.truncate(group_start);
                                        metadata.indices.push(input_index);
                                    }
                                } else if value == max_value && collect_indices {
                                    metadata.indices.push(input_index);
                                }
                            }
                        }

                        out_buf.push(max_value);
                        if collect_indices {
                            metadata.offsets.push(metadata.indices.len());
                        }
                    }
                }
            }
        }

        (
            TensorStorage::from_buffer(vec![b, c, out_h, out_w], out_buf),
            metadata,
        )
    }

    /// Gradient of [`Self::max_pool2d`] using the cached maximum positions.
    ///
    /// Each output gradient is split evenly among all input positions that
    /// attained that output window's maximum. Contributions from overlapping
    /// windows accumulate on a fresh input-shaped buffer.
    pub fn max_pool2d_backward(
        dy: &TensorStorage<T>,
        input_shape: &[usize],
        kernel: (usize, usize),
        stride: (usize, usize),
        max_indices: &MaxPool2dMetadata,
    ) -> TensorStorage<T> {
        if input_shape.len() != 4 {
            panic!(
                "max_pool2d backward expects a [batch, channel, height, width] input_shape, got {:?}.",
                input_shape
            );
        }

        let (b, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 || kh > h || kw > w {
            panic!(
                "max_pool2d backward has invalid kernel {kernel:?} or stride {stride:?} for input shape {input_shape:?}."
            );
        }

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let expected_shape = [b, c, out_h, out_w];
        if dy.shape.as_slice() != expected_shape {
            panic!(
                "max_pool2d backward: the upstream gradient has shape {:?}, expected {:?}.",
                dy.shape, expected_shape
            );
        }
        let expected_offsets = dy.numel + 1;
        if max_indices.offsets.len() != expected_offsets {
            panic!(
                "max_pool2d backward expected {} maximum-index offset boundaries, got {} for {} outputs.",
                expected_offsets,
                max_indices.offsets.len(),
                dy.numel
            );
        }
        if max_indices.offsets.first().copied() != Some(0)
            || max_indices.offsets.last().copied() != Some(max_indices.indices.len())
            || max_indices
                .offsets
                .windows(2)
                .any(|bounds| bounds[0] > bounds[1])
        {
            panic!("max_pool2d backward received invalid maximum-index offsets.");
        }

        let mut out_buf = vec![T::ZERO; input_shape.iter().product()];
        for (dy_index, bounds) in dy.strided_indices().zip(max_indices.offsets.windows(2)) {
            let start = bounds[0];
            let end = bounds[1];
            if start == end {
                panic!("max_pool2d backward encountered an empty maximum-index group.");
            }
            let share = dy.buffer[dy_index] * T::from_f64(1.0 / (end - start) as f64);
            for &input_index in &max_indices.indices[start..end] {
                if input_index >= out_buf.len() {
                    panic!(
                        "max_pool2d backward encountered out-of-range cached input index {input_index}."
                    );
                }
                out_buf[input_index] += share;
            }
        }

        TensorStorage::from_buffer(input_shape.to_vec(), out_buf)
    }

    /// Gradient of [`Self::avg_pool2d`] with respect to its input.
    ///
    /// Each upstream gradient element `dy[b, c, oh, ow]` is scattered into every
    /// input position its window covered, scaled by `1 / (kh * kw)`. Positions
    /// covered by several overlapping windows accumulate (onto a fresh
    /// zero-filled buffer). The result is a fresh `[batch, channel, height,
    /// width]` buffer matching `input_shape`.
    pub fn avg_pool2d_backward(
        dy: &TensorStorage<T>,
        input_shape: &[usize],
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> TensorStorage<T> {
        if input_shape.len() != 4 {
            panic!(
                "avg_pool2d backward expects a [batch, channel, height, width] input_shape, got {:?}.",
                input_shape
            );
        }

        let (b, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        if dy.shape.as_slice() != [b, c, out_h, out_w] {
            panic!(
                "avg_pool2d backward: the upstream gradient has shape {:?}, expected [b={b}, c={c}, out_h={out_h}, out_w={out_w}].",
                dy.shape
            );
        }

        let inv_kernel = T::from_f64(1.0 / (kh * kw) as f64);

        let mut out_buf = vec![T::ZERO; input_shape.iter().product()];
        let (s0, s1, s2, s3) = (dy.strides[0], dy.strides[1], dy.strides[2], dy.strides[3]);

        for batch in 0..b {
            let b_base = dy.offset + batch * s0;
            for chan in 0..c {
                let d_plane = b_base + chan * s1;
                let out_plane = (batch * c + chan) * h * w;
                for oh in 0..out_h {
                    let d_h = d_plane + oh * s2;
                    for ow in 0..out_w {
                        let g = dy.buffer[d_h + ow * s3] * inv_kernel;
                        let oh_row = oh * sh;
                        let ow_col = ow * sw;
                        for i in 0..kh {
                            let row = out_plane + (oh_row + i) * w + ow_col;
                            for j in 0..kw {
                                out_buf[row + j] += g;
                            }
                        }
                    }
                }
            }
        }

        TensorStorage::from_buffer(input_shape.to_vec(), out_buf)
    }
}

/// Elements that can act as one-hot class labels: any float with an integral,
/// non-negative value, or any integer. The float forms validate while reading;
/// integers are labels by construction.
pub trait OneHotLabel: Dtype {
    /// Convert to a class index, panicking on a non-label value.
    fn as_class(self) -> usize;
}

impl OneHotLabel for f32 {
    fn as_class(self) -> usize {
        if self.fract() != 0.0 {
            panic!("One-hotted tensor has value {self} with fractional part.")
        }
        if self < 0.0 {
            panic!("One-hotted tensor has negative value {self}.")
        }
        self as usize
    }
}

impl OneHotLabel for f64 {
    fn as_class(self) -> usize {
        if self.fract() != 0.0 {
            panic!("One-hotted tensor has value {self} with fractional part.")
        }
        if self < 0.0 {
            panic!("One-hotted tensor has negative value {self}.")
        }
        self as usize
    }
}

macro_rules! impl_one_hot_signed {
    ($($t:ty),+ $(,)?) => {
        $(
            impl OneHotLabel for $t {
                fn as_class(self) -> usize {
                    if self < 0 {
                        panic!("One-hotted tensor has negative value {self}.")
                    }
                    self as usize
                }
            }
        )+
    };
}

macro_rules! impl_one_hot_unsigned {
    ($($t:ty),+ $(,)?) => {
        $(
            impl OneHotLabel for $t {
                fn as_class(self) -> usize {
                    self as usize
                }
            }
        )+
    };
}

impl_one_hot_signed!(i8, i16, i32, i64);
impl_one_hot_unsigned!(u8, u16, u32, u64);

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
pub fn reduce_dims<T: Numeric, U: Dtype, A, F, G>(
    a: &TensorStorage<T>,
    dims: &[usize],
    seed: impl Fn(T) -> A,
    fold: F,
    finish: G,
) -> TensorStorage<U>
where
    F: Fn(A, T) -> A,
    G: Fn(A) -> U,
{
    let dims = normalize_dims(dims, &a.shape);

    // single reduced dim: the tight strided loop beats the general odometer for
    // cache-resident tensors, and it is by far the most common case.
    if let [dim] = dims.as_slice() {
        return reduce_single_dim(a, *dim, seed, fold, finish);
    }

    // build output shape
    let out_shape = squeezed_shape(&a.shape, &dims);

    let out_numel: usize = out_shape.iter().product();
    let mut out_buf: Vec<U> = Vec::with_capacity(out_numel);
    let a_buf = &a.buffer;

    // elements along the reduced dims are `reduced_strides[k]` apart in flat space
    let reduced_total: usize = dims.iter().map(|&d| a.shape[d]).product();
    let reduced_sizes: Vec<usize> = dims.iter().map(|&d| a.shape[d]).collect();
    let reduced_strides: Vec<usize> = dims.iter().map(|&d| a.strides[d]).collect();
    let bases = base_offsets(a, &dims);

    // odometer over the reduced dims; reset per output slice
    let mut coords = vec![0usize; dims.len()];

    // iterate over output logical indices
    for base in bases {
        let mut f = base;
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

        out_buf.push(finish(acc));
    }

    TensorStorage::from_buffer(out_shape, out_buf)
}

/// Tight single-dimension reduction: consecutive elements along the reduced dim
/// are `a.strides[dim]` apart in flat space, so the inner loop is just a strided
/// walk with no odometer bookkeeping.
fn reduce_single_dim<T: Numeric, U: Dtype, A, F, G>(
    a: &TensorStorage<T>,
    dim: usize,
    seed: impl Fn(T) -> A,
    fold: F,
    finish: G,
) -> TensorStorage<U>
where
    F: Fn(A, T) -> A,
    G: Fn(A) -> U,
{
    // build output shape
    let out_shape = squeezed_shape(&a.shape, &[dim]);

    let out_numel: usize = out_shape.iter().product();
    let mut out_buf: Vec<U> = Vec::with_capacity(out_numel);
    let a_buf = &a.buffer;

    let reduced_stride = a.strides[dim];
    let bases = base_offsets(a, &[dim]);

    // iterate over output logical indices
    for base in bases {
        let mut f = base;
        let mut acc = seed(a_buf[f]);
        for _ in 1..a.shape[dim] {
            f += reduced_stride;
            acc = fold(acc, a_buf[f]);
        }

        out_buf.push(finish(acc));
    }

    TensorStorage::from_buffer(out_shape, out_buf)
}

/// Flat offset of the first element of each output slice, walked in output
/// row-major order.
///
/// Uses an odometer over the non-reduced dims to avoid any per-element division
/// or multi-dim index allocation.
fn base_offsets<T: Dtype>(a: &TensorStorage<T>, dims: &[usize]) -> Vec<usize> {
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
