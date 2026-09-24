use crate::core::dtype::{Dtype, Numeric};
use crate::core::storage::compute_numel_from_shape;
use std::rc::Rc;

use crate::core::storage::TensorStorage;

impl<T: Dtype> TensorStorage<T> {
    /// Deep copy: duplicate the underlying flat data so the result shares no
    /// buffer with `a`.
    pub fn copy_d(a: &TensorStorage<T>) -> TensorStorage<T> {
        TensorStorage {
            buffer: Rc::new(a.buffer.as_ref().clone()),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }

    /// Shallow copy: a view that shares the underlying flat data with `a`.
    pub fn copy_s(a: &TensorStorage<T>) -> TensorStorage<T> {
        TensorStorage {
            buffer: Rc::clone(&a.buffer),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }
}

impl<T: Dtype> TensorStorage<T> {
    pub fn unsqueeze(a: &TensorStorage<T>, dim: usize) -> TensorStorage<T> {
        if dim > a.shape.len() {
            panic!(
                "Unsqueezed dimension {} out of range for shape of length {:?}.",
                dim, a.shape
            );
        }

        let mut out_strides = a.strides.clone();
        out_strides.insert(dim, 0);

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&a.buffer),
            shape: unsqueeze_shape(&a.shape, dim),
            strides: out_strides,
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }

    /// Re-insert a size-1 axis for every dimension in `dims` (in increasing
    /// order), turning a squeezed reduction result back into a keepdim view.
    /// Purely metadata: no data is copied.
    pub fn unsqueeze_at(a: &TensorStorage<T>, dims: &[usize]) -> TensorStorage<T> {
        if dims.windows(2).any(|w| w[0] >= w[1]) {
            panic!(
                "Dimensions {:?} must be strictly increasing for unsqueeze_at, otherwise the re-inserted axes end up in the wrong positions.",
                dims
            );
        }

        let mut out = Self::copy_s(a);
        for &d in dims {
            out = Self::unsqueeze(&out, d);
        }
        out
    }

    /// Remove the size-1 axis for every dimension in `dims` (given in increasing
    /// order), the inverse of [`Self::unsqueeze_at`]. The highest dims are
    /// removed first so the remaining indices stay valid. Purely metadata: no
    /// data is copied.
    #[allow(dead_code)]
    pub fn squeeze_at(a: &TensorStorage<T>, dims: &[usize]) -> TensorStorage<T> {
        if dims.windows(2).any(|w| w[0] >= w[1]) {
            panic!(
                "Dimensions {:?} must be strictly increasing for squeeze_at, otherwise the removed axes end up in the wrong positions.",
                dims
            );
        }

        // Remove highest dims first so the remaining indices stay valid.
        let mut out = Self::copy_s(a);
        for &d in dims.iter().rev() {
            out = Self::squeeze(&out, d);
        }
        out
    }

    pub fn squeeze(a: &TensorStorage<T>, dim: usize) -> TensorStorage<T> {
        if dim >= a.shape.len() {
            panic!(
                "Squeezed dimension {} out of range for shape of length {:?}.",
                dim, a.shape
            );
        }

        if a.shape[dim] != 1 {
            panic!(
                "Squeezed dimension {} must be singleton. Got size {:?}.",
                dim, a.shape
            );
        }

        // remove the stride corresponding to the squeezed dim
        let mut out_strides = a.strides.clone();
        out_strides.remove(dim);

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&a.buffer),
            shape: squeeze_shape(&a.shape, dim),
            strides: out_strides,
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }

    pub fn transpose(a: &TensorStorage<T>, dim_a: usize, dim_b: usize) -> TensorStorage<T> {
        if dim_a >= a.shape.len() || dim_b >= a.shape.len() {
            panic!(
                "Transposed dimensions {} and {} out of range for shape of length {:?}.",
                dim_a,
                dim_b,
                a.shape.len()
            );
        }

        // transposing a dim with itself is a no-op
        if dim_a == dim_b {
            return Self::copy_s(a);
        }

        let mut out_shape = a.shape.clone();
        out_shape.swap(dim_a, dim_b);

        let mut out_strides = a.strides.clone();
        out_strides.swap(dim_a, dim_b);

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&a.buffer),
            shape: out_shape,
            strides: out_strides,
            contiguous: false,
            numel: a.numel,
            offset: a.offset,
        }
    }

    pub fn expand(a: &TensorStorage<T>, dim: usize, times: usize) -> TensorStorage<T> {
        if dim >= a.shape.len() {
            panic!(
                "Expanded dimension {} out of range for shape of length {:?}.",
                dim, a.shape
            );
        }

        if a.shape[dim] != 1 {
            panic!(
                "Expanded dimension {} must be singleton. Got size {:?}.",
                dim, a.shape
            );
        }

        let mut out_shape = a.shape.clone();
        let mut out_strides = a.strides.clone();
        let mut out_contiguous = true;

        if times != 1 {
            out_shape[dim] = times; // expand the dimension
            out_strides[dim] = 0; // this stride maps to the same underlying element
            out_contiguous = false;
        }

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&a.buffer),
            shape: out_shape,
            strides: out_strides,
            contiguous: out_contiguous && a.contiguous,
            numel: a.numel * times,
            offset: a.offset,
        }
    }

    /// Stack a list of tensors along a new dimension 0.
    ///
    /// All input tensors must have identical shapes. The output has shape
    /// `[N, original_shape...]` where `N` is the number of inputs.
    pub fn stack(storages: &[&TensorStorage<T>]) -> TensorStorage<T> {
        if storages.is_empty() {
            panic!("Cannot stack an empty list of tensors.");
        }

        let elem_shape = storages[0].shape.clone();
        for s in storages.iter().skip(1) {
            if s.shape != elem_shape {
                panic!(
                    "All tensors must have the same shape to stack. Got {:?} vs {:?}.",
                    elem_shape, s.shape
                );
            }
        }

        let n = storages.len();
        let elem_numel = storages[0].numel;

        let mut out_shape = Vec::with_capacity(elem_shape.len() + 1);
        out_shape.push(n);
        out_shape.extend_from_slice(&elem_shape);

        let mut out_buf = Vec::with_capacity(elem_numel * n);
        for s in storages {
            if s.contiguous {
                out_buf.extend_from_slice(&s.buffer[s.offset..s.offset + elem_numel]);
            } else {
                out_buf.extend(s.strided_indices().map(|f| s.buffer[f]));
            }
        }

        TensorStorage::from_buffer(out_shape, out_buf)
    }

    #[allow(dead_code)]
    pub fn broadcast(&self, b: &TensorStorage<T>) -> TensorStorage<T> {
        Self::broadcast_to_shape(self, &b.shape)
    }

    pub fn broadcast_to_shape(&self, shape: &[usize]) -> TensorStorage<T> {
        if !self.is_broadcastable(shape) {
            panic!(
                "Shape {:?} cannot be broadcasted to {:?}",
                self.shape, shape
            );
        }

        // Prepend 1s to align dimensions from the right (NumPy convention)
        let ndim_diff = shape.len() as isize - self.shape.len() as isize;
        let mut out_shape = self.shape.clone();
        let mut out_strides = self.strides.clone();

        for _ in 0..ndim_diff {
            out_shape.insert(0, 1);
            out_strides.insert(0, 0);
        }

        // Broadcast singleton dims (right-aligned now)
        let mut out_contiguous = true;
        for d in 0..shape.len() {
            if out_shape[d] == 1 && out_shape[d] != shape[d] {
                out_strides[d] = 0;
                out_contiguous = false;
                out_shape[d] = shape[d];
            }
        }

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&self.buffer),
            shape: out_shape,
            strides: out_strides,
            // The result is contiguous only if the source was AND this call did
            // not stretch any dimension; otherwise `offset + i` indexing is invalid.
            contiguous: out_contiguous && self.contiguous,
            numel: compute_numel_from_shape(shape),
            offset: self.offset,
        }
    }

    fn is_broadcastable(&self, shape: &[usize]) -> bool {
        let ndim_diff = self.shape.len() as isize - shape.len() as isize;

        // source has more dims than target: extra source dims must be 1
        if ndim_diff > 0 {
            for d in 0..(ndim_diff as usize) {
                if self.shape[d] != 1 {
                    return false;
                }
            }
        }

        let offset = self.shape.len().abs_diff(shape.len());
        let shorter = if ndim_diff >= 0 { shape } else { &self.shape };
        let longer = if ndim_diff >= 0 { &self.shape } else { shape };

        shorter
            .iter()
            .zip(longer.iter().skip(offset))
            .all(|(a, b)| a == b || *a == 1 || *b == 1)
    }
}

impl<T: Numeric> TensorStorage<T> {
    /// Zero-pad every dimension by `pads[d] = (before, after)` elements.
    ///
    /// The result is a fresh contiguous storage (a view could not fold pre/post
    /// offsets into a single stride). `pads` must provide exactly one pair per
    /// dimension; padded regions read as `T::ZERO`.
    pub fn pad(a: &TensorStorage<T>, pads: &[(usize, usize)]) -> TensorStorage<T> {
        let ndim = a.shape.len();
        if pads.len() != ndim {
            panic!(
                "pad requires exactly one (before, after) pair per dimension: got {} pairs for a shape of length {}.",
                pads.len(),
                ndim
            );
        }

        let out_shape: Vec<usize> = a
            .shape
            .iter()
            .zip(pads)
            .map(|(&dim, &(before, after))| dim + before + after)
            .collect();
        let out_numel = compute_numel_from_shape(&out_shape);

        let out_buf = (0..out_numel)
            .map(|f| pad_element(a, pads, &out_shape, f))
            .collect();

        TensorStorage::from_buffer(out_shape, out_buf)
    }

    /// Extract the window `[start, start + len)` along every dimension,
    /// materializing it into a fresh contiguous buffer (not a strided view).
    ///
    /// `ranges` must provide exactly one `(start, len)` pair per dimension, and
    /// every window must stay in bounds.
    pub fn slice(a: &TensorStorage<T>, ranges: &[(usize, usize)]) -> TensorStorage<T> {
        let ndim = a.shape.len();
        if ranges.len() != ndim {
            panic!(
                "slice requires exactly one (start, len) pair per dimension: got {} pairs for a shape of length {}.",
                ranges.len(),
                ndim
            );
        }
        for (d, &(start, len)) in ranges.iter().enumerate() {
            if start + len > a.shape[d] {
                panic!(
                    "slice range (start={start}, len={len}) exceeds dimension {d} of size {}.",
                    a.shape[d]
                );
            }
        }

        let out_shape: Vec<usize> = ranges.iter().map(|&(_, len)| len).collect();
        let out_numel = compute_numel_from_shape(&out_shape);

        let out_buf = (0..out_numel)
            .map(|f| slice_element(a, ranges, f))
            .collect();

        TensorStorage::from_buffer(out_shape, out_buf)
    }

    /// Extract a strided window from every dimension, materializing the result
    /// into a fresh contiguous buffer.
    ///
    /// Each range is `(start, length, step)`. The output has the same rank as
    /// `a`, with each dimension replaced by its requested length. The input may
    /// itself be a strided view; logical coordinates are always resolved through
    /// `a`'s strides.
    pub fn slice_strided(
        a: &TensorStorage<T>,
        ranges: &[(usize, usize, usize)],
    ) -> TensorStorage<T> {
        let ndim = a.shape.len();
        if ranges.len() != ndim {
            panic!(
                "slice_strided requires exactly one (start, length, step) range per dimension: got {} ranges for a shape of length {}.",
                ranges.len(),
                ndim
            );
        }

        for (dim, &(start, length, step)) in ranges.iter().enumerate() {
            if length == 0 || step == 0 {
                panic!(
                    "slice_strided range length and step must be nonzero, got ({start}, {length}, {step}) for dimension {dim}."
                );
            }
            let last = start + (length - 1) * step;
            if last >= a.shape[dim] {
                panic!(
                    "slice_strided range ({start}, {length}, {step}) exceeds dimension {dim} of size {}.",
                    a.shape[dim]
                );
            }
        }

        let out_shape: Vec<usize> = ranges.iter().map(|&(_, length, _)| length).collect();
        let out_numel = compute_numel_from_shape(&out_shape);
        let out_buf = (0..out_numel)
            .map(|flat| {
                let mut remaining = flat;
                let mut input_offset = a.offset;
                for dim in (0..ndim).rev() {
                    let (start, length, step) = ranges[dim];
                    let coord = remaining % length;
                    remaining /= length;
                    input_offset += (start + coord * step) * a.strides[dim];
                }
                a.buffer[input_offset]
            })
            .collect();

        TensorStorage::from_buffer(out_shape, out_buf)
    }

    /// Gradient of [`Self::slice_strided`] with respect to its input.
    ///
    /// The upstream gradient is scattered back to each sampled input position;
    /// overlapping output windows accumulate.
    pub fn slice_strided_backward(
        dy: &TensorStorage<T>,
        input_shape: &[usize],
        ranges: &[(usize, usize, usize)],
    ) -> TensorStorage<T> {
        let ndim = input_shape.len();
        if ranges.len() != ndim {
            panic!(
                "slice_strided backward requires exactly one range per dimension: got {} ranges for a shape of length {}.",
                ranges.len(),
                ndim
            );
        }

        let out_shape: Vec<usize> = ranges.iter().map(|&(_, length, _)| length).collect();
        if dy.shape != out_shape {
            panic!(
                "slice_strided backward: the upstream gradient has shape {:?}, expected {:?}.",
                dy.shape, out_shape
            );
        }

        let mut input_strides = vec![1usize; ndim];
        let mut stride = 1usize;
        for dim in (0..ndim).rev() {
            input_strides[dim] = stride;
            stride *= input_shape[dim];
        }

        let mut out_buf = vec![T::ZERO; input_shape.iter().product()];
        for (flat, dy_offset) in (0..dy.numel).zip(dy.strided_indices()) {
            let mut remaining = flat;
            let mut input_index = 0usize;
            for dim in (0..ndim).rev() {
                let (start, length, step) = ranges[dim];
                let coord = remaining % length;
                remaining /= length;
                input_index += (start + coord * step) * input_strides[dim];
            }
            out_buf[input_index] += dy.buffer[dy_offset];
        }

        TensorStorage::from_buffer(input_shape.to_vec(), out_buf)
    }

    /// Reinterpret the logical elements under a new shape, materializing into a
    /// fresh contiguous buffer.
    ///
    /// A reshape *materializes*: it fixes any strided view into canonical
    /// row-major order, which is what makes it safe to feed transposed or
    /// broadcast views into kernels that assume contiguity. `new_shape` must
    /// preserve the element count.
    pub fn reshape(a: &TensorStorage<T>, new_shape: &[usize]) -> TensorStorage<T> {
        let new_numel = compute_numel_from_shape(new_shape);
        if new_numel != a.numel {
            panic!(
                "reshape cannot change the number of elements: {} elements can't become shape {:?} ({new_numel}).",
                a.numel, new_shape
            );
        }

        let out_buf: Vec<T> = if a.contiguous {
            a.buffer[a.offset..a.offset + a.numel].to_vec()
        } else {
            a.strided_indices().map(|f| a.buffer[f]).collect()
        };

        TensorStorage::from_buffer(new_shape.to_vec(), out_buf)
    }
}

/// Value at flat output index `f` of the padded tensor: `T::ZERO` in the pad
/// region, the source element (through its strides) otherwise. Output
/// coordinates are recovered with div/mod, so this is O(ndim) per element.
fn pad_element<T: Numeric>(
    a: &TensorStorage<T>,
    pads: &[(usize, usize)],
    out_shape: &[usize],
    f: usize,
) -> T {
    let ndim = a.shape.len();
    let mut flat = f;
    let mut in_flat = a.offset;
    for d in (0..ndim).rev() {
        let (before, _) = pads[d];
        let coord = flat % out_shape[d];
        flat /= out_shape[d];

        let in_coord = coord as isize - before as isize;
        if in_coord < 0 || in_coord >= a.shape[d] as isize {
            return T::ZERO;
        }
        in_flat += in_coord as usize * a.strides[d];
    }
    a.buffer[in_flat]
}

/// Source element at flat output index `f` of the sliced tensor, resolved
/// through the source strides (the source may be a strided view).
fn slice_element<T: Dtype>(a: &TensorStorage<T>, ranges: &[(usize, usize)], f: usize) -> T {
    let ndim = a.shape.len();
    let mut flat = f;
    let mut in_flat = a.offset;
    for d in (0..ndim).rev() {
        let (start, len) = ranges[d];
        let coord = flat % len;
        flat /= len;
        in_flat += (start + coord) * a.strides[d];
    }
    a.buffer[in_flat]
}

fn unsqueeze_shape(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut out_shape = shape.to_owned();
    out_shape.insert(dim, 1);
    out_shape
}

pub(crate) fn squeeze_shape(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut out_shape = shape.to_owned();
    out_shape.remove(dim);
    out_shape
}
