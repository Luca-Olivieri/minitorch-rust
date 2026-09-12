use crate::core::storage::compute_numel_from_shape;
use std::rc::Rc;

use crate::core::storage::TensorStorage;

impl TensorStorage {
    pub fn copy_d(a: &TensorStorage) -> TensorStorage {
        TensorStorage {
            buffer: Rc::new(a.buffer.as_ref().clone()),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }

    pub fn copy_s(a: &TensorStorage) -> TensorStorage {
        TensorStorage {
            buffer: Rc::clone(&a.buffer),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset,
        }
    }

    pub fn unsqueeze(a: &TensorStorage, dim: usize) -> TensorStorage {
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
    pub fn unsqueeze_at(a: &TensorStorage, dims: &[usize]) -> TensorStorage {
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
    pub fn squeeze_at(a: &TensorStorage, dims: &[usize]) -> TensorStorage {
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

    pub fn squeeze(a: &TensorStorage, dim: usize) -> TensorStorage {
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

    pub fn transpose(a: &TensorStorage) -> TensorStorage {
        if a.shape.len() != 2 {
            panic!("Transpose requires a 2D tensor, got shape {:?}.", a.shape);
        }

        let mut out_strides = a.strides.clone();
        out_strides.swap(0, 1);

        // make a view: share the underlying flat data and keep the same offset
        Self {
            buffer: Rc::clone(&a.buffer),
            shape: vec![a.shape[1], a.shape[0]],
            strides: out_strides,
            contiguous: false,
            numel: a.numel,
            offset: a.offset,
        }
    }

    pub fn expand(a: &TensorStorage, dim: usize, times: usize) -> TensorStorage {
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
    pub fn stack(storages: &[&TensorStorage]) -> TensorStorage {
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
            for i in 0..elem_numel {
                out_buf.push(s[i]);
            }
        }

        TensorStorage::from_buffer(out_shape, out_buf)
    }

    pub fn broadcast(&self, b: &TensorStorage) -> TensorStorage {
        Self::broadcast_to_shape(self, &b.shape)
    }

    pub fn broadcast_to_shape(&self, shape: &Vec<usize>) -> TensorStorage {
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

    fn is_broadcastable(&self, shape: &Vec<usize>) -> bool {
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
