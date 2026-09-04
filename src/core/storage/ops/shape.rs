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
            offset: a.offset
        }
    }

    pub fn copy_s(a: &TensorStorage) -> TensorStorage {
        TensorStorage {
            buffer: Rc::clone(&a.buffer),
            shape: a.shape.clone(),
            strides: a.strides.clone(),
            contiguous: a.contiguous,
            numel: a.numel,
            offset: a.offset
        }
    }

    pub fn unsqueeze(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        if dim > a.shape.len() {
            panic!("Unsqueezed dimension {} out of range for shape of length {:?}.", dim, a.shape);
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

    pub fn squeeze(
        a: &TensorStorage,
        dim: usize
    ) -> TensorStorage {
        if dim >= a.shape.len() {
            panic!("Squeezed dimension {} out of range for shape of length {:?}.", dim, a.shape);
        }

        if a.shape[dim] != 1 {
            panic!("Squeezed dimension {} must be singleton. Got size {:?}.", dim, a.shape);
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

    pub fn expand(
        a: &TensorStorage,
        dim: usize,
        times: usize
    ) -> TensorStorage {
        if dim >= a.shape.len() {
            panic!("Expanded dimension {} out of range for shape of length {:?}.", dim, a.shape);
        }

        if a.shape[dim] != 1 {
            panic!("Expanded dimension {} must be singleton. Got size {:?}.", dim, a.shape);
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
            contiguous: out_contiguous,
            numel: a.numel*times,
            offset: a.offset,
        }
    }
}

fn unsqueeze_shape(
    shape: &Vec<usize>,
    dim: usize
) -> Vec<usize> {
    let mut out_shape = shape.clone();
    out_shape.insert(dim, 1);
    out_shape
}

pub(crate) fn squeeze_shape(
    shape: &Vec<usize>,
    dim: usize
) -> Vec<usize> {
    let mut out_shape = shape.clone();
    out_shape.remove(dim);
    out_shape
}
