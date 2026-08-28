use std::rc::Rc;

use crate::core::storage::{TensorStorage, compute_numel_from_shape};

impl TensorStorage {
    pub fn copy_d(a: &TensorStorage) -> TensorStorage {
        TensorStorage {
            flat_data: Rc::new(a.flat_data.as_ref().clone()),
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

         // build output shape
        let out_shape = unsqueeze_shape(&a.shape, dim);

        // TODO inefficient: creates the whole tensors, then overrides some of its values

        // make a view: share the underlying flat data and keep the same offset
        let mut out = TensorStorage::new(out_shape, 0.0);
        out.flat_data = Rc::clone(&a.flat_data);
        out.offset = a.offset;

        out.strides = a.strides.clone();
        out.strides.insert(dim, 0);

        out.contiguous = a.contiguous;
        out.numel = compute_numel_from_shape(&out.shape);

        out
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

        // build output shape
        let out_shape = squeeze_shape(&a.shape, dim);

        // make a view: share the underlying flat data and keep the same offset
        let mut out = TensorStorage::new(out_shape, 0.0);
        out.flat_data = Rc::clone(&a.flat_data);
        out.offset = a.offset;

        // remove the stride corresponding to the squeezed dim
        out.strides = a.strides.clone();
        out.strides.insert(dim, 0);

        out.contiguous = a.contiguous;
        out.numel = compute_numel_from_shape(&out.shape);

        out
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

        // build output shape
        let mut out_shape = a.shape.clone();
        out_shape[dim] = times;

        // make a view: share the underlying flat data and keep the same offset
        let mut out = TensorStorage::new(out_shape, 0.0);

        out.flat_data = Rc::clone(&a.flat_data);
        out.offset = a.offset;

        out.strides = a.strides.clone();
        out.strides[dim] = 0; // this stride maps to the same underlying element

        out.contiguous = a.contiguous;
        out.numel = compute_numel_from_shape(&out.shape);

        out
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
