pub mod format;
pub mod indexing;
pub mod init;
pub mod iter;
pub mod ops;

use crate::core::dtype::Dtype;
use std::rc::Rc;

#[derive(Debug)]
pub(crate) struct TensorStorage<T: Dtype = f64> {
    pub(crate) buffer: Rc<Vec<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) contiguous: bool,
    pub(crate) numel: usize,
    pub(crate) offset: usize,
}

impl<T: Dtype> TensorStorage<T> {
    pub(crate) fn new(shape: Vec<usize>, fill_value: T) -> Self {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }

        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);

        Self {
            buffer: Rc::new(vec![fill_value; numel]),
            shape,
            strides,
            contiguous: true,
            numel,
            offset: 0,
        }
    }

    /// Build a storage by computing each element from its flat index.
    ///
    /// Unlike [`Self::new`], there is no up-front fill pass: callers that
    /// overwrite every entry anyway (e.g. `one_hot`) write the buffer exactly
    /// once.
    pub(crate) fn from_fn<F>(shape: Vec<usize>, mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }

        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);

        Self {
            buffer: Rc::new((0..numel).map(&mut f).collect()),
            shape,
            strides,
            contiguous: true,
            numel,
            offset: 0,
        }
    }

    /// Wrap an already-completely-initialized contiguous buffer into a storage.
    /// The resulting tensor is a contiguous, offset-0 view of `buffer`.
    pub(crate) fn from_buffer(shape: Vec<usize>, buffer: Vec<T>) -> Self {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }

        let numel = compute_numel_from_shape(&shape);
        if buffer.len() != numel {
            panic!(
                "Buffer length {} does not match shape numel {}.",
                buffer.len(),
                numel
            )
        }

        let strides = init_strides(&shape);

        Self {
            buffer: Rc::new(buffer),
            shape,
            strides,
            contiguous: true,
            numel,
            offset: 0,
        }
    }

    /// Mutable access to the underlying buffer.
    ///
    /// Panics if the buffer is not uniquely owned (i.e. it is still shared with
    /// another view), since mutating it in place would corrupt sibling views.
    /// Freshly allocated storages and detached copies are always uniquely owned.
    pub(crate) fn buffer_mut(&mut self) -> &mut Vec<T> {
        Rc::get_mut(&mut self.buffer)
            .expect("Cannot mutate a buffer that is still shared by multiple storage views.")
    }
}

fn are_dims_positive(shape: &[usize]) -> bool {
    shape.iter().all(|&dim| dim != 0)
}

fn compute_numel_from_shape(shape: &[usize]) -> usize {
    let mut numel: usize = 1;
    for dim in shape {
        numel *= dim;
    }
    numel
}

fn init_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut curr_stride: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = curr_stride;
        curr_stride *= shape[i];
    }

    strides
}
