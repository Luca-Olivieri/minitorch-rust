pub mod format;
pub mod indexing;
pub mod init;
pub mod ops;

use std::rc::Rc;

// TODO make the numbers generic (not tied to f64)
#[derive(Debug)]
pub struct TensorStorage {
    pub buffer: Rc<Vec<f64>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub contiguous: bool,
    pub numel: usize,
    pub(super) offset: usize,
}

impl TensorStorage {
    // TODO: might think of a constructor which does not initialize the whole flat_data, so that you can iterate through it when building a new flat_data
    pub fn new(shape: Vec<usize>, fill_value: f64) -> Self {
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

    /// Wrap an already-completely-initialized contiguous buffer into a storage.
    /// The resulting tensor is a contiguous, offset-0 view of `buffer`.
    pub fn from_buffer(shape: Vec<usize>, buffer: Vec<f64>) -> Self {
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
    pub(super) fn buffer_mut(&mut self) -> &mut Vec<f64> {
        Rc::get_mut(&mut self.buffer)
            .expect("Cannot mutate a buffer that is still shared by multiple storage views.")
    }
}

fn are_dims_positive(shape: &[usize]) -> bool {
    shape.iter().all(|&dim| dim != 0)
}

fn compute_numel_from_shape(shape: &Vec<usize>) -> usize {
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
