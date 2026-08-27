pub mod indexing;
pub mod ops;
pub mod format;

// TODO make the numbers generic (not tied to f64)
#[derive(Debug)]
pub struct TensorStorage {
    pub(super) flat_data: Vec<f64>,
    pub shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) contiguous: bool,
    pub numel: usize,
    pub(super) offset: usize
}

impl TensorStorage {

    // TODO: might think of a constructor which does not initialize the whole flat_data, so that you can iterate through it when building a new flat_data
    pub fn new(
        shape: Vec<usize>,
        fill_value: f64
    ) -> Self {

        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }

        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);

        Self {
            flat_data: vec![fill_value; numel],
            shape,
            strides: strides,
            contiguous: true,
            numel: numel,
            offset: 0,
        }
    }

    fn is_contiguous(&self) -> bool {
        let contiguous_strides = init_strides(&self.shape);

        for i in 0..self.shape.len() {
            if self.shape[i] == 1 { continue };
            if self.strides[i] != contiguous_strides[i] { return false; }
        }

        true
    }

    fn item(&self) -> f64 {
        if self.numel != 1 {
            panic!("Cannot call item() on a non-singleton tensor (shape {:?}).", self.shape)
        }

        self.flat_data[self.offset]
    }

    fn copy_d(&self) -> TensorStorage {

        Self {
            flat_data: self.flat_data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            contiguous: self.is_contiguous(), // TODO should I make it contiguous?
            numel: self.numel,
            offset: self.offset,
        }
    }
}

fn are_dims_positive(shape: &Vec<usize>) -> bool {
    shape.iter().all(|&dim| dim != 0)
}

fn compute_numel_from_shape(shape: &Vec<usize>) -> usize {
    let mut numel: usize = 1;
    for dim in shape {
        numel *= dim;
    }
    numel
}

fn init_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut curr_stride: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = curr_stride;
        curr_stride *= shape[i];
    }

    strides
}
