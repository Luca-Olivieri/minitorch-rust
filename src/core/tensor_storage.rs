use std::panic;
use std::fmt::self;

// TODO make the numbers generic (not tied to f64)
#[derive(Debug)]
pub struct TensorStorage {
    flat_data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    contiguous: bool,
    numel: usize,
    offset: usize
}

impl TensorStorage {

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64
    ) -> Self {
        // assert_positive_dims(shape);;
        //

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
}

impl fmt::Display for TensorStorage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self))
    }
}

fn are_dims_positive(shape: &Vec<usize>) -> bool {
    // shape.iter().all(|&dim| dim != 0)

    for dim in shape {
        if *dim == 0 {
            return false
        }
    }

    true
}

fn compute_numel_from_shape(shape: &Vec<usize>) -> usize {
    let mut numel: usize = 1;
    for dim in shape {
        numel *= dim;
    }
    numel
}

fn init_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec!(0; shape.len());
    let mut curr_stride: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = curr_stride;
        curr_stride *= shape[i];
    }

    strides
}
