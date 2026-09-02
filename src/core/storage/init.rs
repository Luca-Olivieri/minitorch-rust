use std::rc::Rc;

use crate::core::storage::*;
use rand::rngs::StdRng;
use rand::distr::{Distribution, Uniform};


impl TensorStorage {

    pub fn init_xavier_uniform(
        shape: Vec<usize>,
        mut rng: StdRng
    ) -> Self {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }
        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);
        let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
        let dist = Uniform::new(-limit, limit).unwrap(); // TODO: proper error handling

        let buffer: Vec<f64> = (0..numel)
            .map(|_| dist.sample(&mut rng))
            .collect();

        Self {
            buffer: Rc::new(buffer),
            shape,
            strides,
            contiguous: true,
            numel,
            offset: 0,
        }
    }
}

// void xavier_uniform_inplace(
//         Tensor& x,
//         std::mt19937&& rng
// ) {
//     // limit = sqrt(6 / (in + out))

//     for (size_t i {0}; i < x.numel(); ++i) {
//         x.m_node->m_storage.get_entry_ref(i) = dist(rng);
//     }
// }
