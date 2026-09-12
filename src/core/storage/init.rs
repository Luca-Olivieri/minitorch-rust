use std::rc::Rc;

use crate::core::storage::*;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;

impl TensorStorage {
    pub fn init_xavier_uniform(shape: Vec<usize>, mut rng: StdRng) -> Self {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }
        if shape.len() != 2 {
            panic!("Xavier uniform requires a 2D [in, out] shape. Got {shape:?}.")
        }
        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);
        let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
        // limit > 0 and finite for positive dims, so the range is always valid.
        let dist = Uniform::new(-limit, limit).unwrap();

        let buffer: Vec<f64> = (0..numel).map(|_| dist.sample(&mut rng)).collect();

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
