use std::rc::Rc;

use crate::core::storage::*;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;

impl TensorStorage {
    pub fn init_xavier_uniform(shape: Vec<usize>, mut rng: StdRng) -> Self {
        if !are_dims_positive(&shape) {
            panic!("Tensor shape must have positive dimensions. Got {shape:?}.")
        }
        if shape.len() < 2 {
            panic!("Xavier uniform requires at least two dimensions. Got {shape:?}.")
        }
        let numel = compute_numel_from_shape(&shape);
        let strides = init_strides(&shape);
        // Standard generalization: fan_in is the first dim, fan_out the product
        // of the remaining dims. For a [in, out] Linear weight this reduces to
        // the classic sqrt(6 / (in + out)) bound (identical value), and for a
        // conv weight [in_ch, out_ch, kh, kw] it yields
        // sqrt(6 / (in_ch + out_ch * kh * kw)).
        let fan_out: usize = shape[1..].iter().product();
        let limit = (6.0 / (shape[0] + fan_out) as f64).sqrt();
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
