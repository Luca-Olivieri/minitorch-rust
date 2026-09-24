use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::core::GraphTensor;
use crate::core::nn::module::{Forward1, Module};
use crate::core::tensor::{AbstractTensor, FreeTensor};

/// Inverted-dropout layer with a private, deterministic RNG stream.
///
/// The RNG is derived from the session RNG at construction and then owned by
/// this layer. The mutex permits the existing `Forward1::forward(&self, ...)`
/// API to remain immutable while still advancing the stream on every training
/// call.
pub struct Dropout {
    pub probability: f64,
    pub training: bool,
    rng: Mutex<StdRng>,
}

impl Dropout {
    /// Construct a dropout layer with an independent RNG stream.
    pub fn new(probability: f64, training: bool, session_rng: &mut StdRng) -> Self {
        if !(0.0..1.0).contains(&probability) {
            panic!("Dropout probability must be in [0, 1), got {probability}.");
        }

        Self {
            probability,
            training,
            rng: Mutex::new(StdRng::from_rng(session_rng)),
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn is_training(&self) -> bool {
        self.training
    }

    fn make_mask(&self, input: &GraphTensor, rng: &mut StdRng) -> GraphTensor {
        let scale = 1.0 / (1.0 - self.probability);
        let mut mask = FreeTensor::new(input.shape().clone(), 0.0, false);
        let ndim = input.shape().len();

        for flat in 0..input.numel() {
            let mut remaining = flat;
            let mut coords = vec![0usize; ndim];
            for dim in (0..ndim).rev() {
                coords[dim] = remaining % input.shape()[dim];
                remaining /= input.shape()[dim];
            }

            let value = if rng.random::<f64>() < self.probability {
                0.0
            } else {
                scale
            };
            mask.set(&coords, value);
        }

        mask.to_graph()
    }
}

impl Module for Dropout {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Forward1 for Dropout {
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        if !self.training || self.probability == 0.0 {
            return if no_grad {
                input.detach(false)
            } else {
                input.copy_s()
            };
        }

        let mask = {
            let mut rng = self
                .rng
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.make_mask(input, &mut rng)
        };

        input.mul_with_mode(&mask, no_grad)
    }
}
