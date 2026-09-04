use std::collections::HashMap;

use crate::core::{GraphTensor, nn::module::{Forward1, Module}, tensor::AbstractTensor};

pub struct ReLU {}

impl Module for ReLU {}

impl ReLU {

    pub fn new() -> Self { Self {} }
}

impl Forward1 for ReLU {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let ones = GraphTensor::new(input.shape().clone(), 0.0, false); // TODO substitute with tensor broadcasted from f64 value of 1.0
        GraphTensor::maximum(input, &ones)
    }
}

pub struct Softmax {}

impl Module for Softmax {}

impl Softmax {

    pub fn new() -> Self { Self {} }
}

impl Forward1 for Softmax {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let ndim = input.shape().len();
        if ndim == 0 {
            GraphTensor::new(input.shape().clone(), 1.0, false); // TODO does the requires_grad make sense?
        }

        let dim = ndim - 1; // softmax over the last dimension

        // compute exponentials using e^x via elementwise pow with base e
        let e_const = GraphTensor::new(input.shape().clone(), std::f64::consts::E, false);
        let exps = e_const.pow(input);

        // sum over the target dimension and broadcast for division
        let sums = &exps.sum_dim(dim);
        let denom = sums.unsqueeze(dim).expand(dim, input.shape()[dim]);

        &exps / &denom
    }
}
