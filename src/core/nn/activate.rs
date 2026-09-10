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
        // the scalar broadcasts to the input shape, so no full-size zeros tensor is allocated
        let zero = GraphTensor::new(vec![], 0.0, false);
        GraphTensor::maximum(input, &zero)
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

        // subtract the max over the class dim for numerical stability
        let maxes = input.max_dim(dim);
        let shifted = input - &maxes.unsqueeze(dim).expand(dim, input.shape()[dim]);

        // compute exponentials of the shifted values
        let exps = shifted.exp();

        // sum over the target dimension and broadcast for division
        let sums = &exps.sum_dim(dim);
        let denom = sums.unsqueeze(dim).expand(dim, input.shape()[dim]);

        &exps / &denom
    }
}
