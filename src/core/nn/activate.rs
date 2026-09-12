use crate::core::{
    GraphTensor,
    nn::module::{Forward1, Module},
    tensor::AbstractTensor,
};

pub struct ReLU {}

impl Module for ReLU {}

impl ReLU {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Forward1 for ReLU {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        // the scalar broadcasts to the input shape, so no full-size zeros tensor is allocated
        let zero = GraphTensor::new(vec![], 0.0, false);
        GraphTensor::maximum(input, &zero)
    }
}

pub struct LogSoftmax {}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LogSoftmax {}

impl LogSoftmax {
    pub fn new() -> Self {
        Self {}
    }
}

impl Forward1 for LogSoftmax {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        // log_softmax(x)_i = x_i - max(x) - ln(sum(exp(x - max(x)))) over the last dim.
        // Computing ln(softmax) directly avoids ln(0) -> -inf and the 0 * -inf = NaN
        // that would result from a separate `softmax(x).ln()` pipeline.
        let dim = input.shape().len() - 1; // softmax over the last dimension

        // keepdim: shapes stay broadcastable (…,1) so no manual unsqueeze/expand
        let maxes = input.max(&[dim], true);
        let shifted = input - &maxes;

        let log_denom = shifted.exp().sum(&[dim], true).ln();
        &shifted - &log_denom
    }
}

pub struct Softmax {}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Softmax {}

impl Softmax {
    pub fn new() -> Self {
        Self {}
    }
}

impl Forward1 for Softmax {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let ndim = input.shape().len();
        if ndim == 0 {
            GraphTensor::new(input.shape().clone(), 1.0, false); // TODO does the requires_grad make sense?
        }

        let dim = ndim - 1; // softmax over the last dimension

        // subtract the max over the class dim for numerical stability
        let maxes = input.max(&[dim], true);
        let shifted = input - &maxes;

        // compute exponentials of the shifted values
        let exps = shifted.exp();

        // sum over the target dimension (kept as size-1 for broadcasting)
        let sums = exps.sum(&[dim], true);

        &exps / &sums
    }
}
