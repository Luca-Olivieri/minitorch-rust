use crate::core::GraphTensor;
use crate::core::nn::activate::LogSoftmax;
use crate::core::nn::module::Forward1;
use crate::core::tensor::AbstractTensor;

pub trait Loss {
    fn forward(&self, inputs: &GraphTensor, targets: &GraphTensor, no_grad: bool) -> GraphTensor;
}

pub struct CrossEntropyLoss {
    log_softmax: LogSoftmax,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            log_softmax: LogSoftmax::new(),
        }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, logits: &GraphTensor, targets: &GraphTensor, no_grad: bool) -> GraphTensor {
        let ndim = logits.shape().len();
        if ndim == 0 {
            let logp = self.log_softmax.forward(logits, no_grad);
            let loss = targets.mul_with_mode(&logp, no_grad).neg_with_mode(no_grad);
            return loss.mean_with_mode(&[], false, no_grad);
        }

        let dim = ndim - 1; // cross-entropy over the last dimension

        // Log probabilities computed directly (numerically stable: no ln(0)).
        let log_probs = self.log_softmax.forward(logits, no_grad);

        // Elementwise multiply with one-hot targets.
        let mul = targets.mul_with_mode(&log_probs, no_grad);

        // Sum over the class dimension and negate.
        let summed = mul.sum_with_mode(&[dim], false, no_grad);
        let loss = summed.neg_with_mode(no_grad);

        loss.mean_with_mode(&[0], false, no_grad)
    }
}
