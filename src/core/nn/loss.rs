use crate::core::GraphTensor;
use crate::core::nn::activate::LogSoftmax;
use crate::core::nn::module::Forward1;
use crate::core::tensor::AbstractTensor;

pub trait Loss {
    fn forward(&self, inputs: &GraphTensor, targets: &GraphTensor) -> GraphTensor;
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

impl Loss for CrossEntropyLoss {
    fn forward(&self, logits: &GraphTensor, targets: &GraphTensor) -> GraphTensor {
        let ndim = logits.shape().len();
        if ndim == 0 {
            let logp = self.log_softmax.forward(logits);
            let loss = -&(targets * &logp);
            return loss.mean_dim(0);
        }

        let dim = ndim - 1; // cross-entropy over the last dimension;

        // log probabilities computed directly (numerically stable: no ln(0))
        let log_probs = self.log_softmax.forward(logits);

        // elementwise multiply with targets (expects one-hot targets)
        let mul = targets * &log_probs;

        // sum over class dimension and take negative
        let summed = mul.sum_dim(dim);
        let loss = -&summed;

        loss.mean_dim(0)
    }
}
