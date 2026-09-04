use crate::core::{GraphTensor, nn::{activate::Softmax, module::{Forward1, Module}}, tensor::AbstractTensor};

pub trait Loss {
    fn forward(
        &self,
        inputs: &GraphTensor,
        targets: &GraphTensor,
    ) -> GraphTensor;
}

pub struct CrossEntropyLoss {
    softmax: Softmax
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self { softmax: Softmax::new() }
    }
}

impl Loss for CrossEntropyLoss {

    fn forward(
        &self,
        preds: &GraphTensor,
        targets: &GraphTensor,
    ) -> GraphTensor {
        let ndim = preds.shape().len();
        if ndim == 0 {
            let probs = self.softmax.forward(preds);
            let logp = probs.ln();
            let loss = -&(targets * &logp);
            return loss.mean_dim(0);
        }

        let dim = ndim - 1; // cross-entropy over the last dimension;

        // compute probabilities via softmax
        let probs = self.softmax.forward(preds);

        // log probabilities
        let log_probs = probs.ln();

        // elementwise multiply with targets (expects one-hot targets)
        let mul = targets * &log_probs;

        // sum over class dimension and take negative
        let summed = mul.sum_dim(dim);
        let loss = -&summed;

        loss.mean_dim(0)
    }
}
