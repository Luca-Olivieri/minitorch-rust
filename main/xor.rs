use std::time::Instant;

use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::{ReLU, Softmax};
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};
use minitorch_rust::module;

use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let input_shape = vec![4, 2];
    let mut inputs_f = FreeTensor::new(input_shape.clone(), 0.0, false);
    inputs_f.set(&[0, 0], 0.0);
    inputs_f.set(&[0, 1], 0.0);
    inputs_f.set(&[1, 0], 0.0);
    inputs_f.set(&[1, 1], 1.0);
    inputs_f.set(&[2, 0], 1.0);
    inputs_f.set(&[2, 1], 0.0);
    inputs_f.set(&[3, 0], 1.0);
    inputs_f.set(&[3, 1], 1.0);
    let inputs = inputs_f.to_graph();

    let input_shape = vec![4];
    let mut targets_f = FreeTensor::new(input_shape.clone(), 0.0, true);
    targets_f.set(&[0], 0.0);
    targets_f.set(&[1], 1.0);
    targets_f.set(&[2], 1.0);
    targets_f.set(&[3], 0.0);
    let targets = targets_f.to_graph();

    let rng = StdRng::seed_from_u64(42);

    let mut model = XORClassifier::new(rng);

    let criterion = CrossEntropyLoss::new();

    let optimizer = SGD::new(0.1);

    let num_epochs = 100;

    let softmax = Softmax::new();

    println!("{:?}", model.forward(&inputs, false));

    for epoch in 0..num_epochs {
        let start = Instant::now();
        let logits = model.forward(&inputs, false);
        let forward_time = start.elapsed();

        let gts_oh = targets.one_hot(logits.shape()[1]);

        let start = Instant::now();
        let loss = criterion.forward(&logits, &gts_oh, false);
        let loss_time = start.elapsed();

        let start = Instant::now();
        let grads_map = loss.backward(false);
        let backward_time = start.elapsed();

        let start = Instant::now();

        optimizer.step(&mut model, &grads_map);

        let step_time = start.elapsed();

        if epoch % 10 == 0 {
            let prs = softmax.forward(&logits, false);
            println!("=== [EPOCH {epoch}] === ");
            dbg!(
                forward_time,
                loss_time,
                backward_time,
                step_time,
                GraphTensor::dist(&prs, &gts_oh).item(),
                grads_map.len()
            );
        }
    }
}

module! {
    XORClassifier {
        modules {
            lin1: Linear,
            relu: ReLU,
            lin2: Linear,
            lin3: Linear,
        }
    }
}

impl XORClassifier {
    pub fn new(mut rng: StdRng) -> Self {
        Self {
            lin1: Linear::new(2, 100, true, StdRng::from_rng(&mut rng)),
            relu: ReLU::new(),
            lin2: Linear::new(100, 100, true, StdRng::from_rng(&mut rng)),
            lin3: Linear::new(100, 2, true, StdRng::from_rng(&mut rng)),
        }
    }
}

impl Forward1 for XORClassifier {
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        let input = input.with_no_grad(no_grad);
        let y1 = self.lin1.forward(&input, no_grad);
        let y2 = self.relu.forward(&y1, no_grad);
        let y3 = self.lin2.forward(&y2, no_grad);
        let y4 = self.relu.forward(&y3, no_grad);
        self.lin3.forward(&y4, no_grad) // logits
    }
}
