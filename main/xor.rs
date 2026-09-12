use std::time::Instant;

use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::Softmax;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};
use minitorch_rust::models::XORClassifier;

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

    println!("{:?}", model.forward(&inputs));

    for epoch in 0..num_epochs {
        let start = Instant::now();
        let logits = model.forward(&inputs);
        let forward_time = start.elapsed();

        let gts_oh = targets.one_hot(logits.shape()[1]);

        let start = Instant::now();
        let loss = criterion.forward(&logits, &gts_oh);
        let loss_time = start.elapsed();

        let start = Instant::now();
        let grads_map = loss.backward(false);
        let backward_time = start.elapsed();

        let start = Instant::now();

        optimizer.step(&mut model, &grads_map);

        let step_time = start.elapsed();

        if epoch % 10 == 0 {
            let prs = softmax.forward(&logits);
            println!("=== [EPOCH {epoch}] === ");
            dbg!(
                forward_time,
                loss_time,
                backward_time,
                step_time,
                GraphTensor::dist(&prs, &gts_oh),
                grads_map.len()
            );
        }
    }
}
