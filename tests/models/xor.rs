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
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let y1 = self.lin1.forward(input);
        let y2 = self.relu.forward(&y1);
        let y3 = self.lin2.forward(&y2);
        let y4 = self.relu.forward(&y3);
        self.lin3.forward(&y4) // logits
    }
}

fn xor_inputs() -> GraphTensor {
    let mut inputs_f = FreeTensor::new(vec![4, 2], 0.0, false);
    inputs_f.set(&[0, 0], 0.0);
    inputs_f.set(&[0, 1], 0.0);
    inputs_f.set(&[1, 0], 0.0);
    inputs_f.set(&[1, 1], 1.0);
    inputs_f.set(&[2, 0], 1.0);
    inputs_f.set(&[2, 1], 0.0);
    inputs_f.set(&[3, 0], 1.0);
    inputs_f.set(&[3, 1], 1.0);
    inputs_f.to_graph()
}

fn xor_targets() -> GraphTensor {
    let mut targets_f = FreeTensor::new(vec![4], 0.0, true);
    targets_f.set(&[0], 0.0);
    targets_f.set(&[1], 1.0);
    targets_f.set(&[2], 1.0);
    targets_f.set(&[3], 0.0);
    targets_f.to_graph()
}

#[test]
fn xor_training_reduces_loss_and_tracks_recorded_trajectory() {
    let inputs = xor_inputs();
    let targets = xor_targets();

    let rng = StdRng::seed_from_u64(42);

    let mut model = XORClassifier::new(rng);

    let criterion = CrossEntropyLoss::new();
    let optimizer = SGD::new(0.1);
    let softmax = Softmax::new();

    let num_epochs = 100;

    // Forward the untrained model: records the seeded initialisation state.
    let init_logits = model.forward(&inputs);
    assert_eq!(init_logits.shape(), &[4, 2]);
    let expected_init = [
        [0.0, 0.0],
        [0.06010813653256624, 0.17175688032447833],
        [0.1660415734331862, 0.14839191059980794],
        [0.1101211666294248, 0.25343994988589197],
    ];
    for (i, row) in expected_init.iter().enumerate() {
        for (j, exp) in row.iter().enumerate() {
            let v = *init_logits.at(&[i, j]);
            assert!(
                (v - exp).abs() < 1e-9,
                "init logits[{i}][{j}] = {v}, expected {exp}"
            );
        }
    }

    // Loss/dist trajectory recorded at every 20th epoch (1-indexed).
    let expected_trajectory = [
        (1usize, 0.7003525557795608, 1.4243411767860652),
        (21, 0.5824893354350618, 1.2489189194072574),
        (41, 0.453531611298184, 1.0316319256430848),
        (61, 0.3008931265917528, 0.7354627886236067),
        (81, 0.17208980779325977, 0.4478359035369914),
    ];

    let mut recorded = Vec::new();
    for epoch in 1..=num_epochs {
        let logits = model.forward(&inputs);

        let oh = targets.one_hot(logits.shape()[1]);
        let loss = criterion.forward(&logits, &oh);
        let grads_map = loss.backward(false);

        optimizer.step(&mut model, &grads_map);

        if (epoch - 1) % 20 == 0 {
            let prs = softmax.forward(&logits);
            recorded.push((epoch, loss.item(), GraphTensor::dist(&prs, &oh).item()));
        }
    }

    assert_eq!(recorded.len(), expected_trajectory.len());
    for (actual, expected) in recorded.iter().zip(expected_trajectory.iter()) {
        assert_eq!(actual.0, expected.0, "checkpoint epoch mismatch");
        assert!(
            (actual.1 - expected.1).abs() < 1e-9,
            "loss at epoch {}: got {}, expected {}",
            actual.0,
            actual.1,
            expected.1
        );
        assert!(
            (actual.2 - expected.2).abs() < 1e-9,
            "dist at epoch {}: got {}, expected {}",
            actual.0,
            actual.2,
            expected.2
        );
    }

    // The run must monotonically improve as well, not just match old traces.
    for w in recorded.windows(2) {
        assert!(w[1].1 < w[0].1, "loss increased between checkpoints");
        assert!(w[1].2 < w[0].2, "dist increased between checkpoints");
    }
}
