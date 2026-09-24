use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::dropout::Dropout;
use minitorch_rust::core::nn::dyn_sequential::DynSequential;
use minitorch_rust::core::nn::module::{Forward1, Module};
use minitorch_rust::core::tensor::AbstractTensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}

#[test]
fn dropout_is_reproducible_from_the_session_seed() {
    let input = GraphTensor::new(vec![2, 4], 1.0, true);

    let mut rng_a = StdRng::seed_from_u64(42);
    let dropout_a = Dropout::new(0.3, true, &mut rng_a);
    let mut rng_b = StdRng::seed_from_u64(42);
    let dropout_b = Dropout::new(0.3, true, &mut rng_b);

    let output_a = dropout_a.forward(&input);
    let output_b = dropout_b.forward(&input);
    assert_eq!(output_a.shape(), output_b.shape());
    for index in 0..output_a.numel() {
        let coords = vec![index / 4, index % 4];
        approx(*output_a.at(&coords), *output_b.at(&coords));
    }
}

#[test]
fn dropout_eval_is_identity_and_training_can_be_toggled() {
    let input = GraphTensor::new(vec![2, 3], 2.0, true);
    let mut rng = StdRng::seed_from_u64(7);
    let mut dropout = Dropout::new(0.5, true, &mut rng);
    assert!(dropout.is_training());

    dropout.eval();
    assert!(!dropout.is_training());
    let output = dropout.forward(&input);
    assert_eq!(output.shape(), input.shape());
    assert!(output.requires_grad());
    for row in 0..2 {
        for col in 0..3 {
            approx(*output.at(&[row, col]), *input.at(&[row, col]));
        }
    }

    dropout.train();
    let stochastic = dropout.forward(&input);
    assert_eq!(stochastic.shape(), input.shape());
}

#[test]
fn dropout_backward_uses_the_same_inverted_mask() {
    let input = GraphTensor::new(vec![1, 4], 1.0, true);
    let mut rng = StdRng::seed_from_u64(11);
    let dropout = Dropout::new(0.5, true, &mut rng);

    let output = dropout.forward(&input);
    let grads = output.sum(&[], false).backward(true);
    let dx = grads.get(&input).unwrap();

    for index in 0..4 {
        let coords = vec![0, index];
        approx(*dx.at(&coords), *output.at(&coords));
    }
}

#[test]
fn module_set_training_propagates_into_a_sequence() {
    let input = GraphTensor::new(vec![2, 4], 1.0, true);
    let mut rng = StdRng::seed_from_u64(13);
    let mut sequence = DynSequential::new().with(Dropout::new(0.3, true, &mut rng));

    sequence.set_training(false);
    let eval_output = sequence.forward(&input);
    for index in 0..eval_output.numel() {
        let coords = vec![index / 4, index % 4];
        approx(*eval_output.at(&coords), *input.at(&coords));
    }

    sequence.set_training(true);
    let train_output = sequence.forward(&input);
    assert_eq!(train_output.shape(), input.shape());
}

#[test]
#[should_panic]
fn dropout_rejects_probability_one() {
    let mut rng = StdRng::seed_from_u64(1);
    let _ = Dropout::new(1.0, true, &mut rng);
}

#[test]
#[should_panic]
fn dropout_rejects_negative_probability() {
    let mut rng = StdRng::seed_from_u64(1);
    let _ = Dropout::new(-0.1, true, &mut rng);
}
