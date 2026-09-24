use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::{Conv2d, Conv2dPadding, Linear, MaxPool2d};
use minitorch_rust::core::nn::dropout::Dropout;
use minitorch_rust::core::nn::dyn_sequential::DynSequential;
use minitorch_rust::core::nn::flatten::Flatten;
use minitorch_rust::core::nn::module::{Forward1, Module};
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::tensor::AbstractTensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

fn small_cnn(session_rng: &mut StdRng, training: bool) -> DynSequential {
    DynSequential::new()
        .with(Conv2d::new_with_options(
            1,
            32,
            3,
            1,
            Conv2dPadding::Same,
            1,
            true,
            StdRng::from_rng(session_rng),
        ))
        .with(ReLU::new())
        .with(MaxPool2d::new(2, None))
        .with(Conv2d::new_with_options(
            32,
            64,
            3,
            1,
            Conv2dPadding::Same,
            1,
            true,
            StdRng::from_rng(session_rng),
        ))
        .with(ReLU::new())
        .with(MaxPool2d::new(2, None))
        .with(Flatten::new())
        .with(Dropout::new(0.3, training, session_rng))
        .with(Linear::new(
            64 * 7 * 7,
            128,
            true,
            StdRng::from_rng(session_rng),
        ))
        .with(ReLU::new())
        .with(Linear::new(128, 10, true, StdRng::from_rng(session_rng)))
}

#[test]
fn small_cnn_eval_forward_has_expected_shape() {
    let mut rng = StdRng::seed_from_u64(101);
    let model = small_cnn(&mut rng, false);
    let input = GraphTensor::new(vec![2, 1, 28, 28], 0.25, true);
    let output = model.forward(&input);

    assert_eq!(output.shape(), &[2, 10]);
}

#[test]
fn small_cnn_training_backward_reaches_all_parameterized_layers() {
    let mut rng = StdRng::seed_from_u64(102);
    let mut model = small_cnn(&mut rng, true);
    let input = GraphTensor::new(vec![2, 1, 28, 28], 0.25, true);
    let output = model.forward(&input);
    let loss = output.sum(&[], false);
    let grads = loss.backward(false);

    for path in [
        "0.weight",
        "0.bias",
        "3.weight",
        "3.bias",
        "8.weight",
        "8.bias",
        "10.weight",
        "10.bias",
    ] {
        let parameter = model.param_path(path).unwrap();
        assert!(
            grads.get(parameter).is_some(),
            "missing gradient for {path}"
        );
    }

    SGD::new(1e-3).step(&mut model, &grads);
}

#[test]
fn small_cnn_is_reproducible_from_the_session_seed() {
    let mut rng_a = StdRng::seed_from_u64(103);
    let model_a = small_cnn(&mut rng_a, true);
    let mut rng_b = StdRng::seed_from_u64(103);
    let model_b = small_cnn(&mut rng_b, true);
    let input = GraphTensor::new(vec![1, 1, 28, 28], 0.5, false);

    let output_a = model_a.forward(&input);
    let output_b = model_b.forward(&input);
    for index in 0..output_a.numel() {
        let coords = vec![0, index];
        assert!((*output_a.at(&coords) - *output_b.at(&coords)).abs() < 1e-12);
    }
}
