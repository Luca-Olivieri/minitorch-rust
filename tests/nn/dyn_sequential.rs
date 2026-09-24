use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::dyn_sequential::DynSequential;
use minitorch_rust::core::nn::module::{Forward1, Layer, Module};
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::tensor::AbstractTensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

fn approx(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9,
        "expected {b}, got {a} (diff {})",
        (a - b).abs()
    );
}

#[test]
fn builder_and_vector_constructors_run_the_same_layers() {
    let input = GraphTensor::new(vec![2, 2], 1.0, false);
    let linear = Linear::new(2, 3, true, StdRng::seed_from_u64(7));
    let relu = ReLU::new();
    let expected = relu.forward(&linear.forward(&input));

    let builder = DynSequential::new().with(linear).with(relu);
    let actual = builder.forward(&input);
    assert_eq!(actual.shape(), expected.shape());
    for index in 0..actual.numel() {
        let index = vec![index / 3, index % 3];
        approx(*actual.at(&index), *expected.at(&index));
    }

    let boxed_linear: Box<dyn Layer> = Box::new(Linear::new(2, 3, true, StdRng::seed_from_u64(7)));
    let boxed_relu: Box<dyn Layer> = Box::new(ReLU::new());
    let vector = DynSequential::from_layers(vec![boxed_linear, boxed_relu]);
    assert_eq!(vector.len(), 2);
    assert!(!vector.is_empty());
    assert_eq!(vector.forward(&input).shape(), &[2, 3]);
}

#[test]
fn empty_sequence_is_an_identity_transform() {
    let input = GraphTensor::new(vec![2, 2], 2.5, true);
    let output = DynSequential::new().forward(&input);

    assert_eq!(output.shape(), input.shape());
    assert!(output.requires_grad());
    for row in 0..2 {
        for col in 0..2 {
            approx(*output.at(&[row, col]), *input.at(&[row, col]));
        }
    }
}

#[test]
fn numeric_module_and_parameter_paths_reach_children() {
    let sequence = DynSequential::new()
        .with(Linear::new(2, 3, true, StdRng::seed_from_u64(11)))
        .with(ReLU::new())
        .with(Linear::new(3, 1, false, StdRng::seed_from_u64(12)));

    let mut names = Vec::new();
    sequence.for_each_param(&mut |name, _| names.push(name.to_string()));
    names.sort();
    assert_eq!(names, vec!["0.bias", "0.weight", "2.weight"]);

    assert!(sequence.module("1").is_some());
    assert!(sequence.module("3").is_none());
    assert!(sequence.module_path("0").is_some());
    assert!(sequence.param_path("0.weight").is_some());
    assert!(sequence.param_path("2.weight").is_some());
    assert!(sequence.param_path("1.weight").is_none());
}

#[test]
fn nested_sequences_expose_dotted_paths_and_recursive_grad_flags() {
    let inner = DynSequential::new().with(Linear::new(2, 2, true, StdRng::seed_from_u64(21)));
    let mut outer = DynSequential::new().with(inner).with(ReLU::new());

    assert!(outer.module_path("0.0").is_some());
    assert!(outer.param_path("0.0.weight").is_some());
    assert!(outer.param_path("0.0.bias").is_some());

    outer.set_requires_grad(false, true);
    assert!(!outer.param_path("0.0.weight").unwrap().requires_grad());
    assert!(!outer.param_path("0.0.bias").unwrap().requires_grad());
}

#[test]
fn gradients_and_optimizer_reach_layers_inside_sequence() {
    let mut sequence = DynSequential::new()
        .with(Linear::new(2, 2, true, StdRng::seed_from_u64(31)))
        .with(ReLU::new());
    let input = GraphTensor::new(vec![1, 2], 1.0, true);
    let output = sequence.forward(&input);
    let loss = output.sum(&[], false);
    let grads = loss.backward(false);

    let weight = sequence.param_path("0.weight").unwrap();
    assert!(grads.get(weight).is_some());

    SGD::new(0.1).step(&mut sequence, &grads);
}
