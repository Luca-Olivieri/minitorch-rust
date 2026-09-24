use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::flatten::Flatten;
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::AbstractTensor;

#[test]
fn flatten_defaults_to_batch_preserving_mode() {
    let input = GraphTensor::new(vec![2, 3, 4, 5], 1.5, true);
    let output = Flatten::new().forward(&input, false);

    assert_eq!(output.shape(), &[2, 60]);
    approx(*output.at(&[1, 59]), *input.at(&[1, 2, 3, 4]));
}

#[test]
fn flatten_supports_signed_custom_dimensions() {
    let input = GraphTensor::new(vec![2, 3, 4, 5], 1.0, false);

    let partial = Flatten::with_dims(1, 2).forward(&input, false);
    assert_eq!(partial.shape(), &[2, 12, 5]);

    let all = Flatten::new_with_dims(0, -1).forward(&input, false);
    assert_eq!(all.shape(), &[120]);
}

#[test]
fn flatten_backward_restores_the_input_shape() {
    let input = GraphTensor::new(vec![2, 3, 4, 5], 1.0, true);
    let output = Flatten::new().forward(&input, false);
    let grads = output.sum(&[], false).backward(true);
    let dx = grads.get(&input).unwrap();

    assert_eq!(dx.shape(), input.shape());
    assert_eq!(dx.at(&[1, 2, 3, 4]), &1.0);
}

#[test]
#[should_panic]
fn flatten_rejects_invalid_dimension_order() {
    let input = GraphTensor::new(vec![2, 3, 4, 5], 1.0, false);
    Flatten::with_dims(2, 1).forward(&input, false);
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}
