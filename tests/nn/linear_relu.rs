use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::AbstractTensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn no_grad_forward_disables_graph_edges() {
    let layer = Linear::new(3, 4, true, StdRng::seed_from_u64(42));
    let input = GraphTensor::new(vec![2, 3], 1.0, true);

    let train_output = layer.forward(&input, false);
    assert!(train_output.requires_grad());

    let eval_output = layer.forward(&input, true);
    assert!(!eval_output.requires_grad());
    assert_eq!(eval_output.shape(), train_output.shape());
    for index in 0..eval_output.numel() {
        let coords = vec![index / 4, index % 4];
        assert_eq!(eval_output.at(&coords), train_output.at(&coords));
    }

    let targets = GraphTensor::wrap(vec![0.0, 1.0], false);
    let loss = CrossEntropyLoss::new().forward(&eval_output, &targets.one_hot(4), true);
    assert!(!loss.requires_grad());
}

#[test]
fn linear_relu_forward_and_backward() {
    let rng = StdRng::seed_from_u64(42);

    let lin = Linear::new(3, 4, true, rng);
    let relu = ReLU::new();

    let x = GraphTensor::new(vec![2, 3], 1.0, true);

    let a = lin.forward(&x, false);
    let b = relu.forward(&a, false);

    let grads_map = b.backward(true);

    // Relu clamps the two negative logits of the seeded Linear layer to 0.
    assert_eq!(b.shape(), &[2, 4]);
    let expected_b = [
        [0.0, 0.0, 1.4924861769132234, 0.48383863009522066],
        [0.0, 0.0, 1.4924861769132234, 0.48383863009522066],
    ];
    for (i, row) in expected_b.iter().enumerate() {
        for (j, exp) in row.iter().enumerate() {
            let v = *b.at(&[i, j]);
            assert!((v - exp).abs() < 1e-9, "b[{i}][{j}] = {v}, expected {exp}");
        }
    }

    // dx = W^T @ drelu, where the relu mask zeroes the dead units.
    let dx = grads_map.get(&x).unwrap();
    assert_eq!(dx.shape(), &[2, 3]);
    let expected_dx = [
        [0.07844817624333589, 1.086312925282642, 0.8115637054824661],
        [0.07844817624333589, 1.086312925282642, 0.8115637054824661],
    ];
    for (i, row) in expected_dx.iter().enumerate() {
        for (j, exp) in row.iter().enumerate() {
            let v = *dx.at(&[i, j]);
            assert!((v - exp).abs() < 1e-9, "dx[{i}][{j}] = {v}, expected {exp}");
        }
    }

    // x.grad shuts off ReLU's dead inputs: it is linear in x through `a`,
    // and its gradient wrt x is constant (disconnected node).
    let dx_grads_map = dx.backward(true);
    assert!(
        dx_grads_map.get(&x).is_none(),
        "d( dx )/dx should be disconnected"
    );
}
