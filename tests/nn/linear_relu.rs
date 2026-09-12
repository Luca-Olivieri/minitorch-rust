use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::core::GraphTensor;

use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn linear_relu_forward_and_backward() {
    let rng = StdRng::seed_from_u64(42);

    let lin = Linear::new(3, 4, true, rng);
    let relu = ReLU::new();

    let x = GraphTensor::new(vec![2, 3], 1.0, true);

    let a = lin.forward(&x);
    let b = relu.forward(&a);

    let grads_map = b.backward(true);

    // Relu clamps the two negative logits of the seeded Linear layer to 0.
    assert_eq!(b.shape(), &vec![2, 4]);
    let expected_b = vec![
        vec![0.0, 0.0, 1.4924861769132234, 0.48383863009522066],
        vec![0.0, 0.0, 1.4924861769132234, 0.48383863009522066],
    ];
    for i in 0..2 {
        for j in 0..4 {
            let v = *b.at(&vec![i, j]);
            assert!(
                (v - expected_b[i][j]).abs() < 1e-9,
                "b[{i}][{j}] = {v}, expected {}",
                expected_b[i][j]
            );
        }
    }

    // dx = W^T @ drelu, where the relu mask zeroes the dead units.
    let dx = grads_map.get(&x.to_key()).unwrap();
    assert_eq!(dx.shape(), &vec![2, 3]);
    let expected_dx = vec![
        vec![0.07844817624333589, 1.086312925282642, 0.8115637054824661],
        vec![0.07844817624333589, 1.086312925282642, 0.8115637054824661],
    ];
    for i in 0..2 {
        for j in 0..3 {
            let v = *dx.at(&vec![i, j]);
            assert!(
                (v - expected_dx[i][j]).abs() < 1e-9,
                "dx[{i}][{j}] = {v}, expected {}",
                expected_dx[i][j]
            );
        }
    }

    // x.grad shuts off ReLU's dead inputs: it is linear in x through `a`,
    // and its gradient wrt x is constant (disconnected node).
    let dx_grads_map = dx.backward(true);
    assert!(
        !dx_grads_map.contains_key(&x.to_key()),
        "d( dx )/dx should be disconnected"
    );
}
