use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::compute::MaxPool2d;
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};

fn approx(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9,
        "expected {b}, got {a} (diff {})",
        (a - b).abs()
    );
}

/// Consecutive values `0..numel` in row-major order.
fn ramped(shape: &[usize]) -> GraphTensor {
    let mut tensor = FreeTensor::new(shape.to_vec(), 0.0, true);
    let mut counter = 0.0;

    fn walk(shape: &[usize], coords: &mut Vec<usize>, tensor: &mut FreeTensor, counter: &mut f64) {
        if coords.len() == shape.len() {
            tensor.set(coords, *counter);
            *counter += 1.0;
            return;
        }

        let dim = coords.len();
        for index in 0..shape[dim] {
            coords.push(index);
            walk(shape, coords, tensor, counter);
            coords.pop();
        }
    }

    walk(shape, &mut Vec::new(), &mut tensor, &mut counter);
    tensor.to_graph()
}

fn unflatten(shape: &[usize], flat: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = flat;
    for dim in (0..shape.len()).rev() {
        coords[dim] = remaining % shape[dim];
        remaining /= shape[dim];
    }
    coords
}

#[test]
fn max_pool2d_forward_supports_overlap_and_rectangular_windows() {
    let input = ramped(&[1, 1, 3, 3]);

    let out = input.max_pool2d((2, 2), (1, 1));
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    let expected = [4.0, 5.0, 7.0, 8.0];
    for (flat, value) in expected.iter().enumerate() {
        let index = unflatten(out.shape(), flat);
        approx(*out.at(&index), *value);
    }

    let rectangular = input.max_pool2d((3, 2), (1, 1));
    assert_eq!(rectangular.shape(), &[1, 1, 1, 2]);
    approx(*rectangular.at(&[0, 0, 0, 0]), 7.0);
    approx(*rectangular.at(&[0, 0, 0, 1]), 8.0);
}

#[test]
fn max_pool2d_backward_splits_gradient_between_tied_maxima() {
    let input = GraphTensor::wrap(vec![vec![vec![vec![1.0, 4.0], vec![4.0, 2.0]]]], true);
    let out = input.max_pool2d((2, 2), (1, 1));
    let grads = out.sum(&[], false).backward(true);

    let dx = grads.get(&input).unwrap();
    assert_eq!(dx.shape(), &[1, 1, 2, 2]);
    let expected = [0.0, 0.5, 0.5, 0.0];
    for (flat, value) in expected.iter().enumerate() {
        let index = unflatten(dx.shape(), flat);
        approx(*dx.at(&index), *value);
    }
}

#[test]
fn max_pool2d_backward_accumulates_overlapping_windows() {
    let input = ramped(&[1, 1, 3, 3]);
    let out = input.max_pool2d((2, 2), (1, 1));
    let loss = (&out * &out).sum(&[], false);
    let grads = loss.backward(true);

    let dx = grads.get(&input).unwrap();
    let expected = [
        0.0, 0.0, 0.0, //
        0.0, 8.0, 10.0, //
        0.0, 14.0, 16.0,
    ];
    assert_eq!(dx.shape(), &[1, 1, 3, 3]);
    for (flat, value) in expected.iter().enumerate() {
        let index = unflatten(dx.shape(), flat);
        approx(*dx.at(&index), *value);
    }
}

#[test]
fn max_pool2d_is_generic_over_float_dtype() {
    let input = GraphTensor::<f32>::wrap(vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]], true);
    let out = input.max_pool2d((2, 2), (2, 2));

    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!((*out.at(&[0, 0, 0, 0]) - 4.0).abs() < 1e-6);
}

#[test]
fn maxpool2d_module_defaults_stride_to_kernel() {
    let pool = MaxPool2d::new(2, None);
    let input = GraphTensor::new(vec![1, 1, 4, 4], 2.0, false);
    let out = pool.forward(&input);

    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    for row in 0..2 {
        for col in 0..2 {
            approx(*out.at(&[0, 0, row, col]), 2.0);
        }
    }
}

#[test]
fn maxpool2d_module_accepts_explicit_stride() {
    let pool = MaxPool2d::new(2, Some(1));
    let input = GraphTensor::new(vec![1, 1, 4, 4], 2.0, false);
    let out = pool.forward(&input);

    assert_eq!(out.shape(), &[1, 1, 3, 3]);
}

#[test]
fn max_pool2d_propagates_requires_grad_only_for_grad_inputs() {
    let input = ramped(&[1, 1, 4, 4]);
    assert!(input.max_pool2d((2, 2), (2, 2)).requires_grad());

    let frozen = GraphTensor::new(vec![1, 1, 4, 4], 1.0, false);
    assert!(!frozen.max_pool2d((2, 2), (2, 2)).requires_grad());
}

#[test]
#[should_panic]
fn max_pool2d_rejects_non_4d_input() {
    ramped(&[1, 2, 4]).max_pool2d((2, 2), (2, 2));
}
