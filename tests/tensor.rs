#[path = "tensor/math.rs"]
mod math;
#[path = "tensor/reduce.rs"]
mod reduce;
#[path = "tensor/shape.rs"]
mod shape;

use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::core::GraphTensor;

#[test]
fn from_vec_scalar() {
    let t = GraphTensor::wrap(5.0, false);
    assert_eq!(t.shape(), &vec![]);
    assert_eq!(*t.at(&vec![]), 5.0);
}

#[test]
fn from_vec_1d() {
    let t = GraphTensor::wrap(vec![1.0, 2.0, 3.0], false);
    assert_eq!(t.shape(), &vec![3]);
    assert_eq!(*t.at(&vec![2]), 3.0);
}

#[test]
fn from_vec_2d() {
    let t = GraphTensor::wrap(vec![vec![1.0, 2.0], vec![3.0, 4.0]], false);
    assert_eq!(t.shape(), &vec![2, 2]);
    assert_eq!(*t.at(&vec![1, 0]), 3.0);
    assert_eq!(*t.at(&vec![0, 1]), 2.0);
}

#[test]
fn from_vec_3d() {
    let t = GraphTensor::wrap(
        vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]],
        false,
    );
    assert_eq!(t.shape(), &vec![2, 2, 1]);
    assert_eq!(*t.at(&vec![1, 1, 0]), 4.0);
}

#[test]
#[should_panic]
fn from_vec_ragged_panics() {
    GraphTensor::wrap(vec![vec![1.0, 2.0], vec![3.0]], false);
}

#[test]
fn is_close_within_tolerance() {
    let a = GraphTensor::wrap(vec![1.0, 1.0, 1.0], false);
    let b = GraphTensor::wrap(vec![0.999999, 1.0, 1.1], false);
    let r = a.is_close(&b);
    assert_eq!(*r.at(&vec![0]), 1.0);
    assert_eq!(*r.at(&vec![1]), 1.0);
    assert_eq!(*r.at(&vec![2]), 0.0);
}

#[test]
fn is_close_with_explicit_tolerances() {
    let a = GraphTensor::wrap(vec![10.0], false);
    let b = GraphTensor::wrap(vec![10.0 + 1e-3], false);
    assert_eq!(*a.is_close(&b).at(&vec![0]), 0.0);
    assert_eq!(*a.is_close_with(&b, 1e-4, 0.0).at(&vec![0]), 1.0);
}

#[test]
fn is_close_broadcasts() {
    let a = GraphTensor::wrap(vec![1.0, 2.0, 3.0], false);
    let b = GraphTensor::wrap(1.0, false);
    let r = a.is_close(&b);
    assert_eq!(r.shape(), &vec![3]);
    assert_eq!(*r.at(&vec![0]), 1.0);
    assert_eq!(*r.at(&vec![1]), 0.0);
    assert_eq!(*r.at(&vec![2]), 0.0);
}

#[test]
fn is_close_nan_not_equal() {
    let a = GraphTensor::wrap(vec![f64::NAN], false);
    let b = GraphTensor::wrap(vec![f64::NAN], false);
    let r = a.is_close(&b);
    assert_eq!(*r.at(&vec![0]), 0.0);
}
