use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::tensor::AbstractTensor;

fn assert_shape(t: &GraphTensor, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "shape mismatch for {t:?}");
}

/// Assert the tensor's element values (in logical flat order) against `expected`,
/// within a small absolute tolerance. Grad/sum tensors may be strided broadcast
/// views, so values are read through the public `at` API, walking the shape in
/// row-major (flat) order.
fn assert_values(t: &GraphTensor, expected: &[f64]) {
    assert_eq!(
        t.numel(),
        expected.len(),
        "numel mismatch for shape {:?}",
        t.shape()
    );
    for (i, e) in expected.iter().enumerate() {
        let a = *t.at(&flat_to_md(i, t.shape()));
        let tol = 1e-9;
        assert!(
            (a - e).abs() <= tol,
            "value mismatch at flat index {i} of {:?}: got {a}, expected {e}",
            t.shape()
        );
    }
}

/// Convert a flat row-major index into the multi-dimensional index of `shape`.
fn flat_to_md(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut md = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        md[d] = i % shape[d];
        i /= shape[d];
    }
    md
}

#[test]
fn simple_operation_forward_and_backward() {
    let shape = vec![1, 2, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true);
    let b = GraphTensor::new(shape.clone(), 3.0, true);

    let x = &a * &b;

    let grads_map = x.backward(true);

    // x = a * b = 6 everywhere.
    assert_shape(&x, &shape);
    assert_values(&x, &[6.0; 6]);

    // ∂x/∂a = b = 3, ∂x/∂b = a = 2 (seed gradient is 1).
    let da = grads_map.get(&a).unwrap();
    assert_shape(da, &shape);
    assert_values(da, &[3.0; 6]);

    let db = grads_map.get(&b).unwrap();
    assert_shape(db, &shape);
    assert_values(db, &[2.0; 6]);

    // Higher order: ∂a/∂a is a constant (3), so its gradient wrt `a` is 0
    // (node disconnected); ∂a/∂b = 1.
    let da_grads_map = da.backward(true);
    assert!(
        da_grads_map.get(&a).is_none(),
        "d2x/da2 should be disconnected"
    );

    let d2a_db = da_grads_map.get(&b).unwrap();
    assert_shape(d2a_db, &shape);
    assert_values(d2a_db, &[1.0; 6]);
}

#[test]
fn complex_operation_forward_and_backward() {
    let shape = vec![1, 2, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true);
    let b = GraphTensor::new(shape.clone(), 3.0, true);
    let c = GraphTensor::new(shape.clone(), 4.0, true);

    // r = -ln(((a*b + c - a) / b)^2)
    let x = &a * &b;
    let y = &x + &c;
    let z = &y - &a;
    let w = &z / &b;

    let twos = GraphTensor::new(shape.clone(), 2.0, false);
    let p = w.pow(&twos);
    let q = p.ln();
    let r = -&q;

    let grads_map = r.backward(true);

    let expected_r = -((8.0 / 3.0) as f64).powi(2).ln();
    assert_shape(&r, &shape);
    assert_values(&r, &[expected_r; 6]);

    let da = grads_map.get(&a).unwrap();
    assert_shape(da, &shape);
    assert_values(da, &[-0.5; 6]);

    let db = grads_map.get(&b).unwrap();
    assert_shape(db, &shape);
    assert_values(db, &[1.0 / 6.0; 6]);

    let dc = grads_map.get(&c).unwrap();
    assert_shape(dc, &shape);
    assert_values(dc, &[-0.25; 6]);

    // Second-order gradients of r wrt {a, b, c}, recorded from the graph:
    // ∂a/∂(a,b,c) and ∂b/∂(a,b,c) mix through the division/sum nodes.
    let da_grads_map = da.backward(true);
    let db_grads_map = db.backward(true);
    let dc_grads_map = dc.backward(true);

    assert_values(da_grads_map.get(&a).unwrap(), &[0.125; 6]);
    assert_values(da_grads_map.get(&b).unwrap(), &[-0.125; 6]);
    assert_values(da_grads_map.get(&c).unwrap(), &[0.0625; 6]);

    assert_values(db_grads_map.get(&a).unwrap(), &[-0.125; 6]);
    assert_values(db_grads_map.get(&b).unwrap(), &[-0.09722222222222224; 6]);
    assert_values(db_grads_map.get(&c).unwrap(), &[0.0625; 6]);

    assert_values(dc_grads_map.get(&a).unwrap(), &[0.0625; 6]);
    assert_values(dc_grads_map.get(&b).unwrap(), &[0.0625; 6]);
    assert_values(dc_grads_map.get(&c).unwrap(), &[0.03125; 6]);
}

#[test]
fn matmul_forward_and_backward() {
    let a = GraphTensor::new(vec![2, 3], 1.0, true);
    let b = GraphTensor::new(vec![3, 4], 1.0, true);

    let x = GraphTensor::matmul(&a, &b);

    let grads_map = x.backward(true);

    // ones[2,3] x ones[3,4] -> each entry is the inner product over k=3.
    assert_shape(&x, &[2, 4]);
    assert_values(&x, &[3.0; 8]);

    // ∂x/∂a = x.grad @ b^T: [2,4] @ [4,3] of ones = 4.
    let da = grads_map.get(&a).unwrap();
    assert_shape(da, &[2, 3]);
    assert_values(da, &[4.0; 6]);

    // ∂x/∂b = a^T @ x.grad: [3,2] @ [2,4] of ones = 2.
    let db = grads_map.get(&b).unwrap();
    assert_shape(db, &[3, 4]);
    assert_values(db, &[2.0; 12]);

    // a.grad = x.grad @ b^T is constant in `a` (disconnected), and ∂(a.grad)/∂b = 2.
    let da_grads_map = da.backward(true);
    assert!(
        da_grads_map.get(&a).is_none(),
        "d2x/da2 should be disconnected"
    );

    let d2a_db = da_grads_map.get(&b).unwrap();
    assert_shape(d2a_db, &[3, 4]);
    assert_values(d2a_db, &[2.0; 12]);
}

#[test]
fn abs_forward_and_backward() {
    let a = GraphTensor::<f64>::wrap(vec![-2.0, 3.0, 0.0], true);

    let y = a.abs();
    assert_values(&y, &[2.0, 3.0, 0.0]);

    let grads = y.sum(&[], false).backward(false);
    let da = grads.get(&a).unwrap();
    // sign(a): -1, +1, and 0 (subgradient) at zero.
    assert_values(da, &[-1.0, 1.0, 0.0]);
}

#[test]
fn abs_is_twice_differentiable_away_from_zero() {
    let a = GraphTensor::<f64>::wrap(vec![2.0, -4.0], true);
    let y = a.abs();

    let grads = y.sum(&[], false).backward(true);
    let da = grads.get(&a).unwrap();
    assert_values(da, &[1.0, -1.0]);

    let second = da.sum(&[], false).backward(false);
    // d2|a|/da2 = 0 everywhere except the measure-zero point a == 0: the
    // constant sign gradient is disconnected from `a`.
    assert!(second.get(&a).is_none());
}

#[test]
fn int_abs_is_a_graph_boundary() {
    let a = GraphTensor::<i32>::wrap(vec![2, -3], true);

    let y = a.abs();
    assert_eq!(*y.at(&[0]), 2);
    assert_eq!(*y.at(&[1]), 3);
    assert!(!y.requires_grad());
}

#[test]
fn modul_values_and_boundary() {
    let a = GraphTensor::<f64>::wrap(vec![7.5, -7.5], true);
    let b = GraphTensor::<f64>::wrap(vec![3.0, 3.0], true);

    let r = a.modul(&b);
    assert_eq!(*r.at(&[0]), 1.5);
    assert_eq!(*r.at(&[1]), -1.5);
    assert!(!r.requires_grad());

    // Integers get it too (truncated remainder), as a forward-only boundary.
    let m = GraphTensor::<i32>::wrap(vec![7, -7], false);
    let n = GraphTensor::<i32>::wrap(vec![3, 3], false);
    let rint = m.modul(&n);
    assert_eq!(*rint.at(&[0]), 1);
    assert_eq!(*rint.at(&[1]), -1);
}
