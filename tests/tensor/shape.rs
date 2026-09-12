use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::tensor::AbstractTensor;

#[test]
fn broadcast_forward() {
    let a = GraphTensor::wrap(vec![1.0, 2.0, 3.0], false);
    let b = a.broadcast_to_shape(&[2, 3]);

    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(*b.at(&[0, 0]), 1.0);
    assert_eq!(*b.at(&[1, 1]), 2.0);
    assert_eq!(*b.at(&[1, 2]), 3.0);
}

#[test]
fn broadcast_sums_grad_over_expanded_axes() {
    let a = GraphTensor::wrap(vec![1.0, 2.0, 3.0], true);
    let b = a.broadcast_to_shape(&[2, 3]);

    let grads = b.backward(true);
    let da = grads.get(&a.to_key()).unwrap();

    assert_eq!(da.shape(), &[3]);
    assert_eq!(*da.at(&[0]), 2.0);
    assert_eq!(*da.at(&[1]), 2.0);
    assert_eq!(*da.at(&[2]), 2.0);
}

#[test]
#[should_panic]
fn broadcast_incompatible_shape_panics() {
    let a = GraphTensor::wrap(vec![1.0, 2.0, 3.0], false);
    a.broadcast_to_shape(&[2, 2]);
}

#[test]
fn squeeze_unsqueeze_sum_forward_and_backward() {
    let shape = vec![4, 2, 1, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true); // [4, 2, 1, 3]
    let b = a.squeeze(2); // [4, 2, 3]
    let c = b.unsqueeze(3); // [4, 2, 3, 1]
    let d = c.sum(&[1], false); // [4, 3, 1]
    let e = d.sum(&[], false); // []

    assert_eq!(a.shape(), &[4, 2, 1, 3]);
    assert_eq!(b.shape(), &[4, 2, 3]);
    assert_eq!(c.shape(), &[4, 2, 3, 1]);
    assert_eq!(d.shape(), &[4, 3, 1]);
    assert_eq!(e.shape(), &[]);

    // a is filled with 2.0 over 4*2*1*3 = 24 elements.
    assert_eq!(*e.at(&[]), 48.0);

    let grads_map = e.backward(true);

    // Every op in the chain (squeeze/unsqueeze/sum) broadcasts the seed
    // gradient back to the original shape; all grads are 1.0.
    let da = grads_map.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &[4, 2, 1, 3]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&[i, j, 0, k]), 1.0);
            }
        }
    }

    let db = grads_map.get(&b.to_key()).unwrap();
    assert_eq!(db.shape(), &[4, 2, 3]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*db.at(&[i, j, k]), 1.0);
            }
        }
    }

    let dc = grads_map.get(&c.to_key()).unwrap();
    assert_eq!(dc.shape(), &[4, 2, 3, 1]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*dc.at(&[i, j, k, 0]), 1.0);
            }
        }
    }

    let dd = grads_map.get(&d.to_key()).unwrap();
    assert_eq!(dd.shape(), &[4, 3, 1]);
    for i in 0..4 {
        for k in 0..3 {
            assert_eq!(*dd.at(&[i, k, 0]), 1.0);
        }
    }

    let de = grads_map.get(&e.to_key()).unwrap();
    assert_eq!(de.shape(), &[]);
    assert_eq!(*de.at(&[]), 1.0);
}
