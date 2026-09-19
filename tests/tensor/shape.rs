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
    let da = grads.get(&a).unwrap();

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
fn transpose_arbitrary_shape_swaps_selected_dims() {
    // 2D: transpose of dims 0/1
    let a = GraphTensor::wrap(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], false);
    let t = a.transpose(0, 1); // [[1,4],[2,5],[3,6]]

    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(*t.at(&[0, 0]), 1.0);
    assert_eq!(*t.at(&[0, 1]), 4.0);
    assert_eq!(*t.at(&[1, 0]), 2.0);
    assert_eq!(*t.at(&[1, 1]), 5.0);
    assert_eq!(*t.at(&[2, 0]), 3.0);
    assert_eq!(*t.at(&[2, 1]), 6.0);

    // 3D: swap dims 1/2 of a [2,2,4] tensor (2 stacked 2x4 matrices)
    let a = GraphTensor::wrap(
        vec![
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            vec![vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
        ],
        false,
    );
    let t = a.transpose(1, 2); // [2,4,2]: each inner matrix transposed

    assert_eq!(t.shape(), &[2, 4, 2]);
    for m in 0..2 {
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(*t.at(&[m, j, i]), *a.at(&[m, i, j]));
            }
        }
    }

    // transpose is self-inverse: transposing back restores the original
    let back = t.transpose(1, 2);
    assert_eq!(back.shape(), &[2, 2, 4]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                assert_eq!(*back.at(&[i, j, k]), *a.at(&[i, j, k]));
            }
        }
    }

    // non-adjacent swap on a 4D tensor, dims 0/3
    let a = GraphTensor::wrap(
        vec![
            vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            ],
            vec![
                vec![vec![9.0, 10.0], vec![11.0, 12.0]],
                vec![vec![13.0, 14.0], vec![15.0, 16.0]],
            ],
        ],
        false,
    ); // [2,2,2,2]
    let t = a.transpose(0, 3); // [2,2,2,2]
    assert_eq!(t.shape(), &[2, 2, 2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    assert_eq!(*t.at(&[i, j, k, l]), *a.at(&[l, j, k, i]));
                }
            }
        }
    }
}

#[test]
fn transpose_backward_swaps_selected_dims() {
    // y = transpose(x, 1, 2) of a [2,2,3] tensor, then summed to a scalar.
    // dL/dx[i,j,k] = dL/dy[i,k,j].
    let x = GraphTensor::new(vec![2, 2, 3], 1.0, true);
    let y = x.transpose(1, 2).sum(&[], false);

    let grads = y.backward(true);
    let dx = grads.get(&x).unwrap();
    assert_eq!(dx.shape(), &[2, 2, 3]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*dx.at(&[i, j, k]), 1.0);
            }
        }
    }
}

#[test]
#[should_panic]
fn transpose_out_of_range_dim_panics() {
    let a = GraphTensor::new(vec![2, 3], 1.0, false);
    let _ = a.transpose(0, 2);
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
    let da = grads_map.get(&a).unwrap();
    assert_eq!(da.shape(), &[4, 2, 1, 3]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&[i, j, 0, k]), 1.0);
            }
        }
    }

    let db = grads_map.get(&b).unwrap();
    assert_eq!(db.shape(), &[4, 2, 3]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*db.at(&[i, j, k]), 1.0);
            }
        }
    }

    let dc = grads_map.get(&c).unwrap();
    assert_eq!(dc.shape(), &[4, 2, 3, 1]);
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*dc.at(&[i, j, k, 0]), 1.0);
            }
        }
    }

    let dd = grads_map.get(&d).unwrap();
    assert_eq!(dd.shape(), &[4, 3, 1]);
    for i in 0..4 {
        for k in 0..3 {
            assert_eq!(*dd.at(&[i, k, 0]), 1.0);
        }
    }

    let de = grads_map.get(&e).unwrap();
    assert_eq!(de.shape(), &[]);
    assert_eq!(*de.at(&[]), 1.0);
}
