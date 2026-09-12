use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};
use minitorch_rust::core::GraphTensor;

#[test]
fn one_hot_encodes_last_dim_per_class() {
    let mut f = FreeTensor::new(vec![4, 2], 2.0, true);
    f.set(&vec![0, 0], 3.0);
    f.set(&vec![0, 1], 3.0);
    f.set(&vec![1, 0], 2.0);
    f.set(&vec![1, 1], 2.0);
    f.set(&vec![2, 0], 1.0);
    f.set(&vec![2, 1], 1.0);
    f.set(&vec![3, 0], 0.0);
    f.set(&vec![3, 1], 0.0);

    let a = f.to_graph();
    let oh = a.one_hot(4);

    assert_eq!(oh.shape(), &vec![4, 2, 4]);

    let expected = vec![
        vec![vec![0.0, 0.0, 0.0, 1.0], vec![0.0, 0.0, 0.0, 1.0]],
        vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]],
        vec![vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]],
    ];
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..4 {
                assert_eq!(*oh.at(&vec![i, j, k]), expected[i][j][k]);
            }
        }
    }
}

// shape [2, 2, 3] with values i*6 + j*3 + k
fn tensor() -> GraphTensor {
    GraphTensor::wrap(
        vec![
            vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]],
            vec![vec![6.0, 7.0, 8.0], vec![9.0, 10.0, 11.0]],
        ],
        false,
    )
}

#[test]
fn sum_over_non_adjacent_dims() {
    let out = tensor().sum(&[0, 2], false);
    assert_eq!(out.shape(), &vec![2]);
    assert_eq!(*out.at(&vec![0]), 24.0);
    assert_eq!(*out.at(&vec![1]), 42.0);
}

#[test]
fn sum_over_adjacent_dims() {
    let out = tensor().sum(&[1, 2], false);
    assert_eq!(out.shape(), &vec![2]);
    assert_eq!(*out.at(&vec![0]), 15.0);
    assert_eq!(*out.at(&vec![1]), 51.0);
}

#[test]
fn sum_empty_dims_aggregates_over_all() {
    let out = tensor().sum(&[], false);
    assert_eq!(out.shape(), &vec![]);
    assert_eq!(*out.at(&vec![]), 66.0);

    // empty and explicit-all must agree
    let all = tensor().sum(&[0, 1, 2], false);
    assert_eq!(*out.at(&vec![]), *all.at(&vec![]));
}

#[test]
fn sum_single_dim_uses_single_kernel() {
    // dispatch: single dim -> sum over dim 1 per-slice columns
    let a = tensor().sum(&[1], false);
    assert_eq!(a.shape(), &vec![2, 3]);
    assert_eq!(*a.at(&vec![0, 0]), 3.0);
    assert_eq!(*a.at(&vec![0, 1]), 5.0);
    assert_eq!(*a.at(&vec![0, 2]), 7.0);
    assert_eq!(*a.at(&vec![1, 0]), 15.0);
    assert_eq!(*a.at(&vec![1, 1]), 17.0);
    assert_eq!(*a.at(&vec![1, 2]), 19.0);

    // single-dim chains must equal the generic one-shot multi-dim reduction
    let one_shot = tensor().sum(&[0, 1, 2], false);
    let chained = tensor().sum(&[1], false).sum(&[0], false).sum(&[0], false);
    assert_eq!(*one_shot.at(&vec![]), *chained.at(&vec![]));
    assert_eq!(*one_shot.at(&vec![]), 66.0);
}

#[test]
fn sum_all_total() {
    let out = tensor().sum(&[0, 1, 2], false);
    assert_eq!(out.shape(), &vec![]);
    assert_eq!(*out.at(&vec![]), 66.0);
}

#[test]
fn sum_keepdim_keeps_reduced_axes() {
    let out = tensor().sum(&[0, 2], true);
    assert_eq!(out.shape(), &vec![1, 2, 1]);
    assert_eq!(*out.at(&vec![0, 0, 0]), 24.0);
    assert_eq!(*out.at(&vec![0, 1, 0]), 42.0);
}

#[test]
fn sum_keepdim_empty_dims_keeps_all_axes() {
    let out = tensor().sum(&[], true);
    assert_eq!(out.shape(), &vec![1, 1, 1]);
    assert_eq!(*out.at(&vec![0, 0, 0]), 66.0);
}

#[test]
fn mean_over_non_adjacent_dims() {
    let out = tensor().mean(&[0, 2], false);
    assert_eq!(out.shape(), &vec![2]);
    assert_eq!(*out.at(&vec![0]), 4.0);
    assert_eq!(*out.at(&vec![1]), 7.0);
}

#[test]
fn mean_empty_dims_aggregates_over_all() {
    let out = tensor().mean(&[], false);
    assert_eq!(out.shape(), &vec![]);
    assert_eq!(*out.at(&vec![]), 5.5);
}

#[test]
fn mean_keepdim_single_dim() {
    let out = tensor().mean(&[1], true);
    assert_eq!(out.shape(), &vec![2, 1, 3]);
    assert_eq!(*out.at(&vec![0, 0, 0]), 1.5);
    assert_eq!(*out.at(&vec![0, 0, 1]), 2.5);
    assert_eq!(*out.at(&vec![1, 0, 2]), 9.5);
}

#[test]
fn max_over_non_adjacent_dims() {
    let out = tensor().max(&[0, 2], false);
    assert_eq!(out.shape(), &vec![2]);
    assert_eq!(*out.at(&vec![0]), 8.0);
    assert_eq!(*out.at(&vec![1]), 11.0);
}

#[test]
fn max_over_adjacent_dims() {
    let out = tensor().max(&[1, 2], false);
    assert_eq!(out.shape(), &vec![2]);
    assert_eq!(*out.at(&vec![0]), 5.0);
    assert_eq!(*out.at(&vec![1]), 11.0);
}

#[test]
fn max_empty_dims_aggregates_over_all() {
    let out = tensor().max(&[], false);
    assert_eq!(out.shape(), &vec![]);
    assert_eq!(*out.at(&vec![]), 11.0);
}

#[test]
fn max_keepdim_keeps_reduced_axes() {
    let out = tensor().max(&[0, 2], true);
    assert_eq!(out.shape(), &vec![1, 2, 1]);
    assert_eq!(*out.at(&vec![0, 0, 0]), 8.0);
    assert_eq!(*out.at(&vec![0, 1, 0]), 11.0);
}

#[test]
fn argmax_keepdim_keeps_axis() {
    let out = tensor().argmax(1, true);
    assert_eq!(out.shape(), &vec![2, 1, 3]);
    assert_eq!(*out.at(&vec![0, 0, 0]), 1.0);
    assert_eq!(*out.at(&vec![1, 0, 2]), 1.0);

    let squeezed = tensor().argmax(1, false);
    assert_eq!(squeezed.shape(), &vec![2, 3]);
    assert_eq!(*squeezed.at(&vec![0, 0]), 1.0);
}

#[test]
fn sum_backward_broadcasts_grad_to_all_reduced_dims() {
    let mut a = tensor();
    a.set_requires_grad(true);
    let out = a.sum(&[0, 2], false);

    let grads = out.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &vec![2, 2, 3]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&vec![i, j, k]), 1.0);
            }
        }
    }
}

#[test]
fn sum_backward_through_keepdim_broadcasts_to_input() {
    let mut a = tensor();
    a.set_requires_grad(true);
    // keepdim output [1,2,1] broadcasts against a [2,2,3] multipliers tensor
    let out = &a.sum(&[0, 2], true)
        * &GraphTensor::wrap(
            vec![
                vec![vec![2.0, 2.0, 2.0], vec![3.0, 3.0, 3.0]],
                vec![vec![5.0, 5.0, 5.0], vec![7.0, 7.0, 7.0]],
            ],
            false,
        );
    assert_eq!(out.shape(), &vec![2, 2, 3]);

    let grads = out.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &vec![2, 2, 3]);
    // `out = s * m` with s = keepdim sum over {0,2}, m the multipliers:
    // backprop through `*` sums m over the axes `s` lacks, so each
    // (i, j, k) receives the per-j aggregate Σ_{i,k} m[i,j,k] = 21 / 30.
    let expected = vec![
        vec![vec![21.0, 21.0, 21.0], vec![30.0, 30.0, 30.0]],
        vec![vec![21.0, 21.0, 21.0], vec![30.0, 30.0, 30.0]],
    ];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&vec![i, j, k]), expected[i][j][k]);
            }
        }
    }
}

#[test]
fn sum_empty_dims_backward_is_ones() {
    let mut a = tensor();
    a.set_requires_grad(true);
    let out = a.sum(&[], false);

    let grads = out.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &vec![2, 2, 3]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&vec![i, j, k]), 1.0);
            }
        }
    }
}

#[test]
fn max_backward_flows_only_to_maxima() {
    let mut a = tensor();
    a.set_requires_grad(true);
    let out = a.max(&[0, 2], false);

    let grads = out.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &vec![2, 2, 3]);

    // max over {0, 2} per j: 8 for j=0 (at i=1,k=2), 11 for j=1 (at i=1,k=2)
    let expected = vec![
        vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
        vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]],
    ];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&vec![i, j, k]), expected[i][j][k]);
            }
        }
    }
}

#[test]
fn max_backward_through_keepdim_flows_only_to_maxima() {
    let mut a = tensor();
    a.set_requires_grad(true);
    // same as max backward but the reduce keeps its axes ([1,2,1])
    let out = &a.max(&[0, 2], true)
        * &GraphTensor::wrap(
            vec![
                vec![vec![2.0, 2.0, 2.0], vec![3.0, 3.0, 3.0]],
                vec![vec![5.0, 5.0, 5.0], vec![7.0, 7.0, 7.0]],
            ],
            false,
        );

    let grads = out.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    assert_eq!(da.shape(), &vec![2, 2, 3]);
    // analogous to the sum case: the max-slice gradient ds[j] = Σ_{i,k} m[i,j,k]
    // (21 / 30) is expanded over {0,2} and gated by the per-slice max mask
    let expected = vec![
        vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
        vec![vec![0.0, 0.0, 21.0], vec![0.0, 0.0, 30.0]],
    ];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(*da.at(&vec![i, j, k]), expected[i][j][k]);
            }
        }
    }
}

#[test]
#[should_panic]
fn duplicate_dims_panic() {
    tensor().sum(&[0, 0], false);
}

#[test]
#[should_panic]
fn out_of_range_dim_panics() {
    tensor().max(&[3], false);
}
