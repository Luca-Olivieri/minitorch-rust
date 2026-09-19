use crate::core::storage::TensorStorage;

#[test]
fn squeeze_at_round_trips_unsqueeze_at() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());

    let kept = TensorStorage::unsqueeze_at(&a, &[0, 2]);
    assert_eq!(kept.shape, vec![1, 2, 1, 3]);
    assert_eq!(kept.buffer.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let back = TensorStorage::squeeze_at(&kept, &[0, 2]);
    assert_eq!(back.shape, vec![2, 3]);
    assert_eq!(back.buffer.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(back.strides, a.strides);
}

#[test]
#[should_panic]
fn squeeze_at_panics_on_non_increasing_dims() {
    let a = TensorStorage::from_buffer(vec![1, 2, 1], (1..=2).map(|x| x as f64).collect());
    TensorStorage::squeeze_at(&a, &[2, 0]);
}

#[test]
#[should_panic]
fn squeeze_at_panics_on_non_singleton_dim() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    TensorStorage::squeeze_at(&a, &[0]);
}

#[test]
fn pad_zero_fills_before_and_after_every_dim() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());

    // pad dim0 by (1, 0), dim1 by (0, 2): shape [3, 5]
    let p = TensorStorage::pad(&a, &[(1, 0), (0, 2)]);
    assert_eq!(p.shape, vec![3, 5]);

    let expected: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, // padded row 0
        1.0, 2.0, 3.0, 0.0, 0.0, // original row 0, trailing pad
        4.0, 5.0, 6.0, 0.0, 0.0, // original row 1, trailing pad
    ];
    assert_eq!(p.buffer.as_ref(), &expected);
}

#[test]
fn pad_reads_source_through_strides() {
    // a strided view: transpose of [2,3] is a [3,2] view over the same buffer
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    let t = TensorStorage::transpose(&a, 0, 1); // logical: [[1,4],[2,5],[3,6]]

    let p = TensorStorage::pad(&t, &[(0, 1), (1, 1)]);
    assert_eq!(p.shape, vec![4, 4]);
    let expected: Vec<f64> = vec![
        0.0, 1.0, 4.0, 0.0, //
        0.0, 2.0, 5.0, 0.0, //
        0.0, 3.0, 6.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
    ];
    assert_eq!(p.buffer.as_ref(), &expected);
}

#[test]
fn slice_extracts_contiguous_window() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());

    // keep row 1, take columns [1, 3): shape [1, 2]
    let s = TensorStorage::slice(&a, &[(1, 1), (1, 2)]);
    assert_eq!(s.shape, vec![1, 2]);
    assert_eq!(s.buffer.as_ref(), &[5.0, 6.0]);

    // full-window slice is a materialized copy
    let all = TensorStorage::slice(&a, &[(0, 2), (0, 3)]);
    assert_eq!(all.buffer.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_materializes_a_strided_view() {
    // transpose of [2,3] -> [3,2] view with logical elements [[1,4],[2,5],[3,6]]
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    let t = TensorStorage::transpose(&a, 0, 1);

    let r = TensorStorage::reshape(&t, &[6]);
    assert_eq!(r.shape, vec![6]);
    assert_eq!(r.buffer.as_ref(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(r.contiguous);

    // reshaping back to the 2D shape preserves logical order
    let back = TensorStorage::reshape(&r, &[3, 2]);
    assert_eq!(back.buffer.as_ref(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn reshape_preserves_flat_contiguous_order() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    let r = TensorStorage::reshape(&a, &[3, 2]);
    assert_eq!(r.buffer.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
#[should_panic]
fn pad_wrong_pair_count_panics() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    TensorStorage::pad(&a, &[(1, 1)]);
}

#[test]
#[should_panic]
fn slice_out_of_bounds_panics() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    TensorStorage::slice(&a, &[(1, 2), (0, 3)]);
}

#[test]
#[should_panic]
fn reshape_changing_numel_panics() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    TensorStorage::reshape(&a, &[4]);
}
