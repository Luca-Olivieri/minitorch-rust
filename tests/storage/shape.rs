use minitorch_rust::core::storage::TensorStorage;

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
