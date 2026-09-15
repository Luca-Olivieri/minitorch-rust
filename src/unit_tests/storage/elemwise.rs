use crate::core::storage::TensorStorage;

// Lock in the strong-typed storage layer: the same kernels now serve every
// numeric dtype, booleans get native land/lor/lnot, and compare/one_hot/argmax
// emit the documented output dtype.

#[test]
fn add_is_closed_over_ints() {
    let a = TensorStorage::from_buffer(vec![3], vec![1i32, 2, 3]);
    let b = TensorStorage::from_buffer(vec![3], vec![10i32, 20, 30]);
    let out = TensorStorage::add(&[&a, &b]);
    assert_eq!(out.buffer.as_ref(), &[11i32, 22, 33]);
}

#[test]
fn mul_and_maximum_on_ints() {
    let a = TensorStorage::from_buffer(vec![3], vec![1i8, 7, 3]);
    let b = TensorStorage::from_buffer(vec![3], vec![5i8, 2, 4]);
    let mul = TensorStorage::mul(&[&a, &b]);
    let max = TensorStorage::maximum(&[&a, &b]);
    assert_eq!(mul.buffer.as_ref(), &[5i8, 14, 12]);
    assert_eq!(max.buffer.as_ref(), &[5i8, 7, 4]);
}

#[test]
fn sub_scaled_on_f64() {
    let a = TensorStorage::from_buffer(vec![3], vec![1.0, 2.0, 3.0]);
    let b = TensorStorage::from_buffer(vec![3], vec![1.0, 1.0, 1.0]);
    let out = TensorStorage::sub_scaled(&a, &b, 0.5);
    assert_eq!(out.buffer.as_ref(), &[0.5, 1.5, 2.5]);
}

#[test]
fn matmul_on_ints() {
    let a = TensorStorage::from_buffer(vec![2, 3], vec![1u32, 2, 3, 4, 5, 6]);
    let b = TensorStorage::from_buffer(vec![3, 2], vec![7u32, 8, 9, 10, 11, 12]);
    let out = TensorStorage::matmul(&a, &b);
    assert_eq!(out.buffer.as_ref(), &[58u32, 64, 139, 154]);
    assert_eq!(out.shape, vec![2, 2]);
}

#[test]
fn sum_and_max_on_ints() {
    let a = TensorStorage::from_buffer(vec![2, 2], vec![1i32, 2, 3, 4]);
    let sum = TensorStorage::sum(&a, &[]); // empty dims = all dims
    let max = TensorStorage::max(&a, &[]);
    assert_eq!(sum.buffer.as_ref(), &[10i32]);
    assert_eq!(max.buffer.as_ref(), &[4i32]);
}

#[test]
fn comparisons_yield_masks_in_input_dtype() {
    let a = TensorStorage::from_buffer(vec![4], vec![1i32, 5, 3, 8]);
    let b = TensorStorage::from_buffer(vec![4], vec![2i32, 5, 9, 3]);
    let gt = TensorStorage::gt(&[&a, &b]);
    let lte = TensorStorage::lte(&[&a, &b]);
    assert_eq!(gt.buffer.as_ref(), &[0i32, 0, 0, 1]);
    assert_eq!(lte.buffer.as_ref(), &[1i32, 1, 1, 0]);
}

#[test]
fn is_close_matches_torch_semantics_on_f64() {
    let a = TensorStorage::from_buffer(vec![3], vec![1.0, 2.0, 1e10]);
    let b = TensorStorage::from_buffer(vec![3], vec![1.0 + 1e-8, 2.0 + 1e-4, 1e10 + 1.0]);
    let out = TensorStorage::is_close(&a, &b, 1e-5, 1e-8);
    assert_eq!(out.buffer.as_ref(), &[1.0, 0.0, 1.0]);
}

#[test]
fn one_hot_accepts_int_labels() {
    let labels = TensorStorage::from_buffer(vec![3], vec![1i32, 0, 2]);
    let out = TensorStorage::one_hot(&labels, 3);
    assert_eq!(out.buffer.as_ref(), &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(out.shape, vec![3, 3]);
}

#[test]
fn one_hot_int_out_of_range_panics() {
    let labels = TensorStorage::from_buffer(vec![2], vec![0i64, 3]);
    let result = std::panic::catch_unwind(|| TensorStorage::one_hot(&labels, 3));
    assert!(result.is_err());
}

#[test]
fn argmax_on_ints_yields_f64_indices() {
    let a = TensorStorage::from_buffer(vec![2, 3], vec![1u64, 5, 3, 9, 2, 8]);
    let out = TensorStorage::argmax(&a, 1);
    assert_eq!(out.buffer.as_ref(), &[1.0, 0.0]);
    assert_eq!(out.shape, vec![2]);
}

#[test]
fn bool_land_lor_lnot() {
    let a = TensorStorage::from_buffer(vec![3], vec![true, true, false]);
    let b = TensorStorage::from_buffer(vec![3], vec![true, false, false]);
    let land = TensorStorage::land(&[&a, &b]);
    let lor = TensorStorage::lor(&[&a, &b]);
    let lnot = TensorStorage::lnot(&[&a]);
    assert_eq!(land.buffer.as_ref(), &[true, false, false]);
    assert_eq!(lor.buffer.as_ref(), &[true, true, false]);
    assert_eq!(lnot.buffer.as_ref(), &[false, false, true]);
}

#[test]
fn shape_ops_are_generic_over_bool_and_ints() {
    let a = TensorStorage::from_buffer(vec![2, 3], vec![true, true, false, false, true, false]);
    let t = TensorStorage::transpose(&a, 0, 1);
    assert_eq!(t.shape, vec![3, 2]);
    assert_eq!(t.buffer[t.offset], true);

    let b = TensorStorage::from_buffer(vec![1, 3], vec![4u8, 5, 6]);
    let bb = b.broadcast_to_shape(&vec![2, 3]);
    assert_eq!(bb.strides, vec![0, 1]);
    assert_eq!(TensorStorage::sum(&bb, &[]).buffer.as_ref(), &[30u8]);
}