//! Stage-2 tests: tensor-level dtype casts allocate a new tensor with elements
//! converted through the element-level cast traits.

use crate::core::storage::TensorStorage;
use crate::core::tensor::cast::cast_storage;
use crate::core::tensor::{AbstractTensor, FreeTensor, GraphTensor};

#[test]
fn cast_exact_int_widening() {
    let t = GraphTensor::<i32>::wrap(vec![vec![1, 2], vec![3, 4]], false);
    let c = t.cast::<i64>();

    assert_eq!(c.shape(), &vec![2, 2]);
    assert_eq!(c.at(&[1, 0]), &3i64);
}

#[test]
fn cast_exact_unsigned_to_float() {
    let t = GraphTensor::<u8>::wrap(vec![200, 7], false);
    let c = t.cast::<f64>();

    assert_eq!(c.numel(), 2);
    assert_eq!(c.at(&[0]), &200.0);
    assert_eq!(c.at(&[1]), &7.0);
}

#[test]
fn cast_bool_to_int() {
    let t = GraphTensor::<bool>::wrap(vec![true, false, true], true);
    let c = t.cast::<i32>();

    assert_eq!(c.numel(), 3);
    assert_eq!(c.at(&[0]), &1);
    assert_eq!(c.at(&[2]), &1);
    assert!(!c.requires_grad());
}

#[test]
fn cast_identity_preserves_requires_grad() {
    let t = GraphTensor::<f64>::wrap(vec![1.5], true);
    let c = t.cast::<f64>();

    assert_eq!(c.item(), 1.5);
    assert!(c.requires_grad());
}

#[test]
fn cast_free_tensor() {
    let mut t = FreeTensor::<i16>::new(vec![2], 0, false);
    t.set(&[0], 100);
    t.set(&[1], -100);

    let c = t.cast::<i64>();
    assert_eq!(c.at(&[0]), &100i64);
    assert_eq!(c.at(&[1]), &-100i64);
}

#[test]
fn cast_from_strided_view_is_correct_and_contiguous() {
    let a = TensorStorage::<f64>::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    let strided = TensorStorage::transpose(&a, 0, 1);
    assert!(!strided.contiguous);

    // Logical order of the transposed view is [1, 4, 2, 5, 3, 6].
    let c = cast_storage(&strided, |x| x);
    assert!(c.contiguous);
    for (i, &expected) in [1.0, 4.0, 2.0, 5.0, 3.0, 6.0].iter().enumerate() {
        assert_eq!(c.buffer[i], expected);
    }
}

#[cfg(feature = "allow_lossy_casts")]
mod lossy {
    use super::*;

    #[test]
    fn cast_lossy_float_narrowing_rounds() {
        let t = GraphTensor::<f64>::wrap(vec![1.123456789, 2.0], false);
        let c = t.cast_lossy::<f32>();

        assert_eq!(*c.at(&[0]), 1.123456789_f64 as f32);
        assert_eq!(*c.at(&[1]), 2.0f32);
    }

    #[test]
    fn cast_lossy_int_to_bool() {
        let t = GraphTensor::<i32>::wrap(vec![0, 0, 5], false);
        let c = t.cast_lossy::<bool>();

        assert_eq!(c.at(&[0]), &false);
        assert_eq!(c.at(&[2]), &true);
    }

    #[test]
    fn cast_lossy_int_to_float() {
        let t = GraphTensor::<i64>::wrap(vec![-3, 3], false);
        let c = t.cast_lossy::<f64>();

        assert_eq!(c.at(&[0]), &-3.0);
        assert_eq!(c.at(&[1]), &3.0);
    }

    #[test]
    fn cast_lossy_free_tensor() {
        let t = FreeTensor::<u32>::new(vec![2], 0, false);
        let mut t = t;
        t.set(&[0], 42);
        let c = t.cast_lossy::<f32>();
        assert_eq!(c.at(&[0]), &42.0f32);
    }
}

#[cfg(feature = "allow_dangerous_casts")]
mod dangerous {
    use super::*;

    #[test]
    fn cast_dangerous_float_to_int_in_range() {
        let t = GraphTensor::<f64>::wrap(vec![5.0, -2.0, 0.0], false);
        let c = t.cast_dangerous::<i32>();

        assert_eq!(c.at(&[0]), &5);
        assert_eq!(c.at(&[1]), &-2);
        assert_eq!(c.at(&[2]), &0);
    }

    #[test]
    #[should_panic(expected = "must be a finite, integral number in range")]
    fn cast_dangerous_float_to_int_fractional_panics() {
        let t = GraphTensor::<f64>::wrap(vec![5.5], false);
        let _ = t.cast_dangerous::<i32>();
    }

    #[test]
    fn cast_dangerous_int_narrowing_in_range() {
        let t = GraphTensor::<i64>::wrap(vec![100, -100], false);
        let c = t.cast_dangerous::<i8>();

        assert_eq!(c.at(&[0]), &100i8);
        assert_eq!(c.at(&[1]), &-100i8);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn cast_dangerous_int_narrowing_out_of_range_panics() {
        let t = GraphTensor::<u64>::wrap(vec![300], false);
        let _ = t.cast_dangerous::<u8>();
    }
}
