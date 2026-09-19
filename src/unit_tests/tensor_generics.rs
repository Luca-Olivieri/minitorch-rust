//! Stage-1 smoke tests: the tensor machinery is generic over a dtype
//! (`GraphTensor<i32>`, `FreeTensor<bool>`, ...) while defaulting to `f64` so
//! all existing float code keeps compiling unchanged.

use crate::core::dtype::CastFrom;
use crate::core::storage::TensorStorage;
use crate::core::tensor::{AbstractTensor, FreeTensor, GraphTensor};

#[test]
fn graph_tensor_over_i32_wraps_and_reads() {
    let t = GraphTensor::<i32>::wrap(vec![vec![1, 2], vec![3, 4]], false);

    assert_eq!(t.shape(), &vec![2, 2]);
    assert_eq!(t.numel(), 4);
    assert_eq!(t.at(&[1, 0]), &3);

    let s = format!("{t}");
    assert!(s.contains("dtype=int32"), "{s}");
    assert!(s.contains("[1, 2]"), "{s}");
}

#[test]
fn graph_tensor_wrap_infers_f64_by_default() {
    let t = GraphTensor::wrap(vec![vec![1.0, 2.0], vec![3.0, 4.0]], false);

    assert_eq!(t.shape(), &vec![2, 2]);
    assert_eq!(t.at(&[0, 1]), &2.0);

    let s = format!("{t}");
    assert!(s.contains("dtype=float64"), "{s}");
}

#[test]
fn item_returns_scalar_of_the_element_dtype() {
    let t = GraphTensor::<f32>::wrap(vec![2.5], false);
    assert_eq!(t.item(), 2.5f32);

    let u = GraphTensor::<u16>::wrap(vec![7], false);
    assert_eq!(u.item(), 7u16);
}

#[test]
fn free_tensor_over_bool_sets_and_displays() {
    let mut t = FreeTensor::<bool>::new(vec![2], false, false);
    t.set(&[1], true);

    assert_eq!(t.at(&[0]), &false);
    assert_eq!(t.at(&[1]), &true);

    let s = format!("{t}");
    assert!(s.contains("dtype=bool"), "{s}");
}

#[test]
fn storage_display_uses_the_element_dtype() {
    let s = TensorStorage::<u8>::from_buffer(vec![2], vec![1, 2]);
    let text = format!("{s}");
    assert!(text.contains("dtype=uint8"), "{text}");
    assert!(text.contains("[1, 2]"), "{text}");
}

#[test]
fn casts_work_on_generic_elements() {
    let widened: i64 = i64::cast_from(5i32);
    assert_eq!(widened, 5);

    let widened: f64 = f64::cast_from(3u8);
    assert_eq!(widened, 3.0);
}
