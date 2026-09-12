use std::hint::black_box;
use std::time::{Duration, Instant};

use minitorch_rust::core::storage::TensorStorage;
use minitorch_rust::core::storage::ops::reduce::reduce_dims;

// The `max` kernel before `max_all` existed: the generic reduce_dims odometer.
fn old_max_odometer(a: &TensorStorage) -> TensorStorage {
    let dims: Vec<usize> = (0..a.shape.len()).collect();
    reduce_dims(
        a,
        &dims,
        |v| v,
        |acc, v| if v > acc { v } else { acc },
        |acc| acc,
    )
}

// Time one variant over `iters` runs, returning ns/op.
fn time(f: impl Fn() -> TensorStorage, iters: usize) -> Duration {
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    t.elapsed() / iters as u32
}

fn bench_one(name: &str, a: &TensorStorage) {
    const ITERS: usize = 100;

    // correctness: both produce the same scalar max
    let flat = TensorStorage::max_all(a).buffer.as_ref()[0];
    let odometer = old_max_odometer(a).buffer.as_ref()[0];
    assert_eq!(flat, odometer);

    // warm up both code paths so the comparison is steady-state
    for _ in 0..10 {
        black_box(old_max_odometer(a));
        black_box(TensorStorage::max_all(a));
    }

    let t_odo = time(|| old_max_odometer(a), ITERS).as_secs_f64() * 1e3;
    let t_flat = time(|| TensorStorage::max_all(a), ITERS).as_secs_f64() * 1e3;
    eprintln!(
        "{name:28} odometer {:>9.3}ms   max_all {:>9.3}ms   ratio {:.2}x",
        t_odo,
        t_flat,
        t_odo / t_flat
    );
}

#[test]
fn max_all_matches_reduce_dims() {
    let a = TensorStorage::from_buffer(vec![3, 4], (1..=12).map(|x| x as f64).collect());
    assert_eq!(TensorStorage::max_all(&a).buffer.as_ref()[0], 12.0);
    let t = TensorStorage::transpose(&a);
    assert!(!t.contiguous);
    assert_eq!(TensorStorage::max_all(&t).buffer.as_ref()[0], 12.0);
}

/// Run with: cargo test --release -- --ignored --nocapture bench_max_all
#[test]
#[ignore]
fn bench_max_all() {
    const M: usize = 1 << 20; // 1 Mi elems = 8 MB of f64

    // cache-resident 2D (goes through the general odometer, not the 1D tight loop)
    let a = TensorStorage::from_buffer(vec![64, 16384], (0..M).map(|i| i as f64).collect());
    bench_one("contig [64,16384] (8MB)", &a);

    // cache-resident 3D
    let a = TensorStorage::from_buffer(vec![64, 64, 256], (0..M).map(|i| i as f64).collect());
    bench_one("contig [64,64,256] (8MB)", &a);

    // larger than L2 (128MB)
    let a = TensorStorage::from_buffer(
        vec![1024, 1024, 16],
        (0..M * 16).map(|i| i as f64).collect(),
    );
    bench_one("contig [1024,1024,16] (128MB)", &a);

    // strided view of an 8MB buffer (transpose): both paths use the odometer
    let c = TensorStorage::from_buffer(vec![1024, 1024], (0..M).map(|i| i as f64).collect());
    let t = TensorStorage::transpose(&c);
    bench_one("strided [1024,1024]^T (8MB)", &t);
}

#[test]
fn sum_strided_view_empty_dims_means_all() {
    // 2x2 contiguous, then transposed into a [2,2] strided view.
    let a = TensorStorage::from_buffer(vec![2, 2], (1..=4).map(|x| x as f64).collect());
    let t = TensorStorage::transpose(&a);
    assert!(!t.contiguous);

    let sum = TensorStorage::sum(&t, &[]);
    assert_eq!(sum.shape, Vec::<usize>::new());
    assert_eq!(sum.buffer.as_ref()[0], 10.0);
}

#[test]
fn sum_empty_dims_matches_explicit_all_dims() {
    let a = TensorStorage::from_buffer(vec![2, 2, 2], (1..=8).map(|x| x as f64).collect());

    let via_empty = TensorStorage::sum(&a, &[]);
    let via_all = TensorStorage::sum(&a, &[0, 1, 2]);
    assert_eq!(via_empty.buffer.as_ref()[0], 36.0);
    assert_eq!(via_empty.buffer.as_ref()[0], via_all.buffer.as_ref()[0]);
    assert_eq!(via_empty.shape, Vec::<usize>::new());
}

#[test]
fn sum_single_dim_and_multi_dims_agree() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    // a = [[1,2,3],[4,5,6]]

    let single = TensorStorage::sum(&a, &[0]); // [5,7,9]
    assert_eq!(single.shape, vec![3]);
    assert_eq!(single.buffer.as_ref(), &[5.0, 7.0, 9.0]);

    // single-dim then all-dims equals reducing over {0,1} at once
    let via_multi = TensorStorage::sum(&a, &[0, 1]);
    let via_two_step = TensorStorage::sum(&single, &[0]);
    assert_eq!(via_multi.buffer.as_ref(), via_two_step.buffer.as_ref());
    assert_eq!(via_multi.buffer.as_ref(), &[21.0]);

    // dim order does not matter (sorted internally)
    let swapped = TensorStorage::sum(&a, &[1, 0]);
    assert_eq!(swapped.buffer.as_ref(), &[21.0]);
}
