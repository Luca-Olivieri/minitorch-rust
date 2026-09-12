use std::hint::black_box;
use std::time::{Duration, Instant};

use minitorch_rust::core::storage::TensorStorage;

// The `apply_op` strided path before `StridedIter` existed: per-element
// `operands[j][i]`, i.e. a `logic_to_flat` div/mod decomposition on every
// access. That path is still live behind `Index<usize>`, so `old_mul` exercises
// the real old code.
fn old_mul(a: &TensorStorage, b: &TensorStorage) -> TensorStorage {
    let mut out = Vec::with_capacity(a.numel);
    for i in 0..a.numel {
        out.push(a[i] * b[i]);
    }
    TensorStorage::from_buffer(a.shape.clone(), out)
}

fn time(f: impl Fn() -> TensorStorage, iters: usize) -> Duration {
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    t.elapsed() / iters as u32
}

fn bench_one(name: &str, a: &TensorStorage, b: &TensorStorage) {
    const ITERS: usize = 100;

    // correctness: both paths produce the same elementwise product
    let old = old_mul(a, b);
    let new = TensorStorage::mul(&[a, b]);
    assert_eq!(old.buffer.as_ref(), new.buffer.as_ref());

    for _ in 0..10 {
        black_box(old_mul(a, b));
        black_box(TensorStorage::mul(&[a, b]));
    }

    let t_old = time(|| old_mul(a, b), ITERS).as_secs_f64() * 1e3;
    let t_new = time(|| TensorStorage::mul(&[a, b]), ITERS).as_secs_f64() * 1e3;
    eprintln!(
        "{name:28} logic_to_flat {:>9.3}ms   StridedIter {:>9.3}ms   ratio {:.2}x",
        t_old,
        t_new,
        t_old / t_new
    );
}

#[test]
fn strided_indices_yields_logic_to_flat_order() {
    let a = TensorStorage::from_buffer(vec![2, 3], (1..=6).map(|x| x as f64).collect());
    let t = TensorStorage::transpose(&a); // [3,2] strided view
    assert!(!t.contiguous);

    // [2,3]^T fills column-major: [[1,4],[2,5],[3,6]] => flat [1,4,2,5,3,6]
    let by_odometer: Vec<f64> = t.strided_indices().map(|f| t.buffer[f]).collect();
    let by_logic: Vec<f64> = (0..t.numel).map(|i| t[i]).collect();
    assert_eq!(by_odometer, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(by_odometer, by_logic);
    assert_eq!(by_odometer.len(), 6);
}

#[test]
fn mul_matches_old_path_on_strided_views() {
    // transposed views: both operands strided
    let a = TensorStorage::from_buffer(vec![3, 4], (1..=12).map(|x| x as f64).collect());
    let b = TensorStorage::from_buffer(vec![4, 3], (1..=12).map(|x| x as f64).collect());
    let ta = TensorStorage::transpose(&a); // [4,3]
    let tb = TensorStorage::transpose(&b); // [3,4]
    assert_eq!(ta.shape, vec![4, 3]);
    assert_eq!(tb.shape, vec![3, 4]);

    let old = old_mul(&ta, &tb);
    let new = TensorStorage::mul(&[&ta, &tb]);
    assert_eq!(old.buffer.as_ref(), new.buffer.as_ref());
    assert_eq!(new.shape, vec![4, 3]);
}

#[test]
fn mul_matches_old_path_on_broadcast_views() {
    // bias [1,256] broadcast to [4096,256]: strides [0,1], inner run 256
    let x = TensorStorage::from_buffer(vec![4096, 256], (1..=4096 * 256).map(|x| x as f64).collect());
    let bias = TensorStorage::from_buffer(vec![1, 256], (1..=256).map(|x| x as f64).collect());
    let bb = bias.broadcast_to_shape(&vec![4096, 256]);
    assert!(!bb.contiguous);
    assert_eq!(bb.strides, vec![0, 1]);

    let old = old_mul(&x, &bb);
    let new = TensorStorage::mul(&[&x, &bb]);
    assert_eq!(old.buffer.as_ref(), new.buffer.as_ref());
    assert_eq!(new.shape, vec![4096, 256]);
}

#[test]
fn mul_matches_old_path_on_3d_broadcast() {
    // [64,64,256] x [1,1,256] -> inner run 256, outer dims [64,64]
    let x = TensorStorage::from_buffer(
        vec![64, 64, 256],
        (1..=64 * 64 * 256).map(|x| x as f64).collect(),
    );
    let bias = TensorStorage::from_buffer(vec![1, 1, 256], (1..=256).map(|x| x as f64).collect());
    let bb = bias.broadcast_to_shape(&vec![64, 64, 256]);
    assert_eq!(bb.strides, vec![0, 0, 1]);

    let old = old_mul(&x, &bb);
    let new = TensorStorage::mul(&[&x, &bb]);
    assert_eq!(old.buffer.as_ref(), new.buffer.as_ref());
    assert_eq!(new.shape, vec![64, 64, 256]);
}

/// Run with: cargo test --release -- --ignored --nocapture bench_strided_mul
#[test]
#[ignore]
fn bench_strided_mul() {
    const M: usize = 1 << 20; // 1 Mi elems = 8 MB of f64

    // 2D strided view of an 8MB buffer (transpose): [1024,1024]^T
    let c = TensorStorage::from_buffer(vec![1024, 1024], (0..M).map(|i| i as f64).collect());
    let t2 = TensorStorage::transpose(&c);
    bench_one("strided [1024,1024]^T (8MB)", &t2, &t2);

    // transposed view: [256,4096]^T = [4096,256]
    let c3 = TensorStorage::from_buffer(vec![256, 4096], (0..M).map(|i| i as f64).collect());
    let t3 = TensorStorage::transpose(&c3);
    bench_one("strided [4096,256]^T (8MB)", &t3, &t3);

    // one strided, one contiguous operand (like a broadcasted binary op)
    let d = TensorStorage::from_buffer(vec![4096, 256], (0..M).map(|i| i as f64).collect());
    bench_one("strided^T x contig (8MB)", &t3, &d);

    // bias-broadcast: [256] -> [4096,256], strides [0,1] (inner run 256)
    let bias = TensorStorage::from_buffer(vec![256], (0..256).map(|i| i as f64).collect());
    let bias_b = bias.broadcast_to_shape(&vec![4096, 256]);
    assert_eq!(bias_b.strides, vec![0, 1]);
    bench_one("contig x bias-bcast (8MB)", &d, &bias_b);

    // 3D bias-broadcast: [1,1,256] -> [64,64,256], strides [0,0,1] (inner run 256)
    let x3 = TensorStorage::from_buffer(
        vec![64, 64, 256],
        (0..M).map(|i| i as f64).collect(),
    );
    let bias3 = TensorStorage::from_buffer(vec![1, 1, 256], (0..256).map(|i| i as f64).collect());
    let bias3_b = bias3.broadcast_to_shape(&vec![64, 64, 256]);
    bench_one("contig x 3D-bcast (8MB)", &x3, &bias3_b);
}