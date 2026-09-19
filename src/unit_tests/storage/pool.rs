use crate::core::storage::TensorStorage;

/// Read a (possibly strided/view) storage back into a row-major logical
/// buffer, matching the semantics the kernels are supposed to implement:
/// values are addressed by logical coordinates, not physical layout.
fn row_major(t: &TensorStorage<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(t.numel);
    for flat in 0..t.numel {
        let mut rem = flat;
        let mut offset = t.offset;
        for (d, &size) in t.shape.iter().enumerate().rev() {
            let coord = rem % size;
            offset += coord * t.strides[d];
            rem /= size;
        }
        out.push(t.buffer[offset]);
    }
    out
}

/// Window-mean reference over a row-major `[b, c, h, w]` buffer.
fn ref_forward(
    a: &[f64],
    shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Vec<f64> {
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let inv = 1.0 / (kh * kw) as f64;

    let mut out = Vec::with_capacity(b * c * out_h * out_w);
    for batch in 0..b {
        for chan in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0;
                    for i in 0..kh {
                        for j in 0..kw {
                            let idx = ((batch * c + chan) * h + oh * sh + i) * w + ow * sw + j;
                            acc += a[idx];
                        }
                    }
                    out.push(acc * inv);
                }
            }
        }
    }
    out
}

/// Reference backward: scatter `dy[b, c, oh, ow] / (kh * kw)` over every input
/// position its window covered, accumulating overlaps.
fn ref_backward(
    dy: &[f64],
    out_shape: &[usize],
    input_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Vec<f64> {
    let (b, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let inv = 1.0 / (kh * kw) as f64;
    let (out_h, out_w) = (out_shape[2], out_shape[3]);

    let mut out = vec![0.0; input_shape.iter().product()];
    for batch in 0..b {
        for chan in 0..c {
            let d_plane = (batch * c + chan) * out_h * out_w;
            let out_plane = (batch * c + chan) * h * w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let g = dy[d_plane + oh * out_w + ow] * inv;
                    for i in 0..kh {
                        for j in 0..kw {
                            out[out_plane + (oh * sh + i) * w + ow * sw + j] += g;
                        }
                    }
                }
            }
        }
    }
    out
}

fn approx(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9,
        "expected {b}, got {a} (diff {})",
        (a - b).abs()
    );
}

fn check_forward(a: &TensorStorage<f64>, kernel: (usize, usize), stride: (usize, usize)) {
    let out = TensorStorage::avg_pool2d(a, kernel, stride);
    let expected = ref_forward(&row_major(a), &a.shape, kernel, stride);
    assert_eq!(
        out.shape,
        vec![
            a.shape[0],
            a.shape[1],
            (a.shape[2] - kernel.0) / stride.0 + 1,
            (a.shape[3] - kernel.1) / stride.1 + 1,
        ]
    );
    assert_eq!(out.buffer.len(), expected.len());
    for (got, exp) in out.buffer.iter().zip(&expected) {
        approx(*got, *exp);
    }
}

#[test]
fn avg_pool2d_forward_matches_reference() {
    // [2, 3, 4, 4] tile of consecutive values: exercises every non-degenerate
    // window placement of a 2x2 kernel at both strides.
    let a = TensorStorage::from_buffer(vec![2, 3, 4, 4], (0..96).map(|i| i as f64).collect());
    check_forward(&a, (2, 2), (2, 2));
    check_forward(&a, (2, 2), (1, 1));
    check_forward(&a, (4, 4), (4, 4)); // single full-image window
    check_forward(&a, (1, 1), (2, 2)); // identity kernel, strided sampling

    // Rectangular kernel (3, 2) with a non-uniform stride.
    let b = TensorStorage::from_buffer(vec![1, 2, 4, 4], (0..32).map(|i| i as f64).collect());
    check_forward(&b, (3, 2), (1, 2));
    check_forward(&b, (2, 3), (2, 1));
}

#[test]
fn avg_pool2d_forward_handles_strided_views() {
    // A transposed input is a strided view: results must track the *logical*
    // [b, c, h, w] coordinates.
    let a = TensorStorage::from_buffer(vec![2, 3, 4, 4], (0..96).map(|i| i as f64).collect());
    let t = TensorStorage::transpose(&a, 2, 3);
    assert!(!t.contiguous);

    check_forward(&t, (2, 2), (1, 1));
    check_forward(&t, (2, 2), (2, 2));

    // Double-transposition is a view of the view, with two levels of striding.
    let tt = TensorStorage::transpose(&t, 1, 2);
    check_forward(&tt, (2, 2), (1, 1));
}

#[test]
fn avg_pool2d_backward_matches_reference() {
    let input_shape = vec![2, 3, 4, 4];
    let dy_shape = vec![2, 3, 3, 3]; // kernel (2,2), stride (1,1)

    let dy =
        TensorStorage::from_buffer(dy_shape.clone(), (0..54).map(|i| i as f64 * 0.25).collect());
    let dx = TensorStorage::avg_pool2d_backward(&dy, &input_shape, (2, 2), (1, 1));
    assert_eq!(dx.shape, input_shape);

    let expected = ref_backward(&dy.buffer, &dy_shape, &input_shape, (2, 2), (1, 1));
    for (got, exp) in dx.buffer.iter().zip(&expected) {
        approx(*got, *exp);
    }

    // A non-overlapping stride: every window scatters to a disjoint block, so
    // each input position receives exactly one contribution.
    let dy2 = TensorStorage::from_buffer(vec![2, 2, 2, 2], (0..16).map(|i| i as f64).collect());
    let dx2 = TensorStorage::avg_pool2d_backward(&dy2, &[2, 2, 8, 8], (4, 4), (4, 4));
    let expected2 = ref_backward(&dy2.buffer, &dy2.shape, &[2, 2, 8, 8], (4, 4), (4, 4));
    for (got, exp) in dx2.buffer.iter().zip(&expected2) {
        approx(*got, *exp);
    }
}

#[test]
fn avg_pool2d_backward_accumulates_overlapping_windows() {
    // A single 3x3 plane pooled with a 2x2 kernel at stride 1: the center
    // pixel is covered by all four windows.
    let dy = TensorStorage::from_buffer(vec![1, 1, 2, 2], vec![1.0; 4]);
    let dx = TensorStorage::avg_pool2d_backward(&dy, &[1, 1, 3, 3], (2, 2), (1, 1));

    let expected = [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25];
    for (got, exp) in dx.buffer.iter().zip(&expected) {
        approx(*got, *exp);
    }
}

#[test]
#[should_panic]
fn avg_pool2d_rejects_non_4d() {
    let a = TensorStorage::from_buffer(vec![1, 2, 3], vec![0.0; 6]);
    let _ = TensorStorage::avg_pool2d(&a, (2, 2), (2, 2));
}

#[test]
#[should_panic]
fn avg_pool2d_rejects_zero_stride() {
    let a = TensorStorage::from_buffer(vec![1, 1, 4, 4], vec![0.0; 16]);
    let _ = TensorStorage::avg_pool2d(&a, (2, 2), (0, 2));
}

#[test]
#[should_panic]
fn avg_pool2d_rejects_oversized_kernel() {
    let a = TensorStorage::from_buffer(vec![1, 1, 4, 4], vec![0.0; 16]);
    let _ = TensorStorage::avg_pool2d(&a, (5, 2), (1, 1));
}

#[test]
#[should_panic]
fn avg_pool2d_backward_rejects_mismatched_upstream_shape() {
    let dy = TensorStorage::from_buffer(vec![1, 1, 3, 3], vec![0.0; 9]);
    let _ = TensorStorage::avg_pool2d_backward(&dy, &[1, 1, 4, 4], (3, 3), (1, 1));
}
