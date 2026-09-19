use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::compute::AvgPool2d;
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};

fn approx(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9,
        "expected {b}, got {a} (diff {})",
        (a - b).abs()
    );
}

/// Consecutive values `0..numel` in row-major order (a `FreeTensor` gives us a
/// graph leaf we can take gradients of).
fn ramped(shape: &[usize]) -> GraphTensor {
    let mut t = FreeTensor::new(shape.to_vec(), 0.0, true);
    let mut counter = 0.0;
    fn walk(shape: &[usize], coords: &mut Vec<usize>, t: &mut FreeTensor, counter: &mut f64) {
        if coords.len() == shape.len() {
            t.set(coords, *counter);
            *counter += 1.0;
            return;
        }
        let d = coords.len();
        for i in 0..shape[d] {
            coords.push(i);
            walk(shape, coords, t, counter);
            coords.pop();
        }
    }
    walk(shape, &mut Vec::new(), &mut t, &mut counter);
    t.to_graph()
}

/// Flat -> row-major coordinates under `shape`.
fn unflatten(shape: &[usize], flat: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut rem = flat;
    for d in (0..shape.len()).rev() {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
    coords
}

/// Window-mean reference: `out[b, c, oh, ow] = mean of window at
/// (oh*sh, ow*sw)`.
fn ref_forward(input: &GraphTensor, kernel: (usize, usize), stride: (usize, usize)) -> Vec<f64> {
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let out_h = (input.shape()[2] - kh) / sh + 1;
    let out_w = (input.shape()[3] - kw) / sw + 1;

    let mut out = Vec::new();
    for b in 0..input.shape()[0] {
        for c in 0..input.shape()[1] {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0;
                    for i in 0..kh {
                        for j in 0..kw {
                            acc += *input.at(&[b, c, oh * sh + i, ow * sw + j]);
                        }
                    }
                    out.push(acc / (kh * kw) as f64);
                }
            }
        }
    }
    out
}

/// Gradient of `sum(out^2)` wrt the input: element `(b, c, hi, wi)` receives
/// `2 * out[b, c, oh, ow] / (kh * kw)` for every window that covers it.
fn ref_weighted_input_grad(
    input: &GraphTensor,
    out: &GraphTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Vec<f64> {
    let (kh, kw) = kernel;
    let (sh, sw) = stride;

    let mut grad = vec![0.0; input.shape().iter().product()];
    for b in 0..input.shape()[0] {
        for c in 0..input.shape()[1] {
            for hi in 0..input.shape()[2] {
                for wi in 0..input.shape()[3] {
                    let mut acc = 0.0;
                    for oh in 0..out.shape()[2] {
                        for ow in 0..out.shape()[3] {
                            let covers = oh * sh <= hi
                                && hi < oh * sh + kh
                                && ow * sw <= wi
                                && wi < ow * sw + kw;
                            if covers {
                                acc += 2.0 * *out.at(&[b, c, oh, ow]) / (kh * kw) as f64;
                            }
                        }
                    }
                    let idx = ((b * input.shape()[1] + c) * input.shape()[2] + hi)
                        * input.shape()[3]
                        + wi;
                    grad[idx] = acc;
                }
            }
        }
    }
    grad
}

#[test]
fn avg_pool2d_forward_matches_reference() {
    let input = ramped(&[2, 3, 4, 4]);

    let out = input.avg_pool2d((2, 2), (1, 1));
    assert_eq!(out.shape(), &[2, 3, 3, 3]);

    let expected = ref_forward(&input, (2, 2), (1, 1));
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(out.shape(), flat);
        approx(*out.at(&idx), *exp);
    }

    let pooled = input.avg_pool2d((2, 2), (2, 2));
    assert_eq!(pooled.shape(), &[2, 3, 2, 2]);
    let expected = ref_forward(&input, (2, 2), (2, 2));
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(pooled.shape(), flat);
        approx(*pooled.at(&idx), *exp);
    }

    let rect = input.avg_pool2d((3, 2), (1, 2));
    assert_eq!(rect.shape(), &[2, 3, 2, 2]);
    let expected = ref_forward(&input, (3, 2), (1, 2));
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(rect.shape(), flat);
        approx(*rect.at(&idx), *exp);
    }
}

#[test]
fn avg_pool2d_forward_overlapping_windows() {
    // 3x3 plane 0..=8 pooled with a 2x2 kernel at stride 1.
    let input = ramped(&[1, 1, 3, 3]);
    let out = input.avg_pool2d((2, 2), (1, 1));

    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    let expected = [2.0, 3.0, 5.0, 6.0]; // e.g. [0,0]: (0+1+3+4)/4
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(out.shape(), flat);
        approx(*out.at(&idx), *exp);
    }
}

#[test]
fn avg_pool2d_backward_scatters_over_windows() {
    let input = ramped(&[1, 1, 3, 3]);
    let out = input.avg_pool2d((2, 2), (1, 1));
    let grads = out.sum(&[], false).backward(true);

    let dx = grads.get(&input).unwrap();
    assert_eq!(dx.shape(), &[1, 1, 3, 3]);
    // every window contributes 1/4 of its upstream gradient (1.0 here) per
    // covered position: corners 1 window, edges 2, center 4.
    let expected = [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25];
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(dx.shape(), flat);
        approx(*dx.at(&idx), *exp);
    }
}

#[test]
fn avg_pool2d_weighted_backward_matches_reference() {
    let input = ramped(&[2, 3, 4, 4]);
    let out = input.avg_pool2d((2, 2), (1, 1));

    let loss = (&out * &out).sum(&[], false);
    let grads = loss.backward(true);

    let dx = grads.get(&input).unwrap();
    assert_eq!(dx.shape(), &[2, 3, 4, 4]);
    let expected = ref_weighted_input_grad(&input, &out, (2, 2), (1, 1));
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(dx.shape(), flat);
        approx(*dx.at(&idx), *exp);
    }
}

#[test]
fn avg_pool2d_is_generic_over_float_dtype() {
    let input = GraphTensor::<f32>::wrap(
        vec![vec![vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ]]],
        true,
    );
    let out = input.avg_pool2d((2, 2), (2, 2));
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert!((*out.at(&[0, 0, 0, 0]) - 3.5).abs() < 1e-6);
    assert!((*out.at(&[0, 0, 0, 1]) - 5.5).abs() < 1e-6);
    assert!((*out.at(&[0, 0, 1, 0]) - 11.5).abs() < 1e-6);
    assert!((*out.at(&[0, 0, 1, 1]) - 13.5).abs() < 1e-6);
}

#[test]
fn avgpool2d_module_forward_defaults_stride_to_kernel() {
    let pool = AvgPool2d::new(2, None);
    let input = GraphTensor::new(vec![1, 1, 4, 4], 2.5, false);

    let out = pool.forward(&input);
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    for b in 0..2 {
        for c in 0..2 {
            approx(*out.at(&[0, 0, b, c]), 2.5);
        }
    }
}

#[test]
fn avgpool2d_module_forward_explicit_stride() {
    let pool = AvgPool2d::new(2, Some(1));
    let input = GraphTensor::new(vec![1, 1, 4, 4], 2.0, false);

    let out = pool.forward(&input);
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
    for oh in 0..3 {
        for ow in 0..3 {
            approx(*out.at(&[0, 0, oh, ow]), 2.0);
        }
    }
}

#[test]
fn avg_pool2d_propagates_requires_grad() {
    let input = ramped(&[1, 1, 4, 4]);
    assert!(input.requires_grad());
    let out = input.avg_pool2d((2, 2), (2, 2));
    assert!(out.requires_grad());

    let frozen = GraphTensor::new(vec![1, 1, 4, 4], 1.0, false);
    let hanging = frozen.avg_pool2d((2, 2), (2, 2));
    assert!(!hanging.requires_grad());
}

#[test]
#[should_panic]
fn avg_pool2d_rejects_non_4d_input() {
    ramped(&[1, 2, 4]).avg_pool2d((2, 2), (2, 2));
}
