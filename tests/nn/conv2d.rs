use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::compute::{
    Conv2d, Conv2dPadding, Size2, conv2d, conv2d_with_options,
};
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::tensor::AbstractTensor;

use rand::SeedableRng;
use rand::rngs::StdRng;

fn approx(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-9,
        "expected {b}, got {a} (diff {})",
        (a - b).abs()
    );
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

/// Reference cross-correlation: `out[b, co, oh, ow] = sum_{ci,i,j} input[b, ci,
/// oh+i, ow+j] * weight[ci, co, i, j]` (no kernel flip, like `nn.Conv2d`).
fn ref_forward(input: &GraphTensor, weight: &GraphTensor) -> Vec<f64> {
    let out_h = input.shape()[2] - weight.shape()[2] + 1;
    let out_w = input.shape()[3] - weight.shape()[3] + 1;

    let mut out = Vec::new();
    for b in 0..input.shape()[0] {
        for co in 0..weight.shape()[1] {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0;
                    for ci in 0..input.shape()[1] {
                        for i in 0..weight.shape()[2] {
                            for j in 0..weight.shape()[3] {
                                acc += *input.at(&[b, ci, oh + i, ow + j])
                                    * *weight.at(&[ci, co, i, j]);
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    out
}

/// Gradient of `sum(conv(input, weight))` wrt the input: element `(b, ci, hi,
/// wi)` receives `weight[ci, co, i, j]` for every output position and tap that
/// reads it.
fn ref_input_grad(input_shape: &[usize], weight: &GraphTensor) -> Vec<f64> {
    let out_h = input_shape[2] - weight.shape()[2] + 1;
    let out_w = input_shape[3] - weight.shape()[3] + 1;
    let out_co = weight.shape()[1];

    let mut grad = vec![0.0; input_shape.iter().product()];
    for b in 0..input_shape[0] {
        for ci in 0..input_shape[1] {
            for hi in 0..input_shape[2] {
                for wi in 0..input_shape[3] {
                    let mut acc = 0.0;
                    for co in 0..out_co {
                        for i in 0..weight.shape()[2] {
                            for j in 0..weight.shape()[3] {
                                if hi >= i && wi >= j && hi - i < out_h && wi - j < out_w {
                                    acc += *weight.at(&[ci, co, i, j]);
                                }
                            }
                        }
                    }
                    let idx =
                        ((b * input_shape[1] + ci) * input_shape[2] + hi) * input_shape[3] + wi;
                    grad[idx] = acc;
                }
            }
        }
    }
    grad
}

/// Gradient of `sum(conv(input, weight))` wrt the weight.
fn ref_weight_grad(input: &GraphTensor, weight: &GraphTensor) -> Vec<f64> {
    let out_h = input.shape()[2] - weight.shape()[2] + 1;
    let out_w = input.shape()[3] - weight.shape()[3] + 1;
    let out_co = weight.shape()[1];

    let mut grad = vec![0.0; weight.shape().iter().product()];
    for ci in 0..input.shape()[1] {
        for co in 0..out_co {
            for i in 0..weight.shape()[2] {
                for j in 0..weight.shape()[3] {
                    let mut acc = 0.0;
                    for b in 0..input.shape()[0] {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                acc += *input.at(&[b, ci, oh + i, ow + j]);
                            }
                        }
                    }
                    let idx = ((ci * out_co + co) * weight.shape()[2] + i) * weight.shape()[3] + j;
                    grad[idx] = acc;
                }
            }
        }
    }
    grad
}

fn test_input() -> GraphTensor {
    // [1, 2, 3, 3]
    GraphTensor::wrap(
        vec![vec![
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            vec![
                vec![10.0, 11.0, 12.0],
                vec![13.0, 14.0, 15.0],
                vec![16.0, 17.0, 18.0],
            ],
        ]],
        true,
    )
}

fn test_weight() -> GraphTensor {
    // [2, 1, 2, 2]
    GraphTensor::wrap(
        vec![
            vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]],
            vec![vec![vec![5.0, 6.0], vec![7.0, 8.0]]],
        ],
        true,
    )
}

#[test]
fn conv2d_free_function_forward_matches_reference() {
    let input = test_input();
    let weight = test_weight();

    let out = conv2d(&input, &weight);
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    let expected = [356.0, 392.0, 464.0, 500.0];
    let got = ref_forward(&input, &weight);
    for (got, exp) in got.iter().zip(expected) {
        approx(*got, exp);
    }
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(out.shape(), flat);
        approx(*out.at(&idx), *exp);
    }
}

#[test]
fn conv2d_backward_profiling_reports_split_sections() {
    let input = test_input();
    let weight = test_weight();
    let output = conv2d(&input, &weight);
    let loss = output.sum(&[], false);
    let (grads, timings) = loss.backward_profiled(false);

    assert!(grads.get(&input).is_some());
    let conv_timing = timings
        .iter()
        .find(|timing| timing.operation == "conv2d")
        .expect("profiled convolution timing missing");
    assert_eq!(conv_timing.details.len(), 6);
    assert!(
        conv_timing
            .details
            .iter()
            .any(|(name, _)| *name == "grad_input")
    );
    assert!(
        conv_timing
            .details
            .iter()
            .any(|(name, _)| *name == "grad_weight")
    );
}

#[test]
fn conv2d_free_function_is_generic_over_dtype() {
    let input = GraphTensor::<f32>::wrap(vec![vec![vec![vec![1.0f32, 2.0, 3.0]]]], false); // [1, 1, 1, 3]
    let weight = GraphTensor::<f32>::wrap(vec![vec![vec![vec![1.0f32, 0.0]]]], false); // [1, 1, 1, 2]

    let out = conv2d(&input, &weight);
    assert_eq!(out.shape(), &[1, 1, 1, 2]);
    assert!((*out.at(&[0, 0, 0, 0]) - 1.0).abs() < 1e-6);
    assert!((*out.at(&[0, 0, 0, 1]) - 2.0).abs() < 1e-6);
}

#[test]
fn conv2d_multichannel_batch_matches_reference() {
    // batch of 2, 3 channels, 4x4 input, 2x2 kernel
    let input = GraphTensor::new(vec![2, 3, 4, 4], 1.5, false);
    let weight = GraphTensor::new(vec![3, 2, 2, 2], 0.25, false);

    let out = conv2d(&input, &weight);
    assert_eq!(out.shape(), &[2, 2, 3, 3]);

    // constant inputs: every output is 1.5 * 0.25 * (3 input chan * 4 taps) = 4.5
    for b in 0..2 {
        for co in 0..2 {
            for oh in 0..3 {
                for ow in 0..3 {
                    approx(*out.at(&[b, co, oh, ow]), 4.5);
                }
            }
        }
    }
}

#[test]
fn conv2d_module_forward_applies_bias() {
    let rng = StdRng::seed_from_u64(7);
    let mut conv = Conv2d::new(2, 1, 2, true, rng);

    // overwrite the randomly initialized params with known values
    conv.weight = test_weight();
    conv.bias = Some(GraphTensor::wrap(vec![0.5], true));

    let out = conv.forward(&test_input(), false);
    assert_eq!(out.shape(), &[1, 1, 2, 2]);

    let expected = [356.5, 392.5, 464.5, 500.5];
    for (flat, exp) in expected.iter().enumerate() {
        let idx = unflatten(out.shape(), flat);
        approx(*out.at(&idx), *exp);
    }
}

#[test]
fn conv2d_module_backward_matches_reference_gradients() {
    let rng = StdRng::seed_from_u64(11);
    let mut conv = Conv2d::new(2, 1, 2, true, rng);

    conv.weight = test_weight();
    conv.bias = Some(GraphTensor::wrap(vec![0.5], true));

    let input = test_input();
    let out = conv.forward(&input, false);
    let loss = out.sum(&[], false);

    let grads = loss.backward(true);

    let dx = grads.get(&input).unwrap();
    assert_eq!(dx.shape(), &[1, 2, 3, 3]);
    for (flat, exp) in ref_input_grad(input.shape(), &conv.weight)
        .iter()
        .enumerate()
    {
        let idx = unflatten(dx.shape(), flat);
        approx(*dx.at(&idx), *exp);
    }

    let dw = grads.get(&conv.weight).unwrap();
    assert_eq!(dw.shape(), &[2, 1, 2, 2]);
    for (flat, exp) in ref_weight_grad(&input, &conv.weight).iter().enumerate() {
        let idx = unflatten(dw.shape(), flat);
        approx(*dw.at(&idx), *exp);
    }

    let db = grads.get(conv.bias.as_ref().unwrap()).unwrap();
    assert_eq!(db.shape(), &[1]);
    approx(*db.at(&[0]), 4.0);
}

#[test]
fn conv2d_same_padding_preserves_spatial_dimensions() {
    let input = GraphTensor::new(vec![1, 1, 5, 5], 1.0, false);
    let conv = Conv2d::new_with_options(
        1,
        1,
        3,
        1,
        Conv2dPadding::Same,
        1,
        false,
        StdRng::seed_from_u64(41),
    );

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 5, 5]);
}

#[test]
fn conv2d_square_five_by_five_matches_reference() {
    let input = GraphTensor::new(vec![1, 1, 5, 5], 1.0, true);
    let mut conv = Conv2d::new_with_options(
        1,
        1,
        5,
        1,
        Conv2dPadding::Valid,
        1,
        false,
        StdRng::seed_from_u64(46),
    );
    conv.weight = GraphTensor::new(vec![1, 1, 5, 5], 0.5, true);

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 1, 1]);
    approx(*output.at(&[0, 0, 0, 0]), 12.5);

    let grads = output.sum(&[], false).backward(true);
    let dx = grads.get(&input).unwrap();
    let dw = grads.get(&conv.weight).unwrap();
    for flat in 0..dx.numel() {
        let index = unflatten(dx.shape(), flat);
        approx(*dx.at(&index), 0.5);
    }
    for flat in 0..dw.numel() {
        let index = unflatten(dw.shape(), flat);
        approx(*dw.at(&index), 1.0);
    }
}

#[test]
fn conv2d_supports_stride_and_dilation() {
    let input = GraphTensor::new(vec![1, 1, 7, 7], 1.0, false);
    let conv = Conv2d::new_with_options(
        1,
        1,
        3,
        2,
        Conv2dPadding::Same,
        2,
        false,
        StdRng::seed_from_u64(42),
    );

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 4, 4]);
}

#[test]
fn conv2d_accepts_numeric_padding_shorthand() {
    let input = GraphTensor::new(vec![1, 1, 5, 5], 1.0, false);
    let conv = Conv2d::new_with_options(1, 1, 3, 2, 1, 1, false, StdRng::seed_from_u64(43));

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
}

#[test]
fn conv2d_dilation_forward_and_backward_match_reference() {
    let input = GraphTensor::wrap(
        vec![vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ]]],
        true,
    );
    let mut conv = Conv2d::new_with_options(
        1,
        1,
        2,
        1,
        Conv2dPadding::Valid,
        2,
        false,
        StdRng::seed_from_u64(44),
    );
    conv.weight = GraphTensor::wrap(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], true);

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    let expected = [78.0, 88.0, 118.0, 128.0];
    for (flat, value) in expected.iter().enumerate() {
        let index = unflatten(output.shape(), flat);
        approx(*output.at(&index), *value);
    }

    let grads = output.sum(&[], false).backward(true);
    let dx = grads.get(&input).unwrap();
    let expected_dx = [
        1.0, 1.0, 2.0, 2.0, //
        1.0, 1.0, 2.0, 2.0, //
        3.0, 3.0, 4.0, 4.0, //
        3.0, 3.0, 4.0, 4.0,
    ];
    for (flat, value) in expected_dx.iter().enumerate() {
        let index = unflatten(dx.shape(), flat);
        approx(*dx.at(&index), *value);
    }
}

#[test]
fn conv2d_direct_backward_handles_same_padding() {
    let input = GraphTensor::wrap(
        vec![vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]],
        true,
    );
    let mut conv = Conv2d::new_with_options(
        1,
        1,
        2,
        1,
        Conv2dPadding::Same,
        1,
        false,
        StdRng::seed_from_u64(45),
    );
    conv.weight = GraphTensor::wrap(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], true);

    let output = conv.forward(&input, false);
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    let grads = output.sum(&[], false).backward(true);

    let dx = grads.get(&input).unwrap();
    let expected_dx = [1.0, 3.0, 3.0, 4.0, 10.0, 10.0, 4.0, 10.0, 10.0];
    for (flat, value) in expected_dx.iter().enumerate() {
        let index = unflatten(dx.shape(), flat);
        approx(*dx.at(&index), *value);
    }

    let dw = grads.get(&conv.weight).unwrap();
    let expected_dw = [45.0, 33.0, 39.0, 28.0];
    for (flat, value) in expected_dw.iter().enumerate() {
        let index = unflatten(dw.shape(), flat);
        approx(*dw.at(&index), *value);
    }
}

/// Check forward, `grad_input`, and `grad_weight` against the reference
/// implementations for a valid-padding stride-1/dilation-1 convolution.
///
/// Zero padding is deliberate: it makes edge output positions boundary
/// positions, so the interior/boundary split in the input-gradient kernel is
/// covered by the same test.
fn assert_conv_matches_reference(input: &GraphTensor, weight: &GraphTensor) {
    let output = conv2d(input, weight);
    let expected = ref_forward(input, weight);
    assert_eq!(
        output.shape(),
        &[
            input.shape()[0],
            weight.shape()[1],
            input.shape()[2] - weight.shape()[2] + 1,
            input.shape()[3] - weight.shape()[3] + 1,
        ]
    );
    for (flat, value) in expected.iter().enumerate() {
        approx(*output.at(&unflatten(output.shape(), flat)), *value);
    }

    let grads = output.sum(&[], false).backward(true);

    let dx = grads.get(input).unwrap();
    let expected_dx = ref_input_grad(input.shape(), weight);
    for (flat, value) in expected_dx.iter().enumerate() {
        approx(*dx.at(&unflatten(dx.shape(), flat)), *value);
    }

    let dw = grads.get(weight).unwrap();
    let expected_dw = ref_weight_grad(input, weight);
    for (flat, value) in expected_dw.iter().enumerate() {
        approx(*dw.at(&unflatten(dw.shape(), flat)), *value);
    }
}

/// Build an input whose every element is distinct, so an indexing slip in the
/// tiled kernels cannot cancel out. A constant-filled tensor would make any
/// permutation of the output-channel axis undetectable.
fn distinct_input(shape: &[usize]) -> GraphTensor {
    let mut flat = Vec::with_capacity(shape.iter().product());
    for i in 0..shape.iter().product::<usize>() {
        flat.push(i as f64 * 0.017 - 0.31);
    }
    let mut rows: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(shape[0]);
    for b in 0..shape[0] {
        let mut channels = Vec::with_capacity(shape[1]);
        for c in 0..shape[1] {
            let mut planes = Vec::with_capacity(shape[2]);
            for h in 0..shape[2] {
                let start = (b * shape[1] * shape[2] + c * shape[2] + h) * shape[3];
                planes.push(flat[start..start + shape[3]].to_vec());
            }
            channels.push(planes);
        }
        rows.push(channels);
    }
    GraphTensor::wrap(rows, true)
}

/// Build a weight with a distinct value per `(ci, co, i, j)`.
fn distinct_weight(in_channels: usize, out_channels: usize, kh: usize, kw: usize) -> GraphTensor {
    let mut rows = Vec::with_capacity(in_channels);
    for ci in 0..in_channels {
        let mut channels = Vec::with_capacity(out_channels);
        for co in 0..out_channels {
            let mut plane: Vec<Vec<f64>> = Vec::with_capacity(kh);
            for i in 0..kh {
                let row = (0..kw)
                    .map(|j| {
                        ci as f64 * 0.7 + co as f64 * 0.13 + i as f64 * 0.031 + j as f64 * 0.017
                    })
                    .collect();
                plane.push(row);
            }
            channels.push(plane);
        }
        rows.push(channels);
    }
    GraphTensor::wrap(rows, true)
}

/// The register-tiled loops process output channels in blocks of eight, so a
/// tensor narrower than that exercises only the remainder path. These cover the
/// tiled body, an exact multiple of the tile width, and a remainder.
#[test]
fn conv2d_output_channel_tiling_covers_tile_and_remainder() {
    for out_channels in [8usize, 10, 64] {
        let input = distinct_input(&[2, 3, 5, 4]);
        let weight = distinct_weight(3, out_channels, 3, 2);
        assert_conv_matches_reference(&input, &weight);
    }
}

/// Same padding with more than eight output channels must agree exactly with
/// explicitly zero-padding the input and running the valid-padding path.
#[test]
fn conv2d_tiled_same_padding_matches_explicit_padding() {
    for out_channels in [9usize, 16] {
        let input = distinct_input(&[2, 2, 5, 5]);
        let weight = distinct_weight(2, out_channels, 3, 3);

        let same =
            conv2d_with_options(&input, &weight, Size2::ONE, Conv2dPadding::Same, Size2::ONE);
        let padded = input.pad(&[(0, 0), (0, 0), (1, 1), (1, 1)]);
        let valid = conv2d(&padded, &weight);

        assert_eq!(same.shape(), &[2, out_channels, 5, 5]);
        assert_eq!(same.shape(), valid.shape());
        for flat in 0..same.numel() {
            approx(
                *same.at(&unflatten(same.shape(), flat)),
                *valid.at(&unflatten(valid.shape(), flat)),
            );
        }
    }
}

#[test]
#[should_panic]
fn conv2d_channel_mismatch_panics() {
    let input = GraphTensor::new(vec![1, 3, 4, 4], 1.0, false);
    let weight = GraphTensor::new(vec![2, 1, 2, 2], 1.0, false);
    let _ = conv2d(&input, &weight);
}
