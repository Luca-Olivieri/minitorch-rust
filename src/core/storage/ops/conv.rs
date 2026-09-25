use std::time::{Duration, Instant};

use crate::core::dtype::Numeric;
use crate::core::storage::TensorStorage;

/// Gradients produced by the direct convolution backward kernel.
pub(crate) struct Conv2dBackward<T: Numeric> {
    pub(crate) grad_input: TensorStorage<T>,
    pub(crate) grad_weight: TensorStorage<T>,
}

/// Timing sections collected by a profiled convolution backward pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv2dBackwardTiming {
    pub(crate) pack_grad_output: Duration,
    pub(crate) pack_weight: Duration,
    pub(crate) padded_input: Duration,
    pub(crate) grad_weight: Duration,
    pub(crate) grad_input: Duration,
    pub(crate) unpack_weight: Duration,
}

impl Default for Conv2dBackwardTiming {
    fn default() -> Self {
        Self {
            pack_grad_output: Duration::ZERO,
            pack_weight: Duration::ZERO,
            padded_input: Duration::ZERO,
            grad_weight: Duration::ZERO,
            grad_input: Duration::ZERO,
            unpack_weight: Duration::ZERO,
        }
    }
}

impl Conv2dBackwardTiming {
    pub(crate) fn sections(self) -> Vec<(&'static str, Duration)> {
        vec![
            ("pack grad_output", self.pack_grad_output),
            ("pack weight", self.pack_weight),
            ("padded input", self.padded_input),
            ("grad_weight", self.grad_weight),
            ("grad_input", self.grad_input),
            ("unpack grad_weight", self.unpack_weight),
        ]
    }
}

#[derive(Clone, Copy)]
struct Conv2dBackwardGeometry {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_top: usize,
    pad_left: usize,
}

#[derive(Clone, Copy)]
enum Conv2dBackwardSection {
    PackGradOutput,
    PackWeight,
    PaddedInput,
    GradWeight,
    GradInput,
    UnpackWeight,
}

struct Conv2dBackwardClock<'a> {
    timing: Option<&'a mut Conv2dBackwardTiming>,
}

impl<'a> Conv2dBackwardClock<'a> {
    fn new(timing: Option<&'a mut Conv2dBackwardTiming>) -> Self {
        Self { timing }
    }

    fn start(&self) -> Option<Instant> {
        self.timing.as_ref().map(|_| Instant::now())
    }

    fn finish(&mut self, start: Option<Instant>, section: Conv2dBackwardSection) {
        let Some(start) = start else {
            return;
        };
        let elapsed = start.elapsed();
        let timing = self
            .timing
            .as_deref_mut()
            .expect("convolution backward timing collector disappeared");
        match section {
            Conv2dBackwardSection::PackGradOutput => timing.pack_grad_output = elapsed,
            Conv2dBackwardSection::PackWeight => timing.pack_weight = elapsed,
            Conv2dBackwardSection::PaddedInput => timing.padded_input = elapsed,
            Conv2dBackwardSection::GradWeight => timing.grad_weight = elapsed,
            Conv2dBackwardSection::GradInput => timing.grad_input = elapsed,
            Conv2dBackwardSection::UnpackWeight => timing.unpack_weight = elapsed,
        }
    }
}

/// Output channels accumulated per pass by the register-blocked convolution
/// loops.
///
/// Eight `f64` occupy four 128-bit NEON registers, which is the accumulator
/// width the compiler already selects for these kernels on this target, and
/// leave the remaining registers free for the tap operands. A fixed-size local
/// array of this width is what lets scalar replacement keep the partial sums in
/// registers across the whole tap loop; a `Vec` accumulator of dynamic length
/// cannot be promoted and is reloaded once per tap instead.
const OUT_CHANNEL_TILE: usize = 8;

/// Compute a grouped-one batched 2D cross-correlation directly over storage.
///
/// `input` is `[batch, in_channels, height, width]`, `weight` is
/// `[in_channels, out_channels, kernel_height, kernel_width]`, and padding is
/// represented as `((top, bottom), (left, right))`.
pub(crate) fn conv2d<T: Numeric>(
    input: &TensorStorage<T>,
    weight: &TensorStorage<T>,
    stride: (usize, usize),
    padding: ((usize, usize), (usize, usize)),
    dilation: (usize, usize),
) -> TensorStorage<T> {
    let (batch, in_channels, height, width) = input_shape(input, "input");
    let (weight_in, out_channels, kernel_h, kernel_w) = weight_shape(weight);
    let (stride_h, stride_w) = stride;
    let (dilation_h, dilation_w) = dilation;

    if weight_in != in_channels {
        panic!(
            "conv2d: weight in_channel {weight_in} does not match input in_channels {in_channels}."
        );
    }
    if stride_h == 0 || stride_w == 0 {
        panic!("Conv2d stride components must be nonzero.");
    }
    if dilation_h == 0 || dilation_w == 0 {
        panic!("Conv2d dilation components must be nonzero.");
    }

    let ((pad_top, pad_bottom), (pad_left, pad_right)) = padding;
    let padded_h = height + pad_top + pad_bottom;
    let padded_w = width + pad_left + pad_right;
    let effective_h = (kernel_h - 1) * dilation_h + 1;
    let effective_w = (kernel_w - 1) * dilation_w + 1;
    if padded_h < effective_h || padded_w < effective_w {
        panic!(
            "conv2d effective kernel {effective_h}x{effective_w} does not fit padded input {padded_h}x{padded_w}."
        );
    }

    let out_h = (padded_h - effective_h) / stride_h + 1;
    let out_w = (padded_w - effective_w) / stride_w + 1;
    let pads = [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)];
    let padded_input = if pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0 {
        TensorStorage::copy_s(input)
    } else {
        TensorStorage::pad(input, &pads)
    };
    // Pack weights as [in, kh, kw, out] so the inner output-channel loop is
    // contiguous and each input value can be reused across all output channels.
    let packed_weight = pack_weight(weight, in_channels, out_channels, kernel_h, kernel_w);
    let mut output = vec![T::ZERO; batch * out_channels * out_h * out_w];

    let input_strides = &padded_input.strides;
    let input_offset = padded_input.offset;
    let tiled_channels = out_channels - out_channels % OUT_CHANNEL_TILE;

    for b in 0..batch {
        let input_batch_base = input_offset + b * input_strides[0];
        for oh in 0..out_h {
            for ow in 0..out_w {
                // One register tile of output channels at a time. Holding the
                // tile across the whole tap loop is what keeps the partial sums
                // in registers; a full-width accumulator is reloaded and
                // rewritten once per tap instead.
                for oc_base in (0..tiled_channels).step_by(OUT_CHANNEL_TILE) {
                    let mut tile = [T::ZERO; OUT_CHANNEL_TILE];

                    for ic in 0..in_channels {
                        let input_channel_base = input_batch_base + ic * input_strides[1];
                        for kh in 0..kernel_h {
                            let input_row = input_channel_base
                                + (oh * stride_h + kh * dilation_h) * input_strides[2];
                            for kw in 0..kernel_w {
                                let input_value = padded_input.buffer[input_row
                                    + (ow * stride_w + kw * dilation_w) * input_strides[3]];
                                let packed_base =
                                    ((ic * kernel_h + kh) * kernel_w + kw) * out_channels + oc_base;
                                let weight_slice =
                                    &packed_weight[packed_base..packed_base + OUT_CHANNEL_TILE];
                                for u in 0..OUT_CHANNEL_TILE {
                                    tile[u] += input_value * weight_slice[u];
                                }
                            }
                        }
                    }

                    for (u, &value) in tile.iter().enumerate() {
                        output
                            [(b * out_channels + oc_base + u) * out_h * out_w + oh * out_w + ow] =
                            value;
                    }
                }

                // Output channels left over when `out_channels` is not a
                // multiple of the tile width.
                for oc in tiled_channels..out_channels {
                    let mut sum = T::ZERO;
                    for ic in 0..in_channels {
                        let input_channel_base = input_batch_base + ic * input_strides[1];
                        for kh in 0..kernel_h {
                            let input_row = input_channel_base
                                + (oh * stride_h + kh * dilation_h) * input_strides[2];
                            for kw in 0..kernel_w {
                                let input_value = padded_input.buffer[input_row
                                    + (ow * stride_w + kw * dilation_w) * input_strides[3]];
                                let packed_base =
                                    ((ic * kernel_h + kh) * kernel_w + kw) * out_channels + oc;
                                sum += input_value * packed_weight[packed_base];
                            }
                        }
                    }
                    output[(b * out_channels + oc) * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }

    TensorStorage::from_buffer(vec![batch, out_channels, out_h, out_w], output)
}

/// Compute first-order input and weight gradients for [`conv2d`].
pub(crate) fn conv2d_backward<T: Numeric>(
    grad_output: &TensorStorage<T>,
    input: &TensorStorage<T>,
    weight: &TensorStorage<T>,
    stride: (usize, usize),
    padding: ((usize, usize), (usize, usize)),
    dilation: (usize, usize),
) -> Conv2dBackward<T> {
    conv2d_backward_impl(grad_output, input, weight, stride, padding, dilation, None)
}

/// Compute convolution gradients and collect per-section timings.
pub(crate) fn conv2d_backward_profiled<T: Numeric>(
    grad_output: &TensorStorage<T>,
    input: &TensorStorage<T>,
    weight: &TensorStorage<T>,
    stride: (usize, usize),
    padding: ((usize, usize), (usize, usize)),
    dilation: (usize, usize),
) -> (Conv2dBackward<T>, Conv2dBackwardTiming) {
    let mut timing = Conv2dBackwardTiming::default();
    let backward = conv2d_backward_impl(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        Some(&mut timing),
    );
    (backward, timing)
}

fn conv2d_backward_impl<T: Numeric>(
    grad_output: &TensorStorage<T>,
    input: &TensorStorage<T>,
    weight: &TensorStorage<T>,
    stride: (usize, usize),
    padding: ((usize, usize), (usize, usize)),
    dilation: (usize, usize),
    timing: Option<&mut Conv2dBackwardTiming>,
) -> Conv2dBackward<T> {
    let (batch, in_channels, height, width) = input_shape(input, "input");
    let (weight_in, out_channels, kernel_h, kernel_w) = weight_shape(weight);
    let (stride_h, stride_w) = stride;
    let (dilation_h, dilation_w) = dilation;

    if weight_in != in_channels {
        panic!(
            "conv2d: weight in_channel {weight_in} does not match input in_channels {in_channels}."
        );
    }
    if stride_h == 0 || stride_w == 0 {
        panic!("Conv2d stride components must be nonzero.");
    }
    if dilation_h == 0 || dilation_w == 0 {
        panic!("Conv2d dilation components must be nonzero.");
    }

    let ((pad_top, pad_bottom), (pad_left, pad_right)) = padding;
    let padded_h = height + pad_top + pad_bottom;
    let padded_w = width + pad_left + pad_right;
    let effective_h = (kernel_h - 1) * dilation_h + 1;
    let effective_w = (kernel_w - 1) * dilation_w + 1;
    if padded_h < effective_h || padded_w < effective_w {
        panic!(
            "conv2d effective kernel {effective_h}x{effective_w} does not fit padded input {padded_h}x{padded_w}."
        );
    }

    let out_h = (padded_h - effective_h) / stride_h + 1;
    let out_w = (padded_w - effective_w) / stride_w + 1;
    let expected_output_shape = [batch, out_channels, out_h, out_w];
    if grad_output.shape.as_slice() != expected_output_shape {
        panic!(
            "conv2d backward: grad_output has shape {:?}, expected {:?}.",
            grad_output.shape, expected_output_shape
        );
    }

    let geometry = Conv2dBackwardGeometry {
        batch,
        in_channels,
        out_channels,
        height,
        width,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        pad_top,
        pad_left,
    };
    let mut clock = Conv2dBackwardClock::new(timing);
    let start = clock.start();
    let packed_grad_output = pack_grad_output(grad_output, batch, out_channels, out_h, out_w);
    clock.finish(start, Conv2dBackwardSection::PackGradOutput);

    let start = clock.start();
    let packed_weight = pack_weight(weight, in_channels, out_channels, kernel_h, kernel_w);
    clock.finish(start, Conv2dBackwardSection::PackWeight);

    let pads = [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)];
    let start = clock.start();
    let padded_input = if pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0 {
        TensorStorage::copy_s(input)
    } else {
        TensorStorage::pad(input, &pads)
    };
    clock.finish(start, Conv2dBackwardSection::PaddedInput);

    // Weight gradient: reuse each input value across the contiguous output-channel
    // dimension of the packed tensors.
    let start = clock.start();
    let packed_grad_weight = if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1
    {
        conv2d_grad_weight_stride1_dilation1(&padded_input, &packed_grad_output, geometry)
    } else {
        let mut packed_grad_weight =
            vec![T::ZERO; in_channels * kernel_h * kernel_w * out_channels];
        let padded_input_strides = &padded_input.strides;
        let padded_input_offset = padded_input.offset;
        for b in 0..batch {
            let input_batch_base = padded_input_offset + b * padded_input_strides[0];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let grad_output_base = ((b * out_h + oh) * out_w + ow) * out_channels;
                    for ic in 0..in_channels {
                        let input_channel_base = input_batch_base + ic * padded_input_strides[1];
                        for kh in 0..kernel_h {
                            let input_row = input_channel_base
                                + (oh * stride_h + kh * dilation_h) * padded_input_strides[2];
                            for kw in 0..kernel_w {
                                let input_value = padded_input.buffer[input_row
                                    + (ow * stride_w + kw * dilation_w) * padded_input_strides[3]];
                                let packed_base =
                                    ((ic * kernel_h + kh) * kernel_w + kw) * out_channels;
                                let grad_weight_slice = &mut packed_grad_weight
                                    [packed_base..packed_base + out_channels];
                                let grad_output_slice = &packed_grad_output
                                    [grad_output_base..grad_output_base + out_channels];
                                for (weight_grad, &grad_output_value) in
                                    grad_weight_slice.iter_mut().zip(grad_output_slice.iter())
                                {
                                    *weight_grad += input_value * grad_output_value;
                                }
                            }
                        }
                    }
                }
            }
        }
        packed_grad_weight
    };
    clock.finish(start, Conv2dBackwardSection::GradWeight);

    // Input gradient: enumerate the output positions that read each input value,
    // then reduce the contiguous output-channel vectors into one input gradient.
    let start = clock.start();
    let grad_input = if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        conv2d_grad_input_stride1_dilation1(
            &packed_grad_output,
            &packed_weight,
            input.numel,
            geometry,
        )
    } else {
        let mut grad_input = vec![T::ZERO; input.numel];
        let mut input_accumulator = vec![T::ZERO; out_channels];
        for b in 0..batch {
            for ic in 0..in_channels {
                for ih in 0..height {
                    let padded_ih = ih + pad_top;
                    for iw in 0..width {
                        let padded_iw = iw + pad_left;
                        input_accumulator.fill(T::ZERO);

                        for kh in 0..kernel_h {
                            let height_numerator =
                                padded_ih as isize - kh as isize * dilation_h as isize;
                            if height_numerator < 0 || height_numerator % stride_h as isize != 0 {
                                continue;
                            }
                            let oh = height_numerator as usize / stride_h;
                            if oh >= out_h {
                                continue;
                            }

                            for kw in 0..kernel_w {
                                let width_numerator =
                                    padded_iw as isize - kw as isize * dilation_w as isize;
                                if width_numerator < 0 || width_numerator % stride_w as isize != 0 {
                                    continue;
                                }
                                let ow = width_numerator as usize / stride_w;
                                if ow >= out_w {
                                    continue;
                                }

                                let grad_output_base =
                                    ((b * out_h + oh) * out_w + ow) * out_channels;
                                let packed_base =
                                    ((ic * kernel_h + kh) * kernel_w + kw) * out_channels;
                                let grad_output_slice = &packed_grad_output
                                    [grad_output_base..grad_output_base + out_channels];
                                let weight_slice =
                                    &packed_weight[packed_base..packed_base + out_channels];
                                for (accumulator, (&grad_output_value, &weight_value)) in
                                    input_accumulator
                                        .iter_mut()
                                        .zip(grad_output_slice.iter().zip(weight_slice.iter()))
                                {
                                    *accumulator += grad_output_value * weight_value;
                                }
                            }
                        }

                        let input_index = ((b * in_channels + ic) * height + ih) * width + iw;
                        grad_input[input_index] = input_accumulator
                            .iter()
                            .fold(T::ZERO, |sum, &value| sum + value);
                    }
                }
            }
        }
        grad_input
    };
    clock.finish(start, Conv2dBackwardSection::GradInput);

    // Gradients must use the original parameter layout expected by GradMap and
    // the optimizer; the packed layout is only an internal kernel detail.
    let start = clock.start();
    let mut grad_weight = vec![T::ZERO; weight.numel];
    for ic in 0..in_channels {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let packed_base = ((ic * kernel_h + kh) * kernel_w + kw) * out_channels;
                for oc in 0..out_channels {
                    let original_index = ((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw;
                    grad_weight[original_index] = packed_grad_weight[packed_base + oc];
                }
            }
        }
    }
    clock.finish(start, Conv2dBackwardSection::UnpackWeight);

    Conv2dBackward {
        grad_input: TensorStorage::from_buffer(input.shape.clone(), grad_input),
        grad_weight: TensorStorage::from_buffer(weight.shape.clone(), grad_weight),
    }
}

fn conv2d_grad_weight_stride1_dilation1<T: Numeric>(
    padded_input: &TensorStorage<T>,
    packed_grad_output: &[T],
    geometry: Conv2dBackwardGeometry,
) -> Vec<T> {
    let mut packed_grad_weight =
        vec![
            T::ZERO;
            geometry.in_channels * geometry.kernel_h * geometry.kernel_w * geometry.out_channels
        ];
    let padded_input_strides = &padded_input.strides;
    let padded_input_offset = padded_input.offset;

    for b in 0..geometry.batch {
        let input_batch_base = padded_input_offset + b * padded_input_strides[0];
        for oh in 0..geometry.out_h {
            for ow in 0..geometry.out_w {
                let grad_output_base =
                    ((b * geometry.out_h + oh) * geometry.out_w + ow) * geometry.out_channels;
                for ic in 0..geometry.in_channels {
                    let input_channel_base = input_batch_base + ic * padded_input_strides[1];
                    for kh in 0..geometry.kernel_h {
                        let input_row = input_channel_base + (oh + kh) * padded_input_strides[2];
                        for kw in 0..geometry.kernel_w {
                            let input_value = padded_input.buffer
                                [input_row + (ow + kw) * padded_input_strides[3]];
                            let packed_base = ((ic * geometry.kernel_h + kh) * geometry.kernel_w
                                + kw)
                                * geometry.out_channels;
                            let grad_weight_slice = &mut packed_grad_weight
                                [packed_base..packed_base + geometry.out_channels];
                            let grad_output_slice = &packed_grad_output
                                [grad_output_base..grad_output_base + geometry.out_channels];
                            for (weight_grad, &grad_output_value) in
                                grad_weight_slice.iter_mut().zip(grad_output_slice.iter())
                            {
                                *weight_grad += input_value * grad_output_value;
                            }
                        }
                    }
                }
            }
        }
    }

    packed_grad_weight
}

fn conv2d_grad_input_stride1_dilation1<T: Numeric>(
    packed_grad_output: &[T],
    packed_weight: &[T],
    input_numel: usize,
    geometry: Conv2dBackwardGeometry,
) -> Vec<T> {
    let mut grad_input = vec![T::ZERO; input_numel];
    let tiled_channels = geometry.out_channels - geometry.out_channels % OUT_CHANNEL_TILE;
    // Accumulate one register tile of output channels, so the partial sums stay
    // in vector registers across the tap loop rather than being reloaded from a
    // full-width accumulator once per tap.
    let accumulate_tap = |tile: &mut [T],
                          b: usize,
                          oh: usize,
                          ow: usize,
                          ic: usize,
                          kh: usize,
                          kw: usize,
                          oc_base: usize| {
        let grad_output_base =
            ((b * geometry.out_h + oh) * geometry.out_w + ow) * geometry.out_channels + oc_base;
        let packed_base = ((ic * geometry.kernel_h + kh) * geometry.kernel_w + kw)
            * geometry.out_channels
            + oc_base;
        let grad_output_slice =
            &packed_grad_output[grad_output_base..grad_output_base + tile.len()];
        let weight_slice = &packed_weight[packed_base..packed_base + tile.len()];
        for (tile_value, (&grad_output_value, &weight_value)) in tile
            .iter_mut()
            .zip(grad_output_slice.iter().zip(weight_slice.iter()))
        {
            *tile_value += grad_output_value * weight_value;
        }
    };

    for b in 0..geometry.batch {
        for ic in 0..geometry.in_channels {
            for ih in 0..geometry.height {
                let padded_ih = ih + geometry.pad_top;
                let row_interior =
                    padded_ih >= geometry.kernel_h.saturating_sub(1) && padded_ih < geometry.out_h;
                let input_row_base =
                    ((b * geometry.in_channels + ic) * geometry.height + ih) * geometry.width;
                for iw in 0..geometry.width {
                    let padded_iw = iw + geometry.pad_left;
                    let column_interior = padded_iw >= geometry.kernel_w.saturating_sub(1)
                        && padded_iw < geometry.out_w;
                    let mut total = T::ZERO;

                    for oc_base in (0..tiled_channels).step_by(OUT_CHANNEL_TILE) {
                        let mut tile = [T::ZERO; OUT_CHANNEL_TILE];
                        // Interior positions avoid per-tap coordinate bounds checks.
                        if row_interior && column_interior {
                            for kh in 0..geometry.kernel_h {
                                let oh = padded_ih - kh;
                                for kw in 0..geometry.kernel_w {
                                    let ow = padded_iw - kw;
                                    accumulate_tap(&mut tile, b, oh, ow, ic, kh, kw, oc_base);
                                }
                            }
                        } else {
                            for kh in 0..geometry.kernel_h {
                                if padded_ih < kh {
                                    continue;
                                }
                                let oh = padded_ih - kh;
                                if oh >= geometry.out_h {
                                    continue;
                                }
                                for kw in 0..geometry.kernel_w {
                                    if padded_iw < kw {
                                        continue;
                                    }
                                    let ow = padded_iw - kw;
                                    if ow >= geometry.out_w {
                                        continue;
                                    }
                                    accumulate_tap(&mut tile, b, oh, ow, ic, kh, kw, oc_base);
                                }
                            }
                        }
                        for &value in tile.iter() {
                            total += value;
                        }
                    }

                    // Output channels left over when `out_channels` is not a
                    // multiple of the tile width.
                    for oc in tiled_channels..geometry.out_channels {
                        let mut sum = T::ZERO;
                        for kh in 0..geometry.kernel_h {
                            if padded_ih < kh {
                                continue;
                            }
                            let oh = padded_ih - kh;
                            if oh >= geometry.out_h {
                                continue;
                            }
                            for kw in 0..geometry.kernel_w {
                                if padded_iw < kw {
                                    continue;
                                }
                                let ow = padded_iw - kw;
                                if ow >= geometry.out_w {
                                    continue;
                                }
                                let grad_output_base = ((b * geometry.out_h + oh) * geometry.out_w
                                    + ow)
                                    * geometry.out_channels
                                    + oc;
                                let packed_base =
                                    ((ic * geometry.kernel_h + kh) * geometry.kernel_w + kw)
                                        * geometry.out_channels
                                        + oc;
                                sum += packed_grad_output[grad_output_base]
                                    * packed_weight[packed_base];
                            }
                        }
                        total += sum;
                    }

                    grad_input[input_row_base + iw] = total;
                }
            }
        }
    }

    grad_input
}

fn pack_grad_output<T: Numeric>(
    grad_output: &TensorStorage<T>,
    batch: usize,
    out_channels: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<T> {
    let mut packed = Vec::with_capacity(batch * out_h * out_w * out_channels);
    let strides = &grad_output.strides;
    let offset = grad_output.offset;
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for oc in 0..out_channels {
                    let index = offset
                        + b * strides[0]
                        + oc * strides[1]
                        + oh * strides[2]
                        + ow * strides[3];
                    packed.push(grad_output.buffer[index]);
                }
            }
        }
    }
    packed
}

fn pack_weight<T: Numeric>(
    weight: &TensorStorage<T>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Vec<T> {
    let mut packed = Vec::with_capacity(in_channels * kernel_h * kernel_w * out_channels);
    let strides = &weight.strides;
    let offset = weight.offset;
    for ic in 0..in_channels {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let weight_base = offset + ic * strides[0] + kh * strides[2] + kw * strides[3];
                for oc in 0..out_channels {
                    packed.push(weight.buffer[weight_base + oc * strides[1]]);
                }
            }
        }
    }
    packed
}

fn input_shape<T: Numeric>(storage: &TensorStorage<T>, name: &str) -> (usize, usize, usize, usize) {
    if storage.shape.len() != 4 {
        panic!(
            "conv2d {name} must be rank 4, got shape {:?}.",
            storage.shape
        );
    }
    (
        storage.shape[0],
        storage.shape[1],
        storage.shape[2],
        storage.shape[3],
    )
}

fn weight_shape<T: Numeric>(storage: &TensorStorage<T>) -> (usize, usize, usize, usize) {
    if storage.shape.len() != 4 {
        panic!(
            "conv2d weight must be rank 4, got shape {:?}.",
            storage.shape
        );
    }
    (
        storage.shape[0],
        storage.shape[1],
        storage.shape[2],
        storage.shape[3],
    )
}
