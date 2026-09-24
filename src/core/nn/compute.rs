use rand::rngs::StdRng;

use crate::core::GraphTensor;
use crate::core::dtype::Numeric;
use crate::core::nn::module::{Forward1, Module};
use crate::core::tensor::AbstractTensor;
use crate::module;

/// Batched 2D cross-correlation, kernel square-ish (any `kh x kw`).
///
/// `input` is `[batch, in_channel, height, width]`; `weight` is
/// `[in_channel, out_channel, kernel_h, kernel_w]`. The result is
/// `[batch, out_channel, height - kernel_h + 1, width - kernel_w + 1]`.
///
/// The kernel is decomposed into `kernel_h * kernel_w` taps; each tap
/// correlates one spatial window of `input` with the corresponding weight
/// slice via a single 2D `matmul` over the channel axis (an im2col-style
/// flatten). Gradients flow through every tap independently: each slice's
/// backward zero-pads into the full input/weight shapes, and the autograd
/// engine accumulates the per-tap contributions. No explicit conv gradient
/// math is needed.
pub fn conv2d<T: Numeric>(input: &GraphTensor<T>, weight: &GraphTensor<T>) -> GraphTensor<T> {
    if input.shape().len() != 4 {
        panic!(
            "conv2d expects a [batch, in_channel, height, width] input, got shape {:?}.",
            input.shape()
        );
    }
    if weight.shape().len() != 4 {
        panic!(
            "conv2d expects a [in_channel, out_channel, kernel_h, kernel_w] weight, got shape {:?}.",
            weight.shape()
        );
    }

    let batch = input.shape()[0];
    let in_ch = input.shape()[1];
    let height = input.shape()[2];
    let width = input.shape()[3];

    let out_ch = weight.shape()[1];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];

    if weight.shape()[0] != in_ch {
        panic!(
            "conv2d: weight in_channel {} must match input in_channel {}.",
            weight.shape()[0],
            in_ch
        );
    }

    let out_h = height - kh + 1;
    let out_w = width - kw + 1;

    let mut acc: Option<GraphTensor<T>> = None;
    for i in 0..kh {
        for j in 0..kw {
            // Window [B, in_ch, out_h, out_w] for this tap.
            let window = input.slice(&[(0, batch), (0, in_ch), (i, out_h), (j, out_w)]);

            // Weight slice [in_ch, out_ch] for this tap: [in_ch, out_ch, 1, 1]
            // with the singleton kernel axes squeezed away.
            let kernel = weight
                .slice(&[(0, in_ch), (0, out_ch), (i, 1), (j, 1)])
                .squeeze(2)
                .squeeze(2);

            // [B, in_ch, out_h, out_w] -> [B, out_h, out_w, in_ch] (two
            // transposes build any 4D permutation), flattened to the 2D
            // [B*out_h*out_w, in_ch] matrix the matmul kernel needs.
            let flat = window
                .transpose(1, 2)
                .transpose(2, 3)
                .reshape(&[batch * out_h * out_w, in_ch]);

            let result = GraphTensor::matmul(&flat, &kernel); // [B*out_h*out_w, out_ch]

            // Back to [B, out_h, out_w, out_ch], then [B, out_ch, out_h, out_w].
            let tap = result
                .reshape(&[batch, out_h, out_w, out_ch])
                .transpose(2, 3)
                .transpose(1, 2);

            acc = match acc {
                None => Some(tap),
                Some(prev) => Some(&prev + &tap),
            };
        }
    }

    acc.expect("conv2d over a non-empty kernel always yields a result")
}

module! {
    Linear {
        params {
            weight,
        },
        optional_params {
            bias,
        }
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool, rng: StdRng) -> Self {
        // Xavier/Glorot uniform initialization to break symmetry between units
        let w_shape = vec![in_features, out_features];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng);

        let bias = if has_bias {
            Some(GraphTensor::new(vec![out_features], 0.0, true))
        } else {
            None
        };

        Self { weight, bias }
    }
}

impl Forward1 for Linear {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let mult = GraphTensor::matmul(input, &self.weight);

        // `b` has shape [out_features], `mult` [batch, out_features]: the `+`
        // broadcasts the bias across the batch dim automatically.
        match &self.bias {
            None => mult,
            Some(b) => &mult + b,
        }
    }
}

module! {
    Conv2d {
        params {
            weight,
        },
        optional_params {
            bias,
        }
    }
}

impl Conv2d {
    /// A square-kernel 2D convolution mapping `in_channels` input maps to
    /// `out_channels` output maps, with optional per-output-channel bias.
    ///
    /// The weight is a `[in_channels, out_channels, kernel, kernel]` tensor
    /// (first axis = the channel being summed over, second = the output
    /// channel), so it plugs straight into [`conv2d`].
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        has_bias: bool,
        rng: StdRng,
    ) -> Self {
        let w_shape = vec![in_channels, out_channels, kernel_size, kernel_size];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng);

        let bias = if has_bias {
            Some(GraphTensor::new(vec![out_channels], 0.0, true))
        } else {
            None
        };

        Self { weight, bias }
    }
}

impl Forward1 for Conv2d {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let conv = conv2d(input, &self.weight); // [B, out_ch, out_h, out_w]

        // `b` is [out_ch]; the result is [B, out_ch, out_h, out_w], so it must
        // be shaped to [1, out_ch, 1, 1] before broadcasting — a bare [out_ch]
        // would right-align onto the width axis instead.
        match &self.bias {
            None => conv,
            Some(b) => {
                let b_4d = b.unsqueeze(0).unsqueeze(2).unsqueeze(2);
                &conv + &b_4d
            }
        }
    }
}

/// A parameterless 2D average-pooling layer over `[batch, channel, height,
/// width]` inputs. The window is `kernel x kernel` (square); `stride` defaults
/// to the kernel size (no overlap) when not given. See
/// [`crate::core::GraphTensor::avg_pool2d`] for output-size semantics.
pub struct AvgPool2d {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel: (kernel_size, kernel_size),
            stride: (stride, stride),
        }
    }
}

impl Module for AvgPool2d {}

impl Forward1 for AvgPool2d {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        input.avg_pool2d(self.kernel, self.stride)
    }
}

/// A parameterless 2D max-pooling layer over `[batch, channel, height, width]`
/// inputs. The window is `kernel x kernel`; `stride` defaults to the kernel
/// size (no overlap) when not given. See
/// [`crate::core::GraphTensor::max_pool2d`] for output-size semantics.
pub struct MaxPool2d {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel: (kernel_size, kernel_size),
            stride: (stride, stride),
        }
    }
}

impl Module for MaxPool2d {}

impl Forward1 for MaxPool2d {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        input.max_pool2d(self.kernel, self.stride)
    }
}
