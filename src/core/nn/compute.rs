use rand::rngs::StdRng;

use crate::core::GraphTensor;
use crate::core::dtype::Numeric;
use crate::core::nn::module::{Forward1, Module};
use crate::core::tensor::AbstractTensor;
use crate::module;

/// A spatial pair used by the PyTorch-style convolution options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size2 {
    pub height: usize,
    pub width: usize,
}

impl Size2 {
    pub const ONE: Self = Self {
        height: 1,
        width: 1,
    };
}

impl From<usize> for Size2 {
    fn from(value: usize) -> Self {
        Self {
            height: value,
            width: value,
        }
    }
}

impl From<(usize, usize)> for Size2 {
    fn from((height, width): (usize, usize)) -> Self {
        Self { height, width }
    }
}

/// Zero-padding configuration for [`Conv2d`].
///
/// Non-zero padding modes are intentionally not represented yet; this type is
/// the numeric subset of PyTorch's padding surface needed by the first CNN
/// implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dPadding {
    Valid,
    Same,
    Symmetric((usize, usize)),
}

impl From<usize> for Conv2dPadding {
    fn from(value: usize) -> Self {
        Self::Symmetric((value, value))
    }
}

impl From<(usize, usize)> for Conv2dPadding {
    fn from((height, width): (usize, usize)) -> Self {
        Self::Symmetric((height, width))
    }
}

impl From<&str> for Conv2dPadding {
    fn from(value: &str) -> Self {
        match value {
            "valid" => Self::Valid,
            "same" => Self::Same,
            other => {
                panic!("unsupported Conv2d padding value {other:?}; expected \"same\" or \"valid\"")
            }
        }
    }
}

/// Configuration for a [`Conv2d`] layer.
#[derive(Clone, Copy, Debug)]
pub struct Conv2dConfig {
    pub kernel_size: Size2,
    pub stride: Size2,
    pub padding: Conv2dPadding,
    pub dilation: Size2,
    pub bias: bool,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            kernel_size: Size2 {
                height: 3,
                width: 3,
            },
            stride: Size2::ONE,
            padding: Conv2dPadding::Valid,
            dilation: Size2::ONE,
            bias: true,
        }
    }
}

fn resolve_padding(
    input: Size2,
    kernel: Size2,
    stride: Size2,
    dilation: Size2,
    padding: Conv2dPadding,
) -> ((usize, usize), (usize, usize)) {
    if kernel.height == 0
        || kernel.width == 0
        || stride.height == 0
        || stride.width == 0
        || dilation.height == 0
        || dilation.width == 0
    {
        panic!("Conv2d kernel, stride, and dilation components must be nonzero.");
    }

    let resolve_one = |input: usize, kernel: usize, stride: usize, dilation: usize| {
        let effective_kernel = (kernel - 1) * dilation + 1;
        let output = input.div_ceil(stride);
        let total = ((output - 1) * stride + effective_kernel).saturating_sub(input);
        (total / 2, total - total / 2)
    };

    match padding {
        Conv2dPadding::Valid => ((0, 0), (0, 0)),
        Conv2dPadding::Symmetric((height, width)) => ((height, height), (width, width)),
        Conv2dPadding::Same => (
            resolve_one(input.height, kernel.height, stride.height, dilation.height),
            resolve_one(input.width, kernel.width, stride.width, dilation.width),
        ),
    }
}

/// Batched 2D cross-correlation with PyTorch-style stride, dilation, and
/// zero-padding options.
///
/// `input` is `[batch, in_channel, height, width]`; `weight` is
/// `[in_channel, out_channel, kernel_h, kernel_w]`. Groups are intentionally
/// fixed at one. The kernel is decomposed into `kernel_h * kernel_w` taps;
/// each tap correlates a strided spatial window with the corresponding weight
/// slice via a 2D `matmul` over the channel axis. Gradients flow through the
/// padding, strided slices, and matrix multiplication nodes.
pub fn conv2d<T: Numeric>(input: &GraphTensor<T>, weight: &GraphTensor<T>) -> GraphTensor<T> {
    conv2d_with_options(input, weight, Size2::ONE, Conv2dPadding::Valid, Size2::ONE)
}

/// Batched 2D cross-correlation with configurable stride, padding, and
/// dilation. Only zero padding is currently supported by [`Conv2dPadding`].
pub fn conv2d_with_options<T: Numeric>(
    input: &GraphTensor<T>,
    weight: &GraphTensor<T>,
    stride: Size2,
    padding: Conv2dPadding,
    dilation: Size2,
) -> GraphTensor<T> {
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
    let kernel = Size2 {
        height: weight.shape()[2],
        width: weight.shape()[3],
    };
    let input_size = Size2 { height, width };

    if weight.shape()[0] != in_ch {
        panic!(
            "conv2d: weight in_channel {} must match input in_channel {}.",
            weight.shape()[0],
            in_ch
        );
    }

    let ((pad_top, pad_bottom), (pad_left, pad_right)) =
        resolve_padding(input_size, kernel, stride, dilation, padding);
    let padded_height = height + pad_top + pad_bottom;
    let padded_width = width + pad_left + pad_right;
    let effective_height = (kernel.height - 1) * dilation.height + 1;
    let effective_width = (kernel.width - 1) * dilation.width + 1;
    if padded_height < effective_height || padded_width < effective_width {
        panic!(
            "conv2d effective kernel {effective_height}x{effective_width} does not fit padded input {padded_height}x{padded_width}."
        );
    }

    let out_h = (padded_height - effective_height) / stride.height + 1;
    let out_w = (padded_width - effective_width) / stride.width + 1;
    if out_h == 0 || out_w == 0 {
        panic!("conv2d produced an empty spatial output.");
    }

    let pads = [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)];
    let padded_input = if pads
        .iter()
        .all(|(before, after)| *before == 0 && *after == 0)
    {
        input.copy_s()
    } else {
        input.pad(&pads)
    };

    let mut acc: Option<GraphTensor<T>> = None;
    for i in 0..kernel.height {
        for j in 0..kernel.width {
            // Window [B, in_ch, out_h, out_w] for this tap, sampled at the
            // configured stride and dilation.
            let window = padded_input.slice_strided(&[
                (0, batch, 1),
                (0, in_ch, 1),
                (i * dilation.height, out_h, stride.height),
                (j * dilation.width, out_w, stride.width),
            ]);

            // Weight slice [in_ch, out_ch] for this tap: [in_ch, out_ch, 1, 1]
            // with the singleton kernel axes squeezed away.
            let kernel_slice = weight
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

            let result = GraphTensor::matmul(&flat, &kernel_slice);

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
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        let input = input.with_no_grad(no_grad);
        let mult = GraphTensor::matmul(&input, &self.weight);

        // `b` has shape [out_features], `mult` [batch, out_features]: the `+`
        // broadcasts the bias across the batch dim automatically.
        match &self.bias {
            None => mult,
            Some(b) => &mult + b,
        }
    }
}

/// A PyTorch-style 2D convolution with groups fixed at one.
///
/// The weight is `[in_channels, out_channels, kernel_height, kernel_width]`;
/// the first axis is the channel being summed over and the second is the output
/// channel. `padding` currently supports zero padding, `valid`, and `same`.
pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: Size2,
    pub stride: Size2,
    pub padding: Conv2dPadding,
    pub dilation: Size2,
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
}

impl Module for Conv2d {
    fn for_each_own_param(&self, f: &mut dyn FnMut(&str, &GraphTensor)) {
        f("weight", &self.weight);
        if let Some(bias) = &self.bias {
            f("bias", bias);
        }
    }

    fn for_each_own_param_mut(&mut self, f: &mut dyn FnMut(&str, &mut GraphTensor)) {
        f("weight", &mut self.weight);
        if let Some(bias) = &mut self.bias {
            f("bias", bias);
        }
    }

    fn param(&self, name: &str) -> Option<&GraphTensor> {
        match name {
            "weight" => Some(&self.weight),
            "bias" => self.bias.as_ref(),
            _ => None,
        }
    }

    fn param_mut(&mut self, name: &str) -> Option<&mut GraphTensor> {
        match name {
            "weight" => Some(&mut self.weight),
            "bias" => self.bias.as_mut(),
            _ => None,
        }
    }
}

impl Conv2d {
    /// Construct a valid, stride-one, dilation-one convolution.
    ///
    /// This retains the original compact constructor; use
    /// [`Self::new_with_options`] for PyTorch-style configuration.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        has_bias: bool,
        rng: StdRng,
    ) -> Self {
        Self::new_with_options(
            in_channels,
            out_channels,
            kernel_size,
            Size2::ONE,
            Conv2dPadding::Valid,
            Size2::ONE,
            has_bias,
            rng,
        )
    }

    /// Construct a convolution with configurable stride, zero padding, and
    /// dilation. Integer and `(height, width)` forms are accepted for each
    /// spatial option through the `Into` conversions on [`Size2`] and
    /// [`Conv2dPadding`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options<K, S, P, D>(
        in_channels: usize,
        out_channels: usize,
        kernel_size: K,
        stride: S,
        padding: P,
        dilation: D,
        has_bias: bool,
        rng: StdRng,
    ) -> Self
    where
        K: Into<Size2>,
        S: Into<Size2>,
        P: Into<Conv2dPadding>,
        D: Into<Size2>,
    {
        Self::new_with_config(
            in_channels,
            out_channels,
            Conv2dConfig {
                kernel_size: kernel_size.into(),
                stride: stride.into(),
                padding: padding.into(),
                dilation: dilation.into(),
                bias: has_bias,
            },
            rng,
        )
    }

    /// Construct a convolution from a reusable configuration object.
    pub fn new_with_config(
        in_channels: usize,
        out_channels: usize,
        config: Conv2dConfig,
        rng: StdRng,
    ) -> Self {
        let Conv2dConfig {
            kernel_size,
            stride,
            padding,
            dilation,
            bias: has_bias,
        } = config;
        assert!(
            kernel_size.height > 0 && kernel_size.width > 0,
            "Conv2d kernel size must be nonzero."
        );
        assert!(
            stride.height > 0 && stride.width > 0,
            "Conv2d stride must be nonzero."
        );
        assert!(
            dilation.height > 0 && dilation.width > 0,
            "Conv2d dilation must be nonzero."
        );

        let w_shape = vec![
            in_channels,
            out_channels,
            kernel_size.height,
            kernel_size.width,
        ];
        let weight = GraphTensor::init_xavier_uniform(w_shape, true, rng);
        let bias = if has_bias {
            Some(GraphTensor::new(vec![out_channels], 0.0, true))
        } else {
            None
        };

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            weight,
            bias,
        }
    }
}

impl Forward1 for Conv2d {
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        let input = input.with_no_grad(no_grad);
        let conv = conv2d_with_options(
            &input,
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
        ); // [B, out_ch, out_h, out_w]

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
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        input
            .with_no_grad(no_grad)
            .avg_pool2d(self.kernel, self.stride)
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
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        input
            .with_no_grad(no_grad)
            .max_pool2d(self.kernel, self.stride)
    }
}
