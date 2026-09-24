use crate::core::GraphTensor;
use crate::core::nn::module::{Forward1, Module};
use crate::core::tensor::AbstractTensor;

/// A differentiable `Flatten` module.
///
/// `start_dim` and `end_dim` use PyTorch's signed-dimension convention. The
/// default is `start_dim=1, end_dim=-1`, which turns `[B, C, H, W]` into
/// `[B, C * H * W]`.
pub struct Flatten {
    pub start_dim: isize,
    pub end_dim: isize,
}

impl Flatten {
    /// Create a default PyTorch-style flatten operation.
    pub fn new() -> Self {
        Self {
            start_dim: 1,
            end_dim: -1,
        }
    }

    /// Create a flatten operation with signed dimension bounds.
    pub fn with_dims(start_dim: isize, end_dim: isize) -> Self {
        Self { start_dim, end_dim }
    }

    /// Alias for [`Self::with_dims`].
    pub fn new_with_dims(start_dim: isize, end_dim: isize) -> Self {
        Self::with_dims(start_dim, end_dim)
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {}

impl Forward1 for Flatten {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let ndim = input.shape().len();
        if ndim == 0 {
            panic!("Flatten expects a tensor with at least one dimension.");
        }

        let start = normalize_dim(self.start_dim, ndim, "start_dim");
        let end = normalize_dim(self.end_dim, ndim, "end_dim");
        if start > end {
            panic!(
                "Flatten start_dim {start} must not exceed end_dim {end} for shape {:?}.",
                input.shape()
            );
        }

        let flattened_numel = input.shape()[start..=end].iter().product::<usize>();
        let mut shape = input.shape()[..start].to_vec();
        shape.push(flattened_numel);
        shape.extend_from_slice(&input.shape()[end + 1..]);

        input.reshape(&shape)
    }
}

fn normalize_dim(dim: isize, ndim: usize, name: &str) -> usize {
    let normalized = if dim < 0 { ndim as isize + dim } else { dim };
    if normalized < 0 || normalized >= ndim as isize {
        panic!("Flatten {name} {dim} is out of range for a tensor with {ndim} dimensions.");
    }
    normalized as usize
}
