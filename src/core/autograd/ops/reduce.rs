use crate::core::autograd::grad_fn::*;
use crate::core::tensor::{AbstractTensor, GraphTensor};

// There is a single Sum op that reduces an arbitrary subset of dimensions; summing
// over all dimensions or a single dimension are just special cases of it. `dims` are
// indices into the *input* (pre-reduction) tensor, sorted ascending and de-duplicated.

#[derive(Debug)]
pub struct SumDimsOp {
    pub dims: Vec<usize>,
}

pub type BackwardSumDims = NBackwardOp<SumDimsOp, 1>;

impl GradRule<1> for SumDimsOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        out.push(operands[0].requires_grad().then(|| {
            // Gradient of a sum is the upstream gradient replicated over every
            // reduced dimension (via stride-0 expand views).
            let mut g = in_grad.copy_s();
            let shape = operands[0].shape();
            for &d in &self.dims {
                g = g.unsqueeze(d).expand(d, shape[d]);
            }
            g
        }));
    }
}

#[derive(Debug)]
pub struct MaxDimsOp {
    pub dims: Vec<usize>,
}

pub type BackwardMaxDims = NBackwardOp<MaxDimsOp, 1>;

impl GradRule<1> for MaxDimsOp {
    fn compute_grad(
        &self,
        operands: &[GraphTensor; 1],
        in_grad: &GraphTensor,
        _retain_graph: bool,
        out: &mut Vec<Option<GraphTensor>>,
    ) {
        let input = &operands[0];
        let shape = input.shape();

        // Broadcast the per-slice maximum back to the input shape.
        let mut max_val = input.max_dims(&self.dims);
        for &d in &self.dims {
            max_val = max_val.unsqueeze(d).expand(d, shape[d]);
        }

        // mask of the elements that reach the max (a >= max(value)); no element
        // can exceed it
        let mask = input.gte(&max_val);

        // broadcast the upstream gradient back to the input shape
        let mut grad_b = in_grad.copy_s();
        for &d in &self.dims {
            grad_b = grad_b.unsqueeze(d).expand(d, shape[d]);
        }

        out.push(input.requires_grad().then(|| &grad_b * &mask));
    }
}
