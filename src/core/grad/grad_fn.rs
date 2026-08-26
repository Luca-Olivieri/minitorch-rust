use crate::core::{GraphTensor, tensor::AbstractTensor};

#[derive(Debug)]
pub struct NBackwardOp<const N: usize> {
    pub(crate) operands: [GraphTensor; N] // TODO should the slice contain &GraphTensor?
}

pub trait GradFnTrait: HasOperands + ComputesGrads + std::fmt::Debug {}

// blanket impl so anything satisfying the bounds auto-implements GradFnTrait
impl<T: HasOperands + ComputesGrads + std::fmt::Debug> GradFnTrait for T {}

pub trait HasOperands {
    fn get_operands(&self) -> &[GraphTensor];
}

pub trait ComputesGrads {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>>; // TODO might create a separate struct for this gradients map
}

impl<const N: usize> HasOperands for NBackwardOp<N> {
    fn get_operands(&self) -> &[GraphTensor] { // TODO is there a way to return a &[GraphTensor; N]?
        &self.operands
    }
}

pub type BackwardAdd = NBackwardOp<2>;

impl ComputesGrads for BackwardAdd {
    fn compute_operands_grad(
        &self,
        in_grad: &GraphTensor
    ) -> Vec<Option<GraphTensor>> {

        let mut out_grads = Vec::with_capacity(2);

        let ones = GraphTensor::new(self.operands[0].shape().clone(), 1.0, false); // TODO find a better way to do this

        if self.operands[0].requires_grad() { out_grads.push(Some(in_grad * &ones)); } else { out_grads.push(None);}
        if self.operands[1].requires_grad() { out_grads.push(Some(in_grad * &ones)); } else { out_grads.push(None);}

        out_grads
    }
}

// void BackwardAdd::compute_operands_grad(
//         const Tensor& out
// ) {
//     m_operands[0].accumulate_grad(out.grad());
//     m_operands[1].accumulate_grad(out.grad());
// }
