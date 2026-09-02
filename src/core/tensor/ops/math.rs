use std::rc::Rc;
use std::ops::{
    Add,
    Neg,
    Sub,
    Mul,
    Div,
};

use crate::core::tensor::{AbstractTensor, GraphTensor};
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::autograd::grad_fn::GradFnTrait;
use crate::core::tensor::extract_requires_grad;
use crate::core::autograd::ops::math::{
    AddOp, BackwardAdd, BackwardDiv, BackwardLn, BackwardMaximum, BackwardMul, BackwardNeg, BackwardPow, BackwardSub, DivOp, LnOp, MaximumOp, MulOp, NegOp, PowOp, SubOp
};

impl GraphTensor {
    impl_tensor_unary_op!(ln, TensorStorage::ln, BackwardLn, LnOp);
    impl_tensor_binary_op!(pow, TensorStorage::pow, BackwardPow, PowOp);
    impl_tensor_binary_op!(maximum, TensorStorage::maximum, BackwardMaximum, MaximumOp);
}

impl_tensor_binary_ops! {
    Add, add, TensorStorage::add,  BackwardAdd, AddOp;
    Sub, sub, TensorStorage::sub,  BackwardSub, SubOp;
    Mul, mul, TensorStorage::mul,  BackwardMul, MulOp;
    Div, div, TensorStorage::div,  BackwardDiv, DivOp;
}

impl_tensor_unary_ops! {
    Neg, neg, TensorStorage::neg, BackwardNeg, NegOp;
}

pub fn apply_tensor_op<F, G, const N: usize>(
    op: F,
    grad_fn: Option<G>,
    operands: &[&GraphTensor; N],
) -> GraphTensor
where
    F: Fn(&[&TensorStorage; N]) -> TensorStorage,
    G: FnOnce([GraphTensor; N]) -> Box<dyn GradFnTrait>,
{
    let first_operand_shape = &operands[0].node.storage.shape;
    for o in operands {
        assert_eq!(first_operand_shape, &o.node.storage.shape);
    }

    let storages: [&TensorStorage; N] = std::array::from_fn(|i| &operands[i].node.storage);
    let out_store = op(&storages);

    // Only copy operands and generate grad_fn_box if a grad_fn was provided
    let grad_fn_box: Option<Box<dyn GradFnTrait>> = grad_fn.map(|g| {
        let new_operands: [GraphTensor; N] = std::array::from_fn(|i| operands[i].copy_s());
        g(new_operands)
    });

    let out_node = TensorNode {
        storage: out_store,
        requires_grad: grad_fn_box.is_some() && extract_requires_grad(operands),
        grad_fn: grad_fn_box,
    };

    GraphTensor { node: Rc::new(out_node) }
}

// TODO make this scalar operations into a macro
impl Add<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn add(
        self,
        other: f64
    ) -> GraphTensor {
        let other_t = GraphTensor::new(self.shape().clone(), other, false);
        self + &other_t
    }
}

impl Sub<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn sub(
        self,
        other: f64
    ) -> GraphTensor {
        let other_t = GraphTensor::new(self.shape().clone(), other, false);
        self - &other_t
    }
}

impl Mul<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn mul(
        self,
        other: f64
    ) -> GraphTensor {
        let other_t = GraphTensor::new(self.shape().clone(), other, false);
        self * &other_t
    }
}

impl Div<f64> for &GraphTensor {
    type Output = GraphTensor;
    fn div(
        self,
        other: f64
    ) -> GraphTensor {
        let other_t = GraphTensor::new(self.shape().clone(), other, false);
        self / &other_t
    }
}
