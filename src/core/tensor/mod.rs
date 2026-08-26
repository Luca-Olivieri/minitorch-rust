mod indexing;

use std::rc::Rc;
use std::ops::{Add, Neg, Sub, Mul, Div};

use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::grad::grad_fn::{
    GradFnTrait,
    BackwardAdd,
    BackwardNeg,
    BackwardSub,
    BackwardMul
};

#[derive(Debug)]
pub struct Tensor<'a> {
    g_tensor: &'a GraphTensor,
}

pub trait AbstractTensor {

    fn get_node(&self) -> &TensorNode; // TODO return a reference, or some sort of weak pointer?

    fn at(&self, md_idx: &Vec<usize>) -> &f64 {
        &self.get_node().storage[md_idx]
    }

    fn shape(&self) -> &Vec<usize> {
        &self.get_node().storage.shape
    }

    fn numel(&self) -> usize {
        self.get_node().storage.numel
    }

    fn requires_grad(&self) -> bool {
        self.get_node().requires_grad
    }
}

#[derive(Debug)]
pub struct FreeTensor { // TODO find a definitive name
    node: Box<TensorNode>,
}

impl AbstractTensor for FreeTensor {

    fn get_node(&self) -> &TensorNode {
        &self.node.as_ref()
    }
}

impl FreeTensor {

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool
    ) -> Self {
        let node = TensorNode::new(shape, fill_value, requires_grad);

        Self { node: Box::new(node) }
    }
}

#[derive(Debug)]
pub struct GraphTensor {
    pub(super) node: Rc<TensorNode>
}

impl GraphTensor { // turn this impl and the above one in a macro

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool
    ) -> Self {
        let node = TensorNode::new(shape, fill_value, requires_grad);

        Self { node: Rc::new(node) }
    }

    pub fn copy_s(&self) -> GraphTensor {
        Self { node: self.node.clone()}
    }
}

impl AbstractTensor for GraphTensor {

    fn get_node(&self) -> &TensorNode {
        &self.node.as_ref()
    }
}

macro_rules! impl_tensor_binary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident);* $(;)?) => {
        $(
            impl_tensor_binary_op!($trait, $method, $storage_fn, $grad_fn);
        )*
    };
}
macro_rules! impl_tensor_binary_op {
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self, other: &GraphTensor) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 2]| $storage_fn(&[ops[0], ops[1]]),
                    |operands: [GraphTensor; 2]| {
                        Box::new($grad_fn { operands, _marker: std::marker::PhantomData}) as Box<dyn GradFnTrait>
                    },
                    &[self, other],
                )
            }
        }
    };
}

macro_rules! impl_tensor_unary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident);* $(;)?) => {
        $(
            impl_tensor_unary_op!($trait, $method, $storage_fn, $grad_fn);
        )*
    };
}
macro_rules! impl_tensor_unary_op {
    ($trait:ident, $method:ident, $storage_fn:path, $grad_fn:ident) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self) -> GraphTensor {
                apply_tensor_op(
                    |ops: &[&TensorStorage; 1]| $storage_fn(&[ops[0]]),
                    |operands: [GraphTensor; 1]| {
                        Box::new($grad_fn { operands, _marker: std::marker::PhantomData }) as Box<dyn GradFnTrait>
                    },
                    &[self],
                )
            }
        }
    };
}

impl_tensor_binary_ops! {
    Add, add, TensorStorage::add,  BackwardAdd;
    Sub, sub, TensorStorage::sub,  BackwardSub;
    Mul, mul, TensorStorage::mul,  BackwardMul;
    // Div, div, TensorStorage::div,  BackwardDiv;
}
impl_tensor_unary_ops! {
    Neg, neg, TensorStorage::neg, BackwardNeg;
}

fn apply_tensor_op<F, G, const N: usize>(
    op: F,
    grad_fn: G,
    operands: &[&GraphTensor; N],
) -> GraphTensor
where
    F: Fn(&[&TensorStorage; N]) -> TensorStorage,
    G: FnOnce([GraphTensor; N]) -> Box<dyn GradFnTrait>,
{
    let first_operand_shape = &operands[0].get_node().storage.shape;
    for o in operands {
        assert_eq!(first_operand_shape, &o.get_node().storage.shape);
    }

    let storages: [&TensorStorage; N] = std::array::from_fn(|i| &operands[i].get_node().storage);
    let out_store = op(&storages);

    let new_operands: [GraphTensor; N] = std::array::from_fn(|i| operands[i].copy_s());
    let grad_fn_box: Box<dyn GradFnTrait> = grad_fn(new_operands);

    let out_node = TensorNode {
        storage: out_store,
        requires_grad: extract_requires_grad(operands),
        grad_fn: Some(grad_fn_box),
    };

    GraphTensor { node: Rc::new(out_node) }
}

fn extract_requires_grad(operands: &[&GraphTensor]) -> bool {
    operands.iter().any(|t| t.get_node().requires_grad)
}
