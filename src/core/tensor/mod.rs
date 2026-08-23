mod indexing;

use std::rc::Rc;
use std::ops::{Add, Sub, Mul, Div};

use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;

pub trait Tensor {

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
}

#[derive(Debug)]
pub struct FreeTensor { // TODO find a definitive name
    node: Box<TensorNode>,
}

impl Tensor for FreeTensor {

    fn get_node(&self) -> &TensorNode {
        &self.node
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
    node: Rc<TensorNode>,
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

    fn get_node(&self) -> &TensorNode {
        &self.node
    }
}

macro_rules! impl_tensor_binary_ops {
    ($($trait:ident, $method:ident, $storage_fn:path);* $(;)?) => {
        $(
            impl_tensor_binary_op!($trait, $method, $storage_fn);
        )*
    };
}

macro_rules! impl_tensor_binary_op {
    ($trait:ident, $method:ident, $storage_fn:path) => {
        impl $trait for &GraphTensor {
            type Output = GraphTensor;
            fn $method(self, other: &GraphTensor) -> GraphTensor {
                apply_tensor_op($storage_fn, &[self, other])
            }
        }
    };
}

impl_tensor_binary_ops! {
    Add, add, TensorStorage::add;
    Sub, sub, TensorStorage::sub;
    Mul, mul, TensorStorage::mult;
    Div, div, TensorStorage::div;
}

fn apply_tensor_op<F, const N: usize>(
    op: F,
    operands: &[&GraphTensor; N]
) -> GraphTensor
    where
        F: Fn(&[&TensorStorage; N]) -> TensorStorage,
{
    let first_operand_shape = &operands[0].node.storage.shape;

    for o in operands {
        assert_eq!(first_operand_shape, &o.node.storage.shape);
    }

    let storages: [&TensorStorage; N] = std::array::from_fn(|i| &operands[i].node.storage);

    let out_store = op(&storages);

    let out_node = TensorNode {
        storage: out_store,
        requires_grad: extract_requires_grad(operands)
    };

    GraphTensor { node: Rc::new(out_node) }
}

fn extract_requires_grad(operands: &[&GraphTensor]) -> bool {
    operands.iter().any(|t| t.node.requires_grad)
}
