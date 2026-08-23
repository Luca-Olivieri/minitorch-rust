use std::rc::Rc;
use std::ops::{Add, Sub, Mul, Div};

use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub struct Tensor {
    node: Rc<TensorNode>,
}

impl Tensor {

    pub fn new(
        shape: Vec<usize>,
        fill_value: f64,
        requires_grad: bool
    ) -> Self {
        let node = TensorNode::new(shape, fill_value, requires_grad);

        Self { node: Rc::new(node) }
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
        impl $trait for &Tensor {
            type Output = Tensor;
            fn $method(self, other: &Tensor) -> Tensor {
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
    operands: &[&Tensor; N]
) -> Tensor
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

    Tensor { node: Rc::new(out_node) }
}

fn extract_requires_grad(operands: &[&Tensor]) -> bool {
    operands.iter().any(|t| t.node.requires_grad)
}
