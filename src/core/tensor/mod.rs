mod indexing;
pub mod ops;

use std::rc::Rc;

use crate::core::node::TensorNode;

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

fn dist_l2(
    a: &GraphTensor,
    b: &GraphTensor,
) -> f64 {
    // let diff = a - b;
    // diff.pow(other)
    4.0
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

    // fn norm_lp(&self, p: f64) -> f64 {
    //     let sp: f64 = self.get_node().storage.flat_data.iter().map(|x| x.powf(p)).sum();
    //     sp.powf(1.0/p)
    // }
}

impl AbstractTensor for GraphTensor {

    fn get_node(&self) -> &TensorNode {
        &self.node.as_ref()
    }
}

fn extract_requires_grad(operands: &[&GraphTensor]) -> bool {
    operands.iter().any(|t| t.get_node().requires_grad)
}
