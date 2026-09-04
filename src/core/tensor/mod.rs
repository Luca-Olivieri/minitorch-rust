mod indexing;
pub mod ops;
pub mod init;

use std::rc::Rc;

use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;

#[derive(Debug)]
pub struct Tensor<'a> {
    g_tensor: &'a GraphTensor,
}

pub trait AbstractTensor {

    fn get_node(&self) -> &TensorNode;

    fn get_node_mut(&mut self) -> &mut TensorNode;

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

    fn set_requires_grad(
        &mut self,
        requires_grad: bool
    ) {
        let node = self.get_node_mut();
        node.requires_grad = requires_grad;
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

    fn get_node_mut(&mut self) -> &mut TensorNode {
        self.node.as_mut()
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

    pub fn to_graph(
        self,
    ) -> GraphTensor {
        GraphTensor {
            node: Rc::from(self.node)
        }
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

    pub fn detach(&self, requires_grad: bool) -> GraphTensor {
        let node = TensorNode {
            storage: TensorStorage::copy_s(&self.node.storage),
            requires_grad,
            grad_fn: None,
        };

        Self { node: Rc::new(node) }
    }
}

impl AbstractTensor for GraphTensor {

    fn get_node(&self) -> &TensorNode {
        &self.node.as_ref()
    }

    fn get_node_mut(&mut self) -> &mut TensorNode {
        Rc::get_mut(&mut self.node).expect("Failed to obtain mutable reference of GraphTensor with shared ownership.")
    }
}

fn extract_requires_grad(operands: &[&GraphTensor]) -> bool {
    operands.iter().any(|t| t.get_node().requires_grad)
}
