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

    /// Build a `GraphTensor` from an arbitrarily nested `Vec<f64>`.
    ///
    /// Both the shape and the underlying flat data are inferred from the
    /// structure of the input, e.g. `vec![1.0, 2.0]`, `vec![vec![1.0, 2.0]]`,
    /// `vec![vec![vec![1.0, 2.0]]]`, etc.
    ///
    /// Panics if the nested structure is ragged (children of the same parent
    /// have differing shapes).
    pub fn from_vec<T: IntoNestedStorage>(
        data: T,
        requires_grad: bool
    ) -> Self {
        let mut buffer: Vec<f64> = Vec::new();
        let shape = data.collect_into(&mut buffer);

        let storage = TensorStorage::from_buffer(shape, buffer);
        let node = TensorNode::from_storage(storage, requires_grad);

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

/// Trait abstracting over arbitrarily nested `Vec<f64>` so that `GraphTensor::from_vec`
/// can be called with any nesting depth. Implementations recursively collect the flat
/// data into `buffer` and return the shape of the sub-structure they represent.
pub trait IntoNestedStorage {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize>;
}

impl IntoNestedStorage for f64 {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize> {
        buffer.push(self);
        Vec::new()
    }
}

impl<T: IntoNestedStorage> IntoNestedStorage for Vec<T> {
    fn collect_into(self, buffer: &mut Vec<f64>) -> Vec<usize> {
        let mut shape = vec![self.len()];

        let mut child_shape: Option<Vec<usize>> = None;
        for child in self {
            let cs = child.collect_into(buffer);
            match &child_shape {
                None => child_shape = Some(cs),
                Some(prev) => {
                    if *prev != cs {
                        panic!("Ragged nested Vec: children of the same parent have differing shapes.");
                    }
                }
            }
        }

        if let Some(cs) = child_shape {
            shape.extend(cs);
        }

        shape
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_scalar() {
        let t = GraphTensor::from_vec(5.0, false);
        assert_eq!(t.shape(), &vec![]);
        assert_eq!(*t.at(&vec![]), 5.0);
    }

    #[test]
    fn from_vec_1d() {
        let t = GraphTensor::from_vec(vec![1.0, 2.0, 3.0], false);
        assert_eq!(t.shape(), &vec![3]);
        assert_eq!(*t.at(&vec![2]), 3.0);
    }

    #[test]
    fn from_vec_2d() {
        let t = GraphTensor::from_vec(
            vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0]
            ], false);
        assert_eq!(t.shape(), &vec![2, 2]);
        assert_eq!(*t.at(&vec![1, 0]), 3.0);
        assert_eq!(*t.at(&vec![0, 1]), 2.0);
    }

    #[test]
    fn from_vec_3d() {
        let t = GraphTensor::from_vec(
            vec![
                vec![
                    vec![1.0],
                    vec![2.0]
                ],
                vec![
                    vec![3.0],
                    vec![4.0]
                ]
            ],
            false,
        );
        assert_eq!(t.shape(), &vec![2, 2, 1]);
        assert_eq!(*t.at(&vec![1, 1, 0]), 4.0);
    }

    #[test]
    #[should_panic]
    fn from_vec_ragged_panics() {
        GraphTensor::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0]
        ], false);
    }
}
