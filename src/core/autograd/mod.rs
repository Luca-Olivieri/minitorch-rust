pub mod grad_fn;
pub mod ops;

use std::collections::{HashMap, VecDeque, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::core::GraphTensor;
use crate::core::node::TensorNode;
use crate::core::tensor::AbstractTensor;

pub struct TensorKey {
    node: Rc<TensorNode> // TODO or use GraphTensor directly
}

impl TensorKey {}

impl Clone for TensorKey {
    fn clone(&self) -> Self {
        TensorKey{node: Rc::clone(&self.node)}
    }
}

impl PartialEq for TensorKey {
    fn eq(&self, other: &Self) -> bool {
        Rc::as_ptr(&self.node) == Rc::as_ptr(&other.node)
    }
}

impl Eq for TensorKey {}

impl Hash for TensorKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.node).hash(state);
    }
}

impl GraphTensor {

    pub fn to_key(&self) -> TensorKey {

        TensorKey { node: Rc::clone(&self.node) }
    }
}

impl GraphTensor {
    pub fn backward(
        &self,
        retain_graph: bool
    ) -> HashMap<TensorKey, GraphTensor> {
        if !self.requires_grad() {
            panic!("Cannot call backward() on tensor with requires_grad=False. Likely, the graph has no leaf nodes requiring gradients.")
        }

        let grads_map = topological_backprop(self.copy_s(), retain_graph);
        grads_map
    }
}

fn compute_in_degree(seed: GraphTensor) -> HashMap<TensorKey, u64> { // TODO should seed be owned?
    let mut in_degree: HashMap<TensorKey, u64> = HashMap::new();
    let mut bfs_queue: VecDeque<TensorKey> = VecDeque::new();
    let mut visited: HashSet<TensorKey> = HashSet::new();

    // TODO should all this methods create owning TensorKey?

    let seed_key = seed.to_key();
    bfs_queue.push_back(seed_key.clone()); // TODO I can use the into() or cast() method for automatic conversion to TensorKey
    visited.insert(seed_key.clone());
    in_degree.insert(seed_key.clone(), 0);

    while let Some(u) = bfs_queue.pop_front() {

        // do not propagate through nodes that do not require gradients.
        if !u.node.requires_grad {
            continue;
        }

        if let Some(grad_fn) = &u.node.grad_fn {
            let operands = grad_fn.get_operands();

            for op in operands {
                let op_key = op.to_key();
                *in_degree.entry(op_key.clone()).or_insert(0) += 1;
                if visited.insert(op_key.clone()) {
                    bfs_queue.push_back(op_key.clone());
                }
            }
        }
    }

    in_degree
}

fn topological_backprop(
    seed: GraphTensor, // TODO should seed be owned?
    retain_graph: bool
) -> HashMap<TensorKey, GraphTensor> {

    let mut in_degree = compute_in_degree(seed.copy_s());

    let mut grads_map: HashMap<TensorKey, GraphTensor> = HashMap::new();

    // NOTE: if 'retain_graph' = True, the gradient tensors have 'requires_grad = True'
    //       otherwise, you cannot compute higher-order derivatives

    let seed_grad = GraphTensor::new(seed.shape().clone(), 1.0, retain_graph); // TODO change requires_grad for higher order derivates
    grads_map.insert(seed.to_key(), seed_grad.copy_s());

    let mut process_queue: VecDeque<TensorKey> = VecDeque::new();
    process_queue.push_back(seed.to_key());

    while let Some(u) = process_queue.pop_front() {

        // Fix: Skip nodes that do not require gradients
        if !u.node.requires_grad {
            continue;
        }

        if let Some(grad_fn) = &u.node.grad_fn {
            let in_grad = grads_map.get(&u).unwrap();
            let ops_grad = grad_fn.compute_operands_grad(in_grad);

            for (op, op_grad_opt) in grad_fn.get_operands().iter().zip(ops_grad.iter()) {

                if let Some(op_grad) = op_grad_opt {
                    if grads_map.contains_key(&op.to_key()) {
                        // This unwrap is 100% safe because 'u' requires grad and was reached.
                        let a = grads_map.get(&op.to_key()).unwrap();
                        grads_map.insert(op.to_key(), a + op_grad);
                    } else {
                        // let zeros = GraphTensor::new(u.node.storage.shape.clone(), 0.0, false);
                        // grads_map.insert(op.to_key(), op_grad + &zeros);
                        grads_map.insert(op.to_key(), op_grad.copy_s());
                    }
                }

                // We must still decrement the in-degree of operands (even if they don't require grad)
                // because we incremented them in compute_in_degree.
                if let Some(in_d) = in_degree.get_mut(&op.to_key()) {
                    *in_d -= 1;
                    if *in_d == 0 {
                        process_queue.push_back(op.to_key());
                    }
                }
            }

            // NOTE: cannot execute the free, because grad_fn are inside a mutable Rc, but the Rc should free it when possible
            // if !retain_graph {
            //     u.node.grad_fn = None
            // }
        }
    }

    grads_map
}
