pub mod ops;
pub mod grad_fn;

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

/// Returns (in-degree map, set of leaf tensors reachable from the seed).
///
/// A tensor is a leaf when it has no associated backward function (i.e. it was not
/// produced by an operation). Leaves are captured *before* any gradient computation
/// strips `grad_fn` from intermediate nodes, so they identify true parameters/inputs.
fn compute_in_degree(seed: GraphTensor) -> (HashMap<TensorKey, u64>, HashSet<TensorKey>) { // TODO should seed be owned?
    let mut in_degree: HashMap<TensorKey, u64> = HashMap::new();
    let mut bfs_queue: VecDeque<TensorKey> = VecDeque::new();
    let mut visited: HashSet<TensorKey> = HashSet::new();
    let mut is_leaf: HashSet<TensorKey> = HashSet::new();

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

        if u.node.grad_fn.is_none() {
            is_leaf.insert(u.clone());
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

    (in_degree, is_leaf)
}

fn topological_backprop(
    seed: GraphTensor, // TODO should seed be owned?
    retain_graph: bool
) -> HashMap<TensorKey, GraphTensor> {

    let (mut in_degree, is_leaf) = compute_in_degree(seed.copy_s());

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
            let ops_grad = grad_fn.compute_operands_grad(in_grad, retain_graph);

            for (op, op_grad_opt) in grad_fn.get_operands().iter().zip(ops_grad.iter()) {

                if let Some(op_grad) = op_grad_opt {
                    let op_key = op.to_key();

                    if grads_map.contains_key(&op_key) {
                        if retain_graph {
                            // Higher-order path: preserve the additive structure of the
                            // accumulated gradient so that recomputing its derivative is
                            // still possible. The summing step is an explicit graph node.
                            // This unwrap is 100% safe because 'u' requires grad and was reached.
                            let a = grads_map.get(&op_key).unwrap();
                            grads_map.insert(op_key, a + op_grad);
                        } else {
                            // First-order path: fuse the new contribution into the existing
                            // gradient buffer in place, skipping the allocation and the
                            // graph node entirely. Fall back to the allocating sum when the
                            // buffer cannot be mutated (shared/aliased or strided).
                            let fused = match grads_map.get_mut(&op_key) {
                                Some(a) => accumulate_inplace(a, op_grad),
                                None => false,
                            };
                            if !fused {
                                let a = grads_map.get(&op_key).unwrap();
                                grads_map.insert(op_key, a + op_grad);
                            }
                        }
                    } else {
                        grads_map.insert(op_key, op_grad.copy_s());
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

    // First-order path only: retain only the gradients of leaf tensors. Higher-order
    // callers (retain_graph=True) keep every gradient so they can fetch intermediates.
    if !retain_graph {
        grads_map.retain(|k, _| is_leaf.contains(k));
    }

    grads_map
}

/// Mutate `a`'s buffer in place by adding `b`'s values. Returns `false` without
/// touching anything when the node or its buffer is not uniquely owned or the
/// storage cannot be mutated, in which case the caller falls back to `a + b`.
///
/// This destroys `a`'s pre-sum graph structure, so it is only used on the
/// first-order (retain_graph = False) path.
fn accumulate_inplace(a: &mut GraphTensor, b: &GraphTensor) -> bool {
    let Some(node) = Rc::get_mut(&mut a.node) else {
        return false;
    };

    if !node.storage.add_assign(&b.node.storage) {
        return false;
    }

    // First-order path only: this accumulated gradient no longer carries graph
    // structure (matching compute_operands_grad's stripping of intermediate grads).
    node.requires_grad = false;
    node.grad_fn = None;

    true
}
