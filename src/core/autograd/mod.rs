pub mod grad_fn;
pub mod ops;

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::core::{GraphTensor, dtype::{Dtype, Float}, node::TensorNode};

use self::grad_fn::GradFnTrait;

pub struct TensorKey<T: Dtype = f64> {
    node: Rc<TensorNode<T>>, // TODO or use GraphTensor directly
}

impl<T: Dtype> Clone for TensorKey<T> {
    fn clone(&self) -> Self {
        TensorKey {
            node: Rc::clone(&self.node),
        }
    }
}

impl<T: Dtype> PartialEq for TensorKey<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::as_ptr(&self.node) == Rc::as_ptr(&other.node)
    }
}

impl<T: Dtype> Eq for TensorKey<T> {}

impl<T: Dtype> Hash for TensorKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.node).hash(state);
    }
}

impl<T: Dtype> GraphTensor<T> {
    pub fn to_key(&self) -> TensorKey<T> {
        TensorKey {
            node: Rc::clone(&self.node),
        }
    }
}

impl<T: Float> GraphTensor<T> {
    pub fn backward(&self, retain_graph: bool) -> HashMap<TensorKey<T>, GraphTensor<T>> {
        self.compile_backward().run(retain_graph)
    }

    /// Precompile the backward schedule for the graph reachable from `self`.
    ///
    /// The returned plan owns `Rc` handles to every node of the forward graph, so it
    /// stays valid as long as the graph is alive and can be re-executed any number of
    /// times (e.g. with different `retain_graph` flags), reusing the compiled
    /// topological schedule and the gradient/scratch buffers across runs.
    pub fn compile_backward(&self) -> BackwardPlan<T> {
        BackwardPlan::build(self)
    }
}

/// A fully compiled backward schedule for a fixed forward graph.
///
/// Nodes are assigned dense integer indices once at build time; subsequent runs are
/// pure integer/Vec operations (no hashing, no per-node key churn, no per-node
/// scratch allocations) until the final leaf-gradient map is materialized.
///
/// Backward passes only run for float dtypes (every gradient is a float workload),
/// so the plan's gradient buffers are `GraphTensor<T>` for `T: Float`.
pub struct BackwardPlan<T: Dtype = f64> {
    nodes: Vec<TensorKey<T>>,
    seed_idx: usize,
    operands: Vec<Vec<usize>>,
    // Materialized rules: `grad_fns[u]` is the concrete backward op for node
    // `u`, frozen from its deferred `BackwardSource` at build time (see
    // `BackwardSource::into_grad_fn`). `run` executes these without touching
    // the live forward graph.
    grad_fns: Vec<Option<Box<dyn GradFnTrait<T>>>>,
    is_leaf: Vec<bool>,
    base_in_degree: Vec<usize>,
    grads: Vec<Option<GraphTensor<T>>>,
    scratch: Vec<Option<GraphTensor<T>>>,
}

impl<T: Float> BackwardPlan<T> {
    /// Walk the forward graph once, assigning each reachable, requires-grad node a
    /// dense index and recording its operand indices, leaf-ness, and in-degree.
    fn build(seed: &GraphTensor<T>) -> BackwardPlan<T> {
        let mut nodes: Vec<TensorKey<T>> = Vec::new();
        let mut index_of: HashMap<*const TensorNode<T>, usize> = HashMap::new();
        let mut operands: Vec<Vec<usize>> = Vec::new();
        let mut grad_fns: Vec<Option<Box<dyn GradFnTrait<T>>>> = Vec::new();
        let mut is_leaf: Vec<bool> = Vec::new();
        let mut bfs_queue: VecDeque<usize> = VecDeque::new();

        let seed_idx = 0usize;
        nodes.push(seed.to_key());
        index_of.insert(Rc::as_ptr(&seed.node), seed_idx);
        operands.push(Vec::new());
        grad_fns.push(None);
        is_leaf.push(seed.node.grad_fn.is_none());
        bfs_queue.push_back(seed_idx);

        while let Some(u) = bfs_queue.pop_front() {
            // do not propagate through nodes that do not require gradients.
            if !nodes[u].node.requires_grad {
                continue;
            }

            // Snapshot the operand graph tensors so `nodes` can grow while iterating.
            let op_graphs = {
                let Some(src) = &nodes[u].node.grad_fn else {
                    continue;
                };
                src.operands()
                    .iter()
                    .map(|op| op.copy_s())
                    .collect::<Vec<_>>()
            };

            // Freeze this node's deferred rule into the plan (T: Float here, so
            // every rule's Numeric/Signed/Float bound is satisfiable).
            grad_fns[u] = nodes[u]
                .node
                .grad_fn
                .as_ref()
                .map(|src| (**src).clone().into_grad_fn());

            for op in op_graphs.iter() {
                let ptr = Rc::as_ptr(&op.node);
                let v = match index_of.get(&ptr) {
                    Some(&v) => v,
                    None => {
                        let v = nodes.len();
                        nodes.push(op.to_key());
                        index_of.insert(ptr, v);
                        operands.push(Vec::new());
                        grad_fns.push(None);
                        is_leaf.push(op.node.grad_fn.is_none());
                        bfs_queue.push_back(v);
                        v
                    }
                };
                operands[u].push(v);
            }
        }

        let node_count = nodes.len();
        let mut base_in_degree = vec![0usize; node_count];

        for operand in operands.iter().take(node_count) {
            for &v in operand {
                base_in_degree[v] += 1;
            }
        }

        BackwardPlan {
            nodes,
            seed_idx,
            operands,
            grad_fns,
            is_leaf,
            base_in_degree,
            grads: (0..node_count).map(|_| None).collect(),
            scratch: Vec::new(),
        }
    }

    /// Execute the compiled backward pass from the seed this plan was built for.
    pub fn run(&mut self, retain_graph: bool) -> HashMap<TensorKey<T>, GraphTensor<T>> {
        assert!(
            self.nodes[self.seed_idx].node.requires_grad,
            "Cannot run backward() on a tensor with requires_grad=False. Likely, the graph has no leaf nodes requiring gradients."
        );

        let mut in_degree = self.base_in_degree.clone();
        for grad in self.grads.iter_mut() {
            *grad = None;
        }

        // NOTE: if 'retain_graph' = True, the gradient tensors have 'requires_grad = True'
        //       otherwise, you cannot compute higher-order derivatives
        let seed_shape = self.nodes[self.seed_idx].node.storage.shape.clone();
        let seed_grad = GraphTensor::new(seed_shape, T::from_f64(1.0), retain_graph);
        self.grads[self.seed_idx] = Some(seed_grad.copy_s());

        let mut process_queue: VecDeque<usize> = VecDeque::new();
        process_queue.push_back(self.seed_idx);

        while let Some(u) = process_queue.pop_front() {
            // Skip nodes that do not require gradients.
            if !self.nodes[u].node.requires_grad {
                continue;
            }

            let Some(grad_fn) = &self.grad_fns[u] else {
                continue;
            };

            {
                let in_grad = self.grads[u].as_ref().unwrap();
                grad_fn.compute_operands_grad(in_grad, retain_graph, &mut self.scratch);
            }

            for (&v, op_grad_opt) in self.operands[u].iter().zip(self.scratch.iter()) {
                if let Some(op_grad) = op_grad_opt {
                    if self.grads[v].is_some() {
                        accumulate_grad(&mut self.grads, v, op_grad, retain_graph);
                    } else {
                        self.grads[v] = Some(op_grad.copy_s());
                    }
                }

                // Decrement the in-degree of every operand edge. Operands that never
                // receive a gradient contribution (non-requires-grad) simply never
                // reach zero and are never scheduled.
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    process_queue.push_back(v);
                }
            }
        }

        // Materialize the output map, moving gradients out of the reusable buffer.
        // First-order path: keep only leaf tensors; higher-order callers keep every
        // gradient so they can fetch intermediates.
        let mut grads_map = HashMap::new();
        for (i, grad) in self.grads.iter_mut().enumerate() {
            if (retain_graph || self.is_leaf[i])
                && let Some(g) = grad.take()
            {
                grads_map.insert(self.nodes[i].clone(), g);
            }
        }

        grads_map
    }
}

/// Merge `op_grad` into the gradient accumulated so far for node `v`.
///
/// Higher-order path: keep the additive structure of the accumulated gradient so
/// that recomputing its derivative is still possible. The summing step is an
/// explicit graph node.
///
/// First-order path: fuse the new contribution into the existing gradient buffer
/// in place, skipping the allocation and the graph node entirely. Fall back to
/// the allocating sum when the buffer cannot be mutated (shared/aliased or
/// strided).
fn accumulate_grad<T: Float>(
    grads: &mut [Option<GraphTensor<T>>],
    v: usize,
    op_grad: &GraphTensor<T>,
    retain_graph: bool,
) {
    if retain_graph {
        let a = grads[v].as_ref().unwrap();
        let sum = a + op_grad;
        grads[v] = Some(sum);
    } else {
        let fused = match grads.get_mut(v).and_then(|g| g.as_mut()) {
            Some(a) => try_accumulate_inplace(a, op_grad),
            None => false,
        };
        if !fused {
            let a = grads[v].as_ref().unwrap();
            let sum = a + op_grad;
            grads[v] = Some(sum);
        }
    }
}

/// Try to mutate `a`'s buffer in place by adding `b`'s values. Returns `false`
/// without touching anything when the node or its buffer is not uniquely owned or
/// the storage cannot be mutated, in which case the caller falls back to `a + b`.
///
/// This destroys `a`'s pre-sum graph structure, so it is only used on the
/// first-order (retain_graph = False) path.
fn try_accumulate_inplace<T: Float>(a: &mut GraphTensor<T>, b: &GraphTensor<T>) -> bool {
    let Some(node) = Rc::get_mut(&mut a.node) else {
        return false;
    };

    if !node.storage.try_add_assign(&b.node.storage) {
        return false;
    }

    // First-order path only: this accumulated gradient no longer carries graph
    // structure (matching compute_operands_grad's stripping of intermediate grads).
    node.requires_grad = false;
    node.grad_fn = None;

    true
}