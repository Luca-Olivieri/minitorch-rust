pub(crate) mod erased;
pub mod grad_fn;
pub mod ops;

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::core::autograd::erased::{ErasedTensor, FloatRepr, GradValue};
use crate::core::autograd::grad_fn::{ErasedGradFn, materialize_grad_fn};
use crate::core::{GraphTensor, dtype::Float};

/// Dtype-erased gradient map, keyed by node identity.
///
/// A single backward pass may produce more than one gradient dtype: a
/// differentiable cast is the one edge that connects two different floats, and
/// the invariant "a gradient w.r.t. an input has that input's dtype" then forces
/// both `f32` and `f64` gradients into the same map. So the map is keyed by node
/// identity rather than by a dtype-typed key; read a gradient back in its
/// concrete dtype with [`GradMap::get`].
pub struct GradMap {
    slots: HashMap<*const (), GradValue>,
}

impl GradMap {
    /// Number of gradients in the map (leaf gradients by default).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The gradient accumulated for `tensor`, or `None` if it is absent or its
    /// dtype is not `T` (e.g. `get::<f64>` on an `f32` leaf).
    #[allow(private_bounds)]
    pub fn get<T: FloatRepr>(&self, tensor: &GraphTensor<T>) -> Option<&GraphTensor<T>> {
        let ptr = Rc::as_ptr(&tensor.node) as *const ();
        self.slots.get(&ptr).and_then(T::try_ref_grad_value)
    }
}

#[allow(private_bounds)]
impl<T: FloatRepr> GraphTensor<T> {
    /// Run the backward pass for the graph reachable from `self`.
    ///
    /// The result is a dtype-erased [`GradMap`], because a mixed `f32`/`f64`
    /// graph (created by a differentiable cast) yields gradients in both dtypes.
    pub fn backward(&self, retain_graph: bool) -> GradMap {
        self.compile_backward().run(retain_graph)
    }

    /// Precompile the backward schedule for the graph reachable from `self`.
    ///
    /// The returned plan owns `Rc` handles to every node of the forward graph, so it
    /// stays valid as long as the graph is alive and can be re-executed any number of
    /// times (e.g. with different `retain_graph` flags), reusing the compiled
    /// topological schedule and the gradient/scratch buffers across runs.
    pub fn compile_backward(&self) -> BackwardPlan {
        BackwardPlan::build(self)
    }
}

/// A fully compiled backward schedule for a fixed forward graph.
///
/// Nodes are assigned dense integer indices once at build time; subsequent runs are
/// pure integer/Vec operations (no hashing, no per-node key churn) until the final
/// gradient map is materialized.
///
/// The plan is *dtype-erased*: nodes, rules and gradient slots are all
/// `ErasedTensor`/`GradValue`, dispatched per node over the closed `f32`/`f64`
/// set. That is what lets one graph (and so one backward pass) mix float dtypes,
/// which a differentiable cast requires. The typed math still lives in
/// `GradRule`/`NBackwardOp`, monomorphized per dtype.
pub struct BackwardPlan {
    nodes: Vec<ErasedTensor>,
    seed_idx: usize,
    operands: Vec<Vec<usize>>,
    // Materialized rules: `grad_fns[u]` is the erased backward op for node `u`,
    // frozen from its deferred `BackwardSource` at build time (see
    // [`materialize_grad_fn`]). `run` executes these without touching the live
    // forward graph.
    grad_fns: Vec<Option<Box<dyn ErasedGradFn>>>,
    is_leaf: Vec<bool>,
    base_in_degree: Vec<usize>,
    grads: Vec<Option<GradValue>>,
    scratch: Vec<Option<GradValue>>,
}

impl BackwardPlan {
    /// Walk the forward graph once, assigning each reachable, requires-grad node a
    /// dense index and recording its operand indices, leaf-ness, and in-degree.
    fn build<T: FloatRepr>(seed: &GraphTensor<T>) -> BackwardPlan {
        let mut nodes: Vec<ErasedTensor> = Vec::new();
        let mut index_of: HashMap<*const (), usize> = HashMap::new();
        let mut operands: Vec<Vec<usize>> = Vec::new();
        let mut grad_fns: Vec<Option<Box<dyn ErasedGradFn>>> = Vec::new();
        let mut is_leaf: Vec<bool> = Vec::new();
        let mut bfs_queue: VecDeque<usize> = VecDeque::new();

        let seed_idx = 0usize;
        let seed = T::into_erased(seed.copy_s());
        index_of.insert(seed.ptr(), seed_idx);
        is_leaf.push(seed.is_leaf());
        operands.push(Vec::new());
        grad_fns.push(None);
        nodes.push(seed);
        bfs_queue.push_back(seed_idx);

        while let Some(u) = bfs_queue.pop_front() {
            // do not propagate through nodes that do not require gradients.
            if !nodes[u].requires_grad() {
                continue;
            }

            // Snapshot the erased operands so `nodes` can grow while iterating,
            // then freeze this node's deferred rule into the plan.
            let op_erased = nodes[u].erased_operands();
            grad_fns[u] = materialize_grad_fn(&nodes[u]);

            for op in op_erased {
                let ptr = op.ptr();
                let v = match index_of.get(&ptr) {
                    Some(&v) => v,
                    None => {
                        let v = nodes.len();
                        is_leaf.push(op.is_leaf());
                        operands.push(Vec::new());
                        grad_fns.push(None);
                        nodes.push(op);
                        index_of.insert(ptr, v);
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
    pub fn run(&mut self, retain_graph: bool) -> GradMap {
        self.execute(retain_graph);

        // Materialize the output map, moving gradients out of the reusable buffer.
        // First-order path: keep only leaf tensors; higher-order callers keep every
        // gradient so they can fetch intermediates.
        let mut slots = HashMap::new();
        for (i, grad) in self.grads.iter_mut().enumerate() {
            if (retain_graph || self.is_leaf[i])
                && let Some(g) = grad.take()
            {
                slots.insert(self.nodes[i].ptr(), g);
            }
        }

        GradMap { slots }
    }

    /// Walk the scheduled graph, filling `self.grads` with erased gradient slots.
    fn execute(&mut self, retain_graph: bool) {
        assert!(
            self.nodes[self.seed_idx].requires_grad(),
            "Cannot run backward() on a tensor with requires_grad=False. Likely, the graph has no leaf nodes requiring gradients."
        );

        let mut in_degree = self.base_in_degree.clone();
        for grad in self.grads.iter_mut() {
            *grad = None;
        }

        // NOTE: if 'retain_graph' = True, the gradient tensors have 'requires_grad = True'
        //       otherwise, you cannot compute higher-order derivatives
        let seed_grad = self.nodes[self.seed_idx].one_grad(retain_graph);
        self.grads[self.seed_idx] = Some(seed_grad.copy_s());

        let mut process_queue: VecDeque<usize> = VecDeque::new();
        process_queue.push_back(self.seed_idx);

        while let Some(u) = process_queue.pop_front() {
            // Skip nodes that do not require gradients.
            if !self.nodes[u].requires_grad() {
                continue;
            }

            let Some(grad_fn) = self.grad_fns[u].as_mut() else {
                continue;
            };

            {
                let in_grad = self.grads[u].as_ref().unwrap();
                grad_fn.compute(in_grad, retain_graph, &mut self.scratch);
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
fn accumulate_grad(
    grads: &mut [Option<GradValue>],
    v: usize,
    op_grad: &GradValue,
    retain_graph: bool,
) {
    if retain_graph {
        let a = grads[v].as_ref().unwrap();
        grads[v] = Some(add_grads(a, op_grad));
    } else {
        let fused = match grads.get_mut(v).and_then(|g| g.as_mut()) {
            Some(a) => try_add_assign_grad(a, op_grad),
            None => false,
        };
        if !fused {
            let a = grads[v].as_ref().unwrap();
            grads[v] = Some(add_grads(a, op_grad));
        }
    }
}

/// Add two gradient slots. They must share a dtype (see `GradValue`).
fn add_grads(a: &GradValue, b: &GradValue) -> GradValue {
    match (a, b) {
        (GradValue::F32(a), GradValue::F32(b)) => GradValue::F32(a + b),
        (GradValue::F64(a), GradValue::F64(b)) => GradValue::F64(a + b),
        _ => panic!("Cannot add gradients of different dtypes."),
    }
}

/// In-place accumulation for matching gradient slots.
fn try_add_assign_grad(a: &mut GradValue, b: &GradValue) -> bool {
    match (a, b) {
        (GradValue::F32(a), GradValue::F32(b)) => try_accumulate_inplace(a, b),
        (GradValue::F64(a), GradValue::F64(b)) => try_accumulate_inplace(a, b),
        _ => panic!("Cannot accumulate gradients of different dtypes."),
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
