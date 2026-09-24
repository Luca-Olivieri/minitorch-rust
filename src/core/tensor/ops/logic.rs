use std::rc::Rc;

use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::GraphTensor;

// Logical operations are boolean-native: they target `Tensor<bool>` directly
// (matching the dtype design), so the old numeric `!= 0.0` f64 forms are gone.
// Booleans are never differentiable, so these are pure data passthroughs
// (no grad_fn, no requires_grad propagation).
impl GraphTensor<bool> {
    pub fn land(&self, b: &GraphTensor<bool>) -> GraphTensor<bool> {
        binary_bool(self, b, TensorStorage::land)
    }

    pub fn lor(&self, b: &GraphTensor<bool>) -> GraphTensor<bool> {
        binary_bool(self, b, TensorStorage::lor)
    }

    pub fn lnot(&self) -> GraphTensor<bool> {
        let out = TensorStorage::lnot(&[&self.node.storage]);
        GraphTensor {
            node: Rc::new(TensorNode {
                storage: out,
                requires_grad: false,
                no_grad: self.is_no_grad(),
                grad_fn: None,
            }),
        }
    }
}

fn binary_bool(
    a: &GraphTensor<bool>,
    b: &GraphTensor<bool>,
    op: fn(&[&TensorStorage<bool>; 2]) -> TensorStorage<bool>,
) -> GraphTensor<bool> {
    let (sa, sb) = broadcast_pair(a, b);
    let out = op(&[&sa, &sb]);
    GraphTensor {
        node: Rc::new(TensorNode {
            storage: out,
            requires_grad: false,
            no_grad: a.is_no_grad() || b.is_no_grad(),
            grad_fn: None,
        }),
    }
}

fn broadcast_pair(
    a: &GraphTensor<bool>,
    b: &GraphTensor<bool>,
) -> (TensorStorage<bool>, TensorStorage<bool>) {
    let target_shape = broadcast_shape(&[&a.node.storage.shape, &b.node.storage.shape]);
    let sa = a.node.storage.broadcast_to_shape(&target_shape);
    let sb = b.node.storage.broadcast_to_shape(&target_shape);
    (sa, sb)
}

fn broadcast_shape(shapes: &[&[usize]]) -> Vec<usize> {
    let ndim = shapes.iter().map(|s| s.len()).max().unwrap();
    let mut out = vec![1usize; ndim];
    for s in shapes {
        let pad = ndim - s.len();
        for (i, &dim) in s.iter().enumerate() {
            let d = pad + i;
            if out[d] != dim && out[d] != 1 && dim != 1 {
                panic!(
                    "Shapes {:?} cannot be broadcast together: dim {} is {} but the common shape requires {}.",
                    shapes, d, dim, out[d]
                );
            }
            out[d] = out[d].max(dim);
        }
    }
    out
}
