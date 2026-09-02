use std::rc::Rc;

use rand::rngs::StdRng;

use crate::core::{GraphTensor, node::TensorNode, storage::TensorStorage};

impl GraphTensor {
    pub fn init_xavier_uniform(
        shape: Vec<usize>,
        requires_grad: bool,
        rng: StdRng
    ) -> Self {
        let node = TensorNode {
            storage: TensorStorage::init_xavier_uniform(shape, rng),
            requires_grad,
            grad_fn: None
        };

        Self { node: Rc::new(node) }
    }
}
