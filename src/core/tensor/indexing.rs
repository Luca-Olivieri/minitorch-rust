use std::{ops::Index, rc::Rc};

use crate::core::Tensor;

// TODO Rc handled immutable cells, use RefCell instead?

impl Index<usize> for Tensor {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.node.storage[i]
    }
}

impl Index<&Vec<usize>> for Tensor {
    type Output = f64;

    fn index(
        &self,
        md_idx: &Vec<usize>
    ) -> &f64 {
        &self.node.storage[md_idx]
    }
}

impl Tensor {
    pub fn set(
        &mut self,
        md_idx: &Vec<usize>,
        value: f64
    ) {
        let s = &mut Rc::get_mut(&mut self.node)
            .expect("cannot mutate a Tensor once it has been shared into the graph")
            .storage;
        s[md_idx] = value;
    }
}
