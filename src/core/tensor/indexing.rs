use crate::core::{GraphTensor, tensor::{AbstractTensor, FreeTensor}};

// TODO use at and at_mut syntax  also in TensorStorage

impl FreeTensor {

    pub fn set(
        &mut self,
        md_idx: &Vec<usize>,
        value: f64
    ) {
        self.node.storage[md_idx] = value;
    }
}

impl GraphTensor {
    pub fn item(&self) -> f64 {
        if self.numel() != 1 {
            panic!("Cannot call item() on a non-singleton tensor (shape {:?}).", self.shape());
        }
        self.get_node().storage.buffer[0]
    }
}
