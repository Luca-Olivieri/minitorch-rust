use crate::core::tensor::FreeTensor;

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
