use crate::core::tensor::FreeTensor;

impl FreeTensor {
    pub fn set(&mut self, md_idx: &[usize], value: f64) {
        self.node.storage.set(md_idx, value);
    }
}
