use crate::core::dtype::Dtype;
use crate::core::tensor::FreeTensor;

impl<T: Dtype> FreeTensor<T> {
    pub fn set(&mut self, md_idx: &[usize], value: T) {
        self.node.storage.set(md_idx, value);
    }
}
