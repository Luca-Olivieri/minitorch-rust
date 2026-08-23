use crate::core::tensor_storage::TensorStorage;
use std::ops::{Index, IndexMut};

impl TensorStorage {

    pub(super) fn md_to_flat(
        &self,
        md_idx: &Vec<usize>
    ) -> usize {
        if md_idx.len() != self.shape.len() {
            panic!("Index size {} does not match tensor shape size {}.", md_idx.len(), self.shape.len())
        }

        // flat index computation
        let mut flat_index = self.offset;
        for i in 0..self.shape.len() {
            if md_idx[i] >= self.shape[i] {
                panic!("Index {} out of bounds for dimension {} of size {}.", md_idx[i], i, self.shape[i]);
            }
            flat_index += self.strides[i]*md_idx[i];
        }

        flat_index
    }

    fn logic_to_flat(
        &self,
        l_idx: usize
    ) -> usize {
        if l_idx >= self.numel {
            panic!("Index {} out of bounds for tensor of size {}.", l_idx, self.numel);
        }

        if self.contiguous {
            l_idx + self.offset
        } else {
            let mut offset = self.offset;
            let mut curr_idx = l_idx;
            for i in (0..self.shape.len()).rev() {
                let dim_size = self.shape[i];
                let coord = curr_idx % dim_size;
                curr_idx /= dim_size;
                offset += coord * self.strides[i];
            }

            offset
        }
    }

    fn logic_to_md(
        &self,
        l_idx: usize
    ) -> Vec<usize> {
        if l_idx >= self.numel {
            panic!("Logical index {} out of bounds for tensor of size {}.", l_idx, self.numel);
        }

        let mut curr_idx = l_idx;

        let mut md = Vec::with_capacity(self.shape.len());

        for i in (0..self.shape.len()).rev() {
            md.push(curr_idx % self.shape[i]);
            curr_idx /= self.shape[i];
        }

        md
    }
}

impl Index<usize> for TensorStorage {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.flat_data[self.logic_to_flat(i)]
    }
}

impl IndexMut<usize> for TensorStorage {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        let f_idx = self.logic_to_flat(i);
        &mut self.flat_data[f_idx]
    }
}
