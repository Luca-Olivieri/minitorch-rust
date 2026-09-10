use crate::core::storage::TensorStorage;
use std::{
    ops::{Index, IndexMut},
    rc::Rc,
};

impl TensorStorage {
    /// Try to add `other`'s values into `self`'s buffer elementwise, in place.
    ///
    /// Returns `false` (without mutating anything) when the accumulation cannot be
    /// done safely in place: shape mismatch, `self` is a strided view, or the buffer
    /// is not uniquely owned. Callers should fall back to an allocating `a + b`.
    pub(crate) fn try_add_assign(&mut self, other: &TensorStorage) -> bool {
        if self.shape != other.shape || self.numel != other.numel {
            return false;
        }

        // Only contiguous accumulation targets are mutated in place.
        if !self.contiguous {
            return false;
        }

        // The buffer must be uniquely owned to mutate it in place.
        let Some(buf) = Rc::get_mut(&mut self.buffer) else {
            return false;
        };

        for i in 0..self.numel {
            buf[self.offset + i] += other[i];
        }

        true
    }

    pub(super) fn md_to_flat(&self, md_idx: &Vec<usize>) -> usize {
        if md_idx.len() != self.shape.len() {
            panic!(
                "Index size {} does not match tensor shape size {}.",
                md_idx.len(),
                self.shape.len()
            )
        }

        // flat index computation
        let mut flat_index = self.offset;
        for i in 0..self.shape.len() {
            if md_idx[i] >= self.shape[i] {
                panic!(
                    "Index {} out of bounds for dimension {} of size {}.",
                    md_idx[i], i, self.shape[i]
                );
            }
            flat_index += self.strides[i] * md_idx[i];
        }

        flat_index
    }

    fn logic_to_flat(&self, l_idx: usize) -> usize {
        if l_idx >= self.numel {
            panic!(
                "Index {} out of bounds for tensor of size {}.",
                l_idx, self.numel
            );
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
}

impl Index<&Vec<usize>> for TensorStorage {
    type Output = f64;

    fn index(&self, md_idx: &Vec<usize>) -> &f64 {
        &self.buffer[self.md_to_flat(md_idx)]
    }
}

impl Index<usize> for TensorStorage {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.buffer[self.logic_to_flat(i)]
    }
}

// TODO see if it makes sense to validate md_idx before fetching the data

// TODO alternatively, the two IdexMut methods down here can be removed, and when they are used,
// modify the flat_data directly BEFORE giving it to the TensorStorage
impl IndexMut<&Vec<usize>> for TensorStorage {
    fn index_mut(&mut self, md_idx: &Vec<usize>) -> &mut f64 {
        let f_idx = self.md_to_flat(md_idx);
        &mut self.buffer_mut()[f_idx]
    }
}

impl IndexMut<usize> for TensorStorage {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        let f_idx = self.logic_to_flat(i);
        &mut self.buffer_mut()[f_idx]
    }
}
