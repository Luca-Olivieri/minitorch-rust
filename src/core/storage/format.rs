use std::fmt;

use crate::core::storage::TensorStorage;

// TODO implementing fmt::Display for TensorStorage is very convenient to print it easily, but in practice you should only dbg!, since TensorStorage is not meant to be exposed to the user

impl fmt::Display for TensorStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype=float,\n", self.shape)?; // TODO insert TensorStorage, and fix padding
        write!(f, "       numel={}, strides={:?}, contiguous={}, offset={},\n", self.numel, self.strides, self.contiguous, self.offset)?;
        write!(f, "       data=")?;

        if self.shape.is_empty() {
            if let Some(&val) = self.buffer.get(0) {
                write!(f, "{:.4}", val)?;
            }
        } else {
            let mut curr_md_idx: Vec<usize> = Vec::new();
            self.print_recursive(f, 0, &mut curr_md_idx, 12)?;
        }

        write!(f, ")")
    }
}

impl TensorStorage {
    fn print_recursive(
        &self,
        f: &mut fmt::Formatter<'_>,
        dim_index: usize,
        curr_md_idx: &mut Vec<usize>,
        indent: usize,
    ) -> fmt::Result {
        let dim_size = self.shape[dim_index];

        if dim_index == self.shape.len() - 1 {
            write!(f, "[")?;
            for i in 0..dim_size {
                curr_md_idx.push(i);
                let val = self[self.md_to_flat(curr_md_idx)];
                curr_md_idx.pop();

                write!(f, "{:.4}", val)?;
                if i < dim_size - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
        } else {
            write!(f, "[")?;
            for i in 0..dim_size {
                if i > 0 {
                    write!(f, ",")?;
                    let newlines = self.shape.len() - dim_index - 1;
                    for _ in 0..newlines {
                        writeln!(f)?;
                    }
                    for _ in 0..(indent + 1) {
                        write!(f, " ")?;
                    }
                }
                curr_md_idx.push(i);
                self.print_recursive(f, dim_index + 1, curr_md_idx, indent + 1)?;
                curr_md_idx.pop();
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}
