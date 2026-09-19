use std::fmt;

use crate::core::dtype::DtypeStyler;
use crate::core::storage::TensorStorage;

impl<T: DtypeStyler> fmt::Display for TensorStorage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data_indent = "       data=".len();
        writeln!(
            f,
            "Tensor(shape={:?}, dtype={},",
            self.shape,
            T::dtype_name()
        )?;
        writeln!(
            f,
            "       numel={}, strides={:?}, contiguous={}, offset={},",
            self.numel, self.strides, self.contiguous, self.offset
        )?;
        write!(f, "       data=")?;

        if self.shape.is_empty() {
            if let Some(val) = self.buffer.first() {
                T::fmt_val(val, f)?;
            }
        } else {
            let mut curr_md_idx: Vec<usize> = Vec::new();
            self.print_recursive(f, 0, &mut curr_md_idx, data_indent)?;
        }

        write!(f, ")")
    }
}

impl<T: DtypeStyler> TensorStorage<T> {
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
                let val = self.at(curr_md_idx);
                curr_md_idx.pop();

                T::fmt_val(val, f)?;
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
