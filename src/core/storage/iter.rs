use crate::core::storage::TensorStorage;

/// Incremental row-major walk over a tensor's logical indices.
///
/// Yields the flat buffer index of each logical element in row-major order.
/// A shared odometer advances the flat index by the corresponding stride on
/// every step (and subtracts `shape * stride` on the rare carry), so the
/// amortized cost is O(1) per element instead of the O(ndim) div/mod
/// decomposition used by per-element logical indexing. The innermost dim
/// usually does not carry, so its advance is a single `flat += stride`.
pub struct StridedIter<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    coords: Vec<usize>,
    flat: usize,
    remaining: usize,
}

impl<'a> StridedIter<'a> {
    pub(crate) fn new(
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
        numel: usize,
    ) -> Self {
        Self {
            shape,
            strides,
            coords: vec![0; shape.len()],
            flat: offset,
            remaining: numel,
        }
    }
}

impl Iterator for StridedIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }

        let cur = self.flat;
        self.remaining -= 1;

        if self.remaining > 0 {
            // advance the odometer like a mixed-radix counter: bump the least
            // significant (innermost) coordinate, cascading the carry outward.
            let mut k = self.coords.len();
            while k > 0 {
                k -= 1;
                self.coords[k] += 1;
                self.flat += self.strides[k];
                if self.coords[k] < self.shape[k] {
                    break;
                }
                self.coords[k] = 0;
                self.flat -= self.strides[k] * self.shape[k];
            }
        }

        Some(cur)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIter<'_> {}

impl TensorStorage {
    /// Flat buffer indices of each logical element, in row-major order.
    ///
    /// Prefer this over per-element `self[i]` indexing when looping over every
    /// element of a possibly strided tensor: it replaces O(ndim) divisions per
    /// element with an O(1) amortized odometer walk.
    pub fn strided_indices(&self) -> StridedIter<'_> {
        StridedIter::new(&self.shape, &self.strides, self.offset, self.numel)
    }
}
