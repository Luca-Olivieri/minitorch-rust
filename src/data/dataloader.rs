use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::core::GraphTensor;
use crate::data::dataset::CovertypeDataset;

/// Wraps a dataset and produces batches of stacked input/target tensors.
///
/// Mirrors the C++ `DataLoader<Rs...>` abstraction: indices are shuffled once at
/// construction, reshufflable per epoch, and each batch stacks the collected
/// samples along a new leading dimension.
pub struct DataLoader {
    dataset: CovertypeDataset,
    batch_size: usize,
    num_batches: usize,
    shuffle: bool,
    rng: StdRng,
    indices: Vec<usize>,
}

fn num_batches_for(len: usize, batch_size: usize) -> usize {
    len.div_ceil(batch_size)
}

impl DataLoader {
    pub fn new(dataset: CovertypeDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        if batch_size == 0 {
            panic!("batch_size must be nonzero.");
        }

        let num_batches = num_batches_for(dataset.len(), batch_size);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            num_batches,
            shuffle,
            rng,
            indices,
        }
    }

    pub fn size(&self) -> usize {
        self.num_batches
    }

    /// Reshuffle indices when shuffle mode is enabled.
    pub fn reshuffle(&mut self) {
        if !self.shuffle {
            return;
        }
        self.indices.shuffle(&mut self.rng);
    }

    /// Return batch `index` as a `(inputs, targets)` tuple of stacked tensors.
    ///
    /// Each batch stacks the samples along a new leading dimension:
    /// inputs become `[N, 54]` and targets become `[N, ...]`.
    pub fn get_batch(&mut self, index: usize) -> (GraphTensor, GraphTensor) {
        let start = index * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.dataset.len());

        let samples: Vec<(GraphTensor, GraphTensor)> = (start..end)
            .map(|i| {
                let ds_idx = if self.shuffle { self.indices[i] } else { i };
                self.dataset.get_item(ds_idx)
            })
            .collect();

        let input_stacks: Vec<GraphTensor> = samples.iter().map(|s| s.0.copy_s()).collect();
        let target_stacks: Vec<GraphTensor> = samples.iter().map(|s| s.1.copy_s()).collect();

        (
            GraphTensor::stack(&input_stacks),
            GraphTensor::stack(&target_stacks),
        )
    }
}
