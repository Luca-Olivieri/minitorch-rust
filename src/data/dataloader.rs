use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::core::GraphTensor;
use crate::data::dataset::{CovertypeDataset, MNISTDataset};

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

/// Batches normalized MNIST tensors with the same shuffle/reshuffle behavior
/// as the existing Covertype loader.
pub struct MnistDataLoader {
    pub dataset: MNISTDataset,
    batch_size: usize,
    num_batches: usize,
    shuffle: bool,
    rng: StdRng,
    indices: Vec<usize>,
}

impl MnistDataLoader {
    pub fn new(dataset: MNISTDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        if batch_size == 0 {
            panic!("batch_size must be nonzero.");
        }

        let num_batches = dataset.len().div_ceil(batch_size);
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

    pub fn len(&self) -> usize {
        self.num_batches
    }

    pub fn is_empty(&self) -> bool {
        self.num_batches == 0
    }

    pub fn reshuffle(&mut self) {
        if self.shuffle {
            self.indices.shuffle(&mut self.rng);
        }
    }

    pub fn get_batch(&self, index: usize) -> (GraphTensor, GraphTensor) {
        if index >= self.num_batches {
            panic!(
                "MNIST batch index {index} is out of range for {} batches.",
                self.num_batches
            );
        }

        let start = index * self.batch_size;
        let end = (start + self.batch_size).min(self.dataset.len());
        let samples: Vec<(GraphTensor, GraphTensor)> = (start..end)
            .map(|offset| {
                let dataset_index = if self.shuffle {
                    self.indices[offset]
                } else {
                    offset
                };
                self.dataset.get_item(dataset_index)
            })
            .collect();

        let inputs: Vec<GraphTensor> = samples.iter().map(|sample| sample.0.copy_s()).collect();
        let targets: Vec<GraphTensor> = samples.iter().map(|sample| sample.1.copy_s()).collect();
        (GraphTensor::stack(&inputs), GraphTensor::stack(&targets))
    }
}
