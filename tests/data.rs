use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::data::dataloader::MnistDataLoader;
use minitorch_rust::data::dataset::{MNISTDataset, MnistSplit};

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "minitorch-mnist-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_images(path: &Path, count: usize, rows: usize, cols: usize, pixels: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2051u32.to_be_bytes());
    bytes.extend_from_slice(&(count as u32).to_be_bytes());
    bytes.extend_from_slice(&(rows as u32).to_be_bytes());
    bytes.extend_from_slice(&(cols as u32).to_be_bytes());
    bytes.extend_from_slice(pixels);
    fs::write(path, bytes).unwrap();
}

fn write_labels(path: &Path, labels: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2049u32.to_be_bytes());
    bytes.extend_from_slice(&(labels.len() as u32).to_be_bytes());
    bytes.extend_from_slice(labels);
    fs::write(path, bytes).unwrap();
}

fn write_split(root: &Path, split: MnistSplit, labels: &[u8]) {
    let (images_name, labels_name) = match split {
        MnistSplit::Train => ("train-images.idx3-ubyte", "train-labels.idx1-ubyte"),
        MnistSplit::Test => ("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"),
    };
    let pixels = (0..labels.len() * 4)
        .map(|value| value as u8)
        .collect::<Vec<_>>();
    write_images(&root.join(images_name), labels.len(), 2, 2, &pixels);
    write_labels(&root.join(labels_name), labels);
}

#[test]
fn mnist_dataset_parses_and_normalizes_idx_files() {
    let root = TempRoot::new();
    write_split(root.path(), MnistSplit::Train, &[3, 7]);

    let dataset = MNISTDataset::new(root.path(), MnistSplit::Train);
    assert_eq!(dataset.len(), 2);
    assert!(!dataset.is_empty());
    assert_eq!(dataset.image_shape(), vec![1, 2, 2]);

    let (image, label) = dataset.get_item(0);
    assert_eq!(image.shape().as_slice(), &[1, 2, 2]);
    assert!((image.at(&[0, 0, 0]) - 0.0).abs() < 1e-12);
    assert!((image.at(&[0, 0, 1]) - (1.0 / 255.0)).abs() < 1e-12);
    assert!((label.at(&[]) - 3.0).abs() < 1e-12);
}

#[test]
fn mnist_loader_returns_image_and_label_batches() {
    let root = TempRoot::new();
    write_split(root.path(), MnistSplit::Test, &[1, 4, 9]);

    let dataset = MNISTDataset::new(root.path(), MnistSplit::Test);
    let loader = MnistDataLoader::new(dataset, 2, false, 42);
    assert_eq!(loader.len(), 2);
    assert!(!loader.is_empty());

    let (images, labels) = loader.get_batch(1);
    assert_eq!(images.shape().as_slice(), &[1, 1, 2, 2]);
    assert_eq!(labels.shape().as_slice(), &[1]);
    assert!((labels.at(&[0]) - 9.0).abs() < 1e-12);
}

#[test]
#[should_panic]
fn mnist_dataset_rejects_invalid_idx_magic() {
    let root = TempRoot::new();
    let images = root.path().join("train-images.idx3-ubyte");
    let labels = root.path().join("train-labels.idx1-ubyte");
    write_images(&images, 1, 2, 2, &[0, 1, 2, 3]);
    fs::write(&labels, [0, 0, 0, 1, 0, 0, 0, 1, 4]).unwrap();

    let _ = MNISTDataset::new(root.path(), MnistSplit::Train);
}
