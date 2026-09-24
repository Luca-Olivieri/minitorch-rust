use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use crate::core::GraphTensor;

/// Selects the standard MNIST training or test split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MnistSplit {
    Train,
    Test,
}

impl MnistSplit {
    fn file_names(self) -> (&'static str, &'static str) {
        match self {
            Self::Train => ("train-images.idx3-ubyte", "train-labels.idx1-ubyte"),
            Self::Test => ("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"),
        }
    }
}

/// In-memory MNIST dataset backed by the standard IDX files.
///
/// Images are stored as raw bytes and converted to normalized `[1, rows,
/// cols]` tensors when requested. Labels are returned as scalar `f64` tensors.
pub struct MNISTDataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl MNISTDataset {
    /// Load one split from a directory containing the four root-level IDX files.
    pub fn new<P: AsRef<Path>>(root: P, split: MnistSplit) -> Self {
        let root = root.as_ref();
        let (images_name, labels_name) = split.file_names();
        let (images, rows, cols) = read_idx_images(&root.join(images_name));
        let labels = read_idx_labels(&root.join(labels_name));
        if images.len() / (rows * cols) != labels.len() {
            panic!(
                "MNIST image/label count mismatch: {} images versus {} labels.",
                images.len() / (rows * cols),
                labels.len()
            );
        }

        Self {
            images,
            labels,
            rows,
            cols,
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image_shape(&self) -> Vec<usize> {
        vec![1, self.rows, self.cols]
    }

    /// Return a normalized image and its scalar label.
    pub fn get_item(&self, index: usize) -> (GraphTensor, GraphTensor) {
        if index >= self.len() {
            panic!(
                "MNIST item index {index} is out of range for {} items.",
                self.len()
            );
        }

        let image_numel = self.rows * self.cols;
        let start = index * image_numel;
        let end = start + image_numel;
        let pixels: Vec<f64> = self.images[start..end]
            .iter()
            .map(|&value| f64::from(value) / 255.0)
            .collect();
        let image = GraphTensor::from_flat_buffer(self.image_shape(), pixels, false);
        let label = GraphTensor::wrap(f64::from(self.labels[index]), false);
        (image, label)
    }
}

fn read_idx_images(path: &Path) -> (Vec<u8>, usize, usize) {
    let bytes = read_idx_file(path);
    if bytes.len() < 16 {
        panic!(
            "MNIST image file {} is shorter than its IDX header.",
            path.display()
        );
    }
    let magic = read_be_u32(&bytes, 0, "image magic");
    if magic != 2051 {
        panic!(
            "Unsupported MNIST image magic {magic} in {}.",
            path.display()
        );
    }
    let count = read_be_u32(&bytes, 4, "image count") as usize;
    let rows = read_be_u32(&bytes, 8, "image rows") as usize;
    let cols = read_be_u32(&bytes, 12, "image columns") as usize;
    if rows == 0 || cols == 0 {
        panic!("MNIST image dimensions must be nonzero, got {rows}x{cols}.");
    }
    let image_numel = rows
        .checked_mul(cols)
        .expect("MNIST image dimensions overflow usize");
    let expected_len = 16usize
        .checked_add(
            count
                .checked_mul(image_numel)
                .expect("MNIST image payload size overflows usize"),
        )
        .expect("MNIST image file size overflows usize");
    if bytes.len() != expected_len {
        panic!(
            "MNIST image file {} has {} bytes, expected {expected_len}.",
            path.display(),
            bytes.len()
        );
    }
    (bytes[16..].to_vec(), rows, cols)
}

fn read_idx_labels(path: &Path) -> Vec<u8> {
    let bytes = read_idx_file(path);
    if bytes.len() < 8 {
        panic!(
            "MNIST label file {} is shorter than its IDX header.",
            path.display()
        );
    }
    let magic = read_be_u32(&bytes, 0, "label magic");
    if magic != 2049 {
        panic!(
            "Unsupported MNIST label magic {magic} in {}.",
            path.display()
        );
    }
    let count = read_be_u32(&bytes, 4, "label count") as usize;
    if bytes.len() != 8 + count {
        panic!(
            "MNIST label file {} has {} bytes, expected {}.",
            path.display(),
            bytes.len(),
            8 + count
        );
    }
    bytes[8..].to_vec()
}

fn read_idx_file(path: &Path) -> Vec<u8> {
    fs::read(path)
        .unwrap_or_else(|error| panic!("Could not read MNIST file {}: {error}", path.display()))
}

fn read_be_u32(bytes: &[u8], offset: usize, field: &str) -> u32 {
    let end = offset + 4;
    let raw = bytes
        .get(offset..end)
        .unwrap_or_else(|| panic!("MNIST IDX header is missing {field}."));
    u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]])
}

pub struct CovertypeDataset {
    pub path: String,
    pub len: usize,
    reader: BufReader<File>,
    lut: Vec<u64>,
}

impl CovertypeDataset {
    pub fn new(path: String, limit: Option<usize>) -> Self {
        let file_res = File::open(&path);
        if file_res.is_err() {
            panic!("Could not open file {}", path);
        }

        let file = file_res.unwrap();

        let lut = Self::build_lut(&file).unwrap();

        Self {
            path,
            len: limit.unwrap_or(lut.len()),
            reader: BufReader::new(file),
            lut,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn build_lut(file: &File) -> io::Result<Vec<u64>> {
        let mut reader = BufReader::new(file.try_clone()?);
        let mut position = 0u64;

        // Skip the first line without indexing it
        let mut first_line = String::new();
        let first_bytes = reader.read_line(&mut first_line)?;
        position += first_bytes as u64;

        let mut index = Vec::new();

        // If the file only had one line (or was empty), there's nothing left to index
        if first_bytes == 0 {
            return Ok(index);
        }

        loop {
            index.push(position);

            let mut line = String::new();
            let bytes = reader.read_line(&mut line)?;

            if bytes == 0 {
                index.pop(); // don't keep position after EOF
                break;
            }

            position += bytes as u64;
        }

        Ok(index)
    }

    pub fn get_item(&mut self, idx: usize) -> (GraphTensor, GraphTensor) {
        let seek_res = self.reader.seek(SeekFrom::Start(self.lut[idx]));
        if seek_res.is_err() {
            panic!("Could not seek line at idx {} from LUT", idx);
        }

        let mut line = String::new();
        let res = self.reader.read_line(&mut line);
        if res.is_err() {
            panic!("Could not read line {} of file at path {}", idx, self.path);
        }

        let mut buffer: Vec<f64> = line
            .trim_end()
            .split(',')
            .skip(1) // skip "index" column
            .map(|v| v.parse().unwrap())
            .collect();
        let gt = buffer.pop().unwrap() - 1.0;

        (
            GraphTensor::wrap(buffer, false),
            GraphTensor::wrap(gt, false),
        )
    }
}
