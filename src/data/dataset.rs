use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};

use crate::core::GraphTensor;
use crate::core::node::TensorNode;
use crate::core::storage::TensorStorage;
use crate::core::tensor::AbstractTensor;

pub struct CovertypeDataset{
    pub path: String,
    pub len: usize,
    reader: BufReader<File>,
    lut: Vec<u64>,
}

impl CovertypeDataset {
    pub fn new(
        path: String,
        limit: Option<usize>
    ) -> Self {
        let file_res = File::open(&path);
        if file_res.is_err() {
            panic!("Could not open file {}", &path);
        }

        let file = file_res.unwrap();

        let lut = Self::build_lut(&file).unwrap();

        Self {
            path: path,
            len: limit.unwrap_or_else(|| lut.len()),
            reader: BufReader::new(file),
            lut: lut
        }
    }

    pub fn len(&self) -> usize {
        self.len
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

    pub fn get_item(
        &mut self,
        idx: usize
    ) -> (GraphTensor, GraphTensor) {

        let seek_res = self.reader.seek(SeekFrom::Start(self.lut[idx]));
        if seek_res.is_err() {
            panic!("Could not seek line at idx {} from LUT", idx);
        }

        let mut line = String::new();
        let res = self.reader.read_line(&mut line);
        if res.is_err() {
            panic!("Could not read line {} of file at path {}", idx, &self.path);
        }

        let mut buffer: Vec<f64> = line.trim_end().split(',')
            .skip(1) // skip "index" column
            .map(|v| v.parse().unwrap())
            .collect();
        let gt = buffer.pop().unwrap() - 1.0;

        (
            GraphTensor::from_vec(buffer, false),
            GraphTensor::from_vec(gt, false)
        )
    }
}
