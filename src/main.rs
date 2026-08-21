mod data;
mod core;

use core::tensor_storage::TensorStorage;

fn main() {
    println!("Hello, world!");

    let shape = vec![1, 2, 3];

    let a = TensorStorage::new(shape.clone(), 5.0);
    let b = TensorStorage::new(shape.clone(), 9.0);

    println!("{}", TensorStorage::add(&a, &b));
    // dbg!(TensorStorage::add(&a, &b));
}
