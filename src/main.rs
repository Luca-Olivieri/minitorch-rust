mod data;
mod core;

use core::tensor_storage::TensorStorage;

fn main() {
    println!("Hello, world!");

    let shape = vec![1, 2, 3];
    let fill_value = 5.0;

    let x = TensorStorage::new(shape, fill_value);

    // println!("{}", x);
    dbg!(x);
}
