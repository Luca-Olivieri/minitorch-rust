mod core;
mod data;

use core::Tensor;

fn main() {
    println!("Hello, world!");

    let shape = vec![1, 2, 3];

    // let a = TensorStorage::new(shape.clone(), 5.1);
    // let b = TensorStorage::new(shape.clone(), 9.2);


    // println!("{}", TensorStorage::add(&a, &b));
    // println!("{}", TensorStorage::minus(&a));
    // println!("{}", TensorStorage::sub(&a, &b));
    // println!("{}", TensorStorage::modul(&a, &b));
    // println!("{}", TensorStorage::mult(&a, &b));
    // println!("{}", TensorStorage::div(&a, &b));
    // println!("{}", TensorStorage::pow(&a, &b));
    // println!("{}", TensorStorage::log(&a, &b));
    // dbg!(TensorStorage::add(&a, &b));

    let a = Tensor::new(shape.clone(), 5.1, true);
    let b = Tensor::new(shape.clone(), 9.2, true);

    let c = &a + &b;
    let d = &a - &b;

    dbg!(c);
    dbg!(d);
}
