mod core;
mod data;

use core::GraphTensor;

use crate::core::tensor::{FreeTensor, AbstractTensor};
use crate::core::grad::TensorKey;

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

    let md_idx: Vec<usize> = vec![0, 1, 2];

    let mut f = FreeTensor::new(shape.clone(), 10.0, true);

    let md_idx = vec![0, 1, 2];

    dbg!(f.at(&md_idx));
    f.set(&md_idx, 3.0);
    dbg!(f.at(&md_idx));

    let a = GraphTensor::new(shape.clone(), 5.1, true);
    let b = GraphTensor::new(shape.clone(), 9.2, true);

    let c = &a + &b;
    let d = &c + &a;

    let grads_map = d.backward(false);

    let grad = grads_map.get(&a.to_key()).unwrap();

    dbg!(&grad.get_node().storage);

}
