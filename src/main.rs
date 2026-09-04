mod core;
mod data;
mod models;

use core::GraphTensor;

use crate::{core::{nn::{activate::ReLU, compute::Linear, loss::{CrossEntropyLoss, Loss}, module::{Forward1, Module}, optimizer::{Optimizer, SGD}}, storage::TensorStorage, tensor::{AbstractTensor, FreeTensor}}, models::XORClassifier};

use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    println!("Hello, world!");

    // let shape = vec![1, 2, 3];

    // // let a = TensorStorage::new(shape.clone(), 5.1);
    // // let b = TensorStorage::new(shape.clone(), 9.2);

    // // println!("{}", TensorStorage::add(&a, &b));
    // // println!("{}", TensorStorage::minus(&a));
    // // println!("{}", TensorStorage::sub(&a, &b));
    // // println!("{}", TensorStorage::modul(&a, &b));
    // // println!("{}", TensorStorage::mult(&a, &b));
    // // println!("{}", TensorStorage::div(&a, &b));
    // // println!("{}", TensorStorage::pow(&a, &b));
    // // println!("{}", TensorStorage::log(&a, &b));
    // // dbg!(TensorStorage::add(&a, &b));

    // let md_idx: Vec<usize> = vec![0, 1, 2];

    // let mut f = FreeTensor::new(shape.clone(), 10.0, true);

    // let md_idx = vec![0, 1, 2];

    // dbg!(f.at(&md_idx));
    // f.set(&md_idx, 3.0);
    // dbg!(f.at(&md_idx));

    // let a = GraphTensor::new(shape.clone(), 5.1, true);
    // let b = GraphTensor::new(shape.clone(), 9.2, true);

    // let c = &a + &b;
    // let d = &c - &a;

    // let grads_map = d.backward(false);

    // let grad = grads_map.get(&a.to_key()).unwrap();

    // dbg!(&grad.get_node().storage);

    // let sum = TensorStorage::sum(&s);
    // dbg!(sum);

    // test_complex_operation()
    // test_simple_operation()

    // test_shapes();
    // test_sum_dim();
    // test_one_hot();
    // test_matmul();

    // test_linear_relu();

    try_xor();
}

fn try_xor() {
    let input_shape = vec![4, 2];
    let mut inputs_f = FreeTensor::new(input_shape.clone(), 0.0, false);
    inputs_f.set(&vec![0, 0], 0.0);
    inputs_f.set(&vec![0, 1], 0.0);
    inputs_f.set(&vec![1, 0], 0.0);
    inputs_f.set(&vec![1, 1], 1.0);
    inputs_f.set(&vec![2, 0], 1.0);
    inputs_f.set(&vec![2, 1], 0.0);
    inputs_f.set(&vec![3, 0], 1.0);
    inputs_f.set(&vec![3, 1], 1.0);
    let inputs = inputs_f.to_graph();

    let input_shape = vec![4];
    let mut targets_f = FreeTensor::new(input_shape.clone(), 0.0, true);
    targets_f.set(&vec![0], 0.0);
    targets_f.set(&vec![1], 1.0);
    targets_f.set(&vec![2], 1.0);
    targets_f.set(&vec![3], 0.0);
    let targets = targets_f.to_graph();

    let rng = StdRng::seed_from_u64(42);

    let mut model = XORClassifier::new(rng);

    let criterion = CrossEntropyLoss::new();

    let optimizer = SGD::new(0.1);

    let num_epochs = 100;

    dbg!(&model.forward(&inputs).get_node().storage.buffer);

    for epoch in 0..num_epochs {

        let logits = model.forward(&inputs);

        let gts_oh = targets.one_hot(logits.shape()[1]);

        let loss = criterion.forward(&logits, &gts_oh);

        let grads_map = loss.backward(false);

        let d_lin1_w = grads_map.get(&model.lin1.weight.to_key()).unwrap();
        let d_lin1_b = grads_map.get(&model.lin1.bias.as_ref().unwrap().to_key()).unwrap();
        let d_lin2_w = grads_map.get(&model.lin2.weight.to_key()).unwrap();
        let d_lin2_b = grads_map.get(&model.lin2.bias.as_ref().unwrap().to_key()).unwrap();
        let d_lin3_w = grads_map.get(&model.lin3.weight.to_key()).unwrap();
        let d_lin3_b = grads_map.get(&model.lin3.bias.as_ref().unwrap().to_key()).unwrap();

        model.lin1.weight = optimizer.step(&model.lin1.weight, d_lin1_w);
        *model.lin1.bias.as_mut().unwrap() = optimizer.step(&model.lin1.bias.as_ref().unwrap(), d_lin1_b);

        model.lin2.weight = optimizer.step(&model.lin2.weight, d_lin2_w);
        *model.lin2.bias.as_mut().unwrap() = optimizer.step(&model.lin2.bias.as_ref().unwrap(), d_lin2_b);

        model.lin3.weight = optimizer.step(&model.lin3.weight, d_lin3_w);
        *model.lin3.bias.as_mut().unwrap() = optimizer.step(&model.lin3.bias.as_ref().unwrap(), d_lin3_b);

        if epoch % 10 == 0 {
            dbg!(epoch);
        }
    }

    let logits = model.forward(&inputs);
    let gts_oh = targets.one_hot(logits.shape()[1]);

    dbg!(&logits.get_node().storage);
    dbg!(&gts_oh.get_node().storage);
}

fn test_complex_operation() {

    let shape = vec![1, 2, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true);
    let b = GraphTensor::new(shape.clone(), 3.0, true);
    let c = GraphTensor::new(shape.clone(), 4.0, true);

    let x = &a * &b;
    let y = &x + &c;
    let z = &y - &a;
    let w = &z / &b;

    let twos = GraphTensor::new(shape.clone(), 2.0, false);
    let p = w.pow(&twos);
    let q = p.ln();
    let r = -&q;

    let grads_map = r.backward(false);

    println!( "============ r ============");
    dbg!(&r.get_node().storage);
    println!( "========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!( "========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    println!( "========== c.grad =========");
    let dc = grads_map.get(&c.to_key()).unwrap();
    dbg!(&dc.get_node().storage);

    let da_grads_map = da.backward(true);
    let db_grads_map = db.backward(true);
    let dc_grads_map = dc.backward(true);

    println!( "========== d2a_da =========");
    let d2a_da = da_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2a_da.get_node().storage);
    println!( "========== d2a_db =========");
    let d2a_db = da_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2a_db.get_node().storage);
    println!( "========== d2a_dc =========");
    let d2a_dc = da_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2a_dc.get_node().storage);

    println!( "========== d2b_da =========");
    let d2b_da = db_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2b_da.get_node().storage);
    println!( "========== d2b_db =========");
    let d2b_db = db_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2b_db.get_node().storage);
    println!( "========== d2b_dc =========");
    let d2b_dc = db_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2b_dc.get_node().storage);

    println!( "========== d2c_da =========");
    let d2c_da = dc_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2c_da.get_node().storage);
    println!( "========== d2c_db =========");
    let d2c_db = dc_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2c_db.get_node().storage);
    println!( "========== d2c_dc =========");
    let d2c_dc = dc_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2c_dc.get_node().storage);
}

fn test_simple_operation() {

    let shape = vec![1, 2, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true);
    let b = GraphTensor::new(shape.clone(), 3.0, true);

    let x = &a * &b;

    let grads_map = x.backward(false);

    println!( "============ r ============");
    dbg!(&x.get_node().storage);
    println!( "========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!( "========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    let da_grads_map = da.backward(true);

    println!( "========== da.grad =========");
    if let Some(d2a_da) = da_grads_map.get(&a.to_key()) {
        dbg!(&d2a_da.get_node().storage);
    } else {
        println!("Gradient is 0 (Node disconnected from HOD graph)");
    }
}

fn test_shapes() {

    let shape = vec![4, 2, 1, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true); // [4, 2, 1, 3]
    dbg!(&a.shape());
    let b = a.squeeze(2); // [4, 2, 3]
    dbg!(&b.shape());
    let c = b.unsqueeze(3); // [4, 2, 3, 1]
    dbg!(&c.shape());
    let d = c.sum_dim(1); // [2, 3, 1]
    dbg!(&d.shape());
    let e = d.sum(); // []
    dbg!(&e.shape());

    let grads_map = e.backward(true);

    println!( "============ r ============");
    dbg!(&e.get_node().storage);
    println!( "========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);
    println!( "========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);
    println!( "========== c.grad =========");
    let dc = grads_map.get(&c.to_key()).unwrap();
    dbg!(&dc.get_node().storage);
    println!( "========== d.grad =========");
    let dd = grads_map.get(&d.to_key()).unwrap();
    dbg!(&dd.get_node().storage);
    println!( "========== e.grad =========");
    let de = grads_map.get(&e.to_key()).unwrap();
    dbg!(&de.get_node().storage);

    let da_grads_map = da.backward(true);

    // println!( "========== da.grad =========");
    // if let Some(d2a_da) = da_grads_map.get(&a.to_key()) {
    //     dbg!(&d2a_da.get_node().storage);
    // } else {
    //     println!("Gradient is 0 (Node disconnected from HOD graph)");
    // }
}

fn test_one_hot() {

    let shape = vec![4, 2];

    let mut f = FreeTensor::new(shape.clone(), 2.0, true);
    f.set(&vec![0, 0], 3.0);
    f.set(&vec![0, 1], 3.0);
    f.set(&vec![1, 0], 2.0);
    f.set(&vec![1, 1], 2.0);
    f.set(&vec![2, 0], 1.0);
    f.set(&vec![2, 1], 1.0);
    f.set(&vec![3, 0], 0.0);
    f.set(&vec![3, 1], 0.0);

    let a = f.to_graph();
    let oh = a.one_hot(4);

    println!( "============ r ============");
    println!("{}", &oh.get_node().storage);
}

fn test_matmul() {

    let a_shape = vec![2, 3];
    let b_shape = vec![3, 4];

    let a = GraphTensor::new(a_shape.clone(), 1.0, true);
    let b = GraphTensor::new(b_shape.clone(), 1.0, true);

    let x = GraphTensor::matmul(&a, &b);

    let grads_map = x.backward(true);

    println!( "============ r ============");
    dbg!(&x.get_node().storage);
    println!( "========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!( "========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    let da_grads_map = da.backward(true);

    println!( "========== da.grad =========");
    if let Some(d2a_da) = da_grads_map.get(&a.to_key()) {
        dbg!(&d2a_da.get_node().storage);
    } else {
        println!("Gradient is 0 (Node disconnected from HOD graph)");
    }
}

fn test_linear_relu() {

    let rng = StdRng::seed_from_u64(42);

    let lin = Linear::new(3, 4, true, rng);
    let relu = ReLU::new();

    let x_shape = vec![2, 3];

    let x = GraphTensor::new(x_shape.clone(), 1.0, true);

    let a = lin.forward(&x);
    let b = relu.forward(&a);

    let grads_map = b.backward(true);

    println!( "============ b ============");
    dbg!(&b.get_node().storage);
    println!( "========== x.grad =========");
    let dx = grads_map.get(&x.to_key()).unwrap();
    dbg!(&dx.get_node().storage);

    let dx_grads_map = dx.backward(true);

    println!( "========== dx.grad =========");
    if let Some(d2x_dx) = dx_grads_map.get(&x.to_key()) {
        dbg!(&d2x_dx.get_node().storage);
    } else {
        println!("Gradient is 0 (Node disconnected from HOD graph)");
    }
}
