mod core;
mod data;
mod models;

use core::GraphTensor;
use std::time::Instant;

// TODO implement
// TODO use graphTensor.reshape() to build sum() over multiple dimensions

use crate::{
    core::{
        nn::{
            activate::{ReLU, Softmax},
            compute::Linear,
            loss::{CrossEntropyLoss, Loss},
            module::{Forward1, Module},
            optimizer::{Optimizer, SGD},
            smoothing::SimpleExpSmoothing,
        },
        tensor::{AbstractTensor, FreeTensor},
    },
    data::{dataloader::DataLoader, dataset::CovertypeDataset},
    models::{CovertypeClassifier, XORClassifier},
};

use rand::SeedableRng;
use rand::rngs::StdRng;

macro_rules! timeit {
    ($fmt:literal; $($stmt:stmt)*) => {
        let __start = std::time::Instant::now();

        $($stmt)*

        let __elapsed = __start.elapsed();
        println!("{}", $fmt.replace("{elapsed}", &format!("{:?}", __elapsed)));
    };
}

fn main() {
    // try_covertype();
    try_xor();
}

fn try_covertype() {
    timeit!("Datasets set up (took {elapsed})";
    let limit = 100;
    let train_ds = CovertypeDataset::new(String::from("/Users/lucaolivieri/Desktop/CS/coding/C++/minitorch/data/covertype_train.csv"), Some(limit));
    let val_ds = CovertypeDataset::new(String::from("/Users/lucaolivieri/Desktop/CS/coding/C++/minitorch/data/covertype_train.csv"), Some(limit));
    );

    timeit!("Dataloaders set up (took {elapsed})";
    let batch_size = 4;
    let mut train_dl = DataLoader::new(train_ds, batch_size, true, 42);
    let mut val_dl = DataLoader::new(val_ds, batch_size, false, 42);
    );

    let rng = StdRng::seed_from_u64(42);

    timeit!("Model set up (took {elapsed})";
    let mut model = CovertypeClassifier::new(rng);
    );

    let base_lr = 1e-2;

    timeit!("Criterion and optimizer set up (took {elapsed})";
    let criterion = CrossEntropyLoss::new();
    let optimizer = SGD::new(base_lr);
    );

    timeit!("Initial loss evaluation (took {elapsed})";
    let init_loss = model.evaluate(&mut val_dl, &criterion);
    println!("Initial loss value: {:?}", init_loss.item());
    );

    let num_epochs = 20;

    let loss_smoothing_factor = 1e-1;
    let mut epoch_loss_smoother = SimpleExpSmoothing::new(loss_smoothing_factor);

    for epoch in 0..num_epochs {
        let epoch_viz = epoch + 1;

        train_dl.reshuffle();

        let mut num_steps = 0;

        timeit!("Training epoch completed (took {elapsed})";
        for step in 0..train_dl.size() {
            let step_viz = step+1;

            let (inputs, targets) = train_dl.get_batch(step);

            let start = Instant::now();
            let logits = model.forward(&inputs);
            let forward_time = start.elapsed();

            let gts_oh = targets.one_hot(logits.shape()[1]);

            let start = Instant::now();
            let loss = criterion.forward(&logits, &gts_oh);
            let loss_time = start.elapsed();

            let start = Instant::now();
            let grads_map = loss.backward(false);
            let backward_time = start.elapsed();

            let start = Instant::now();

            let params = &mut model.all_params_mut();

            optimizer.step(params, &grads_map);

            let step_time = start.elapsed();

            epoch_loss_smoother.update(loss.item());
            num_steps += 1;

            if (epoch+1) % 2 == 0 && (step+1) == 1000 {
                println!("=== [EPOCH {epoch_viz}/{num_epochs}] STEP {step_viz}/{num_steps} === "); // TODO implement correctly padded numbers
                dbg!(forward_time, loss_time, backward_time, step_time, grads_map.len());
            }
        });

        if (epoch + 1) % 2 == 0 {
            println!(
                "=== [EPOCH {epoch_viz}/{num_epochs}] avg. train loss = {} over {num_steps} steps ===",
                epoch_loss_smoother.value()
            );
        }

        let epoch_val_loss = model.evaluate(&mut val_dl, &criterion);

        if (epoch + 1) % 2 == 0 {
            println!(
                "=== [EPOCH {epoch_viz}/{num_epochs}] val. loss = {:?} ===",
                epoch_val_loss.item()
            );
        }

        epoch_loss_smoother.reset();
    }
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

    let softmax = Softmax::new();

    println!("{}", &model.forward(&inputs).get_node().storage);

    for epoch in 0..num_epochs {
        let start = Instant::now();
        let logits = model.forward(&inputs);
        let forward_time = start.elapsed();

        let gts_oh = targets.one_hot(logits.shape()[1]);

        let start = Instant::now();
        let loss = criterion.forward(&logits, &gts_oh);
        let loss_time = start.elapsed();

        let start = Instant::now();
        let grads_map = loss.backward(false);
        let backward_time = start.elapsed();

        let start = Instant::now();

        let params = &mut model.all_params_mut();

        optimizer.step(params, &grads_map);

        let step_time = start.elapsed();

        if epoch % 10 == 0 {
            let prs = softmax.forward(&logits);
            println!("=== [EPOCH {epoch}] === ");
            dbg!(
                forward_time,
                loss_time,
                backward_time,
                step_time,
                GraphTensor::dist(&prs, &gts_oh),
                grads_map.len()
            );
        }
    }
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

    let grads_map = r.backward(true);

    println!("============ r ============");
    dbg!(&r.get_node().storage);
    println!("========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!("========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    println!("========== c.grad =========");
    let dc = grads_map.get(&c.to_key()).unwrap();
    dbg!(&dc.get_node().storage);

    let da_grads_map = da.backward(true);
    let db_grads_map = db.backward(true);
    let dc_grads_map = dc.backward(true);

    println!("========== d2a_da =========");
    let d2a_da = da_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2a_da.get_node().storage);
    println!("========== d2a_db =========");
    let d2a_db = da_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2a_db.get_node().storage);
    println!("========== d2a_dc =========");
    let d2a_dc = da_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2a_dc.get_node().storage);

    println!("========== d2b_da =========");
    let d2b_da = db_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2b_da.get_node().storage);
    println!("========== d2b_db =========");
    let d2b_db = db_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2b_db.get_node().storage);
    println!("========== d2b_dc =========");
    let d2b_dc = db_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2b_dc.get_node().storage);

    println!("========== d2c_da =========");
    let d2c_da = dc_grads_map.get(&a.to_key()).unwrap();
    dbg!(&d2c_da.get_node().storage);
    println!("========== d2c_db =========");
    let d2c_db = dc_grads_map.get(&b.to_key()).unwrap();
    dbg!(&d2c_db.get_node().storage);
    println!("========== d2c_dc =========");
    let d2c_dc = dc_grads_map.get(&c.to_key()).unwrap();
    dbg!(&d2c_dc.get_node().storage);
}

fn test_simple_operation() {
    let shape = vec![1, 2, 3];

    let a = GraphTensor::new(shape.clone(), 2.0, true);
    let b = GraphTensor::new(shape.clone(), 3.0, true);

    let x = &a * &b;

    let grads_map = x.backward(true);

    println!("============ r ============");
    dbg!(&x.get_node().storage);
    println!("========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!("========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    let da_grads_map = da.backward(true);

    println!("========== da.grad =========");
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

    println!("============ r ============");
    dbg!(&e.get_node().storage);
    println!("========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);
    println!("========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);
    println!("========== c.grad =========");
    let dc = grads_map.get(&c.to_key()).unwrap();
    dbg!(&dc.get_node().storage);
    println!("========== d.grad =========");
    let dd = grads_map.get(&d.to_key()).unwrap();
    dbg!(&dd.get_node().storage);
    println!("========== e.grad =========");
    let de = grads_map.get(&e.to_key()).unwrap();
    dbg!(&de.get_node().storage);

    let _da_grads_map = da.backward(true);

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

    println!("============ r ============");
    println!("{}", &oh.get_node().storage);
}

fn test_matmul() {
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 4];

    let a = GraphTensor::new(a_shape.clone(), 1.0, true);
    let b = GraphTensor::new(b_shape.clone(), 1.0, true);

    let x = GraphTensor::matmul(&a, &b);

    let grads_map = x.backward(true);

    println!("============ r ============");
    dbg!(&x.get_node().storage);
    println!("========== a.grad =========");
    let da = grads_map.get(&a.to_key()).unwrap();
    dbg!(&da.get_node().storage);

    println!("========== b.grad =========");
    let db = grads_map.get(&b.to_key()).unwrap();
    dbg!(&db.get_node().storage);

    let da_grads_map = da.backward(true);

    println!("========== da.grad =========");
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

    println!("============ b ============");
    dbg!(&b.get_node().storage);
    println!("========== x.grad =========");
    let dx = grads_map.get(&x.to_key()).unwrap();
    dbg!(&dx.get_node().storage);

    let dx_grads_map = dx.backward(true);

    println!("========== dx.grad =========");
    if let Some(d2x_dx) = dx_grads_map.get(&x.to_key()) {
        dbg!(&d2x_dx.get_node().storage);
    } else {
        println!("Gradient is 0 (Node disconnected from HOD graph)");
    }
}
