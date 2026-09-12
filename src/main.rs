use std::time::Instant;

use minitorch_rust::core::nn::activate::Softmax;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::{Forward1, Module};
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::nn::smoothing::SimpleExpSmoothing;
use minitorch_rust::core::tensor::{AbstractTensor, FreeTensor};
use minitorch_rust::core::GraphTensor;
use minitorch_rust::data::dataloader::DataLoader;
use minitorch_rust::data::dataset::CovertypeDataset;
use minitorch_rust::models::{CovertypeClassifier, XORClassifier};

use rand::rngs::StdRng;
use rand::SeedableRng;

macro_rules! timeit {
    ($fmt:literal; $($stmt:stmt;)*) => {
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
        };);

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
