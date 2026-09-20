use std::time::Instant;

use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::Forward1;
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::nn::smoothing::SimpleExpSmoothing;
use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::data::dataloader::DataLoader;
use minitorch_rust::data::dataset::CovertypeDataset;
use minitorch_rust::module;

use rand::SeedableRng;
use rand::rngs::StdRng;

macro_rules! timeit {
    ($fmt:literal; $($stmt:stmt;)*) => {
        let __start = std::time::Instant::now();

        $($stmt)*

        let __elapsed = __start.elapsed();
        println!("{}", $fmt.replace("{elapsed}", &format!("{:?}", __elapsed)));
    };
}

fn main() {
    timeit!("Datasets set up (took {elapsed})";
    let limit = 100;
    let train_ds = CovertypeDataset::new(String::from("/Users/lucaolivieri/Documents/CS/coding/C++/minitorch/data/covertype_train.csv"), Some(limit));
    let val_ds = CovertypeDataset::new(String::from("/Users/lucaolivieri/Documents/CS/coding/C++/minitorch/data/covertype_val.csv"), Some(limit));
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
    let epoch_w = num_epochs.to_string().len();

    let loss_smoothing_factor = 1e-1;
    let mut epoch_loss_smoother = SimpleExpSmoothing::new(loss_smoothing_factor);

    for epoch in 0..num_epochs {
        let epoch_viz = epoch + 1;

        train_dl.reshuffle();

        let mut num_steps = 0;

        timeit!("Training epoch completed (took {elapsed})";
        for step in 0..train_dl.size() {
            let step_viz = step+1;
            let step_w = train_dl.size().to_string().len();

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

            optimizer.step(&mut model, &grads_map);

            let step_time = start.elapsed();

            epoch_loss_smoother.update(loss.item());
            num_steps += 1;

            if (epoch+1) % 2 == 0 && (step+1) == 1000 {
                println!("=== [EPOCH {epoch_viz:>epoch_w$}/{num_epochs}] STEP {step_viz:>step_w$}/{num_steps} ===");
                dbg!(forward_time, loss_time, backward_time, step_time, grads_map.len());
            }
        };);

        if (epoch + 1) % 2 == 0 {
            println!(
                "=== [EPOCH {epoch_viz:>epoch_w$}/{num_epochs}] avg. train loss = {} over {num_steps} steps ===",
                epoch_loss_smoother.value()
            );
        }

        let epoch_val_loss = model.evaluate(&mut val_dl, &criterion);

        if (epoch + 1) % 2 == 0 {
            println!(
                "=== [EPOCH {epoch_viz:>epoch_w$}/{num_epochs}] val. loss = {:?} ===",
                epoch_val_loss.item()
            );
        }

        epoch_loss_smoother.reset();
    }
}

module! {
    CovertypeClassifier {
        modules {
            lin1: Linear,
            relu: ReLU,
            lin2: Linear,
            lin3: Linear,
        }
    }
}

impl CovertypeClassifier {
    pub fn new(mut rng: StdRng) -> Self {
        Self {
            lin1: Linear::new(54, 100, true, StdRng::from_rng(&mut rng)),
            relu: ReLU::new(),
            lin2: Linear::new(100, 100, true, StdRng::from_rng(&mut rng)),
            lin3: Linear::new(100, 7, true, StdRng::from_rng(&mut rng)),
        }
    }

    pub fn evaluate(&self, dl: &mut DataLoader, criterion: &dyn Loss) -> GraphTensor {
        let mut curr_loss = 0.0;
        let mut curr_sample_count = 0;

        for step in 0..dl.size() {
            let (inputs, gts) = dl.get_batch(step);

            let prs_oh = self.forward(&inputs);

            let gts_oh = gts.one_hot(prs_oh.shape()[1]);

            let loss = criterion.forward(&prs_oh, &gts_oh);

            curr_loss += loss.item() * (inputs.shape()[0] as f64);
            curr_sample_count += inputs.shape()[0];
        }

        let total_loss = curr_loss / (curr_sample_count as f64);
        GraphTensor::wrap(total_loss, false)
    }
}

impl Forward1 for CovertypeClassifier {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let y1 = self.lin1.forward(input);
        let y2 = self.relu.forward(&y1);
        let y3 = self.lin2.forward(&y2);
        let y4 = self.relu.forward(&y3);
        self.lin3.forward(&y4) // logits
    }
}
