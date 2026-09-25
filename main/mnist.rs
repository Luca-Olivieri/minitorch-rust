use std::collections::BTreeMap;
use std::fs;
use std::time::{Duration, Instant};

use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::autograd::BackwardTiming;
use minitorch_rust::core::nn::activate::ReLU;
use minitorch_rust::core::nn::compute::{Conv2d, Conv2dPadding, Linear, MaxPool2d};
use minitorch_rust::core::nn::dropout::Dropout;
use minitorch_rust::core::nn::dyn_sequential::DynSequential;
use minitorch_rust::core::nn::flatten::Flatten;
use minitorch_rust::core::nn::loss::{CrossEntropyLoss, Loss};
use minitorch_rust::core::nn::module::{Forward1, Module};
use minitorch_rust::core::nn::optimizer::{Optimizer, SGD};
use minitorch_rust::core::nn::smoothing::SimpleExpSmoothing;
use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::data::dataloader::MnistDataLoader;
use minitorch_rust::data::dataset::{MNISTDataset, MnistSplit};
use minitorch_rust::module;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/config/train_config.yml");

#[derive(Debug, Deserialize)]
struct TrainConfig {
    dataset_root: String,
    batch_size: usize,
    base_lr: f64,
    epochs: usize,
    smoothing_factor: f64,
    seed: u64,
    log_every: usize,
    dropout_p: f64,
}

impl TrainConfig {
    fn load() -> Self {
        let contents = fs::read_to_string(CONFIG_PATH).unwrap_or_else(|error| {
            panic!("failed to read training config `{CONFIG_PATH}`: {error}");
        });
        let config: Self = serde_yaml::from_str(&contents).unwrap_or_else(|error| {
            panic!("failed to parse training config `{CONFIG_PATH}`: {error}");
        });
        config.validate();
        config
    }

    fn validate(&self) {
        assert!(self.batch_size > 0, "batch_size must be greater than zero.");
        assert!(self.log_every > 0, "log_every must be greater than zero.");
        assert!(
            self.base_lr.is_finite() && self.base_lr >= 0.0,
            "base_lr must be a finite, non-negative number."
        );
        assert!(
            (0.0..=1.0).contains(&self.smoothing_factor),
            "smoothing_factor must be in [0, 1]."
        );
        assert!(
            (0.0..1.0).contains(&self.dropout_p),
            "dropout_p must be in [0, 1)."
        );
    }
}

macro_rules! timeit {
    ($fmt:literal; $($stmt:stmt;)*) => {
        let __start = Instant::now();

        $($stmt)*

        let __elapsed = __start.elapsed();
        println!("{}", $fmt.replace("{elapsed}", &format!("{:?}", __elapsed)));
    };
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_layer_timings(epoch: usize, step: usize, timings: &[(&'static str, Duration)]) {
    println!("[PROFILE] Forward timings | epoch {epoch} | step {step}");
    let name_width = timings
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);
    for (name, duration) in timings {
        println!("  {name:<name_width$}  {:>12.6} ms", duration_ms(*duration));
    }
}

fn print_backward_timings(epoch: usize, step: usize, timings: &[BackwardTiming]) {
    let mut totals: BTreeMap<&'static str, (usize, Duration)> = BTreeMap::new();
    for timing in timings {
        let entry = totals
            .entry(timing.operation)
            .or_insert((0, Duration::ZERO));
        entry.0 += 1;
        entry.1 += timing.duration;
    }

    println!("[PROFILE] Backward operation totals | epoch {epoch} | step {step}");
    println!("  {:<20}  {:>7}  {:>14}", "operation", "count", "total");
    for (operation, (count, total)) in totals {
        println!(
            "  {operation:<20}  {count:>7}  {:>12.6} ms",
            duration_ms(total)
        );
    }

    let conv_timings: Vec<_> = timings
        .iter()
        .filter(|timing| timing.operation == "conv2d")
        .collect();
    if conv_timings.is_empty() {
        return;
    }

    println!("  convolution details (reverse graph order)");
    for (index, timing) in conv_timings.iter().enumerate() {
        println!(
            "    conv2d_backward[{}]  {:>12.6} ms",
            index + 1,
            duration_ms(timing.duration)
        );
        for (section, duration) in &timing.details {
            println!("      {section:<20}  {:>12.6} ms", duration_ms(*duration));
        }
    }
}

module! {
    SmallCNN {
        modules {
            features: DynSequential,
            classifier: DynSequential,
        }
    }
}

impl SmallCNN {
    /// Build the MNIST-sized CNN using independent RNG streams derived from
    /// the supplied session RNG.
    pub fn new(session_rng: &mut StdRng, training: bool, dropout_p: f64) -> Self {
        let features = DynSequential::new()
            .with(Conv2d::new_with_options(
                1,
                32,
                3,
                1,
                Conv2dPadding::Same,
                1,
                true,
                StdRng::from_rng(session_rng),
            ))
            .with(ReLU::new())
            .with(MaxPool2d::new(2, None))
            .with(Conv2d::new_with_options(
                32,
                64,
                3,
                1,
                Conv2dPadding::Same,
                1,
                true,
                StdRng::from_rng(session_rng),
            ))
            .with(ReLU::new())
            .with(MaxPool2d::new(2, None));

        let classifier = DynSequential::new()
            .with(Flatten::new())
            .with(Dropout::new(dropout_p, training, session_rng))
            .with(Linear::new(
                64 * 7 * 7,
                128,
                true,
                StdRng::from_rng(session_rng),
            ))
            .with(ReLU::new())
            .with(Linear::new(128, 10, true, StdRng::from_rng(session_rng)));

        Self {
            features,
            classifier,
        }
    }

    pub fn forward_with_timings(
        &self,
        input: &GraphTensor,
        no_grad: bool,
    ) -> (GraphTensor, Vec<(&'static str, Duration)>) {
        const FEATURE_LAYERS: [&str; 6] = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"];
        const CLASSIFIER_LAYERS: [&str; 5] = ["flatten", "dropout", "linear1", "relu3", "linear2"];

        let (features, feature_timings) = self.features.forward_with_timings(input, no_grad);
        let (output, classifier_timings) = self.classifier.forward_with_timings(&features, no_grad);
        assert_eq!(feature_timings.len(), FEATURE_LAYERS.len());
        assert_eq!(classifier_timings.len(), CLASSIFIER_LAYERS.len());

        let mut timings = Vec::with_capacity(FEATURE_LAYERS.len() + CLASSIFIER_LAYERS.len());
        timings.extend(FEATURE_LAYERS.iter().copied().zip(feature_timings));
        timings.extend(CLASSIFIER_LAYERS.iter().copied().zip(classifier_timings));
        (output, timings)
    }

    pub fn evaluate(&self, loader: &MnistDataLoader, criterion: &dyn Loss) -> GraphTensor {
        let mut total_loss = 0.0;
        let mut sample_count = 0usize;

        for step in 0..loader.size() {
            let (inputs, targets) = loader.get_batch(step);
            let logits = self.forward(&inputs, true);
            let targets_oh = targets.one_hot(logits.shape()[1]);
            let loss = criterion.forward(&logits, &targets_oh, true);

            total_loss += loss.item() * inputs.shape()[0] as f64;
            sample_count += inputs.shape()[0];
        }

        assert!(sample_count > 0, "cannot evaluate an empty MNIST loader");
        GraphTensor::wrap(total_loss / sample_count as f64, false)
    }
}

impl Forward1 for SmallCNN {
    fn forward(&self, input: &GraphTensor, no_grad: bool) -> GraphTensor {
        let features = self.features.forward(input, no_grad);
        self.classifier.forward(&features, no_grad)
    }
}

fn main() {
    let config = TrainConfig::load();
    let profile_layers = std::env::var_os("MINITORCH_PROFILE_LAYERS").is_some();
    if profile_layers {
        println!("Layer profiling enabled for logged training steps.");
    }

    timeit!("Datasets set up (took {elapsed})";
        let train_dataset = MNISTDataset::new(
            &config.dataset_root,
            MnistSplit::Train,
        );
        let test_dataset = MNISTDataset::new(
            &config.dataset_root,
            MnistSplit::Test,
        );
    );

    timeit!("Dataloaders set up (took {elapsed})";
        let mut train_loader = MnistDataLoader::new(
            train_dataset,
            config.batch_size,
            true,
            config.seed,
        );
        let test_loader = MnistDataLoader::new(
            test_dataset,
            config.batch_size,
            false,
            config.seed,
        );
    );

    let mut session_rng = StdRng::seed_from_u64(config.seed);
    timeit!("Model set up (took {elapsed})";
        let mut model = SmallCNN::new(
            &mut session_rng,
            true,
            config.dropout_p,
        );
    );

    println!(
        "Sample count (train_dataset): {}",
        train_loader.dataset.len()
    );
    println!("Sample count (test_dataset): {}", test_loader.dataset.len());
    println!("Num. batches (train_loader): {}", train_loader.size());
    println!("Num. batches (test_loader): {}", test_loader.size());

    let criterion = CrossEntropyLoss::new();
    let optimizer = SGD::new(config.base_lr);
    let num_epochs = config.epochs;
    let epoch_width = num_epochs.to_string().len();
    let mut loss_smoother = SimpleExpSmoothing::new(config.smoothing_factor);
    let log_every = config.log_every;

    timeit!("Initial loss evaluation (took {elapsed})";
        model.set_training(false);
        let initial_loss = model.evaluate(&test_loader, &criterion);
        println!("Initial loss value: {:?}", initial_loss.item());
    );

    for epoch in 0..num_epochs {
        let epoch_number = epoch + 1;
        model.set_training(true);
        train_loader.reshuffle();
        let mut num_steps = 0usize;
        let num_batches = train_loader.size();

        let training_start = Instant::now();
        for step in 0..num_batches {
            let step_number = step + 1;
            let step_width = num_batches.to_string().len();
            let (inputs, targets) = train_loader.get_batch(step);

            let profile_this_step =
                profile_layers && (step_number % log_every == 0 || step_number == num_batches);
            let start = Instant::now();
            let mut layer_timings = None;
            let logits = if profile_this_step {
                let (output, timings) = model.forward_with_timings(&inputs, false);
                layer_timings = Some(timings);
                output
            } else {
                model.forward(&inputs, false)
            };
            let forward_time = start.elapsed();
            if let Some(timings) = layer_timings.as_ref() {
                print_layer_timings(epoch_number, step_number, timings);
            }

            let targets_oh = targets.one_hot(logits.shape()[1]);

            let start = Instant::now();
            let loss = criterion.forward(&logits, &targets_oh, false);
            let loss_time = start.elapsed();

            let start = Instant::now();
            let mut backward_timings: Option<Vec<BackwardTiming>> = None;
            let grads = if profile_this_step {
                let (grads, timings) = loss.backward_profiled(false);
                backward_timings = Some(timings);
                grads
            } else {
                loss.backward(false)
            };
            let backward_time = start.elapsed();
            if let Some(timings) = backward_timings.as_ref() {
                print_backward_timings(epoch_number, step_number, timings);
            }

            let start = Instant::now();
            optimizer.step(&mut model, &grads);
            let step_time = start.elapsed();

            loss_smoother.update(loss.item());
            num_steps += 1;

            if step_number % log_every == 0 || step_number == num_batches {
                println!(
                    "=== [EPOCH {epoch_number:>epoch_width$}/{num_epochs}] STEP {step_number:>step_width$}/{num_batches} ==="
                );
                dbg!(
                    forward_time,
                    loss_time,
                    loss_smoother.value(),
                    backward_time,
                    step_time,
                    grads.len()
                );
            }
        }

        let training_time = training_start.elapsed();
        println!(
            "=== [EPOCH {epoch_number:>epoch_width$}/{num_epochs}] training time = {:?} ===",
            training_time
        );

        println!(
            "=== [EPOCH {epoch_number:>epoch_width$}/{num_epochs}] smoothed train loss = {} over {num_steps} steps ===",
            loss_smoother.value()
        );

        model.set_training(false);
        let validation_start = Instant::now();
        let validation_loss = model.evaluate(&test_loader, &criterion);
        let validation_time = validation_start.elapsed();
        println!(
            "=== [EPOCH {epoch_number:>epoch_width$}/{num_epochs}] validation loss = {:?} (time = {:?}) ===",
            validation_loss.item(),
            validation_time
        );

        loss_smoother.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::TrainConfig;

    #[test]
    fn train_config_loads() {
        let config = TrainConfig::load();
        assert!(config.batch_size > 0);
        assert!(config.log_every > 0);
    }
}
