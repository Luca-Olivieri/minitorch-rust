use rand::{SeedableRng, rngs::StdRng};

use crate::{
    core::{
        GraphTensor,
        nn::{activate::ReLU, compute::Linear, loss::Loss, module::Forward1},
        tensor::AbstractTensor,
    },
    data::dataloader::DataLoader,
};

use crate::module;

module! {
    XORClassifier {
        modules {
            lin1: Linear,
            relu: ReLU,
            lin2: Linear,
            lin3: Linear,
        }
    }
}

impl XORClassifier {
    pub fn new(mut rng: StdRng) -> Self {
        Self {
            lin1: Linear::new(2, 100, true, StdRng::from_rng(&mut rng)),
            relu: ReLU::new(),
            lin2: Linear::new(100, 100, true, StdRng::from_rng(&mut rng)),
            lin3: Linear::new(100, 2, true, StdRng::from_rng(&mut rng)),
        }
    }
}

impl Forward1 for XORClassifier {
    fn forward(&self, input: &GraphTensor) -> GraphTensor {
        let y1 = self.lin1.forward(input);
        let y2 = self.relu.forward(&y1);
        let y3 = self.lin2.forward(&y2);
        let y4 = self.relu.forward(&y3);
        self.lin3.forward(&y4) // logits
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
            let (mut inputs, mut gts) = dl.get_batch(step);
            inputs.set_requires_grad(false);
            gts.set_requires_grad(false);

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
